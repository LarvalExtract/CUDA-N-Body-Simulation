#include "NBodySystemCUDA.h"

#include <cuda_runtime_api.h>
#include <curand_kernel.h>

namespace kernels
{
	__global__ void init_particles(unsigned int, float2*, float2*, float*, size_t, float);
	__global__ void step_particles(float2*, float2*, float2*, float*, size_t, float, float, float);
	__global__ void write_particle_descriptions_to_texture(float4*, float2*, float2*, float*, size_t);
	
	__device__ float2 calculate_gravitational_accel(float2, float2, float, float);
	__device__ float2 compute_acceleration_tiled(float2, float*, float2*, float);
	__device__ float dot(float2, float2);
	__device__ float length(float2);
}

void NBodySystemCUDA::CallInitParticlesKernel()
{
	const auto
	[
		numParticles, 
		seed, 
		softeningFactor, 
		timeScale, 
		aspectRatio
	] = SimulationParameters.Simulation;

	dev_OldPositions.resize(numParticles);
	dev_NewPositions.resize(numParticles);
	dev_Velocities.resize(numParticles);
	dev_Masses.resize(numParticles);

	dev_OldPositionsPtr = dev_OldPositions.data();

	const dim3 Db(256);
	const dim3 Dg((numParticles + Db.x - 1) / Db.x);

	kernels::init_particles<<<Dg, Db>>>(
		seed,
		thrust::raw_pointer_cast(dev_OldPositionsPtr),
		thrust::raw_pointer_cast(dev_Velocities.data()),
		thrust::raw_pointer_cast(dev_Masses.data()),
		numParticles,
		aspectRatio
		);

	dev_NewPositions = dev_OldPositions;
	dev_NewPositionsPtr = dev_NewPositions.data();
}

void NBodySystemCUDA::CallStepParticlesKernel(float deltaTime)
{
	const dim3 Db(SimulationParameters.CUDA.TileSize);
	const dim3 Dg((GetNumParticles() + Db.x - 1) / Db.x);
	const size_t S = Db.x * (sizeof(float2) + sizeof(float));

	ThreadsPerBlock = Db.x;
	BlockCount = Dg.x;
	SharedMem = S;

	kernels::step_particles<<<Dg, Db, S>>>(
		thrust::raw_pointer_cast(dev_OldPositionsPtr),
		thrust::raw_pointer_cast(dev_NewPositionsPtr),
		thrust::raw_pointer_cast(dev_Velocities.data()),
		thrust::raw_pointer_cast(dev_Masses.data()),
		SimulationParameters.Simulation.NumParticles,
		SimulationParameters.Simulation.TimeScale * (deltaTime / 1000.0f),
		SimulationParameters.Simulation.ScreenAspectRatio,
		SimulationParameters.Simulation.SofteningFactor
		);
	
	std::swap(dev_OldPositionsPtr, dev_NewPositionsPtr);
}

void NBodySystemCUDA::CallFillParticleTextureKernel()
{
	const dim3 Db(256);
	const dim3 Dg((GetNumParticles() + Db.x - 1) / Db.x);

	kernels::write_particle_descriptions_to_texture<<<Dg, Db>>>(
		pParticleTexture,
		thrust::raw_pointer_cast(dev_NewPositions.data()),
		thrust::raw_pointer_cast(dev_Velocities.data()),
		thrust::raw_pointer_cast(dev_Masses.data()),
		SimulationParameters.Simulation.NumParticles
		);
}

namespace kernels
{
	__device__ float dot(float2 a, float2 b)
	{
		return a.x * b.x + a.y * b.y;
	}

	__device__ float length(float2 vec)
	{
		return sqrtf(dot(vec, vec));
	}

	__global__ void init_particles(unsigned int seed, float2* particlePositions, float2* particleVelocities, float* particleMasses, size_t numParticles, float aspectRatio)
	{
		const int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= numParticles)
		{
			return;
		}

		curandStateXORWOW_t state;
		curand_init(seed, i, 0, &state);

		const auto angle = curand_uniform(&state) * (2.0f * 3.14159f);
		const auto radius = (curand_normal(&state) + 1.0f) / 4.0f;
		float2 pos = {
			radius * cosf(angle),
			radius * sinf(angle)
		};

		const float k = 150.0f;

		particlePositions[i] = pos;
		particleVelocities[i] = { -pos.y / radius * k, pos.x / radius * k};
		particleMasses[i] = curand_uniform(&state);
	}

	__device__ float2 calculate_gravitational_accel(float2 pi, float2 pj, float mj, float e)
	{
		const float2 r = { pj.x - pi.x, pj.y - pi.y };

		const float distSqr = (r.x * r.x + r.y * r.y) + e;
		const float invDist = rsqrtf(distSqr);
		const float invDistCube = invDist * invDist * invDist;
		const float GM = invDistCube * mj;

		return { GM * r.x, GM * r.y };
	}

	__device__ float2 compute_acceleration_tiled(float2 particlePosition, float* particleMasses, float2* oldPositions, float softeningFactor)
	{
		__shared__ extern float shared[];

		const auto shared_positions = reinterpret_cast<float2*>(shared);
		const auto shared_masses = &shared[blockDim.x * 2];

		float2 accel = { 0.0f, 0.0f };

		for (size_t tileId = 0; tileId < gridDim.x; tileId++)
		{
			const size_t idx = tileId * blockDim.x + threadIdx.x;

			shared_positions[threadIdx.x] = oldPositions[idx];
			shared_masses[threadIdx.x] = particleMasses[idx];

			__syncthreads();
			for (size_t i = 0; i < blockDim.x; i++)
			{
				const auto [ax, ay] = calculate_gravitational_accel(particlePosition, shared_positions[i], shared_masses[i], softeningFactor);
				accel.x += ax;
				accel.y += ay;
			}
			__syncthreads();
		}

		return accel;
	}

	__global__ void step_particles(float2* oldPositions, float2* newPositions, float2 *velocities, float* masses, size_t numBodies, float deltaTime, float aspectRatio, float softeningFactor)
	{
		const int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= numBodies)
		{
			return;
		}

		float2 position = oldPositions[i];
		float2 velocity = velocities[i];
		float2 accel = compute_acceleration_tiled(position, masses, oldPositions, softeningFactor);

		velocity.x += accel.x * deltaTime;
		velocity.y += accel.y * deltaTime;

		position.x += velocity.x * deltaTime;
		position.y += velocity.y * deltaTime;

		// bounce particles off screen edges
		if (position.x < -aspectRatio || position.x > aspectRatio) velocity.x = -velocity.x * 0.5f;
		if (position.y < -1.0f || position.y > 1.0f) velocity.y = -velocity.y * 0.5f;
	   
		newPositions[i] = position;
		velocities[i] = velocity;
	}

	__global__ void write_particle_descriptions_to_texture(float4* buffer, float2* particlePositions, float2* particleVelocities, float* particleMasses, size_t numParticles)
	{
		const int i = blockIdx.x * blockDim.x + threadIdx.x;

		if (i >= numParticles)
		{
			return;
		}

		float2 vel = particleVelocities[i];
		float particleSpeed = length(vel);

		buffer[i].x = particlePositions[i].x;
		buffer[i].y = particlePositions[i].y;
		buffer[i].z = particleSpeed;
		buffer[i].w = particleMasses[i];
	}
}
