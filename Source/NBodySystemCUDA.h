#pragma once

#include "INBodySystem.h"

#include <thrust/device_vector.h>

class NBodySystemCUDA : public INBodySystem
{
	friend class NBodyApplication;

public:
	~NBodySystemCUDA() override;

	void InitSimulation(const NBodySystemParameters& parameters) override;
	void StepSimulation(float deltaTime) override;
	void Render(unsigned int textureWidth, unsigned int textureHeight) override;
	void SetupRenderingResources(Glyph3::ResourcePtr texture) override;
	unsigned int GetNumParticles() const override;

	unsigned int GetThreadsPerBlock() const;
	unsigned int GetBlockCount() const;
	unsigned int GetSharedMemSize() const;

private:
	void CallInitParticlesKernel();
	void CallStepParticlesKernel(float deltaTime);
	void CallFillParticleTextureKernel();
	
	NBodySystemParameters SimulationParameters;
	std::chrono::duration<double> KernelStepTime;
	unsigned int ThreadsPerBlock;
	unsigned int BlockCount;
	unsigned int SharedMem;

	cudaGraphicsResource* pCudaGraphicsResource = nullptr;
	float4* pParticleTexture = nullptr;
	size_t Pitch = 0;
	
	thrust::device_vector<float2> dev_OldPositions;
	thrust::device_vector<float2> dev_NewPositions;
	thrust::device_vector<float2> dev_Velocities;
	thrust::device_vector<float> dev_Masses;

	thrust::device_ptr<float2> dev_OldPositionsPtr;
	thrust::device_ptr<float2> dev_NewPositionsPtr;
};