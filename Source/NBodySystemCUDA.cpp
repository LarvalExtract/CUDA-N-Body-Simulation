#include "NBodySystemCUDA.h"
#include "CUDAHelpers.h"

#include <cuda_d3d11_interop.h>
#include <cuda_runtime_api.h>

#include <RendererDX11.h>
#include <Texture2dDX11.h>

NBodySystemCUDA::~NBodySystemCUDA()
{
	CUDA_CALL(cudaGraphicsUnregisterResource(pCudaGraphicsResource));
	CUDA_CALL(cudaFree(pParticleTexture));
}

void NBodySystemCUDA::InitSimulation(const NBodySystemParameters& parameters)
{
	SimulationParameters = parameters;

	CallInitParticlesKernel();
}

void NBodySystemCUDA::StepSimulation(float deltaTime)
{
	CUDA_CALL(cudaDeviceSynchronize());
	const auto start = std::chrono::high_resolution_clock::now();

	CallStepParticlesKernel(deltaTime);

	CUDA_CALL(cudaDeviceSynchronize());
	const auto end = std::chrono::high_resolution_clock::now();

	KernelStepTime = end - start;
}

void NBodySystemCUDA::Render(unsigned int textureWidth, unsigned int textureHeight)
{
	CUDA_CALL(cudaGraphicsMapResources(1, &pCudaGraphicsResource));

	cudaArray* cuArray = nullptr;
	CUDA_CALL(cudaGraphicsSubResourceGetMappedArray(&cuArray, pCudaGraphicsResource, 0, 0));

	CallFillParticleTextureKernel();

	CUDA_CALL(cudaMemcpy2DToArray(cuArray, 0, 0, pParticleTexture, Pitch, textureWidth * sizeof(float4), textureHeight, cudaMemcpyDeviceToDevice));
	CUDA_CALL(cudaGraphicsUnmapResources(1, &pCudaGraphicsResource));
}

using namespace Glyph3;
void NBodySystemCUDA::SetupRenderingResources(ResourcePtr texture)
{
	const auto renderer = RendererDX11::Get();

	const auto resource = renderer->GetTexture2DByIndex(texture->m_iResource);
	const auto width = resource->GetActualDescription().Width;
	const auto height = resource->GetActualDescription().Height;
	const auto d3d11Resource = resource->GetResource();

	if (pCudaGraphicsResource)
	{
		CUDA_CALL(cudaGraphicsUnregisterResource(pCudaGraphicsResource));
		CUDA_CALL(cudaFree(pParticleTexture));
	}

	CUDA_CALL_ASSERT(cudaGraphicsD3D11RegisterResource(&pCudaGraphicsResource, d3d11Resource, cudaGraphicsRegisterFlagsNone));
	CUDA_CALL_ASSERT(cudaMallocPitch(reinterpret_cast<void**>(&pParticleTexture), &Pitch, width * sizeof(float4), height));
	CUDA_CALL_ASSERT(cudaMemset(pParticleTexture, 0, SimulationParameters.Simulation.NumParticles * sizeof(float4)));
}

unsigned int NBodySystemCUDA::GetNumParticles() const
{
	return SimulationParameters.Simulation.NumParticles;
}

unsigned NBodySystemCUDA::GetThreadsPerBlock() const
{
	return ThreadsPerBlock;
}

unsigned NBodySystemCUDA::GetBlockCount() const
{
	return BlockCount;
}

unsigned NBodySystemCUDA::GetSharedMemSize() const
{
	return SharedMem;
}
