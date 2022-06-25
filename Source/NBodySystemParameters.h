#pragma once

struct NBodySystemParameters
{
	struct SimulationParameters
	{
		unsigned int NumParticles = 32768;
		unsigned int Seed = 12345;
		float SofteningFactor = 0.0001f;
		float TimeScale = 1.0f;
		float ScreenAspectRatio = 1.0f;
	} Simulation;

	struct CUDAParameters
	{
		unsigned int TileSize = 256;
	} CUDA;
};