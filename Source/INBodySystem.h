#pragma once

#include "NBodySystemParameters.h"

#include <ResourceDX11.h>

class INBodySystem
{
public:
	virtual ~INBodySystem() = default;

	virtual void InitSimulation(const NBodySystemParameters& parameters) = 0;
	virtual void StepSimulation(float deltaTime) = 0;
	virtual void Render(unsigned int textureWidth, unsigned int textureHeight) = 0;
	virtual void SetupRenderingResources(Glyph3::ResourcePtr texture) = 0;
	virtual unsigned int GetNumParticles() const = 0;
};
