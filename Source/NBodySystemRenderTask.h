#pragma once

#include <Task.h>

#include "INBodySystem.h"

class NBodySystemRenderTask : public Glyph3::Task
{
	friend class INBodySystem;

public:
	NBodySystemRenderTask(Glyph3::ResourcePtr renderTarget);
	~NBodySystemRenderTask() override = default;

	void Update(float fTime) override;
	void QueuePreTasks(Glyph3::RendererDX11* pRenderer) override;
	void ExecuteTask(Glyph3::PipelineManagerDX11* pPipelineManager, Glyph3::IParameterManager* pParamManager) override;
	void SetRenderParams(Glyph3::IParameterManager* pParamManager) override;
	void SetUsageParams(Glyph3::IParameterManager* pParamManager) override;
	std::wstring GetName() override;

	void SetBodySystem(INBodySystem* pBodySystem);

private:
	INBodySystem* pBodySystem = nullptr;

	Glyph3::ResourcePtr RenderTarget;
	Glyph3::RenderEffectDX11 RenderEffect;
	Glyph3::ResourcePtr Texture;

	int iViewportIndex = 0;
	int iSamplerState = 0;
};