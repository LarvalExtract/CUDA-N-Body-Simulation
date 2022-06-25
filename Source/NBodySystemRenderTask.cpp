#include "NBodySystemRenderTask.h"

#include <format>
#include <cuda_d3d11_interop.h>

#include <Texture2dConfigDX11.h>
#include <RasterizerStateConfigDX11.h>

using namespace Glyph3;

NBodySystemRenderTask::NBodySystemRenderTask(ResourcePtr renderTarget) :
	RenderTarget(renderTarget)
{
	const auto renderer = RendererDX11::Get();
	const auto screenWidth = RenderTarget->m_pTexture2dConfig->GetTextureDesc().Width;
	const auto screenHeight = RenderTarget->m_pTexture2dConfig->GetTextureDesc().Height;

	std::wstring shaderFile(L"TextureView.hlsl");
	std::wstring vertexShaderFunction(L"VSMain");
	std::wstring pixelShaderFunction(L"PSMain");
	std::wstring vertexShaderModel(L"vs_4_0");
	std::wstring pixelShaderModel(L"ps_4_0");

	RenderEffect.SetVertexShader(renderer->LoadShader(VERTEX_SHADER, shaderFile, vertexShaderFunction, vertexShaderModel, true, true));
	RenderEffect.SetPixelShader(renderer->LoadShader(PIXEL_SHADER, shaderFile, pixelShaderFunction, pixelShaderModel));

	RasterizerStateConfigDX11 rasteriserConfig;
	rasteriserConfig.FillMode = D3D11_FILL_SOLID;
	rasteriserConfig.CullMode = D3D11_CULL_NONE;
	RenderEffect.m_iRasterizerState = renderer->CreateRasterizerState(&rasteriserConfig);

	D3D11_SAMPLER_DESC samplerDesc;
	samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
	samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
	samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
	samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
	samplerDesc.MaxAnisotropy = 0;
	samplerDesc.MipLODBias = 0.0f;
	iSamplerState = renderer->CreateSamplerState(&samplerDesc);

	D3D11_VIEWPORT viewport;
	viewport.Width = static_cast<float>(screenWidth);
	viewport.Height = static_cast<float>(screenHeight);
	viewport.MinDepth = 0.0f;
	viewport.MaxDepth = 1.0f;
	viewport.TopLeftX = 0;
	viewport.TopLeftY = 0;

	iViewportIndex = renderer->CreateViewPort(viewport);
}

void NBodySystemRenderTask::Update(float fTime)
{
	if (pBodySystem)
	{
		pBodySystem->StepSimulation(fTime);
	}
}

void NBodySystemRenderTask::QueuePreTasks(RendererDX11* pRenderer)
{
	pRenderer->QueueTask(this);
}

void NBodySystemRenderTask::ExecuteTask(PipelineManagerDX11* pPipelineManager, IParameterManager* pParamManager)
{
	if (!pBodySystem)
	{
		return;
	}

	SetRenderParams(pParamManager);

	pBodySystem->Render(Texture->m_pTexture2dConfig->GetTextureDesc().Width, Texture->m_pTexture2dConfig->GetTextureDesc().Height);

	pPipelineManager->InputAssemblerStage.ClearDesiredState();
	pPipelineManager->InputAssemblerStage.DesiredState.PrimitiveTopology.SetState(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
	pPipelineManager->InputAssemblerStage.ApplyDesiredState(pPipelineManager->GetDeviceContext());

	pPipelineManager->RasterizerStage.ClearDesiredState();
	pPipelineManager->RasterizerStage.DesiredState.ViewportCount.SetState(1);
	pPipelineManager->RasterizerStage.DesiredState.Viewports.SetState(0, iViewportIndex);
	pPipelineManager->RasterizerStage.ApplyDesiredState(pPipelineManager->GetDeviceContext());

	pPipelineManager->ClearRenderTargets();
	pPipelineManager->OutputMergerStage.DesiredState.RenderTargetViews.SetState(0, RenderTarget->m_iResourceRTV);
	pPipelineManager->ApplyRenderTargets();

	pPipelineManager->ClearPipelineResources();
	RenderEffect.ConfigurePipeline(pPipelineManager, pParamManager);
	pPipelineManager->ApplyPipelineResources();

	pPipelineManager->Draw(pBodySystem->GetNumParticles(), 0);
}

void NBodySystemRenderTask::SetRenderParams(IParameterManager* pParamManager)
{
	if (!pBodySystem)
	{
		return;
	}

	pParamManager->SetShaderResourceParameter(L"ParticleTexture", Texture);
	pParamManager->SetSamplerParameter(L"ParticleTextureSampler", &iSamplerState);

	const auto renderTargetDesc = RenderTarget->m_pTexture2dConfig->GetTextureDesc();
	const auto screenWidth = static_cast<float>(renderTargetDesc.Width);
	const auto screenHeight = static_cast<float>(renderTargetDesc.Height);

	const auto texdesc = Texture->m_pTexture2dConfig->GetTextureDesc();

	Vector4f params{
		static_cast<float>(texdesc.Width),
		static_cast<float>(texdesc.Height),
		screenWidth / screenHeight,
		0.0f };

	pParamManager->SetVectorParameter(L"ParticleParams", &params);
}

void NBodySystemRenderTask::SetUsageParams(IParameterManager* pParamManager)
{
}

std::wstring NBodySystemRenderTask::GetName()
{
	return L"NBodySystem render task";
}

void NBodySystemRenderTask::SetBodySystem(INBodySystem* pBodySystem)
{
	this->pBodySystem = pBodySystem;

	if (this->pBodySystem)
	{
		const auto renderer = RendererDX11::Get();

		if (Texture != nullptr)
		{
			renderer->DeleteResource(Texture);
		}

		const auto numParticles = this->pBodySystem->GetNumParticles();
		const auto dim = static_cast<unsigned int>(std::ceilf(sqrtf(numParticles)));

		Texture2dConfigDX11 texconfig;
		texconfig.SetWidth(dim);
		texconfig.SetHeight(dim);
		texconfig.SetUsage(D3D11_USAGE_DEFAULT);
		texconfig.SetArraySize(1);
		texconfig.SetMipLevels(1);
		texconfig.SetBindFlags(D3D11_BIND_SHADER_RESOURCE);
		texconfig.SetFormat(DXGI_FORMAT_R32G32B32A32_FLOAT);

		Texture = renderer->CreateTexture2D(&texconfig, 0);

		this->pBodySystem->SetupRenderingResources(Texture);
	}
}
