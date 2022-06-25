#include "NBodyApplication.h"
#include "CUDAHelpers.h"

#include <EventManager.h>
#include <EvtFrameStart.h>
#include <SwapChainConfigDX11.h>

#include <imgui.h>
#include <imgui_impl_dx11.h>
#include <imgui_impl_win32.h>

using namespace Glyph3;

NBodyApplication AppInstance;

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

LRESULT NBodyApplication::WindowProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam)
{
	return ImGui_ImplWin32_WndProcHandler(hwnd, msg, wparam, lparam) | Application::WindowProc(hwnd, msg, wparam, lparam);
}

NBodyApplication::NBodyApplication()
{
	// Find CUDA device
	if (int cudaDeviceCount; cudaGetDeviceCount(&cudaDeviceCount) == cudaSuccess && cudaDeviceCount > 0)
	{
		m_CudaDevices.resize(cudaDeviceCount);
		for (int device = 0; device < cudaDeviceCount; device++)
		{
			CUDA_CALL(cudaGetDeviceProperties(&m_CudaDevices.at(device), device));
		}
	}
	else
	{
		throw std::runtime_error("No CUDA devices found");
	}
}

void NBodyApplication::BeforeRegisterWindowClass(WNDCLASSEX& wc)
{
	
}

std::wstring NBodyApplication::GetName()
{
	return L"6100COMP Artefact";
}

void NBodyApplication::Initialize()
{
	m_NBodySystemParams.Simulation.ScreenAspectRatio = static_cast<float>(m_iScreenWidth) / static_cast<float>(m_iScreenHeight);
	m_NBodySystem.InitSimulation(m_NBodySystemParams);

	m_NBodyRenderer = std::make_unique<NBodySystemRenderTask>(m_RenderTarget);
	m_NBodyRenderer->SetBodySystem(&m_NBodySystem);
}

void NBodyApplication::Update()
{
	m_pRenderer11->pImmPipeline->ClearBuffers({0.0f, 0.0f, 0.0f, 1.0f});

	m_pTimer->Update();
	EvtManager.ProcessEvent(std::make_shared<EvtFrameStart>(m_pTimer->Elapsed()));

	m_NBodyRenderer->Update(m_pTimer->Elapsed());
	
	m_NBodyRenderer->QueuePreTasks(m_pRenderer11.get());

	m_pRenderer11->ProcessTaskQueue();

	DrawImGui();

	m_pRenderer11->Present(m_pWindow->GetHandle(), m_pWindow->GetSwapChain());
}

bool NBodyApplication::ConfigureEngineComponents()
{

	// Set the render window parameters and initialize the window
	m_pWindow = std::make_unique<Win32RenderWindow>();
	m_pWindow->SetPosition(25, 25);
	m_pWindow->SetSize(m_iScreenWidth, m_iScreenHeight);
	auto name = GetName();
	m_pWindow->SetCaption(name);
	m_pWindow->Initialize(this);


	// Create the renderer and initialize it for the desired device
	// type and feature level.
	m_pRenderer11 = std::make_unique<RendererDX11>();

	if (!m_pRenderer11->Initialize(D3D_DRIVER_TYPE_HARDWARE, D3D_FEATURE_LEVEL_11_0))
	{
		Log::Get().Write(L"Could not create hardware device, trying to create the reference device...");

		if (!m_pRenderer11->Initialize(D3D_DRIVER_TYPE_REFERENCE, D3D_FEATURE_LEVEL_10_0))
		{
			ShowWindow(m_pWindow->GetHandle(), SW_HIDE);
			MessageBox(m_pWindow->GetHandle(), "Could not create a hardware or software Direct3D 11 device!", "6100COMP Artefact", MB_ICONEXCLAMATION | MB_SYSTEMMODAL);
			RequestTermination();
			return(false);
		}
	}

	SwapChainConfigDX11 tconfig;
	tconfig.SetWidth(m_pWindow->GetWidth());
	tconfig.SetHeight(m_pWindow->GetHeight());
	tconfig.SetOutputWindow(m_pWindow->GetHandle());
	const auto iSwapChain = m_pRenderer11->CreateSwapChain(&tconfig);
	m_pWindow->SetSwapChain(iSwapChain);
	
	m_RenderTarget = m_pRenderer11->GetSwapChainResource(iSwapChain);

	// Init ImGui
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();

	// Setup Platform/Renderer backends
	ID3D11DeviceContext* pDeviceContext;
	m_pRenderer11->GetDevice()->GetImmediateContext(&pDeviceContext);

	ImGui_ImplWin32_Init(m_pWindow->GetHandle());
	ImGui_ImplDX11_Init(m_pRenderer11->GetDevice(), pDeviceContext);

	return true;
}

void NBodyApplication::ShutdownEngineComponents()
{
	m_pRenderer11->Shutdown();
	m_pWindow->Shutdown();

	ImGui_ImplDX11_Shutdown();
	ImGui_ImplWin32_Shutdown();
	ImGui::DestroyContext();
}

void NBodyApplication::Shutdown()
{
	
}

bool NBodyApplication::ConfigureCommandLine(LPSTR lpcmdline)
{
	unsigned int particleCount, tileSize;

	// ss defaults particleCount and tileSize to 0 if cl args aren't set/are bad
	std::stringstream ss(lpcmdline);
	ss >> particleCount;
	ss >> tileSize;

	const unsigned int defaultParticleCount = m_CudaDevices[0].multiProcessorCount * m_CudaDevices[0].maxThreadsPerMultiProcessor;
	const unsigned int defaultTileSize = 256;

	m_NBodySystemParams.Simulation.NumParticles = particleCount >= 1 && particleCount <= 131072
		? particleCount
		: defaultParticleCount;

	m_NBodySystemParams.CUDA.TileSize = tileSize >= 1 && tileSize <= 1024
		? tileSize
		: defaultTileSize;

	return true;
}

void NBodyApplication::TakeScreenShot()
{
	
}

void NBodyApplication::DrawImGui()
{
	ImGui_ImplDX11_NewFrame();
	ImGui_ImplWin32_NewFrame();
	ImGui::NewFrame();

	DrawCustomUi();

	ImGui::Render();

	m_pRenderer11->pImmPipeline->ClearRenderTargets();
	m_pRenderer11->pImmPipeline->OutputMergerStage.DesiredState.RenderTargetViews.SetState(0, m_RenderTarget->m_iResourceRTV);
	m_pRenderer11->pImmPipeline->ApplyRenderTargets();

	m_pRenderer11->pImmPipeline->RasterizerStage.DesiredState.ViewportCount.SetState(1);
	m_pRenderer11->pImmPipeline->RasterizerStage.DesiredState.Viewports.SetState(0, 0);
	m_pRenderer11->pImmPipeline->RasterizerStage.DesiredState.RasterizerState.SetState(0);

	m_pRenderer11->pImmPipeline->OutputMergerStage.DesiredState.DepthStencilState.SetState(0);
	m_pRenderer11->pImmPipeline->OutputMergerStage.DesiredState.BlendState.SetState(0);

	ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
}

void NBodyApplication::DrawCustomUi()
{
	ImGui::Begin("NBody Simulation");

	constexpr auto formattedBytes = [](size_t bytes)
	{
		constexpr size_t kb = 1024;
		constexpr size_t mb = kb * kb;
		constexpr size_t gb = mb * kb;

		if (bytes > gb)	return std::format("{:.2f}GB", static_cast<double>(bytes) / static_cast<double>(gb));
		if (bytes > mb)	return std::format("{:.2f}MB", static_cast<double>(bytes) / static_cast<double>(mb));
		if (bytes > kb)	return std::format("{:.2f}kB", static_cast<double>(bytes) / static_cast<double>(kb));
		return std::format("{} bytes", bytes);
	};

	const auto blocks = m_NBodySystem.GetBlockCount();
	const auto tpb = m_NBodySystem.GetThreadsPerBlock();
	const auto threads = blocks * tpb;

	const auto globalMemory =
		m_NBodySystem.dev_Masses.size() * sizeof(float) +
		m_NBodySystem.dev_OldPositions.size() * sizeof(float2) +
		m_NBodySystem.dev_NewPositions.size() * sizeof(float2) +
		m_NBodySystem.dev_Velocities.size() * sizeof(float2);
	const auto sharedMemory = m_NBodySystem.GetSharedMemSize() * blocks;
	const auto totalMemory = globalMemory + sharedMemory;

	const auto gms = formattedBytes(globalMemory);
	const auto sms = formattedBytes(sharedMemory);
	const auto tms = formattedBytes(totalMemory);

	const auto ms = m_NBodySystem.KernelStepTime.count() * 1000.0;
	ImVec4 colour;
	if (ms < 16.66)
		colour = { 0.0f, 1.0f, 0.0f, 1.0f };
	else if (ms < 33.33)
		colour = { 1.0f, 1.0f, 0.0f, 1.0f };
	else
		colour = { 1.0f, 0.0f, 0.0f, 1.0f };

	ImGui::Text("Performance");
	ImGui::Separator();

	ImGui::TextColored(colour, "Simulation time: %.3fms (%i FPS)", ms, m_pTimer->Framerate());
	ImGui::Text("Memory: %s (%s global + %s shared)", tms.c_str(), gms.c_str(), sms.c_str());
	
	ImGui::Spacing();

	ImGui::Text("Settings");
	ImGui::Separator();
	ImGui::SliderInt("Tiles", reinterpret_cast<int*>(&m_NBodySystem.SimulationParameters.CUDA.TileSize), 1, min(m_NBodySystem.SimulationParameters.Simulation.NumParticles, 1024));
	ImGui::Text("Blocks: %u", blocks);
	ImGui::Text("Threads/block: %u", tpb);
	ImGui::Text("Threads: %u", threads);
	ImGui::Separator();
	ImGui::SliderFloat("Time scale", &m_NBodySystem.SimulationParameters.Simulation.TimeScale, 0.0f, 1.0f);
	ImGui::DragFloat("Softening factor", &m_NBodySystem.SimulationParameters.Simulation.SofteningFactor, 0.00001f, 0.00001f, 0.1f, "%.5f");
	ImGui::Separator();
	ImGui::SliderInt("Particles", reinterpret_cast<int*>(&m_NBodySystemParams.Simulation.NumParticles), 128, 131072);
	ImGui::DragInt("Seed", reinterpret_cast<int*>(&m_NBodySystemParams.Simulation.Seed));
	if (ImGui::Button("Reset"))
	{
		m_NBodySystem.InitSimulation(m_NBodySystemParams);
		m_NBodyRenderer->SetBodySystem(&m_NBodySystem);
	}

	ImGui::Spacing();

	const auto& device = m_CudaDevices[0];
	const auto totalmem = formattedBytes(device.totalGlobalMem);
	ImGui::Text("Device");
	ImGui::Separator();
	ImGui::Text("                 Name: %s", device.name);
	ImGui::Text("      Core clock rate: %uMHz", device.clockRate / 1000);
	ImGui::Text("    Memory clock rate: %uMHz", device.memoryClockRate / 1000);
	ImGui::Text("        Global memory: %s", totalmem.c_str());
	ImGui::Text("  Compute capabilitiy: %u.%u", device.major, device.minor);
	ImGui::Text("                  SMs: %u", device.multiProcessorCount);
	ImGui::Text("   Max threads per SM: %u", device.maxThreadsPerMultiProcessor);
	ImGui::Text("     Threads per warp: %u", device.warpSize);
	ImGui::Text("    Max blocks per SM: %u", device.maxBlocksPerMultiProcessor);
	ImGui::Text("     Registers per SM: %u", device.regsPerMultiprocessor);
	ImGui::Spacing();

	ImGui::End();
}