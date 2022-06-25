#pragma once

#include "NBodySystemCUDA.h"
#include "NBodySystemRenderTask.h"

#include <Application.h>
#include <Win32RenderWindow.h>

#include <vector>

class NBodyApplication : public Glyph3::Application
{
public:
	NBodyApplication();
	void Initialize() override;
	void Update() override;
	void Shutdown() override;
	bool ConfigureCommandLine(LPSTR lpcmdline) override;
	bool ConfigureEngineComponents() override;
	void ShutdownEngineComponents() override;
	void TakeScreenShot() override;
	std::wstring GetName() override;
	LRESULT WindowProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) override;
	void BeforeRegisterWindowClass(WNDCLASSEX& wc) override;

	void DrawImGui();
	void DrawCustomUi();

protected:
	std::unique_ptr<Glyph3::RendererDX11> m_pRenderer11;
	std::unique_ptr<Glyph3::Win32RenderWindow> m_pWindow;
	Glyph3::ResourcePtr	m_RenderTarget;

	NBodySystemParameters m_NBodySystemParams;
	NBodySystemCUDA m_NBodySystem;
	std::unique_ptr<NBodySystemRenderTask> m_NBodyRenderer;
	std::vector<cudaDeviceProp> m_CudaDevices;
	
	int	m_iScreenWidth = 1920;
	int m_iScreenHeight = 1080;
};


