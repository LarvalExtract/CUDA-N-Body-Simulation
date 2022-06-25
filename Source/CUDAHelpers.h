#pragma once

// C++ 20 :)
#include <format>
#include <source_location>

#pragma warning(disable:4091)

#define CUDA_ERROR_STRING(function, result) std::format("CUDA error: {} returned {} ({}) [{}: {}]\n", function, cudaGetErrorName(result), cudaGetErrorString(result), std::source_location::current().file_name(), std::source_location::current().line())

#ifdef _DEBUG
// Helper macro which logs the result of the given CUDA call to output if the CUDA call returned an error
// DEBUG mode: breaks execution on error
#define CUDA_CALL(function) \
	if (const auto result = function; result != cudaSuccess) \
	{ \
		OutputDebugString(CUDA_ERROR_STRING(#function, result).c_str()); \
		__debugbreak(); \
	}void 
#else
// Helper macro which logs the result of the given CUDA call to output if the CUDA call returned an error
#define CUDA_CALL(function) \
	if (const auto result = function; result != cudaSuccess) \
	{ \
		OutputDebugString(CUDA_ERROR_STRING(#function, result).c_str()); \
	}void 
#endif

// Helper macro which logs the result of the given CUDA call to output and throws an std::runtime_exception if the CUDA call returned an error
#define CUDA_CALL_ASSERT(function) \
	if (const auto result = function; result != cudaSuccess) \
	{ \
		const auto msg = CUDA_ERROR_STRING(#function, result); \
		OutputDebugString(msg.c_str()); \
		throw std::runtime_error(msg); \
	}void