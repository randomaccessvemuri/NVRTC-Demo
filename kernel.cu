//NVRTC DEMO: We take the iteration function for a fractal and compile it into a CUDA kernel at runtime.
const int IMAGE_X = 1920;
const int IMAGE_Y = 1080;
#define randomness 2

#include <cuda_runtime.h>
#include <nvrtc.h>
#include <cuda.h>
#include <plog/Log.h>
#include "plog/Initializers/ConsoleInitializer.h"
#include <iostream>
#include <string>
#include <plog/Formatters/TxtFormatter.h>
#include <cuda/std/complex>
#include <fstream>
#include <regex>
#include <imgui.h>
#include <imgui_stdlib.h>
#include <imgui-SFML.h>
#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>







const std::string ITERATION_FUNCTION = "z = z*z + c";
const char* headerSourceDir[] = {""};
const char* headerNames[] = { "cuComplex.h" };



const std::string code = R"(
#include <cuda/std/complex>
extern "C" __global__ void iterFunc(uchar4 * img, float2 cartesianBounds, float2 cartesianOrigin, int2 imageDims, uchar4* colorMap, int colorMapSize, int maxBounces) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= imageDims.x || y >= imageDims.y) return;

	
	
	float cartesianX = (x - imageDims.x / 2) * cartesianBounds.x / imageDims.x + cartesianOrigin.x;
	float cartesianY = (y - imageDims.y / 2) * cartesianBounds.y / imageDims.y + cartesianOrigin.y;

	cuda::std::complex<float> c(cartesianX, cartesianY);
	cuda::std::complex<float> z = c;
	int bounces = 0;
	
	for (int i = 0; i < maxBounces; i++) {
		ITERATION_FUNCTION;
		if ((z.real() * z.real() + z.imag() * z.imag()) > cartesianBounds.x * cartesianBounds.y) {
			break;
		}
		bounces++;
	}

	if (bounces == maxBounces) {
		img[y * imageDims.x + x] = colorMap[0];
		return;
	}

	


	float bouncesForBlending = (float)bounces + 1 - log(z.real() * z.real() + z.imag() * z.imag())/log(2.f);
	
	float t = abs(bouncesForBlending - (int)bouncesForBlending); // fractional part
	
	int index1 = ((int)bouncesForBlending % (colorMapSize - 1)) + 1;
	int index2 = (index1 % (colorMapSize - 1)) + 1;

	uchar4 color1 = colorMap[index1==0?1: index1];
	uchar4 color2 = colorMap[index2==0?1: index2];

	// Linear interpolation
	uchar4 color;
	color.x = color1.x * (t) + color2.x * (1.0-t);
	color.y = color1.y * (t) + color2.y * (1.0-t);
	color.z = color1.z * (t) + color2.z * (1.0-t);
	color.w = 255;

	img[y * imageDims.x + x] = color;
}
)";

struct imgDims{
	int width;
	int height;
};


struct ptxExecEnv {
	CUdevice device;
	CUcontext context;
	CUmodule module;
	CUfunction kernel;
	void* args;
	const char* ptx;
	unsigned long long ptxSize;
	int gridSize;
	int blockSize;
	CUdeviceptr imgDevice;
	uchar4* imgHost;
	unsigned long long bufferSize;
	nvrtcProgram prog;
};
struct preRenderConfig {
	float2 cartesianBounds;
	float2 cartesianOrigin;
	imgDims imageDims;
	int maxBounces;
	uchar4* colorMap;
	int colorMapSize;
	std::string iterationFunction;
};

ptxExecEnv initCuda() {
	cuInit(0);
	CUdevice device;
	CUcontext context;

	cuDeviceGet(&device, 0);
	cuCtxCreate(&context, 0, device);

	return {
		device,
		context,
		nullptr,
		nullptr,
		nullptr,
		nullptr,
		0,
		0,
		0,
		0,
		nullptr,
		0,
		0
	};

}

void generatePTX(std::string kernelCode, preRenderConfig preRenderConfigObj, ptxExecEnv* env) {

	kernelCode= std::regex_replace(kernelCode, std::regex("ITERATION_FUNCTION"), preRenderConfigObj.iterationFunction);

	std::cout << "Kernel Code: \n" << kernelCode << "\n";
	_sleep(1000);


	PLOGD << "Generating PTX Code From Kernel Code..";
	nvrtcProgram prog;
	nvrtcCreateProgram(
		&prog, 
		kernelCode.c_str(),
		"iterFunc",
		1,
		headerSourceDir,
		headerNames
	);

	//One of the bugs encountered was here. GTX 1650 doesn't support Compute Capability 8.0 so we need to set it to 7.5
	//TODO: Get the compute capability of the device and set it accordingly
	const char* opts[] = { "--gpu-architecture=compute_52",
		 "--fmad=false", "--include-path=D:/include/"};
	PLOGD << "Compiling Kernel To CUDA PTX..\n";
	if (nvrtcCompileProgram(prog, 3, opts) != NVRTC_SUCCESS) {
		PLOGE << "Failed to compile NVRTC Program!";
	}

	size_t logSize;
	if (nvrtcGetProgramLogSize(prog, &logSize) != NVRTC_SUCCESS) {
		PLOGE << "Failed to get NVRTC Log Size!";
	}
	char* log = new char[logSize];
	nvrtcGetProgramLog(prog, log);
	// Obtain PTX from the program.
	size_t ptxSize;
	nvrtcGetPTXSize(prog, &ptxSize);
	std::string ptxLog(log, logSize);
	PLOGW << "Generated NVRTC Log: \n" << ptxLog;
	char* ptx = new char[ptxSize];
	nvrtcGetPTX(prog, ptx);
	PLOGD << "PTX Code generated!";
	
	CUmodule module;
	CUfunction kernel;
	cuInit(0);
	if (cuDeviceGet(&env->device, 0)!= CUDA_SUCCESS) {
		PLOGE << "Failed to get CUDA device!";
	}
	cuCtxCreate(&env->context, 0, env->device);
	auto moduleLoadResult = cuModuleLoadDataEx(&module, ptx, 0, 0, 0);
	PLOGW << "Module Load Result: " << cudaGetErrorString(static_cast<cudaError_t>(moduleLoadResult));

	auto kernelLoadResult = cuModuleGetFunction(&kernel, module, "iterFunc");
	PLOGW << "Kernel Load Result: " << cudaGetErrorString(static_cast<cudaError_t>(kernelLoadResult));
	//Kernel load was failing because I didn't use extern C which resulted in name mangling and the kernel not being found post compilation
	unsigned long long n = static_cast<unsigned long long>(preRenderConfigObj.imageDims.width) * static_cast<unsigned long long>(preRenderConfigObj.imageDims.height);
	unsigned long long bufferSize = n * sizeof(uchar4);
	int blockSize = 32;
	int gridSize = (n + blockSize - 1) / blockSize;
	uchar4* hostPtr = new uchar4[bufferSize];
	CUdeviceptr imgDevice;
	cuMemAlloc(&imgDevice, bufferSize);
	CUdeviceptr colorMapDevice;
	cuMemAlloc(&colorMapDevice, preRenderConfigObj.colorMapSize * sizeof(uchar4));
	cuMemcpyHtoD(colorMapDevice, preRenderConfigObj.colorMap, preRenderConfigObj.colorMapSize * sizeof(uchar4));

	void* args[] = {&imgDevice,  &preRenderConfigObj.cartesianBounds, &preRenderConfigObj.cartesianOrigin, &preRenderConfigObj.imageDims, &colorMapDevice, &preRenderConfigObj.colorMapSize, &preRenderConfigObj.maxBounces};

	env->module = module;
	env->kernel = kernel;
	env->ptx = ptx;
	env->ptxSize = ptxSize;
	env->gridSize = gridSize;
	env->blockSize = blockSize;
	env->imgDevice = imgDevice;
	env->imgHost = hostPtr;
	env->bufferSize = bufferSize;
	env->args = args;
	env->prog = prog;
}

void executePTXCode(ptxExecEnv runConfig) {
	//List out everything about the execution environment
	int gridSizeX = (IMAGE_X + runConfig.blockSize - 1) / runConfig.blockSize;
	int gridSizeY = (IMAGE_Y + runConfig.blockSize - 1) / runConfig.blockSize;
	
	auto result = cuLaunchKernel(
		runConfig.kernel,
		gridSizeX, gridSizeY, 1,
		runConfig.blockSize, runConfig.blockSize, 1,
		0,
		NULL,
		(void**)runConfig.args,
		NULL
		);

	PLOGW << "Kernel Launch Result: " << cudaGetErrorString(static_cast<cudaError_t>(result));
	cuCtxSynchronize();
	PLOGD << "Kernel Executed!";
	cuMemcpyDtoH(runConfig.imgHost, runConfig.imgDevice, runConfig.bufferSize);
}

void terminatePTXEnv(ptxExecEnv runConfig) {
	cuMemFree(runConfig.imgDevice);
	cuModuleUnload(runConfig.module);
	cuCtxDestroy(runConfig.context);
	nvrtcDestroyProgram(&runConfig.prog);
	delete[] runConfig.imgHost;
	delete[] runConfig.ptx;
}

//Initialize window:



int main() {
	//Initialize the logger
	plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender;
	plog::init(plog::verbose, &consoleAppender);


	//Setup the image
	imgDims imgConfig = { IMAGE_X, IMAGE_Y };
	float2 cartesianBounds = { 4, 4};
	float2 cartesianOrigin = { 0, 0 };
	int maxBounces = 1500;
	uchar4 thermalColorMap[] = {
	{0, 0, 0, 255}, // Black
	{0, 0, 255, 255}, // Blue
	{0, 255, 255, 255}, // Cyan
	{0, 255, 0, 255}, // Green
	{255, 255, 0, 255}, // Yellow
	{255, 0, 0, 255}, // Red
	{255, 255, 255, 255}, // White
	};

	preRenderConfig config = {
		cartesianBounds,
		cartesianOrigin, 
		imgConfig, 
		maxBounces, 
		thermalColorMap, 
		sizeof(thermalColorMap) / sizeof(uchar4),
		"z = z*z*z*z + c;"
	};

	PLOGD << "Color Map: " << sizeof(thermalColorMap) / sizeof(uchar4) << "colors";

	ptxExecEnv runConfig = initCuda();
	generatePTX(code, config, &runConfig);
	executePTXCode(runConfig);
	uchar4* img = runConfig.imgHost;
	////Write the image to a file
	//std::ofstream imgFile("D:/temp/img.ppm");
	//imgFile << "P3\n" << imgConfig.width << " " << imgConfig.height << "\n255\n";
	//for (int i = 0; i < imgConfig.width * imgConfig.height; i++) {
	//	imgFile << static_cast<int>(img[i].x) << " " << static_cast<int>(img[i].y) << " " << static_cast<int>(img[i].z) << "\n";
	//}


	sf::RenderWindow window(sf::VideoMode(IMAGE_X, IMAGE_Y), "Fractal Explorer");
	sf::Image sfmlImage;
	sfmlImage.create(imgConfig.width, imgConfig.height, reinterpret_cast<sf::Uint8*>(img));
	sf::Texture texture;
	texture.loadFromImage(sfmlImage);
	sf::Sprite sprite;
	sprite.setTexture(texture);

	ImGui::SFML::Init(window);

	sf::Event event;
	sf::Clock deltaClock;
	while (window.isOpen())
	{
		sf::Event event;
		while (window.pollEvent(event)) {
			ImGui::SFML::ProcessEvent(window, event);

			if (event.type == sf::Event::Closed) {
				window.close();
			}
		}

		ImGui::SFML::Update(window, deltaClock.restart());

		ImGui::Begin("Hello, world!");
		ImGui::SliderFloat2("Cartesian Bounds", &config.cartesianBounds.x, 0.0f, 10.0f);
		ImGui::SliderFloat2("Cartesian Origin", &config.cartesianOrigin.x, -10.0f, 10.0f);
		ImGui::SliderInt("Max Bounces", &config.maxBounces, 0, 10000);
		ImGui::InputText("Iteration Function", &config.iterationFunction);
	if (ImGui::Button("Render")) {
			generatePTX(code, config, &runConfig);
			executePTXCode(runConfig);
			sfmlImage.create(imgConfig.width, imgConfig.height, reinterpret_cast<sf::Uint8*>(runConfig.imgHost));
			texture.loadFromImage(sfmlImage);
			sprite.setTexture(texture);

		}
		ImGui::End();

		window.clear();
		window.draw(sprite);
		ImGui::SFML::Render(window);
		window.display();
	}

	ImGui::SFML::Shutdown();

	terminatePTXEnv(runConfig);
	return 0;
}