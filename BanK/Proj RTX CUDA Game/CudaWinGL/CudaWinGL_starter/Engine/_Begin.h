#include <iostream>
#include <time.h>
#include <float.h>
#include <vector>
#include <random>
using namespace std;

#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))


#define STB_IMAGE_IMPLEMENTATION
#include "includes/stb_image.h"
#include "includes/KHR/khrplatform.h"
#include "includes/glew.h"
#include "includes/GLFW/glfw3.h"
#include "includes/GLFW/glfw3native.h"
#include "includes/shader.h"



// must include after gl lib
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"

#include <windows.h>
#include <sstream>
#include <iostream>
#include <time.h>
#include <float.h>


#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"





struct Time {
	float Deltatime = 0;
	float TrueDeltatime = 0;
	float CurrentTime = 0;
	double CurrentFrameTime = 0;
	int Frame = 0;
	float Fps = 0;
	float Scale = 1;

	struct Advanced {
		double lastFrameTime = 0;
		double lastSecondTime = 0;
		int framesInLastSecond = 0;
	} Advanced;

	void Calculate() {
		//float currentFrame = static_cast<float>(glfwGetTime());
		//deltaTime = currentFrame - lastFrame;
		//lastFrame = currentFrame;

		CurrentFrameTime = glfwGetTime();
		TrueDeltatime = static_cast<float>(CurrentFrameTime - Advanced.lastFrameTime); // Time in seconds
		Deltatime = TrueDeltatime * Scale;
		Advanced.lastFrameTime = CurrentFrameTime;
		CurrentTime += Deltatime;
		Frame++;

		Fps = 1.0f / Deltatime;
		//std::cout << "FPS: " << Fps << std::endl;
	}
}Time;



vec3 normalize(const vec3& v) {
    float length = sqrt(v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
    if (length > 0) {
        return v / length;
    }
    return v; // Return the original vector if length is zero to avoid division by zero
}




float B_frand(float min, float max) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(min, max);
	return dis(gen);
}
int B_irand(int min, int max) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dis(min, max);
	return dis(gen);
}

float B_distance1D(float A, float B) {
	return abs(A - B);
}
float B_distance2D(glm::vec2& p1, const glm::vec2& p2) {
	float dx = p2.x - p1.x;
	float dy = p2.y - p1.y;
	return std::sqrt(dx * dx + dy * dy);
}
float B_distance2D(float x1, float y1, float x2, float y2) {
	float dx = x2 - x1;
	float dy = y2 - y1;
	return std::sqrt(dx * dx + dy * dy);
}
float B_distance3D(float x1, float y1, float z1, float x2, float y2, float z2) {
	float dx = x2 - x1;
	float dy = y2 - y1;
	float dz = z2 - z1;
	return std::sqrt(dx * dx + dy * dy + dz * dz);
}

float B_normalize(float value, float min_val, float max_val) {
	return (value - min_val) / (max_val - min_val);
}
float B_normalize_reversed(float value, float min_val, float max_val) {
	return (value - max_val) / (min_val - max_val);
}
float B_clamp(float value, float minValue, float maxValue) {
	//max(A,B) returns A if A is the max value
	return std::max(minValue, std::min(value, maxValue));
}
float B_clampLoop(float value, float minValue, float maxValue) {
	if (value > maxValue) {
		return minValue;
	}
	else if (value < minValue) {
		return maxValue;
	}
	return value;
}
float B_clampSkin(float value, float minValue, float maxValue) {
	float middle = (minValue + maxValue) / 2.0f;
	if (value < middle) { return minValue; }
	return maxValue;
}

float B_lerp(float start, float end, float t) {
	//t = B_clamp(t, 0, 1);
	return start + t * (end - start);
}
glm::vec3 B_lerpVec3(glm::vec3 A, glm::vec3 B, float t) {
	return (1.0f - t) * A + t * B;
}

float B_SnapToGrid(float& V, float gridSize) {
	return gridSize * std::roundf(V / gridSize);
}

 