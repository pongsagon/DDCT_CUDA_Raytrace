#include <iostream>
#include <time.h>
#include <float.h>


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
#include "triangle.h"


void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

// settings
#define WIDTH 1200
#define HEIGHT 800
double prevTime = 0.0;
double currTime = 0.0;

// Windows
HWND handle;
WNDPROC currentWndProc;
MSG Msg;
WNDPROC btnWndProc;
std::stringstream ss;

// Touch
#define MAXPOINTS 20
int points[MAXPOINTS][2];       // touch coor
int diff_points[MAXPOINTS][2];  // touch offset each frame
int idLookup[MAXPOINTS];
int last_points[MAXPOINTS][2];

// mouse
float lastX = WIDTH / 2.0f;
float lastY = HEIGHT / 2.0f;
bool firstMouse = true;


// cuda opengl interop
GLuint shDrawTex;           // shader
GLuint tex_cudaResult;      // result texture to glBindTexture(GL_TEXTURE_2D, texture);
unsigned int* cuda_dest_resource;  // output from cuda
struct cudaGraphicsResource* cuda_tex_result_resource;


// cuda rt
int nx = WIDTH;
int ny = HEIGHT;
int ns = 16;
int tx = 24;
int ty = 24;
int num_pixels = WIDTH * HEIGHT;
// cam
float Yaw = -90;
float Pitch = 0;
vec3 lookfrom(0, 5, 20);
vec3 lookat(0, 2, 0);
float dist_to_focus = 20.0;
float aperture = 0.01;
camera d_camera;

// create a world of spheres
spheres a_world;
triangle triangles;

//HANDLING GAME STATE

void warpToRoomCenter(spheres& world, int targetRoomID, vec3& lookFrom, vec3& lookAt, const vec3& offset = vec3(0.0f, 0.0f, 0.0f), const vec3& setlookAt = vec3(0.0f, 0.0f, -1.0f)) {

    Room targetRoom = world.rooms[targetRoomID];
    world.currentRoomID = targetRoomID;

    vec3 roomCenter = (targetRoom.minCorner + targetRoom.maxCorner) * 0.5f;
    roomCenter.e[1] = lookFrom.e[1];

    lookFrom = roomCenter + offset;
    lookAt = setlookAt;

    world.update_current_room(lookfrom);

    d_camera.updateCam(0, Yaw, Pitch, lookfrom, lookat, vec3(0, 1, 0), 30.0, float(nx) / float(ny), aperture, dist_to_focus);

}

enum ProgramState {
    ROOM_1,           // Initial state in Room 1
    PILLAR_TRIGGERED, // Pillar has changed color
    ROOM_2_CORRIDOR,  // In the corridor of Room 2
    ROOM_3  // Transitioning to Room 3
};

ProgramState currentState = ROOM_1;

void updateGameState(spheres& sphere_world, triangle& triangle_world, vec3& lookFrom, vec3& lookAt, float deltaTime) {

    switch (currentState) {
    case ROOM_1:
        // Check for orb-ceiling color match

        sphere_world.room1_sphere_follow_camera(deltaTime, lookfrom, lookat);
        //triangle_world.updateCeilingLight();

        if (sphere_world.check_orb_match(triangle_world.getCurrentCeilingColorIndex(), lookFrom)) {
            triangle_world.changePillarColor();
            currentState = PILLAR_TRIGGERED;
            std::cout << "Pillar color triggered!\n";
        }

        break;

    case PILLAR_TRIGGERED:

        sphere_world.room1_sphere_follow_camera(deltaTime, lookfrom, lookat);
        // Check if player enters the pillar (center of room)
        if (sphere_world.isCameraNearPillar(lookFrom)) {
            warpToRoomCenter(sphere_world, 1, lookFrom, lookAt, vec3(0.0f, 0.0f, -95.0f)); // Warp to Room 2
            currentState = ROOM_2_CORRIDOR;
        }
        break;

    case ROOM_2_CORRIDOR:
        // Check if the camera has reached the end of the corridor
        if (sphere_world.isCameraNearCorridorEnd(lookFrom)) {
            warpToRoomCenter(sphere_world, 2, lookFrom, lookAt, vec3(0.0f, 15.0f, 0.0f), vec3(0.0f,0.0f,0.0f)); // Warp to Room 3
            currentState = ROOM_3;
        }
        break;

    case ROOM_3:

        // Final state or new logic for Room 3

        sphere_world.room3_followcam(deltaTime, lookfrom, lookat);
         
        sphere_world.update_moving_spheres(deltaTime);

        break;

    }

}


// ---------------------------------------
// Touch handler
// ---------------------------------------

// This function is used to return an index given an ID
int GetContactIndex(int dwID) {
    for (int i = 0; i < MAXPOINTS; i++) {
        if (idLookup[i] == dwID) {
            return i;
        }
    }

    for (int i = 0; i < MAXPOINTS; i++) {
        if (idLookup[i] == -1) {
            idLookup[i] = dwID;
            return i;
        }
    }
    // Out of contacts
    return -1;
}

// Mark the specified index as initialized for new use
BOOL RemoveContactIndex(int index) {
    if (index >= 0 && index < MAXPOINTS) {
        idLookup[index] = -1;
        return true;
    }

    return false;
}

LRESULT OnTouch(HWND hWnd, WPARAM wParam, LPARAM lParam) {
    BOOL bHandled = FALSE;
    UINT cInputs = LOWORD(wParam);
    PTOUCHINPUT pInputs = new TOUCHINPUT[cInputs];
    POINT ptInput;
    if (pInputs) {
        if (GetTouchInputInfo((HTOUCHINPUT)lParam, cInputs, pInputs, sizeof(TOUCHINPUT))) {
            for (UINT i = 0; i < cInputs; i++) {
                TOUCHINPUT ti = pInputs[i];
                int index = GetContactIndex(ti.dwID);
                if (ti.dwID != 0 && index < MAXPOINTS) {

                    // Do something with your touch input handle
                    ptInput.x = TOUCH_COORD_TO_PIXEL(ti.x);
                    ptInput.y = TOUCH_COORD_TO_PIXEL(ti.y);
                    ScreenToClient(hWnd, &ptInput);

                    if (ti.dwFlags & TOUCHEVENTF_UP) {
                        points[index][0] = -1;
                        points[index][1] = -1;
                        last_points[index][0] = -1;
                        last_points[index][1] = -1;
                        diff_points[index][0] = 0;
                        diff_points[index][1] = 0;

                        // Remove the old contact index to make it available for the new incremented dwID.
                        // On some touch devices, the dwID value is continuously incremented.
                        RemoveContactIndex(index);
                    }
                    else {
                        if (points[index][0] > 0) {
                            last_points[index][0] = points[index][0];
                            last_points[index][1] = points[index][1];
                        }

                        points[index][0] = ptInput.x;
                        points[index][1] = ptInput.y;

                        if (last_points[index][0] > 0) {
                            diff_points[index][0] = points[index][0] - last_points[index][0];
                            diff_points[index][1] = points[index][1] - last_points[index][1];
                        }
                    }
                }
            }
            bHandled = TRUE;
        }
        else {
            /* handle the error here */
        }
        delete[] pInputs;
    }
    else {
        /* handle the error here, probably out of memory */
    }
    if (bHandled) {
        // if you handled the message, close the touch input handle and return
        CloseTouchInputHandle((HTOUCHINPUT)lParam);
        return 0;
    }
    else {
        // if you didn't handle the message, let DefWindowProc handle it
        return DefWindowProc(hWnd, WM_TOUCH, wParam, lParam);
    }
}

LRESULT CALLBACK SubclassWindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
    case WM_TOUCH:
        OnTouch(hWnd, wParam, lParam);
        break;
    case WM_LBUTTONDOWN:
    {

    }
    break;
    case WM_CLOSE:
        DestroyWindow(hWnd);
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    }

    return CallWindowProc(btnWndProc, hWnd, uMsg, wParam, lParam);
}




// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{  

    /*if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        d_camera.updateCam(1, Yaw, Pitch, lookfrom, lookat, vec3(0, 1, 0), 30.0, float(nx) / float(ny), aperture, dist_to_focus);
    }

    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        d_camera.updateCam(2, Yaw, Pitch, lookfrom, lookat, vec3(0, 1, 0), 30.0, float(nx) / float(ny), aperture, dist_to_focus);
    }

    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        d_camera.updateCam(3, Yaw, Pitch, lookfrom, lookat, vec3(0, 1, 0), 30.0, float(nx) / float(ny), aperture, dist_to_focus);
    }

    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        d_camera.updateCam(4, Yaw, Pitch, lookfrom, lookat, vec3(0, 1, 0), 30.0, float(nx) / float(ny), aperture, dist_to_focus);
    }*/
    

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    

    // Adjust these values to modify the speed and sensitivity
    float moveSpeed = 0.2f;

    // Compute forward and right direction vectors based on yaw angle
    vec3 forward(cos(glm::radians(Yaw)) * moveSpeed, 0.0, sin(glm::radians(Yaw)) * moveSpeed);
    vec3 right(sin(glm::radians(Yaw)) * moveSpeed, 0.0, -cos(glm::radians(Yaw)) * moveSpeed);

    vec3 newPosition = lookfrom; // Start with the current position

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        newPosition += forward;
    }

    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        newPosition -= forward;
    }

    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        newPosition += right;
    }

    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        
        newPosition -= right;
    }

    /// Get the current room bounds from the spheres class

    int currentRoomID = a_world.currentRoomID;

    // Switch to Room 1 and update the position
    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
        //currentRoomID = 0; // Room 1
        //vec3 roomCenter = (a_world.rooms[0].minCorner + a_world.rooms[0].maxCorner) * 0.5f;
        //float zOffset = 3.0f;
        //roomCenter.e[2] += zOffset;
        //roomCenter.e[1] = lookfrom.e[1]; // Preserve current camera height
        //lookfrom = roomCenter;          // Move the camera to the center of Room 1
        //a_world.update_current_room(lookfrom); // Update the current room
        //return; // Skip further processing

        warpToRoomCenter(a_world, 0, lookfrom, lookat, vec3(0,0,-3.0f));
        return;

    }

    if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {
        //currentRoomID = 1; // Room 2
        //vec3 roomCenter = (a_world.rooms[1].minCorner + a_world.rooms[1].maxCorner) * 0.5f;
        //roomCenter.e[1] = lookfrom.e[1]; // Preserve current camera height
        //lookfrom = roomCenter;          // Move the camera to the center of Room 2
        //a_world.update_current_room(lookfrom); // Update the current room
        //return; // Skip further processing

        warpToRoomCenter(a_world, 1, lookfrom, lookat, vec3(0.0f, 0.0f, -95.0f), vec3(0.0f, 3.0f, 210.0f)); // Warp to Room 2;
        currentState = ROOM_2_CORRIDOR;

        return;

    }

    if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) {
        //currentRoomID = 1; // Room 2
        //vec3 roomCenter = (a_world.rooms[1].minCorner + a_world.rooms[1].maxCorner) * 0.5f;
        //roomCenter.e[1] = lookfrom.e[1]; // Preserve current camera height
        //lookfrom = roomCenter;          // Move the camera to the center of Room 2
        //a_world.update_current_room(lookfrom); // Update the current room
        //return; // Skip further processing

        warpToRoomCenter(a_world, 2, lookfrom, lookat, vec3(0.0f, 10.0f, 0.0f), vec3(110.0f, 0.0f, -60.0f)); // Warp to Room 3
        currentState = ROOM_3;
        return;

    }

    // Handle room-specific logic
    currentRoomID = a_world.currentRoomID;
    // Get the current room bounds from the spheres class
    vec3 roomMin = a_world.rooms[currentRoomID].minCorner;
    vec3 roomMax = a_world.rooms[currentRoomID].maxCorner;


    if (currentRoomID == 1) { // Room 2: Restrict movement

        float boundaryOffset = 1.0f; // Adjust this value as needed to prevent "seeing through" the boundary

        // Check boundaries and restrict position with offset
        if (newPosition.e[0] < roomMin.x() + boundaryOffset)
            newPosition.e[0] = roomMin.x() + boundaryOffset;
        if (newPosition.e[0] > roomMax.x() - boundaryOffset)
            newPosition.e[0] = roomMax.x() - boundaryOffset;
        if (newPosition.e[2] < roomMin.z() + boundaryOffset)
            newPosition.e[2] = roomMin.z() + boundaryOffset;
        if (newPosition.e[2] > roomMax.z() - boundaryOffset)
            newPosition.e[2] = roomMax.z() - boundaryOffset;


    }
    else { // Room 1 and Room 3: Enable warping mechanism

        // Check boundaries and warp position if necessary
        if (newPosition.e[0] < roomMin.x()) {
            triangles.updateCeilingLight();
            newPosition.e[0] = roomMax.x(); // Warp to the opposite X boundary
        }
        if (newPosition.e[0] > roomMax.x()) {
            triangles.updateCeilingLight();
            newPosition.e[0] = roomMin.x(); // Warp to the opposite X boundary
        }
        if (newPosition.e[2] < roomMin.z()) {
            triangles.updateCeilingLight();
            newPosition.e[2] = roomMax.z(); // Warp to the opposite Z boundary
        }
        if (newPosition.e[2] > roomMax.z()) {
            triangles.updateCeilingLight();
            newPosition.e[2] = roomMin.z(); // Warp to the opposite Z boundary
        }

    }

    lookfrom = newPosition;

    float cosPitch = cos(glm::radians(Pitch));
    lookat.e[0] = lookfrom.e[0] + cos(glm::radians(Yaw)) * cosPitch;
    lookat.e[1] = lookfrom.e[1] + sin(glm::radians(Pitch));
    lookat.e[2] = lookfrom.e[2] + sin(glm::radians(Yaw)) * cosPitch;

    d_camera.updateCam(0, Yaw, Pitch, lookfrom, lookat, vec3(0, 1, 0), 30.0, float(nx) / float(ny), aperture, dist_to_focus);

}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{

    //if (currentState == ROOM_3) {
    //    // Lock mouse movement in Room 3
    //    return;
    //}


    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE)
    {
        firstMouse = true;
        return;
    }

    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;   
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    xoffset *= 0.2f;
    yoffset *= 0.2f;

    Yaw += xoffset;
    Pitch += yoffset;

    if (Pitch > 89.0f)
        Pitch = 89.0f;
    if (Pitch < -89.0f)
        Pitch = -89.0f;

   d_camera.updateCam(0, Yaw, Pitch, lookfrom, lookat, vec3(0, 1, 0), 30.0, float(nx) / float(ny), aperture, dist_to_focus);

       
}



// ---------------------------------------
// CUDA code
// ---------------------------------------



// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        // Make sure we call CUDA Device Reset before exiting
        ss << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        OutputDebugStringA(ss.str().c_str());
        ss.str("");
        cudaDeviceReset();
        exit(99);
    }
}

__device__ int clamp(int x, int a, int b) { return MAX(a, MIN(b, x)); }

// convert floating point rgb color to 8-bit integer
__device__ int rgbToInt(float r, float g, float b) {
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);

    return (int(b) << 16) | (int(g) << 8) | int(r);
}



__device__ vec3 color(const ray& r, const spheres& sphere_world, const triangle& tri_world, curandState* local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);

    for (int i = 0; i < 10; i++) {
        hit_record rec_sphere, rec_triangle;
        bool hit_sphere = sphere_world.hitall(cur_ray, 0.001f, FLT_MAX, rec_sphere);
        bool hit_triangle = tri_world.hitall(cur_ray, 0.001f, FLT_MAX, rec_triangle);

        if (hit_sphere && (!hit_triangle || rec_sphere.t < rec_triangle.t)) {
            // Sphere was hit and is closer (or triangle wasn't hit)
            ray scattered;
            vec3 attenuation;

            switch (rec_sphere.mat_type) {
            case 1:
                if (scatter_lambert(cur_ray, rec_sphere, attenuation, scattered, local_rand_state)) {
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                }
                else {
                    return vec3(0.0, 0.0, 0.0);
                }
                break;
            case 2:
                if (scatter_metal(cur_ray, rec_sphere, attenuation, scattered, local_rand_state)) {
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                }
                else {
                    return vec3(0.0, 0.0, 0.0);
                }
                break;
            case 3:
                if (scatter_dielectric(cur_ray, rec_sphere, attenuation, scattered, local_rand_state)) {
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                }
                else {
                    return vec3(0.0, 0.0, 0.0);
                }
                break;
            case 4: // CHECKERED
                if (scatter_checkered(cur_ray, rec_sphere, attenuation, scattered, local_rand_state)) {
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                }
                else {
                    return vec3(0.0, 0.0, 0.0);
                }
                break;
            case 5: // Light source
                if (scatter_metal(cur_ray, rec_sphere, attenuation, scattered, local_rand_state)) {
                    cur_attenuation *= attenuation;
                    return cur_attenuation * attenuation;
                }
                else {
                    return vec3(0.0, 0.0, 0.0);
                }
                break;

            default:
                break;
            }
        }
        else if (hit_triangle) {
            // Triangle was hit (and sphere was either not hit or farther away)
            ray scattered;
            vec3 attenuation;

            switch (rec_triangle.mat_type) {
            case 1:
                if (scatter_lambert(cur_ray, rec_triangle, attenuation, scattered, local_rand_state)) {
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                }
                else {
                    return vec3(0.0, 0.0, 0.0);
                }
                break;
            case 2:
                if (scatter_metal(cur_ray, rec_triangle, attenuation, scattered, local_rand_state)) {
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                }
                else {
                    return vec3(0.0, 0.0, 0.0);
                }
                break;
            case 3:
                if (scatter_dielectric(cur_ray, rec_triangle, attenuation, scattered, local_rand_state)) {
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                }
                else {
                    return vec3(0.0, 0.0, 0.0);
                }
                break;
            case 5: //light source
                if (scatter_metal(cur_ray, rec_triangle, attenuation, scattered, local_rand_state)) {
                    cur_attenuation *= attenuation;
                    return cur_attenuation * attenuation;
                }
                else {
                    return vec3(0.0, 0.0, 0.0);
                }
                break;
            case 6:
                if (scatter_squiggly_wave(cur_ray, rec_triangle, attenuation, scattered, local_rand_state)) {
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                }
                else {
                    return vec3(0.0, 0.0, 0.0);  // No contribution if scattering fails
                }
                break;
            case 7:
                if (scatter_checkered(cur_ray, rec_triangle, attenuation, scattered, local_rand_state)) {
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                }
                else {
                    return vec3(0.0, 0.0, 0.0);  // No contribution if scattering fails
                }
                break;
           

            default:
                break;
            }
        }
        else {
            // No hits, return sky color
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.7, 0.0, 0.0);
            return cur_attenuation * c;
        }
    }

    return vec3(0.0, 0.0, 0.0); // exceeded recursion limit
}




__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    //int pixel_index = j;
    // Original: Each thread gets same seed, a different sequence number, no offset
    // curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}


__global__ void render(unsigned int* fb, int max_x, int max_y, int ns, __grid_constant__ const camera cam,
                        __grid_constant__ const spheres sphere_world, __grid_constant__ const triangle tri_world, curandState* rand_state) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);


    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = cam.get_ray(u, v, &local_rand_state);
        col += color(r, sphere_world, tri_world, &local_rand_state);
    }

    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = 255.99 * sqrt(col[0]);
    col[1] = 255.99 * sqrt(col[1]);
    col[2] = 255.99 * sqrt(col[2]);
    fb[pixel_index] = rgbToInt(col[0], col[1], col[2]);

}


int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
    LPSTR lpCmdLine, int nCmdShow)
{

#pragma region "gl setup"
    // ------------------------------
    // glfw: initialize and configure
    // ------------------------------

    // not need
    //cudaGLSetGLDevice(0);
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);


    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "LearnOpenGL", NULL, NULL);

    handle = glfwGetWin32Window(window);
    btnWndProc = (WNDPROC)SetWindowLongPtrW(handle, GWLP_WNDPROC, (LONG_PTR)SubclassWindowProc);
    int touch_success = RegisterTouchWindow(handle, 0);

    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // set this to 0, will swap at fullspeed, but app will close very slow, sometime hang
    glfwSwapInterval(1);

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        getchar();
        glfwTerminate();
        return -1;
    }

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);

    // init touch data
    for (int i = 0; i < MAXPOINTS; i++) {
        points[i][0] = -1;
        points[i][1] = -1;
        last_points[i][0] = -1;
        last_points[i][1] = -1;
        diff_points[i][0] = 0;
        diff_points[i][1] = 0;
        idLookup[i] = -1;
    }


    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    Shader ourShader("tex.vs", "tex.fs");

    float vertices[] = {
        // positions          // texture coords
         1.0f,  1.0f, 0.0f,   1.0f, 1.0f, // top right
         1.0f, -1.0f, 0.0f,   1.0f, 0.0f, // bottom right
        -1.0f, -1.0f, 0.0f,   0.0f, 0.0f, // bottom left
        -1.0f,  1.0f, 0.0f,   0.0f, 1.0f  // top left 
    };
    unsigned int indices[] = {
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };
    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);


    // cuda mem out bind to tex
    // ---------------------------------------
    int num_texels = WIDTH * HEIGHT;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;
    checkCudaErrors(cudaMalloc((void**)&cuda_dest_resource, size_tex_data));

    // create a texture, output from cuda
    glGenTextures(1, &tex_cudaResult);
    glBindTexture(GL_TEXTURE_2D, tex_cudaResult);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_tex_result_resource, tex_cudaResult, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));

    // fps
    prevTime = glfwGetTime();
#pragma endregion "gl setup"


    // ------------------------------
    // CUDA: RT
    // ------------------------------


    // allocate random state
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));

    // create cam
    d_camera.updateCam(0, Yaw, Pitch, lookfrom, lookat, vec3(0, 1, 0), 30.0, float(nx) / float(ny), aperture, dist_to_focus);

    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render_init << <blocks, threads >> > (nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());



    while (!glfwWindowShouldClose(window))//(Msg.message != WM_QUIT)
    {
        // update cam
        processInput(window);

        glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // begin measure gpu
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);       

        render << <blocks, threads >> > (cuda_dest_resource, nx, ny, ns, d_camera, a_world, triangles, d_rand_state);

        checkCudaErrors(cudaGetLastError());
        //checkCudaErrors(cudaDeviceSynchronize());
        // 
        // copy cuda_dest_resource data to the texture
        cudaArray* texture_ptr;
        checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_result_resource, 0));
        checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_result_resource, 0, 0));
        checkCudaErrors(cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dest_resource, size_tex_data, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_result_resource, 0));
    

        // end measure gpu
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        ss << elapsedTime << "ms\n";
        OutputDebugStringA(ss.str().c_str());
        ss.str("");
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        //ROOM 1
        
        //triangles.updateCeilingLight();

        //int currentCeilingColorIndex = triangles.getCurrentCeilingColorIndex();
        //if (a_world.check_orb_match(currentCeilingColorIndex, lookfrom)) {
        //    // Call the function in the triangle class to change the pillar color
        //    triangles.changePillarColor();
        //}

        //ROOM 2   
        //a_world.room2_check_collision(1.5f, 6.0f, lookfrom, lookat);
        //a_world.update_floating_spheres(elapsedTime);
        
        updateGameState(a_world, triangles, lookfrom, lookat, elapsedTime);
        

        // render gl
        glUniform1i(glGetUniformLocation(ourShader.ID, "texture1"), 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex_cudaResult);
        ourShader.use();
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();

        // fps
        prevTime = currTime;
    }


    // Free the device memory
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    cudaFree(cuda_dest_resource);
    cudaFree(d_rand_state);
    cudaDeviceReset();

    glfwTerminate();
    return 0;
}
