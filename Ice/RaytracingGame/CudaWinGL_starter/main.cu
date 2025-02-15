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
#include "triangle.h"
#include "quad.h"
#include "camera.h"
#include "material.h"
#include "hittable.h"
#include "hittable_list.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

// settings
#define WIDTH 1000
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
int ns = 3;
int tx = 24;
int ty = 24;
int num_pixels = WIDTH * HEIGHT;
// cam
float Yaw = -90;
float Pitch = 0;
vec3 lookfrom(0, 2, 20);
vec3 lookat(0, 2, 0);
float dist_to_focus = 10.0;
float aperture = 0.1;
//camera d_camera;


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
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    /*
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
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
    }
    */


}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}


bool closeCam = true;
bool spaceKeyPressed = false;

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_SPACE) {
        if (action == GLFW_PRESS && !spaceKeyPressed) {
            // Toggle the logic state
            closeCam = !closeCam;

            // Mark space key as pressed
            spaceKeyPressed = true;

        }
        else if (action == GLFW_RELEASE) {
            // Mark space key as released
            spaceKeyPressed = false;
        }
    }
}

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
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

    xoffset *= 0.3f;
    yoffset *= 0.3f;

    //Yaw += xoffset;
    //Pitch += yoffset;

    if (Pitch > 89.0f)
        Pitch = 89.0f;
    if (Pitch < -89.0f)
        Pitch = -89.0f;


    //d_camera.updateCam(0, Yaw, Pitch, lookfrom, lookat, vec3(0, 1, 0), 30.0, float(nx) / float(ny), aperture, dist_to_focus);


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

/*
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
*/

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

/*
__device__ color ray_color(const ray& r, hittable** world, curandState* local_rand_state)
{
    color accumulated_color(0, 0, 0);
    color attenuation(1, 1, 1);
    ray current_ray = r;
    color background_color = color(0.1f, 0.2f, 0.3f);

    for (int i = 0; i < 2; ++i) {
        hit_record rec;

        // Check if the ray hits anything in the scene.
        if (!((*world)->hit(current_ray, 0.001f, FLT_MAX, rec)))
        {
            accumulated_color += attenuation * background_color;
            break;
        }

        accumulated_color += attenuation * rec.mat_ptr->emitted(rec.u, rec.v, rec.p);

        ray scattered;
        color current_attenuation;

        if (!rec.mat_ptr->scatter(current_ray, rec, current_attenuation, scattered, local_rand_state))
        {
            break;
        }

        attenuation *= current_attenuation;
        current_ray = scattered;
    }

    return accumulated_color;
}
*/

#define START_COLLISION_INDEX 2
#define END_COLLISION_INDEX 12

#define START_ITEM_INDEX 13
#define END_ITEM_INDEX 28
const int HITTABLE_NUMS = END_ITEM_INDEX + 1;

__device__ vec3 ray_color(const ray& r, hittable** world, curandState* local_rand_state)
{
    color background_color = color(0.9f, 0.9f, 1.0f);
    ray cur_ray = r;

    vec3 cur_attenuation = vec3(0.5, 0.5, 0.5);


    for (int i = 0; i < 10; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
                //return vec3(0.0, 0.0, 0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * background_color;
            return cur_attenuation * c;
        }
    }
    return vec3(0.0, 0.0, 0.0); // exceeded recursion
}


__global__ void create_world(hittable** d_list, hittable** d_world, camera** d_camera, int nx, int ny, int num) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_camera = new camera();

        float groundSize = 16;
        float halfSize = groundSize / 2.0f;
        float quadSize = halfSize / 2.0f;
        float defaultY = 0.0f;
        float boundaryHeight = 1.0f;
        float wallHeight = 1.25f;

        vec3 wallColor = vec3(0.361, 0.702, 0.22);
        vec3 groundColor = vec3(0.984, 0.255, 0.255);

        // Ball
        d_list[0] = new sphere(vec3(0, defaultY + 0.4, -1), 0.4,
            new metal(vec3(0.3, 0.2, 0.15), 0.0));

        // Ground
        d_list[1] = new quad(vec3(-halfSize, defaultY, -halfSize), vec3(groundSize, 0, 0), vec3(0, 0, groundSize),
            //new lambertian(groundColor));
            new lambertian(texture(color(.05, .05, .05), color(.95, .95, .95))));

        // Boundary Walls

        d_list[2] = new quad(vec3(halfSize, defaultY, -halfSize), vec3(0, 0, groundSize), vec3(0, boundaryHeight, 0),
            new lambertian(wallColor));

        d_list[3] = new quad(vec3(-halfSize, defaultY, -halfSize), vec3(groundSize, 0, 0), vec3(0, boundaryHeight, 0),
            new lambertian(wallColor));

        d_list[4] = new quad(vec3(-halfSize, defaultY, halfSize), vec3(groundSize, 0, 0), vec3(0, boundaryHeight, 0),
            new lambertian(wallColor));

        d_list[5] = new quad(vec3(-halfSize, defaultY, -halfSize), vec3(0, 0, groundSize), vec3(0, boundaryHeight, 0),
            new lambertian(wallColor));

        // Interior walls

        d_list[6] = new quad(vec3(0 - quadSize, defaultY, 0), vec3(quadSize, 0, 0), vec3(0, wallHeight + 0.5, 0),
            new metal(vec3(0.8, 0.6, 0.2), 0.0));

        d_list[7] = new quad(vec3(-halfSize, defaultY, 2), vec3(quadSize, 0, 0), vec3(0, wallHeight, 0),
            new lambertian(vec3(0.2, 0.5, 0.6)));

        d_list[8] = new quad(vec3(-halfSize, defaultY, -2), vec3(quadSize, 0, 0), vec3(0, wallHeight, 0),
            new lambertian(vec3(0.2, 0.5, 0.6)));

        d_list[9] = new quad(vec3(0, defaultY, 2), vec3(quadSize, 0, quadSize), vec3(0, wallHeight, 0),
            new dielectric(1.5f));

        d_list[10] = new quad(vec3(quadSize, defaultY, 2), vec3(0, 0, quadSize), vec3(0, wallHeight, 0),
            new lambertian(vec3(0.2, 0.5, 0.6)));

        d_list[11] = new quad(vec3(0, defaultY, -halfSize), vec3(0.1, 0, quadSize), vec3(0, wallHeight, 0),
            new lambertian(vec3(0.2, 0.95, 0.2)));

        d_list[12] = new quad(vec3(0, defaultY, -2), vec3(quadSize, 0, quadSize), vec3(0, wallHeight, 0),
            new lambertian(vec3(0.95, 0.2, 0.2)));

        // Light items
        int count = 0;

        for (int i = START_ITEM_INDEX; i <= END_ITEM_INDEX; i++, count++)
        {
            int xOffset = count / 4;
            int zOffset = count % 4;

            d_list[i] = new sphere(vec3(
                halfSize - (xOffset * quadSize + 2.0),
                defaultY + 0.5,
                halfSize - (zOffset * quadSize + 1.0)), 0.15,
                new diffuse_light(vec3(0.8, 0.7, 0.2)));
        }

        *d_world = new hittable_list(d_list, num);
    }
}

__global__ void free_world(hittable** d_list, hittable** d_world, int num, camera** cam) {
    for (int i = 0; i < num; i++)
    {
        if (d_list[i]->type == hittable_type::Sphere)
        {
            delete ((sphere*)d_list[i])->mat_ptr;
        }

        else if (d_list[i]->type == hittable_type::Triangle)
        {
            delete ((triangle*)d_list[i])->mat_ptr;
        }

        else if (d_list[i]->type == hittable_type::Quad)
        {
            delete ((quad*)d_list[i])->mat_ptr;
        }

        delete d_list[i];
    }

    delete* d_world;
    delete* cam;
}

__global__ void render(unsigned int* fb, int max_x, int max_y, int ns, /*__grid_constant__ const*/ camera **cam, hittable** world, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);

    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        //ray r = (*cam)->get_ray(u, v);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += ray_color(r, world, &local_rand_state);
    }

    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = 255.99 * sqrt(col[0]);
    col[1] = 255.99 * sqrt(col[1]);
    col[2] = 255.99 * sqrt(col[2]);
    fb[pixel_index] = rgbToInt(col[0], col[1], col[2]);
}

__global__ void update_game(hittable** d_list, int moveInput, float dt, camera** cam, bool closeCam)
{
    if (d_list[0]->type == hittable_type::Sphere)
    {
        sphere* ball = (sphere*)d_list[0];

        vec3 oldVelocity = ball->velocity;

        float xVel = oldVelocity.x();
        float zVel = oldVelocity.z();

        float acc = 8.0f;
        float maxVelocity = 3.0f;
        
        if (moveInput & 1)
        {
            zVel += dt * acc;
        }
        if (moveInput & 2)
        {
            zVel -= dt * acc;
        }
        if (moveInput & 4)
        {
            xVel += dt * acc;

        }
        if (moveInput & 8)
        {
            xVel -= dt * acc;
        }

        xVel *= 0.98f;
        zVel *= 0.98f;

        if (zVel > maxVelocity)
            zVel = maxVelocity;
        else if (zVel < -maxVelocity)
            zVel = -maxVelocity;

        if (xVel > maxVelocity)
            xVel = maxVelocity;
        else if (xVel < -maxVelocity)
            xVel = -maxVelocity;

        ball->change_velocity(vec3(xVel, oldVelocity.y(), zVel));

        vec3 newPos = vec3(
            ball->center.x() + ball->velocity.x() * dt,
            ball->center.y() + ball->velocity.y() * dt,
            ball->center.z() + ball->velocity.z() * dt
        );

        for (int i = START_COLLISION_INDEX; i <= END_COLLISION_INDEX; i++)
        {
            if (d_list[i]->type == hittable_type::Quad)
            {
                quad* q = (quad*)d_list[i];

                {
                    float distToPlane = dot(newPos - q->Q, q->normal);

                    if (fabs(distToPlane) > ball->radius)
                        continue;

                    vec3 projectedCenter = newPos - distToPlane * q->normal;

                    // Express the projected point in the quad's local basis
                    vec3 relPoint = projectedCenter - q->Q;
                    float uProj = dot(relPoint, unit_vector(q->u)) / q->u.length();
                    float vProj = dot(relPoint, unit_vector(q->v)) / q->v.length();

                    vec3 closestPoint;

                    // Determine the closest point on the quad
                    if (uProj >= 0 && uProj <= 1 && vProj >= 0 && vProj <= 1) {
                        // Point is inside the quad
                        closestPoint = projectedCenter;
                    }
                    else {
                        // Point is outside; clamp to the quad edges
                        vec3 edgeClosest;
                        float minDist2 = FLT_MAX;

                        // Check all edges
                        vec3 corners[4] = { q->Q, q->Q + q->u, q->Q + q->v, q->Q + q->u + q->v };

                        for (int i = 0; i < 4; ++i)
                        {
                            vec3 edgeStart = corners[i];
                            vec3 edgeDir = corners[(i + 1) % 4] - corners[i];
                            float t = clamp(dot(newPos - edgeStart, edgeDir) / dot(edgeDir, edgeDir), 0.0f, 1.0f);
                            vec3 pointOnEdge = edgeStart + t * edgeDir;

                            float dist2 = (newPos - pointOnEdge).length();
                            if (dist2 < minDist2) {
                                minDist2 = dist2;
                                edgeClosest = pointOnEdge;
                            }
                        }

                        closestPoint = edgeClosest;
                    }

                    vec3 t = newPos - closestPoint;
                    float distToClosest = t.length();
                    if (distToClosest > ball->radius)
                        continue;


                    vec3 collisionNormal = unit_vector(t);

                    newPos = newPos + (ball->radius - distToClosest) * collisionNormal;

                    vec3 newVelocity = ball->velocity - 1.0f * dot(ball->velocity, q->normal) * q->normal;
                    ball->change_velocity(newVelocity);
                }

            }
        }

        ball->change_position(newPos);

        vec3 lookat = ball->center;
        vec3 lookfrom = vec3(0, 1.5, -4.0) + lookat;
        float dist_to_focus = 7.0f;
        float aperture = 0.05f;

        if (!closeCam)
        {
            lookfrom = vec3(0, 12.0f, -8.0f) + lookat;
            dist_to_focus = 14.0f;
        }


        (*cam)->setValue(lookfrom, lookat, vec3(0, 1, 0), 30.0, float(WIDTH) / float(HEIGHT), aperture, dist_to_focus);
    }
}

__global__ void check_collision(hittable** d_list)
{

    if (d_list[0]->type == hittable_type::Sphere)
    {
        sphere* ball = (sphere*)d_list[0];

        for (int i = START_ITEM_INDEX; i <= END_ITEM_INDEX; i++)
        {
            if (d_list[i]->type == hittable_type::Sphere)
            {
                sphere* item = (sphere*)d_list[i];

                if (!item->active)
                    continue;

                float distance = (ball->center - item->center).length();

                float sumRadius = ball->radius + item->radius;

                if (distance <= sumRadius)
                {
                    item->active = false;
                    ball->mat_ptr->set_color(vec3(0.1, 0.0f, 0.0f));
                }

            }

        }
    }
}


int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
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
    glfwSetKeyCallback(window, key_callback);

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

    // make our world of hitables & the camera
    hittable** d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, HITTABLE_NUMS * sizeof(hittable*)));
    hittable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));

    camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
    
    create_world << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, HITTABLE_NUMS);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // create cam
    //d_camera.updateCam(0, Yaw, Pitch, lookfrom, lookat, vec3(0, 1, 0), 30.0, float(nx) / float(ny), aperture, dist_to_focus);

    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render_init << <blocks, threads >> > (nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    float deltaTime = 0.0f;
    float lastFrame = 0.0f;

    float acc = 8.0f;
    float maxVelocity = 4.0f;

    float xVel = 0.0f;
    float zVel = 0.0f;

    int moveInput = 0;

    while (!glfwWindowShouldClose(window))//(Msg.message != WM_QUIT)
    {
        // --------------------
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;


        // update cam
       //processInput(window);

        moveInput = 0;

        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
        {
            moveInput |= 1;
        }
        else if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
        {
            moveInput |= 2;
        }

        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
        {
            moveInput |= 4;
        }
        else if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
        {
            moveInput |= 8;
        }

        update_game << <1, 1 >> > (d_list, moveInput, deltaTime, d_camera, closeCam);
        check_collision << <1, 1 >> > (d_list);


        glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // begin measure gpu
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        render << <blocks, threads >> > (cuda_dest_resource, nx, ny, ns, d_camera, d_world, d_rand_state);
        //render << <blocks, threads >> > (cuda_dest_resource, nx, ny, ns, d_camera, d_world, d_rand_state);
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

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world << <1, 1 >> > (d_list, d_world, HITTABLE_NUMS, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(cuda_dest_resource));

    cudaDeviceReset();

    glfwTerminate();
    return 0;
}
