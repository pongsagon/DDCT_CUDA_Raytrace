
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
#include <fcntl.h>
#include <io.h>


#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "triangle.h"
#include "camera.h"
#include "material.h"


void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

// settings
#define WIDTH 1200
#define HEIGHT 800
double prevTime = 0.0;
double currTime = 0.0;
float deltaTime = 0.0f;

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
int ns = 60;
int tx = 24;
int ty = 24;
int num_pixels = WIDTH * HEIGHT;
// cam
float Yaw = -90;
float Pitch = 0;
vec3 lookfrom(0, 2, 20);
vec3 lookat(0, 2, 0);
vec3 camForward;
float dist_to_focus = 20.0;
float aperture = 0.01;
float fov = 45.0f;  // Field of view
vec3 offset = vec3(0, 10, -15);
camera d_camera;
float sphereYaw = 0.0f;
float spherePitch = 0.0f;

spheres a_world;
triangles a_tri_world;


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

__device__ int clamp(int x, int a, int b) { return MAX(a, MIN(b, x)); }


template<typename T>
T clamp_host(T x, T a, T b) {
    return (x < a) ? a : (x > b) ? b : x;
}

// convert floating point rgb color to 8-bit integer
__device__ int rgbToInt(float r, float g, float b) {
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);

    return (int(b) << 16) | (int(g) << 8) | int(r);
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
vec3 velocity = vec3(0, 0, 0);
vec3 acceleration = vec3(0, 0, 0);
float thrust = 1.0f;  // Adjust thrust value to control acceleration
float friction = 0.98f;  // Friction or damping factor; adjust for desired resistance

void processInput(GLFWwindow* window, float deltaTime) {
    camForward = normalize(lookat - lookfrom); // Camera's forward vector
    vec3 right = normalize(cross(camForward, vec3(0, 1, 0))); // Camera's right vector
    vec3 up = vec3(0, 1, 0);  // Up vector for vertical movement

    float speedMultiplier = 1.0f;  // Default speed

    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
        speedMultiplier = 2.0f;  // Double the speed when shift is pressed
    }

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        acceleration += camForward * thrust * speedMultiplier;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        acceleration -= camForward * thrust * speedMultiplier;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        acceleration -= right * thrust * speedMultiplier;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        acceleration += right * thrust * speedMultiplier;
    }
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
        acceleration += up * thrust * speedMultiplier;  // Move up
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) {
        acceleration -= up * thrust * speedMultiplier;  // Move down
    }
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwTerminate();
    }
        
    // Apply acceleration to velocity
    velocity += acceleration;
    // Apply friction
    velocity *= friction;
    // Reset acceleration to 0 for the next frame
    acceleration = vec3(0, 0, 0);

    // Update sphere position based on velocity
    a_world.moveSphere(1, velocity * deltaTime); // Assuming sphere 1 is the player

}


void updateCamera() {
    vec3 front;
    front.setX(cos(glm::radians(Yaw)) * cos(glm::radians(Pitch)));
    front.setY(sin(glm::radians(Pitch)));
    front.setZ(sin(glm::radians(Yaw)) * cos(glm::radians(Pitch)));
    camForward = normalize(front); // Make sure this vector is normalized

    lookat = a_world.getSpherePosition(1) + camForward;  // Camera looks from sphere 1 position in forward direction
    lookfrom = lookat - camForward * dist_to_focus;  // Position camera back along the forward direction by dist_to_focus

    d_camera.setCamera(lookfrom, lookat, vec3(0, 1, 0), fov, float(WIDTH) / float(HEIGHT), aperture, dist_to_focus);
}


// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

void updateSphereOrientation(float xoffset, float yoffset) {
    // Update yaw and pitch based on mouse movement
    sphereYaw += xoffset;
    spherePitch += yoffset;

    // Constrain pitch to prevent flipping
    spherePitch = std::max(std::min(spherePitch, 89.0f), -89.0f);
}



void mouse_callback(GLFWwindow* window, double xposIn, double yposIn) {
    static float lastX = WIDTH / 2.0f;
    static float lastY = HEIGHT / 2.0f;
    static bool firstMouse = true;

    if (firstMouse) {
        lastX = xposIn;
        lastY = yposIn;
        firstMouse = false;
    }

    float xoffset = xposIn - lastX;
    float yoffset = lastY - yposIn; // Reversed since y-coordinates go from bottom to top
    lastX = xposIn;
    lastY = yposIn;

    float sensitivity = 0.1f; // Adjust sensitivity as needed
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    Yaw += xoffset;
    Pitch += yoffset;  // Reversed to align with the inverted y-axis

    // Constrain pitch to prevent screen flip
    Pitch = std::max(std::min(Pitch, 89.0f), -89.0f);
    updateSphereOrientation(xoffset, yoffset);
    // Update camera direction based on the new yaw and pitch
}


void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        vec3 shootDirection = camForward;  // Use updated camForward as the shooting direction
        float shootSpeed = 15.0f;  // Define the speed of the projectile
        a_world.shootSphere(shootDirection, shootSpeed);  // Ensure spheres are shot using this direction
    }
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



__device__ vec3 color(const ray& r, const spheres& world, const triangles& tri_world, curandState* local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    float closest_so_far = FLT_MAX;

    for (int i = 0; i < 10; i++) {

        hit_record rec_tri;
        hit_record rec_sphere;

        bool hitTri = tri_world.hitall(cur_ray, 0.001f, closest_so_far, rec_tri);
        bool hitSphere = world.hitall(cur_ray, 0.001f, closest_so_far, rec_sphere);

        if(hitTri && (!hitSphere || rec_tri.t < rec_sphere.t )) {
            ray scattered;
            vec3 attenuation;

            // Handle materials based on triangle hit
            switch (rec_tri.mat_type) {
            case 1:
                if (scatter_lambert(cur_ray, rec_tri, attenuation, scattered, local_rand_state)) {
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                }
                else {
                    return vec3(0.0, 0.0, 0.0);
                }
                break;
            case 2:
                if (scatter_metal(cur_ray, rec_tri, attenuation, scattered, local_rand_state)) {
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                }
                else {
                    return vec3(0.0, 0.0, 0.0);
                }
                break;
            case 3:
                if (scatter_dielectric(cur_ray, rec_tri, attenuation, scattered, local_rand_state)) {
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                }
                else {
                    return vec3(0.0, 0.0, 0.0);
                }
                break;
            case 4:
                if (scatter_lightmetal(cur_ray, rec_tri, attenuation, scattered, local_rand_state)) {
                    cur_attenuation *= attenuation;
                    return cur_attenuation *= attenuation;  // Returned twice, might be an error unless intended
                }
                else {
                    return vec3(0.0, 0.0, 0.0);
                }
                break;
            case 5:
                if (scatter_checkered(cur_ray, rec_tri, attenuation, scattered, local_rand_state)) {
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                }
                else {
                    return vec3(0.0, 0.0, 0.0);
                }
                break;
            case 7:
                if (scatter_stardust_galaxy(cur_ray, rec_tri, attenuation, scattered, local_rand_state)) {
                   
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                }
                else {
                    return vec3(0.0, 0.0, 0.0);
                }
                break;
                
            case 8:
                if (scatter_wave(cur_ray, rec_tri, attenuation, scattered, local_rand_state)) {
                   
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                    return cur_attenuation *= attenuation;
                }
                else {
                    return vec3(0.0, 0.0, 0.0);  // Return black if no scattering occurs
                }
                break;

            }
        }
        else if (hitSphere && (!hitTri || rec_sphere.t < rec_tri.t)) {
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
                case 4:
                    if (scatter_lightmetal(cur_ray, rec_sphere, attenuation, scattered, local_rand_state)) {
                        cur_attenuation *= attenuation;
                        return cur_attenuation *= attenuation;
                    }
                    else {
                        return vec3(0.0, 0.0, 0.0);
                    }
                    break;
                case 5:
                    if (scatter_checkered(cur_ray, rec_sphere, attenuation, scattered, local_rand_state)) {
                        cur_attenuation *= attenuation;
                        cur_ray = scattered;
                    }
                    else {
                        return vec3(0.0, 0.0, 0.0); // Absorb the light if no scattering
                    }
                    break;
				case 6:
					if (scatter_disco(cur_ray, rec_sphere, attenuation, scattered, local_rand_state)) {
						cur_attenuation *= attenuation;
						return cur_attenuation *= attenuation;
					}
					else {
						return vec3(0.0, 0.0, 0.0);
					}
					break;
                case 7:
                    if (scatter_stardust_galaxy(cur_ray, rec_sphere, attenuation, scattered, local_rand_state)) {
                        cur_attenuation *= attenuation;
                        cur_ray = scattered;
                    }
                    else {
                        return vec3(0.0, 0.0, 0.0);
                    }
                    break;

                default:
                    break;
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0,0.1, 0.1) * 1.0f + t * vec3(1, 1, 1) * 0.5f;
            return cur_attenuation * c;
        }
    }
    return vec3(0.0, 0.0, 0.0); // exceeded recursion
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
                        __grid_constant__ const spheres world, __grid_constant__ const triangles tri_world, curandState* rand_state) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        float u = (i + (s + curand_uniform(&local_rand_state)) / ns) / float(max_x);
        float v = (j + (s + curand_uniform(&local_rand_state)) / ns) / float(max_y);
        ray r = cam.get_ray(u, v, &local_rand_state);
        col += color(r, world,tri_world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = 255.99 * sqrt(col[0]);
    col[1] = 255.99 * sqrt(col[1]);
    col[2] = 255.99 * sqrt(col[2]);
    fb[pixel_index] = rgbToInt(col[0], col[1], col[2]);
}


void InitializeConsole() {
    AllocConsole();  // Allocates a new console for the calling process.

    // Redirect standard input, output, and error to the console
    freopen_s((FILE**)stdout, "CONOUT$", "w", stdout);
    freopen_s((FILE**)stdin, "CONIN$", "r", stdin);
    freopen_s((FILE**)stderr, "CONOUT$", "w", stderr);

    std::cout << "Debugging Console Initialized\n"; // Test message
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
    LPSTR lpCmdLine, int nCmdShow)
{
    InitializeConsole();
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
    glfwSetMouseButtonCallback(window, mouse_button_callback);


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
    

    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render_init << <blocks, threads >> > (nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);  // Hide cursor and capture it

    while (!glfwWindowShouldClose(window)) {
        currTime = glfwGetTime();
        float deltaTime = currTime - prevTime;
        prevTime = currTime;


        processInput(window,deltaTime);
        updateCamera();
        glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // begin measure gpu
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        vec3 playerPosition = a_world.getSpherePosition(1); // Get player's current position
      

        a_world.updateSpheres(deltaTime, playerPosition, a_tri_world.getRamielCenter());
        a_tri_world.update_position(deltaTime);
        render << <blocks, threads >> > (cuda_dest_resource, nx, ny, ns, d_camera, a_world,a_tri_world, d_rand_state);
        //updateCamera();
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

        
      
       
        d_camera.followTarget(a_world.getSpherePosition(1), offset, vec3(0, 10, 0), 45.0, float(WIDTH) / float(HEIGHT), 0.1, 10.0);
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
