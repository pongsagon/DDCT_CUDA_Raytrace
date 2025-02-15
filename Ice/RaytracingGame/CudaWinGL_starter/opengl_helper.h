#pragma once
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

#include "vec3_test.h"
#include "camera.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

// settings
#define WIDTH 1200
#define HEIGHT 800
double prevTime = 0.0;
double currTime = 0.0;


int nx = WIDTH;
int ny = HEIGHT;
int ns = 2;
int tx = 24;
int ty = 24;
int num_pixels = WIDTH * HEIGHT;

float Yaw = -90;
float Pitch = 0;
vec3 lookfrom(0, 2, 20);
vec3 lookat(0, 2, 0);
float dist_to_focus = 10.0;
float aperture = 0.1;
camera d_camera;


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
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

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

    Yaw += xoffset;
    Pitch += yoffset;

    if (Pitch > 89.0f)
        Pitch = 89.0f;
    if (Pitch < -89.0f)
        Pitch = -89.0f;


    d_camera.updateCam(0, Yaw, Pitch, lookfrom, lookat, vec3(0, 1, 0), 30.0, float(nx) / float(ny), aperture, dist_to_focus);


}


int InitOpenGL()
{
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
}