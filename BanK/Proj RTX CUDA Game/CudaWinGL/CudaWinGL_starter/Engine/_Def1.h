#include "_Begin.h"






// settings
#define WIDTH 1200
#define HEIGHT 600
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
int ns = 4;
int tx = 24;
int ty = 24;
int num_pixels = WIDTH * HEIGHT;
// cam
float Yaw = -90;
float Pitch = 0;
vec3 lookfrom(0, 2, 20);
vec3 lookat(0, 2, 0);
float dist_to_focus = 10.0;
float aperture = 0.02;
camera d_camera;

 

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
    // Ensure the cursor is locked in the window (set up once during initialization)
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    // Calculate the offsets based on the mouse movement
    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    // Update the last known mouse position
    lastX = xpos;
    lastY = ypos;

    // Apply sensitivity to the offsets
    xoffset *= 0.8f;
    yoffset *= 0.8f;

    // Calculate the target yaw and pitch
    float TargetYaw = Yaw + xoffset;
    float TargetPitch = Pitch + yoffset;

    // Smoothly interpolate towards the target yaw and pitch
    float LerpSpeed = Time.Deltatime * 5;
    Yaw = B_lerp(Yaw, TargetYaw, LerpSpeed);
    Pitch = B_lerp(Pitch, TargetPitch, LerpSpeed);

    // Clamp the pitch to prevent the camera from flipping
    if (Pitch > 89.0f)
        Pitch = 89.0f;
    if (Pitch < -89.0f)
        Pitch = -89.0f;

    // Update the camera
    d_camera.updateCam(0, Yaw, Pitch, lookfrom, lookat, vec3(0, 1, 0), 30.0, float(nx) / float(ny), aperture, dist_to_focus);
}


















class InputEvent;
vector <InputEvent*> SystemInputList;
class InputEvent {

public:
    InputEvent(int GLFW_Key)
        :My_GLFW_Key(GLFW_Key) {
        SystemInputList.reserve(500);
        SystemInputList.push_back(this);
    }
    int My_GLFW_Key;
    bool Pressed = false;
    bool OnPressed = false;
    bool OnReleased = false;

    virtual void UpdateState(GLFWwindow* window) = 0;
    void UpdateEvent(int glfGet) {
        if (glfGet == GLFW_PRESS) {

            if (Pressed) {
                OnPressed = false;
            }
            else {
                OnPressed = true;
            }
            Pressed = true;
            OnReleased = false;
        }
        else {

            if (!Pressed) {
                OnReleased = false;
            }
            else {
                OnReleased = true;
            }
            Pressed = false;
            OnPressed = false;
        }
    }
};


struct BanK_SystemKeys {

    class KeyEvent : public InputEvent {

    public:
        KeyEvent(int GLFW_Key)
            :InputEvent(GLFW_Key) {}

        void UpdateState(GLFWwindow* window) {
            UpdateEvent(glfwGetKey(window, My_GLFW_Key));
        }
    };
    class MouseEvent : public InputEvent {

    public:
        MouseEvent(int GLFW_Key)
            :InputEvent(GLFW_Key) {}

        void UpdateState(GLFWwindow* window) {
            UpdateEvent(glfwGetMouseButton(window, My_GLFW_Key));
        }
    };

    MouseEvent LMB = MouseEvent(GLFW_MOUSE_BUTTON_LEFT);
    MouseEvent RMB = MouseEvent(GLFW_MOUSE_BUTTON_RIGHT);

    KeyEvent A = KeyEvent(GLFW_KEY_A);
    KeyEvent B = KeyEvent(GLFW_KEY_B);
    KeyEvent C = KeyEvent(GLFW_KEY_C);
    KeyEvent D = KeyEvent(GLFW_KEY_D);
    KeyEvent E = KeyEvent(GLFW_KEY_E);
    KeyEvent F = KeyEvent(GLFW_KEY_F);
    KeyEvent G = KeyEvent(GLFW_KEY_G);
    KeyEvent H = KeyEvent(GLFW_KEY_H);
    KeyEvent I = KeyEvent(GLFW_KEY_I);
    KeyEvent J = KeyEvent(GLFW_KEY_J);
    KeyEvent K = KeyEvent(GLFW_KEY_K);
    KeyEvent L = KeyEvent(GLFW_KEY_L);
    KeyEvent M = KeyEvent(GLFW_KEY_M);
    KeyEvent N = KeyEvent(GLFW_KEY_N);
    KeyEvent O = KeyEvent(GLFW_KEY_O);
    KeyEvent P = KeyEvent(GLFW_KEY_P);
    KeyEvent Q = KeyEvent(GLFW_KEY_Q);
    KeyEvent R = KeyEvent(GLFW_KEY_R);
    KeyEvent S = KeyEvent(GLFW_KEY_S);
    KeyEvent T = KeyEvent(GLFW_KEY_T);
    KeyEvent U = KeyEvent(GLFW_KEY_U);
    KeyEvent V = KeyEvent(GLFW_KEY_V);
    KeyEvent W = KeyEvent(GLFW_KEY_W);
    KeyEvent X = KeyEvent(GLFW_KEY_X);
    KeyEvent Y = KeyEvent(GLFW_KEY_Y);
    KeyEvent Z = KeyEvent(GLFW_KEY_Z);

    KeyEvent Escape = KeyEvent(GLFW_KEY_ESCAPE);
    KeyEvent Tab = KeyEvent(GLFW_KEY_TAB);
    KeyEvent Space = KeyEvent(GLFW_KEY_SPACE);
    KeyEvent Backspace = KeyEvent(GLFW_KEY_BACKSPACE);
    KeyEvent Enter = KeyEvent(GLFW_KEY_ENTER);
    KeyEvent LeftShift = KeyEvent(GLFW_KEY_LEFT_SHIFT);
    KeyEvent RightShift = KeyEvent(GLFW_KEY_RIGHT_SHIFT);
    KeyEvent LeftCtrl = KeyEvent(GLFW_KEY_LEFT_CONTROL);
    KeyEvent RightCtrl = KeyEvent(GLFW_KEY_RIGHT_CONTROL);
    KeyEvent LFT_Alt = KeyEvent(GLFW_KEY_LEFT_ALT);
    KeyEvent RHT_Alt = KeyEvent(GLFW_KEY_RIGHT_ALT);
    KeyEvent CapsLock = KeyEvent(GLFW_KEY_CAPS_LOCK);
    KeyEvent F1 = KeyEvent(GLFW_KEY_F1);
    KeyEvent F2 = KeyEvent(GLFW_KEY_F2);
    KeyEvent F3 = KeyEvent(GLFW_KEY_F3);
    KeyEvent F4 = KeyEvent(GLFW_KEY_F4);
    KeyEvent F5 = KeyEvent(GLFW_KEY_F5);
    KeyEvent F6 = KeyEvent(GLFW_KEY_F6);
    KeyEvent F7 = KeyEvent(GLFW_KEY_F7);
    KeyEvent F8 = KeyEvent(GLFW_KEY_F8);
    KeyEvent F9 = KeyEvent(GLFW_KEY_F9);
    KeyEvent F10 = KeyEvent(GLFW_KEY_F10);
    KeyEvent F11 = KeyEvent(GLFW_KEY_F11);
    KeyEvent F12 = KeyEvent(GLFW_KEY_F12);
    KeyEvent LFT_BRACKET = KeyEvent(GLFW_KEY_LEFT_BRACKET);
    KeyEvent RHT_BRACKET = KeyEvent(GLFW_KEY_RIGHT_BRACKET);
    KeyEvent BACKSLASH = KeyEvent(GLFW_KEY_BACKSLASH);
    KeyEvent SLASH = KeyEvent(GLFW_KEY_SLASH);
    KeyEvent Equal = KeyEvent(GLFW_KEY_EQUAL);
    KeyEvent Minus = KeyEvent(GLFW_KEY_MINUS);

}SystemKeys;


void B_UpdateInputs(GLFWwindow* window)
{
    // read input
    glfwPollEvents();

    for (InputEvent* pInst : SystemInputList) {
        pInst->UpdateState(window);
    }

}