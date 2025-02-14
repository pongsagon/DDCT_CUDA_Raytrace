#include "Engine/_Final.h"

// create a world of spheres and triangles
spheres a_sphere_world;
triangles a_tri_world;
glm::vec3 Blackhole_Vel = glm::vec3(0, 0, 0);


int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
    LPSTR lpCmdLine, int nCmdShow)
{
     
#pragma region "gl setup"
    // ------------------------------
    // glfw: initialize and configure
    // ------------------------------

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
    int num_texels = WIDTH * HEIGHT;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;
    checkCudaErrors(cudaMalloc((void**)&cuda_dest_resource, size_tex_data));

    glGenTextures(1, &tex_cudaResult);
    glBindTexture(GL_TEXTURE_2D, tex_cudaResult);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_tex_result_resource, tex_cudaResult, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));

    prevTime = glfwGetTime();
#pragma endregion "gl setup"

    // CUDA: RT
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));

    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render_init << <blocks, threads >> > (nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    while (!glfwWindowShouldClose(window))
    {
        processInput(window);
        glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        // Updated render call to include a_tri_world
        render << <blocks, threads >> > (cuda_dest_resource, nx, ny, ns, d_camera, a_sphere_world, a_tri_world, d_rand_state);
        checkCudaErrors(cudaGetLastError());

        cudaArray* texture_ptr;
        checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_result_resource, 0));
        checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_result_resource, 0, 0));
        checkCudaErrors(cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dest_resource, size_tex_data, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_result_resource, 0));

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        ss << elapsedTime << "ms\n";
        OutputDebugStringA(ss.str().c_str());
        ss.str("");
        cudaEventDestroy(start);
        cudaEventDestroy(stop);


        glUniform1i(glGetUniformLocation(ourShader.ID, "texture1"), 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex_cudaResult);
        ourShader.use();
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
        B_UpdateInputs(window);
        Time.Calculate();











        a_sphere_world.Update();

        ////////           BLACKHOLE
        ////////
        /////////////////////////////////////////////////////////

        vec3& Blackhole = a_sphere_world.center[a_sphere_world.B_Blackhole];
        vec3& BlackholeHorizon = a_sphere_world.center[a_sphere_world.B_BlackholeHorizon];
        vec3& CenterStarGel = a_sphere_world.center[a_sphere_world.B_CenterStarGel];
        vec3& CenterStar = a_sphere_world.center[a_sphere_world.B_CenterStar];

        vec3& BlackholeColor = a_sphere_world.albedo[a_sphere_world.B_Blackhole];


            glm::vec3 Vector_Front;
            Vector_Front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch)); 
            Vector_Front.y = sin(glm::radians(Pitch));
            Vector_Front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
            Vector_Front = glm::normalize(Vector_Front);
            glm::vec3 Vector_Right;
            Vector_Right = glm::normalize(glm::cross(Vector_Front, glm::vec3(0, 1, 0)));
            glm::vec3 Vector_Up;
            Vector_Up = glm::normalize(glm::cross(Vector_Right, Vector_Front));



            //Camera pos is set behind Blackhole
            float LerpSpeed = Time.Deltatime * 5;
            float CamDistance = 7.2f;
            float TargetposX = Blackhole.e[0] - Vector_Front.x * CamDistance;
            float TargetposY = Blackhole.e[1] - Vector_Front.y * CamDistance;
            float TargetposZ = Blackhole.e[2] - Vector_Front.z * CamDistance;
            lookfrom.e[0] = B_lerp(lookfrom.e[0], TargetposX, LerpSpeed);
            lookfrom.e[1] = B_lerp(lookfrom.e[1], TargetposY, LerpSpeed);
            lookfrom.e[2] = B_lerp(lookfrom.e[2], TargetposZ, LerpSpeed);

                    //Blackhole Movements
                    float Acel_Strength = 8;
                    float Velo_Strength = 1.5f;
                    if (SystemKeys.W.Pressed) {
                        Blackhole_Vel.x += Vector_Front.x * Time.Deltatime * Acel_Strength;
                        Blackhole_Vel.y += Vector_Front.y * Time.Deltatime * Acel_Strength;
                        Blackhole_Vel.z += Vector_Front.z * Time.Deltatime * Acel_Strength;
                    }
                    else if (SystemKeys.S.Pressed){
                        Blackhole_Vel.x -= Vector_Front.x * Time.Deltatime * Acel_Strength;
                        Blackhole_Vel.y -= Vector_Front.y * Time.Deltatime * Acel_Strength;
                        Blackhole_Vel.z -= Vector_Front.z * Time.Deltatime * Acel_Strength;
                    }

                    if (SystemKeys.A.Pressed) {
                        Blackhole_Vel.x -= Vector_Right.x * Time.Deltatime * Acel_Strength;
                        Blackhole_Vel.y -= Vector_Right.y * Time.Deltatime * Acel_Strength;
                        Blackhole_Vel.z -= Vector_Right.z * Time.Deltatime * Acel_Strength;
                    }
                    else if (SystemKeys.D.Pressed){
                        Blackhole_Vel.x += Vector_Right.x * Time.Deltatime * Acel_Strength;
                        Blackhole_Vel.y += Vector_Right.y * Time.Deltatime * Acel_Strength;
                        Blackhole_Vel.z += Vector_Right.z * Time.Deltatime * Acel_Strength;
                    }

                    if (SystemKeys.E.Pressed) {
                        Blackhole_Vel.y += Time.Deltatime * Acel_Strength * 1.5f;
                    }
                    else if (SystemKeys.Q.Pressed) {
                        Blackhole_Vel.y -= Time.Deltatime * Acel_Strength * 1.5f;
                    }

                    float MaxSpeed = 5;
                    Blackhole_Vel = B_lerpVec3(Blackhole_Vel, glm::vec3(0), Time.Deltatime*0.5f);
                    Blackhole_Vel.x = B_clamp(Blackhole_Vel.x, -MaxSpeed, MaxSpeed);
                    Blackhole_Vel.y = B_clamp(Blackhole_Vel.y, -MaxSpeed, MaxSpeed);
                    Blackhole_Vel.z = B_clamp(Blackhole_Vel.z, -MaxSpeed, MaxSpeed);
                    Blackhole.e[0] += Blackhole_Vel.x * Time.Deltatime * Velo_Strength;
                    Blackhole.e[1] += Blackhole_Vel.y * Time.Deltatime * Velo_Strength;
                    Blackhole.e[2] += Blackhole_Vel.z * Time.Deltatime * Velo_Strength;

                    //Borders
                    Blackhole.e[1] = B_clamp(Blackhole.e[1], -0.16, 100);
                    lookfrom[1] = B_clamp(lookfrom[1], 0.01, 100);

                    //Colors
                    LerpSpeed = Time.Deltatime * 2;
                    BlackholeColor.e[0] = B_lerp(BlackholeColor.e[0], 0, LerpSpeed);
                    BlackholeColor.e[1] = B_lerp(BlackholeColor.e[1], 0, LerpSpeed);
                    BlackholeColor.e[2] = B_lerp(BlackholeColor.e[2], 0, LerpSpeed);

                    //Copy Position
                    BlackholeHorizon = Blackhole; 
                    //CenterStarGel = Blackhole;




                    //Collision
                    glm::vec3 CenterStar_Temp;
                    CenterStar_Temp.x = CenterStar.x();
                    CenterStar_Temp.y = CenterStar.y();
                    CenterStar_Temp.z = CenterStar.z();
                                        
                    glm::vec3 Blackhole_Temp;
                    Blackhole_Temp.x = Blackhole.x();
                    Blackhole_Temp.y = Blackhole.y();
                    Blackhole_Temp.z = Blackhole.z();

                    float DistanceA = glm::distance(CenterStar_Temp, Blackhole_Temp);

                    if (DistanceA < 4.5) {
                        glm::vec3 direction = glm::normalize(CenterStar_Temp - Blackhole_Temp);
                        Blackhole_Temp = CenterStar_Temp + (direction * -4.5f);

                        Blackhole.e[0] = Blackhole_Temp.x;
                        Blackhole.e[1] = Blackhole_Temp.y;
                        Blackhole.e[2] = Blackhole_Temp.z;
                    }




        d_camera.updateCam(0, Yaw, Pitch, lookfrom, lookat, vec3(0, 1, 0), 30.0, float(nx) / float(ny), aperture, dist_to_focus);








    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    cudaFree(cuda_dest_resource);
    cudaFree(d_rand_state);
    cudaDeviceReset();
    glfwTerminate();
    return 0;
}
