#include "_Def2.h"




// ---------------------------------------
//  CUDA code 
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



__device__ vec3 FinalColor(const ray& r, const spheres& sphere_world, const triangles& tri_world, curandState* local_rand_state) {
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
            case Mat_Solid:
                if (scatter_lambert(cur_ray, rec_sphere, attenuation, scattered, local_rand_state)) {
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                }
                else {
                    return vec3(0.0, 0.0, 0.0);
                }
                break;
            case Mat_Metal:
                if (scatter_metal(cur_ray, rec_sphere, attenuation, scattered, local_rand_state)) {
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                }
                else {
                    return vec3(0.0, 0.0, 0.0);
                }
                break;
            case Mat_Mirror:
                if (scatter_dielectric(cur_ray, rec_sphere, attenuation, scattered, local_rand_state)) {
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                }
                else {
                    return vec3(0.0, 0.0, 0.0);
                }
                break;
            case Mat_Checkered: // CHECKERED
                if (scatter_checkered(cur_ray, rec_sphere, sphere_world.center[sphere_world.list_size - 1], attenuation, scattered, local_rand_state)) {
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                }
                else {
                    return vec3(0.0, 0.0, 0.0);
                }
                break;
            case Mat_Emissive: // Light source
                if (scatter_dielectric(cur_ray, rec_sphere, attenuation, scattered, local_rand_state)) {
                    //cur_ray = scattered;

                    cur_attenuation *= attenuation;
                    return cur_attenuation * attenuation * 800;
                } 
                else {
                    return vec3(0.0, 0.0, 0.0);
                }
                break;

            case Mat_Papermint: // Light source
                if (scatter_papermint(cur_ray, rec_sphere, attenuation, scattered, local_rand_state)) {
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                }  
                else {
                    return vec3(0.0, 0.0, 0.0);
                }
                break;

             case Mat_Kaleidoscope: // Light source
                if (scatter_kaleidoscope(cur_ray, rec_sphere, attenuation, scattered, local_rand_state)) {
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                } 
                else {
                    return vec3(0.0, 0.0, 0.0);
                }
                break;

            case Mat_ScatterDot: // Light source
                if (scatter_scatterdot(cur_ray, rec_sphere, sphere_world.center[sphere_world.list_size - 5], attenuation, scattered, local_rand_state)) {
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
        else if (hit_triangle) {
            // Triangle was hit (and sphere was either not hit or farther away)
            ray scattered;
            vec3 attenuation;

            switch (rec_triangle.mat_type) {
            case Mat_Solid:
                if (scatter_lambert(cur_ray, rec_triangle, attenuation, scattered, local_rand_state)) {
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                }
                else {
                    return vec3(0.0, 0.0, 0.0);
                }
                break;
            case Mat_Metal:
                if (scatter_metal(cur_ray, rec_triangle, attenuation, scattered, local_rand_state)) {
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                }
                else {
                    return vec3(0.0, 0.0, 0.0);
                }
                break;
            case Mat_Mirror:
                if (scatter_dielectric(cur_ray, rec_triangle, attenuation, scattered, local_rand_state)) {
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                }
                else {
                    return vec3(0.0, 0.0, 0.0);
                }
                break;
            case Mat_Mirror_Dark:
                if (scatter_dielectric(cur_ray, rec_triangle, attenuation, scattered, local_rand_state)) {
                    cur_attenuation *= attenuation * 0.75f;
                    cur_ray = scattered;
                }
                else {
                    return vec3(0.0, 0.0, 0.0);
                }
                break;
            case Mat_Checkered: // CHECKERED
                if (scatter_checkered(cur_ray, rec_triangle, vec3(0,0,0), attenuation, scattered, local_rand_state)) {
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                }
                else {
                    return vec3(0.0, 0.0, 0.0);
                }
                break;
            case Mat_Emissive: // Light source
                if (scatter_dielectric(cur_ray, rec_triangle, attenuation, scattered, local_rand_state)) {
                    cur_attenuation *= attenuation;
                    return cur_attenuation * attenuation * 800;
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
            // No hits, return sky color
            vec3 unit_direction = unit_vector(cur_ray.direction());

            //float t = 0.5f * (unit_direction.y() + 1.0f);
            //vec3 c = (1.0f - t) * vec3(1, 1, 1)*0.0f + t * vec3(0.5, 0.7, 1.0)*0.2f;

            //float t = 0.5f * (unit_direction.y() + 1.0f);
            //vec3 c = (1.0f - t) * vec3(1, 1, 1) * -0.005f + t * vec3(0.6, 0.5, 1.0) * 0.032f;

            //float t = 0.5f * (unit_direction.y() + 1.0f); 
            //vec3 c = (1.0f - t) * vec3(1, 1, 1) * -0.005f + t * vec3(1, 0.2, 1.0)*0.32f;


            float t = 0.5f * (unit_direction.y() + 1.0f); 
            vec3 c = (1.0f - t) * vec3(1, 1, 1) * -0.005f + t * vec3(1, 1, 1)*1.5;

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
    __grid_constant__ const spheres sphere_world, __grid_constant__ const triangles tri_world, curandState* rand_state) {
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
        col += FinalColor(r, sphere_world, tri_world, &local_rand_state); // Updated to pass tri_world
    }





    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = 255.99 * sqrt(col[0]);
    col[1] = 255.99 * sqrt(col[1]);
    col[2] = 255.99 * sqrt(col[2]);
    fb[pixel_index] = rgbToInt(col[0], col[1], col[2]);
}
 