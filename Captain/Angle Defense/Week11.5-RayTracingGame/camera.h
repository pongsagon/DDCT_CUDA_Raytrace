#ifndef CAMERAH
#define CAMERAH

#include "vec3.h"
#include "ray.h"
#include <curand_kernel.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ vec3 random_in_unit_disk(curandState* local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - vec3(1, 1, 0);
    } while (dot(p, p) >= 1.0f);
    return p;
}

class camera {
public:
    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lens_radius;

    camera() : lens_radius(0.0f), origin(vec3(0, 0, 0)), lower_left_corner(vec3(-2.0, -1.0, -1.0)), horizontal(vec3(4.0, 0.0, 0.0)), vertical(vec3(0.0, 2.0, 0.0)), u(vec3(1, 0, 0)), v(vec3(0, 1, 0)), w(vec3(0, 0, 1)) {}

    camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist) {
        lens_radius = aperture / 2;
        float theta = vfov * M_PI / 180;
        float half_height = tan(theta / 2);
        float half_width = aspect * half_height;
        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
        horizontal = 2 * half_width * focus_dist * u;
        vertical = 2 * half_height * focus_dist * v;
    }


    void setCamera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist) {
        float theta = vfov * M_PI / 180.0f;
        float half_height = tan(theta / 2);
        float half_width = aspect * half_height;

        origin = lookfrom;
        lens_radius = aperture / 2;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
        horizontal = 2 * half_width * focus_dist * u;
        vertical = 2 * half_height * focus_dist * v;
    }

    void followTarget(const vec3& target_position, const vec3& offset, vec3 vup, float vfov, float aspect, float aperture, float focus_dist) {
        vec3 new_lookfrom = target_position + offset;
        vec3 new_lookat = target_position;

        vup = vec3(0, 1, 0);

        w = unit_vector(new_lookfrom - new_lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        float theta = vfov * M_PI / 180;
        float half_height = tan(theta / 2);
        float half_width = aspect * half_height;

        origin = new_lookfrom;
        lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
        horizontal = 2 * half_width * focus_dist * u;
        vertical = 2 * half_height * focus_dist * v;

        // Debugging output
        /*std::cout << "Camera vectors:" << std::endl;
        std::cout << "Origin: " << origin << std::endl;
        std::cout << "Lower left corner: " << lower_left_corner << std::endl;
        std::cout << "Horizontal: " << horizontal << std::endl;
        std::cout << "Vertical: " << vertical << std::endl;*/
    }

    __device__ ray get_ray(float s, float t, curandState* local_rand_state) const {
        vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
        vec3 offset = u * rd.x() + v * rd.y();
        return ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
    }


};

#endif
