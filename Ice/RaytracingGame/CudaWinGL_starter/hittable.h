#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"

class material;

enum hittable_type {
    None,
    Sphere,
    Triangle,
    Quad,
};

struct hit_record
{
    float t;
    float u;
    float v;
    vec3 p;
    vec3 normal;
    material* mat_ptr;
    bool front_face;

    __device__ void set_face_normal(const ray& r, const vec3& outward_normal) {
        // Sets the hit record normal vector.
        // NOTE: the parameter `outward_normal` is assumed to have unit length.

        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hittable {
public:
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;

    int type = hittable_type::None;
};

#endif
