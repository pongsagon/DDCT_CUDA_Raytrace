#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "hittable.h"

class triangle : public hittable {
public:
    __device__ triangle() { type = hittable_type::Triangle; }
    __device__ triangle(vec3 v0, vec3 v1, vec3 v2, material* m)
        : vertices { v0, v1, v2 }
        , mat_ptr(m)
        , normal(unit_vector(cross(v1 - v0, v2 - v0)))
    {
        type = hittable_type::Triangle;
    };

    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
    material* mat_ptr = nullptr;
    vec3 vertices[3];
    vec3 normal;
};

__device__ bool triangle::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 v0v1 = vertices[1] - vertices[0];
    vec3 v0v2 = vertices[2] - vertices[0];
    vec3 pvec = cross(r.direction(), v0v2);
    float det = dot(v0v1, pvec);

    // If the determinant is near zero, the ray lies in the plane of the triangle
    if (fabs(det) < 1e-8) return false;
    float invDet = 1.0 / det;

    vec3 tvec = r.origin() - vertices[0];
    float u = dot(tvec, pvec) * invDet;
    if (u < 0 || u > 1) return false;

    vec3 qvec = cross(tvec, v0v1);
    float v = dot(r.direction(), qvec) * invDet;
    if (v < 0 || u + v > 1) return false;

    float t = dot(v0v2, qvec) * invDet;
    if (t < t_min || t > t_max) return false;

    rec.t = t;
    rec.p = r.point_at_parameter(t);
    rec.normal = normal;
    rec.mat_ptr = mat_ptr;

    return true;
}


#endif
