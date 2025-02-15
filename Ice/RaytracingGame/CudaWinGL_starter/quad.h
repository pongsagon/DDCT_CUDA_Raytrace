#pragma once

#include "vec3.h"
#include "material.h"
#include "hittable.h"

#include "interval.h"

class quad : public hittable
{
public:

    vec3 normal;
    double D;
    vec3 w;
    vec3 u, v;
    point3 Q;
    material* mat_ptr = nullptr;
    //aabb bbox;

    __host__ __device__  quad() { type = hittable_type::Quad; }

    __host__ __device__  quad(const point3& Q, const vec3& u, const vec3& v, material* m)
        : Q(Q), u(u), v(v), mat_ptr(m)
    {
        type = hittable_type::Quad;

        auto n = cross(u, v);
        normal = unit_vector(n);
        D = dot(normal, Q);
        w = n / dot(n, n);
    }

    /*
    __device__ virtual void set_bounding_box() {
        // Compute the bounding box of all four vertices.
        
        auto bbox_diagonal1 = aabb(Q, Q + u + v);
        auto bbox_diagonal2 = aabb(Q + u, Q + v);
        bbox = aabb(bbox_diagonal1, bbox_diagonal2);
        
    }
    */

    //aabb bounding_box() const override { return bbox; }

    //__device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const 
    __device__ bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const
    {
        auto denom = dot(normal, r.direction());

        // No hit if the ray is parallel to the plane.
        if (std::fabs(denom) < 1e-8)
            return false;

        // Return false if the hit point parameter t is outside the ray interval.
        auto t = (D - dot(normal, r.origin())) / denom;
        if (!(t < tmax && t > tmin))
        //if (!ray_t.contains(t))
            return false;

        auto intersection = r.point_at_parameter(t);
        vec3 planar_hitpt_vector = intersection - Q;
        auto alpha = dot(w, cross(planar_hitpt_vector, v));
        auto beta = dot(w, cross(u, planar_hitpt_vector));

        if (!is_interior(alpha, beta, rec))
            return false;

        // Ray hits the 2D shape; set the rest of the hit record and return true.

        rec.t = t;
        rec.p = intersection;
        //rec = vec3(0.8, 0.2, 0.1);
        //rec.mat_type = 1;
        rec.normal = set_face_normal(r, normal);
        rec.mat_ptr = mat_ptr;

        return true;
    }

    __device__ vec3 set_face_normal(const ray& r, const vec3& outward_normal) const {
        // Sets the hit record normal vector.
        // NOTE: the parameter `outward_normal` is assumed to have unit length.

        bool front_face = dot(r.direction(), outward_normal) < 0;
        return front_face ? outward_normal : -outward_normal;
    }

    __device__ bool is_interior(double a, double b, hit_record& rec) const
    {
        float tmin = 0.0f;
        float tmax = 1.0f;

        // Given the hit point in plane coordinates, return false if it is outside the
        // primitive, otherwise set the hit record UV coordinates and return true.

        if (!(a < tmax && a > tmin) || !(b < tmax && b > tmin))
            return false;

        rec.u = a;
        rec.v = b;
        return true;
    }
};


