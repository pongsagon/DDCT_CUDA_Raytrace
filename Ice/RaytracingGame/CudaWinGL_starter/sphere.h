#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"

__constant__ const double pi = 3.1415926535897932385;

__device__ void get_sphere_uv(const point3& p, float& u, float& v, float radius) {
    
    auto theta = std::acos(-p.y());
    auto phi = std::atan2(-p.z(), p.x()) + pi;

    u = phi / (2 * pi);
    v = theta / pi;
    
    
    /*
    
    auto theta = std::atan2(p.x(), p.z());
    auto magnitude = p.length();
    auto phi = std::acos(p.y() / radius);
    auto raw_u = theta / (2 * pi);

    u = 1 - (raw_u + 0.5f);
    v = 1 - phi / pi;
    */

    
}

class sphere : public hittable {
public:
    __device__ sphere()
    {
        type = hittable_type::Sphere;
        velocity = vec3(0, 0, 0);
    }
    __device__ sphere(vec3 cen, float r, material* m) : center(cen), radius(r), mat_ptr(m)
    {
        type = hittable_type::Sphere;
        velocity = vec3(0, 0, 0);
    };
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;

    __device__ void change_position(vec3 new_position)
    {
        center = new_position;
    }

    __device__ void change_velocity(vec3 new_velocity)
    {
        velocity = new_velocity;
    }

    bool active = true;

    vec3 center;
    vec3 velocity;
    float radius;
    material* mat_ptr = nullptr;
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    if (!active)
        return false;

    vec3 oc = center - r.origin();
    auto a = r.direction().squared_length();
    auto h = dot(r.direction(), oc);
    auto c = oc.squared_length() - radius * radius;

    auto discriminant = h * h - a * c;
    if (discriminant < 0)
        return false;

    auto sqrtd = std::sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    auto root = (h - sqrtd) / a;
    //if (!ray_t.surrounds(root)) {
    if (!(t_min < root && root < t_max)){
        root = (h + sqrtd) / a;
        //if (!ray_t.surrounds(root))
        if (!(t_min < root && root < t_max))
            return false;
    }

    rec.t = root;
    //rec.p = r.at(rec.t);
    rec.p = r.point_at_parameter(rec.t);
    vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
    //rec.normal = (rec.p - center) / radius;
    get_sphere_uv(outward_normal, rec.u, rec.v, radius);
    rec.mat_ptr = mat_ptr;

    return true;
}
/*
__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - a * c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            rec.u = 1;
            rec.v = 1;
            //get_sphere_uv(rec.normal, rec.u, rec.v);

            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            rec.u = 1;
            rec.v = 1;
            //get_sphere_uv(rec.normal, rec.u, rec.v);


            return true;
        }
    }
    return false;
}
*/

#endif
