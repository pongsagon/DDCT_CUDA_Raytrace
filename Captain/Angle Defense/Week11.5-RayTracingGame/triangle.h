#ifndef TRIANGLEH
#define TRIANGLEH

#include "material.h"
#include "vec3.h"
#include "ray.h"
#include <random>

class triangles {
public:
    vec3 p1[200], p2[200], p3[200];
    vec3 original_p1[200], original_p2[200], original_p3[200]; 
    vec3 albedo[200];
    float fuzz[200];
    float ref_idx[200];
    int mat_type[200];
    float deltaTime[200]; 
    int list_size;
    int ramiel_mat_idx = 2;
    vec3 ramiel_albedo = vec3(0.0, 0.0, 0.8);
    vec3 initial_center = vec3(0.0, 0.0, 0.0); 
    vec3 ramiel_center; 
    float radius = 800.0; 
    float angle = 0.0;

    triangles() {
        int idx = 0;
        std::random_device rd;
        std::mt19937 eng(rd());
        std::uniform_real_distribution<> distrHeight(500, 1200);
        std::uniform_real_distribution<> distrAngle(0, 2 * M_PI);
        std::uniform_real_distribution<> distrRadius(1000, 1200);

        float plane_size = 10000.0; 

        // Triangle 1 of the plane
        p3[idx] = vec3(-plane_size, -400, -plane_size);
        p2[idx] = vec3(plane_size, -400, -plane_size);
        p1[idx] = vec3(-plane_size, -400, plane_size);
        albedo[idx] = vec3(1.0, 1.0, 1.0); 
        fuzz[idx] = 0;
        mat_type[idx] = 8; 
        idx++;

        // Triangle 2 of the plane
        p3[idx] = vec3(plane_size, -400, -plane_size);
        p2[idx] = vec3(plane_size, -400, plane_size);
        p1[idx] = vec3(-plane_size, -400, plane_size);
        albedo[idx] = vec3(1.0, 1.0, 1.0); 
        fuzz[idx] = 0;
        mat_type[idx] = 8; 
        idx++;

        for (int i = 0; i < 40; i++) {
            float pillarHeight = distrHeight(eng);
            float angle = distrAngle(eng);
            float distance = distrRadius(eng); 
            float x = initial_center.x() + distance * cos(angle);
            float z = initial_center.z() + distance * sin(angle);
            vec3 pillarBaseCenter(x, -400, z);

            float baseHalfSize = 50.0;
            vec3 apex = pillarBaseCenter + vec3(0, pillarHeight, 0);

            vec3 baseCorner1 = pillarBaseCenter + vec3(-baseHalfSize, 0, -baseHalfSize);
            vec3 baseCorner2 = pillarBaseCenter + vec3(baseHalfSize, 0, -baseHalfSize);
            vec3 baseCorner3 = pillarBaseCenter + vec3(0, 0, baseHalfSize * 1.732); 

            // Side 1
            p3[idx] = apex;
            p2[idx] = baseCorner1;
            p1[idx] = baseCorner2;
            albedo[idx] = vec3(0.5, 0.5, 0.8);
            fuzz[idx] = 0.05;
            mat_type[idx] = 2;
            idx++;

            // Side 2
            p3[idx] = apex;
            p2[idx] = baseCorner2;
            p1[idx] = baseCorner3;
            albedo[idx] = vec3(0.5, 0.5, 0.8);
            fuzz[idx] = 0.05;
            mat_type[idx] = 2;
            idx++;

            // Side 3
            p3[idx] = apex;
            p2[idx] = baseCorner3;
            p1[idx] = baseCorner1;
            albedo[idx] = vec3(0.5, 0.5, 0.8);
            fuzz[idx] = 0.05;
            mat_type[idx] = 2;
            idx++;
        }



        vec3 apexUp = vec3(0, 150, 0);   // Apex of the upward pyramid
        vec3 apexDown = vec3(0, -150, 0); // Apex of the downward pyramid
        float s = 250.0;
        float half_s = s / 2;

        // Upward Pyramid
        // Triangle 1 - First side
        p1[idx] = vec3(half_s, 0, -half_s);
        p2[idx] = vec3(-half_s, 0, -half_s);
        p3[idx] = apexUp;
        albedo[idx] = ramiel_albedo;
        fuzz[idx] = 0.0;
        mat_type[idx] = ramiel_mat_idx;
        idx++;

        // Triangle 2 - Second side
        p1[idx] = vec3(half_s, 0, half_s);
        p2[idx] = vec3(half_s, 0, -half_s);
        p3[idx] = apexUp;
        albedo[idx] = ramiel_albedo;
        fuzz[idx] = 0.0;
        mat_type[idx] = ramiel_mat_idx;
        idx++;

        // Triangle 3 - Third side
        p1[idx] = vec3(-half_s, 0, half_s);
        p2[idx] = vec3(half_s, 0, half_s);
        p3[idx] = apexUp;
        albedo[idx] = ramiel_albedo;
        fuzz[idx] = 0.0;
        mat_type[idx] = ramiel_mat_idx;
        idx++;

        // Triangle 4 - Fourth side
        p1[idx] = vec3(-half_s, 0, -half_s);
        p2[idx] = vec3(-half_s, 0, half_s);
        p3[idx] = apexUp;
        albedo[idx] = ramiel_albedo;
        fuzz[idx] = 0.0;
        mat_type[idx] = ramiel_mat_idx;
        idx++;

        // Inverted Pyramid
    // Triangle 1 - First side
        p1[idx] = vec3(half_s, 0, -half_s);
        p2[idx] = vec3(half_s, 0, half_s); 
        p3[idx] = apexDown;
        albedo[idx] = ramiel_albedo;
        fuzz[idx] = 0.0;
        mat_type[idx] = ramiel_mat_idx;
        idx++;

        // Triangle 2 - Second side
        p1[idx] = vec3(half_s, 0, half_s);
        p2[idx] = vec3(-half_s, 0, half_s);
        p3[idx] = apexDown;
        albedo[idx] = ramiel_albedo;
        fuzz[idx] = 0.0;
        mat_type[idx] = ramiel_mat_idx;
        idx++;

        // Triangle 3 - Third side
        p1[idx] = vec3(-half_s, 0, half_s);
        p2[idx] = vec3(-half_s, 0, -half_s); 
        p3[idx] = apexDown;
        albedo[idx] = ramiel_albedo;
        fuzz[idx] = 0.0;
        mat_type[idx] = ramiel_mat_idx;
        idx++;

        // Triangle 4 - Fourth side
        p1[idx] = vec3(-half_s, 0, -half_s);
        p2[idx] = vec3(half_s, 0, -half_s);
        p3[idx] = apexDown;
        albedo[idx] = ramiel_albedo;
        fuzz[idx] = 0.0;
        mat_type[idx] = ramiel_mat_idx;
        idx++;



        list_size = idx;

        calculate_original_positions();
    }

    void calculate_original_positions() {
        for (int i = 2; i < list_size; i++) {
            original_p1[i] = p1[i] - initial_center;
            original_p2[i] = p2[i] - initial_center;
            original_p3[i] = p3[i] - initial_center;
        }
    }




    void update_position(float deltaTime) {
        angle += 0.1 * deltaTime;
        float x = initial_center.getX() + radius * cos(angle);
        float z = initial_center.getZ() + radius * sin(angle);
        vec3 new_center = vec3(x, 0.0, z);
        ramiel_center = new_center;

        for (int i = 122; i < list_size; i++) {
            p1[i] = new_center + original_p1[i];
            p2[i] = new_center + original_p2[i];
            p3[i] = new_center + original_p3[i];
        }

        this->deltaTime[0] = deltaTime;
		this->deltaTime[1] = deltaTime;
    }

    vec3 getRamielCenter() {
        //printf("Ramiel Center: %f %f %f\n", ramiel_center.getX(), ramiel_center.getY(), ramiel_center.getZ());
		return ramiel_center;
	}


    __device__ bool hitall(const ray& r, float t_min, float t_max, hit_record& rec) const;
    __device__ bool hit(int idx, const ray& r, float tmin, float tmax, hit_record& rec) const;
};

__device__ bool triangles::hitall(const ray& r, float t_min, float t_max, hit_record& rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;
    for (int i = 0; i < list_size; i++) {
        if (hit(i, r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}

__device__ bool triangles::hit(int idx, const ray& r, float t_min, float t_max, hit_record& rec) const {
    const float EPSILON = 1e-8;
    vec3 edge1 = p2[idx] - p1[idx];
    vec3 edge2 = p3[idx] - p1[idx];
    vec3 h = cross(r.direction(), edge2);
    float a = dot(edge1, h);
    if (a > -EPSILON && a < EPSILON)
        return false;  

    float f = 1.0 / a;
    vec3 s = r.origin() - p1[idx];
    float u = f * dot(s, h);
    if (u < 0.0 || u > 1.0)
        return false;

    vec3 q = cross(s, edge1);
    float v = f * dot(r.direction(), q);
    if (v < 0.0 || u + v > 1.0)
        return false;

    // Calculate t to see where the intersection point is on the line.
    float t = f * dot(edge2, q);
    if (t > t_min && t < t_max) { 
        rec.t = t;
        rec.p = r.point_at_parameter(t);

        rec.normal = cross(edge1, edge2).normalize();
        
        rec.albedo = albedo[idx];
        rec.fuzz = fuzz[idx];
        rec.ref_idx = ref_idx[idx];
        rec.mat_type = mat_type[idx];
        rec.deltaTime = deltaTime[idx];
        return true;
    }
    return false;
}

#endif // TRIANGLEH
