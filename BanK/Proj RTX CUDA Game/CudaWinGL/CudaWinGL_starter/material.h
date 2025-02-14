#ifndef MATERIALH
#define MATERIALH



enum MaterialEnum
{
    Nan=0,

    Mat_Solid,
    Mat_Metal,
    Mat_Mirror,
    Mat_Mirror_Dark,
    Mat_Glass,
    Mat_Checkered,
    Mat_Emissive,
    Mat_Papermint,
    Mat_Kaleidoscope,
    Mat_ScatterDot,

    NaN
};


struct hit_record
{
    float t;
    vec3 p;
    vec3 normal;
    vec3 albedo;
    float fuzz;
    float ref_idx;
    int mat_type;
};

#include "ray.h"



__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f-ref_idx) / (1.0f+ref_idx);
    r0 = r0*r0;
    return r0 + (1.0f-r0)*pow((1.0f - cosine),5.0f);
}

__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt*ni_over_nt*(1-dt*dt);
    if (discriminant > 0) {
        refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
        return true;
    }
    else
        return false;
}

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
    vec3 p;
    do {
        p = 2.0f*RANDVEC3 - vec3(1,1,1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

__device__ vec3 reflect(const vec3& v, const vec3& n) {
     return v - 2.0f*dot(v,n)*n;
}


__device__ bool scatter_lambert(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) {
    vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
    scattered = ray(rec.p, target - rec.p);
    attenuation = rec.albedo;
    return true;
}

__device__ bool scatter_metal(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) {
    vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    scattered = ray(rec.p, reflected + rec.fuzz * random_in_unit_sphere(local_rand_state));
    attenuation = rec.albedo;
    return (dot(scattered.direction(), rec.normal) > 0.0f);
}

__device__ bool scatter_dielectric(const ray& r_in,
                    const hit_record& rec,
                    vec3& attenuation,
                    ray& scattered,
                    curandState* local_rand_state)  {
    vec3 outward_normal;
    vec3 reflected = reflect(r_in.direction(), rec.normal);
    float ni_over_nt;
    attenuation = vec3(1.0, 1.0, 1.0);
    vec3 refracted;
    float reflect_prob;
    float cosine;
    if (dot(r_in.direction(), rec.normal) > 0.0f) {
        outward_normal = -rec.normal;
        ni_over_nt = rec.ref_idx;
        cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
        cosine = sqrt(1.0f - rec.ref_idx * rec.ref_idx * (1 - cosine * cosine));
    }
    else {
        outward_normal = rec.normal;
        ni_over_nt = 1.0f / rec.ref_idx;
        cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
    }
    if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
        reflect_prob = schlick(cosine, rec.ref_idx);
    else
        reflect_prob = 1.0f;
    if (curand_uniform(local_rand_state) < reflect_prob)
        scattered = ray(rec.p, reflected);
    else
        scattered = ray(rec.p, refracted);
    return true;
}



__device__ bool scatter_papermint(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) {
    //vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);

    // Define stripe frequency and angle in degrees
    float stripe_frequency = 16.0f/10000;  // Frequency of stripes
    float angle_degrees = 45.0f;     // Desired angle in degrees
    float angle_radians = angle_degrees * (22/7.0f) / 180.0f;  // Convert angle to radians

    // Calculate rotated coordinates based on the angle
    float rotated_position = rec.p.x() * cos(angle_radians) + rec.p.y() * sin(angle_radians);

    // Determine if it's a red or white stripe
    bool is_red_stripe = (sin(stripe_frequency * rotated_position) > 0.0f);

    // Set color based on stripe
    if (is_red_stripe) {
        attenuation = vec3(1.0f, 0.0f, 0.0f);  // Red color for the red stripe
    }
    else {
        attenuation = vec3(1.0f, 0.8f, 0.4)*5;  // White color for the white stripe
    }

    // Lambertian scatter
    vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
    scattered = ray(rec.p, target - rec.p);
    //scattered = ray(rec.p + reflected, target - rec.p);

    return true;
}

__device__ bool scatter_checkered(const ray& r_in, const hit_record& rec, vec3 object_position, vec3& attenuation, ray& scattered, curandState* local_rand_state) {
    // Define grid frequency for larger squares
    float grid_frequency = 2.0f;  // Adjust frequency as needed

    // Calculate the local position relative to the object
    vec3 local_position = rec.p - object_position;

    // Convert local position to spherical coordinates
    float theta = acos(local_position.z() / local_position.length());  // Angle from the Z-axis
    float phi = atan2(local_position.y(), local_position.x());         // Angle around the Z-axis

    // Scale theta and phi by grid frequency
    float u = theta * grid_frequency;
    float v = phi * grid_frequency;

    // Calculate checkered pattern using spherical coordinates
    bool is_black_square = (int(floor(u) + floor(v)) % 2) == 0;

    // Set color based on checkered pattern
    if (is_black_square) {
        attenuation = vec3(0.0f, 0.0f, 0.0f);  // Black color for black squares
    }
    else {
        attenuation = vec3(1.0f, 1.0f, 1.0f);  // White color for white squares
    }

    // Lambertian scatter
    vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
    scattered = ray(rec.p, target - rec.p);

    return true;
}



__device__ bool scatter_kaleidoscope(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) {
    // Define 4 different colors for the triangles
    vec3 colors[4] = {
        vec3(1.0f, 0.0f, 0.0f),  // Red
        vec3(0.0f, 1.0f, 0.0f),  // Green
        vec3(0.0f, 0.0f, 1.0f),  // Blue
        vec3(1.0f, 1.0f, 0.0f)   // Yellow
    };

    // Define the scale factor (set to 1000 as per your request)
    float scale = 1000.0f;

    // Determine which triangle in the kaleidoscope pattern this is based on the position
    int pattern_index = int(floor(rec.p.x() / scale)) % 4; // Assign one of the 4 triangles

    // Set the color based on the pattern
    attenuation = colors[pattern_index];

    // Apply transformations (rotation/reflection) to create the kaleidoscope pattern
    vec3 offset(0.0f, 0.0f, 0.0f);  // Initialize offset vector

    // Apply rotation and reflection to achieve kaleidoscope pattern
    if (pattern_index == 0) {
        offset = vec3(scale, scale, 0.0f);  // First triangle at +1000 scale
    }
    else if (pattern_index == 1) {
        offset = vec3(-scale, scale, 0.0f);  // Second triangle mirrored on the x-axis
    }
    else if (pattern_index == 2) {
        offset = vec3(scale, -scale, 0.0f);  // Third triangle mirrored on the y-axis
    }
    else if (pattern_index == 3) {
        offset = vec3(-scale, -scale, 0.0f);  // Fourth triangle mirrored on both axes
    }

    // Manually perform the vector addition (since `vec3` does not support the `+` operator)
    vec3 target;
    target[0] = rec.p.x() + offset.x() + rec.normal.x();
    target[1] = rec.p.y() + offset.y() + rec.normal.y();
    target[2] = rec.p.z() + offset.z() + rec.normal.z();

    // Now, define the vertices of the triangles in 3D space.
    // We will manually define the triangle's positions to ensure it remains triangular, not rectangular.

    vec3 triangle_vertex_1 = target; // Point 1 of the triangle
    vec3 triangle_vertex_2 = target + vec3(scale * 0.5f, 0.0f, 0.0f);  // Point 2 of the triangle
    vec3 triangle_vertex_3 = target + vec3(0.0f, scale * 0.5f, 0.0f);  // Point 3 of the triangle

    // Scatter the ray within the triangle's boundaries
    vec3 scatter_point = triangle_vertex_1 + random_in_unit_sphere(local_rand_state) * (triangle_vertex_2 - triangle_vertex_1).length();
    scattered = ray(rec.p, scatter_point - rec.p);  // Scatter in a direction towards the triangle vertices

    return true;
}






__device__ void cartesian_to_spherical(const vec3& p, float& theta, float& phi) {
    // Convert from Cartesian to spherical coordinates
    theta = atan2(p.y(), p.x());  // azimuthal angle (longitude)
    phi = acos(p.z() / p.length());  // polar angle (latitude)
}


__device__ bool Draw_Dot(float x_within_cell, float y_within_cell, float dot_radius) {
    // Calculate distance from the center of the cell
    float distance_to_dot_center = sqrt(pow(x_within_cell - 0.5f, 2) + pow(y_within_cell - 0.5f, 2));
    // Return true if within dot radius, ensure dot fully fits within cell
    return distance_to_dot_center < dot_radius && x_within_cell >= 0 && x_within_cell <= 1 && y_within_cell >= 0 && y_within_cell <= 1;
}

__device__ bool Draw_Cross(float x_within_cell, float y_within_cell, float cross_width, float angle_radians) {
    // Apply rotation to the coordinates within the grid cell
    float x_rot = x_within_cell * cos(angle_radians) - y_within_cell * sin(angle_radians);
    float y_rot = x_within_cell * sin(angle_radians) + y_within_cell * cos(angle_radians);

    // Ensure cross arms fully fit within the cell by checking bounds
    bool in_vertical_cross = fabs(x_rot - 0.5f) < cross_width;
    bool in_horizontal_cross = fabs(y_rot - 0.5f) < cross_width;

    // Return true if within either arm of the rotated cross, ensuring full visibility within bounds
    return in_vertical_cross || in_horizontal_cross;
}
__device__ bool scatter_scatterdot(const ray& r_in, const hit_record& rec, vec3 object_position, vec3& attenuation, ray& scattered, curandState* local_rand_state) {

    vec3 Black = vec3(0.1f, 0.1f, 0.1f);
    vec3 White = vec3(1, 0, 0);
    // Define grid frequency and pattern parameters
    float pattern_frequency = 16*5; // Adjust frequency as needed
    float dot_radius = 0.5f;        // Radius of each dot (adjusted for visibility)
    float cross_width = 0.1f;       // Width of the cross arms (adjusted for visibility)
    float angle_degrees = 45;        // Angle to rotate the cross (in degrees)
    float angle_radians = angle_degrees * (22 / 7.0f) / 180.0f;  // Convert angle to radians

    // Calculate the local position relative to the sphere
    vec3 local_position = rec.p - object_position;

    // Convert to spherical coordinates (theta, phi)
    float theta, phi;
    cartesian_to_spherical(local_position, theta, phi);

    // Normalize theta and phi to [0, 1] for texture mapping
    float u = (theta + 3.14159f) / (2.0f * 3.14159f); // Map theta to [0, 1]
    float v = phi / 3.14159f; // Map phi to [0, 1]

    // Determine the grid indices based on the spherical coordinates
    float x_grid = floor(u * pattern_frequency);
    float y_grid = floor(v * pattern_frequency);

    // Check whether the cell contains a dot or cross
    bool is_dot_cell = ((int(x_grid) + int(y_grid)) % 2 == 0); // Alternating pattern

    // Calculate local position within the grid cell
    float x_within_cell = fmod(u * pattern_frequency, 1.0f);
    float y_within_cell = fmod(v * pattern_frequency, 1.0f);

    // Determine color based on the pattern
    if (is_dot_cell && Draw_Dot(x_within_cell, y_within_cell, dot_radius)) {
        attenuation = White; 
    }
    else if (!is_dot_cell && Draw_Cross(x_within_cell, y_within_cell, cross_width, angle_radians)) {
        attenuation = White;  
    }
    else {
        attenuation = Black; 
    }

    // Lambertian scatter
    vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
    scattered = ray(rec.p, target - rec.p);

    return true;
}




#endif
