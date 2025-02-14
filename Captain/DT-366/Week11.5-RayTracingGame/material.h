#ifndef MATERIALH
#define MATERIALH

#define M_PI 3.14159265358979323846

struct hit_record
{
    float t;
    vec3 p;
    vec3 normal;
    vec3 albedo;
    float fuzz;
    float ref_idx;
    int mat_type;
    float deltaTime;
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

__device__ vec3 random_in_disk(curandState* local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - vec3(1, 1, 0);
    } while (dot(p, p) >= 1.0f);
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

__device__ bool scatter_lightmetal(const ray& r_in,
    const hit_record& rec,
    vec3& attenuation,
    ray& scattered,
    curandState* local_rand_state) {
    vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    scattered = ray(rec.p, reflected + rec.fuzz * random_in_unit_sphere(local_rand_state));

    // Use albedo for color like in metal
    attenuation = rec.albedo;

    // Light emission properties
    float luminance = 10.0f;  // Adjust this for more or less emission
    vec3 emitted = vec3(luminance, luminance, luminance) * rec.albedo;  // Emission based on albedo

    // Combine color with emission
    attenuation = attenuation * 0.5 + emitted * 0.5;

    // Only reflect if the angle is correct
    return (dot(scattered.direction(), rec.normal) > 0.0f);
}



__device__ bool scatter_disco(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) {
   // Convert hit point to spherical coordinates (for 2D-like mapping)
    float theta = acosf(rec.normal.y());                // Angle from the "up" axis
    float phi = atan2f(rec.normal.z(), rec.normal.x()); // Angle around the equator

    // Scale the pattern to adjust checkered size
    float scale = 10.0f;
    float u = theta / M_PI * scale;                     // Normalized latitude
    float v = (phi + M_PI) / (2.0f * M_PI) * scale;     // Normalized longitude

    // Create a checkered pattern using u and v
    float pattern = fmodf(floorf(u) + floorf(v), 2.0f);

    // Reflect like a metal based on the pattern
    vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    if (pattern < 1.0f) {
        // Metallic reflection with possible color
        scattered = ray(rec.p, reflected + rec.fuzz * random_in_unit_sphere(local_rand_state));
        attenuation = vec3(0.8f, 0.8f, 0.8f);  // Silver color
    }
    else {
        // Metallic reflection with different possible color
        scattered = ray(rec.p, reflected + rec.fuzz * random_in_unit_sphere(local_rand_state));
        attenuation = rec.albedo;  // Golden color
    }

    // Ensure the reflected ray is above the surface
    return (dot(scattered.direction(), rec.normal) > 0.0f);
}

__device__ bool scatter_stardust_galaxy(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) {
    // Reflect the ray as usual, using the inward normal for interaction
    vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    scattered = ray(rec.p, reflected + rec.fuzz * random_in_unit_sphere(local_rand_state));

    // Convert hit point to spherical coordinates
    float theta = acosf(-rec.normal.y());  // Using -rec.normal.y to simulate inside view
    float phi = atan2f(-rec.normal.z(), -rec.normal.x());  // Using -rec.normal components
    float u = (phi + M_PI) / (2.0f * M_PI);  // Normalized longitude
    float v = theta / M_PI;  // Normalized latitude

    // Modulate star presence based on UV coordinates
    float pattern = fmodf(u * 15.0f + v * 20.0f, 1.0f);
	float pattern2 = fmodf(u * 20.0f - v * 15.0f, 1.0f);
	float pattern3 = fmodf(u * 10.0f + v * 10.0f, 1.0f);
    float s = curand_uniform(local_rand_state);
    if (s < pattern) {  // Adjust probability based on UV pattern
        float star_brightness = 0.0f + 0.1f * curand_uniform(local_rand_state); // Brighter stars
        attenuation = vec3(star_brightness, star_brightness, star_brightness);
    }
    else if (s < pattern2) {
		float star_brightness = 0.0f + 0.1f * curand_uniform(local_rand_state); // Brighter stars
		attenuation = vec3(star_brightness, star_brightness, star_brightness);
	}
    else if (s > pattern3) {
        float star_brightness = 0.0f + 0.1f * curand_uniform(local_rand_state); // Brighter stars
        attenuation = vec3(star_brightness, star_brightness, star_brightness);
    }
    else {
        attenuation = rec.albedo;  // Dark space
    }

   

    return (dot(scattered.direction(), rec.normal) > 0.0f);
}

__device__ bool scatter_checkered(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) {
    vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    scattered = ray(rec.p, reflected + rec.fuzz * random_in_unit_sphere(local_rand_state));
    // Use position to determine the texture coordinates, assuming the plane is aligned with one of the coordinate planes
    float scale = 0.05f; // This scale adjusts the size of the checkers

    // Assuming the plane is aligned along the Y-axis, using X and Z coordinates for the pattern
    float u = fmodf(fabs(rec.p.x() * scale + 1000.0f), 2.0f);
    float v = fmodf(fabs(rec.p.z() * scale + 1000.0f), 2.0f);

    // Create a checkered pattern using u and v
    bool odd_u = (int(floor(u)) % 2) == 1;
    bool odd_v = (int(floor(v)) % 2) == 1;
    bool checker = odd_u ^ odd_v; // XOR to determine check pattern

    // Alternate between two colors based on the pattern
    if (checker) {
        attenuation = rec.albedo; // White
    }
    else {
        attenuation = vec3(0, 0, 0); // Black
    }

    return (dot(scattered.direction(), rec.normal) > 0);
}

__device__ bool scatter_wave(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) {
    
vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    scattered = ray(rec.p, reflected + rec.fuzz * random_in_unit_sphere(local_rand_state));

    // Adjust the scale to control the frequency and appearance of the waves
    float scale = 0.05f;


    //printf("Time Phase: %f\n",rec.deltaTime);
    // Create a dynamic wave pattern using both x and z coordinates
    float xComponent = rec.p.x() * scale;
    float zComponent = rec.p.z() * scale;
    float combinedWave = sin(xComponent +  zComponent + sin(zComponent));

    // Define a threshold to create visible curved lines
    bool isWave = fabs(combinedWave) < 0.2f; // Adjust this threshold to control the thickness and visibility of the lines

    // Alternate between two colors based on whether the point lies on a wave or not
    if (isWave) {
        // Light grey lines to make the wave visible against a dark background
        attenuation = rec.albedo;
    }
    else {
        // Dark background to contrast with the light grey lines
       
        attenuation = vec3(0.0, 0.0, 0.0);
    }

    return (dot(scattered.direction(), rec.normal) > 0);
}






#endif
