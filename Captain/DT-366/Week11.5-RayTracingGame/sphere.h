#ifndef SPHEREH
#define SPHEREH


#include "material.h"
#include "triangle.h"

float rand01() {
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

vec3 normalize(const vec3& v) {
    float length = sqrt(v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
    // Check if length is zero
    if (length == 0) return vec3(0, 0, 0); // or handle as an error
    return vec3(v.x() / length, v.y() / length, v.z() / length);
}

class spheres {
public:
    // this example use 20KB (max kernel const parameter 32KB)
    vec3 center[100];
    float radius[100];
    vec3 albedo[100];
    float fuzz[100];
    float ref_idx[100];
    int mat_type[100];
    vec3 velocity[100]; // Add velocity for each sphere
    float speed[100];
    bool isProjectile[100]; // To identify if the sphere is a projectile
    bool isBullet[100];  // To identify if the sphere is a bullet
    bool spawning[100];  // To identify if the sphere is currently spawning
    int list_size;
    int lastShotIndex = 2;
    vec3 boundaryCenter = vec3(0, 0, 0); // New boundary center
    bool isEnemy[100];
    float boundaryRadius = 800;
    int score = 0;

    spheres()
    {
        int idx = 0;
        center[idx] = vec3(0, 0, 0);
        radius[idx] = 100;
        albedo[idx] = vec3(1.0, 1.0, 1.0);
        mat_type[idx] = 7;
        idx = 1;

        center[idx] = vec3(0, 103, 0);
        radius[idx] = 3;
        ref_idx[idx] = 1;
        mat_type[idx] = 3;
        isEnemy[idx] = false;
        idx++;



        // Enemies
        for (int i = 0; i < 20; i++) {
            // Position enemies at the boundary
            float theta = rand01() * 2 * M_PI;
            float phi = acos(2 * rand01() - 1);
            float r = boundaryRadius;  // Set radius to boundary radius to start at the boundary

            float x = r * sin(phi) * cos(theta);
            float y = r * sin(phi) * sin(theta);
            float z = r * cos(phi);

            center[idx] = vec3(0, 0, 0);  // Positioned at the boundary
            radius[idx] = 6;
            albedo[idx] = vec3(rand01(), rand01(), rand01());
            mat_type[idx] = 4;
            isEnemy[idx] = true;
            isProjectile[idx] = true;  // Consider if enemies should be considered projectiles
            spawning[idx] = true;
			speed[idx] = 0.5 + rand01() * 0.5;  // Random speed between 0.5 and 1.0 units per update
            velocity[idx] = -normalize(vec3(x, y, z)) * speed[idx];  // Random speed between 0.5 and 1.0 units per update
            idx++;
        }


        //spare bullets
        for (int i = 0; i < 20; i++) {
            // Random position inside sphere[0] with radius less than radius[0]
            float theta = rand01() * 2 * M_PI;
            float phi = acos(2 * rand01() - 1);
            float r = boundaryRadius * cbrt(rand01());  // Use cubic root of random number to ensure uniform distribution

            float x = r * sin(phi) * cos(theta);
            float y = r * sin(phi) * sin(theta);
            float z = r * cos(phi);

            center[idx] = boundaryCenter + vec3(x, y, z);  // Positioned relative to the center of sphere[0]
            radius[idx] = 0.5;  // Smaller radius for the projectile spheres
            albedo[idx] = vec3(rand01(), rand01(), rand01());
            mat_type[idx] = 2;  // Arbitrary material type for example
            isProjectile[idx] = true;  // Set as projectile
            velocity[idx] = vec3(rand01(), rand01(), rand01());  // Initial random velocity
            isBullet[idx] = true;
            isEnemy[idx] = false;
            idx++;
        }


        //Bullets
        float angle_increment = 2 * M_PI / 10;
        for (int i = 0; i < 20; i++) {
            float angle = i * angle_increment;
            float x = center[1].x() + 5 * cos(angle);
            float z = center[1].z() + 5 * sin(angle);
            float y = 1;
            center[idx] = vec3(x, y, z);
            radius[idx] = 0.5;
            albedo[idx] = vec3(rand01(), rand01(), rand01());
            isBullet[idx] = true;
            isProjectile[idx] = false;
            isEnemy[idx] = false;
            mat_type[idx] = 2;
            idx++;
        }




        list_size = idx;
    };

    __device__ bool hitall(const ray& r, float t_min, float t_max, hit_record& rec) const;
    __device__ bool hit(int idx, const ray& r, float tmin, float tmax, hit_record& rec) const;

    vec3 getSpherePosition(int idx) const {
        if (idx >= 0 && idx < list_size) {
            return center[idx];
        }
        return vec3(0, 0, 0); // Default position if index is out of bounds
    }

    void moveSphere(int idx, vec3 displacement) {
        vec3 new_position = center[idx] + displacement;

        // Check if the new position is outside the boundary of the main environment.
        if (!isWithinBounds(new_position, boundaryCenter, boundaryRadius - radius[idx])) {
            center[idx] = adjustPosition(center[idx], displacement, boundaryCenter, boundaryRadius - radius[idx]);
        }
        else {
            // Check for collision with Sphere 0 if it's not Sphere 0 itself.
            if (idx != 0 && isApproachingSphere(center[idx], new_position, center[0], radius[0] + radius[idx])) { // Include radius of the moving sphere
                new_position = preventEntry(center[idx], new_position, center[0], radius[0] + radius[idx]);
            }
            center[idx] = new_position;
        }
    }


    // Determines if the sphere is moving towards or into Sphere 0
    bool isApproachingSphere(vec3 currentPos, vec3 newPos, vec3 sphereCenter, float sphereRadius) {
        float currentDistance = (currentPos - sphereCenter).length();
        float newDistance = (newPos - sphereCenter).length();
        return newDistance <= sphereRadius && currentDistance > newDistance;
    }

    // Adjusts the new position to prevent entry into Sphere 0
    vec3 preventEntry(vec3 currentPos, vec3 newPos, vec3 sphereCenter, float sphereRadius) {
        vec3 fromCenterToNewPos = newPos - sphereCenter;
        float distFromCenterToNewPos = fromCenterToNewPos.length();

        // Adjust the new position so it stops right at the boundary of Sphere 0, minus its own radius
        if (distFromCenterToNewPos < sphereRadius) {
            // Calculate the normalized direction from the sphere center to the new position
            vec3 direction = normalize(fromCenterToNewPos);

            // Adjust new position to be right on the boundary of Sphere 0, but outside
            newPos = sphereCenter + direction * (sphereRadius);
        }

        return newPos;
    }




    bool isWithinBounds(vec3 position, vec3 center, float effectiveRadius) {
        float dist = (position - center).length();
        return dist <= effectiveRadius;
    }

    vec3 adjustPosition(vec3 currentPos, vec3 displacement, vec3 boundaryCenter, float boundaryRadius) {
        vec3 direction = (currentPos + displacement - boundaryCenter).normalize();
        return boundaryCenter + direction * boundaryRadius;
    }


    void updateSpheres(float deltaTime, const vec3& playerPosition, const vec3& bossPosition) {
        static vec3 lastPlayerPosition = playerPosition;  // Keep track of the last player position
        float orbit_radius = 5.0;  // Fixed orbit radius
        float base_orbit_speed = 2.0f;  // Base orbit speed reduced for stability
        float friction = 0.98;
        
      
        for (int i = 2; i < list_size; i++) {
            if (isProjectile[i]) {
                // Apply friction to slow down the projectile

                if (spawning[i]) {
                    // Incrementally increase the radius until it reaches the target
                    float targetRadius = radius[i];  // Arbitrary target radiuss
                    radius[i] += (targetRadius / 1.0f) * deltaTime;

                    if (radius[i] >= targetRadius) {
                        radius[i] = targetRadius;  // Ensure it does not exceed target radius
                        spawning[i] = false;  // Stop spawning animation
                    }
                    else {
                        continue;  // Skip movement updates if still spawning
                    }
                }

                velocity[i] *= friction;

                // Predict the next position
                vec3 nextPosition = center[i] + velocity[i];

                // Collision check with boundary
                vec3 displacementFromBoundary = nextPosition - boundaryCenter;
                float distanceFromBoundary = displacementFromBoundary.length();
                if (distanceFromBoundary > boundaryRadius - radius[i]) {
                    displacementFromBoundary = normalize(displacementFromBoundary);
                    nextPosition = boundaryCenter + displacementFromBoundary * (boundaryRadius - radius[i]);
                    velocity[i] = reflect(velocity[i], displacementFromBoundary);
                    velocity[i] = velocity[i] / 2.0f;
                }

                //collide with player
                vec3 displacementFromPlayer = nextPosition - center[1];
                float distanceFromPlayer = displacementFromPlayer.length();
                if (distanceFromPlayer < radius[1] + radius[i]) {
                    if (!isEnemy[i]) {  // Only change state if it's not an enemy
                        isProjectile[i] = false;
                        // float angle = atan2(displacementFromPlayer.z(), displacementFromPlayer.x());
                         //center[i] = playerPosition + vec3(orbit_radius * cos(angle), center[i].y(), orbit_radius * sin(angle));
                        continue;  // Skip further updates for this sphere
                    }
                }

                vec3 displacementFromEarth = nextPosition - center[0];
                float distanceFromEarth = displacementFromEarth.length();
                if (distanceFromEarth < radius[0] + radius[i]) {
                    displacementFromEarth = normalize(displacementFromEarth);
                    nextPosition = center[0] + displacementFromEarth * (radius[0] - radius[i]);
                    velocity[i] = reflect(velocity[i], displacementFromEarth);
                    velocity[i] = velocity[i] / 2.0f;
                }

                // Update the center of the projectile
                center[i] = nextPosition;
            }
            else {
                float playerMovementEffect = 0.01f;  // Reducing effect of player movement
                float current_angle = atan2(center[i].z() - lastPlayerPosition.z(), center[i].x() - lastPlayerPosition.x());

                // Calculate speed adjustment based on player movement
                vec3 diff = playerPosition - lastPlayerPosition;
                float distanceMoved = diff.length();
                float speedAdjustment = distanceMoved * playerMovementEffect;

                // Update the angle
                current_angle += (base_orbit_speed + speedAdjustment) * deltaTime;
                current_angle = fmod(current_angle, 2 * M_PI);  // Normalize the angle

                float newX = playerPosition.x() + orbit_radius * cos(current_angle);
                float newZ = playerPosition.z() + orbit_radius * sin(current_angle);
                float newY = playerPosition.y();  // Maintain Y position

                center[i] = vec3(newX, newY, newZ);
            }


            if (isEnemy[i]) {
                // Move enemies towards the center

                vec3 toCenter = -normalize(center[i]);  // Direction vector pointing towards the center
                velocity[i] = toCenter * speed[i]; // Set velocity towards center with varying speed
                vec3 nextPosition = center[i] + velocity[i] * deltaTime;
                bool collisionDetected = false;

                // Collision check with the boundary (if they move outside, bring them back to boundary)
                if ((nextPosition - boundaryCenter).length() > boundaryRadius) {
                    nextPosition = boundaryCenter + normalize(nextPosition - boundaryCenter) * boundaryRadius;
                }

                for (int j = 2; j < list_size; j++) {
                    if (isProjectile[j] && isBullet[j]) {
                        vec3 displacementFromEnemies = nextPosition - center[j];
                        float distanceFromEnemies = displacementFromEnemies.length();
                        if (distanceFromEnemies < radius[j] + radius[i]) {
                            int activatedCount = 0;
                            score ++;
							printf("Score: %d\n", score);
							speed[j] = 0.5 + rand01() * 0.5;  // Random speed between 0.5 and 1.0 units per update
                            collisionDetected = true;
                            repositionAtBoundary(i);
                            spawnBullets(5);
                        }
                    }
                }

                vec3 displacementFromEarth = nextPosition - center[0];
                float distanceFromEarth = displacementFromEarth.length();

                if (distanceFromEarth < radius[0] + radius[i]) {
                    displacementFromEarth = normalize(displacementFromEarth);
                    // Position enemies at the boundary
                    radius[0] -= 1.0;
                    albedo[0].setX(albedo[0].getX() + 0.01f);
                    collisionDetected = true;
                    repositionAtBoundary(i);
                }

                if (!collisionDetected) {
                    // No collision, update the position
                    center[i] = nextPosition;
                }
            }
        }

        lastPlayerPosition = playerPosition;
    }

    void repositionAtBoundary(int idx) {
        float theta = rand01() * 2 * M_PI;
        float phi = acos(2 * rand01() - 1);
        float r = boundaryRadius;  // Set radius to boundary radius

        float x = r * sin(phi) * cos(theta);
        float y = r * sin(phi) * sin(theta);
        float z = r * cos(phi);

        center[idx] = vec3(x, y, z);  // New random position on the boundary
        // Optionally reset velocity towards center
        velocity[idx] = -normalize(vec3(x, y, z)) * (rand01() * 0.5 + 0.5);
        spawning[idx] = true;

    }

    void spawnBullets(int count) {
        int activatedCount = 0;
        for (int i = 2; i < list_size && activatedCount < count; i++) {
            if (isBullet[i] && isProjectile[i]) {  // Correct condition to find active projectiles
                isProjectile[i] = false;
                activatedCount++;
            }
        }
    }

    int findNextSphereToShoot() {
        // Find the next non-projectile sphere to shoot
        for (int i = 2; i <= list_size; i++) {
            if (!isProjectile[i]) {
                return i;
            }
        }
        return -1;  // No available sphere to shoot
    }

    void shootSphere(vec3 direction, float speed) {  // Increased default speed for testing
        int idx = findNextSphereToShoot();
        if (idx != -1) {
            isProjectile[idx] = true;
            velocity[idx] = normalize(direction) * speed;
        }
    }


    vec3 reflect(const vec3& v, const vec3& n) {
        return v - 2 * dot(v, n) * n;
    }

    vec3 rotate(const vec3& v, float angle, const vec3& axis) {
        // Convert angle to radians for trigonometric functions
        float rad = angle * M_PI / 180.0f;
        vec3 axis_normalized = axis.normalize();
        return v * cos(rad) + cross(axis_normalized, v) * sin(rad) + axis_normalized * dot(axis_normalized, v) * (1 - cos(rad));
    }


    int findOrbitingSphere() const {
        for (int i = 0; i < list_size; i++) {
            if (!isProjectile[i] && center[i].y() > 0) {  // Example condition
                return i;
            }
        }
        return -1;  // Return -1 if no suitable sphere is found
    }

};

__device__ bool spheres::hitall(const ray& r, float t_min, float t_max, hit_record& rec) const {
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

__device__ bool spheres::hit(int idx, const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center[idx];
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius[idx] * radius[idx];
    float discriminant = b * b - a * c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center[idx]) / radius[idx];
            rec.albedo = albedo[idx];
            rec.fuzz = fuzz[idx];
            rec.ref_idx = ref_idx[idx];
            rec.mat_type = mat_type[idx];
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center[idx]) / radius[idx];
            rec.albedo = albedo[idx];
            rec.fuzz = fuzz[idx];
            rec.ref_idx = ref_idx[idx];
            rec.mat_type = mat_type[idx];
            return true;
        }
    }
    return false;
}


#endif