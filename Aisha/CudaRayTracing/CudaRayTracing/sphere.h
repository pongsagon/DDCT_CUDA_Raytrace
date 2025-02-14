#ifndef SPHEREH
#define SPHEREH


#include "material.h"


float rand01() {
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

struct Room {
    vec3 minCorner; // Minimum x, y, z boundary
    vec3 maxCorner; // Maximum x, y, z boundary
    int roomID;     // Identifier for the room
};

class spheres {

public:

    Room rooms[3];      // Define two rooms
    int currentRoomID;  // Tracks the current room ID of the camera

    spheres()
    {

        //MIRROR ROOM
        rooms[0] = { vec3(-30.0f, 0.0f, -30.0f), vec3(30.0f, 30.0f, 30.0f), 0 }; // Mirror room


        //CORRIDOR
        rooms[1] = {
            vec3(-70.0f, 0.0f, 10.0f),                 // Minimum corner (bottom-left)
            vec3(-60.0f, 20.0f, 70.0f),               // Maximum corner (top-right)
            1                                         // Room ID for the corridor
        };

        float room2StartX = 80.0f; 
        float room2StartZ = -90.0f;

        //REFRACTION ROOM
        rooms[2] = {
            vec3(room2StartX, 0.0f, room2StartZ),
            vec3(room2StartX + 60.0f, 30.0f, room2StartZ + 60.0f),
            2
        };


        currentRoomID = 0;


        //int idx = 0;

        //// Adjusted loop limits for fewer spheres
        //for (int a = -5; a < 5; a++) {
        //    for (int b = -5; b < 5; b++) {
        //        float choose_mat = rand01();
        //        vec3 _center(a + rand01(), 0.2, b + rand01());
        //        if (choose_mat < 0.8f) {
        //            center[idx] = _center;
        //            radius[idx] = 0.2;
        //            albedo[idx] = vec3(rand01() * rand01(), rand01() * rand01(), rand01() * rand01());
        //            mat_type[idx] = 1;
        //            idx++;
        //        }
        //        else if (choose_mat < 0.9f) {
        //            center[idx] = _center;
        //            radius[idx] = 0.2;
        //            albedo[idx] = vec3(0.5f * (1.0f + rand01()), 0.5f * (1.0f + rand01()), 0.5f * (1.0f + rand01()));
        //            fuzz[idx] = 0.5f * rand01();
        //            mat_type[idx] = 2;
        //            idx++;
        //        }
        //        else {
        //            center[idx] = _center;
        //            radius[idx] = 0.2;
        //            ref_idx[idx] = 1.5;
        //            mat_type[idx] = 3;
        //            idx++;
        //        }
        //    }
        //}

        //// Checkered material sphere
        //center[idx] = vec3(2, 1, 0);  // Position of the checkered sphere
        //radius[idx] = 1.0;
        //mat_type[idx] = 4;  // Checkered material
        //idx++;

        //center[idx] = vec3(-7, 1, 0); // Light sphere
        //radius[idx] = 1.0;
        //mat_type[idx] = 5;
        //idx++;

        //center[idx] = vec3(0, 1, 0);
        //radius[idx] = 1;
        //ref_idx[idx] = 1.5;
        //mat_type[idx] = 3;
        //idx++;

        //center[idx] = vec3(-4, 1, 0);
        //radius[idx] = 1;
        //albedo[idx] = vec3(0.4, 0.2, 0.1);
        //mat_type[idx] = 1;
        //idx++;

        //center[idx] = vec3(4, 1, 0);
        //radius[idx] = 1;
        //albedo[idx] = vec3(0.7, 0.6, 0.5);
        //fuzz[idx] = 0.0;
        //mat_type[idx] = 2;
        //idx++;

        //list_size = 10 * 10 + 5;


        //Camera sphere
        int idx = 0;
        center[idx] = vec3(0, 5.0, 0.0);
        radius[idx] = 1.0f;
        albedo[idx] = vec3(0.7, 0.5, 0.2);
        mat_type[idx] = 2;
        idx++; 

        //--------------------ROOM 1-------------------------

        float roomMin = -29.0f;
        float roomMax = 29.0f;
        float centerAvoidMin = -17.0f;
        float centerAvoidMax = 17.0f;
        float minDistance = 2.8f;

        int totalGroundSpheres = 60;
        int placedPairs = 0;
        int totalFloatingSpheres = 10;
        int totalColorChangingSheres = 15;
     
        orbStartIndex = idx;

        while (idx < totalGroundSpheres + 10) {

            float margin = 2.0f;
            float x = (rooms[0].minCorner.x() + margin) + rand01() * ((rooms[0].maxCorner.x() - margin) - (rooms[0].minCorner.x() + margin));
            float z = (rooms[0].minCorner.z() + margin) + rand01() * ((rooms[0].maxCorner.z() - margin) - (rooms[0].minCorner.z() + margin));


            if (x > centerAvoidMin && x < centerAvoidMax && z > centerAvoidMin && z < centerAvoidMax) {
                continue;
            }

            bool overlaps = false;
            for (int i = 0; i < idx; i++) {
                float distSq = (x - center[i].x()) * (x - center[i].x()) + (z - center[i].z()) * (z - center[i].z());
                if (distSq < minDistance * minDistance) {
                    overlaps = true;
                    break;
                }
            }

            if (overlaps) continue;

            if (placedPairs < 5) {

                vec3 pairColor = orbColors[placedPairs];

                // First sphere (colored)
                center[idx] = vec3(x, 0.35f, z);
                radius[idx] = 0.35f;
                albedo[idx] = pairColor; // Match ceiling color
                mat_type[idx] = 2;       // Colored sphere material type
                idx++;

                // Second sphere (mirror material), slightly offset
                center[idx] = vec3(x, 0.35f, z); // Slight offset in Z direction
                radius[idx] = 0.45f;
                albedo[idx] = vec3(0.0, 0.0, 0.0); 
                ref_idx[idx] = -1;                 
                mat_type[idx] = 3;                 
                idx++;

                placedPairs++; // Increment placed pair count
            }
            else {
                // Place a regular ground sphere
                float rand_radius = 1.2f + rand01() * 0.8f;
                radius[idx] = rand_radius;
                center[idx] = vec3(x, rand_radius, z);
                albedo[idx] = vec3(0.7, 0.7, 0.7);
                fuzz[idx] = 0.0f;
                mat_type[idx] = 2;
                idx++;
            }
        }


        while (idx < totalGroundSpheres + totalFloatingSpheres) {     

            float margin = 2.0f;
            float x = (rooms[0].minCorner.x() + margin) + rand01() * ((rooms[0].maxCorner.x() - margin) - (rooms[0].minCorner.x() + margin));
            float z = (rooms[0].minCorner.z() + margin) + rand01() * ((rooms[0].maxCorner.z() - margin) - (rooms[0].minCorner.z() + margin));

            float height = 5.0f + rand01() * 9.0f;  

            bool overlaps = false;
            for (int i = totalGroundSpheres; i < idx; i++) {
                float distSq = (x - center[i].x()) * (x - center[i].x()) + (z - center[i].z()) * (z - center[i].z());
                if (distSq < minDistance * minDistance * (height / 10.0f)) {  
                    overlaps = true;
                    break;
                }
            }

            if (!overlaps) {
                float rand_radius = 1.2f + rand01() * 0.8f;
                center[idx] = vec3(x, height, z);  // Floating sphere positioned at random height
                radius[idx] = rand_radius;
                albedo[idx] = vec3(0.7, 0.7, 0.7);
                fuzz[idx] = 0.0f; 
                mat_type[idx] = 2;
                idx++;
            }
        }

        
        StartIdx_Room2Spheres = idx;

        while (idx < totalGroundSpheres + totalFloatingSpheres + totalColorChangingSheres) {

            float margin = 4.0f;
            float x = (rooms[2].minCorner.x() + margin) + rand01() * ((rooms[2].maxCorner.x() - margin) - (rooms[2].minCorner.x() + margin));
            float z = (rooms[2].minCorner.z() + margin) + rand01() * ((rooms[2].maxCorner.z() - margin) - (rooms[2].minCorner.z() + margin));

            // Check for overlaps
            bool overlaps = false;
            for (int i = 0; i < idx; i++) {
                float distSq = (x - center[i].x()) * (x - center[i].x()) + (z - center[i].z()) * (z - center[i].z());
                if (distSq < minDistance * minDistance) {
                    overlaps = true;
                    break;
                }
            }

            if (!overlaps) {
 
                radius[idx] = 3.0f;
                center[idx] = vec3(x, 3.0f, z);

                velocity[idx] = vec3(
                    (rand01() - 0.5f) * 0.002f,  
                    0.0f,                        
                    (rand01() - 0.5f) * 0.002f   
                );

                albedo[idx] = vec3(0.0f, 0.0f, 0.0f);
                ref_idx[idx] = 1.5f;
                mat_type[idx] = 3;
                idx++;

            }

        }

        
        // Set the list size
        list_size = idx;

        //------------------ROOM 2---------------------------


    };

    __device__ bool hitall(const ray& r, float t_min, float t_max, hit_record& rec) const;
    __device__ bool hit(int idx, const ray& r, float tmin, float tmax, hit_record& rec) const;
   

    // this example use 20KB (max kernel const parameter 32KB)
    vec3 center[200];
    float radius[200];
    vec3 albedo[200];
    float fuzz[200];
    float ref_idx[200];
    int mat_type[200];
    int list_size;
    bool isColliding[200] = { false };
    vec3 velocity[200];


    vec3 predefinedColors[5] = {
        vec3(0.8, 0.9, 0.9), // Soft pastel teal
        vec3(0.9, 0.8, 0.8), // Soft pastel pink
        vec3(0.8, 0.9, 1.0), // Soft pastel blue
        vec3(0.9, 0.9, 0.8), // Soft pastel yellow
        vec3(0.8, 0.9, 0.8)  // Soft pastel green
    };

    vec3 orbColors[5] = {
       vec3(0.9, 0.2, 0.2), // Vivid red
       vec3(0.2, 0.9, 0.2), // Vivid green
       vec3(0.2, 0.2, 0.9), // Vivid blue
       vec3(0.9, 0.9, 0.2), // Vivid yellow
       vec3(0.9, 0.2, 0.9)  // Vivid magenta
    };



    void update_current_room(const vec3& cameraPosition) {

        for (int i = 0; i < 2; i++) {
            if (cameraPosition.x() >= rooms[i].minCorner.x() && cameraPosition.x() <= rooms[i].maxCorner.x() &&
                cameraPosition.y() >= rooms[i].minCorner.y() && cameraPosition.y() <= rooms[i].maxCorner.y() &&
                cameraPosition.z() >= rooms[i].minCorner.z() && cameraPosition.z() <= rooms[i].maxCorner.z()) {
                currentRoomID = rooms[i].roomID;
                break;
            }
        }

    }


    void room1_sphere_follow_camera(float elapsedTime, vec3 lookFrom, vec3 lookAt) {

        float distanceBehind = 2.0f;  // Distance behind the camera

        vec3 direction = unit_vector(lookFrom - lookAt);

        center[0].setX(lookFrom.e[0] + direction.e[0] * distanceBehind);   
        center[0].setZ(lookFrom.e[2] + direction.e[2] * distanceBehind);

    }

    void room3_followcam(float elapsedTime, vec3 lookFrom, vec3 lookAt) {

        // Keep the sphere on the ground at y = 1.0f
        center[0].setY(1.0f);

        // Lock x and z to the camera's x and z position
        center[0].setX(lookFrom.x());
        center[0].setZ(lookFrom.z());

    }

    void update_moving_spheres(float deltaTime) {

        const Room& room2 = rooms[2];

        for (int i = StartIdx_Room2Spheres; i < list_size; i++) {
            if (i == 0) continue; // Skip the camera-follow sphere

            // Update position based on velocity
            center[i] += velocity[i] * deltaTime;

            // (Walls)
            if (center[i].x() - radius[i] < room2.minCorner.x() || center[i].x() + radius[i] > room2.maxCorner.x()) {
                velocity[i].setX(-velocity[i].x()); // Reverse X direction
            }
            if (center[i].z() - radius[i] < room2.minCorner.z() || center[i].z() + radius[i] > room2.maxCorner.z()) {
                velocity[i].setZ(-velocity[i].z()); // Reverse Z direction
            }

            // Sphere-to-Sphere Collisions
            for (int j = i + 1; j < list_size; j++) {
                if (j == 0) continue; // Skip camera-follow sphere

                vec3 diff = center[j] - center[i];
                float distanceSq = dot(diff, diff);
                float radiusSum = radius[i] + radius[j];

                if (distanceSq < radiusSum * radiusSum) { 

                    vec3 collisionNormal = unit_vector(diff);            
                    // Reflect velocities along the collision normal while reducing speed slightly
                    vec3 newVelocity_i = velocity[i] - 2.0f * dot(velocity[i], collisionNormal) * collisionNormal;
                    vec3 newVelocity_j = velocity[j] - 2.0f * dot(velocity[j], collisionNormal) * collisionNormal;

                    // Apply small dampening effect to avoid excessive bouncing
                    newVelocity_i *= 0.9f;
                    newVelocity_j *= 0.9f;

                    // Set Y to 0 to ensure no vertical movement
                    newVelocity_i.setY(0.0f);
                    newVelocity_j.setY(0.0f);

                    velocity[i] = newVelocity_i;
                    velocity[j] = newVelocity_j;
                    
                    float distance = sqrt(distanceSq);
                    float overlap = radiusSum - distance;
                    vec3 correction = collisionNormal * (overlap * 0.5f);
                    center[i] -= correction;
                    center[j] += correction;

                }
            }

            //Collision with Camera Sphere (Index 0)
            vec3 toCamera = center[0] - center[i];
            float distToCameraSq = dot(toCamera, toCamera);
            float cameraRadiusSum = radius[i] + radius[0];

            if (distToCameraSq < cameraRadiusSum * cameraRadiusSum) {

            


                //vec3 collisionNormal = unit_vector(toCamera);

                //// Reflect velocity away from the camera-follow sphere
                //vec3 newVelocity_i = velocity[i] - 2.0f * dot(velocity[i], collisionNormal) * collisionNormal;
                //newVelocity_i *= 0.9f; // Apply slight dampening
                //newVelocity_i.setY(0.0f); // Keep Y at 0

                //velocity[i] = newVelocity_i;


                //// Separate from the camera-follow sphere
                //float distance = sqrt(distToCameraSq);
                //float overlap = cameraRadiusSum - distance;
                //center[i] -= collisionNormal * overlap * 0.5f;
                //center[0] += collisionNormal * overlap * 0.5f; 

                //// **Prevent Camera Sphere from Entering Other Spheres**
                //// If overlap still exists, push the camera-follow sphere out to maintain proper separation
                //float adjustedDist = sqrt(dot(center[0] - center[i], center[0] - center[i]));
                //if (adjustedDist < cameraRadiusSum) {
                //    vec3 correction = collisionNormal * (cameraRadiusSum - adjustedDist);
                //    center[0] += correction;
                //}
            }
        }

    }


    void update_floating_spheres(float deltaTime) {

        const Room& room2 = rooms[2];

        for (int i = StartIdx_Room2Spheres; i < StartIdx_Room2Spheres + 15; i++) {

            center[i] += velocity[i] * deltaTime * 0.1f;
                
            if (center[i].x() - radius[i] < room2.minCorner.x() || center[i].x() + radius[i] > room2.maxCorner.x()) {
                velocity[i].setX(-velocity[i].x()); // Reverse X direction
            }
            if (center[i].y() - radius[i] < room2.minCorner.y() || center[i].y() + radius[i] > room2.maxCorner.y()) {
                velocity[i].setY(-velocity[i].y()); // Reverse Y direction
            }
            if (center[i].z() - radius[i] < room2.minCorner.z() || center[i].z() + radius[i] > room2.maxCorner.z()) {
                velocity[i].setZ(-velocity[i].z()); // Reverse Z direction
            }

            // Sphere-to-sphere collision checks
            for (int j = i + 1; j < list_size; j++) {
                if (mat_type[j] == 5) {
                    vec3 diff = center[j] - center[i];
                    float distanceSq = dot(diff, diff);
                    float radiusSum = radius[i] + radius[j];

                    if (distanceSq < radiusSum * radiusSum) {
                        // Collision detected
                        vec3 collisionNormal = unit_vector(diff); // Unit vector along collision normal

                        // Reverse velocity of each sphere along the collision normal
                        velocity[i] -= 2.0f * dot(velocity[i], collisionNormal) * collisionNormal;
                        velocity[j] -= 2.0f * dot(velocity[j], collisionNormal) * collisionNormal;

                        // Slightly move spheres apart to resolve overlap
                        float distance = sqrt(distanceSq);
                        float overlap = radiusSum - distance;
                        vec3 correction = collisionNormal * (overlap * 0.5f);
                        center[i] -= correction;
                        center[j] += correction;
                    }
                }

            }

        }
        

    }

    bool check_orb_match(int ceilingColorIndex, vec3 lookFrom) {

        for (int i = orbStartIndex; i <= orbStartIndex + 10; i += 2) {
            vec3 orbColor = albedo[i]; // Colored sphere albedo
            int orbIndex = (i - orbStartIndex) / 2; // Match orb index to ceiling index

            // Check if the orb's index matches the ceiling's color index
            if (orbIndex == ceilingColorIndex) {
                // Ignore the Y-axis difference
                vec3 horizontalDifference = vec3(lookFrom.x() - center[i].x(), 0.0f, lookFrom.z() - center[i].z());
                float distanceSq = dot(horizontalDifference, horizontalDifference);

                if (distanceSq < 2.0f * 2.0f) {
                    return true; // Player is close to the orb in the X-Z plane
                }
            }
        }
        return false;

    }

    bool isCameraNearPillar(const vec3& lookFrom) {

        vec3 pillarCenter(0.0f, 0.0f, 0.0f);
        vec3 horizontalDifference = vec3(lookFrom.x() - pillarCenter.x(), 0.0f, lookFrom.z() - pillarCenter.z());
        float distanceSq = dot(horizontalDifference, horizontalDifference);
        return distanceSq <= 4.0f * 4.0f;

    }

    bool isCameraNearCorridorEnd(const vec3& lookFrom) {

        float corridorEndZ = 10.0f + 60.0f; // Z position of the glowing wall at the end of the corridor
        float tolerance = 1.0f; // Allow some margin for triggering

        return lookFrom.z() >= corridorEndZ - tolerance;
    }




    private:
        int orbStartIndex = 0;
        int StartIdx_Room2Spheres = 0;
        int warpSphereIndex = 0;

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
