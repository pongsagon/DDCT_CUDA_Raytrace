#ifndef TRIANGLEH
#define TRIANGLEH


#include "material.h"


class triangle {
public:
    triangle()
    {

        int idx = 0;

        //----------------------- ROOM 1 ---------------------------------------------------
        
        v0[idx] = vec3(30.0, 0.0, 30.0);
        v1[idx] = vec3(30.0, 0.0, -30.0);
        v2[idx] = vec3(-30.0, 0.0, -30.0);

        albedo[0] = vec3(0.1, 0.25, 0.1);
        mat_type[0] = 1;

        idx++;

        v0[idx] = vec3(-30.0, 0.0, 30.0);
        v1[idx] = vec3(30.0, 0.0, 30.0);
        v2[idx] = vec3(-30.0, 0.0, -30.0);

        albedo[idx] = vec3(0.1, 0.25, 0.1);
        mat_type[idx] = 1;

        idx++;

        list_size = idx;

        //ROOM WALLS

        float wallHeight = 30.0f;

        // Wall 1 (Left wall at X = -30.0)
        v0[idx] = vec3(-30.0, 0.0, -30.0); // Bottom-left
        v1[idx] = vec3(-30.0, wallHeight, 30.0); // Top-right
        v2[idx] = vec3(-30.0, 0.0, 30.0);  // Bottom-right
        idx++;

        v0[idx] = vec3(-30.0, 0.0, -30.0); // Bottom-left
        v1[idx] = vec3(-30.0, wallHeight, -30.0); // Top-left
        v2[idx] = vec3(-30.0, wallHeight, 30.0); // Top-right
        idx++;

        // Wall 2 (Bottom wall at Z = -30.0)
        v0[idx] = vec3(-30.0, 0.0, -30.0); // Bottom-left
        v1[idx] = vec3(30.0, 0.0, -30.0);  // Bottom-right
        v2[idx] = vec3(30.0, wallHeight, -30.0); // Top-right
        idx++;

        v0[idx] = vec3(-30.0, 0.0, -30.0); // Bottom-left
        v1[idx] = vec3(30.0, wallHeight, -30.0); // Top-right
        v2[idx] = vec3(-30.0, wallHeight, -30.0); // Top-left
        idx++;

        // Wall 3 (Right wall at X = 30.0)
        v0[idx] = vec3(30.0, 0.0, -30.0); // Bottom-left
        v1[idx] = vec3(30.0, 0.0, 30.0);  // Bottom-right
        v2[idx] = vec3(30.0, wallHeight, 30.0); // Top-right
        idx++;

        v0[idx] = vec3(30.0, 0.0, -30.0); // Bottom-left
        v1[idx] = vec3(30.0, wallHeight, 30.0); // Top-right
        v2[idx] = vec3(30.0, wallHeight, -30.0); // Top-left
        idx++;

        // Wall 4 (Top wall at Z = 30.0)
        v0[idx] = vec3(-30.0, 0.0, 30.0); // Bottom-left
        v1[idx] = vec3(30.0, wallHeight, 30.0); // Top-right
        v2[idx] = vec3(30.0, 0.0, 30.0);  // Bottom-right
        idx++;

        v0[idx] = vec3(-30.0, 0.0, 30.0); // Bottom-left
        v1[idx] = vec3(-30.0, wallHeight, 30.0); // Top-left
        v2[idx] = vec3(30.0, wallHeight, 30.0); // Top-right
        idx++;


        for (int i = list_size; i < 10; i++) {
            albedo[i] = vec3(0.7, 0.7, 0.7);
            fuzz[i] = 0.0;
            mat_type[i] = 2;
        }
        
        
        //GLASS PILLAR

        float pillarHeight = wallHeight;
        float pillarWidth = 4.0f;
        float halfWidth = pillarWidth / 2.0f;

        //10 
        
        // Side 1 (Front)
        v0[idx] = vec3(-halfWidth, 0.0, -halfWidth);    
        v1[idx] = vec3(halfWidth, pillarHeight, -halfWidth); 
        v2[idx] = vec3(halfWidth, 0.0, -halfWidth);    
        idx++;

        v0[idx] = vec3(-halfWidth, 0.0, -halfWidth);   
        v1[idx] = vec3(-halfWidth, pillarHeight, -halfWidth); 
        v2[idx] = vec3(halfWidth, pillarHeight, -halfWidth); 
        idx++;

        // Side 2 (Right)
        v0[idx] = vec3(halfWidth, 0.0, -halfWidth);    
        v1[idx] = vec3(halfWidth, pillarHeight, halfWidth); 
        v2[idx] = vec3(halfWidth, 0.0, halfWidth);   
        idx++;

        v0[idx] = vec3(halfWidth, 0.0, -halfWidth);    
        v1[idx] = vec3(halfWidth, pillarHeight, -halfWidth); 
        v2[idx] = vec3(halfWidth, pillarHeight, halfWidth); 
        idx++;

        // Side 3 (Back)
        v0[idx] = vec3(halfWidth, 0.0, halfWidth); 
        v1[idx] = vec3(-halfWidth, pillarHeight, halfWidth); 
        v2[idx] = vec3(-halfWidth, 0.0, halfWidth);    
        idx++;

        v0[idx] = vec3(halfWidth, 0.0, halfWidth);   
        v1[idx] = vec3(halfWidth, pillarHeight, halfWidth); 
        v2[idx] = vec3(-halfWidth, pillarHeight, halfWidth); 
        idx++;

        // Side 4 (Left)
        v0[idx] = vec3(-halfWidth, 0.0, halfWidth);  
        v1[idx] = vec3(-halfWidth, pillarHeight, -halfWidth);
        v2[idx] = vec3(-halfWidth, 0.0, -halfWidth);  
        idx++;

        v0[idx] = vec3(-halfWidth, 0.0, halfWidth);    
        v1[idx] = vec3(-halfWidth, pillarHeight, halfWidth); 
        v2[idx] = vec3(-halfWidth, pillarHeight, -halfWidth);
        idx++;

        for (int i = list_size; i < idx; i++) {
            albedo[i] = vec3(0.7, 0.7, 0.7);
            fuzz[i] = 0.0;
            mat_type[i] = 2; 
        }

        //LIGHT CEILING
        
        // Triangle 1
        v0[idx] = vec3(-30.0, wallHeight, 30.0);  // Top Left
        v1[idx] = vec3(30.0, wallHeight, -30.0);  // Bottom Right
        v2[idx] = vec3(30.0, wallHeight, 30.0);   // Top Right
        albedo[idx] = vec3(1.6, 1.0, 2.0);
        mat_type[idx] = 5;
        idx++;

        // Triangle 2
        v0[idx] = vec3(-30.0, wallHeight, 30.0);  // Top Left
        v1[idx] = vec3(-30.0, wallHeight, -30.0); // Bottom Left
        v2[idx] = vec3(30.0, wallHeight, -30.0);  // Bottom Right
        albedo[idx] = vec3(1.6, 1.0, 2.0);
        mat_type[idx] = 5;
        idx++;

        //----------------------- ROOM 3 ---------------------------------------------------

        float room2StartX = 80.0;  // Room 2 offset to the right
        float room2StartZ = -90.0; // Room 2 offset behind Room 1

        v0[idx] = vec3(room2StartX, 0.0f, room2StartZ);                      // Bottom-left
        v1[idx] = vec3(room2StartX + 60.0f, 0.0f, room2StartZ);              // Bottom-right
        v2[idx] = vec3(room2StartX + 60.0f, 0.0f, room2StartZ + 60.0f);      // Top-right
        albedo[idx] = vec3(0.0f, 0.0f, 0.0f);  // Black floor
        mat_type[idx] = 6;   // Floor material
        idx++;

        // Triangle 2 for the floor
        v0[idx] = vec3(room2StartX, 0.0f, room2StartZ);                      // Bottom-left
        v1[idx] = vec3(room2StartX + 60.0f, 0.0f, room2StartZ + 60.0f);      // Top-right
        v2[idx] = vec3(room2StartX, 0.0f, room2StartZ + 60.0f);              // Top-left
        albedo[idx] = vec3(0.0f, 0.0f, 0.0f);  // Black floor 
        mat_type[idx] = 6;   // Floor material
        idx++;

        fuzz[idx] = 0.05f;
        create_wall(
            vec3(room2StartX, 0.0, room2StartZ),                // Bottom-left
            vec3(room2StartX, wallHeight, room2StartZ),         // Top-left
            vec3(room2StartX, wallHeight, room2StartZ + 60.0),  // Top-right
            vec3(room2StartX, 0.0, room2StartZ + 60.0),         // Bottom-right
            vec3(0.8, 0.1, 0.1), idx, 2);                         // Dark grey

        // Right wall at room2StartX + 60
        fuzz[idx] = 0.05f;
        create_wall(
            vec3(room2StartX + 60.0, 0.0, room2StartZ),                // Bottom-left
            vec3(room2StartX + 60.0, wallHeight, room2StartZ),         // Top-left
            vec3(room2StartX + 60.0, wallHeight, room2StartZ + 60.0),  // Top-right
            vec3(room2StartX + 60.0, 0.0, room2StartZ + 60.0),         // Bottom-right
            vec3(0.8, 0.1, 0.1), idx, 2);                                // Dark grey

        // Back wall at room2StartZ
        create_wall(
            vec3(room2StartX, 0.0, room2StartZ),                // Bottom-left
            vec3(room2StartX + 60.0, 0.0, room2StartZ),         // Bottom-right
            vec3(room2StartX + 60.0, wallHeight, room2StartZ),  // Top-right
            vec3(room2StartX, wallHeight, room2StartZ),         // Top-left
            vec3(0.1, 0.1, 0.1), idx);                         // Dark grey

        // Front wall at room2StartZ + 60
        create_wall(
            vec3(room2StartX, 0.0, room2StartZ + 60.0),                // Bottom-left
            vec3(room2StartX, wallHeight, room2StartZ + 60.0),         // Top-left
            vec3(room2StartX + 60.0, wallHeight, room2StartZ + 60.0),  // Top-right
            vec3(room2StartX + 60.0, 0.0, room2StartZ + 60.0),         // Bottom-right
            vec3(0.1, 0.1, 0.1), idx);

        //// Ceiling of Room 2 (Black)
        //v0[idx] = vec3(room2StartX, wallHeight, room2StartZ);
        //v1[idx] = vec3(room2StartX + 60.0, wallHeight, room2StartZ);
        //v2[idx] = vec3(room2StartX + 60.0, wallHeight, room2StartZ + 60.0);
        //albedo[idx] = vec3(0.1, 0.1, 0.1);  // Black color
        //mat_type[idx] = 1;
        //idx++;

        //v0[idx] = vec3(room2StartX, wallHeight, room2StartZ);
        //v1[idx] = vec3(room2StartX + 60.0, wallHeight, room2StartZ + 60.0);
        //v2[idx] = vec3(room2StartX, wallHeight, room2StartZ + 60.0);
        //albedo[idx] = vec3(0.1, 0.1, 0.1);  // Black color
        //mat_type[idx] = 1;
        //idx++;


        //----------------------- ROOM 2 ---------------------------------------------------
        float corridorStartX = -70.0f;  // Starting X position
        float corridorStartZ = 10.0f;   // Starting Z position
        float corridorWidth = 10.0f;    // Narrow width
        float corridorHeight = 20.0f;   // Height of the corridor
        float corridorLength = 60.0f;  // Very long corridor length

        // Floor
        
       // First triangle
        v0[idx] = vec3(corridorStartX, 0.0f, corridorStartZ);                         // Bottom-left
        v1[idx] = vec3(corridorStartX + corridorWidth, 0.0f, corridorStartZ + corridorLength); // Top-right
        v2[idx] = vec3(corridorStartX + corridorWidth, 0.0f, corridorStartZ);         // Bottom-right
        albedo[idx] = vec3(0.1f, 0.1f, 0.1f); // Dark grey
        fuzz[idx] = 0.05f;
        mat_type[idx] = 2;
        idx++;

        // Second triangle
        v0[idx] = vec3(corridorStartX, 0.0f, corridorStartZ);                         // Bottom-left
        v1[idx] = vec3(corridorStartX, 0.0f, corridorStartZ + corridorLength);        // Top-left
        v2[idx] = vec3(corridorStartX + corridorWidth, 0.0f, corridorStartZ + corridorLength); // Top-right
        albedo[idx] = vec3(0.1f, 0.1f, 0.1f); // Dark grey
        fuzz[idx] = 0.05f;
        mat_type[idx] = 2;
        idx++;


        //cieling
        
        v0[idx] = vec3(corridorStartX, corridorHeight, corridorStartZ);                        // Bottom-left
        v1[idx] = vec3(corridorStartX + corridorWidth, corridorHeight, corridorStartZ);        // Bottom-right
        v2[idx] = vec3(corridorStartX + corridorWidth, corridorHeight, corridorStartZ + corridorLength); // Top-right
        albedo[idx] = vec3(0.1f, 0.1f, 0.1f); // Dark grey
        fuzz[idx] = 0.05f;
        mat_type[idx] = 2;
        idx++;

        v0[idx] = vec3(corridorStartX, corridorHeight, corridorStartZ);                        // Bottom-left
        v1[idx] = vec3(corridorStartX + corridorWidth, corridorHeight, corridorStartZ + corridorLength); // Top-right
        v2[idx] = vec3(corridorStartX, corridorHeight, corridorStartZ + corridorLength);       // Top-left
        albedo[idx] = vec3(0.1f, 0.1f, 0.1f); // Dark grey
        fuzz[idx] = 0.05f;
        mat_type[idx] = 2;
        idx++;

        // Right Wall
        
        v0[idx] = vec3(corridorStartX, 0.0f, corridorStartZ);                        // Bottom-left
        v1[idx] = vec3(corridorStartX, corridorHeight, corridorStartZ);              // Top-left
        v2[idx] = vec3(corridorStartX, corridorHeight, corridorStartZ + corridorLength); // Top-right
        albedo[idx] = vec3(0.1f, 0.1f, 0.1f); // Dark grey
        fuzz[idx] = 0.05f;
        mat_type[idx] = 2;
        idx++;

        v0[idx] = vec3(corridorStartX, 0.0f, corridorStartZ);                        // Bottom-left
        v1[idx] = vec3(corridorStartX, corridorHeight, corridorStartZ + corridorLength); // Top-right
        v2[idx] = vec3(corridorStartX, 0.0f, corridorStartZ + corridorLength);
        albedo[idx] = vec3(0.1f, 0.1f, 0.1f); // Dark grey
        fuzz[idx] = 0.05f;
        mat_type[idx] = 2;
        idx++;

        // left Wall
        
        v0[idx] = vec3(corridorStartX + corridorWidth, 0.0f, corridorStartZ);                        // Bottom-left
        v1[idx] = vec3(corridorStartX + corridorWidth, 0.0f, corridorStartZ + corridorLength);       // Bottom-right
        v2[idx] = vec3(corridorStartX + corridorWidth, corridorHeight, corridorStartZ + corridorLength); // Top-right
        albedo[idx] = vec3(0.1f, 0.1f, 0.1f); // Dark grey
        fuzz[idx] = 0.05f;
        mat_type[idx] = 2;
        idx++;

        v0[idx] = vec3(corridorStartX + corridorWidth, 0.0f, corridorStartZ);                        // Bottom-left
        v1[idx] = vec3(corridorStartX + corridorWidth, corridorHeight, corridorStartZ + corridorLength); // Top-right
        v2[idx] = vec3(corridorStartX + corridorWidth, corridorHeight, corridorStartZ);             // Top-left
        albedo[idx] = vec3(0.1f, 0.1f, 0.1f); // Dark grey
        fuzz[idx] = 0.05f;
        mat_type[idx] = 2;
        idx++;

        // Back Wall
        fuzz[idx] = 0.05f;
        v0[idx] = vec3(corridorStartX, 0.0f, corridorStartZ);                        // Bottom-left
        v1[idx] = vec3(corridorStartX + corridorWidth, 0.0f, corridorStartZ);        // Bottom-right
        v2[idx] = vec3(corridorStartX + corridorWidth, corridorHeight, corridorStartZ); // Top-right
        albedo[idx] = vec3(0.1f, 0.1f, 0.1f); // Dark grey
        fuzz[idx] = 0.05f;
        mat_type[idx] = 2;
        idx++;

        v0[idx] = vec3(corridorStartX, 0.0f, corridorStartZ);                        // Bottom-left
        v1[idx] = vec3(corridorStartX + corridorWidth, corridorHeight, corridorStartZ); // Top-right
        v2[idx] = vec3(corridorStartX, corridorHeight, corridorStartZ);              // Top-left
        albedo[idx] = vec3(0.1f, 0.1f, 0.1f); // Dark grey
        fuzz[idx] = 0.05f;
        mat_type[idx] = 2;
        idx++;



        // Front wall at the end of the corridor (glowing)
        create_wall(
            vec3(corridorStartX, 0.0f, corridorStartZ + corridorLength),                
            vec3(corridorStartX + corridorWidth, 0.0f, corridorStartZ + corridorLength), 
            vec3(corridorStartX + corridorWidth, corridorHeight, corridorStartZ + corridorLength), 
            vec3(corridorStartX, corridorHeight, corridorStartZ + corridorLength),    
            vec3(1.7f, 1.7f, 1.7f), idx, 5);

        list_size = idx;
  

    };

    __device__ bool hitall(const ray& r, float t_min, float t_max, hit_record& rec) const;
    __device__ bool hit(int idx, const ray& r, float tmin, float tmax, hit_record& rec) const;

    // this example use 20KB (max kernel const parameter 32KB)  

    vec3 v0[50], v1[50], v2[50];
    vec3 albedo[50];
    float fuzz[50];
    float ref_idx[50];
    int mat_type[50];
    int list_size;
    int pillar_count; 

    vec3 predefinedColors[5] = {
        vec3(1.6, 1.0, 2.0),  // Bright pink
        vec3(1.0, 1.6, 1.2),  // Bright mint green
        vec3(1.0, 1.0, 1.8),  // Bright sky blue
        vec3(1.8, 1.4, 1.0),  // Bright golden yellow
        vec3(1.4, 1.0, 1.6)   // purple
    };

    void updateCeilingLight() {     

        currentColorIndex = (currentColorIndex + 1) % 5;

        albedo[18] = predefinedColors[currentColorIndex];
        albedo[19] = predefinedColors[currentColorIndex];

    }

    int getCurrentCeilingColorIndex() const {
        return currentColorIndex;
    }

    void changePillarColor() {

        for (int i = 10; i <= 17; i++) {
            albedo[i] = vec3(0.9, 0.2, 0.4);
            fuzz[i] = 0.05;
        }
    }

    private:
        
        int currentColorIndex = 0;

        void create_wall(const vec3& v0a, const vec3& v1a, const vec3& v2a, const vec3& v3a, vec3& wallAlbedo, int& idx, int material_type = 1) {
            v0[idx] = v0a;
            v1[idx] = v2a;
            v2[idx] = v1a;
            albedo[idx] = wallAlbedo;
            mat_type[idx] = material_type;
            idx++;

            v0[idx] = v0a;
            v1[idx] = v3a;
            v2[idx] = v2a;
            albedo[idx] = wallAlbedo;
            mat_type[idx] = material_type;
            idx++;
        }

};

__device__ bool triangle::hitall(const ray& r, float t_min, float t_max, hit_record& rec) const {

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

__device__  bool triangle::hit(int idx, const ray& r, float t_min, float t_max, hit_record& rec) const {
    const float EPSILON = 1e-8;
    vec3 edge1 = v1[idx] - v0[idx];
    vec3 edge2 = v2[idx] - v0[idx];
    vec3 h = cross(r.direction(), edge2);
    float a = dot(edge1, h);
    if (a > -EPSILON && a < EPSILON)
        return false;    // This ray is parallel to this triangle.

    float f = 1.0 / a;
    vec3 s = r.origin() - v0[idx];
    float u = f * dot(s, h);
    if (u < 0.0 || u > 1.0)
        return false;

    vec3 q = cross(s, edge1);
    float v = f * dot(r.direction(), q);
    if (v < 0.0 || u + v > 1.0)
        return false;

    // Calculate t to see where the intersection point is on the line.
    float t = f * dot(edge2, q);
    if (t > t_min && t < t_max) { // ray intersection
        rec.t = t;
        rec.p = r.point_at_parameter(t);

        rec.normal = cross(edge1, edge2).normalize(); // Compute normal

        rec.albedo = albedo[idx];
        rec.fuzz = fuzz[idx];
        rec.ref_idx = ref_idx[idx];
        rec.mat_type = mat_type[idx];
        return true;
    }
    return false;
}






#endif
