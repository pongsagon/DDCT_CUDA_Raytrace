#include "_Def1.h"



glm::vec3 CentralGravity = glm::vec3(0, 10, 0);




float rand01() {
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

class spheres {
public:

    int B_Blackhole;
    int B_BlackholeHorizon;
    int B_CenterStarGel;
    int B_CenterStar;

    spheres()
    {
        int idx = 0;


        //center[idx] = vec3(0,0,0);
        //radius[idx] = 10000;
        //albedo[idx] = vec3(0, 0, 0);
        //mat_type[idx] = Mat_Papermint;
        //idx++;


        int Amount = 3;
        float Spread = 1.75f;
        for (int a = -Amount; a < Amount; a++) {
            for (int b = -Amount; b < Amount; b++) {
                float choose_mat = rand01();

                float RandRange = 10;
                float RandSpread = 5;
                vec3 _center;
                _center[0] = B_irand(-RandRange, RandRange);
                _center[1] = B_irand(-RandRange, RandRange);
                _center[2] = B_irand(-RandRange, RandRange);
                glm::vec3 _Vel = glm::vec3(B_irand(-5, 5), B_irand(-5, 5), B_irand(-5, 5));
                float _Rad = B_frand(0.16, 0.3);

                if (choose_mat < 0.64f) {
                    center[idx] = _center;
                    radius[idx] = _Rad;
                    albedo[idx] = vec3(1, rand01() * rand01() * 0.8f + 0.02f, 0.02f);
                    mat_type[idx] = Mat_Solid;//1
                    Velocity[idx] = _Vel;
                    idx++;
                }
                else if (choose_mat < 0.8f) {
                    center[idx] = _center;
                    radius[idx] = _Rad;
                    albedo[idx] = vec3(0.05, 0.05, 0.05);
                    fuzz[idx] = B_frand(0.32, 0.64);
                    mat_type[idx] = Mat_Metal;//2
                    Velocity[idx] = _Vel;
                    idx++;
                }
                else {
                    center[idx] = _center;
                    radius[idx] = _Rad * 0.72;
                    ref_idx[idx] = 1.5;
                    albedo[idx] = vec3(99, 99, 99);
                    mat_type[idx] = Mat_Emissive;
                    Velocity[idx] = _Vel;
                    idx++;
                }
            }
        }

        float CoreScale = 0.64;

        B_Blackhole = idx;
        center[idx] = vec3(CentralGravity.x, CentralGravity.y, CentralGravity.z + 4);
        radius[idx] = 0.45 * CoreScale;
        albedo[idx] = vec3(0, 0, 0);
        mat_type[idx] = Mat_Solid;
        idx++;

        B_BlackholeHorizon = idx;
        center[idx] = lookfrom;
        radius[idx] = 0.64;
        albedo[idx] = vec3(0, 0, 0);
        fuzz[idx] = 0.5 * CoreScale;
        ref_idx[idx] = -1;
        mat_type[idx] = Mat_Mirror;
        idx++;

        //B_CenterStarGel = idx;
        //center[idx] = vec3(CentralGravity.x, CentralGravity.y, CentralGravity.z);
        //radius[idx] = 6;
        //albedo[idx] = vec3(0.2, 0.2, 0.2);
        //fuzz[idx] = 0.5;
        //ref_idx[idx] = 1.64;
        //mat_type[idx] = Mat_Mirror;
        //idx++;



        B_CenterStar = idx;
        center[idx] = vec3(CentralGravity.x, CentralGravity.y, CentralGravity.z);
        radius[idx] = 4;
        albedo[idx] = vec3(0.2, 0.2, 0.2);
        fuzz[idx] = 0.0;
        ref_idx[idx] = 0;
        mat_type[idx] = Mat_ScatterDot;
        idx++;



        list_size = idx;
    };

    __device__ bool hitall(const ray& r, float t_min, float t_max, hit_record& rec) const;
    __device__ bool hit(int idx, const ray& r, float tmin, float tmax, hit_record& rec) const;

    float time = 0;
    void Update() {

        if (Time.Deltatime > 0.5) { return; }

        glm::vec3 BlackholeCenter;
        BlackholeCenter.x = center[B_Blackhole].x();
        BlackholeCenter.y = center[B_Blackhole].y();
        BlackholeCenter.z = center[B_Blackhole].z();

        vec3& BlackholeColor = albedo[B_Blackhole];


        glm::vec3 EachCenter_Temp;
        float movementSpeed = 8; // Adjust this value to control the speed of movement
        for (int i = 1; i < list_size; i++) {

            if (i == B_Blackhole) { continue; }
            if (i == B_CenterStar) { continue; }
            if (i == B_BlackholeHorizon) { continue; }
            if (i == B_CenterStarGel) { continue; }

            EachCenter_Temp.x = center[i].x();
            EachCenter_Temp.y = center[i].y();
            EachCenter_Temp.z = center[i].z();

            float DistanceToBlack = B_distance3D(
                BlackholeCenter.x, BlackholeCenter.y, BlackholeCenter.z
                , EachCenter_Temp.x, EachCenter_Temp.y, EachCenter_Temp.z);

            if (DistanceToBlack < 5) {


                glm::vec3 direction = glm::normalize(BlackholeCenter - EachCenter_Temp);
                Velocity[i] += direction * movementSpeed * Time.Deltatime;

                if (DistanceToBlack < 1.1f) {

                    float LerpSpeed = Time.Deltatime * 1.5f;
                    EachCenter_Temp = B_lerpVec3(EachCenter_Temp, BlackholeCenter, LerpSpeed);

                    if (DistanceToBlack < 0.25) {
                        float RandRange = 8;
                        float RandSpread = 3;
                        EachCenter_Temp = glm::vec3(B_irand(-RandRange, RandRange), B_irand(-RandRange, RandRange), B_irand(-RandRange, RandRange)) * RandSpread;
                        Velocity[0] = glm::vec3(0);

                        BlackholeColor = albedo[i] * 50;
                    }
                }

            }
            else
            {
                //Velocity[i] =B_lerpVec3(Velocity[i] , glm::vec3(0),Time.Deltatime*0.5f);

                //float angle_increment = 0.1f;  // Controls the speed of rotation
                //float radius = 1.0f;           // Radius of the circular motion
                //time += Time.Deltatime*0.05;

                //float theta = time + i * (2 * 3.14f / list_size);

                //EachCenter_Temp.x = (radius * cos(theta));
                //EachCenter_Temp.y = (radius * sin(theta));

                glm::vec3 EachCenter_Temp2;
                for (int j = 1; j < list_size; j++) {

                    if (j == B_Blackhole) { continue; }
                    if (j == B_CenterStar) { continue; }
                    if (j == B_BlackholeHorizon) { continue; }
                    if (j == B_CenterStarGel) { continue; }
                    if (j == i) { continue; }

                    EachCenter_Temp2.x = center[j].x();
                    EachCenter_Temp2.y = center[j].y();
                    EachCenter_Temp2.z = center[j].z();

                    float DistanceToOther = B_distance3D(
                        EachCenter_Temp.x, EachCenter_Temp.y, EachCenter_Temp.z
                        , EachCenter_Temp2.x, EachCenter_Temp2.y, EachCenter_Temp2.z);
                    if (DistanceToOther < 16) {
                        glm::vec3 direction = glm::normalize(EachCenter_Temp - EachCenter_Temp2);
                        Velocity[j] += direction * Time.Deltatime * 0.1f;
                    }

                }

                glm::vec3 direction = glm::normalize(CentralGravity - EachCenter_Temp);
                Velocity[i] += direction * Time.Deltatime * 0.2f;


            }



            float MaxSpeed = 8;
            Velocity[i][0] = B_clamp(Velocity[i][0], -MaxSpeed, MaxSpeed);
            Velocity[i][1] = B_clamp(Velocity[i][1], -MaxSpeed, MaxSpeed);
            Velocity[i][2] = B_clamp(Velocity[i][2], -MaxSpeed, MaxSpeed);
            EachCenter_Temp += Velocity[i] * Time.Deltatime;

            EachCenter_Temp.y = B_clamp(EachCenter_Temp.y, 0, 100);
            center[i][0] = EachCenter_Temp.x;
            center[i][1] = EachCenter_Temp.y;
            center[i][2] = EachCenter_Temp.z;
        }



    }

    // this example use 20KB (max kernel const parameter 32KB)
    vec3 center[500];
    glm::vec3 Velocity[500];
    float radius[500];
    vec3 albedo[500];
    float fuzz[500];
    float ref_idx[500];
    int mat_type[500];
    int list_size;
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























class triangles {
public:
    triangles()
    {
        // Example triangle initialization (add more as needed)
        int idx = 0;

        vertices[idx][0] = vec3(-250000, 0, -500000);
        vertices[idx][1] = vec3(500000, 0, 0);
        vertices[idx][2] = vec3(-250000, 0, 500000);
        normals[idx] = calculate_normal(vertices[idx][0], vertices[idx][1], vertices[idx][2]);
        albedo[idx] = vec3(0.5, 0.5, 0.5);
        mat_type[idx] = Mat_Mirror_Dark;
        idx++;


        vertices[idx][0] = vec3(-250000, -0.1, -500000);
        vertices[idx][1] = vec3(500000, -0.1, 0);
        vertices[idx][2] = vec3(-250000, -0.1, 500000);
        normals[idx] = calculate_normal(vertices[idx][0], vertices[idx][1], vertices[idx][2]);
        albedo[idx] = vec3(1, 0, 0);
        mat_type[idx] = Mat_Emissive;
        idx++;



        //float TrisSize = 100;
        //float Gap = 0.032;
        //float Redindex = 0.016;

        float TrisSize = 100;
        float Gap = 0.025;
        float Redindex = 0.32;

        float TriWidth = 40;
        float TriHeight = 60;
        for (int i = 0; i < 1; i++) {


            vertices[idx][0] = vec3(TriWidth, 0, TriWidth + Gap) * TrisSize;
            vertices[idx][1] = vec3(0, TriHeight, Gap) * TrisSize;
            vertices[idx][2] = vec3(-TriWidth, 0, TriWidth + Gap) * TrisSize;
            normals[idx] = calculate_normal(vertices[idx][0], vertices[idx][1], vertices[idx][2]);
            albedo[idx] = vec3(0.5, 0.5, 0.5);
            mat_type[idx] = Mat_Mirror;
            ref_idx[idx] = Redindex;
            idx++;

            vertices[idx][0] = vec3(TriWidth, 0, -TriWidth - Gap) * TrisSize;
            vertices[idx][1] = vec3(0, TriHeight, -Gap) * TrisSize;
            vertices[idx][2] = vec3(-TriWidth, 0, -TriWidth - Gap) * TrisSize;
            normals[idx] = calculate_normal(vertices[idx][0], vertices[idx][1], vertices[idx][2]);
            albedo[idx] = vec3(0.5, 0.5, 0.5);
            mat_type[idx] = Mat_Mirror;
            ref_idx[idx] = Redindex;
            idx++;

            vertices[idx][0] = vec3(-Gap - TriWidth, 0, -TriWidth) * TrisSize;
            vertices[idx][1] = vec3(-Gap, TriHeight, 0) * TrisSize;
            vertices[idx][2] = vec3(-Gap - TriWidth, 0, TriWidth) * TrisSize;
            normals[idx] = calculate_normal(vertices[idx][2], vertices[idx][1], vertices[idx][0]);
            albedo[idx] = vec3(0.5, 0.5, 0.5);
            mat_type[idx] = Mat_Mirror;
            ref_idx[idx] = Redindex;
            idx++;

            vertices[idx][0] = vec3(Gap + TriWidth, 0, -TriWidth) * TrisSize;
            vertices[idx][1] = vec3(Gap, TriHeight, 0) * TrisSize;
            vertices[idx][2] = vec3(Gap + TriWidth, 0, TriWidth) * TrisSize;
            normals[idx] = calculate_normal(vertices[idx][2], vertices[idx][1], vertices[idx][0]);
            albedo[idx] = vec3(0.5, 0.5, 0.5);
            mat_type[idx] = Mat_Mirror;
            ref_idx[idx] = Redindex;
            idx++;

             TriWidth *= 0.5;
             TriHeight *= 2;
        }



        list_size = idx;
    };

    __device__ bool hitall(const ray& r, float t_min, float t_max, hit_record& rec) const;
    __device__ bool hit(int idx, const ray& r, float tmin, float tmax, hit_record& rec) const;

private:
    // Calculate the normal vector of a triangle
    vec3 calculate_normal(const vec3& v0, const vec3& v1, const vec3& v2) const {
        return unit_vector(cross(v1 - v0, v2 - v0));
    }

    vec3 vertices[30][3]; // Stores vertices for each triangle
    vec3 normals[30];     // Stores normals for each triangle
    vec3 albedo[30];
    float fuzz[30];
    float ref_idx[30];
    int mat_type[30];
    int list_size;
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
    // Möller–Trumbore ray-triangle intersection algorithm
    vec3 v0v1 = vertices[idx][1] - vertices[idx][0];
    vec3 v0v2 = vertices[idx][2] - vertices[idx][0];
    vec3 pvec = cross(r.direction(), v0v2);
    float det = dot(v0v1, pvec);

    // If the determinant is near zero, the ray lies in the plane of the triangle
    if (fabs(det) < 1e-8) return false;
    float invDet = 1.0 / det;

    vec3 tvec = r.origin() - vertices[idx][0];
    float u = dot(tvec, pvec) * invDet;
    if (u < 0 || u > 1) return false;

    vec3 qvec = cross(tvec, v0v1);
    float v = dot(r.direction(), qvec) * invDet;
    if (v < 0 || u + v > 1) return false;

    float t = dot(v0v2, qvec) * invDet;
    if (t < t_min || t > t_max) return false;

    rec.t = t;
    rec.p = r.point_at_parameter(t);
    rec.normal = normals[idx]; // Use precomputed normal
    rec.albedo = albedo[idx];
    rec.fuzz = fuzz[idx];
    rec.ref_idx = ref_idx[idx];
    rec.mat_type = mat_type[idx];

    return true;
}
