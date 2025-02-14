#pragma once

#include "material.h"
#include "vec3.h"
#include "ray.h"

class hitable {
public:
	virtual bool hit(int idx, const ray& r, float t_min, float t_max, hit_record& rec) const = 0;

	
};