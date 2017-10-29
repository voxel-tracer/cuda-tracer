#ifndef HITABLE_H_
#define HITABLE_H_

#include "ray.h"

class material;

struct hit_record {
	float t;
	float3 p;
	float3 normal;
	const material *mat_ptr;
};

class hitable {
public:
	virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
	virtual float pdf_value(const float3& o, const float3& v) const { return 0.0; }
	virtual float3 random(const float3& o) const { return make_float3(1, 0, 0); }
	
	bool sphere_dbg;
};




#endif /* HITABLE_H_ */
