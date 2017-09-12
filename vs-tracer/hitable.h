#ifndef HITABLE_H_
#define HITABLE_H_

#include "ray.h"

class material;

struct hit_record {
	float t;
	vec3 p;
	vec3 normal;
	const material *mat_ptr;
};

class hitable {
	public:
		virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};




#endif /* HITABLE_H_ */
