#ifndef RAY_H_
#define RAY_H_

#include "vec3.h"

class ray {
public:
	ray() {}
	ray(const vec3& a, const vec3& b) { A = a; B = b; }
	ray(const ray& r) { A = r.A; B = r.B; }
	const vec3& origin() const { return A; }
	const vec3& direction() const { return B; }
	vec3 point_at_parameter(float t) const { return A + t*B; }

	vec3 A;
	vec3 B;
};

struct cu_ray {
	unsigned int pixelId;
	float3 origin;
	float3 direction;

	cu_ray() {}
	cu_ray(cu_ray& r) { pixelId = r.pixelId; origin = r.origin; direction = r.direction; }
};

struct sample {
	unsigned int pixelId;
	unsigned int depth;
	vec3 color;
	vec3 not_absorbed;

	sample() {}
	sample(sample& s) { pixelId = s.pixelId; depth = s.depth; color = s.color; not_absorbed = s.not_absorbed; }
};

#endif /* RAY_H_ */
