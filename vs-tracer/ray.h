#ifndef RAY_H_
#define RAY_H_

#include "utils.h"

class ray {
public:
	ray() {}
	ray(const float3& a, const float3& b) { A = a; B = b; }
	ray(const ray& r) { A = r.A; B = r.B; }
	const float3& origin() const { return A; }
	const float3& direction() const { return B; }
	float3 point_at_parameter(float t) const { return A + t*B; }

	float3 A;
	float3 B;
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
	float3 color;
	float3 not_absorbed;

	sample() {}
	sample(sample& s) { pixelId = s.pixelId; depth = s.depth; color = s.color; not_absorbed = s.not_absorbed; }
};

#endif /* RAY_H_ */
