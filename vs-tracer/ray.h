#ifndef RAY_H_
#define RAY_H_

#include "utils.h"

struct ray {
	unsigned int pixelId;
	float3 origin;
	float3 direction;

	ray() {}
	ray(const float3& o, const float3& d) { origin = o; direction = d; }
	ray(ray& r) { pixelId = r.pixelId; origin = r.origin; direction = r.direction; }

	float3 point_at_parameter(float t) const { return origin + t*direction; }
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
