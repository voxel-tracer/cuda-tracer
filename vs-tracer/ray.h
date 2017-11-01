#ifndef RAY_H_
#define RAY_H_

#include "utils.h"

struct ray {
	unsigned int pixelId;
	float3 origin;
	float3 direction;
	unsigned int depth;
	float3 color;
	float3 not_absorbed;

	ray() {}
	ray(const float3& o, const float3& d) { origin = o; direction = d; }
	ray(ray& r) { pixelId = r.pixelId; origin = r.origin; direction = r.direction; }

	float3 point_at_parameter(float t) const { return origin + t*direction; }
};

#endif /* RAY_H_ */
