#ifndef RAY_H_
#define RAY_H_

#include "utils.h"

struct ray {
	float3 origin;
	float3 direction;
	unsigned int depth;

	ray() {}
	ray(const float3& o, const float3& d) { origin = o; direction = d; }
	ray(ray& r) { origin = r.origin; direction = r.direction; }

	float3 point_at_parameter(float t) const { return origin + t*direction; }
};

struct sample {
	unsigned int pixelId;
	float3 color;
	float3 not_absorbed;

	sample() {}
	sample(int pId): pixelId(pId) {
		color = make_float3(0, 0, 0);
		not_absorbed = make_float3(1, 1, 1);
	}
	sample(const sample& s) { pixelId = s.pixelId; color = s.color; not_absorbed = s.not_absorbed; }
};

struct clr_rec {
	float3 color;
	float3 origin;
	float3 direction;
	bool done;

	clr_rec() {
		color = make_float3(0, 0, 0);
		origin = make_float3(0, 0, 0);
		direction = make_float3(0, 0, 0);
	}
};

#endif /* RAY_H_ */
