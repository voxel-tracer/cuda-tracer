#ifndef RAY_H_
#define RAY_H_

#include "utils.h"

struct ray {
	float3 origin;
	float3 direction;

	__host__ __device__ ray() {}
	__host__ __device__ ray(const float3& o, const float3& d) :origin(o), direction(d) {}
	__host__ __device__ ray(ray& r):origin(r.origin), direction(r.direction) {}

	__host__ __device__ float3 point_at_parameter(float t) const { return origin + t*direction; }
};

struct sample {
	uint pixelId;
	float3 not_absorbed;
	uint depth = 0;
	bool done = false;

	sample() {}
	sample(int pId): pixelId(pId) {
		not_absorbed = make_float3(1, 1, 1);
	}
	sample(const sample& s) :pixelId(s.pixelId), not_absorbed(s.not_absorbed), depth(s.depth), done(s.done) {}
};

struct clr_rec {
	float3 color;
	float3 origin;
	float3 direction;
	bool done = false;

	clr_rec() {
		color = make_float3(0, 0, 0);
		origin = make_float3(0, 0, 0);
		direction = make_float3(0, 0, 0);
	}
};

#endif /* RAY_H_ */
