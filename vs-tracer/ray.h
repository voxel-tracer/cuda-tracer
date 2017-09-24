#ifndef RAY_H_
#define RAY_H_

#include "vec3.h"

class ray {
public:
	ray() {}
	ray(const vec3& a, const vec3& b) { A = a; B = b;}
	vec3 origin() const { return A; }
	vec3 direction() const { return B; }
	vec3 point_at_parameter(float t) const { return A + t*B; }

	vec3 A;
	vec3 B;
};

struct cu_ray {
	float3 origin;
	float3 direction;
	unsigned int pixelId;
	unsigned int depth;

	cu_ray() {}
	cu_ray(cu_ray& r) { origin = r.origin; direction = r.direction; pixelId = r.pixelId; depth = r.depth; }
};

#endif /* RAY_H_ */
