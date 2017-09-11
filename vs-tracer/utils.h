#ifndef UTILS_H_
#define UTILS_H_

#include <cstdlib>

#include "vec3.h"

float drand48()
{
	return (float) rand() / RAND_MAX;
}

vec3 random_in_unit_sphere() {
	vec3 p;
	do {
		p = 2.0*vec3(drand48(), drand48(), drand48()) - vec3(1,1,1);
	} while (p.squared_length() >= 1.0);
	return p;
}

vec3 reflect(const vec3& v, const vec3& n) {
	return v - 2*dot(v, n)*n;
}

bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
	vec3 uv = unit_vector(v);
	float dt = dot(uv, n);
	float discriminant = 1.0 - ni_over_nt*ni_over_nt*(1-dt*dt);
	if (discriminant > 0) {
		refracted = ni_over_nt*(uv - n*dt) - n*sqrtf(discriminant);
		return true;
	}
	return false;
}

float schlick(float cosine, float ref_idx) {
	float r0 = (1-ref_idx) / (1+ref_idx);
	r0 = r0*r0;
	return r0 + (1-r0)*pow((1 - cosine), 5);
}

vec3 random_in_unit_disk() {
	vec3 p;
	do {
		p = 2.0*vec3(drand48(), drand48(), 0) - vec3(1,1,0);
	} while (dot(p,p) >= 1.0);
	return p;
}

#endif /* UTILS_H_ */
