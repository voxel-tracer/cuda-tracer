#include "utils.h"
#include <stdio.h>

static unsigned int g_seed;

// Compute a pseudorandom integer.
// Output value in range [0, 32767]
float drand48(void) {
	g_seed = (214013 * g_seed + 2531011);
	return (float)((g_seed >> 16) & 0x7FFF) / 32767;
}

__device__ float cu_drand48(seed_t seed) {
	float res = curand_uniform(seed);
	//printf("%f\n", res);
	return res;
}

float3 hex2float3(const int hexval) {
	return make_float3(((hexval >> 16) & 0xFF) / 255.0, ((hexval >> 8) & 0xFF) / 255.0, (hexval & 0xFF) / 255.0);
}

__device__ float3 random_to_sphere(seed_t seed) {
	float3 p;
	do {
		p = 2.0*make_float3(cu_drand48(seed), cu_drand48(seed), cu_drand48(seed)) - make_float3(1, 1, 1);
	} while (dot(p, p) >= 1.0);
	return normalize(p);
}

__device__ float3 random_to_sphere(seed_t seed, float radius, float distance_squared) {
	float r1 = cu_drand48(seed);
	float r2 = cu_drand48(seed);
	float z = 1 + r2*(sqrt(1 - radius*radius / distance_squared) - 1);
	float phi = 2 * M_PI*r1;
	float x = cos(phi)*sqrt(1 - z*z);
	float y = sin(phi)*sqrt(1 - z*z);
	return make_float3(x, y, z);
}

__device__ float3 random_cosine_direction(seed_t seed) {
	float r1 = cu_drand48(seed);
	float r2 = cu_drand48(seed);
	float z = sqrtf(1 - r2);
	float phi = 2 * M_PI*r1;
	float x = cosf(phi) * 2 * sqrtf(r2);
	float y = sinf(phi) * 2 * sqrtf(r2);
	return make_float3(x, y, z);
}

__device__ bool refract(const float3& v, const float3& n, float ni_over_nt, float3& refracted) {
	float3 uv = normalize(v);
	float dt = dot(uv, n);
	float discriminant = 1.0 - ni_over_nt*ni_over_nt*(1 - dt*dt);
	if (discriminant > 0) {
		refracted = ni_over_nt*(uv - n*dt) - n*sqrtf(discriminant);
		return true;
	}
	return false;
}

__device__ float schlick(float cosine, float ref_idx) {
	float r0 = (1 - ref_idx) / (1 + ref_idx);
	r0 = r0*r0;
	return r0 + (1 - r0)*pow((1 - cosine), 5);
}

float3 random_in_unit_disk() {
	float3 p;
	do {
		p = 2.0*make_float3(drand48(), drand48(), 0) - make_float3(1, 1, 0);
	} while (dot(p, p) >= 1.0);
	return p;
}

__device__ float squared_length(const float3& v) {
	return dot(v, v);
}
