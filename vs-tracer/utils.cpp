#include "utils.h"

static unsigned int g_seed;

// Used to seed the generator.           
inline void fast_srand(int seed) {
	g_seed = seed;
}

// Compute a pseudorandom integer.
// Output value in range [0, 32767]
inline float drand48(void) {
	g_seed = (214013 * g_seed + 2531011);
	return (float)((g_seed >> 16) & 0x7FFF) / 32767;
}

vec3 random_in_unit_sphere() {
	vec3 p;
	do {
		p = vec3(2 * drand48() - 1, 2 * drand48() - 1, 2 * drand48() - 1);
	} while (p.squared_length() >= 1.0);
	return p;
	//vec3 p = vec3(2 * drand48() - 1, 2 * drand48() - 1, 2 * drand48() - 1);
	//if (p.squared_length() > 1.0)
	//	return unit_vector(p);
	//return p;
}

vec3 random_to_sphere() {
	vec3 p;
	do {
		p = 2.0*vec3(drand48(), drand48(), drand48()) - vec3(1, 1, 1);
	} while (dot(p, p) >= 1.0);
	return unit_vector(p);
}

vec3 random_cosine_direction() {
	float r1 = drand48();
	float r2 = drand48();
	float z = sqrtf(1 - r2);
	float phi = 2 * M_PI*r1;
	float x = cosf(phi) * 2 * sqrtf(r2);
	float y = sinf(phi) * 2 * sqrtf(r2);
	return vec3(x, y, z);
}

vec3 reflect(const vec3& v, const vec3& n) {
	return v - 2 * dot(v, n)*n;
}

bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
	vec3 uv = unit_vector(v);
	float dt = dot(uv, n);
	float discriminant = 1.0 - ni_over_nt*ni_over_nt*(1 - dt*dt);
	if (discriminant > 0) {
		refracted = ni_over_nt*(uv - n*dt) - n*sqrtf(discriminant);
		return true;
	}
	return false;
}

float schlick(float cosine, float ref_idx) {
	float r0 = (1 - ref_idx) / (1 + ref_idx);
	r0 = r0*r0;
	return r0 + (1 - r0)*pow((1 - cosine), 5);
}

vec3 random_in_unit_disk() {
	vec3 p;
	do {
		p = 2.0*vec3(drand48(), drand48(), 0) - vec3(1, 1, 0);
	} while (dot(p, p) >= 1.0);
	return p;
}
