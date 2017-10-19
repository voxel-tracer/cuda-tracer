#ifndef UTILS_H_
#define UTILS_H_

#include <cstdlib>
#include "vec3.h"

// Used to seed the generator.           
inline void fast_srand(int seed);

// Compute a pseudorandom integer.
// Output value in range [0, 32767]
inline float drand48(void);
vec3 random_in_unit_sphere();
vec3 random_cosine_direction();
vec3 random_to_sphere();
vec3 reflect(const vec3& v, const vec3& n);
bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted);
float schlick(float cosine, float ref_idx);
vec3 random_in_unit_disk();

#endif /* UTILS_H_ */
