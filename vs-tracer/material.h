#ifndef MATERIAL_H_
#define MATERIAL_H_

#include <vector_functions.hpp>
#include "ray.h"
#include "hitable.h"
#include "utils.h"

enum materialType 
{
	LAMBERTIAN,
	METAL,
	DIELECTRIC
};

struct material 
{
	materialType type;
	vec3 albedo;
	vec3 emitted;
	float param;
	bool scatter(const ray& ray_in, const hit_record& rec, vec3& attenuation, ray& scattered) const;
};

material* make_lambertian(const vec3& a);
material* make_metal(const vec3& albedo, float fuzz);
material* make_dielectric(float ref_idx);
material* make_diffuse_light(const vec3& e);

#endif /* MATERIAL_H_ */
