#ifndef MATERIAL_H_
#define MATERIAL_H_

#include <vector_functions.hpp>
#include "ray.h"
#include "pdf.h"
#include "hitable.h"
#include "utils.h"

enum materialType 
{
	LAMBERTIAN,
	METAL,
	DIELECTRIC,
	DIFFUSE_LIGHT
};

struct scatter_record
{
	ray specular_ray;
	bool is_specular;
	vec3 attenuation;
	pdf *pdf_ptr;
};

struct material 
{
	materialType type;
	vec3 albedo;
	vec3 _emitted;
	float param;
	bool scatter(const ray& ray_in, const hit_record& rec, scatter_record& srec) const;
	float scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered) const;
	vec3 emitted(const ray& r_in, const hit_record& rec, const vec3 &p) const;
};

material* make_lambertian(const vec3& a);
material* make_metal(const vec3& albedo, float fuzz);
material* make_dielectric(float ref_idx);
material* make_diffuse_light(const vec3& e);

#endif /* MATERIAL_H_ */
