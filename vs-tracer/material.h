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
	float param;
};

material* make_lambertian(const vec3& a) 
{
	material *mat = new material();
	mat->type = LAMBERTIAN;
	mat->albedo = a;
	return mat;
}

material* make_metal(const vec3& albedo, float fuzz)
{
	material *mat = new material();
	mat->type = METAL;
	mat->albedo = albedo;
	mat->param = fuzz;
	return mat;
}

material* make_dielectric(float ref_idx)
{
	material *mat = new material();
	mat->type = DIELECTRIC;
	mat->param = ref_idx;
	return mat;
}

bool scatter_lambertian(const material& mat, const ray& ray_in, const hit_record& rec, vec3& attenuation, ray& scattered)
{
	vec3 target = rec.p + rec.normal + random_in_unit_sphere();
	scattered = ray(rec.p, target - rec.p);
	attenuation = mat.albedo;
	return true;
}

bool scatter_metal(const material& mat, const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered)
{
	vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
	scattered = ray(rec.p, reflected + mat.param*random_in_unit_sphere());
	attenuation = mat.albedo;
	return (dot(scattered.direction(), rec.normal) > 0);
}

bool scatter_dielectric(const material& mat, const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) {
	vec3 outward_normal;
	vec3 reflected = reflect(r_in.direction(), rec.normal);
	float ni_over_nt;
	attenuation = vec3(1, 1, 1);
	vec3 refracted;
	float reflect_probe;
	float cosine;
	if (dot(r_in.direction(), rec.normal) > 0) {
		outward_normal = -rec.normal;
		ni_over_nt = mat.param;
		cosine = mat.param * dot(r_in.direction(), rec.normal) / r_in.direction().length();
	}
	else {
		outward_normal = rec.normal;
		ni_over_nt = 1.0 / mat.param;
		cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
	}
	if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted)) {
		reflect_probe = schlick(cosine, mat.param);
	}
	else {
		reflect_probe = 1.0;
	}
	if (drand48() < reflect_probe) {
		scattered = ray(rec.p, reflected);
	}
	else {
		scattered = ray(rec.p, refracted);
	}
	return true;
}

bool scatter(const material& mat, const ray& ray_in, const hit_record& rec, vec3& attenuation, ray& scattered)
{
	switch (mat.type)
	{
	case LAMBERTIAN:
		return scatter_lambertian(mat, ray_in, rec, attenuation, scattered);
	case METAL:
		return scatter_metal(mat, ray_in, rec, attenuation, scattered);
	case DIELECTRIC:
		return scatter_dielectric(mat, ray_in, rec, attenuation, scattered);
	default:
		// should never happen
		return false;
	}
}

#endif /* MATERIAL_H_ */
