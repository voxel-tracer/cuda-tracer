#include "material.h"
#include "onb.h"


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

material* make_diffuse_light(const vec3& e)
{
	material *mat = new material();
	mat->type = DIFFUSE_LIGHT;
	mat->_emitted = e;
	return mat;
}

inline bool scatter_lambertian(const material* mat, const ray& ray_in, const hit_record& hrec, scatter_record& srec)
{
	srec.is_specular = false;
	srec.attenuation = mat->albedo;
	srec.pdf_ptr = new cosine_density(hrec.normal);
	return true;
}

inline bool scatter_metal(const material* mat, const ray& r_in, const hit_record& hrec, scatter_record& srec)
{
	vec3 reflected = reflect(unit_vector(r_in.direction()), hrec.normal);
	srec.specular_ray = ray(hrec.p, reflected + mat->param*random_in_unit_sphere());
	srec.attenuation = mat->albedo;
	srec.is_specular = true;
	srec.pdf_ptr = NULL;
	return true;
}

inline bool scatter_dielectric(const material* mat, const ray& r_in, const hit_record& hrec, scatter_record& srec) {
	vec3 outward_normal;
	vec3 reflected = reflect(r_in.direction(), hrec.normal);
	float ni_over_nt;
	srec.attenuation = vec3(1, 1, 1);
	vec3 refracted;
	float reflect_probe;
	float cosine;

	srec.is_specular = true;
	if (dot(r_in.direction(), hrec.normal) > 0) {
		outward_normal = -hrec.normal;
		ni_over_nt = mat->param;
		cosine = mat->param * dot(r_in.direction(), hrec.normal) / r_in.direction().length();
	}
	else {
		outward_normal = hrec.normal;
		ni_over_nt = 1.0 / mat->param;
		cosine = -dot(r_in.direction(), hrec.normal) / r_in.direction().length();
	}
	if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted)) {
		reflect_probe = schlick(cosine, mat->param);
	}
	else {
		reflect_probe = 1.0;
	}
	if (drand48() < reflect_probe) {
		srec.specular_ray = ray(hrec.p, reflected);
	}
	else {
		srec.specular_ray = ray(hrec.p, refracted);
	}
	return true;
}

bool material::scatter(const ray& ray_in, const hit_record& rec, scatter_record& srec) const
{
	switch (type)
	{
	case LAMBERTIAN:
		return scatter_lambertian(this, ray_in, rec, srec);
	case METAL:
		return scatter_metal(this, ray_in, rec, srec);
	case DIELECTRIC:
		return scatter_dielectric(this, ray_in, rec, srec);
	default:
		// should never happen
		return false;
	}
}

float lambertian_scattering_pdf(const material* mat, const ray& r_in, const hit_record& rec, const ray& scattered) {
	float cosine = dot(rec.normal, unit_vector(scattered.direction()));
	if (cosine < 0) return 0;
	return cosine / M_PI;
}

float material::scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered) const
{
	if (type == LAMBERTIAN)
		return lambertian_scattering_pdf(this, r_in, rec, scattered);
	return 0;
}

vec3 material::emitted(const ray& r_in, const hit_record& rec, const vec3& p) const {
	if (type == DIFFUSE_LIGHT) {
		if (dot(rec.normal, r_in.direction()) < 0.0)
			return _emitted;
	}
	return vec3(0, 0, 0);
}
