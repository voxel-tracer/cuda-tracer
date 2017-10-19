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

inline bool scatter_lambertian(const material* mat, const ray& ray_in, const hit_record& rec, vec3& alb, ray& scattered, float& pdf)
{
	onb uvw;
	uvw.build_from_w(rec.normal);
	vec3 direction = uvw.local(random_cosine_direction());
	scattered = ray(rec.p, unit_vector(direction));
	alb = mat->albedo;
	pdf = dot(uvw.w(), scattered.direction()) / M_PI;

	//vec3 direction;
	//do {
	//	direction = random_in_unit_sphere();
	//} while (dot(direction, rec.normal) < 0);
	//scattered = ray(rec.p, unit_vector(direction));
	//alb = mat->albedo;
	//pdf = 0.5 / M_PI;

	return pdf > 0;
}

inline bool scatter_metal(const material* mat, const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered)
{
	vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
	scattered = ray(rec.p, reflected + mat->param*random_in_unit_sphere());
	attenuation = mat->albedo;
	return (dot(scattered.direction(), rec.normal) > 0);
}

inline bool scatter_dielectric(const material* mat, const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) {
	vec3 outward_normal;
	vec3 reflected = reflect(r_in.direction(), rec.normal);
	float ni_over_nt;
	attenuation = vec3(1, 1, 1);
	vec3 refracted;
	float reflect_probe;
	float cosine;
	if (dot(r_in.direction(), rec.normal) > 0) {
		outward_normal = -rec.normal;
		ni_over_nt = mat->param;
		cosine = mat->param * dot(r_in.direction(), rec.normal) / r_in.direction().length();
	}
	else {
		outward_normal = rec.normal;
		ni_over_nt = 1.0 / mat->param;
		cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
	}
	if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted)) {
		reflect_probe = schlick(cosine, mat->param);
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

bool material::scatter(const ray& ray_in, const hit_record& rec, vec3& attenuation, ray& scattered, float& pdf) const
{
	switch (type)
	{
	case LAMBERTIAN:
		return scatter_lambertian(this, ray_in, rec, attenuation, scattered, pdf);
	case METAL:
		return scatter_metal(this, ray_in, rec, attenuation, scattered);
	case DIELECTRIC:
		return scatter_dielectric(this, ray_in, rec, attenuation, scattered);
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
