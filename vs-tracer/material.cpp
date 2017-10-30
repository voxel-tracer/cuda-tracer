#include "material.h"
#include "onb.h"
#include "utils.h"


material* make_lambertian(const float3& a)
{
	material *mat = new material();
	mat->type = LAMBERTIAN;
	mat->albedo = a;
	return mat;
}

material* make_metal(const float3& albedo, float fuzz)
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

material* make_diffuse_light(const float3& e)
{
	material *mat = new material();
	mat->type = DIFFUSE_LIGHT;
	mat->_emitted = e;
	return mat;
}

inline bool scatter_lambertian(const material* mat, const ray& ray_in, const hit_record& hrec, const sphere* light_shape, scatter_record& srec)
{
	srec.is_specular = false;
	hitable_pdf plight(light_shape, hrec.p);
	srec.pdf_ptr = new cosine_density(hrec.normal);
	mixture_pdf p(&plight, srec.pdf_ptr);
	srec.scattered = ray(hrec.p, p.generate());
	float pdf_val = p.value(srec.scattered.direction);
	float scattering_pdf = mat->scattering_pdf(ray_in, hrec, srec.scattered);
	srec.attenuation = mat->albedo*scattering_pdf / pdf_val;
	return pdf_val > 0;
}

inline bool scatter_metal(const material* mat, const ray& r_in, const hit_record& hrec, scatter_record& srec)
{
	float3 reflected = reflect(normalize(r_in.direction), hrec.normal);
	srec.scattered = ray(hrec.p, reflected + mat->param*random_to_sphere());
	srec.attenuation = mat->albedo;
	srec.is_specular = true;
	srec.pdf_ptr = NULL;
	return true;
}

inline bool scatter_dielectric(const material* mat, const ray& r_in, const hit_record& hrec, scatter_record& srec) {
	float3 outward_normal;
	float3 reflected = reflect(r_in.direction, hrec.normal);
	float ni_over_nt;
	srec.attenuation = make_float3(1, 1, 1);
	float3 refracted;
	float reflect_probe;
	float cosine;

	srec.is_specular = true;
	if (dot(r_in.direction, hrec.normal) > 0) {
		outward_normal = -1*hrec.normal;
		ni_over_nt = mat->param;
		cosine = mat->param * dot(r_in.direction, hrec.normal) / length(r_in.direction);
	}
	else {
		outward_normal = hrec.normal;
		ni_over_nt = 1.0 / mat->param;
		cosine = -dot(r_in.direction, hrec.normal) / length(r_in.direction);
	}
	if (refract(r_in.direction, outward_normal, ni_over_nt, refracted)) {
		reflect_probe = schlick(cosine, mat->param);
	}
	else {
		reflect_probe = 1.0;
	}
	if (drand48() < reflect_probe) {
		srec.scattered = ray(hrec.p, reflected);
	}
	else {
		srec.scattered = ray(hrec.p, refracted);
	}
	return true;
}

bool material::scatter(const ray& ray_in, const hit_record& rec, const sphere* light_shape, scatter_record& srec) const
{
	switch (type)
	{
	case LAMBERTIAN:
		return scatter_lambertian(this, ray_in, rec, light_shape, srec);
	case METAL:
		return scatter_metal(this, ray_in, rec, srec);
	case DIELECTRIC:
		return scatter_dielectric(this, ray_in, rec, srec);
	default:
		// DIFFUSE_LIGHT is an example of a material that doesn't scatter light
		return false;
	}
}

float lambertian_scattering_pdf(const material* mat, const ray& r_in, const hit_record& rec, const ray& scattered) {
	float cosine = dot(rec.normal, normalize(scattered.direction));
	if (cosine < 0) return 0;
	return cosine / M_PI;
}

float material::scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered) const
{
	if (type == LAMBERTIAN)
		return lambertian_scattering_pdf(this, r_in, rec, scattered);
	return 0;
}

float3 material::emitted(const ray& r_in, const hit_record& rec, const float3& p) const {
	if (type == DIFFUSE_LIGHT) {
		if (dot(rec.normal, r_in.direction) < 0.0)
			return _emitted;
	}
	return make_float3(0, 0, 0);
}
