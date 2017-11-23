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

__device__ bool scatter_lambertian(const material* mat, const float3& r_dir, const hit_record& hrec, const sphere* light_shape, seed_t seed, scatter_record& srec) {
	srec.is_specular = false;
	pdf p = pdf(hrec.normal);
	//if (light_shape != NULL) {
	//	pdf  *plight = make_hitable_pdf(light_shape, hrec.p);
	//	p = make_mixture_pdf(plight, p);
	//}
	srec.scattered = ray(hrec.p, p.generate(seed));
	float pdf_val = p.value(srec.scattered.direction);
	float scattering_pdf = mat->scattering_pdf(r_dir, hrec, srec.scattered);
	srec.attenuation = mat->albedo*scattering_pdf / pdf_val;
	return pdf_val > 0;
}

__device__ inline bool scatter_metal(const material* mat, const float3& r_dir, const hit_record& hrec, seed_t seed, scatter_record& srec)
{
	float3 reflected = reflect(normalize(r_dir), hrec.normal);
	srec.scattered = ray(hrec.p, reflected + mat->param*random_to_sphere(seed));
	srec.attenuation = mat->albedo;
	srec.is_specular = true;
	return true;
}

__device__ inline bool scatter_dielectric(const material* mat, const float3& r_dir, const hit_record& hrec, seed_t seed, scatter_record& srec) {
	float3 outward_normal;
	float3 reflected = reflect(r_dir, hrec.normal);
	float ni_over_nt;
	srec.attenuation = make_float3(1, 1, 1);
	float3 refracted;
	float reflect_probe;
	float cosine;

	srec.is_specular = true;
	if (dot(r_dir, hrec.normal) > 0) {
		outward_normal = -1 * hrec.normal;
		ni_over_nt = mat->param;
		cosine = mat->param * dot(r_dir, hrec.normal) / length(r_dir);
	}
	else {
		outward_normal = hrec.normal;
		ni_over_nt = 1.0f / mat->param;
		cosine = -dot(r_dir, hrec.normal) / length(r_dir);
	}
	if (refract(r_dir, outward_normal, ni_over_nt, refracted)) {
		reflect_probe = schlick(cosine, mat->param);
	}
	else {
		reflect_probe = 1.0;
	}
	if (cu_drand48(seed) < reflect_probe) {
		srec.scattered = ray(hrec.p, reflected);
	}
	else {
		srec.scattered = ray(hrec.p, refracted);
	}
	return true;
}

__device__ bool material::scatter(const float3& r_dir, const hit_record& rec, const sphere* light_shape, seed_t seed, scatter_record& srec) const
{
	switch (type)
	{
	case LAMBERTIAN:
		return scatter_lambertian(this, r_dir, rec, light_shape, seed, srec);
	case METAL:
		return scatter_metal(this, r_dir, rec, seed, srec);
	case DIELECTRIC:
		return scatter_dielectric(this, r_dir, rec, seed, srec);
	default:
		// DIFFUSE_LIGHT is an example of a material that doesn't scatter light
		return false;
	}
}

__device__ float lambertian_scattering_pdf(const material* mat, const hit_record& rec, const ray& scattered) {
	float cosine = dot(rec.normal, normalize(scattered.direction));
	if (cosine < 0) return 0;
	return cosine / M_PI;
}

__device__ float material::scattering_pdf(const float3& r_dir, const hit_record& rec, const ray& scattered) const
{
	if (type == LAMBERTIAN)
		return lambertian_scattering_pdf(this, rec, scattered);
	return 0;
}

float3 material::emitted(const ray& r_in, const hit_record& rec, const float3& p) const {
	if (type == DIFFUSE_LIGHT) {
		if (dot(rec.normal, r_in.direction) < 0.0)
			return _emitted;
	}
	return make_float3(0, 0, 0);
}
