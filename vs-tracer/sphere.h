#ifndef SPHERE_H_
#define SPHERE_H_

#include "hitable.h"
#include "material.h"

class sphere: public hitable {
	public:
		sphere(vec3 cen, float r, const material *mat) { center = cen; radius = r; mat_ptr = mat; }
		virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;

		vec3 center;
		float radius;
		const material *mat_ptr;
};

bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	vec3 oc = r.origin() - center;
	float a = r.direction().squared_length();
	float b = 2.0 * dot(oc, r.direction());
	float c = oc.squared_length() - radius*radius;
	float discriminant = b*b - 4*a*c;
	if (discriminant > 0) {
		float temp = (-b - sqrtf(discriminant)) / (2.0*a);
		if (temp < t_max && temp > t_min) {
			rec.t = temp;
			rec.p = r.point_at_parameter(temp);
			rec.normal = (rec.p - center) / radius;
			rec.mat_ptr = mat_ptr;
			return true;
		}
	}
	return false;
}

struct cu_sphere {
	float3 center;
	float radius;
};

#endif /* SPHERE_H_ */
