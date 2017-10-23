#ifndef SPHERE_H_
#define SPHERE_H_

#include "hitable.h"
#include "material.h"
#include "pdf.h"

class sphere: public hitable {
	public:
		sphere(vec3 cen, float r, const material *mat) { center = cen; radius = r; mat_ptr = mat; }
		virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
			vec3 oc = r.origin() - center;
			float a = dot(r.direction(), r.direction());
			float b = dot(oc, r.direction());
			float c = dot(oc, oc) - radius*radius;
			float discriminant = b*b - a*c;
			//if (sphere_dbg) printf("sphere_hit: a %.6f, b %.6f, c %.6f, d %.6f\n", a, b, c, discriminant);
			if (discriminant > 0) {
				float temp = (-b - sqrt(b*b - a*c)) / a;
				if (temp < t_max && temp > t_min) {
					rec.t = temp;
					rec.p = r.point_at_parameter(rec.t);
					rec.normal = (rec.p - center) / radius;
					rec.mat_ptr = mat_ptr;
					return true;
				}
				temp = (-b + sqrt(b*b - a*c)) / a;
				if (temp < t_max && temp > t_min) {
					rec.t = temp;
					rec.p = r.point_at_parameter(rec.t);
					rec.normal = (rec.p - center) / radius;
					rec.mat_ptr = mat_ptr;
					return true;
				}
			}
			return false;
		}
		virtual float pdf_value(const vec3& o, const vec3& v) const {
			hit_record rec;
			if (this->hit(ray(o, v), 0.001, FLT_MAX, rec)) {
				float cos_theta_max = sqrtf(1 - radius*radius / (center - o).squared_length());
				float solid_angle = 2 * M_PI*(1 - cos_theta_max);
				return 1 / solid_angle;
			}
			return 0;
		}
		virtual vec3 random(const vec3& o) const {
			vec3 direction = center - o;
			float distance_squared = direction.squared_length();
			onb uvw;
			uvw.build_from_w(direction);
			return uvw.local(random_to_sphere(radius, distance_squared));

		}
		vec3 center;
		float radius;
		const material *mat_ptr;
};

struct cu_sphere {
	float3 center;
	float radius;
};

#endif /* SPHERE_H_ */
