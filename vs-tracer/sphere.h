#ifndef SPHERE_H_
#define SPHERE_H_

#include <float.h>

#include "ray.h"
#include "onb.h"

struct material;

struct hit_record {
	hit_record() {}
	hit_record(float _t, const float3& _p, const float3& n, int _mat_idx): t(_t), p(_p), normal(n), mat_idx(_mat_idx) {}
	hit_record(const hit_record& rec): t(rec.t), p(rec.p), normal(rec.normal), mat_idx(rec.mat_idx) {}

	float t;
	float3 p;
	float3 normal;
	int mat_idx;
};

struct sphere {
	sphere(float3 cen, float r, int midx): center(cen), radius(r), mat_idx(midx) { }
	bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
		float3 oc = r.origin - center;
		float a = dot(r.direction, r.direction);
		float b = dot(oc, r.direction);
		float c = dot(oc, oc) - radius*radius;
		float discriminant = b*b - a*c;
		//if (sphere_dbg) printf("sphere_hit: a %.6f, b %.6f, c %.6f, d %.6f\n", a, b, c, discriminant);
		if (discriminant > 0) {
			float temp = (-b - sqrt(b*b - a*c)) / a;
			if (temp < t_max && temp > t_min) {
				rec = hit_record(temp, r.point_at_parameter(temp), (rec.p - center) / radius, mat_idx);
				return true;
			}
			temp = (-b + sqrt(b*b - a*c)) / a;
			if (temp < t_max && temp > t_min) {
				rec = hit_record(temp, r.point_at_parameter(temp), (rec.p - center) / radius, mat_idx);
				return true;
			}
		}
		return false;
	}

	float pdf_value(const float3& o, const float3& v) const {
		hit_record rec;
		if (this->hit(ray(o, v), 0.001, FLT_MAX, rec)) {
			float cos_theta_max = sqrtf(1 - radius*radius / squared_length(center - o));
			float solid_angle = 2 * M_PI*(1 - cos_theta_max);
			return 1 / solid_angle;
		}
		return 0;
	}

	float3 random(const float3& o) const {
		float3 direction = center - o;
		float distance_squared = squared_length(direction);
		onb uvw;
		uvw.build_from_w(direction);
		return uvw.local(random_to_sphere(radius, distance_squared));
	}

	float3 center;
	float radius;
	const int mat_idx;
};

#endif /* SPHERE_H_ */
