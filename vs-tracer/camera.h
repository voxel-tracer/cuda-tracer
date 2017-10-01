#ifndef CAMERA_H_
#define CAMERA_H_

#include "utils.h"
#include "ray.h"

class camera {
public:
	camera(vec3 lookfrom, vec3 _lookat, vec3 _vup, float vfov, float aspect, float aperture, float _focus_dist) { // vfov is top to bottom in degrees
		lens_radius = aperture / 2;
		float theta = vfov*M_PI/180;
		half_height = tan(theta/2);
		half_width = aspect * half_height;
		lookat = _lookat;
		focus_dist = _focus_dist;
		vup = _vup;
		radial_distance = (lookfrom - lookat).length();
		init(lookfrom);
	}

	void init(vec3 lookfrom) {
		origin = lookfrom;
		w = unit_vector(lookfrom - lookat);
		u = unit_vector(cross(vup, w));
		v = cross(w, u);
		lower_left_corner = origin - half_width*focus_dist*u - half_height*focus_dist*v - focus_dist*w;
		horizontal = 2*half_width*focus_dist*u;
		vertical = 2*half_height*focus_dist*v;
	}

	void look_from(float theta, float phi) {
		init(vec3(
			radial_distance*sinf(theta)*sinf(phi),
			radial_distance*cosf(theta),
			radial_distance*sinf(theta)*cosf(phi)) + lookat
		);
	}

	void get_ray(float s, float t, ray& r) const {
		vec3 rd = lens_radius*random_in_unit_disk();
		vec3 offset = u*rd.x() + v*rd.y();

		r.A = origin + offset;
		r.B = lower_left_corner + s*horizontal + t*vertical - origin - offset;
	}

	void get_ray(float s, float t, cu_ray& r) const {
		vec3 rd = lens_radius*random_in_unit_disk();
		vec3 offset = u*rd.x() + v*rd.y();

		r.origin = (origin + offset).to_float3();
		r.direction = (lower_left_corner + s*horizontal + t*vertical - origin - offset).to_float3();
	}

	vec3 lookat;
	vec3 origin;
	vec3 vup;
	float half_width;
	float half_height;
	float focus_dist;
	vec3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;
	vec3 u, v, w;
	float lens_radius;
private:
	float radial_distance;
};




#endif /* CAMERA_H_ */
