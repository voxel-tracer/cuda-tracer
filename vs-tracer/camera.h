#ifndef CAMERA_H_
#define CAMERA_H_

#include "utils.h"
#include "ray.h"

class camera {
public:
	camera(float3 lookfrom, float3 _lookat, float3 _vup, float vfov, float aspect, float aperture, float _focus_dist) { // vfov is top to bottom in degrees
		lens_radius = aperture / 2;
		float theta = (float)(vfov*M_PI / 180);
		half_height = tan(theta/2);
		half_width = aspect * half_height;
		lookat = _lookat;
		focus_dist = _focus_dist;
		vup = _vup;
		radial_distance = length(lookfrom - lookat);
		init(lookfrom);
	}

	void init(float3 lookfrom) {
		origin = lookfrom;
		w = normalize(lookfrom - lookat);
		u = normalize(cross(vup, w));
		v = cross(w, u);
		lower_left_corner = origin - half_width*focus_dist*u - half_height*focus_dist*v - focus_dist*w;
		horizontal = 2*half_width*focus_dist*u;
		vertical = 2*half_height*focus_dist*v;
	}

	void look_from(float theta, float phi) {
		init(make_float3(
			radial_distance*sinf(theta)*sinf(phi),
			radial_distance*cosf(theta),
			radial_distance*sinf(theta)*cosf(phi)) + lookat
		);
	}

	void get_ray(float s, float t, ray& r) const {
		float3 rd = lens_radius*random_in_unit_disk();
		float3 offset = u*rd.x + v*rd.y;

		r.origin = origin + offset;
		r.direction = lower_left_corner + s*horizontal + t*vertical - origin - offset;
	}

	float3 lookat;
	float3 origin;
	float3 vup;
	float half_width;
	float half_height;
	float focus_dist;
	float3 lower_left_corner;
	float3 horizontal;
	float3 vertical;
	float3 u, v, w;
	float lens_radius;
private:
	float radial_distance;
};




#endif /* CAMERA_H_ */
