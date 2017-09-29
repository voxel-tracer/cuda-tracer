#pragma once

#include <ctime>

#include "camera.h"
#include "hitable_list.h"
#include "sphere.h"

struct cu_hit {
	int hit_idx;
	float hit_t;
};

struct pixel {
	//TODO pixel should know it's coordinates and it's id should be a computed field
	unsigned int id;
	unsigned int samples;

	pixel() { id = 0; samples = 0; }
};

class renderer {
public:
	renderer(camera* _cam, hitable_list* w, unsigned int _nx, unsigned int _ny, unsigned int _ns, unsigned int _max_depth, float _min_attenuation) { 
		cam = _cam;
		world = w;
		nx = _nx; 
		ny = _ny; 
		ns = _ns;
		max_depth = _max_depth;
		min_attenuation = _min_attenuation;
	}

	unsigned int numpixels() const { return nx*ny; }
	unsigned int numrays() const { return num_rays; }
	unsigned int get_pixelId(int x, int y) { return (ny - y - 1)*nx + x; }
	vec3 get_pixel_color(int x, int y) {
		const unsigned int pixelId = get_pixelId(x, y);
		if (pixels[pixelId].samples == 0) return vec3(0, 0, 0);
		return h_colors[pixelId] / float(pixels[pixelId].samples);
	}

	void prepare_kernel();

	bool color(cu_ray& cu_r, const cu_hit& hit, vec3& sample_clr);
	cu_ray* generate_rays(cu_ray* rays);
	void run_kernel();
	void compact_rays();

	void destroy();

	camera* cam;
	hitable_list *world;
	unsigned int nx;
	unsigned int ny;
	unsigned int ns;
	unsigned int max_depth;
	float min_attenuation;

	cu_ray* h_rays;
	cu_ray* d_rays;
	cu_hit* h_hits;
	cu_hit* d_hits;
	cu_sphere* d_scene;
	unsigned int scene_size;

	pixel* pixels;
	vec3* h_colors;
	vec3* h_sample_colors;
	unsigned int* pix_array;

	clock_t kernel = 0;
	clock_t generate = 0;
	clock_t compact = 0;

	unsigned int num_rays;
private:
	int* pixel_idx;
};
inline void generate_ray(const camera* cam, cu_ray& r, const unsigned int x, const unsigned int y, const unsigned int nx, const unsigned int ny);