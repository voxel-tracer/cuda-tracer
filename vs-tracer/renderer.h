#pragma once

#include <ctime>

#include "camera.h"
#include "hitable_list.h"

struct cu_hit {
	int hit_idx = 0;
	float hit_t = 0;
};

struct pixel {
	//TODO pixel should know it's coordinates and it's id should be a computed field
	uint id = 0;
	uint unit_idx = 0;

	uint samples = 0;
	uint done = 0; // needed to differentiate between done vs ongoing samples, when doing progressive rendering

	pixel() {}
	pixel(uint _id, uint _unit_idx) : id(_id), unit_idx(_unit_idx), samples(0), done(0) {}
	pixel(const pixel &p) : id(p.id), unit_idx(p.unit_idx), samples(p.samples), done(p.done) {}
};

struct work_unit {
	cudaStream_t stream;
	const uint start_idx;
	const uint end_idx;
	ray* h_rays;
	ray* d_rays;
	cu_hit* d_hits;
	clr_rec* h_clrs;
	clr_rec* d_clrs;
	int * pixel_idx;
	sample* samples;

	bool done = false;

	work_unit(uint start, uint end) :start_idx(start), end_idx(end) {}
	uint length() const { return end_idx - start_idx; }
};

class renderer {
public:
	renderer(const camera* _cam, const hitable_list* w, const sphere *ls, unsigned int _nx, unsigned int _ny, unsigned int _ns, unsigned int _max_depth, float _min_attenuation, uint nunits):
		cam(_cam), world(w), light_shape(ls), nx(_nx), ny(_ny), ns(_ns), max_depth(_max_depth), min_attenuation(_min_attenuation), num_units(nunits) {}

	unsigned int numpixels() const { return nx*ny; }
	bool is_not_done() const { return !(wunits[0]->done && wunits[1]->done); }
	unsigned int get_pixelId(int x, int y) const { return (ny - y - 1)*nx + x; }
	float3 get_pixel_color(int x, int y) const {
		const unsigned int pixelId = get_pixelId(x, y);
		if (pixels[pixelId].done == 0) return make_float3(0, 0, 0);
		return h_colors[pixelId] / float(pixels[pixelId].done);
	}
	uint totalrays() const { return total_rays; }

	void prepare_kernel();
	void update_camera();

	bool color(int ray_idx);
	void generate_rays();
	
	void render_work_unit(uint unit_idx);

	void destroy();

	const camera* const cam;
	const hitable_list * const world;
	const uint nx;
	const uint ny;
	const uint ns;
	const uint max_depth;
	const float min_attenuation;

	sphere* d_scene;
	material* d_materials;
	bool init_rnds = true;

	pixel* pixels;
	float3* h_colors;

	clock_t kernel = 0;
	clock_t generate = 0;
	clock_t compact = 0;

private:
	void copy_rays_to_gpu(const work_unit* wu);
	void start_kernel(const work_unit* wu);
	void copy_colors_from_gpu(const work_unit* wu);
	void compact_rays(work_unit* wu);
	inline void generate_ray(work_unit* wu, const uint ray_idx, int x, int y);

	uint total_rays = 0;
	work_unit **wunits;
	const uint num_units;
	uint next_pixel = 0;
	int remaining_pixels = 0;
	uint num_runs = 0;
	const sphere * const light_shape;
};