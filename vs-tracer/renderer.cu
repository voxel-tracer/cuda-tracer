#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>
#include <algorithm>
#include <iterator>

#include "renderer.h"
#include "sphere.h"
#include "device_launch_parameters.h"
#include "pdf.h"
#include "material.h"

#define DBG_IDX	-1 //42091

void err(cudaError_t err, char *msg)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to %s (error code %s)!\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

struct pixel_compare {
	const pixel* pixels;
	pixel_compare(pixel* _pixels) { pixels = _pixels; }

	bool operator() (int p0, int p1)
	{
		return pixels[p0].samples < pixels[p1].samples;
	}
};


inline void renderer::generate_ray(int ray_idx, int x, int y)
{
	float u = float(x + drand48()) / float(nx);
	float v = float(y + drand48()) / float(ny);
	cam->get_ray(u, v, h_rays[ray_idx]);
	samples[ray_idx] = sample((ny - y - 1)*nx + x);
}

void renderer::prepare_kernel()
{
	const unsigned int num_pixels = nx*ny;
	
	pixels = new pixel[num_pixels];
	h_clrs = new clr_rec[num_pixels];
	samples = new sample[num_pixels];
	h_rays = new ray[num_pixels];
	h_colors = new float3[num_pixels];
	pixel_idx = new int[num_pixels];
	
	// allocate device memory for input
    d_scene = NULL;
	err(cudaMalloc((void **)&d_scene, world->list_size * sizeof(sphere)), "allocate device d_scene");
	d_materials = NULL;
	err(cudaMalloc((void **)&d_materials, world->material_size * sizeof(material)), "allocate device d_materials");

    d_rays = NULL;
	err(cudaMalloc((void **)&d_rays, num_pixels * sizeof(ray)), "allocate device d_rays");

    d_hits = NULL;
	err(cudaMalloc((void **)&d_hits, num_pixels * sizeof(cu_hit)), "allocate device d_hits");

	d_clrs = NULL;
	err(cudaMalloc((void **)&d_clrs, num_pixels * sizeof(clr_rec)), "allocate device d_clrs");

	d_rnd_states = NULL;
	err(cudaMalloc((void **)&d_rnd_states, num_pixels * sizeof(curandStatePhilox4_32_10_t)), "allocate device rnd states");

    // Copy the host input in host memory to the device input in device memory
	err(cudaMemcpy(d_scene, world->list, world->list_size * sizeof(sphere), cudaMemcpyHostToDevice), "copy scene from host to device");
	err(cudaMemcpy(d_materials, world->materials, world->material_size * sizeof(material), cudaMemcpyHostToDevice), "copy materials from host to device");

	// set temporary variables
	for (unsigned int i = 0; i < num_pixels; i++)
	{
		pixels[i].id = i;
		pixels[i].samples = 1;
		pixel_idx[i] = i;
	}

	//clock_t start = clock();
	generate_rays();
	//generate += clock() - start;

	num_rays = num_pixels;
}

void renderer::update_camera()
{
	const unsigned int num_pixels = numpixels();

	// set temporary variables
	for (unsigned int i = 0; i < num_pixels; i++)
	{
		h_colors[i] = make_float3(0, 0, 0);
		pixels[i].id = i;
		pixels[i].samples = 1;
		pixels[i].done = 0;
		pixel_idx[i] = i;
	}

	//clock_t start = clock();
	generate_rays();
	//generate += clock() - start;
	num_rays = num_pixels;
}

void renderer::generate_rays()
{
	unsigned int ray_idx = 0;
	for (int j = ny - 1; j >= 0; j--)
		for (int i = 0; i < nx; ++i, ++ray_idx)
			generate_ray(ray_idx, i, j);
}

bool renderer::color(int ray_idx) {
	//ray& r = h_rays[ray_idx];
	//sample& s = samples[ray_idx];
	//const cu_hit& hit = h_hits[ray_idx];

	//if (hit.hit_idx == -1) {
	//	//if (s.pixelId == DBG_IDX)	printf("NO_HIT\n");

	//	// no intersection with spheres, return sky color
	//	float3 unit_direction = normalize(r.direction);
	//	float t = 0.5*(unit_direction.y + 1.0);
	//	float3 sky_clr = 1.0* ((1 - t)*make_float3(1.0, 1.0, 1.0) + t*make_float3(0.5, 0.7, 1.0));
	//	//float3 sky_clr(0, 0, 0);
	//	s.color += s.not_absorbed*sky_clr;
	//	return false;
	//}

	//sphere *sphr = (sphere*)(world->list[hit.hit_idx]);
	//float3 hit_p = r.point_at_parameter(hit.hit_t);
	//hit_record rec(hit.hit_t, hit_p, (hit_p - sphr->center) / sphr->radius, sphr->mat_idx);
	//const material *hit_mat = world->materials[rec.mat_idx];

	//scatter_record srec;
	//const float3& emitted =  hit_mat->emitted(r, rec, rec.p);
	//s.color += s.not_absorbed*emitted;
	////if (s.pixelId==DBG_IDX && s.color.squared_length() > 10) printf("white acne at %d\n", s.pixelId);
	////if (s.pixelId == DBG_IDX) printf("emitted=(%.2f,%.2f,%.2f), not_absorbed=%.6f\n", emitted[0], emitted[1], emitted[2], s.not_absorbed.squared_length());
	//if ((++r.depth) <= max_depth && hit_mat->scatter(r, rec, light_shape, srec)) {
	//	r.direction = srec.scattered.direction;
	//	r.origin = srec.scattered.origin;
	//	s.not_absorbed *= srec.attenuation;
	//	return true;
	//}

	return false;
}

//void renderer::simple_color() {
//
//	for (int ray_idx = 0; ray_idx < numpixels(); ++ray_idx) {
//		const ray& r = h_rays[ray_idx];
//		const cu_hit& hit = h_hits[ray_idx];
//		clr_rec& crec = h_clrs[ray_idx];
//
//		if (hit.hit_idx == -1) {
//			// no intersection with spheres, return sky color
//			float3 unit_direction = normalize(r.direction);
//			float t = 0.5*(unit_direction.y + 1.0);
//			crec.color = 1.0* ((1 - t)*make_float3(1.0, 1.0, 1.0) + t*make_float3(0.5, 0.7, 1.0));
//			crec.done = true;
//			continue;
//		}
//
//		sphere *sphr = (sphere*)(world->list[hit.hit_idx]);
//		float3 hit_p = r.point_at_parameter(hit.hit_t);
//		hit_record rec(hit.hit_t, hit_p, (hit_p - sphr->center) / sphr->radius, sphr->mat_idx); //TODO move this to sphere class
//		const material *hit_mat = world->materials[rec.mat_idx];
//
//		scatter_record srec;
//		if (r.depth < max_depth && scatter_lambertian(hit_mat, r, rec, light_shape, srec)) {
//			crec.origin = srec.scattered.origin;
//			crec.direction = srec.scattered.direction;
//			crec.color = srec.attenuation;
//			crec.done = false;
//		} else {
//			crec.color = make_float3(0, 0, 0);
//			crec.done = true;
//		}
//	}
//}

__global__ void
hit_scene(const ray* rays, const unsigned int num_rays, const sphere* scene, const unsigned int scene_size, float t_min, float t_max, cu_hit* hits)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= num_rays)
		return;

	const ray *r = &(rays[i]);
	const float3 ro = r->origin;
	const float3 rd = r->direction;

	float closest_hit = t_max;
	int hit_idx = -1;

	for (int s = 0; s < scene_size; s++)
	{
		const sphere sphere = scene[s];
		const float3 sc = sphere.center;
		const float sr = sphere.radius;

		float3 oc = make_float3(ro.x - sc.x, ro.y - sc.y, ro.z - sc.z);
		float a = rd.x*rd.x + rd.y*rd.y + rd.z*rd.z;
		float b = oc.x*rd.x + oc.y*rd.y + oc.z*rd.z;
		float c = (oc.x*oc.x + oc.y*oc.y + oc.z*oc.z) - sr*sr;
		float discriminant = b*b - a*c;
		if (discriminant > 0.01f) {
			float t = (-b - sqrtf(discriminant)) / a;
			//if (r->pixelId == DBG_IDX && s == 4) printf("hit_scene: a %.6f, b %.6f, c %.6f, d %.6f, t %.6f\n", a, b, c, discriminant, t);
			if (t < closest_hit && t > t_min) {
				closest_hit = t;
				hit_idx = s;
				continue;
			}
			t = (-b + sqrtf(discriminant)) / a;
			//if (r->pixelId == DBG_IDX && s == 4) printf("hit_scene: a %.6f, b %.6f, c %.6f, d %.6f, t %.6f\n", a, b, c, discriminant, t);
			if (t < closest_hit && t > t_min) {
				closest_hit = t;
				hit_idx = s;
			}
		}
	}

	//if (r->pixelId == DBG_IDX) printf("hit_scene: hit_idx %d, closest_hit %.2f\n", hit_idx, closest_hit);
	hits[i].hit_t = closest_hit;
	hits[i].hit_idx = hit_idx;
}

__global__ void init_kernel(curandStatePhilox4_32_10_t* states, const uint num_rays, const uint seed) {
	const int ray_idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (ray_idx >= num_rays)
		return;

	curand_init(seed, ray_idx, 0, &states[ray_idx]);
}

__global__ void
simple_color(const ray* rays, const cu_hit* hits, clr_rec* clrs, curandStatePhilox4_32_10_t* states, const int num_rays, const sphere* spheres, const int num_spheres, const material* materials, const int num_materials, const int max_depth) {
	const int ray_idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (ray_idx >= num_rays)
		return;

	const ray& r = rays[ray_idx];
	const cu_hit& hit = hits[ray_idx];
	clr_rec& crec = clrs[ray_idx];

	if (hit.hit_idx == -1) {
		// no intersection with spheres, return sky color
		float3 unit_direction = normalize(r.direction);
		float t = 0.5*(unit_direction.y + 1.0);
		crec.color = 1.0* ((1 - t)*make_float3(1.0, 1.0, 1.0) + t*make_float3(0.5, 0.7, 1.0));
		crec.done = true;
		return;
	}

	const sphere& sphr = spheres[hit.hit_idx];
	float3 hit_p = r.point_at_parameter(hit.hit_t);
	hit_record rec(hit.hit_t, hit_p, (hit_p - sphr.center) / sphr.radius, sphr.mat_idx); //TODO move this to sphere class
	const material& hit_mat = materials[rec.mat_idx];
	//printf("ray_idx %d, hit_idx %d, mat_idx %d, mat_type %d\n", ray_idx, hit.hit_idx, rec.mat_idx, hit_mat.type);

	curandStatePhilox4_32_10_t localState = states[ray_idx];

	scatter_record srec;
	if (scatter_lambertian(&hit_mat, r.direction, rec, NULL, &localState, srec)) {
		crec.origin = srec.scattered.origin;
		crec.direction = srec.scattered.direction;
		crec.color = srec.attenuation;
		crec.done = false;
	} else {
		crec.color = make_float3(0, 0, 0);
		crec.done = true;
	}

	states[ray_idx] = localState;
}

void renderer::run_kernel()
{
	cudaProfilerStart();
	clock_t start = clock();

	// copying rays to device
	err(cudaMemcpy(d_rays, h_rays, num_rays * sizeof(ray), cudaMemcpyHostToDevice), "copy rays from host to device");

	// Launch the CUDA Kernel
	int threadsPerBlock = 128;
	int blocksPerGrid = (num_rays + threadsPerBlock - 1) / threadsPerBlock;
	if (init_rnds) {
		init_kernel <<<blocksPerGrid, threadsPerBlock >>> (d_rnd_states, num_rays, 0);
		init_rnds = false;
	}
	hit_scene<<<blocksPerGrid, threadsPerBlock>>>(d_rays, num_rays, d_scene, world->list_size, 0.001f, FLT_MAX, d_hits);
	err(cudaGetLastError(), "launch hit_scene kernel");
	simple_color<<<blocksPerGrid, threadsPerBlock>>>(d_rays, d_hits, d_clrs, d_rnd_states, num_rays, d_scene, world->list_size, d_materials, world->material_size, max_depth);
	err(cudaGetLastError(), "launch simple_color kernel");

	// Copy the results to host
	err(cudaMemcpy(h_clrs, d_clrs, num_rays * sizeof(clr_rec), cudaMemcpyDeviceToHost), "copy results from device to host");

	kernel += clock() - start;
	cudaProfilerStop();
}

void renderer::compact_rays() {
	unsigned int sampled = 0;
	// first step only generate scattered rays and compact them
	for (unsigned int i = 0; i < numpixels(); ++i)
	{
		unsigned int pixelId = samples[i].pixelId;
		if (!color(i)) { // is ray no longer active ?
			// cumulate its color
			h_colors[pixelId] += samples[i].color;
			++(pixels[pixelId].done);
			//if (pixelId == DBG_IDX) printf("sample done\n");

			// generate new ray
			pixelId = pixel_idx[(sampled++) % numpixels()];
			pixels[pixelId].samples++;
			// then, generate a new sample
			const unsigned int x = pixelId % nx;
			const unsigned int y = ny - 1 - (pixelId / nx);
			generate_ray(i, x, y);
		}
	}
	std::sort(pixel_idx, pixel_idx + numpixels(), pixel_compare(pixels));
	num_rays = (pixels[pixel_idx[0]].samples <= ns) ? numpixels() : 0;
}

void renderer::simple_compact_rays() {
	clock_t start = clock();

	unsigned int sampled = 0;
	// first step only generate scattered rays and compact them
	for (unsigned int i = 0; i < numpixels(); ++i)
	{
		const clr_rec& crec = h_clrs[i];
		sample& s = samples[i];
		unsigned int pixelId = s.pixelId;
		if (s.depth == max_depth || crec.done) { // ray no longer active ?
			if (crec.done) // cumulate its color
				h_colors[pixelId] += s.not_absorbed*crec.color;
			++(pixels[pixelId].done);
			//if (pixelId == DBG_IDX) printf("sample done\n");

			// generate new ray
			pixelId = pixel_idx[(sampled++) % numpixels()];
			pixels[pixelId].samples++;
			// then, generate a new sample
			const unsigned int x = pixelId % nx;
			const unsigned int y = ny - 1 - (pixelId / nx);
			generate_ray(i, x, y);
		} else { // ray has been scattered
			s.not_absorbed *= crec.color;
			h_rays[i].origin = crec.origin;
			h_rays[i].direction = crec.direction;
			++s.depth;
		}
	}
	std::sort(pixel_idx, pixel_idx + numpixels(), pixel_compare(pixels));
	num_rays = (pixels[pixel_idx[0]].samples <= ns) ? numpixels() : 0;

	compact += clock() - start;
}

void renderer::destroy() {
	// Free device global memory
	err(cudaFree(d_scene), "free device d_scene");
	err(cudaFree(d_materials), "free device d_materials");
	err(cudaFree(d_rays), "free device d_rays");
	err(cudaFree(d_hits), "free device d_hits");
	err(cudaFree(d_clrs), "free device d_clrs");
	err(cudaFree(d_rnd_states), "free device d_rnd_states");

	// Free host memory
	delete[] samples;
	delete[] h_clrs;
}