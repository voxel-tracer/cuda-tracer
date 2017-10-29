#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>
#include <algorithm>
#include <iterator>

#include "renderer.h"
#include "sphere.h"
#include "device_launch_parameters.h"
#include "pdf.h"

#define DBG_IDX	-1 //42091

void err(cudaError_t err, char *msg)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to %s (error code %s)!\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

cu_sphere*
init_cu_scene(const hitable_list* world)
{
	const unsigned int size = world->list_size;
	cu_sphere* scene = (cu_sphere*)malloc(size * sizeof(cu_sphere));
	for (int i = 0; i < size; i++)
	{
		const sphere *s = (sphere*)world->list[i];
		scene[i].center = make_float3(s->center.x, s->center.y, s->center.z);
		scene[i].radius = s->radius;
	}

	return scene;
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
	samples[ray_idx].depth = 0;
	samples[ray_idx].pixelId = (ny - y - 1)*nx + x;
	h_rays[ray_idx].pixelId = samples[ray_idx].pixelId;
}

void renderer::prepare_kernel()
{
	const unsigned int num_pixels = nx*ny;
	cu_sphere *h_scene = init_cu_scene(world);
	scene_size = 500;
	
	pixels = new pixel[num_pixels];
	samples = new sample[num_pixels];
	h_rays = new ray[num_pixels];
	h_colors = new float3[num_pixels];
	h_hits = new cu_hit[num_pixels];
	pixel_idx = new int[num_pixels];
	
	// allocate device memory for input
    d_scene = NULL;
	err(cudaMalloc((void **)&d_scene, scene_size * sizeof(cu_sphere)), "allocate device d_scene");

    d_rays = NULL;
	err(cudaMalloc((void **)&d_rays, num_pixels * sizeof(ray)), "allocate device d_rays");

    d_hits = NULL;
	err(cudaMalloc((void **)&d_hits, num_pixels * sizeof(cu_hit)), "allocate device d_hits");

    // Copy the host input in host memory to the device input in device memory
	err(cudaMemcpy(d_scene, h_scene, world->list_size * sizeof(cu_sphere), cudaMemcpyHostToDevice), "copy scene from host to device");

	// set temporary variables
	for (int i = 0; i < num_pixels; i++)
	{
		samples[i].color = make_float3(0, 0, 0);
		samples[i].not_absorbed = make_float3(1, 1, 1);
		pixels[i].id = i;
		pixels[i].samples = 1;
		pixel_idx[i] = i;
	}

	//clock_t start = clock();
	generate_rays(h_rays);
	//generate += clock() - start;

	num_rays = num_pixels;

	free(h_scene);
}


void renderer::update_camera()
{
	const unsigned int num_pixels = numpixels();

	// set temporary variables
	for (int i = 0; i < num_pixels; i++)
	{
		h_colors[i] = make_float3(0, 0, 0);
		samples[i].color = make_float3(0, 0, 0);
		samples[i].not_absorbed = make_float3(1, 1, 1);
		pixels[i].id = i;
		pixels[i].samples = 1;
		pixels[i].done = 0;
		pixel_idx[i] = i;
	}

	//clock_t start = clock();
	generate_rays(h_rays);
	//generate += clock() - start;
	num_rays = num_pixels;
}

ray* renderer::generate_rays(ray* rays)
{
	unsigned int ray_idx = 0;
	for (int j = ny - 1; j >= 0; j--)
		for (int i = 0; i < nx; ++i, ++ray_idx)
			generate_ray(ray_idx, i, j);

	return rays;
}

bool renderer::color(int ray_idx) {
	ray& cu_r = h_rays[ray_idx];
	const cu_hit& hit = h_hits[ray_idx];
	sample& s = samples[ray_idx];
	ray r = ray(cu_r);

	if (hit.hit_idx == -1) {
		//if (s.pixelId == DBG_IDX)	printf("NO_HIT\n");

		// no intersection with spheres, return sky color
		float3 unit_direction = normalize(r.direction);
		float t = 0.5*(unit_direction.y + 1.0);
		float3 sky_clr = 1.0* ((1 - t)*make_float3(1.0, 1.0, 1.0) + t*make_float3(0.5, 0.7, 1.0));
		//float3 sky_clr(0, 0, 0);
		s.color += s.not_absorbed*sky_clr;
		return false;
	}

	hit_record rec;
	sphere *sphr = (sphere*)(world->list[hit.hit_idx]);
	rec.t = hit.hit_t;
	rec.p = r.point_at_parameter(hit.hit_t);
	rec.normal = (rec.p - sphr->center) / sphr->radius;
	rec.mat_ptr = sphr->mat_ptr;

	scatter_record srec;
	const float3& emitted = rec.mat_ptr->emitted(r, rec, rec.p);
	s.color += s.not_absorbed*emitted;
	//if (s.pixelId==DBG_IDX && s.color.squared_length() > 10) printf("white acne at %d\n", s.pixelId);
	//if (s.pixelId == DBG_IDX) printf("emitted=(%.2f,%.2f,%.2f), not_absorbed=%.6f\n", emitted[0], emitted[1], emitted[2], s.not_absorbed.squared_length());
	if ((++s.depth) <= max_depth && rec.mat_ptr->scatter(r, rec, light_shape, srec)) {
		cu_r.origin = srec.scattered.origin;
		cu_r.direction = srec.scattered.direction;
		s.not_absorbed *= srec.attenuation;
		return true;
	}

	return false;
}

__global__ void
hit_scene(const ray* rays, const unsigned int num_rays, const cu_sphere* scene, const unsigned int scene_size, float t_min, float t_max, cu_hit* hits)
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
		const cu_sphere sphere = scene[s];
		const float3 sc = sphere.center;
		const float sr = sphere.radius;

		float3 oc = make_float3(ro.x - sc.x, ro.y - sc.y, ro.z - sc.z);
		float a = rd.x*rd.x + rd.y*rd.y + rd.z*rd.z;
		float b = oc.x*rd.x + oc.y*rd.y + oc.z*rd.z;
		float c = (oc.x*oc.x + oc.y*oc.y + oc.z*oc.z) - sr*sr;
		float discriminant = b*b - a*c;
		if (discriminant > 0.01) {
			float t = (-b - sqrtf(discriminant)) / a;
			if (r->pixelId == DBG_IDX && s == 4) printf("hit_scene: a %.6f, b %.6f, c %.6f, d %.6f, t %.6f\n", a, b, c, discriminant, t);
			if (t < closest_hit && t > t_min) {
				closest_hit = t;
				hit_idx = s;
				continue;
			}
			t = (-b + sqrtf(discriminant)) / a;
			if (r->pixelId == DBG_IDX && s == 4) printf("hit_scene: a %.6f, b %.6f, c %.6f, d %.6f, t %.6f\n", a, b, c, discriminant, t);
			if (t < closest_hit && t > t_min) {
				closest_hit = t;
				hit_idx = s;
			}
		}
	}

	if (r->pixelId == DBG_IDX) printf("hit_scene: hit_idx %d, closest_hit %.2f\n", hit_idx, closest_hit);
	hits[i].hit_t = closest_hit;
	hits[i].hit_idx = hit_idx;
}

void renderer::run_kernel()
{
	cudaProfilerStart();
	//clock_t start = clock();

	// copying rays to device
	err(cudaMemcpy(d_rays, h_rays, num_rays * sizeof(ray), cudaMemcpyHostToDevice), "copy rays from host to device");

	// Launch the CUDA Kernel
	int threadsPerBlock = 128;
	int blocksPerGrid = (num_rays + threadsPerBlock - 1) / threadsPerBlock;
	hit_scene<<<blocksPerGrid, threadsPerBlock>>>(d_rays, num_rays, d_scene, scene_size, 0.001, FLT_MAX, d_hits);
	err(cudaGetLastError(), "launch kernel");

	// Copy the results to host
	err(cudaMemcpy(h_hits, d_hits, num_rays * sizeof(cu_hit), cudaMemcpyDeviceToHost), "copy results from device to host");

	//kernel += clock() - start;
	cudaProfilerStop();
}

void renderer::compact_rays()
{
	unsigned int sampled = 0;
	// first step only generate scattered rays and compact them
	for (unsigned int i = 0; i < numpixels(); ++i)
	{
		unsigned int pixelId = samples[i].pixelId;
		//clock_t start = clock();
		bool active = color(i);
		//compact += clock() - start;
		if (!active) { // is ray no longer active ?
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
			samples[i].color = make_float3(0, 0, 0);
			samples[i].not_absorbed = make_float3(1, 1, 1);
		}
	}
	std::sort(pixel_idx, pixel_idx + numpixels(), pixel_compare(pixels));
	num_rays = (pixels[pixel_idx[0]].samples <= ns) ? numpixels() : 0;
}

void renderer::destroy() {
	// Free device global memory
	err(cudaFree(d_scene), "free device d_scene");
	err(cudaFree(d_rays), "free device d_rays");
	err(cudaFree(d_hits), "free device d_hits");

	// Free host memory
	free(h_hits);
	delete[] samples;
}