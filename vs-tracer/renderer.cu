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
	const uint ns;
	pixel_compare(const pixel* _pixels, uint _ns): pixels(_pixels), ns(_ns) {}

	bool operator() (int p0, int p1) {
		const uint s0 = pixels[p0].samples;
		const uint s1 = pixels[p1].samples;
		if (s1 >= ns) return true;
		else if (s0 >= ns) return false;

		const uint d0 = pixels[p0].done;
		const uint d1 = pixels[p1].done;
		if (d0 == d1) return s0 < s1;
		return d0 < d1;
	}
};


void renderer::prepare_kernel() {
	const unsigned int num_pixels = nx*ny;
	const uint unit_numpixels = num_pixels / num_units; //TODO make sure we don't miss any rays because of precision loss

	remaining_pixels = num_pixels;
	next_pixel = 0;
	total_rays = 0;
	
	h_colors = new float3[num_pixels];

	// allocate device memory for input
    d_scene = NULL;
	err(cudaMalloc((void **)&d_scene, world->list_size * sizeof(sphere)), "allocate device d_scene");
	d_materials = NULL;
	err(cudaMalloc((void **)&d_materials, world->material_size * sizeof(material)), "allocate device d_materials");

    // Copy the host input in host memory to the device input in device memory
	err(cudaMemcpy(d_scene, world->list, world->list_size * sizeof(sphere), cudaMemcpyHostToDevice), "copy scene from host to device");
	err(cudaMemcpy(d_materials, world->materials, world->material_size * sizeof(material), cudaMemcpyHostToDevice), "copy materials from host to device");

	wunits = new work_unit*[num_units];
	uint cur_idx = 0;
	for (uint unit = 0; unit < num_units; unit++) {
		uint next_idx = cur_idx + unit_numpixels;
		work_unit *wu = new work_unit(cur_idx, next_idx);
		const uint unit_len = wu->length();

		wu->pixel_idx = new int[unit_len];
		//for (uint i = 0; i < wu->length(); ++i) 
		//	wu->pixel_idx[i] = wu->start_idx + i;
		wu->samples = new sample[unit_len];

		wu->pixels = new pixel[unit_len];
		for (uint i = 0; i < unit_len; i++) {
			wu->pixels[i] = pixel(cur_idx + i, unit);
			wu->pixels[i].samples = 1;
		}

		err(cudaMallocHost(&wu->h_rays, unit_len * sizeof(ray)), "allocate h_rays");
		err(cudaMalloc((void **)&(wu->d_rays), unit_len * sizeof(ray)), "allocate device d_rays");
		err(cudaMalloc((void **)&(wu->d_hits), unit_len * sizeof(cu_hit)), "allocate device d_hits");
		err(cudaMallocHost(&(wu->h_clrs), unit_len * sizeof(clr_rec)), "allocate h_clrs");
		err(cudaMalloc((void **)&(wu->d_clrs), unit_len * sizeof(clr_rec)), "allocate device d_clrs");
		err(cudaStreamCreate(&wu->stream), "cuda stream create");

		wunits[unit] = wu;
		cur_idx = next_idx;
	}

	//clock_t start = clock();
	generate_rays();
	//generate += clock() - start;
}

void renderer::update_camera()
{
	const unsigned int num_pixels = numpixels();

	// set temporary variables
	for (unsigned int i = 0; i < num_pixels; i++) {
		h_colors[i] = make_float3(0, 0, 0);
	}

	for (uint unit = 0; unit < num_units; unit++) {
		work_unit* wu = wunits[unit];
		wu->done = false;
		for (uint i = 0; i < wu->length(); i++) {
			wu->pixels[i].id = wu->start_idx + i;
			wu->pixels[i].samples = 1;
			wu->pixels[i].done = 0;
		}
	}

	generate_rays();
	num_runs = 0;
}

void renderer::generate_rays() {
	uint ray_idx = 0;
	for (int j = ny - 1; j >= 0; j--)
		for (int i = 0; i < nx; ++i, ++ray_idx) {
			// for initial generation ray_idx == pixelId
			const uint unit_idx = get_unitIdx(ray_idx);
			generate_ray(wunits[unit_idx], ray_idx, i, j);
		}
}

inline void renderer::generate_ray(work_unit* wu, const uint sampleId, int x, int y) {
	const float u = float(x + drand48()) / float(nx);
	const float v = float(y + drand48()) / float(ny);
	const uint local_ray_idx = sampleId - wu->start_idx;
	cam->get_ray(u, v,wu->h_rays[local_ray_idx]);
	wu->samples[local_ray_idx] = sample((ny - y - 1)*nx + x);
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
__global__ void hit_scene(const ray* rays, const uint num_rays, const sphere* scene, const unsigned int scene_size, float t_min, float t_max, cu_hit* hits)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= num_rays) return;

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

	hits[i].hit_t = closest_hit;
	hits[i].hit_idx = hit_idx;
}

__global__ void simple_color(const ray* rays, const cu_hit* hits, clr_rec* clrs, const uint seed, const uint num_rays, const sphere* spheres, const int num_spheres, const material* materials, const int num_materials, const int max_depth) {
	const int ray_idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (ray_idx >= num_rays) return;

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
	const material hit_mat = materials[rec.mat_idx];
	curandStatePhilox4_32_10_t localState;
	curand_init(0, seed*blockDim.x + threadIdx.x, 0, &localState);

	scatter_record srec;
	//const float3& emitted =  hit_mat->emitted(r, rec, rec.p);
	//crec.color += crec.not_absorbed*emitted;
	if (hit_mat.scatter(r.direction, rec, NULL, &localState, srec)) {
		crec.origin = srec.scattered.origin;
		crec.direction = srec.scattered.direction;
		crec.color = srec.attenuation;
		crec.done = false;
	} else {
		crec.color = make_float3(0, 0, 0);
		crec.done = true;
	}
}

void renderer::copy_rays_to_gpu(const work_unit* wu) {
	err(cudaMemcpyAsync(wu->d_rays, wu->h_rays, wu->length() * sizeof(ray), cudaMemcpyHostToDevice, wu->stream), "copy rays from host to device");
}

void renderer::copy_colors_from_gpu(const work_unit* wu) {
	err(cudaMemcpyAsync(wu->h_clrs, wu->d_clrs, wu->length() * sizeof(clr_rec), cudaMemcpyDeviceToHost, wu->stream), "copy results from device to host");
}

void renderer::start_kernel(const work_unit* wu) {
	int threadsPerBlock = 128;
	int blocksPerGrid = (wu->length() + threadsPerBlock - 1) / threadsPerBlock;
	hit_scene <<<blocksPerGrid, threadsPerBlock, 0, wu->stream >>>(wu->d_rays, wu->length(), d_scene, world->list_size, 0.1f, FLT_MAX, wu->d_hits);
	//err(cudaGetLastError(), "launch hit_scene kernel");
	simple_color <<<blocksPerGrid, threadsPerBlock, 0, wu->stream >>>(wu->d_rays, wu->d_hits, wu->d_clrs, num_runs++, wu->length(), d_scene, world->list_size, d_materials, world->material_size, max_depth);
	//err(cudaGetLastError(), "launch simple_color kernel");
}

void renderer::render_work_unit(uint unit_idx) {
	work_unit* wu = wunits[unit_idx];
	while (!wu->done) {
		copy_rays_to_gpu(wu);
		start_kernel(wu);
		copy_colors_from_gpu(wu);
		cudaStreamQuery(wu->stream); // flush stream to start the kernel 
		cudaStreamSynchronize(wu->stream);
		compact_rays(wu);
	}
}

void renderer::compact_rays(work_unit* wu) {
	uint done_samples = 0;
	bool not_done = false;
	for (uint i = 0; i < wu->length(); ++i) {
		const clr_rec& crec = wu->h_clrs[i];
		sample& s = wu->samples[i];
		const uint local_pixelId = s.pixelId - wu->start_idx;
		s.done = crec.done || s.depth == max_depth;
		if (s.done) {
			if (crec.done) h_colors[s.pixelId] += s.not_absorbed*crec.color;
			++(wu->pixels[local_pixelId].done);
			++done_samples;
		} else {
			s.not_absorbed *= crec.color;
			wu->h_rays[i].origin = crec.origin;
			wu->h_rays[i].direction = crec.direction;
			++s.depth;
		}
		not_done = not_done || (wu->pixels[local_pixelId].done < ns);
	}

	if (done_samples > 0 && not_done) {
		// sort uint ray [wu->start_idx, wu->end_idx[
		for (uint i = 0; i < wu->length(); ++i) wu->pixel_idx[i] = i;
		std::sort(wu->pixel_idx, wu->pixel_idx + wu->length(), pixel_compare(wu->pixels, ns));
		uint sampled = 0;
		for (uint i = 0; i < wu->length(); ++i) {
			const uint sId = wu->start_idx + i;
			sample& s = wu->samples[i];
			if (s.done) {
				// generate new ray
				const uint local_pixelId = wu->pixel_idx[sampled++];
				const uint pixelId = wu->start_idx + local_pixelId;
				wu->pixels[local_pixelId].samples++;
				// then, generate a new sample
				const unsigned int x = pixelId % nx;
				const unsigned int y = ny - 1 - (pixelId / nx);
				generate_ray(wu, sId, x, y);
			}
		}
	}

	wu->done = !not_done;
}

void renderer::destroy() {
	// Free device global memory
	err(cudaFree(d_scene), "free device d_scene");
	err(cudaFree(d_materials), "free device d_materials");

	for (uint unit = 0; unit < num_units; unit++) {
		work_unit *wu = wunits[unit];
		err(cudaFree(wu->d_rays), "free device d_rays");
		err(cudaFree(wu->d_hits), "free device d_hits");
		err(cudaFree(wu->d_clrs), "free device d_clrs");

		err(cudaStreamDestroy(wu->stream), "destroy cuda stream");

		cudaFreeHost(wu->h_clrs);
		cudaFreeHost(wu->h_rays);

		delete[] wu->pixel_idx;
		delete[] wu->samples;
		delete[] wu->pixels;
	}

	// Free host memory
	delete[] wunits;
}