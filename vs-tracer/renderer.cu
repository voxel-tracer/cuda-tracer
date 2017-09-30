#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>
#include <algorithm>
#include <iterator>

#include "renderer.h"
#include "sphere.h"
#include "device_launch_parameters.h"



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
		scene[i].center = make_float3(s->center.x(), s->center.y(), s->center.z());
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


inline void generate_ray(const camera* cam, cu_ray& r, const unsigned int x, const unsigned int y, const unsigned int nx, const unsigned int ny)
{
	float u = float(x + drand48()) / float(nx);
	float v = float(y + drand48()) / float(ny);
	cam->get_ray(u, v, r);
	r.depth = 0;
	r.pixelId = (ny - y - 1)*nx + x;
}

void renderer::prepare_kernel()
{
	const unsigned int num_pixels = nx*ny;
	cu_sphere *h_scene = init_cu_scene(world);
	scene_size = 500;
	
	pixels = new pixel[num_pixels];
	h_rays = new cu_ray[num_pixels];
	h_colors = new vec3[num_pixels];
	h_sample_colors = new vec3[num_pixels];
	h_hits = new cu_hit[num_pixels];
	pixel_idx = new int[num_pixels];
	
	// allocate device memory for input
    d_scene = NULL;
	err(cudaMalloc((void **)&d_scene, scene_size * sizeof(cu_sphere)), "allocate device d_scene");

    d_rays = NULL;
	err(cudaMalloc((void **)&d_rays, num_pixels * sizeof(cu_ray)), "allocate device d_rays");

    d_hits = NULL;
	err(cudaMalloc((void **)&d_hits, num_pixels * sizeof(cu_hit)), "allocate device d_hits");

    // Copy the host input in host memory to the device input in device memory
	err(cudaMemcpy(d_scene, h_scene, world->list_size * sizeof(cu_sphere), cudaMemcpyHostToDevice), "copy scene from host to device");

	// set temporary variables
	for (int i = 0; i < num_pixels; i++)
	{
		h_sample_colors[i] = vec3(1, 1, 1);
		pixels[i].id = i;
		pixels[i].samples = 1;
		pixel_idx[i] = i;
	}

	clock_t start = clock();
	generate_rays(h_rays);
	generate += clock() - start;

	num_rays = num_pixels;

	free(h_scene);

	pix_array = new unsigned int[nx * ny];
}

cu_ray* renderer::generate_rays(cu_ray* rays)
{
	unsigned int ray_idx = 0;
	for (int j = ny - 1; j >= 0; j--)
		for (int i = 0; i < nx; ++i, ++ray_idx)
			generate_ray(cam, rays[ray_idx], i, j, nx, ny);

	return rays;
}

bool renderer::color(cu_ray& cu_r, const cu_hit& hit, vec3& sample_clr) {
	ray r = ray(vec3(cu_r.origin), vec3(cu_r.direction));

	if (hit.hit_idx == -1) {
		// no intersection with spheres, return sky color
		vec3 unit_direction = unit_vector(r.direction());
		float t = 0.5*(unit_direction.y() + 1.0);
		sample_clr *= (1 - t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
		return false;
	}

	hit_record rec;
	sphere *s = (sphere*)(world->list[hit.hit_idx]);
	rec.t = hit.hit_t;
	rec.p = r.point_at_parameter(hit.hit_t);
	rec.normal = (rec.p - s->center) / s->radius;
	rec.mat_ptr = s->mat_ptr;

	vec3 attenuation;
	if ((++cu_r.depth) <= max_depth && scatter(*rec.mat_ptr, r, rec, attenuation, r)) {
		cu_r.origin = r.origin().to_float3();
		cu_r.direction = r.direction().to_float3();

		sample_clr *= attenuation;
		return true;
	}

	sample_clr = vec3(0, 0, 0);
	return false;
}

__global__ void
hit_scene(const cu_ray* rays, const unsigned int num_rays, const cu_sphere* scene, const unsigned int scene_size, float t_min, float t_max, cu_hit* hits)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= num_rays)
		return;

	const cu_ray *r = &(rays[i]);
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
		float b = 2.0f * (oc.x*rd.x + oc.y*rd.y + oc.z*rd.z);
		float c = oc.x*oc.x + oc.y*oc.y + oc.z*oc.z - sr*sr;
		float discriminant = b*b - 4 * a*c;
		if (discriminant > 0)
		{
			float t = (-b - sqrtf(discriminant)) / (2.0f*a);
			if (t < closest_hit && t > t_min) {
				closest_hit = t;
				hit_idx = s;
			}
		}
	}

	hits[i].hit_t = closest_hit;
	hits[i].hit_idx = hit_idx;
}

void renderer::run_kernel()
{
	cudaProfilerStart();
	clock_t start = clock();

	// copying rays to device
	err(cudaMemcpy(d_rays, h_rays, num_rays * sizeof(cu_ray), cudaMemcpyHostToDevice), "copy rays from host to device");

	// Launch the CUDA Kernel
	int threadsPerBlock = 128;
	int blocksPerGrid = (num_rays + threadsPerBlock - 1) / threadsPerBlock;
	hit_scene<<<blocksPerGrid, threadsPerBlock>>>(d_rays, num_rays, d_scene, scene_size, 0.001, FLT_MAX, d_hits);
	err(cudaGetLastError(), "launch kernel");

	// Copy the results to host
	err(cudaMemcpy(h_hits, d_hits, num_rays * sizeof(cu_hit), cudaMemcpyDeviceToHost), "copy results from device to host");

	kernel += clock() - start;
	cudaProfilerStop();
}


void renderer::compact_rays()
{
	// first step only generate scattered rays and compact them
	clock_t start = clock();
	unsigned int ray_idx = 0;
	for (unsigned int i = 0; i < num_rays; ++i)
	{
		const unsigned int pixelId = h_rays[i].pixelId;
		if (color(h_rays[i], h_hits[i], h_sample_colors[i]) && h_sample_colors[i].squared_length() > min_attenuation)
		{
			// compact ray
			h_rays[ray_idx] = h_rays[i];
			h_sample_colors[ray_idx] = h_sample_colors[i];
			++ray_idx;
		}
		else
		{
			// ray is no longer active, cumulate its color
			h_colors[pixelId] += h_sample_colors[i];
			unsigned int num_samples = ++(pixels[pixelId].done);
			//TODO extract this logic out of the renderer
			vec3 col = h_colors[pixelId] / float(num_samples);
			col = vec3(sqrtf(col[0]), sqrtf(col[1]), sqrtf(col[2]));
			int ir = int(255.99*col.r());
			int ig = int(255.99*col.g());
			int ib = int(255.99*col.b());
			pix_array[pixelId] = (ir << 16) + (ig << 8) + ib;
		}
	}
	compact += clock() - start;
	// for each ray that's no longer active, sample a pixel that's not fully sampled yet
	start = clock();
	std::sort(pixel_idx, pixel_idx+numpixels(), pixel_compare(pixels));
	unsigned int sampled = 0;
	do
	{
		sampled = 0;
		for (unsigned int i = 0; i < numpixels() && ray_idx < numpixels(); ++i)
		{
			const unsigned int pixelId = pixel_idx[i];
			if (pixels[pixelId].samples < ns)
			{
				pixels[pixelId].samples++;
				// then, generate a new sample
				const unsigned int x = pixelId % nx;
				const unsigned int y = ny - 1 - (pixelId / nx);
				generate_ray(cam, h_rays[ray_idx], x, y, nx, ny);
				h_sample_colors[ray_idx] = vec3(1, 1, 1);
				++ray_idx;
				++sampled;
			}
		}
	} while (ray_idx < numpixels() && sampled > 0);
	generate += clock() - start;
	num_rays = ray_idx;
}

void renderer::destroy() {
	// Free device global memory
	err(cudaFree(d_scene), "free device d_scene");
	err(cudaFree(d_rays), "free device d_rays");
	err(cudaFree(d_hits), "free device d_hits");

	// Free host memory
	free(h_hits);
	delete[] pix_array;
}