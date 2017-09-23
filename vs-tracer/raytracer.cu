#include <iostream>
#include <float.h>
#include <fstream>
#include <ctime>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>

#include "sphere.h"
#include "camera.h"
#include "hitable_list.h"

//#define DBG_ID (150*600+300)
char buffer[100];

using namespace std;

struct cu_hit {
	int hit_idx;
	float hit_t;
};

hitable_list *random_scene()
{
    int n = 500;
    hitable **list = new hitable*[n+1];
    list[0] =  new sphere(vec3(0,-1000,0), 1000, make_lambertian(vec3(0.5, 0.5, 0.5)));
    int i = 1;
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose_mat = drand48();
            vec3 center(a+0.9*drand48(),0.2,b+0.9*drand48());
            if ((center-vec3(4,0.2,0)).length() > 0.9) {
                if (choose_mat < 0.8) {  // diffuse
                    list[i++] = new sphere(center, 0.2, make_lambertian(vec3(drand48()*drand48(), drand48()*drand48(), drand48()*drand48())));
                }
                else if (choose_mat < 0.95) { // metal
                    list[i++] = new sphere(center, 0.2,
                            make_metal(vec3(0.5*(1 + drand48()), 0.5*(1 + drand48()), 0.5*(1 + drand48())),  0.5*drand48()));
                }
                else {  // glass
                    list[i++] = new sphere(center, 0.2, make_dielectric(1.5));
                }
            }
        }
    }

    list[i++] = new sphere(vec3(0, 1, 0), 1.0, make_dielectric(1.5));
    list[i++] = new sphere(vec3(-4, 1, 0), 1.0, make_lambertian(vec3(0.4, 0.2, 0.1)));
    list[i++] = new sphere(vec3(4, 1, 0), 1.0, make_metal(vec3(0.7, 0.6, 0.5), 0.0));

    return new hitable_list(list,i);
}

cu_sphere*
init_cu_scene(const hitable_list* world)
{
	const unsigned int size = world->list_size;
	cu_sphere* scene = (cu_sphere*) malloc(size*sizeof(cu_sphere));
	for (int i = 0; i < size; i++)
	{
		const sphere *s = (sphere*) world->list[i];
		scene[i].center = make_float3(s->center.x(), s->center.y(), s->center.z());
		scene[i].radius = s->radius;
	}

	return scene;
}

inline void generate_ray(const camera* cam, cu_ray& r, const unsigned int x, const unsigned int y, const unsigned int nx, const unsigned int ny)
{
	float u = float(x + drand48()) / float(nx);
	float v = float(y + drand48()) / float(ny);
	cam->get_ray(u, v, r);
}

cu_ray*
generate_rays(const camera* cam, cu_ray* rays, const unsigned int nx, const unsigned int ny)
{
	unsigned int ray_idx = 0;
    for (int j = ny-1; j >= 0; j--)
		for (int i = 0; i < nx; ++i, ++ray_idx)
			generate_ray(cam, rays[ray_idx], i, j, nx, ny);

    return rays;
}

camera*
init_camera(unsigned int nx, unsigned int ny)
{
    vec3 lookfrom(13,2,3);
    vec3 lookat(0,0,0);
    float dist_to_focus = 10.0;
    float aperture = 0.1;

    return new camera(lookfrom, lookat, vec3(0,1,0), 20, float(nx)/float(ny), aperture, dist_to_focus);
}

__global__ void
hit_scene(const cu_ray* rays, const unsigned int num_rays, const cu_sphere* scene, const unsigned int scene_size, float t_min, float t_max, cu_hit* hits)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_rays)
    	return;

    const cu_ray r = rays[i];
    const float3 ro = r.origin;
    const float3 rd = r.direction;

    float closest_hit = t_max;
    int hit_idx = -1;

//    if (i == DBG_ID) printf("hit_scene: ro = (%.2f, %.2f, %.2f) rd = (%.2f, %.2f, %.2f) \n", ro.x, ro.y, ro.z, rd.x, rd.y, rd.z);

    for (int s = 0; s < scene_size; s++)
    {
    	const cu_sphere sphere = scene[s];
    	const float3 sc = sphere.center;
    	const float sr = sphere.radius;

    	float3 oc = make_float3(ro.x-sc.x, ro.y-sc.y, ro.z-sc.z);
    	float a = rd.x*rd.x + rd.y*rd.y + rd.z*rd.z;
    	float b = 2.0f * (oc.x*rd.x + oc.y*rd.y + oc.z*rd.z);
    	float c = oc.x*oc.x + oc.y*oc.y + oc.z*oc.z - sr*sr;
    	float discriminant = b*b - 4*a*c;
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

//    if (i == DBG_ID) printf("hit_scene: hit_idx = %d, hit_t = %.2f\n", hit_idx, closest_hit);
}

bool color(const unsigned int sample_id, cu_ray& cu_r, const cu_hit& hit, const hitable_list *world, vec3& sample_clr, const unsigned int max_depth, cu_ray& scattered) {
	ray r = ray(vec3(cu_r.origin), vec3(cu_r.direction));

	if (hit.hit_idx == -1) {
		// no intersection with spheres, return sky color
		vec3 unit_direction = unit_vector(r.direction());
		float t = 0.5*(unit_direction.y() + 1.0);
		sample_clr *= (1 - t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
//		if (sample_id == DBG_ID) printf("no_hit: %s\n", sample_clr.to_string(buffer));
		return false;
	}

	hit_record rec;
	sphere *s = (sphere*) (world->list[hit.hit_idx]);
	rec.t = hit.hit_t;
	rec.p = r.point_at_parameter(hit.hit_t);
	rec.normal = (rec.p - s->center) / s->radius;
	rec.mat_ptr = s->mat_ptr;

	vec3 attenuation;
	if ((++cu_r.depth) <= max_depth && scatter(*rec.mat_ptr, r, rec, attenuation, r)) {
		scattered.origin = r.origin().to_float3();
		scattered.direction = r.direction().to_float3();
		scattered.depth = cu_r.depth;
		scattered.samples = cu_r.samples;

		sample_clr *= attenuation;
//		if (sample_id == DBG_ID) printf("scatter: %s\n", sample_clr.to_string(buffer));
		return true;
	}

	sample_clr = vec3(0, 0, 0);
//	if (sample_id == DBG_ID) printf("no_scatter: %s\n", sample_clr.to_string(buffer));
	return false;
}

void err(cudaError_t err, char *msg)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to %s (error code %s)!\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
/**
 * Host main routine
 */
int
main(void)
{
	const unsigned int scene_size = 500;

	printf("preparing renderer...\n");

	const int nx = 600;
	const int ny = 300;
	const int ns = 1000;
	const int max_depth = 50;
    const hitable_list *world = random_scene();

	cu_sphere *h_scene = init_cu_scene(world);
    const camera *cam = init_camera(nx, ny);
    const unsigned int all_rays = nx*ny;
	cu_ray *h_rays = new cu_ray[all_rays];
	vec3 *h_colors = new vec3[all_rays];
	unsigned int *h_ray_sample_ids = new unsigned int[all_rays];
	vec3 *h_sample_colors = new vec3[all_rays];

	cu_hit *h_hits = new cu_hit[all_rays];

    // allocate device memory for input
    cu_sphere *d_scene = NULL;
	err(cudaMalloc((void **)&d_scene, scene_size * sizeof(cu_sphere)), "allocate device d_scene");

    cu_ray *d_rays = NULL;
	err(cudaMalloc((void **)&d_rays, all_rays * sizeof(cu_ray)), "allocate device d_rays");

    cu_hit *d_hits = NULL;
	err(cudaMalloc((void **)&d_hits, all_rays * sizeof(cu_hit)), "allocate device d_hits");

    // Copy the host input in host memory to the device input in device memory
	err(cudaMemcpy(d_scene, h_scene, world->list_size * sizeof(cu_sphere), cudaMemcpyHostToDevice), "copy scene from host to device");

    clock_t begin = clock();
	clock_t kernel = 0;
	clock_t generate = 0;
	clock_t compact = 0;
	clock_t cumul = 0;

	// set temporary variables
	for (int i = 0; i < all_rays; i++)
	{
		h_sample_colors[i] = vec3(1, 1, 1);
		h_ray_sample_ids[i] = i;
	}

	// generate initial samples: one per pixel
	clock_t start = clock();
	generate_rays(cam, h_rays, nx, ny);
	generate += clock() - start;
	
	unsigned int num_rays = all_rays;
	unsigned int iteration = 0;
	while (num_rays > 0)
	{
		if (iteration % 100 == 0)
		{
			//cout << "iteration " << iteration << "(" << num_rays << " rays)\n";
			//cout << "iteration " << iteration << "\r";
			cout.flush();
		}

		// compute ray-world intersections
		cudaProfilerStart();
		clock_t start = clock();
		// copying rays to device
		err(cudaMemcpy(d_rays, h_rays, num_rays * sizeof(cu_ray), cudaMemcpyHostToDevice), "copy rays from host to device");

		// Launch the CUDA Kernel
		int threadsPerBlock = 128;
		int blocksPerGrid = (num_rays + threadsPerBlock - 1) / threadsPerBlock;
		hit_scene << <blocksPerGrid, threadsPerBlock >> >(d_rays, num_rays, d_scene, scene_size, 0.001, FLT_MAX, d_hits);
		err(cudaGetLastError(), "launch kernel");
		
		// Copy the results to host
		err(cudaMemcpy(h_hits, d_hits, num_rays * sizeof(cu_hit), cudaMemcpyDeviceToHost), "copy results from device to host");
		kernel += clock() - start;
		cudaProfilerStop();

		// compact active rays
		// whenever a rays becomes inactive, generate another sample until we hit spp limit
		start = clock();
		unsigned int ray_idx = 0;
		for (unsigned int i = 0; i < num_rays; ++i)
		{
			const unsigned int sampleId = h_ray_sample_ids[i];
			if (color(sampleId, h_rays[i], h_hits[i], world, h_sample_colors[sampleId], max_depth, h_rays[ray_idx]))
			{
				h_ray_sample_ids[ray_idx] = sampleId;
				++ray_idx;
			}
			else
			{
				// ray is no longer active, first cumulate its color
				h_colors[sampleId] += h_sample_colors[sampleId];
				if (++(h_rays[i].samples) < ns)
				{
					h_sample_colors[sampleId] = vec3(1, 1, 1);
					// then, generate a new sample
					const unsigned int x = sampleId % nx;
					const unsigned int y = ny - 1 - (sampleId / nx);
					generate_ray(cam, h_rays[ray_idx], x, y, nx, ny);
					h_rays[ray_idx].depth = 0;
					h_rays[ray_idx].samples = h_rays[i].samples;

					h_ray_sample_ids[ray_idx] = sampleId;
					++ray_idx;
				}
			}
		}
		compact += clock() - start;
		num_rays = ray_idx;

		++iteration;
	}

    clock_t end = clock();
    printf("rendering duration %.2f seconds\nkernel %.2f seconds\ngenerate %.2f seconds\ncompact %.2f seconds\ncumul %.2f seconds\n", 
		double(end - begin) / CLOCKS_PER_SEC, 
		double(kernel) / CLOCKS_PER_SEC,
		double(generate) / CLOCKS_PER_SEC,
		double(compact) / CLOCKS_PER_SEC,
		double(cumul) / CLOCKS_PER_SEC);

    // Free device global memory
	err(cudaFree(d_scene), "free device d_scene");
   	err(cudaFree(d_rays), "free device d_rays");
	err(cudaFree(d_hits), "free device d_hits");

    // Free host memory
    free(h_scene);
    free(h_hits);

    // generate final image
    ofstream image;
    image.open("picture.ppm");
	image << "P3\n" << nx << " " << ny << "\n255\n";
    unsigned int sample_idx = 0;
    for (int j = ny-1; j >= 0; j--)
    {
		for (int i = 0; i < nx; ++i, sample_idx++)
		{
			vec3 col = h_colors[sample_idx] / float(ns);
			col = vec3( sqrtf(col[0]), sqrtf(col[1]), sqrtf(col[2]) );
			int ir = int(255.99*col.r());
			int ig = int(255.99*col.g());
			int ib = int(255.99*col.b());

			image << ir << " " << ig << " " << ib << "\n";
		}
    }

	//cin.ignore();
    return 0;
}

