/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <iostream>
#include <float.h>
#include <fstream>
#include <ctime>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>

#include "sphere.h"
#include "camera.h"
#include "hitable_list.h"

//#define DBG_ID (150*600+300)
char buffer[100];

using namespace std;

struct cu_sphere {
	float3 center;
	float radius;
};

struct cu_ray {
	float3 origin;
	float3 direction;
};

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

cu_ray*
generate_rays(const camera* cam, cu_ray* rays, const unsigned int nx, const unsigned int ny)
{
	unsigned int ray_idx = 0;
	ray r;
    for (int j = ny-1; j >= 0; j--)
    {
		for (int i = 0; i < nx; ++i, ++ray_idx)
		{
			float u = float(i + drand48()) / float(nx);
			float v = float(j + drand48()) / float(ny);
			cam->get_ray(u, v, r);
			rays[ray_idx].origin = make_float3(r.A.x(), r.A.y(), r.A.z());
			rays[ray_idx].direction = make_float3(r.B.x(), r.B.y(), r.B.z());
		}
    }

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

bool color(const unsigned int sample_id, const cu_ray& cu_r, const cu_hit& hit, const hitable_list *world, vec3& sample_clr, cu_ray& scattered) {
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
	if (scatter(*rec.mat_ptr, r, rec, attenuation, r)) {
		scattered.origin = r.origin().to_float3();
		scattered.direction = r.direction().to_float3();

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
	const int ns = 100;
	int max_depth = 50;
    const hitable_list *world = random_scene();

	cu_sphere *h_scene = init_cu_scene(world);
    const camera *cam = init_camera(nx, ny);
    const unsigned int all_rays = nx*ny;
	cu_ray *h_rays = new cu_ray[all_rays]; //(cu_ray*) malloc(all_rays*sizeof(cu_ray));
	vec3 *h_colors = new vec3[all_rays]; //(vec3**) malloc(all_rays*sizeof(vec3*));
	unsigned int *h_ray_sample_ids = new unsigned int[all_rays]; //(unsigned int *) malloc(all_rays*sizeof(unsigned int));
	vec3 *h_sample_colors = new vec3[all_rays]; //(vec3**) malloc(all_rays* sizeof(vec3*));

    // init cumulated colors
    //for (int i = 0; i < all_rays; i++)
    //		h_colors[i] = new vec3(0, 0, 0);

    // init sample colors
    //for (int i = 0; i < all_rays; i++)
    //		h_sample_colors[i] = new vec3(1, 1, 1);

	cu_hit *h_hits = new cu_hit[all_rays]; //(cu_hit*) malloc(all_rays*sizeof(cu_hit));

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
    for (unsigned int s = 0; s < ns; ++s)
    {
		if (s % 10 == 0)
		{
			cout << "sample " << s << "/" << ns << "\r";
			cout.flush();
		}

		// reset temporary variables
		//cout << "reset vars\n";
        for (int i = 0; i < all_rays; i++)
        {
			h_sample_colors[i] = vec3(1, 1, 1);
        	h_ray_sample_ids[i] = i;
        }

		//cout << "generate rays\n";
		clock_t start = clock();
        generate_rays(cam, h_rays, nx, ny);
		generate += clock() - start;

        // compute ray-world intersections
        unsigned int depth = 0;
        unsigned int num_rays = all_rays;
        while (depth < max_depth && num_rays > 0)
        {
			clock_t start = clock();
			//cout << "copying rays to device...";
			err(cudaMemcpy(d_rays, h_rays, num_rays * sizeof(cu_ray), cudaMemcpyHostToDevice), "copy rays from host to device");
			//cout << "done\n";

            // Launch the CUDA Kernel
            int threadsPerBlock = 256;
            int blocksPerGrid =(num_rays + threadsPerBlock - 1) / threadsPerBlock;
			//cout << "launching kernel...";
            hit_scene<<<blocksPerGrid, threadsPerBlock>>>(d_rays, num_rays, d_scene, scene_size, 0.001, FLT_MAX, d_hits);
			err(cudaGetLastError(), "launch kernel");
			//cout << "done\n";
            // Copy the device result in device memory to the host result vector
			err(cudaMemcpy(h_hits, d_hits, num_rays * sizeof(cu_hit), cudaMemcpyDeviceToHost), "copy results from device to host");
			kernel += clock() - start;

            // compact active rays
        	unsigned int ray_idx = 0;
            for (unsigned int i = 0; i < num_rays; ++i)
            {
            		if (color(h_ray_sample_ids[i], h_rays[i], h_hits[i], world, h_sample_colors[h_ray_sample_ids[i]], h_rays[ray_idx]))
            		{
            			h_ray_sample_ids[ray_idx] = h_ray_sample_ids[i];
            			++ray_idx;
            		}
            }
            num_rays = ray_idx;
            ++depth;
        }

        // cumulate sample colors
        for (unsigned int i = 0; i < all_rays; ++i)
        		h_colors[i] += h_sample_colors[i];
    }

    clock_t end = clock();
    printf("rendering duration %.2f seconds\nkernel %.2f seconds\ngenerate %.2f seconds\n", 
		double(end - begin) / CLOCKS_PER_SEC, 
		double(kernel) / CLOCKS_PER_SEC,
		double(generate) / CLOCKS_PER_SEC);

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

