#include <iostream>
#include <float.h>
#include <fstream>
#include <algorithm>
#include <ctime>
#include <SDL.h>

// For the CUDA runtime routines (prefixed with "cuda_")
//#include <cuda_runtime.h>
//#include <cuda_profiler_api.h>
//#include <helper_cuda.h>

#include "sphere.h"
#include "hitable_list.h"
#include "renderer.h"

//#define DBG_ID (150*600+300)
char buffer[100];

using namespace std;

hitable_list*
random_scene()
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
	list[i++] = new sphere(vec3(10, 10, 10), 0.5, make_diffuse_light(vec3(100, 100, 100)));

    return new hitable_list(list,i);
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

/**
 * Host main routine
 */
int main(int argc, char** argv)
{
	SDL_Init(SDL_INIT_VIDEO);

	const unsigned int scene_size = 500;

	printf("preparing renderer...\n");

	const int nx = 600;
	const int ny = 300;
	const int ns = 10000;
	hitable_list *world = random_scene();

    camera *cam = init_camera(nx, ny);
	float theta = 80 * M_PI / 180;
	float phi = 45 * M_PI / 180;
	const float delta = 1 * M_PI / 180;
	cam->look_from(theta, phi);
	renderer r(cam, world, nx, ny, ns, 50, 0.001);

    clock_t begin = clock();

	r.prepare_kernel();

	SDL_Window* screen = SDL_CreateWindow("Voxel Tracer (rendering)", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, nx, ny, 0);
	SDL_Renderer* renderer = SDL_CreateRenderer(screen, -1, 0);
	SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STATIC, nx, ny);
	unsigned int* pix_array = new unsigned int[r.numpixels()];

	bool quit = false;
	bool mouse_drag = false;
	SDL_Event event;
	//do {
	//	SDL_WaitEvent(&event);
	//} while (event.type != SDL_KEYDOWN);

	unsigned int iteration = 0;
	unsigned int total_rays = 0;
	while (!quit)
	{
		if (r.numrays() > 0)
		{
			total_rays += r.numrays();
			//if (iteration % 1 == 0)
			//{
			//	//cout << "iteration " << iteration << "(" << num_rays << " rays)\n";
			//	cout << "iteration " << iteration << "(" << r.numrays() << " rays)\r";
			//	cout.flush();
			//}

			// compute ray-world intersections
			r.run_kernel();

			// compact active rays
			r.compact_rays();
		}

		// update pixels
		for (int x = 0; x < nx; x++)
		{
			for (int y = 0; y < ny; y++)
			{
				vec3 col = r.get_pixel_color(x, y);
				col = vec3(sqrtf(col[0]), sqrtf(col[1]), sqrtf(col[2]));
				int ir = min(255, int(255.99*col.r()));
				int ig = min(255, int(255.99*col.g()));
				int ib = min(255, int(255.99*col.b()));
				pix_array[(ny - 1 - y)*nx + x] = (ir << 16) + (ig << 8) + ib;
			}
		}
		SDL_UpdateTexture(texture, NULL, pix_array, nx * sizeof(unsigned int));
		SDL_RenderClear(renderer);
		SDL_RenderCopy(renderer, texture, NULL, NULL);
		SDL_RenderPresent(renderer);

		++iteration;

		while (SDL_PollEvent(&event)) {
			switch (event.type) {
			case SDL_QUIT:
				quit = true;
				break;
			case SDL_MOUSEMOTION:
				if (mouse_drag) {
					int mx = event.motion.xrel;
					int my = event.motion.yrel;
					theta += -my*delta; 
					if (theta < delta) theta = delta;
					if (theta > (M_PI_2 - delta)) theta = M_PI_2 - delta;
					phi += -mx*delta;
					cam->look_from(theta, phi);
					r.update_camera();
				}
				break;
			case SDL_MOUSEBUTTONDOWN:
				mouse_drag = true;
				break;
			case SDL_MOUSEBUTTONUP:
				mouse_drag = false;
				break;
			}
		}
	}

    clock_t end = clock();
	printf("rendering %d rays, duration %.2f seconds\nkernel %.2f seconds\ngenerate %.2f seconds\ncompact %.2f seconds\n",
		total_rays,
		double(end - begin) / CLOCKS_PER_SEC,
		double(r.kernel) / CLOCKS_PER_SEC,
		double(r.generate) / CLOCKS_PER_SEC,
		double(r.compact) / CLOCKS_PER_SEC);
	cout.flush();

	SDL_SetWindowTitle(screen, "Voxel Tracer");

	while (!quit) {
		SDL_WaitEvent(&event);
		quit = event.type == SDL_QUIT;
	}

	SDL_DestroyTexture(texture);
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(screen);
	delete[] pix_array;
/*
    // generate final image
    ofstream image;
    image.open("picture.ppm");
	image << "P3\n" << nx << " " << ny << "\n255\n";
    unsigned int sample_idx = 0;
    for (int j = ny-1; j >= 0; j--)
    {
		for (int i = 0; i < nx; ++i, sample_idx++)
		{
			vec3 col = r.get_pixel_color(i, j);
			col = vec3( sqrtf(col[0]), sqrtf(col[1]), sqrtf(col[2]) );
			int ir = int(255.99*col.r());
			int ig = int(255.99*col.g());
			int ib = int(255.99*col.b());

			image << ir << " " << ig << " " << ib << "\n";
		}
    }
*/	
	SDL_Quit();
	r.destroy();

    return 0;
}

