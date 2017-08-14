#include <iostream>
#include <fstream>
#include <ctime>

#include "hitable_list.h"
#include "sphere.h"
#include "ray.h"
#include "camera.h"
#include "material.h"
#include "utils.h"

using namespace std;

hitable *random_scene() {
    int n = 500;
    hitable **list = new hitable*[n+1];
    list[0] =  new sphere(vec3(0,-1000,0), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));
    int i = 1;
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose_mat = drand48();
            vec3 center(a+0.9*drand48(),0.2,b+0.9*drand48());
            if ((center-vec3(4,0.2,0)).length() > 0.9) {
                if (choose_mat < 0.8) {  // diffuse
                    list[i++] = new sphere(center, 0.2, new lambertian(vec3(drand48()*drand48(), drand48()*drand48(), drand48()*drand48())));
                }
                else if (choose_mat < 0.95) { // metal
                    list[i++] = new sphere(center, 0.2,
                            new metal(vec3(0.5*(1 + drand48()), 0.5*(1 + drand48()), 0.5*(1 + drand48())),  0.5*drand48()));
                }
                else {  // glass
                    list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
    }

    list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
    list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
    list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));

    return new hitable_list(list,i);
}

bool color(const ray& r, hitable *world, vec3& sample_clr, ray& scattered) {
	hit_record rec;
	if (!world->hit(r, 0.001, MAXFLOAT, rec)) {
		// no intersection with spheres, return sky color
		vec3 unit_direction = unit_vector(r.direction());
		float t = 0.5*(unit_direction.y() + 1.0);
		sample_clr *= (1 - t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
		return false;
	}

	vec3 attenuation;
	if (rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {
		sample_clr *= attenuation;
		return true;
	}

	sample_clr = vec3(0, 0, 0);
	return false;
}

int main() {
	int nx = 600;
	int ny = 300;
	int ns = 100;
	int max_depth = 50;
	hitable *world = random_scene();

    // init camera
    vec3 lookfrom(13,2,3);
    vec3 lookat(0,0,0);
    float dist_to_focus = 10.0;
    float aperture = 0.1;
    camera cam(lookfrom, lookat, vec3(0,1,0), 20, float(nx)/float(ny), aperture, dist_to_focus);

    const unsigned int all_rays = nx*ny;
    ray *rays[all_rays];
    vec3 *colors[all_rays];
    unsigned int ray_sample_ids[all_rays];
    vec3 *sample_colors[all_rays];

    // init cumulated colors
    for (int i = 0; i < all_rays; i++)
    		colors[i] = new vec3(0, 0, 0);

    // init sample colors
    for (int i = 0; i < all_rays; i++)
    		sample_colors[i] = new vec3(1, 1, 1);

    clock_t begin = clock();
    for (unsigned int s = 0; s < ns; ++s)
    {
    		cout << "sample " << s << "/" << ns << "\r";
    		cout.flush();

    		// reset samples
        for (int i = 0; i < all_rays; i++)
        		*(sample_colors[i]) = vec3(1, 1, 1);

        // generate all camera rays, but just for one sample per pixel
        unsigned int ray_idx = 0;
        for (int j = ny-1; j >= 0; j--)
        {
			for (int i = 0; i < nx; ++i, ++ray_idx)
			{
				float u = float(i + drand48()) / float(nx);
				float v = float(j + drand48()) / float(ny);
				rays[ray_idx] = new ray();
				cam.get_ray(u, v, *(rays[ray_idx]));
				ray_sample_ids[ray_idx] = ray_idx;
			}
        }

        // compute ray-world intersections
        unsigned int depth = 0;
        unsigned int num_rays = all_rays;
        while (depth < max_depth && num_rays > 0)
        {
            ray_idx = 0;
            for (unsigned int i = 0; i < num_rays; ++i)
            {
            		if (color(*rays[i], world, *sample_colors[ray_sample_ids[i]], *rays[ray_idx]))
            		{
            			ray_sample_ids[ray_idx] = ray_sample_ids[i];
            			++ray_idx;
            		}
            }
            num_rays = ray_idx;
            ++depth;
        }

        // cumulate sample colors
        for (unsigned int i = 0; i < all_rays; ++i)
        		*(colors[i]) += *(sample_colors[i]);
    }

    clock_t end = clock();
    printf("rendering duration %.2f seconds", double(end - begin) / CLOCKS_PER_SEC);

    // generate final image
    ofstream image;
    image.open("picture.ppm");
	image << "P3\n" << nx << " " << ny << "\n255\n";
    unsigned int sample_idx = 0;
    for (int j = ny-1; j >= 0; j--)
    {
		for (int i = 0; i < nx; ++i, sample_idx++)
		{
			vec3 col = *(colors[sample_idx]) / float(ns);
			col = vec3( sqrtf(col[0]), sqrtf(col[1]), sqrtf(col[2]) );
			int ir = int(255.99*col.r());
			int ig = int(255.99*col.g());
			int ib = int(255.99*col.b());

			image << ir << " " << ig << " " << ib << "\n";
		}
    }
	return 0;
}
