#include <iostream>
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
	int ns = 2;
	int max_depth = 50;
	hitable *world = random_scene();

    vec3 lookfrom(13,2,3);
    vec3 lookat(0,0,0);
    float dist_to_focus = 10.0;
    float aperture = 0.1;

    camera cam(lookfrom, lookat, vec3(0,1,0), 20, float(nx)/float(ny), aperture, dist_to_focus);

    unsigned int num_rays = nx*ny*ns;
    ray *rays[num_rays];
    unsigned int ray_sample_ids[num_rays];
    vec3 *sample_colors[num_rays];

    // generate all camera rays
    unsigned int ray_idx = 0;
    for (int j = ny-1; j >= 0; j--)
    {
		for (int i = 0; i < nx; ++i)
		{
			for (int s = 0; s < ns; ++s, ++ray_idx)
			{
				float u = float(i + drand48()) / float(nx);
				float v = float(j + drand48()) / float(ny);
				rays[ray_idx] = new ray();
				cam.get_ray(u, v, *(rays[ray_idx]));
				ray_sample_ids[ray_idx] = ray_idx;

				sample_colors[ray_idx] = new vec3(1, 1, 1);
			}
		}
    }

    // compute ray-world intersections
    unsigned int depth = 0;
    while (depth < max_depth && num_rays > 0)
    {
        ray_idx = 0;
        for (unsigned int i = 0; i < num_rays; i++)
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

    // combine pixels samples and generate final image
	cout << "P3\n" << nx << " " << ny << "\n255\n";
    unsigned int sample_idx = 0;
    for (int j = ny-1; j >= 0; j--)
    {
		for (int i = 0; i < nx; ++i)
		{
			vec3 col(0, 0, 0);
			for (int s = 0; s < ns; ++s, ++sample_idx)
			{
				col += *sample_colors[sample_idx];
			}

			col /= float(ns);
			col = vec3( sqrtf(col[0]), sqrtf(col[1]), sqrtf(col[2]) );
			int ir = int(255.99*col.r());
			int ig = int(255.99*col.g());
			int ib = int(255.99*col.b());

			cout << ir << " " << ig << " " << ib << "\n";
		}
    }
	return 0;
}
