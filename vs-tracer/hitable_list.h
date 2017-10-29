#ifndef HITABLE_LIST_H_
#define HITABLE_LIST_H_

#include "hitable.h"

class hitable_list: public hitable {
	public:
		hitable_list(hitable **l, int n) { list = l; list_size = n; }
		virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
			hit_record temp_rec;
			bool hit_anything = false;
			double closest_so_far = tmax;
			for (int i = 0; i < list_size; ++i) {
				if (list[i]->hit(r, tmin, closest_so_far, temp_rec)) {
					hit_anything = true;
					closest_so_far = temp_rec.t;
					rec = temp_rec;
				}
			}
			return hit_anything;
		}

		hitable **list;
		int list_size;
		virtual float pdf_value(const float3& o, const float3& v) const {
			float weight = 1.0 / list_size;
			float sum = 0;
			for (int i = 0; i < list_size; i++)
				sum += weight*list[i]->pdf_value(o, v);
			return sum;
		}
		virtual float3 random(const float3& o) const {
			int index = int(drand48() * list_size);
			return list[index]->random(o);
		}
};

#endif /* HITABLE_LIST_H_ */
