#ifndef HITABLE_LIST_H_
#define HITABLE_LIST_H_

#include "sphere.h"

struct hitable_list {
	hitable_list(const sphere **l, int ls, const material **m, int ms): list(l), list_size(ls), materials(m), material_size(ms) {}

	const sphere **list;
	const int list_size;
	const material **materials;
	const int material_size;
};

#endif /* HITABLE_LIST_H_ */
