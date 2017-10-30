#ifndef HITABLE_LIST_H_
#define HITABLE_LIST_H_

#include "sphere.h"

struct hitable_list {
	hitable_list(sphere **l, int n) { list = l; list_size = n; }

	sphere **list;
	int list_size;
};

#endif /* HITABLE_LIST_H_ */
