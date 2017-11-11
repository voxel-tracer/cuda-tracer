#include "pdf.h"

float pdf::value(const float3& direction) const {
	float cosine;
	switch (type) {
	case COSINE:
		cosine = dot(normalize(direction), uvw.w());
		if (cosine > 0)
			return cosine / M_PI;
		return 0;
	case HITABLE:
		return ptr->pdf_value(o, direction);
	case MIXTURE:
		return 0.5*p[0]->value(direction) + 0.5*p[1]->value(direction);
	}

	return 0;
}

__device__ float3 pdf::generate(seed_t seed) const {
	switch (type) {
	case COSINE:
		return uvw.local(random_cosine_direction(seed));
	case HITABLE:
		return ptr->random(seed, o);
	case MIXTURE:
		return (cu_drand48(seed) < 0.5) ? p[0]->generate(seed) : p[1]->generate(seed);
	}

	return make_float3(1, 0, 0); // we should throw an error
}

__host__ __device__ pdf* make_cosine_pdf(const float3& w) {
	return new pdf(w);
}

__host__ __device__ pdf* make_hitable_pdf(const sphere *p, const float3& origin) {
	return new pdf(p, origin);
}

__host__ __device__ pdf* make_mixture_pdf(pdf *p0, pdf *p1) {
	return new pdf(p0, p1);
}
