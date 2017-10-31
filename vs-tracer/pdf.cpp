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

float3 pdf::generate() const {
	switch (type) {
	case COSINE:
		return uvw.local(random_cosine_direction());
	case HITABLE:
		return ptr->random(o);
	case MIXTURE:
		return (drand48() < 0.5) ? p[0]->generate() : p[1]->generate();
	}

	return make_float3(1, 0, 0); // we should throw an error
}

pdf* make_cosine_pdf(const float3& w) {
	return new pdf(w);
}

pdf* make_hitable_pdf(const sphere *p, const float3& origin) {
	return new pdf(p, origin);
}

pdf* make_mixture_pdf(pdf *p0, pdf *p1) {
	return new pdf(p0, p1);
}
