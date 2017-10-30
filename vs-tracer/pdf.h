#pragma once

#include <limits.h>
#include "onb.h"
#include "sphere.h"

class pdf {
public:
	virtual float value(const float3& direction) const = 0;
	virtual	float3 generate() const = 0;
};

class cosine_density : public pdf {
public:
	cosine_density(const float3& w) { uvw.build_from_w(w); }
	virtual float value(const float3& direction) const {
		float cosine = dot(normalize(direction), uvw.w());
		if (cosine > 0)
			return cosine / M_PI;
		return 0;
	}
	virtual float3 generate() const {
		return uvw.local(random_cosine_direction());
	}

	onb uvw;
};

class hitable_pdf :public pdf {
public:
	hitable_pdf(const sphere *p, const float3& origin) : ptr(p), o(origin) {}
	virtual float value(const float3& direction) const {
		return ptr->pdf_value(o, direction);
	}
	virtual float3 generate() const {
		return ptr->random(o);
	}

	float3 o;
	const sphere *ptr;
};

class mixture_pdf :public pdf {
public:
	mixture_pdf(pdf *p0, pdf *p1) { p[0] = p0; p[1] = p1; }
	virtual float value(const float3& direction) const {
		return 0.5*p[0]->value(direction) + 0.5*p[1]->value(direction);
	}
	virtual float3 generate() const {
		if (drand48() < 0.5)
			return p[0]->generate();
		return p[1]->generate();
	}
	pdf *p[2];
};