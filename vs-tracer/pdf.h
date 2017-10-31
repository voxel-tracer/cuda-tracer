#pragma once

#include <limits.h>
#include "onb.h"
#include "sphere.h"

enum pdf_type {
	COSINE,
	HITABLE,
	MIXTURE
};

struct pdf {
	pdf(const float3& w): type(COSINE), uvw(w), o() {} // cosine density pdf
	pdf(const sphere *p, const float3& origin): type(HITABLE), ptr(p), o(origin) {} // hitable pdf
	pdf(pdf *p0, pdf *p1) : type(MIXTURE), o() { p[0] = p0; p[1] = p1; } // mixture pdf

	float value(const float3& direction) const;
	float3 generate() const;

	const pdf_type type;

	// COSINE
	const onb uvw;
	// HITABLE
	const float3 o;
	const sphere *ptr;
	// MIXTURE
	pdf *p[2];
};

pdf* make_cosine_pdf(const float3& w);
pdf* make_hitable_pdf(const sphere *p, const float3& origin);
pdf* make_mixture_pdf(pdf *p0, pdf *p1);