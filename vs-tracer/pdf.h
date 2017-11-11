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
	__host__ __device__ pdf(const float3& w): type(COSINE), uvw(w), o() {} // cosine density pdf
	__host__ __device__ pdf(const sphere *p, const float3& origin): type(HITABLE), ptr(p), o(origin) {} // hitable pdf
	__host__ __device__ pdf(pdf *p0, pdf *p1) : type(MIXTURE), o() { p[0] = p0; p[1] = p1; } // mixture pdf

	__device__ float value(const float3& direction) const;
	__device__ float3 generate(seed_t seed) const;
	__device__ pdf() : type(COSINE), o() {}

	const pdf_type type;

	// COSINE
	const onb uvw;
	// HITABLE
	const float3 o;
	const sphere *ptr;
	// MIXTURE
	pdf *p[2];
};

__host__ __device__ pdf* make_cosine_pdf(const float3& w);
__host__ __device__ pdf* make_hitable_pdf(const sphere *p, const float3& origin);
__host__ __device__ pdf* make_mixture_pdf(pdf *p0, pdf *p1);