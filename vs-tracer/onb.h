#pragma once

#include "utils.h"

struct onb {
	__device__ onb() {}
	__device__ onb(const float3& w) { build_from_w(w); }
	inline float3 operator[](int i) const { return axis[i]; }
	__device__ float3 u() const { return axis[0]; }
	__device__ float3 v() const { return axis[1]; }
	__device__ float3 w() const { return axis[2]; }
	__device__ float3 local(float a, float b, float c) const { return a*u() + b*v() + c*w(); }
	__device__ float3 local(const float3& a) const { return a.x*u() + a.y*v() + a.z*w(); }
	__device__ void build_from_w(const float3& n) {
		axis[2] = normalize(n);
		float3 a;
		if (fabsf(w().x) > 0.9999f)
			a = make_float3(0, 1, 0);
		else
			a = make_float3(1, 0, 0);
		axis[1] = normalize(cross(w(), a));
		axis[0] = cross(w(), v());
	}
	float3 axis[3];
};