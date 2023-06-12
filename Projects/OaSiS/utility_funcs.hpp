#ifndef UTILITY_FUNCS_HPP
#define UTILITY_FUNCS_HPP
#include "settings.h"

namespace mn {

//TODO: Maybe create parameters for some of this magic numbers
//NOLINTBEGIN(readability-magic-numbers) Magic numbers are formula-specific
/// assume p is already within kernel range [-1.5, 1.5]
template <typename T, size_t Degree>
constexpr vec<T, Degree + 1> bspline_weight(T p);

//Copied from IQ-MPM repro
template<>
constexpr vec<float, 1> bspline_weight<float, 0>(float p) 
{
	vec<float, 1> dw {1.0f};
	return dw;
}

template<>
constexpr vec<float, 2> bspline_weight<float, 1>(float p) 
{
	vec<float, 2> dw {0.0f, 0.0f};
	float dx = p * config::G_DX_INV;///< normalized offset
	
	dw[0] = 1 - dx;
	dw[1] = dx;
	
	return dw;
}

template<>
constexpr vec<float, 3> bspline_weight<float, 2>(float p) 
{
	vec<float, 3> dw {0.0f, 0.0f, 0.0f};
	float d0 = p * config::G_DX_INV;///< normalized offset
	
	float z = 1.5f - d0;
	float z2 = z * z;
	dw[0] = 0.5f * z2;
	float d1 = d0 - 1.0f;
	dw[1] = 0.75f - d1 * d1;
	float d2 = 1.0f - d1;
	float zz = 1.5f - d2;
	float zz2 = zz * zz;
	dw[2] = 0.5f * zz2;
	
	return dw;
}

template<>
constexpr vec<float, 4> bspline_weight<float, 3>(float p) 
{
	vec<float, 4> dw {0.0f, 0.0f, 0.0f, 0.0f};
	float d0 = p * config::G_DX_INV;///< normalized offset

	float z = 2.0f - d0;
	float z3 = z * z * z;
	dw[0] = (1.0f / 6.0f) * z3;
	float d1 = d0 - 1.0f;
	float zz2 = d1 * d1;
	dw[1] = (0.5f * d1 - 1.0f) * zz2 + 2.0f / 3.0f;
	float d2 = 1.0f - d1;
	float zzz2 = d2 * d2;
	dw[2] = (0.5f * d2 - 1.0f) * zzz2 + 2.0f / 3.0f;
	float d3 = 1.0f + d2;
	float zzzz = 2.0f - d3;
	float zzzz3 = zzzz * zzzz * zzzz;
	dw[3] = (1.0f / 6.0f) * zzzz3;
	
	return dw;
}

constexpr ivec3 get_cell_id(const std::array<float, 3>& position, const std::array<float, 3>& relative_offset) {
	return ivec3(static_cast<int>(std::lround(position[0] * config::G_DX_INV - relative_offset[0] * config::G_BLOCKSIZE)), static_cast<int>(std::lround(position[1] * config::G_DX_INV - relative_offset[1] * config::G_BLOCKSIZE)), static_cast<int>(std::lround(position[2] * config::G_DX_INV - relative_offset[2] * config::G_BLOCKSIZE)));
}

constexpr int dir_offset(const std::array<int, 3>& d) {
	return (d[0] + 1) * 9 + (d[1] + 1) * 3 + d[2] + 1;
}
constexpr void dir_components(int dir, std::array<int, 3>& d) {
	d[2] = (dir % 3) - 1;
	d[1] = ((dir / 3) % 3) - 1;
	d[0] = ((dir / 9) % 3) - 1;
}
//NOLINTEND(readability-magic-numbers) Magic numbers are formula-specific

//NOLINTBEGIN(readability-magic-numbers) Magic numbers are formula-specific
constexpr Duration compute_dt(float max_vel, const Duration cur_time, const Duration next_time, const Duration dt_default) noexcept {
	//Choose dt such that particles with maximum velocity cannot move more than G_DX * CFL
	//This ensures CFL condition is satisfied
	Duration dt = dt_default;
	if(max_vel > 0.0f) {
		const Duration new_dt(config::G_DX * config::CFL / max_vel);
		dt = std::min(new_dt, dt);
	}

	//If next_time - cur_time is smaller as current dt, use this.
	dt = std::min(dt, next_time - cur_time);

	return dt;
}
//NOLINTEND(readability-magic-numbers) Magic numbers are formula-specific

//Copied from https://stackoverflow.com/questions/8622256/in-c11-is-sqrt-defined-as-constexpr
namespace Detail
{
	template<typename T>
    constexpr T sqrtNewtonRaphson(T x, T curr, T prev)
    {
        return curr == prev
            ? curr
            : sqrtNewtonRaphson(x, static_cast<T>(0.5) * (curr + x / curr), curr);
    }
}

/*
* Constexpr version of the square root
* Return value:
*   - For a finite and non-negative value of "x", returns an approximation for the square root of "x"
*   - Otherwise, returns NaN
*/
template<typename T>
constexpr T const_sqrt(T x)
{
    return x >= static_cast<T>(0.0) && x < std::numeric_limits<T>::infinity()
        ? Detail::sqrtNewtonRaphson(x, x, static_cast<T>(0.0))
        : std::numeric_limits<T>::quiet_NaN();
}

}// namespace mn

#endif