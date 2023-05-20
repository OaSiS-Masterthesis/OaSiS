#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include "Utility.h"
#include "Givens.cuh"

namespace mn {

//TODO: When we have c++20 use std::span

// matrix multiplification
// return mat1 * diagnal_matrix * mat2^T
// input diag: only have diagnal entries
template<typename T>
constexpr void matmul_mat_diag_mat_t_2d(std::array<T, 4>& out, const std::array<T, 4>& mat1, const std::array<T, 2>& diag, const std::array<T, 4>& mat2_t) {
	out[0] = mat1[0] * diag[0] * mat2_t[0] + mat1[2] * diag[1] * mat2_t[2];
	out[1] = mat1[1] * diag[0] * mat2_t[0] + mat1[3] * diag[1] * mat2_t[2];

	out[2] = mat1[0] * diag[0] * mat2_t[1] + mat1[2] * diag[1] * mat2_t[3];
	out[3] = mat1[1] * diag[0] * mat2_t[1] + mat1[3] * diag[1] * mat2_t[3];
}

/* matrix indexes
        mat1   |  diag   |  mat2^T
        0 3 6  |  0      |  0 1 2
        1 4 7  |    1    |  3 4 5
        2 5 8  |      2  |  6 7 8
*/
template<typename T>
constexpr void matmul_mat_diag_mat_t_3d(std::array<T, 9>& out, const std::array<T, 9>& mat1, const std::array<T, 3>& diag, const std::array<T, 9>& mat2_t) {
	out[0] = mat1[0] * diag[0] * mat2_t[0] + mat1[3] * diag[1] * mat2_t[3] + mat1[6] * diag[2] * mat2_t[6];
	out[1] = mat1[1] * diag[0] * mat2_t[0] + mat1[4] * diag[1] * mat2_t[3] + mat1[7] * diag[2] * mat2_t[6];
	out[2] = mat1[2] * diag[0] * mat2_t[0] + mat1[5] * diag[1] * mat2_t[3] + mat1[8] * diag[2] * mat2_t[6];

	out[3] = mat1[0] * diag[0] * mat2_t[1] + mat1[3] * diag[1] * mat2_t[4] + mat1[6] * diag[2] * mat2_t[7];
	out[4] = mat1[1] * diag[0] * mat2_t[1] + mat1[4] * diag[1] * mat2_t[4] + mat1[7] * diag[2] * mat2_t[7];
	out[5] = mat1[2] * diag[0] * mat2_t[1] + mat1[5] * diag[1] * mat2_t[4] + mat1[8] * diag[2] * mat2_t[7];

	out[6] = mat1[0] * diag[0] * mat2_t[2] + mat1[3] * diag[1] * mat2_t[5] + mat1[6] * diag[2] * mat2_t[8];
	out[7] = mat1[1] * diag[0] * mat2_t[2] + mat1[4] * diag[1] * mat2_t[5] + mat1[7] * diag[2] * mat2_t[8];
	out[8] = mat1[2] * diag[0] * mat2_t[2] + mat1[5] * diag[1] * mat2_t[5] + mat1[8] * diag[2] * mat2_t[8];
}

/* out = mat^T * vec */
template<typename T>
constexpr void mat_t_mul_vec_3d(std::array<T, 3>& out, const std::array<T, 9>& mat, const std::array<T, 3>& vec) {
	out[0] = mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2];
	out[1] = mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2];
	out[2] = mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2];
}

/* out = a x b (cross product)
 *      = {[a]_x} * b        */
template<typename T>
constexpr void vec_cross_mul_vec_3d(std::array<T, 3>& out, const std::array<T, 3>& a, const std::array<T, 3>& b) {
	out[0] = a[1] * b[2] + a[2] * b[1];
	out[1] = a[2] * b[0] + a[0] * b[2];
	out[2] = a[0] * b[1] + a[1] * b[0];
}

template<typename T>
constexpr void vec_cross_vec_3d(std::array<T, 3>& out, const std::array<T, 3>& a, const std::array<T, 3>& b) {
	out[0] = a[1] * b[2] - a[2] * b[1];
	out[1] = a[2] * b[0] - a[0] * b[2];
	out[2] = a[0] * b[1] - a[1] * b[0];
}


template<typename T>
constexpr void matrix_cofactor_2d(const std::array<T, 4>& x, std::array<T, 4>& cof) {
	cof[0] = x[3];
	cof[1] = -x[2];
	cof[2] = -x[1];
	cof[3] = x[0];
}

template<typename T>
constexpr void matrix_cofactor_3d(const std::array<T, 9>& x, std::array<T, 9>& cof) {
	T cofactor11 = x[4] * x[8] - x[7] * x[5];
	T cofactor12 = x[7] * x[2] - x[1] * x[8];
	T cofactor13 = x[1] * x[5] - x[4] * x[2];
	cof[0]		 = cofactor11;
	cof[1]		 = cofactor12;
	cof[2]		 = cofactor13;
	cof[3]		 = x[6] * x[5] - x[3] * x[8];
	cof[4]		 = x[0] * x[8] - x[6] * x[2];
	cof[5]		 = x[3] * x[2] - x[0] * x[5];
	cof[6]		 = x[3] * x[7] - x[6] * x[4];
	cof[7]		 = x[6] * x[1] - x[0] * x[7];
	cof[8]		 = x[0] * x[4] - x[3] * x[1];
}

#if 0
template <typename T>
constexpr void matrix_inverse(const std::array<T, 9>& x, std::array<T, 9>& inv)
{
    T cofactor11 = x[4] * x[8] - x[7] * x[5];
	T cofactor12 = x[7] * x[2] - x[1] * x[8];
	T cofactor13 = x[1] * x[5] - x[4] * x[2];
    T determinant = x[0] * cofactor11 + x[3] * cofactor12 + x[6] * cofactor13;
    T s = 1 / determinant;
    inv[0] = s * cofactor11;
    inv[1] = s * cofactor12;
    inv[2] = s * cofactor13;
    inv[3] = s * x[6] * x[5] - s * x[3] * x[8];
    inv[4] = s * x[0] * x[8] - s * x[6] * x[2];
    inv[5] = s * x[3] * x[2] - s * x[0] * x[5];
    inv[6] = s * x[3] * x[7] - s * x[6] * x[4];
    inv[7] = s * x[6] * x[1] - s * x[0] * x[7];
    inv[8] = s * x[0] * x[4] - s * x[3] * x[1];
}
#endif

//Solves ax = b for x
template <typename T>
__forceinline__ __host__ __device__ void solve_linear_system(const std::array<T, 9>& a, std::array<T, 3>& x, const std::array<T, 3>& b){
	//Calculate QR
	std::array<T, 9> r = a;
	
	const mn::math::GivensRotation rot0(r[1], r[2], 1, 2);
	rot0.template mat_rotation<3, T>(r);
	const mn::math::GivensRotation rot1(r[0], r[1], 0, 1);
	rot1.template mat_rotation<3, T>(r);
	const mn::math::GivensRotation rot2(r[4], r[5], 1, 2);
	rot2.template mat_rotation<3, T>(r);
	
	std::array<T, 9> rot0_mat;
	std::array<T, 9> rot1_mat;
	std::array<T, 9> rot2_mat;
	rot0.template fill<3, T>(rot0_mat);
	rot1.template fill<3, T>(rot1_mat);
	rot2.template fill<3, T>(rot2_mat);
	
	std::array<T, 9> q_transpose_tmp;
	matrix_matrix_multiplication_3d(rot1_mat, rot0_mat, q_transpose_tmp);
	std::array<T, 9> q_transpose;
	matrix_matrix_multiplication_3d(rot2_mat, q_transpose_tmp, q_transpose);
	
	//Calculate y
	std::array<T, 3> y;
	matrix_vector_multiplication_3d(q_transpose, b, y);
	
	//Back substitution
	x[2] = y[2] / (std::abs(r[8]) < 1e-4 ? static_cast<T>(1.0) : r[8]);
	x[1] = (y[1] - x[2] * r[7]) / (std::abs(r[4]) < 1e-4 ? static_cast<T>(1.0) : r[4]);
	x[0] = (y[0] - x[2] * r[6] - x[1] * r[3]) / (std::abs(r[0]) < 1e-4 ? static_cast<T>(1.0) : r[0]);
}

template <typename T, std::size_t Dim>
__forceinline__ __host__ __device__ void solve_linear_system(const std::array<T, Dim * Dim>& a, std::array<T, Dim>& x, const std::array<T, Dim>& b){
	constexpr size_t num_rotations = (Dim * (Dim - 1))/2;
	
	//Calculate QR
	std::array<T, Dim * Dim> r = a;
	
	std::array<T, Dim * Dim> rot_mat[num_rotations];
	size_t index = 0;
	for(size_t i = 0; i < Dim - 1; ++i){
		for(size_t j = 0; j < (Dim - i - 1); ++j){
			const T row0 = Dim - j - 1;
			const T row1 = Dim - j;
			
			const mn::math::GivensRotation rot(r[Dim * i + row0], r[row1], row0, row1);
			rot.template mat_rotation<Dim, T>(r);
			rot.template fill<3, T>(rot_mat[index++]);
		}
	}
	
	std::array<T, Dim * Dim> q_transpose_tmp = rot_mat[0];
	std::array<T, Dim * Dim> q_transpose;
	for(size_t i = 1; i < num_rotations; ++i){
		matrix_matrix_multiplication(rot_mat[i], q_transpose_tmp, q_transpose);
		q_transpose_tmp = q_transpose;
	}
	
	//Calculate y
	std::array<T, Dim> y;
	matrix_vector_multiplication(q_transpose, b, y);
	
	//Back substitution
	for(size_t i = Dim - 1; (i + 1) >= 1; ++i){
		T summed_y = y[i];
		for(size_t j = Dim - 1; j > i; ++j){
			summed_y -= x[j] * r[j * Dim + i];
		}
		x[i] = summed_y / (std::abs(r[i * Dim + i]) < 1e-4 ? static_cast<T>(1.0) : r[i * Dim + i]);
	}
}

//Not tested
/*
//Matrix a should be definite for stability
template <typename T, std::size_t Dim>
__forceinline__ __host__ __device__ void cholesky_decomposition_definite(const std::array<T, Dim * Dim>& a, std::array<T, Dim * Dim>& r, std::array<T, Dim * Dim>& r_t){
	//Calculate decomposition
	std::array<T, Dim> d;
	std::array<T, Dim * Dim> l;
	for(size_t i = 0; i < Dim - 1; ++i){
		d[i] = a[Dim * i + i];
		if(i > 0){
			for(size_t k = 0; k < i - 1; ++k){
				d[i] += l[Dim * i + k] * l[Dim * i + k] * d[k];
			}
		}
		
		for(size_t j = i + 1; j < Dim - 1; ++j){
			l[Dim * i + j] = a[Dim * i + j] / d[i];
			if(i > 0){
				for(size_t k = 0; k < i - 1; ++k){
					l[Dim * i + j] += l[Dim * j + k] * l[Dim * i + k] * d[k] / d[i];
				}
			}
		}
	}
	
	//Fill matrices
	for(size_t i = 0; i < Dim - 1; ++i){
		for(size_t j = 0; j < Dim - 1; ++j){
			if(i < j){
				r[Dim * i + j] = l[Dim * i + j];
				r_t[Dim * i + j] = 0.0f;
			}else if(i < j){
				r[Dim * i + j] = 0.0f;
				r_t[Dim * i + j] = l[Dim * j + i];
			}else{//i == j
				r[Dim * i + j] = d[i];
				r_t[Dim * i + j] = d[i];
			}
		}
	}
}
*/

//Not needed, not finished, not tested
/*
template <typename T, std::size_t Dim>
__forceinline__ __host__ __device__ void minimize(const std::array<T, Dim * Dim>& a, std::array<T, Dim>& x, const std::array<T, Dim>& b, const T t, const T epsilon){
	//Calculate delta x
	solve_linear_system(a, x, b);
	
	x += t * delta_x;
}

//https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
//NOTE: Remember to set x to a feasible staring point
//NOTE: This assumes that g and gradient of f are linear combinations Ax + b
template <typename T, std::size_t Dim, std::size_t NumberOfInequalityConstraints>
__forceinline__ __host__ __device__ void constraint_convex_optimization(const std::array<T, Dim * Dim>& a, const std::array<T, Dim>& b, const std::array<T, Dim * Dim>& gradient_f_mat, const std::array<T, Dim>& gradient_f_const, const std::array<T, Dim * Dim>& g_mat[NumberOfInequalityConstraints], const std::array<T, Dim>& g_const[NumberOfInequalityConstraints], const std::array<T, Dim>& x_0, std::array<T, Dim>& x, const T t_0, const T mu, const T epsilon){
	//Init x and t
	x = x_0;
	T t = t_0;
	
	//Calculate second gradients
	std::array<T, Dim>& second_gradient_f_const;
	std::array<T, Dim>& gradient_g_const[NumberOfInequalityConstraints];
	for(size_t i = 0; i < Dim; ++i){
		second_gradient_f_const[i] = gradient_f_const[i * Dim + i];
		for(size_t j = 0; j < NumberOfInequalityConstraints; ++j){
			gradient_g_const[j][i] = g_mat[j][i * Dim + i];
		}
	}
	
	//Break if condition is reached
	while((static_cast<T>(NumberOfInequalityConstraints) / t) < epsilon){
		//Centering
		
		//H = t * second_gradient_f_const + sum(1/g(x)^2 * gradient_g * gradient_g^T) + sum(1/-g(x) * gradient_g^2)
		//g = t * gradient_f + sum(1/-g * gradient_g)
		std::array<T, Dim> h;
		std::array<T, Dim> g;
		for(size_t i = 0; i < Dim; ++i){
			H[i] = t * second_gradient_f_const[i];
			g[i] = t * gradient_f_const * x;
			for(size_t k = 0; k < NumberOfInequalityConstraints; ++k){
				const T g_k = (g_mat[k] * x);
				//FIXME: Gradient^T correct?
				H[i] += static_cast<T>(1.0) / (g_k * g_k) * (gradient_g_const[i] * gradient_g_const[i]);
				g[i] -= static_cast<T>(1.0) / g_k * gradient_g_const[i];
			}
		}
		
		// H A^T  x  = -g
		// A 0    v     0
		std::array<T, (2 * Dim) * (Dim + 1)> min_a;
		std::array<T, 2 * Dim> min_b;
		std::array<T, 2 * Dim> min_x;
		for(size_t i = 0; i < Dim; ++i){
			min_a[i] = H[i];//Store H
			min_b[i] = -g[i];//Store -g
			for(size_t j = 0; j < Dim; ++j){
				min_a[i * (2 * Dim) + Dim + j] = a[i * Dim + j];//Store A
				min_a[(1 + i) * (2 * Dim) + j] = a[j * Dim + i];//Store A^T
			}
		}
		
		//Minimize
		minimize(min_a, min_x, min_b);
		
		//Update x
		for(size_t i = 0; i < Dim; ++i){
			x[i] += min_x[i];
		}
		
		//Update t
		t *= mu;
	}
}
*/

//Gram-Schmidt orthogonalization
template <typename T>
constexpr void matrix_orthogonalize(const std::array<T, 9>& x, std::array<T, 9>& orth)
{
	//FIXME: Not sure if that works correctly (does seem to produce wring results)
	const T length_0 = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
	orth[0] = x[0]/length_0;
	orth[1] = x[1]/length_0;
	orth[2] = x[2]/length_0;
	
	const vec<T, 3> column_0 {orth[0], orth[1], orth[2]};
	const vec<T, 3> column_1 {x[3], x[4], x[5]};
	const vec<T, 3> dot_div = column_1 - column_0.dot(column_1) * column_0;
	
	const T length_1 = sqrt(dot_div[0] * dot_div[0] + dot_div[1] * dot_div[1] + dot_div[2] * dot_div[2]);
	orth[3] = dot_div[0]/length_1;
	orth[4] = dot_div[1]/length_1;
	orth[5] = dot_div[2]/length_1;
	
	const vec<T, 3> column_1_new {orth[3], orth[4], orth[5]};
	vec<T, 3> cross;
	vec_cross_vec_3d(cross.data_arr(), column_0.data_arr(), column_1_new.data_arr());
	
	const T length_2 = sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
	orth[6] = cross[0]/length_2;
	orth[7] = cross[1]/length_2;
	orth[8] = cross[2]/length_2;
}

template<typename T>
constexpr T matrix_determinant_3d(const std::array<T, 9>& x) {
	return x[0] * (x[4] * x[8] - x[7] * x[5]) + x[3] * (x[7] * x[2] - x[1] * x[8]) + x[6] * (x[1] * x[5] - x[4] * x[2]);
}

template<typename T>
constexpr T matrix_determinant_2d(const std::array<T, 4>& x) {
	return x[0] * x[3] - x[1] * x[2];
}

template<typename T>
constexpr void matrix_transpose_3d(const std::array<T, 9>& x, std::array<T, 9>& transpose) {
	transpose[0] = x[0];
	transpose[1] = x[3];
	transpose[2] = x[6];
	transpose[3] = x[1];
	transpose[4] = x[4];
	transpose[5] = x[7];
	transpose[6] = x[2];
	transpose[7] = x[5];
	transpose[8] = x[8];
}

template<typename T>
constexpr void matrix_transpose_2d(const std::array<T, 4>& x, std::array<T, 4>& transpose) {
	transpose[0] = x[0];
	transpose[1] = x[2];
	transpose[2] = x[1];
	transpose[3] = x[3];
}

template<typename T>
constexpr T matrix_trace_3d(const std::array<T, 9>& x) {
	return x[0] + x[4] + x[8];
}

template<typename T>
constexpr T matrix_trace_2d(const std::array<T, 4>& x) {
	return x[0] + x[3];
}

template<typename T>
constexpr void matrix_matrix_multiplication_3d(const std::array<T, 9>& a, const std::array<T, 9>& b, std::array<T, 9>& c) {
	c[0] = a[0] * b[0] + a[3] * b[1] + a[6] * b[2];
	c[1] = a[1] * b[0] + a[4] * b[1] + a[7] * b[2];
	c[2] = a[2] * b[0] + a[5] * b[1] + a[8] * b[2];
	c[3] = a[0] * b[3] + a[3] * b[4] + a[6] * b[5];
	c[4] = a[1] * b[3] + a[4] * b[4] + a[7] * b[5];
	c[5] = a[2] * b[3] + a[5] * b[4] + a[8] * b[5];
	c[6] = a[0] * b[6] + a[3] * b[7] + a[6] * b[8];
	c[7] = a[1] * b[6] + a[4] * b[7] + a[7] * b[8];
	c[8] = a[2] * b[6] + a[5] * b[7] + a[8] * b[8];
}

template<typename T, size_t Dim>
constexpr void matrix_matrix_multiplication(const std::array<T, Dim * Dim>& a, const std::array<T, Dim * Dim>& b, std::array<T, Dim * Dim>& c) {
	for(size_t i = 0; i < Dim; ++i){
		for(size_t j = 0; j < Dim; ++j){
			c[i * Dim + j] = 0;
			for(size_t k = 0; k < Dim; ++k){
				c[i * Dim + j] += a[k * Dim + i] * b[j * Dim + k];
			}
		}
	}
}

template<typename T>
constexpr void matrix_matrix_multiplication_2d(const std::array<T, 4>& a, const std::array<T, 4>& b, std::array<T, 4>& c) {
	c[0] = a[0] * b[0] + a[2] * b[1];
	c[1] = a[1] * b[0] + a[3] * b[1];
	c[2] = a[0] * b[2] + a[2] * b[3];
	c[3] = a[1] * b[2] + a[3] * b[3];
}

template<typename T>
constexpr void matrix_diagonal_matrix_multiplication_3d(const std::array<T, 9>& a, const std::array<T, 9>& b, std::array<T, 9>& c) {
	c[0] = a[0] * b[0];
	c[1] = a[1] * b[0];
	c[2] = a[2] * b[0];
	c[3] = a[3] * b[1];
	c[4] = a[4] * b[1];
	c[5] = a[5] * b[1];
	c[6] = a[6] * b[2];
	c[7] = a[7] * b[2];
	c[8] = a[8] * b[2];
}

template<typename T>
constexpr void matrix_diagonal_matrix_multiplication_2d(const std::array<T, 4>& a, const std::array<T, 4>& b, std::array<T, 4>& c) {
	c[0] = a[0] * b[0];
	c[1] = a[1] * b[0];
	c[2] = a[2] * b[1];
	c[3] = a[3] * b[1];
}

template<typename T>
constexpr void matrix_transpose_matrix_multiplication_3d(const std::array<T, 9>& a, const std::array<T, 9>& b, std::array<T, 9>& c) {
	c[0] = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
	c[1] = a[3] * b[0] + a[4] * b[1] + a[5] * b[2];
	c[2] = a[6] * b[0] + a[7] * b[1] + a[8] * b[2];
	c[3] = a[0] * b[3] + a[1] * b[4] + a[2] * b[5];
	c[4] = a[3] * b[3] + a[4] * b[4] + a[5] * b[5];
	c[5] = a[6] * b[3] + a[7] * b[4] + a[8] * b[5];
	c[6] = a[0] * b[6] + a[1] * b[7] + a[2] * b[8];
	c[7] = a[3] * b[6] + a[4] * b[7] + a[5] * b[8];
	c[8] = a[6] * b[6] + a[7] * b[7] + a[8] * b[8];
}

template<typename T>
constexpr void matrix_transpose_matrix_multiplication_2d(const std::array<T, 4>& a, const std::array<T, 4>& b, std::array<T, 4>& c) {
	c[0] = a[0] * b[0] + a[1] * b[1];
	c[1] = a[2] * b[0] + a[3] * b[1];
	c[2] = a[0] * b[2] + a[1] * b[3];
	c[3] = a[2] * b[2] + a[3] * b[3];
}

template<typename T>
constexpr void matrix_vector_multiplication_3d(const std::array<T, 9>& x, const std::array<T, 3>& v, std::array<T, 3>& result) {
	result[0] = x[0] * v[0] + x[3] * v[1] + x[6] * v[2];
	result[1] = x[1] * v[0] + x[4] * v[1] + x[7] * v[2];
	result[2] = x[2] * v[0] + x[5] * v[1] + x[8] * v[2];
}

template<typename T, size_t Dim>
constexpr void matrix_vector_multiplication(const std::array<T, Dim * Dim>& x, const std::array<T, Dim>& v, std::array<T, Dim>& result) {
	for(size_t i = 0; i < Dim; ++i){
		result[i] = 0;
		for(size_t j = 0; j < Dim; ++j){
			result[i] += x[j * Dim + i] * v[j];
		}
	}
}

template<typename T>
constexpr void matrix_vector_multiplication_2d(const std::array<T, 4>& x, const std::array<T, 2>& v, std::array<T, 2>& result) {
	result[0] = x[0] * v[0] + x[2] * v[1];
	result[1] = x[1] * v[0] + x[3] * v[1];
}

template<typename T>
constexpr void vector_matrix_multiplication_3d(const std::array<T, 3>& v, const std::array<T, 9>& x, std::array<T, 3>& result) {
	result[0] = x[0] * v[0] + x[1] * v[1] + x[2] * v[2];
	result[1] = x[3] * v[0] + x[4] * v[1] + x[5] * v[2];
	result[2] = x[6] * v[0] + x[7] * v[1] + x[8] * v[2];
}

template<typename T>
constexpr void vector_matrix_multiplication_2d(const std::array<T, 2>& v, const std::array<T, 4>& x, std::array<T, 2>& result) {
	result[0] = x[0] * v[0] + x[1] * v[1];
	result[1] = x[2] * v[0] + x[3] * v[1];
}

template<typename T>
constexpr void matrix_matrix_transpose_multiplication_3d(const std::array<T, 9>& a, const std::array<T, 9>& b, std::array<T, 9>& c) {
	c[0] = a[0] * b[0] + a[3] * b[3] + a[6] * b[6];
	c[1] = a[1] * b[0] + a[4] * b[3] + a[7] * b[6];
	c[2] = a[2] * b[0] + a[5] * b[3] + a[8] * b[6];
	c[3] = a[0] * b[1] + a[3] * b[4] + a[6] * b[7];
	c[4] = a[1] * b[1] + a[4] * b[4] + a[7] * b[7];
	c[5] = a[2] * b[1] + a[5] * b[4] + a[8] * b[7];
	c[6] = a[0] * b[2] + a[3] * b[5] + a[6] * b[8];
	c[7] = a[1] * b[2] + a[4] * b[5] + a[7] * b[8];
	c[8] = a[2] * b[2] + a[5] * b[5] + a[8] * b[8];
}

//c = b * a * b^T
template<typename T>
constexpr void matrix_matrix_matrix_transpose_multiplication_3d(const std::array<T, 9>& a, const std::array<T, 9>& b, std::array<T, 9>& c) {
	c[0] = b[0] * a[0] * b[0] + b[1] * a[3] * b[3] + b[2] * a[6] * b[6];
	c[1] = b[0] * a[1] * b[0] + b[1] * a[4] * b[3] + b[2] * a[7] * b[6];
	c[2] = b[0] * a[2] * b[0] + b[1] * a[5] * b[3] + b[2] * a[8] * b[6];
	c[3] = b[3] * a[0] * b[1] + b[4] * a[3] * b[4] + b[5] * a[6] * b[7];
	c[4] = b[3] * a[1] * b[1] + b[4] * a[4] * b[4] + b[5] * a[7] * b[7];
	c[5] = b[3] * a[2] * b[1] + b[4] * a[5] * b[4] + b[5] * a[8] * b[7];
	c[6] = b[6] * a[0] * b[2] + b[7] * a[3] * b[5] + b[8] * a[6] * b[8];
	c[7] = b[6] * a[1] * b[2] + b[7] * a[4] * b[5] + b[8] * a[7] * b[8];
	c[8] = b[6] * a[2] * b[2] + b[7] * a[5] * b[5] + b[8] * a[8] * b[8];
}

template<typename T>
constexpr void matrix_matrix_transpose_multiplication_2d(const std::array<T, 4>& a, const std::array<T, 4>& b, std::array<T, 4>& c) {
	c[0] = a[0] * b[0] + a[2] * b[2];
	c[1] = a[1] * b[0] + a[3] * b[2];
	c[2] = a[0] * b[1] + a[2] * b[3];
	c[3] = a[1] * b[1] + a[3] * b[3];
}

template<typename T>
constexpr void matrix_matrix_tranpose_multiplication_3d(const std::array<T, 9>& in, std::array<T, 9>& out) {
	out[0] = in[0] * in[0] + in[3] * in[3] + in[6] * in[6];
	out[1] = in[1] * in[0] + in[4] * in[3] + in[7] * in[6];
	out[2] = in[2] * in[0] + in[5] * in[3] + in[8] * in[6];

	out[3] = in[0] * in[1] + in[3] * in[4] + in[6] * in[7];
	out[4] = in[1] * in[1] + in[4] * in[4] + in[7] * in[7];
	out[5] = in[2] * in[1] + in[5] * in[4] + in[8] * in[7];

	out[6] = in[0] * in[2] + in[3] * in[5] + in[6] * in[8];
	out[7] = in[1] * in[2] + in[4] * in[5] + in[7] * in[8];
	out[8] = in[2] * in[2] + in[5] * in[5] + in[8] * in[8];
}

template<typename T>
constexpr void matrix_deviatoric_3d(const std::array<T, 9>& in, std::array<T, 9>& out) {
	//FIXME: Rewrote this, cause for some reasons sometimes precision errors occured (Probably compiler bug)
	//T trace_in_div_d = (in[0] + in[4] + in[8]) / static_cast<T>(3.0);
	out[0] = in[0] * static_cast<T>(2.0 / 3.0) - (in[4] + in[8]) / static_cast<T>(3.0);
	out[1] = in[1];
	out[2] = in[2];

	out[3] = in[3];
	out[4] = in[4] * static_cast<T>(2.0 / 3.0) - (in[0] + in[8]) / static_cast<T>(3.0);
	out[5] = in[5];

	out[6] = in[6];
	out[7] = in[7];
	out[8] = in[8] * static_cast<T>(2.0 / 3.0) - (in[0] + in[4]) / static_cast<T>(3.0);
}

template <typename T>
constexpr void quat_quat_multiplication(const std::array<T, 4>& a, const std::array<T, 4>& b, std::array<T, 4>& c) {
	c[0] = a[0] * b[3] + b[0] * a[3] + a[1] * b[2] - a[2] * b[1];
	c[1] = a[1] * b[3] + b[1] * a[3] + a[2] * b[0] - a[0] * b[2];
	c[2] = a[2] * b[3] + b[2] * a[3] + a[0] * b[1] - a[1] * b[0];
	c[3] = a[3] * b[3] - (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
}

template <typename T>
constexpr void vec_quat_multiplication(const std::array<T, 3>& a, const std::array<T, 4>& b, std::array<T, 4>& c) {
	c[0] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1];
	c[1] = a[1] * b[3] + a[2] * b[0] - a[0] * b[2];
	c[2] = a[2] * b[3] + a[0] * b[1] - a[1] * b[0];
	c[3] = -(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
}

template <typename T>
constexpr void quat_quat_cross(const std::array<T, 4>& a, const std::array<T, 4>& b, std::array<T, 4>& c) {
	c[0] = a[0] * b[3] + b[0] * a[3] + a[1] * b[2] - a[2] * b[1];
	c[1] = a[1] * b[3] + b[1] * a[3] + a[2] * b[0] - a[0] * b[2];
	c[2] = a[2] * b[3] + b[2] * a[3] + a[0] * b[1] - a[1] * b[0];
	c[3] = a[3] * b[3];
}

template <typename T>
constexpr void vec_quat_cross(const std::array<T, 3>& a, const std::array<T, 4>& b, std::array<T, 4>& c) {
	c[0] = a[0] * b[3] + b[0] * a[3] + a[1] * b[2] - a[2] * b[1];
	c[1] = a[1] * b[3] + b[1] * a[3] + a[2] * b[0] - a[0] * b[2];
	c[2] = a[2] * b[3] + b[2] * a[3] + a[0] * b[1] - a[1] * b[0];
	c[3] = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
}

template <typename T>
constexpr void quat_vec_cross(const std::array<T, 4>& a, const std::array<T, 3>& b, std::array<T, 4>& c) {
	c[0] = a[0] * b[3] + b[0] * a[3] + a[1] * b[2] - a[2] * b[1];
	c[1] = a[1] * b[3] + b[1] * a[3] + a[2] * b[0] - a[0] * b[2];
	c[2] = a[2] * b[3] + b[2] * a[3] + a[0] * b[1] - a[1] * b[0];
	c[3] = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
}

template <typename T>
constexpr void rotate_by_quat(const std::array<T, 3>& in, const std::array<T, 4>& quat, std::array<T, 3>& out)
{
	out[0] = in[0] + static_cast<T>(2.0) * quat[1] * (quat[0] * in[1] - quat[1] * in[0] + quat[3] * in[2]) - static_cast<T>(2.0) * quat[2] * (quat[2] * in[0] - quat[0] * in[2] + quat[3] * in[1]);
	out[1] = in[1] + static_cast<T>(2.0) * quat[2] * (quat[1] * in[2] - quat[2] * in[1] + quat[3] * in[0]) - static_cast<T>(2.0) * quat[0] * (quat[0] * in[1] - quat[1] * in[0] + quat[3] * in[2]);
	out[2] = in[2] + static_cast<T>(2.0) * quat[0] * (quat[2] * in[0] - quat[0] * in[2] + quat[3] * in[1]) - static_cast<T>(2.0) * quat[1] * (quat[1] * in[2] - quat[2] * in[1] + quat[3] * in[0]);
}

template <typename T>
constexpr void rotate_by_quat(const std::array<T, 9>& in, const std::array<T, 4>& quat, std::array<T, 9>& out)
{
	const std::array<T, 3> column_0 {in[0], in[1], in[2]};
	const std::array<T, 3> column_1 {in[3], in[4], in[5]};
	const std::array<T, 3> column_2 {in[6], in[7], in[8]};
	
	std::array<T, 3> column_0_rotated;
	std::array<T, 3> column_1_rotated;
	std::array<T, 3> column_2_rotated;
	rotate_by_quat(column_0, quat, column_0_rotated);
	rotate_by_quat(column_1, quat, column_1_rotated);
	rotate_by_quat(column_2, quat, column_2_rotated);
	
	out[0] = column_0_rotated[0];
	out[1] = column_0_rotated[1];
	out[2] = column_0_rotated[2];
	out[3] = column_1_rotated[0];
	out[4] = column_1_rotated[1];
	out[5] = column_1_rotated[2];
	out[6] = column_2_rotated[0];
	out[7] = column_2_rotated[1];
	out[8] = column_2_rotated[2];
}

#if 0
template <typename T>
constexpr T vector_max_component(const std::array<T, 3>& x)
{
    T tmp = x[0];
    if (tmp < x[1]){
        tmp = x[1];
	}
    if (tmp < x[2]){
        tmp = x[2];
	}
    return tmp;
}

template <typename T>
constexpr T vector_magnitude(const std::array<T, 3>& x)
{
    return sqrtf(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
}

template <typename T>
constexpr void vector_component_max(const std::array<T, 3>& x, const std::array<T, 3>& y, std::array<T, 3>& result)
{
    for (int v = 0; v < 3; ++v){
        result[v] = x[v] > y[v] ? x[v] : y[v];
	}
}

template <typename T>
constexpr T signed_distance_oriented_box(const std::array<T, 3>& point, const std::array<T, 3>& box_center, const std::array<T, 3>& edges, const std::array<T, 9>& rotation) {
    std::array<T, 3> tmp;
    for (int v = 0; v < 3; ++v){
        tmp[v] = point[v] - box_center[v];
	}
    std::array<T, 3> diff;
    matrixvectormultiplication(rotation, tmp, diff);
    std::array<T, 3> phi;
    for (int v = 0; v < 3; ++v){
        phi[v] = (diff[v] > 0 ? diff[v] : -diff[v]) - edges[v] * .5f;
	}

    if (phi[0] <= 0 && phi[1] <= 0 && phi[2] <= 0){
        return vector_max_component(phi);
    }else {
        T zeros[3] = {0, 0, 0};
        vector_component_max(phi, zeros, diff);
        return vector_magnitude(diff);
    }
}

template <typename T>
constexpr void quat_cast(std::array<T, 9>& mat, std::array<T, 4>& quat)
{
    std::array<std::array<T, 3>, 3> m;
    for (int i = 0; i < 3; ++i){
    for (int j = 0; j < 3; ++j){
        m[i][j] = mat[i + j * 3];
	}
	}
    T four_x_squared_minus1 = m[0][0] - m[1][1] - m[2][2];
    T four_y_squared_minus1 = m[1][1] - m[0][0] - m[2][2];
    T four_z_squared_minus1 = m[2][2] - m[0][0] - m[1][1];
    T four_w_squared_minus1 = m[0][0] + m[1][1] + m[2][2];

    int biggest_index = 0;
    T four_biggest_squared_minus1 = four_w_squared_minus1;
    if (four_x_squared_minus1 > four_biggest_squared_minus1)
    {
        four_biggest_squared_minus1 = four_x_squared_minus1;
        biggest_index = 1;
    }
    if (four_y_squared_minus1 > four_biggest_squared_minus1)
    {
        four_biggest_squared_minus1 = four_y_squared_minus1;
        biggest_index = 2;
    }
    if (four_z_squared_minus1 > four_biggest_squared_minus1)
    {
        four_biggest_squared_minus1 = four_z_squared_minus1;
        biggest_index = 3;
    }

    T biggest_val = sqrt(four_biggest_squared_minus1 + static_cast<T>(1)) * static_cast<T>(0.5);
    T mult = static_cast<T>(0.25) / biggest_val;

    switch (biggest_index)
    {
    case 0:
        quat[0] = biggest_val;
        quat[1] = (m[1][2] - m[2][1]) * mult;
        quat[2] = (m[2][0] - m[0][2]) * mult;
        quat[3] = (m[0][1] - m[1][0]) * mult;
        break;
    case 1:
        quat[0] = (m[1][2] - m[2][1]) * mult;
        quat[1] = biggest_val;
        quat[2] = (m[0][1] + m[1][0]) * mult;
        quat[3] = (m[2][0] + m[0][2]) * mult;
        break;
    case 2:
        quat[0] = (m[2][0] - m[0][2]) * mult;
        quat[1] = (m[0][1] + m[1][0]) * mult;
        quat[2] = biggest_val;
        quat[3] = (m[1][2] + m[2][1]) * mult;
        break;
    case 3:
        quat[0] = (m[0][1] - m[1][0]) * mult;
        quat[1] = (m[2][0] + m[0][2]) * mult;
        quat[2] = (m[1][2] + m[2][1]) * mult;
        quat[3] = biggest_val;
        break;
    default: // Silence a -Wswitch-default warning in GCC. Should never actually get here. Assert is just for sanity.
        //assert(false);
        break;
    }
}

template <typename T>
constexpr void mat3_cast(const std::array<T, 4>& q, std::array<T, 9>& mat)
{
    T qxx(q[0] * q[0]);
    T qyy(q[1] * q[1]);
    T qzz(q[2] * q[2]);
    T qxz(q[0] * q[2]);
    T qxy(q[0] * q[1]);
    T qyz(q[1] * q[2]);
    T qwx(q[3] * q[0]);
    T qwy(q[3] * q[1]);
    T qwz(q[3] * q[2]);

    /*Result[0][0] = */ mat[0] = T(1) - T(2) * (qyy + qzz);
    /*Result[0][1] = */ mat[3] = T(2) * (qxy + qwz);
    /*Result[0][2] = */ mat[6] = T(2) * (qxz - qwy);

    /*Result[1][0] = */ mat[1] = T(2) * (qxy - qwz);
    /*Result[1][1] = */ mat[4] = T(1) - T(2) * (qxx + qzz);
    /*Result[1][2] = */ mat[7] = T(2) * (qyz + qwx);

    /*Result[2][0] = */ mat[2] = T(2) * (qxz + qwy);
    /*Result[2][1] = */ mat[5] = T(2) * (qyz - qwx);
    /*Result[2][2] = */ mat[8] = T(1) - T(2) * (qxx + qyy);
}
#endif

}// namespace mn

#endif
