#ifndef ALPHA_SHAPES_CUH
#define ALPHA_SHAPES_CUH

#include <MnBase/Math/Matrix/MatrixUtils.h>

#include <thrust/sort.h>

#include "particle_buffer.cuh"
#include "settings.h"
#include "utility_funcs.hpp"
#include "triangle_mesh.cuh"
#include "kernels.cuh"

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

//Alpha shapes

//TODO: Acceleration struct for fast neighbour search; Weighted alpha (by density); Different triangulation (e.g. regular);
//TODO: Large alpha spanning across more than two cells (using threads to handle several cells x^3 as a block; if spanning way too far we need more sophistocated approaches (e.g. somehow tracking the end points or using uniform grid search as proposed in some papers)
//TODO: Handle near isolated points as curves or thin surfaces instead all as drops (e.g. based on alpha radius; must happen accross boundary; after all other steps or beforhand?)
//TODO: Use threads for acceleration?

namespace mn {
	
//Big enough to cover all cells near the current cell that can contain particles near enough to make a face an alpha face
constexpr size_t ALPHA_SHAPES_KERNEL_SIZE = static_cast<size_t>(const_sqrt(config::MAX_ALPHA) / config::G_DX) + 1;//NOTE:Static cast required for expression being const
constexpr size_t ALPHA_SHAPES_KERNEL_LENGTH = 2 * ALPHA_SHAPES_KERNEL_SIZE + 1;//Sidelength of the kernel cube
constexpr size_t ALPHA_SHAPES_NUMBER_OF_CELLS = ALPHA_SHAPES_KERNEL_LENGTH * ALPHA_SHAPES_KERNEL_LENGTH * ALPHA_SHAPES_KERNEL_LENGTH;

constexpr size_t ALPHA_SHAPES_MAX_PARTICLE_COUNT = ALPHA_SHAPES_NUMBER_OF_CELLS * config::G_MAX_PARTICLES_IN_CELL;
constexpr size_t ALPHA_SHAPES_MAX_TRIANGLE_COUNT = 2 * ALPHA_SHAPES_MAX_PARTICLE_COUNT - 4;//Max value by euler characteristic: 2 * MAX_PARTICLES_IN_CELL - 4 

constexpr unsigned int ALPHA_SHAPES_MAX_KERNEL_SIZE = config::G_MAX_ACTIVE_BLOCK;

constexpr float ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD = 1e-9;//TODO: Maybe adjust threshold

using AlphaShapesBlockDomain	   = CompactDomain<char, config::G_BLOCKSIZE, config::G_BLOCKSIZE, config::G_BLOCKSIZE>;
using AlphaShapesGridBufferDomain  = CompactDomain<int, config::G_MAX_ACTIVE_BLOCK>;
using AlphaShapesParticleDomain	   = CompactDomain<int, ALPHA_SHAPES_MAX_PARTICLE_COUNT>;
using AlphaShapesTriangleDomain	   = CompactDomain<int, ALPHA_SHAPES_MAX_TRIANGLE_COUNT>;

//NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables, readability-identifier-naming) Check is buggy and reports variable errors for template arguments
using AlphaShapesParticleBufferData  = Structural<StructuralType::DENSE, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::SUM_POW2_ALIGN>, ParticleBinDomain, attrib_layout::SOA, f32_, f32_, f32_, f32_, f32_, f32_, f32_, f32_>;//Point type (integer bytes as float, needs to be casted accordingly), normal, mean_curvature, gauss_curvature ; temporary: summed_area, normal, summed_angles, summed_laplacians

//TODO: Mayber different gouping, alignment;
using AlphaShapesTriangle  = Structural<StructuralType::DENSE, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::COMPACT>, CompactDomain<char, 3>, attrib_layout::AOS, i32_, i32_, i32_>;

//using AlphaShapesGridParticleData  = Structural<StructuralType::DENSE, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::SUM_POW2_ALIGN>, AlphaShapesParticleDomain, attrib_layout::SOA, i32_>;//index
using AlphaShapesGridTriangleData  = Structural<StructuralType::DENSE, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::SUM_POW2_ALIGN>, AlphaShapesTriangleDomain, attrib_layout::SOA, AlphaShapesTriangle, StructuralEntity<bool>, AlphaShapesTriangle>;//triangle, is_alpha, alpha_triangle

//using AlphaShapesGridBufferCellData = Structural<StructuralType::DYNAMIC, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::COMPACT>, AlphaShapesBlockDomain, attrib_layout::AOS, AlphaShapesGridParticleData, AlphaShapesGridTriangleData>;
using AlphaShapesGridBufferCellData = Structural<StructuralType::DENSE, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::COMPACT>, AlphaShapesBlockDomain, attrib_layout::AOS, AlphaShapesGridTriangleData>;
using AlphaShapesGridBufferBlockData = Structural<StructuralType::DYNAMIC, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::COMPACT>, AlphaShapesGridBufferDomain, attrib_layout::AOS, AlphaShapesGridBufferCellData>;
//NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables, readability-identifier-naming)

struct AlphaShapesParticleBuffer : Instance<particle_buffer_<AlphaShapesParticleBufferData>> {
	using base_t							 = Instance<particle_buffer_<AlphaShapesParticleBufferData>>;

	AlphaShapesParticleBuffer() = default;

	template<typename Allocator>
	AlphaShapesParticleBuffer(Allocator allocator, std::size_t count)
		: base_t {spawn<particle_buffer_<AlphaShapesParticleBufferData>, orphan_signature>(allocator, count)}
		{}
};

/*struct AlphaShapesGridBuffer : Instance<AlphaShapesGridBufferBlockData> {
	using base_t = Instance<AlphaShapesGridBufferBlockData>;

	template<typename Allocator>
	explicit AlphaShapesGridBuffer(Allocator allocator)
		: base_t {spawn<AlphaShapesGridBufferBlockData, orphan_signature>(allocator)} {}

	template<typename Allocator>
	void check_capacity(Allocator allocator, std::size_t capacity) {
		if(capacity > this->capacity) {
			this->resize(allocator, capacity);
		}
	}
};*/

struct AlphaShapesGridBuffer {

	template<typename Allocator>
	explicit AlphaShapesGridBuffer(Allocator allocator) {}

	template<typename Allocator>
	void check_capacity(Allocator allocator, std::size_t capacity) {
		
	}
	
	template<typename Allocator>
	void resize(Allocator allocator, std::size_t capacity) {
		
	}
};

//TODO: Make magic numbers to constants where suitable
//TODO: Ensure call dimensions and such are small enough to allow narrowing conversations. Or directly use unsigned where possible
//TODO: Maybe use names instead of formula signs for better understanding
//NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers, readability-identifier-naming, misc-definitions-in-headers) CUDA does not yet support std::span; Common names for physical formulas; Cannot declare __global__ functions inline

enum class AlphaShapesPointType {
	OUTER_POINT = 0,
	INNER_POINT,
	ISOLATED_POINT,
	CURVE_1D,
	CURVE_2D,
	TOTAL
};

template<typename Partition, MaterialE MaterialType>
__forceinline__ __device__ void alpha_shapes_fetch_particles(const ParticleBuffer<MaterialType> particle_buffer, const Partition prev_partition, int* particle_indices, const int particles_in_cell, const int src_blockno, const ivec3 blockid, const ivec3 blockid_offset, const int cellno, const int start_index){
	for(int particle_id = 0; particle_id < particles_in_cell; particle_id++) {
		//Fetch index of the advection source
		ivec3 offset;
		int source_pidib;
		{
			//Fetch advection (direction at high bits, particle in in cell at low bits)
			const int advect = particle_buffer.cellbuckets[src_blockno * config::G_PARTICLE_NUM_PER_BLOCK + cellno * config::G_MAX_PARTICLES_IN_CELL + particle_id];
			
			//Retrieve the direction (first stripping the particle id by division)
			dir_components(advect / config::G_PARTICLE_NUM_PER_BLOCK, offset.data_arr());

			//Retrieve the particle id by AND for lower bits
			source_pidib = advect & (config::G_PARTICLE_NUM_PER_BLOCK - 1);
		}
		
		//Calculate offset from current blockid
		const int dirtag = dir_offset((blockid_offset + offset).data_arr());
		
		particle_indices[start_index + particle_id] = (dirtag * config::G_PARTICLE_NUM_PER_BLOCK) | source_pidib;
	}
}

template<typename Partition, MaterialE MaterialType>
__forceinline__ __device__ void alpha_shapes_fetch_id(const ParticleBuffer<MaterialType> particle_buffer, const Partition prev_partition, const int particle_id, const ivec3 blockid, int& advection_source_blockno, int& source_pidib){
	//Fetch index of the advection source
	{
		//Fetch advection (direction at high bits, particle in in cell at low bits)
		const int advect = particle_id;

		//Retrieve the direction (first stripping the particle id by division)
		ivec3 offset;
		dir_components(advect / config::G_PARTICLE_NUM_PER_BLOCK, offset.data_arr());

		//Retrieve the particle id by AND for lower bits
		source_pidib = advect & (config::G_PARTICLE_NUM_PER_BLOCK - 1);

		//Get global index by adding blockid and offset
		const ivec3 global_advection_index = blockid + offset;

		//Get block_no from partition
		advection_source_blockno = prev_partition.query(global_advection_index);
	}
}

template<typename Partition, MaterialE MaterialType>
__forceinline__ __device__ std::array<float, 3> alpha_shapes_get_particle_position(const ParticleBuffer<MaterialType> particle_buffer, const Partition prev_partition, const int particle_id, const ivec3 blockid){
	//Fetch index of the advection source
	int advection_source_blockno;
	int source_pidib;
	alpha_shapes_fetch_id(particle_buffer, prev_partition, particle_id, blockid, advection_source_blockno, source_pidib);
	
	FetchParticleBufferDataIntermediate fetch_particle_buffer_tmp = {};
	fetch_particle_buffer_data<MaterialType>(particle_buffer, advection_source_blockno, source_pidib, fetch_particle_buffer_tmp);
	return fetch_particle_buffer_tmp.pos;
}



__forceinline__ __host__ __device__ std::array<float, 3> alpha_shapes_calculate_triangle_normal(const std::array<std::array<float, 3>, 3>& triangle_positions){
	const std::array<vec3, 3> positions {
		  vec3(triangle_positions[0][0], triangle_positions[0][1], triangle_positions[0][2])
		, vec3(triangle_positions[1][0], triangle_positions[1][1], triangle_positions[1][2])
		, vec3(triangle_positions[2][0], triangle_positions[2][1], triangle_positions[2][2])
	};
	
	vec3 face_normal;
	vec_cross_vec_3d(face_normal.data_arr(), (positions[1] - positions[0]).data_arr(), (positions[2] - positions[0]).data_arr());
	
	const float face_normal_length = sqrt(face_normal[0] * face_normal[0] + face_normal[1] * face_normal[1] + face_normal[2] * face_normal[2]);
	
	//Normalize
	face_normal = face_normal / face_normal_length;
	
	return face_normal.data_arr();
}
/*
//Basically copied from https://rodolphe-vaillant.fr/entry/127/find-a-tetrahedron-circumcenter and https://www.mcs.anl.gov/~fathom/meshkit-docs/html/circumcenter_8cpp_source.html
__forceinline__ __device__ void alpha_shapes_get_circumsphere(const std::array<float, 3>& a, const std::array<float, 3>& b, const std::array<float, 3>& c, const std::array<float, 3>& d, std::array<float, 3>& center, float& radius){
    // Use coordinates relative to point 'a' of the tetrahedron.
 
    // ba = b - a
    const float ba_x = b[0] - a[0];
    const float ba_y = b[1] - a[1];
    const float ba_z = b[2] - a[2];
 
    // ca = c - a
    const float ca_x = c[0] - a[0];
    const float ca_y = c[1] - a[1];
    const float ca_z = c[2] - a[2];
 
    // da = d - a
    const float da_x = d[0] - a[0];
    const float da_y = d[1] - a[1];
    const float da_z = d[2] - a[2];
 
    // Squares of lengths of the edges incident to 'a'.
    const float len_ba = ba_x * ba_x + ba_y * ba_y + ba_z * ba_z;
    const float len_ca = ca_x * ca_x + ca_y * ca_y + ca_z * ca_z;
    const float len_da = da_x * da_x + da_y * da_y + da_z * da_z;
 
    // Cross products of these edges.
 
    // c cross d
    const float cross_cd_x = ca_y * da_z - da_y * ca_z;
    const float cross_cd_y = ca_z * da_x - da_z * ca_x;
    const float cross_cd_z = ca_x * da_y - da_x * ca_y;
 
    // d cross b
    const float cross_db_x = da_y * ba_z - ba_y * da_z;
    const float cross_db_y = da_z * ba_x - ba_z * da_x;
    const float cross_db_z = da_x * ba_y - ba_x * da_y;
 
    // b cross c
    const float cross_bc_x = ba_y * ca_z - ca_y * ba_z;
    const float cross_bc_y = ba_z * ca_x - ca_z * ba_x;
    const float cross_bc_z = ba_x * ca_y - ca_x * ba_y;
 
    // Calculate the denominator of the formula.
    const float denominator = 0.5f / (ba_x * cross_cd_x + ba_y * cross_cd_y + ba_z * cross_cd_z);
	
	//FIXME: Maybe clamp denominator if near to zero to ensure nearly degenerated tetrahedra are not unstable
 
    // Calculate offset (from 'a') of circumcenter.
    const float circ_x = (len_ba * cross_cd_x + len_ca * cross_db_x + len_da * cross_bc_x);
    const float circ_y = (len_ba * cross_cd_y + len_ca * cross_db_y + len_da * cross_bc_y);
    const float circ_z = (len_ba * cross_cd_z + len_ca * cross_db_z + len_da * cross_bc_z);
 
    center[0] = a[0] + circ_x * denominator;
    center[1] = a[1] + circ_y * denominator;
    center[2] = a[2] + circ_z * denominator;
	radius = std::sqrt(circ_x * circ_x + circ_y * circ_y + circ_z * circ_z) * std::abs(denominator);
	
	if(radius > 3.0f * sqrt(3.0f) * config::G_DX){
		printf("X %f %f # %f %f %f # %f %f %f # %f %f %f # %f %f %f # %f %f %f # %f - ", radius, 3.0f * sqrt(3.0f) * config::G_DX, center[0], center[1], center[2], a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], d[0], d[1], d[2], denominator);
	}
}*/

//Based on https://math.stackexchange.com/questions/2414640/circumsphere-of-a-tetrahedron https://www.cambridge.org/core/books/matrices-and-graphs-in-geometry/B225AA8C5EA96EAD4D0280173930FE03
__forceinline__ __device__ void alpha_shapes_get_circumsphere(const std::array<float, 3>& a, const std::array<float, 3>& b, const std::array<float, 3>& c, const std::array<float, 3>& d, std::array<float, 3>& center, float& radius){
	// ba = b - a
    const float ba_x = b[0] - a[0];
    const float ba_y = b[1] - a[1];
    const float ba_z = b[2] - a[2];
 
    // ca = c - a
    const float ca_x = c[0] - a[0];
    const float ca_y = c[1] - a[1];
    const float ca_z = c[2] - a[2];
 
    // da = d - a
    const float da_x = d[0] - a[0];
    const float da_y = d[1] - a[1];
    const float da_z = d[2] - a[2];
	
	// cb = c - b
    const float cb_x = c[0] - b[0];
    const float cb_y = c[1] - b[1];
    const float cb_z = c[2] - b[2];
 
    // db = d - b
    const float db_x = d[0] - b[0];
    const float db_y = d[1] - b[1];
    const float db_z = d[2] - b[2];
	
	// dc = d - c
    const float dc_x = d[0] - c[0];
    const float dc_y = d[1] - c[1];
    const float dc_z = d[2] - c[2];
	
	// Squares of lengths of the edges incident to 'a'.
    const float len_ba = ba_x * ba_x + ba_y * ba_y + ba_z * ba_z;
    const float len_ca = ca_x * ca_x + ca_y * ca_y + ca_z * ca_z;
    const float len_da = da_x * da_x + da_y * da_y + da_z * da_z;
	const float len_cb = cb_x * cb_x + cb_y * cb_y + cb_z * cb_z;
	const float len_db = db_x * db_x + db_y * db_y + db_z * db_z;
	const float len_dc = dc_x * dc_x + dc_y * dc_y + dc_z * dc_z;

   const std::array<float, 25> left_side {
		  0.0f, 1.0f, 1.0f, 1.0f, 1.0f
		, 1.0f, 0.0f, len_ba, len_ca, len_da
		, 1.0f, len_ba, 0.0f, len_cb, len_db
		, 1.0f, len_ca, len_cb, 0.0f, len_dc
		, 1.0f, len_da, len_db, len_dc, 0.0f
	};
	const std::array<float, 5> right_side {-2.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	
	//Solve  C^T * m0 = {-2, 0, 0, 0, 0}; from M = -2C^-1 => MC = -2 * I;
	std::array<float, 5> m0;
	solve_linear_system(left_side, m0, right_side);
	
	const float sum = 1.0f;//m0[1] + m0[2] + m0[3] + m0[4];
	
	center[0] = (m0[1] * a[0] + m0[2] * b[0] + m0[3] * c[0] + m0[4] * d[0]) / sum;
    center[1] = (m0[1] * a[1] + m0[2] * b[1] + m0[3] * c[1] + m0[4] * d[1]) / sum;
    center[2] = (m0[1] * a[2] + m0[2] * b[2] + m0[3] * c[2] + m0[4] * d[2]) / sum;
	radius = 0.5f * sqrt(m0[0]);
	
	//if(radius > 3.0f * sqrt(3.0f) * config::G_DX){
		printf("X %f %f # %f %f %f # %f %f %f # %f %f %f # %f %f %f # %f %f %f # %f %f %f %f %f - ", radius, 3.0f * sqrt(3.0f) * config::G_DX, center[0], center[1], center[2], a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], d[0], d[1], d[2], m0[0], m0[1], m0[2], m0[3], m0[4]);
	//}
}

template<typename Partition, MaterialE MaterialType>
__forceinline__ __device__ bool alpha_shapes_get_first_triangle(const ParticleBuffer<MaterialType> particle_buffer, const Partition prev_partition, const int* particle_indices, const ivec3 blockid, const int count, std::array<int, 3>& triangle){
	//Pick first point
	const int p0 = particle_indices[0];
	const std::array<float, 3> p0_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, p0, blockid);
	const vec3 p0_position {p0_position_arr[0], p0_position_arr[1], p0_position_arr[2]};
	
	//Find nearest point
	float current_minimum_distance = std::numeric_limits<float>::max();
	int p1;
	for(int particle_id = 1; particle_id < count; particle_id++) {
		const std::array<float, 3> particle_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, particle_indices[particle_id], blockid);
		const vec3 particle_position {particle_position_arr[0], particle_position_arr[1], particle_position_arr[2]};
		
		//Skip particles with same position
		if(particle_position[0] != p0_position[0] || particle_position[1] != p0_position[01] || particle_position[2] != p0_position[2]){
			//Calculate distance
			const vec3 diff = p0_position - particle_position;
			const float squared_distance = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
			
			if(squared_distance < current_minimum_distance){
				current_minimum_distance = squared_distance;
				p1 = particle_indices[particle_id];
			}
		}
	}
	
	if(current_minimum_distance < std::numeric_limits<float>::max()){
		const std::array<float, 3> p1_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, p1, blockid);
		const vec3 p1_position {p1_position_arr[0], p1_position_arr[1], p1_position_arr[2]};
		
		//Find smallest meridian sphere
		float current_smallest_meridian_sphere_radius = std::numeric_limits<float>::max();
		int p2;
		for(int particle_id = 1; particle_id < count; particle_id++) {
			const std::array<float, 3> particle_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, particle_indices[particle_id], blockid);
			const vec3 particle_position {particle_position_arr[0], particle_position_arr[1], particle_position_arr[2]};
			
			const float t = (p1_position - p0_position).dot(particle_position - p0_position) / (p1_position - p0_position).dot(p1_position - p0_position);
			const vec3 nearest_point_on_line = p0_position + t * (p1_position - p0_position);
			
			//Skip particles on same line
			if(particle_position[0] != nearest_point_on_line[0] || particle_position[1] != nearest_point_on_line[01] || particle_position[2] != nearest_point_on_line[2]){
				//Calculate meridian sphere radius
				const vec3 mid = (p0_position + p1_position + particle_position) / 3.0f;
				const vec3 diff0 = mid - p0_position;
				const vec3 diff1 = mid - p1_position;
				const vec3 diff2 = mid - particle_position;
				
				const float squared_distance0 = diff0[0] * diff0[0] + diff0[1] * diff0[1] + diff0[2] * diff0[2];
				const float squared_distance1 = diff1[0] * diff1[0] + diff1[1] * diff1[1] + diff1[2] * diff1[2];
				const float squared_distance2 = diff2[0] * diff2[0] + diff2[1] * diff2[1] + diff2[2] * diff2[2];
				
				const float meridian_sphere_radius = std::max(squared_distance0, std::max(squared_distance1, squared_distance2));
				
				if(meridian_sphere_radius < current_smallest_meridian_sphere_radius){
					current_smallest_meridian_sphere_radius = meridian_sphere_radius;
					p2 = particle_indices[particle_id];
				}
			}
		}
		
		//Return indices
		triangle[0] = p0;
		triangle[1] = p1;
		triangle[2] = p2;
		
		return (current_smallest_meridian_sphere_radius < std::numeric_limits<float>::max());
	}
}

template<typename Partition, MaterialE MaterialType, typename ForwardIterator>
__forceinline__ __device__ bool alpha_shapes_get_fourth_point(const ParticleBuffer<MaterialType> particle_buffer, const Partition prev_partition, const ForwardIterator& begin, const ForwardIterator& end, const ivec3 blockid, const std::array<std::array<float, 3>, 3>& triangle_positions, int& point_id){
	const std::array<float, 3>& triangle_normal = alpha_shapes_calculate_triangle_normal(triangle_positions);
	
	const vec3 triangle_normal_vec {triangle_normal[0], triangle_normal[1], triangle_normal[2]};
	
	const vec3 p0_position {triangle_positions[0][0], triangle_positions[0][1], triangle_positions[0][2]};
	
	//Only search in normal direction
	ForwardIterator p3_iter = thrust::min_element(thrust::seq, begin, end, [&particle_buffer, &prev_partition, &blockid, &triangle_positions, &triangle_normal_vec, &p0_position](const int& a, const int& b){
		const std::array<float, 3> particle_position_arr_a = alpha_shapes_get_particle_position(particle_buffer, prev_partition, a, blockid);
		const std::array<float, 3> particle_position_arr_b = alpha_shapes_get_particle_position(particle_buffer, prev_partition, b, blockid);
		
		const vec3 particle_position_a {particle_position_arr_a[0], particle_position_arr_a[1], particle_position_arr_a[2]};
		const vec3 particle_position_b {particle_position_arr_b[0], particle_position_arr_b[1], particle_position_arr_b[2]};
		
		//Test if in half_space; Also sorts out particles that lie in a plane with the triangle
		bool in_halfspace_a = triangle_normal_vec.dot(particle_position_a - p0_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
		bool in_halfspace_b = triangle_normal_vec.dot(particle_position_b - p0_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
		
		if(!in_halfspace_a && !in_halfspace_b){
			//Calculate delaunay spheres
			vec3 sphere_center_a;
			vec3 sphere_center_b;
			float radius_a;
			float radius_b;
			alpha_shapes_get_circumsphere(triangle_positions[0], triangle_positions[1], triangle_positions[2], particle_position_arr_a, sphere_center_a.data_arr(), radius_a);
			alpha_shapes_get_circumsphere(triangle_positions[0], triangle_positions[1], triangle_positions[2], particle_position_arr_b, sphere_center_b.data_arr(), radius_b);
			
			//Smaller delaunay sphere is the on that does not contain the other point
			const vec3 diff_a = sphere_center_b - particle_position_a;
			const float squared_distance_a = diff_a[0] * diff_a[0] + diff_a[1] * diff_a[1] + diff_a[2] * diff_a[2];
			const vec3 diff_b = sphere_center_a - particle_position_b;
			const float squared_distance_b = diff_b[0] * diff_b[0] + diff_b[1] * diff_b[1] + diff_b[2] * diff_b[2];
			
			return (squared_distance_a > (radius_b * radius_b) && squared_distance_b <= (radius_a * radius_a));
		}else{//If at leaast one is not in halfspace, consider a smaller if it is in halfspace (and b is not); otherwise a is not smaller (but equal)
			return in_halfspace_a;
		}
	});
	
	point_id = *p3_iter;
	
	const std::array<float, 3> p3_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, point_id, blockid);
	const vec3 p3_position {p3_position_arr[0], p3_position_arr[1], p3_position_arr[2]};
	
	//FIXME: Do we have to recheck that no other point lies in circumsphere or is this guranteed?
		
	//Test if in half_space; Also sorts out particles that lie in a plane with the triangle
	return (triangle_normal_vec.dot(p3_position - p0_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD);
}

template<typename Partition, MaterialE MaterialType>
__forceinline__ __device__ void alpha_shapes_check_contact_condition(const ParticleBuffer<MaterialType> particle_buffer, const Partition prev_partition, AlphaShapesParticleBuffer alpha_shapes_particle_buffer, const int* own_particle_indices, std::array<int, 3>* triangles, bool* triangles_is_alpha, int& current_triangle_count, int& next_triangle_count, const ivec3 blockid, const int p3_id, const bool is_alpha, const int finalize_particles_start, const int finalize_particles_end){
	const std::array<int, 3> current_triangle = triangles[0];
	
	std::array<int, 4> contact_indices;
	int face_contacts = 0;
	
	//Find triangles in touch
	//NOTE: current_triangle always is in contact
	for(int contact_triangle_index = 0; contact_triangle_index < current_triangle_count + next_triangle_count; contact_triangle_index++) {
		const std::array<int, 3>& contact_triangle = triangles[contact_triangle_index];
		
		//Test for face contact
		if(
			   (contact_triangle[0] == current_triangle[0] || contact_triangle[0] == current_triangle[1] || contact_triangle[0] == current_triangle[2] || contact_triangle[0] == p3_id)
			&& (contact_triangle[1] == current_triangle[0] || contact_triangle[1] == current_triangle[1] || contact_triangle[1] == current_triangle[2] || contact_triangle[1] == p3_id)
			&& (contact_triangle[2] == current_triangle[0] || contact_triangle[2] == current_triangle[1] || contact_triangle[2] == current_triangle[2] || contact_triangle[2] == p3_id)
		){
			contact_indices[face_contacts++] = contact_triangle_index;
		}
	}
	
	int next_triangle_start = current_triangle_count;
	
	for(int contact_triangle_index = face_contacts - 1; contact_triangle_index >= 0; contact_triangle_index--) {//NOTE: Loop goes backwards to handle big indices first which allows easier swapping
		//If the new tetrahedron is in alpha and the current face is not, then the current face is a boundary face that has to be kept; Same the other way round
		if(triangles_is_alpha[contact_indices[contact_triangle_index]] != is_alpha){
			alpha_shapes_finalize_triangle(particle_buffer, prev_partition, alpha_shapes_particle_buffer, own_particle_indices, triangles[contact_indices[contact_triangle_index]], blockid, finalize_particles_start, finalize_particles_end);
		}
		
		//Swap contact triangles to end of list to remove them
		int swap_index;
		if(contact_indices[contact_triangle_index] < current_triangle_count){
			swap_index = current_triangle_count - 1;//Swap with last active triangle
			
			//Decrease triangle count
			current_triangle_count--;
		}else{
			swap_index = next_triangle_start + next_triangle_count - 1;//Swap with first next triangle
			
			//Decrease next triangle count
			next_triangle_count--;
		}
		
		//Swap contacting triangle to the end
		thrust::swap(triangles[contact_indices[contact_triangle_index]], triangles[swap_index]);
		thrust::swap(triangles_is_alpha[contact_indices[contact_triangle_index]], triangles_is_alpha[swap_index]);
	}
	
	//Fill gap between current list and next list
	for(int i = 0; i < next_triangle_start - current_triangle_count; ++i) {
		thrust::swap(triangles[next_triangle_start - 1 - i], triangles[next_triangle_start + next_triangle_count - 1 - i]);
		thrust::swap(triangles_is_alpha[next_triangle_start - 1 - i], triangles_is_alpha[next_triangle_start + next_triangle_count - 1 - i]);
	}
	
	int next_triangle_end = current_triangle_count + next_triangle_count;
	next_triangle_count += (4 - face_contacts);
	
	if(current_triangle_count + next_triangle_count > static_cast<int>(ALPHA_SHAPES_MAX_TRIANGLE_COUNT)){
		printf("Too much triangles: May not be more than %d, but is %d\n", static_cast<int>(ALPHA_SHAPES_MAX_TRIANGLE_COUNT), current_triangle_count + next_triangle_count);
	}
	
	//Add new triangles
	//Ensure correct order (current_triangle cw normal points outwards)
	if(face_contacts == 1){
		triangles[next_triangle_end][0] = current_triangle[0];
		triangles[next_triangle_end][1] = current_triangle[1];
		triangles[next_triangle_end][2] = p3_id;
		triangles[next_triangle_end + 1][0] = current_triangle[1];
		triangles[next_triangle_end + 1][1] = current_triangle[2];
		triangles[next_triangle_end + 1][2] = p3_id;
		triangles[next_triangle_end + 2][0] = current_triangle[2];
		triangles[next_triangle_end + 2][1] = current_triangle[0];
		triangles[next_triangle_end + 2][2] = p3_id;
		
		triangles_is_alpha[next_triangle_end] = is_alpha;
		triangles_is_alpha[next_triangle_end + 1] = is_alpha;
		triangles_is_alpha[next_triangle_end + 2] = is_alpha;
	}else if(face_contacts == 2){
		std::array<int, 3> triangle_counts_per_vertex;
		for(int contact_triangle_index = 0; contact_triangle_index < face_contacts; contact_triangle_index++) {
			//Check which vertices are contacting
			for(int vertex_index = 0; vertex_index < 3; vertex_index++) {
				const int particle_index = triangles[contact_indices[contact_triangle_index]][vertex_index];
				if(particle_index == current_triangle[0]){
					triangle_counts_per_vertex[0]++;
				}else if(particle_index == current_triangle[1]){
					triangle_counts_per_vertex[1]++;
				}else{//particle_index == current_triangle[2]
					triangle_counts_per_vertex[2]++;
				}
			}
		}
		
		if(triangle_counts_per_vertex[0] == 1){
			triangles[next_triangle_end][0] = current_triangle[0];
			triangles[next_triangle_end][1] = current_triangle[1];
			triangles[next_triangle_end][2] = p3_id;
			triangles[next_triangle_end + 1][0] = current_triangle[2];
			triangles[next_triangle_end + 1][1] = current_triangle[0];
			triangles[next_triangle_end + 1][2] = p3_id;
		}else if(triangle_counts_per_vertex[1] == 1){
			triangles[next_triangle_end][0] = current_triangle[0];
			triangles[next_triangle_end][1] = current_triangle[1];
			triangles[next_triangle_end][2] = p3_id;
			triangles[next_triangle_end + 1][0] = current_triangle[1];
			triangles[next_triangle_end + 1][1] = current_triangle[2];
			triangles[next_triangle_end + 1][2] = p3_id;
		}else {//triangle_counts_per_vertex[2] == 1
			triangles[next_triangle_end][0] = current_triangle[1];
			triangles[next_triangle_end][1] = current_triangle[2];
			triangles[next_triangle_end][2] = p3_id;
			triangles[next_triangle_end + 1][0] = current_triangle[2];
			triangles[next_triangle_end + 1][1] = current_triangle[0];
			triangles[next_triangle_end + 1][2] = p3_id;
		}
		
		triangles_is_alpha[next_triangle_end] = is_alpha;
		triangles_is_alpha[next_triangle_end + 1] = is_alpha;
	}else if(face_contacts == 3){
		std::array<int, 3> triangle_counts_per_vertex;
		for(int contact_triangle_index = 0; contact_triangle_index < face_contacts; contact_triangle_index++) {
			//Check which vertices are contacting
			for(int vertex_index = 0; vertex_index < 3; vertex_index++) {
				const int particle_index = triangles[contact_indices[contact_triangle_index]][vertex_index];
				if(particle_index == current_triangle[0]){
					triangle_counts_per_vertex[0]++;
				}else if(particle_index == current_triangle[1]){
					triangle_counts_per_vertex[1]++;
				}else{//particle_index == current_triangle[2]
					triangle_counts_per_vertex[2]++;
				}
			}
		}
		
		if(triangle_counts_per_vertex[0] == 3){
			triangles[next_triangle_end][0] = current_triangle[1];
			triangles[next_triangle_end][1] = current_triangle[2];
			triangles[next_triangle_end][2] = p3_id;
		}else if(triangle_counts_per_vertex[1] == 3){
			triangles[next_triangle_end][0] = current_triangle[2];
			triangles[next_triangle_end][1] = current_triangle[0];
			triangles[next_triangle_end][2] = p3_id;
		}else {//triangle_counts_per_vertex[2] == 3
			triangles[next_triangle_end][0] = current_triangle[0];
			triangles[next_triangle_end][1] = current_triangle[1];
			triangles[next_triangle_end][2] = p3_id;
		}
		
		triangles_is_alpha[next_triangle_end] = is_alpha;
	}//Otherwise nothing to do, just faces removed
}

//TODO: In this and the following functions actually we do not need atomic add (just normal add). Also we need synchronization if we would use several threads and atomic
template<typename Partition, MaterialE MaterialType>
__forceinline__ __device__ void alpha_shapes_accumulate_triangle_at_vertex(const ParticleBuffer<MaterialType> particle_buffer, const Partition prev_partition, AlphaShapesParticleBuffer alpha_shapes_particle_buffer, const std::array<int, 3> triangle, const ivec3 blockid, const int contact_index){
	const float cotan_clamp_min_rad = 1.0f / std::tan(3.0f * (180.0f / M_PI));
	const float cotan_clamp_max_rad = 1.0f / std::tan(177.0f * (180.0f / M_PI));
	
	const std::array<float, 3> p0_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangle[0], blockid);
	const std::array<float, 3> p1_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangle[1], blockid);
	const std::array<float, 3> p2_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangle[2], blockid);
	
	const vec3 p0_position {p0_position_arr[0], p0_position_arr[1], p0_position_arr[2]};
	const vec3 p1_position {p1_position_arr[0], p1_position_arr[1], p1_position_arr[2]};
	const vec3 p2_position {p2_position_arr[0], p2_position_arr[1], p2_position_arr[2]};
	
	const std::array<vec3, 3> triangle_positions {
		  p0_position
		, p1_position
		, p2_position
	};
	
	vec3 face_normal;
	vec_cross_vec_3d(face_normal.data_arr(), (triangle_positions[1] - triangle_positions[0]).data_arr(), (triangle_positions[2] - triangle_positions[0]).data_arr());
	
	const float face_normal_length = sqrt(face_normal[0] * face_normal[0] + face_normal[1] * face_normal[1] + face_normal[2] * face_normal[2]);
	
	//Normalize
	face_normal = face_normal / face_normal_length;
	
	const float face_area = 0.5f * face_normal_length;
	
	const vec3 particle_position = triangle_positions[contact_index];
		
	int advection_source_blockno;
	int source_pidib;
	alpha_shapes_fetch_id(particle_buffer, prev_partition, triangle[contact_index], blockid, advection_source_blockno, source_pidib);
	
	auto particle_bin													= alpha_shapes_particle_buffer.ch(_0, particle_buffer.bin_offsets[advection_source_blockno] + source_pidib / config::G_BIN_CAPACITY);
	const int particle_id_in_bin = source_pidib  % config::G_BIN_CAPACITY;
	
	float cosine = (triangle_positions[(contact_index + 1) % 3] - triangle_positions[contact_index]).dot(triangle_positions[(contact_index + 2) % 3] - triangle_positions[contact_index]) / sqrt((triangle_positions[(contact_index + 1) % 3] - triangle_positions[contact_index]).dot(triangle_positions[(contact_index + 1) % 3] - triangle_positions[contact_index]) * (triangle_positions[(contact_index + 2) % 3] - triangle_positions[contact_index]).dot(triangle_positions[(contact_index + 2) % 3] - triangle_positions[contact_index]));
	cosine = std::min(std::max(cosine, -1.0f), 1.0f);
	const float angle = std::acos(cosine);
	
	//Normal
	atomicAdd(&particle_bin.val(_1, particle_id_in_bin), angle * face_normal[0]);
	atomicAdd(&particle_bin.val(_2, particle_id_in_bin), angle * face_normal[1]);
	atomicAdd(&particle_bin.val(_3, particle_id_in_bin), angle * face_normal[2]);
	
	//Gauss curvature
	atomicAdd(&particle_bin.val(_0, particle_id_in_bin), face_area * (1.0f / 3.0f));
	atomicAdd(&particle_bin.val(_4, particle_id_in_bin), angle);
	
	//Mean curvature
	float current_cotan;
	float next_cotan;
	{
		const vec3 a = particle_position - triangle_positions[(contact_index + 2) % 3];
		const vec3 b = triangle_positions[(contact_index + 1) % 3] - triangle_positions[(contact_index + 2) % 3];
		
		vec3 cross;
		vec_cross_vec_3d(cross.data_arr(), a.data_arr(), b.data_arr());
		const float cross_norm = sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);

		current_cotan = a.dot(b) / cross_norm;
		current_cotan = std::min(std::max(current_cotan, cotan_clamp_min_rad), cotan_clamp_max_rad);
		
		next_cotan = a.dot(b) / cross_norm;
		next_cotan = std::min(std::max(next_cotan, cotan_clamp_min_rad), cotan_clamp_max_rad);
	}
	
	const vec3 laplacian = current_cotan * (triangle_positions[contact_index] - particle_position) + next_cotan * (triangle_positions[(contact_index + 1) % 3] - particle_position);
	atomicAdd(&particle_bin.val(_5, particle_id_in_bin), laplacian[0]);
	atomicAdd(&particle_bin.val(_6, particle_id_in_bin), laplacian[1]);
	atomicAdd(&particle_bin.val(_7, particle_id_in_bin), laplacian[2]);
}

template<typename Partition, MaterialE MaterialType>
__forceinline__ __device__ void alpha_shapes_finalize_triangle(const ParticleBuffer<MaterialType> particle_buffer, const Partition prev_partition, AlphaShapesParticleBuffer alpha_shapes_particle_buffer, const int* particle_indices, const std::array<int, 3> triangle, const ivec3 blockid, const int range_start, const int range_end){
	const std::array<float, 3> p0_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangle[0], blockid);
	const std::array<float, 3> p1_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangle[1], blockid);
	const std::array<float, 3> p2_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangle[2], blockid);
	
	const vec3 p0_position {p0_position_arr[0], p0_position_arr[1], p0_position_arr[2]};
	const vec3 p1_position {p1_position_arr[0], p1_position_arr[1], p1_position_arr[2]};
	const vec3 p2_position {p2_position_arr[0], p2_position_arr[1], p2_position_arr[2]};
	
	const std::array<vec3, 3> triangle_positions {
		  p0_position
		, p1_position
		, p2_position
	};
	
	vec3 face_normal;
	vec_cross_vec_3d(face_normal.data_arr(), (triangle_positions[1] - triangle_positions[0]).data_arr(), (triangle_positions[2] - triangle_positions[0]).data_arr());
	
	const float face_normal_length = sqrt(face_normal[0] * face_normal[0] + face_normal[1] * face_normal[1] + face_normal[2] * face_normal[2]);
	
	//Normalize
	face_normal = face_normal / face_normal_length;
	
	const float face_area = 0.5f * face_normal_length;
	
	for(size_t contact_index = 0; contact_index < 3; ++contact_index){
		alpha_shapes_accumulate_triangle_at_vertex(particle_buffer, prev_partition, alpha_shapes_particle_buffer, triangle, blockid, contact_index);
	}
	
	for(int particle_id = range_start; particle_id < range_end; particle_id++) {
		const int current_particle_index = particle_indices[particle_id];
		
		int advection_source_blockno;
		int source_pidib;
		alpha_shapes_fetch_id(particle_buffer, prev_partition, particle_indices[particle_id], blockid, advection_source_blockno, source_pidib);
		
		auto particle_bin													= alpha_shapes_particle_buffer.ch(_0, particle_buffer.bin_offsets[advection_source_blockno] + source_pidib / config::G_BIN_CAPACITY);
		const int particle_id_in_bin = source_pidib  % config::G_BIN_CAPACITY;
	
		
		const std::array<float, 3> particle_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, current_particle_index, blockid);
		const vec3 particle_position {particle_position_arr[0], particle_position_arr[1], particle_position_arr[2]};
		
		//If point is not part of a triangle check if it lies on another triangle; This may be the case for degenerated points
				
		const vec9 barycentric_projection_matrix {
				  triangle_positions[0][0] - triangle_positions[2][0], triangle_positions[0][1] - triangle_positions[2][1], triangle_positions[0][2] - triangle_positions[2][2]
				, triangle_positions[1][0] - triangle_positions[2][0], triangle_positions[1][1] - triangle_positions[2][1], triangle_positions[1][2] - triangle_positions[2][2]
				, -face_normal[0], -face_normal[1], -face_normal[2]
		};
		
		vec3 contact_barycentric;
		solve_linear_system(barycentric_projection_matrix.data_arr(), contact_barycentric.data_arr(),  (particle_position - triangle_positions[2]).data_arr());
		
		//If first two coordinates are bigger than 0 and smaller than one, the projected point is on the triangle; if if the last coordinate is 1 it lies on the plane
		if(contact_barycentric[0] > 0.0f && contact_barycentric[1] > 0.0f && contact_barycentric[2] == 1.0f && (contact_barycentric[0] + contact_barycentric[1]) <= 1.0f){
			//Calculate last barycentric coordinate
			contact_barycentric[2] = 1.0f - contact_barycentric[0] - contact_barycentric[1];
			
			//Accumulate values
			if(contact_barycentric[0] > 0.0f){//Point somewhere in triangle
				//Use face normal
				atomicAdd(&particle_bin.val(_1, particle_id_in_bin), face_normal[0]);
				atomicAdd(&particle_bin.val(_2, particle_id_in_bin), face_normal[1]);
				atomicAdd(&particle_bin.val(_3, particle_id_in_bin), face_normal[2]);
				
				//Gauss curvature
				atomicAdd(&particle_bin.val(_0, particle_id_in_bin), 1.0f);//Just ensure this is not zero
				atomicAdd(&particle_bin.val(_4, particle_id_in_bin), 2.0f * M_PI);
			}else if(contact_barycentric[0] == 1.0f || contact_barycentric[1] == 1.0f || contact_barycentric[0] == 1.0f){//Point on vertex
				int contact_index;
				if(contact_barycentric[0] == 1.0f){
					contact_index = 0;
				}else if(contact_barycentric[1] == 1.0f){
					contact_index = 1;
				}else {//contact_barycentric[2] == 1.0f
					contact_index = 2;
				}
			
				alpha_shapes_accumulate_triangle_at_vertex(particle_buffer, prev_partition, alpha_shapes_particle_buffer, triangle, blockid, contact_index);
			}else{//Point on edge
				//Use half normal
				atomicAdd(&particle_bin.val(_1, particle_id_in_bin), 0.5f * face_normal[0]);
				atomicAdd(&particle_bin.val(_2, particle_id_in_bin), 0.5f * face_normal[1]);
				atomicAdd(&particle_bin.val(_3, particle_id_in_bin), 0.5f * face_normal[2]);
				
				//Gauss curvature
				atomicAdd(&particle_bin.val(_0, particle_id_in_bin), face_area * (1.0f / 3.0f));//Just ensure this is not zero
				atomicAdd(&particle_bin.val(_4, particle_id_in_bin), M_PI);
				
				int opposite_index;
				if(contact_barycentric[0] == 0.0f){
					opposite_index = 0;
				}else if(contact_barycentric[1] == 0.0f){
					opposite_index = 1;
				}else {//contact_barycentric[2] == 0.0f
					opposite_index = 2;
				}
				
				//Mean curvature
				//FIXME: Is this correct?
				const vec3 laplacian = (triangle_positions[(opposite_index + 1) % 3] - particle_position) + (triangle_positions[(opposite_index + 2) % 3] - particle_position) + 2.0f * (triangle_positions[opposite_index] - particle_position);
				atomicAdd(&particle_bin.val(_5, particle_id_in_bin), laplacian[0]);
				atomicAdd(&particle_bin.val(_6, particle_id_in_bin), laplacian[1]);
				atomicAdd(&particle_bin.val(_7, particle_id_in_bin), laplacian[2]);
			}
		}
	}
}

template<typename Partition, MaterialE MaterialType>
__forceinline__ __device__ void alpha_shapes_finalize_particles(const ParticleBuffer<MaterialType> particle_buffer, const Partition prev_partition, AlphaShapesParticleBuffer alpha_shapes_particle_buffer, const int* particle_indices, const ivec3 blockid, const int range_start, const int range_end){
	for(int particle_id = range_start; particle_id < range_end; particle_id++) {
		const int current_particle_index = particle_indices[particle_id];
		
		int advection_source_blockno;
		int source_pidib;
		alpha_shapes_fetch_id(particle_buffer, prev_partition, current_particle_index, blockid, advection_source_blockno, source_pidib);
		
		auto particle_bin													= alpha_shapes_particle_buffer.ch(_0, particle_buffer.bin_offsets[advection_source_blockno] + source_pidib / config::G_BIN_CAPACITY);
		const int particle_id_in_bin = source_pidib  % config::G_BIN_CAPACITY;
		
		float summed_angles = particle_bin.val(_4, particle_id_in_bin);
		float summed_area = particle_bin.val(_0, particle_id_in_bin);
		vec3 summed_laplacians;
		summed_laplacians[0] = particle_bin.val(_5, particle_id_in_bin);
		summed_laplacians[0] = particle_bin.val(_6, particle_id_in_bin);
		summed_laplacians[0] = particle_bin.val(_7, particle_id_in_bin);
		
		AlphaShapesPointType point_type;
		if(summed_area >= 0.0f){
			point_type = AlphaShapesPointType::OUTER_POINT;
		}else{
			//Isolated point or point in shell
			point_type = AlphaShapesPointType::ISOLATED_POINT;//FIXME: All are currently treated as isolated points
			
			summed_angles += 2 * M_PI;
			summed_area += 1.0f;//Just ensure this is not zero
			
			//TODO: Or might be part of curve or thin surface
			//TODO: Decide wheter interior or exterior point
		}
		
		summed_laplacians /= 2.0f * summed_area;
		const float laplacian_norm = sqrt(summed_laplacians[0] * summed_laplacians[0] + summed_laplacians[1] * summed_laplacians[1] + summed_laplacians[2] * summed_laplacians[2]);

		const float gauss_curvature = (2.0f * M_PI - summed_angles) / summed_area;
		const float mean_curvature = 0.5f * laplacian_norm;
		
		particle_bin.val(_0, particle_id_in_bin) = *reinterpret_cast<float*>(&point_type);
		particle_bin.val(_4, particle_id_in_bin) = mean_curvature;
		particle_bin.val(_5, particle_id_in_bin) = gauss_curvature;
	}
}

/*
template<typename Partition, MaterialE MaterialType>
__forceinline__ __device__ void alpha_shapes_finalize_particles(const ParticleBuffer<MaterialType> particle_buffer, const Partition prev_partition, AlphaShapesParticleBuffer alpha_shapes_particle_buffer, const int* particle_indices, const std::array<int, 3>* alpha_triangles, const int alpha_triangle_count, const ivec3 blockid, const int range_start, const int range_end){
	for(int particle_id = range_start; particle_id < range_end; particle_id++) {
		const int current_particle_index = particle_indices[particle_id];
		
		const std::array<float, 3> particle_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, current_particle_index, blockid);
		const vec3 particle_position {particle_position_arr[0], particle_position_arr[1], particle_position_arr[2]};
	
		//TODO: Maybe different curvature calculation?
		vec3 particle_normal {0.0f, 0.0f, 0.0f};
		
		float summed_angles = 0.0f;
		float summed_area = 0.0f;
		vec3 summed_laplacians {0.0f, 0.0f, 0.0f};
	
		std::array<int, MAX_TRIANGLE_COUNT_PER_VERTEX> neighbour_points;
		int contact_triangles_count = 0;
		for(int current_triangle_index = 0; current_triangle_index < alpha_triangle_count; current_triangle_index++) {
			const std::array<float, 3> p0_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, alpha_triangles[current_triangle_index][0], blockid);
			const std::array<float, 3> p1_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, alpha_triangles[current_triangle_index][1], blockid);
			const std::array<float, 3> p2_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, alpha_triangles[current_triangle_index][2], blockid);
			
			const vec3 p0_position {p0_position_arr[0], p0_position_arr[1], p0_position_arr[2]};
			const vec3 p1_position {p1_position_arr[0], p1_position_arr[1], p1_position_arr[2]};
			const vec3 p2_position {p2_position_arr[0], p2_position_arr[1], p2_position_arr[2]};
			
			const std::array<vec3, 3> triangle_positions {
				  p0_position
				, p1_position
				, p2_position
			};
			
			vec3 face_normal;
			vec_cross_vec_3d(face_normal.data_arr(), (triangle_positions[1] - triangle_positions[0]).data_arr(), (triangle_positions[2] - triangle_positions[0]).data_arr());
			
			const float face_normal_length = sqrt(face_normal[0] * face_normal[0] + face_normal[1] * face_normal[1] + face_normal[2] * face_normal[2]);
			
			//Normalize
			face_normal = face_normal / face_normal_length;
			
			const float face_area = 0.5f * face_normal_length;
			
			int contact_index;
			int neighbour_index;
			vec3 contact_barycentric {0.0f, 0.0f, 0.0f};
			bool vertex_contact = false;
			if(alpha_triangles[current_triangle_index][0] == current_particle_index){
				contact_index = 0;
				neighbour_index = 1;
				contact_barycentric[0] = 1.0f;
				vertex_contact = true;
				contact_triangles_count++;
			}else if(alpha_triangles[current_triangle_index][1] == current_particle_index){
				contact_index = 1;
				neighbour_index = 2;
				contact_barycentric[1] = 1.0f;
				vertex_contact = true;
				contact_triangles_count++;
			}else if(alpha_triangles[current_triangle_index][2] == current_particle_index){
				contact_index = 2;
				neighbour_index = 0;
				contact_barycentric[2] = 1.0f;
				vertex_contact = true;
				contact_triangles_count++;
			}else{//If point is not part of a triangle check if it lies on another triangle; This may be the case for degenerated points
				
				const vec9 barycentric_projection_matrix {
						  triangle_positions[0][0] - triangle_positions[2][0], triangle_positions[0][1] - triangle_positions[2][1], triangle_positions[0][2] - triangle_positions[2][2]
						, triangle_positions[1][0] - triangle_positions[2][0], triangle_positions[1][1] - triangle_positions[2][1], triangle_positions[1][2] - triangle_positions[2][2]
						, -face_normal[0], -face_normal[1], -face_normal[2]
				};
				
				solve_linear_system(barycentric_projection_matrix.data_arr(), contact_barycentric.data_arr(),  (particle_position - triangle_positions[2]).data_arr());
				
				//If first two coordinates are bigger than 0 and smaller than one, the projected point is on the triangle; if if the last coordinate is 1 it lies on the plane
				if(contact_barycentric[0] > 0.0f && contact_barycentric[1] > 0.0f && contact_barycentric[2] == 1.0f && (contact_barycentric[0] + contact_barycentric[1]) <= 1.0f){
					//Calculate last barycentric coordinate
					contact_barycentric[2] = 1.0f - contact_barycentric[0] - contact_barycentric[1];
					
					contact_triangles_count++;
					
					//Accumulate values
					if(contact_barycentric[0] > 0.0f){//Point somewhere in triangle
						//Use face normal
						particle_normal += face_normal;
						
						//Gauss curvature
						summed_angles += 2 * M_PI;
						summed_area += 1.0f;//Just ensure this is not zero
						
						//Break cause a point can only lie on one triangle if not on their edges
						break;
					}else if(contact_barycentric[0] == 1.0f || contact_barycentric[1] == 1.0f || contact_barycentric[0] == 1.0f){//Point on vertex
						if(contact_barycentric[0] == 1.0f){
							contact_index = 0;
							neighbour_index = 1;
						}else if(contact_barycentric[1] == 1.0f){
							contact_index = 1;
							neighbour_index = 2;
						}else {//contact_barycentric[2] == 1.0f
							contact_index = 2;
							neighbour_index = 0;
						}
					
						vertex_contact = true;
					}else{//Point on edge
						//Use half normal
						particle_normal += 0.5f * face_normal;
						
						//Gauss curvature
						summed_angles += M_PI;
						summed_area += face_area * (1.0f / 3.0f);
						
						int opposite_index;
						if(contact_barycentric[0] == 0.0f){
							opposite_index = 0;
						}else if(contact_barycentric[1] == 0.0f){
							opposite_index = 1;
						}else {//contact_barycentric[2] == 0.0f
							opposite_index = 2;
						}
						
						//Mean curvature
						//FIXME: Is this correct?
						summed_laplacians += (triangle_positions[(opposite_index + 1) % 3] - particle_position) + (triangle_positions[(opposite_index + 2) % 3] - particle_position) + 2.0f * (triangle_positions[opposite_index] - particle_position);
					}
				}
			}
			
			if(vertex_contact){
				float cosine = (triangle_positions[(contact_index + 1) % 3] - triangle_positions[contact_index]).dot(triangle_positions[(contact_index + 2) % 3] - triangle_positions[contact_index]) / sqrt((triangle_positions[(contact_index + 1) % 3] - triangle_positions[contact_index]).dot(triangle_positions[(contact_index + 1) % 3] - triangle_positions[contact_index]) * (triangle_positions[(contact_index + 2) % 3] - triangle_positions[contact_index]).dot(triangle_positions[(contact_index + 2) % 3] - triangle_positions[contact_index]));
				cosine = std::min(std::max(cosine, -1.0f), 1.0f);
				const float angle = std::acos(cosine);
				
				//Normal
				particle_normal += angle * face_normal;
				
				//Gauss curvature
				summed_angles += angle;
				summed_area += face_area * (1.0f / 3.0f);
				
				//Store neighbour
				if(contact_triangles_count >= MAX_TRIANGLE_COUNT_PER_VERTEX){
					printf("Too much triangles for this vertex: Is %d but may not be greater then %d\n", contact_triangles_count, static_cast<int>(MAX_TRIANGLE_COUNT_PER_VERTEX));
				}
				neighbour_points[contact_triangles_count - 1] = alpha_triangles[current_triangle_index][neighbour_index];
				
				printf("A %d %.28f %.28f # %d - ", current_particle_index, face_area, summed_area, contact_triangles_count);
		
			}
		}
		
		AlphaShapesPointType point_type = AlphaShapesPointType::ISOLATED_POINT;
		
		//Calculate particle states
		if(contact_triangles_count == 0){
			//Isolated point or point in shell
			point_type = AlphaShapesPointType::ISOLATED_POINT;//FIXME: All are currently treated as isolated points
			
			summed_angles += 2 * M_PI;
			summed_area += 1.0f;//Just ensure this is not zero
			
			//TODO: Or might be part of curve or thin surface
			//TODO: Decide wheter interior or exterior point
		}else if(contact_triangles_count > 2){
			point_type = AlphaShapesPointType::OUTER_POINT;
			
			//Sort points by angle so that incident faces are near each other

			//Copied from https://stackoverflow.com/questions/47949485/sorting-a-list-of-3d-points-in-clockwise-order
			//FIXME: Verify this is correct!
			
			const int first_neighbour = neighbour_points[0];
			const std::array<float, 3> first_neighbour_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, first_neighbour, blockid);
			const vec3 first_neighbour_position {first_neighbour_position_arr[0], first_neighbour_position_arr[1], first_neighbour_position_arr[2]};
			
			const vec3 first_diff = first_neighbour_position - particle_position;
			
			vec3 first_cross;
			vec_cross_vec_3d(first_cross.data_arr(), first_diff.data_arr(), particle_position.data_arr());
			
			thrust::sort(thrust::seq, neighbour_points.begin(), neighbour_points.begin() + contact_triangles_count, [&particle_buffer, &prev_partition, &blockid, &particle_position, &first_diff, &first_cross](const int& a, const int& b){
				const std::array<float, 3> neighbour_a_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, a, blockid);
				const std::array<float, 3> neighbour_b_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, b, blockid);
				const vec3 neighbour_a_position {neighbour_a_position_arr[0], neighbour_a_position_arr[1], neighbour_a_position_arr[2]};
				const vec3 neighbour_b_position {neighbour_b_position_arr[0], neighbour_b_position_arr[1], neighbour_b_position_arr[2]};
				
				const vec3 diff_a = neighbour_a_position - particle_position;
				const vec3 diff_b = neighbour_b_position - particle_position;
				
				const float dot_a = first_cross.dot(diff_a);	
				const float dot_b = first_cross.dot(diff_b);
				
				//           h2 > 0     h2 = 0     h2 < 0
				//          ————————   ————————   ————————
				//  h1 > 0 |   *        v1 > v2    v1 > v2
				//  h1 = 0 | v1 < v2       †          *
				//  h1 < 0 | v1 < v2       *          *

				//  *   means we can use the triple product because the (cylindrical)
				//      angle between u1 and u2 is less than π
				//  †   means u1 and u2 are either 0 or π around from the zero reference
				//      in which case u1 < u2 only if dot(u1, r) > 0 and dot(u2, r) < 0
				if(dot_a > 0.0f && dot_b <= 0.0f){
					return false;
				}else if(dot_a <= 0.0f && dot_b > 0.0f){
					return true;
				}else if(dot_a == 0.0f && dot_b == 0.0f){
					return (diff_a.dot(first_diff) > 0 && diff_b.dot(first_diff) < 0);
				}else{
					vec3 cross;
					vec_cross_vec_3d(cross.data_arr(), diff_a.data_arr(), diff_b.data_arr());
					return cross.dot(particle_position) > 0;
				}
			});
			
			for(int current_neighbour_index = 0; current_neighbour_index < contact_triangles_count; current_neighbour_index++) {
				const int prev_neighbour_index = (current_neighbour_index + contact_triangles_count - 1) % contact_triangles_count;
				const int next_neighbour_index = (current_neighbour_index + 1) % contact_triangles_count;
				
				const int prev_neighbour = neighbour_points[prev_neighbour_index];
				const int current_neighbour = neighbour_points[current_neighbour_index];
				const int next_neighbour = neighbour_points[next_neighbour_index];
				
				const std::array<float, 3> prev_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, prev_neighbour, blockid);
				const std::array<float, 3> current_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, current_neighbour, blockid);
				const std::array<float, 3> next_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, next_neighbour, blockid);
				
				const vec3 prev_position {prev_position_arr[0], prev_position_arr[1], prev_position_arr[2]};
				const vec3 current_position {current_position_arr[0], current_position_arr[1], current_position_arr[2]};
				const vec3 next_position {next_position_arr[0], next_position_arr[1], next_position_arr[2]};
				
				
				//Calculate cotans
				const float clamp_min_rad = 1.0f / std::tan(3.0f * (180.0f / M_PI));
				const float clamp_max_rad = 1.0f / std::tan(177.0f * (180.0f / M_PI));
				
				float next_cotan;
				{
					const vec3 a = particle_position - next_position;
					const vec3 b = current_position - next_position;
					
					vec3 cross;
					vec_cross_vec_3d(cross.data_arr(), a.data_arr(), b.data_arr());
					const float cross_norm = sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);

					next_cotan = a.dot(b) / cross_norm;
					next_cotan = std::min(std::max(next_cotan, clamp_min_rad), clamp_max_rad);
				}

				float prev_cotan;
				{
					const vec3 a = particle_position - prev_position;
					const vec3 b = current_position - prev_position;
					
					vec3 cross;
					vec_cross_vec_3d(cross.data_arr(), a.data_arr(), b.data_arr());
					const float cross_norm = sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);

					prev_cotan = a.dot(b) / cross_norm;
					prev_cotan = std::min(std::max(prev_cotan, clamp_min_rad), clamp_max_rad);
				}
				
				summed_laplacians += (next_cotan + prev_cotan) * (current_position - particle_position);
			}
		}
		
		summed_laplacians /= 2.0f * summed_area;
		const float laplacian_norm = sqrt(summed_laplacians[0] * summed_laplacians[0] + summed_laplacians[1] * summed_laplacians[1] + summed_laplacians[2] * summed_laplacians[2]);

		const float gauss_curvature = (2.0f * M_PI - summed_angles) / summed_area;
		const float mean_curvature = 0.5f * laplacian_norm;
		
		if(point_type != AlphaShapesPointType::ISOLATED_POINT){
			printf("B %d # %d %f %f %f %f %f # %.28f %d # %f %f %f - ", current_particle_index, static_cast<int>(point_type), particle_normal[0], particle_normal[1], particle_normal[2], mean_curvature, gauss_curvature, summed_area, contact_triangles_count, summed_laplacians[0], summed_laplacians[1], summed_laplacians[2]);
		}
		
		alpha_shapes_store_particle(particle_buffer, prev_partition, alpha_shapes_particle_buffer, current_particle_index, blockid, point_type, particle_normal.data_arr(), mean_curvature, gauss_curvature);
	}
}
*/


//FIXME: if n+2 (=5) or more points lie on a sphere there are several possible triangulations; Can we ignore this or do we have to handle this and if so how can we do this; Maybe seep line ensures consistency?
template<typename Partition, typename Grid, MaterialE MaterialType>
__global__ void alpha_shapes(const ParticleBuffer<MaterialType> particle_buffer, const Partition prev_partition, const Partition partition, const Grid grid, AlphaShapesParticleBuffer alpha_shapes_particle_buffer, AlphaShapesGridBuffer alpha_shapes_grid_buffer, const unsigned int start_index) {
	//const int src_blockno		   = static_cast<int>(blockIdx.x);
	const ivec3 blockid			   = partition.active_keys[blockIdx.x];
	const ivec3 cellid = blockid * static_cast<int>(config::G_BLOCKSIZE) + ivec3(static_cast<int>(threadIdx.x), static_cast<int>(threadIdx.y), static_cast<int>(threadIdx.z));
	
	const int prev_blockno = prev_partition.query(blockid);
	const int cellno = ((cellid[0] & config::G_BLOCKMASK) << (config::G_BLOCKBITS << 1)) | ((cellid[1] & config::G_BLOCKMASK) << config::G_BLOCKBITS) | (cellid[2] & config::G_BLOCKMASK);
	const int particles_in_cell = particle_buffer.cell_particle_counts[prev_blockno * config::G_BLOCKVOLUME + cellno];
	
	//auto current_particle_data = alpha_shapes_grid_buffer.ch(_0, src_blockno).ch(_0, cellid - blockid);
	//auto current_triangle_data = alpha_shapes_grid_buffer.ch(_0, src_blockno).ch(_1, cellid - blockid);
	//auto current_triangle_data = alpha_shapes_grid_buffer.ch(_0, src_blockno).ch(_0, (cellid - blockid).cast<char>());
	
	//TODO: Use arrays instead?
	//int* particle_indices = &(current_particle_data.val_1d(_0, 0));
	
	//std::array<int, 3>* triangles = reinterpret_cast<std::array<int, 3>*>(current_triangle_data.val_1d(_0, 0));
	//bool* triangles_is_alpha = &(current_triangle_data.val_1d(_1, 0));
	//std::array<int, 3>* alpha_triangles = reinterpret_cast<std::array<int, 3>*>(current_triangle_data.val_1d(_2, 0));
	
	//TODO: If still not enogh memory we can iterate through all particles. Also maybe we can reduce triangle count and maybe merge the arrays or only save alpha triangles; Maybe also we can somehow utilize shared memory?; Also we can split up into several iterations maybe, reusing the same memory; Or first sort globally
	std::array<int, ALPHA_SHAPES_MAX_PARTICLE_COUNT> particle_indices;
	std::array<int, config::G_MAX_PARTICLES_IN_CELL> own_particle_indices;
	
	std::array<std::array<int, 3>, ALPHA_SHAPES_MAX_TRIANGLE_COUNT> triangles;
	std::array<bool, ALPHA_SHAPES_MAX_TRIANGLE_COUNT> triangles_is_alpha;
	
	/*
	constexpr size_t TOTAL_ARRAY_SIZE = sizeof(particle_indices)
			   + sizeof(own_particle_indices)
			   + sizeof(triangles)
			   + sizeof(triangles_is_alpha);
	
	constexpr unsigned long LOCAL_MEMORY_SIZE = (static_cast<size_t>(1) << 17);
			   
	printf("%lu %lu - ", static_cast<unsigned long>(TOTAL_ARRAY_SIZE), static_cast<unsigned long>(LOCAL_MEMORY_SIZE));
	
	static_assert(TOTAL_ARRAY_SIZE < LOCAL_MEMORY_SIZE && "Not enough local memory");
	*/
	
	int own_particle_bucket_size = particles_in_cell;
	
	//If we have no particles in the bucket return
	if(own_particle_bucket_size == 0) {
		return;
	}
	
	//TODO: Smaller alpha based on density or cohesion maybe
	const float alpha = config::MAX_ALPHA;
	
	//FIXME: Actually we only need to handle blocks in radius of sqrt(alpha) around box
	//Fetch particles
	int particle_bucket_size = 0;
	for(int i = -static_cast<int>(ALPHA_SHAPES_KERNEL_SIZE); i <= static_cast<int>(ALPHA_SHAPES_KERNEL_SIZE); ++i){
		for(int j = -static_cast<int>(ALPHA_SHAPES_KERNEL_SIZE); j <= static_cast<int>(ALPHA_SHAPES_KERNEL_SIZE); ++j){
			for(int k = -static_cast<int>(ALPHA_SHAPES_KERNEL_SIZE); k <= static_cast<int>(ALPHA_SHAPES_KERNEL_SIZE); ++k){
				const ivec3 cellid_offset {i, j, k};
				const ivec3 current_cellid = cellid + cellid_offset;
				const int current_cellno = ((current_cellid[0] & config::G_BLOCKMASK) << (config::G_BLOCKBITS << 1)) | ((current_cellid[1] & config::G_BLOCKMASK) << config::G_BLOCKBITS) | (current_cellid[2] & config::G_BLOCKMASK);
				
				const ivec3 current_blockid = current_cellid / static_cast<int>(config::G_BLOCKSIZE);
				const int current_blockno = prev_partition.query(current_blockid);
				const ivec3 blockid_offset = current_blockid - blockid;
				
				//FIXME:Hide conditional in std::max; Add in every loop iteration; Otherwise cuda crashes when using particle_bucket_size for array access; Also this might increase performance
				//For empty blocks (blockno = -1) current_bucket_size will be zero
				const int current_bucket_size = particle_buffer.cell_particle_counts[(std::max(current_blockno, 0) * config::G_BLOCKVOLUME + current_cellno)] * std::min(current_blockno + 1, 1);
				
				alpha_shapes_fetch_particles(particle_buffer, prev_partition, particle_indices.data(), current_bucket_size, current_blockno, current_blockid, blockid_offset, current_cellno, particle_bucket_size);
				
				if(i == 0 && j == 0 && k == 0){
					for(int particle_id = 0; particle_id < own_particle_bucket_size; particle_id++) {
						own_particle_indices[particle_id] = particle_indices[particle_bucket_size + particle_id];
					}
				}
				
				particle_bucket_size += current_bucket_size;
			}
		}
	}
	
	//FIXME: Actually we only need to handle blocks in radius of sqrt(alpha) around box
	//Filter by max distance; This cannot affect particles of current cell
	const vec3 cell_center = (cellid + 2.0f + vec3(grid.get_offset()[0], grid.get_offset()[1], grid.get_offset()[2]) * config::G_BLOCKSIZE) * config::G_DX;//0->2.0 1->3.0 ...; 1.5 is lower bound of block 0, 5.5 is lower bound of block 1, ...; 1.5 is lower bound of cell 0, 2.5 is lower bound of cell 1, ...
	for(int particle_id = 0; particle_id < particle_bucket_size; particle_id++) {
		const std::array<float, 3> particle_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, particle_indices[particle_id], blockid);
		const vec3 particle_position {particle_position_arr[0], particle_position_arr[1], particle_position_arr[2]};

		//Calculate distance to center
		const vec3 diff = cell_center - particle_position;
		const float squared_distance = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
		
		//Cell boundary + 2.0f * alpha as max radius
		if(squared_distance > (0.75f * config::G_DX * config::G_DX + 2.0f * alpha)){
			//Remove by exchanging with last element
			thrust::swap(particle_indices[particle_id], particle_indices[particle_bucket_size - 1]);
			
			//Decrease count
			particle_bucket_size--;
			
			//Revisit swapped particle
			particle_id--;
		}
	}
	
	//Sort by ascending x
	thrust::sort(thrust::seq, particle_indices.begin(), particle_indices.begin() + particle_bucket_size, [&particle_buffer, &prev_partition, &blockid](const int& a, const int& b){
		const std::array<float, 3> particle_position_arr_a = alpha_shapes_get_particle_position(particle_buffer, prev_partition, a, blockid);
		const std::array<float, 3> particle_position_arr_b = alpha_shapes_get_particle_position(particle_buffer, prev_partition, b, blockid);
		
		return particle_position_arr_a[0] < particle_position_arr_b[0];
	});
	
	thrust::sort(thrust::seq, own_particle_indices.begin(), own_particle_indices.begin() + own_particle_bucket_size, [&particle_buffer, &prev_partition, &blockid](const int& a, const int& b){
		const std::array<float, 3> particle_position_arr_a = alpha_shapes_get_particle_position(particle_buffer, prev_partition, a, blockid);
		const std::array<float, 3> particle_position_arr_b = alpha_shapes_get_particle_position(particle_buffer, prev_partition, b, blockid);
		
		return particle_position_arr_a[0] < particle_position_arr_b[0];
	});
	
	//Build delaunay triangulation with this points and keep all these intersecting the node
	
	//Create first triangle
	const bool found_initial_triangle = alpha_shapes_get_first_triangle(particle_buffer, prev_partition, particle_indices.data(), blockid, particle_bucket_size, triangles[0]);
	
	bool found_initial_tetrahedron = false;
	float minimum_x;
	float maximum_x;
	if(found_initial_triangle){
		//Create first tetrahedron
		{
			const std::array<int, 3> current_triangle = triangles[0];
			
			const std::array<float, 3> p0_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, current_triangle[0], blockid);
			std::array<float, 3> p1_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, current_triangle[1], blockid);
			std::array<float, 3> p2_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, current_triangle[2], blockid);
			
			const vec3 p0_position {p0_position_arr[0], p0_position_arr[1], p0_position_arr[2]};
			vec3 p1_position {p1_position_arr[0], p1_position_arr[1], p1_position_arr[2]};
			vec3 p2_position {p2_position_arr[0], p2_position_arr[1], p2_position_arr[2]};
			
			//Find smallest delaunay tetrahedron
			std::array<std::array<float, 3>, 3> current_triangle_positions {
				  p0_position.data_arr()
				, p1_position.data_arr()
				, p2_position.data_arr()
			};
			
			int p3_id;
			found_initial_tetrahedron = alpha_shapes_get_fourth_point(particle_buffer, prev_partition, particle_indices.begin(), particle_indices.begin() + particle_bucket_size, blockid, current_triangle_positions, p3_id);
			
			//If found, current normal is pointing inwards (towards the forth point), so it has to be flipped. Otherwise we only flip the normal temporarly to find the forth point
			if(!found_initial_tetrahedron){
				//Flip face normal
				current_triangle_positions = {
					  p0_position.data_arr()
					, p2_position.data_arr()
					, p1_position.data_arr()
				};

				found_initial_tetrahedron = alpha_shapes_get_fourth_point(particle_buffer, prev_partition, particle_indices.begin(), particle_indices.begin() + particle_bucket_size, blockid, current_triangle_positions, p3_id);
			}else{
				//Flip triangle order to get correct normal direction pointing outwards
				thrust::swap(triangles[0][1], triangles[0][2]);
				thrust::swap(p1_position_arr, p2_position_arr);
				thrust::swap(p1_position, p2_position);
			}
			
			//NOTE: current_triangle_positions still has inwards pointing normal here
			
			if(found_initial_tetrahedron){
				const std::array<float, 3> p3_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, p3_id, blockid);
				const vec3 p3_position {p3_position_arr[0], p3_position_arr[1], p3_position_arr[2]};
			
				//Check alpha shape condition and mark faces accordingly
				std::array<float, 3> sphere_center;
				float radius;
				alpha_shapes_get_circumsphere(p0_position_arr, p1_position_arr, p2_position_arr, p3_position_arr, sphere_center, radius);
				const bool is_alpha = (radius * radius <= alpha);
				
				//Add triangles
				//Ensure correct order (current_triangle cw normal points outwards)
				triangles[1][0] = current_triangle[0];
				triangles[1][1] = current_triangle[1];
				triangles[1][2] = p3_id;
				triangles[2][0] = current_triangle[1];
				triangles[2][1] = current_triangle[2];
				triangles[2][2] = p3_id;
				triangles[3][0] = current_triangle[2];
				triangles[3][1] = current_triangle[0];
				triangles[3][2] = p3_id;
				
				triangles_is_alpha[0] = is_alpha;
				triangles_is_alpha[1] = is_alpha;
				triangles_is_alpha[2] = is_alpha;
				triangles_is_alpha[3] = is_alpha;
				
				//Init bounds
				minimum_x = sphere_center[0] - radius;
				maximum_x = sphere_center[0] + radius;
			}
		}
	}
	//printf("B %.28f %.28f - ", minimum_x, maximum_x);
	
	int active_particles_start = 0;
	int active_particles_count = 0;
	
	int own_active_particles_start = 0;
	int own_active_particles_count = 0;
	int own_last_active_particles_start = 0;
	
	int current_triangle_count = 0;
	
	if(found_initial_tetrahedron){
		//Init sweep line
		
		//Move upper bound; Activate additional particles based on range to new triangles
		for(int particle_id = active_particles_start + active_particles_count; particle_id < particle_bucket_size; particle_id++) {
			const std::array<float, 3> particle_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, particle_indices[particle_id], blockid);
			if(particle_position_arr[0] > maximum_x){
				break;
			}
			active_particles_count++;
		}
		
		for(int particle_id = own_active_particles_start + own_active_particles_count; particle_id < own_particle_bucket_size; particle_id++) {
			const std::array<float, 3> particle_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, own_particle_indices[particle_id], blockid);
			if(particle_position_arr[0] > maximum_x){
				break;
			}
			own_active_particles_count++;
		}
		
		current_triangle_count = 4;
		int next_triangle_count = 0;
		
		while(own_active_particles_count > 0){
			minimum_x = std::numeric_limits<float>::max();
			maximum_x = std::numeric_limits<float>::min();
			
			while(current_triangle_count > 0){
				const std::array<int, 3> current_triangle = triangles[0];
				
				const std::array<float, 3> p0_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, current_triangle[0], blockid);
				const std::array<float, 3> p1_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, current_triangle[1], blockid);
				const std::array<float, 3> p2_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, current_triangle[2], blockid);
				
				const vec3 p0_position {p0_position_arr[0], p0_position_arr[1], p0_position_arr[2]};
				const vec3 p1_position {p1_position_arr[0], p1_position_arr[1], p1_position_arr[2]};
				const vec3 p2_position {p2_position_arr[0], p2_position_arr[1], p2_position_arr[2]};
				
				//Find smallest delaunay tetrahedron
				const std::array<std::array<float, 3>, 3> current_triangle_positions {
					  p0_position.data_arr()
					, p1_position.data_arr()
					, p2_position.data_arr()
				};
				
				int p3_id;
				const bool found = alpha_shapes_get_fourth_point(particle_buffer, prev_partition, particle_indices.begin() + active_particles_start, particle_indices.begin() + active_particles_start + active_particles_count, blockid, current_triangle_positions, p3_id);
				const std::array<float, 3> p3_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, p3_id, blockid);
				const vec3 p3_position {p3_position_arr[0], p3_position_arr[1], p3_position_arr[2]};
				
				//If no tetrahedron could be created we have a boundary face of the convex hull
				if(found){
					//Check alpha shape condition and mark faces accordingly
					std::array<float, 3> sphere_center;
					float radius;
					alpha_shapes_get_circumsphere(p0_position_arr, p1_position_arr, p2_position_arr, p3_position_arr, sphere_center, radius);
					const bool is_alpha = (radius * radius <= alpha);
					
					//Check contact conditions and update triangle list
					//NOTE: This changes the triangle counts
					alpha_shapes_check_contact_condition(particle_buffer, prev_partition, alpha_shapes_particle_buffer, own_particle_indices.data(), triangles.data(), triangles_is_alpha.data(), current_triangle_count, next_triangle_count, blockid, p3_id, is_alpha, own_last_active_particles_start, own_active_particles_start + own_active_particles_count);
					
					//Update bounds
					//printf("C %f %f %f # %f - ", sphere_center[0], sphere_center[1], sphere_center[2], radius);
					minimum_x = std::min(minimum_x, sphere_center[0] - radius);
					maximum_x = std::max(maximum_x, sphere_center[0] + radius);
				}else{
					//Remove the face and move it to alpha if it is alpha
					
					if(triangles_is_alpha[0]){
						alpha_shapes_finalize_triangle(particle_buffer, prev_partition, alpha_shapes_particle_buffer, own_particle_indices.data(), current_triangle, blockid, own_last_active_particles_start, own_active_particles_start + own_active_particles_count);
					}
					
					//Swap contacting triangle to the end
					thrust::swap(triangles[0], triangles[current_triangle_count - 1]);
					thrust::swap(triangles_is_alpha[0], triangles_is_alpha[current_triangle_count - 1]);
					
					//Decrease triangle count
					current_triangle_count--;
				}
			}
			
			//printf("A %d %d %d # %.28f %.28f - ", current_triangle_count, active_particles_start, active_particles_count, minimum_x, maximum_x);
			
			//All triangles have been handled, either checked for tetrahedron or removed due to contacting
			
			//Swap triangle lists
			current_triangle_count = next_triangle_count;
			next_triangle_count = 0;
			
			//Move sweep line
			
			//Move lower bound; Remopve particles that are out of range
			for(int particle_id = active_particles_start; particle_id < active_particles_start + active_particles_count; particle_id++) {
				const std::array<float, 3> particle_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, own_particle_indices[particle_id], blockid);
				if(particle_position_arr[0] >= minimum_x){
					break;
				}
				active_particles_start++;
				active_particles_count--;
			}
			
			//Move upper bound; Activate additional particles based on range to new triangles
			for(int particle_id = active_particles_start + active_particles_count; particle_id < particle_bucket_size; particle_id++) {
				const std::array<float, 3> particle_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, own_particle_indices[particle_id], blockid);
				if(particle_position_arr[0] > maximum_x){
					break;
				}
				active_particles_count++;
			}
			
			//Move lower bound; Remopve particles that are out of range
			for(int particle_id = own_active_particles_start; particle_id < own_active_particles_start + own_active_particles_count; particle_id++) {
				const std::array<float, 3> particle_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, own_particle_indices[particle_id], blockid);
				if(particle_position_arr[0] >= minimum_x){
					break;
				}
				own_active_particles_start++;
				own_active_particles_count--;
			}
			
			//Move upper bound; Activate additional particles based on range to new triangles
			for(int particle_id = own_active_particles_start + own_active_particles_count; particle_id < own_particle_bucket_size; particle_id++) {
				const std::array<float, 3> particle_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, own_particle_indices[particle_id], blockid);
				if(particle_position_arr[0] > maximum_x){
					break;
				}
				own_active_particles_count++;
			}
			
			//Finalize particles and faces if possible (based on sweep line)
			alpha_shapes_finalize_particles(particle_buffer, prev_partition, alpha_shapes_particle_buffer, own_particle_indices.data(), blockid, own_last_active_particles_start, own_active_particles_start);
			
			//Save last bounds
			own_last_active_particles_start = own_active_particles_start;
		}
	}else{
		own_active_particles_start = own_particle_bucket_size;//Handle all particles
	}
	
	//Add all faces left over at the end to alpha_triangles when no more points are found if they are marked as alpha.
	//NOTE: We are counting backwards to avoid overriding triangles
	for(int current_triangle_index = current_triangle_count - 1; current_triangle_index >=0; current_triangle_index--) {
		if(triangles_is_alpha[current_triangle_index]){
			alpha_shapes_finalize_triangle(particle_buffer, prev_partition, alpha_shapes_particle_buffer, own_particle_indices.data(), triangles[current_triangle_index], blockid, own_last_active_particles_start, own_active_particles_start);
		}
	}
	
	//Finalize all particles left
	alpha_shapes_finalize_particles(particle_buffer, prev_partition, alpha_shapes_particle_buffer, own_particle_indices.data(), blockid, own_last_active_particles_start, own_active_particles_start);
}

template<typename Partition, MaterialE MaterialType>
__global__ void clear_alpha_shapes_particle_buffer(const ParticleBuffer<MaterialType> particle_buffer, const ParticleBuffer<MaterialType> next_particle_buffer, const Partition prev_partition, AlphaShapesParticleBuffer alpha_shapes_particle_buffer){
	const int src_blockno		   = static_cast<int>(blockIdx.x);
	const int particle_bucket_size = next_particle_buffer.particle_bucket_sizes[src_blockno];
	
	//If we have no particles in the bucket return
	if(particle_bucket_size == 0) {
		return;
	}
	
	for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < particle_bucket_size; particle_id_in_block += static_cast<int>(blockDim.x)) {
		auto particle_bin													= alpha_shapes_particle_buffer.ch(_0, next_particle_buffer.bin_offsets[src_blockno] + particle_id_in_block / config::G_BIN_CAPACITY);
		//point_type/summed_area
		particle_bin.val(_0, particle_id_in_block % config::G_BIN_CAPACITY) = 0;
		//normal
		particle_bin.val(_1, particle_id_in_block % config::G_BIN_CAPACITY) = 0.0f;
		particle_bin.val(_2, particle_id_in_block % config::G_BIN_CAPACITY) = 0.0f;
		particle_bin.val(_3, particle_id_in_block % config::G_BIN_CAPACITY) = 0.0f;
		//mean_curvature/summed_angle
		particle_bin.val(_4, particle_id_in_block % config::G_BIN_CAPACITY) = 0.0f;
		//gauss_curvature/summed_laplacian
		particle_bin.val(_5, particle_id_in_block % config::G_BIN_CAPACITY) = 0.0f;
		particle_bin.val(_6, particle_id_in_block % config::G_BIN_CAPACITY) = 0.0f;
		particle_bin.val(_7, particle_id_in_block % config::G_BIN_CAPACITY) = 0.0f;
	}
}

//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers, readability-identifier-naming, misc-definitions-in-headers)
}// namespace mn

#endif