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

constexpr float ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD = 1e-9;//FIXME:1e-9;//TODO: Maybe adjust threshold
constexpr float ALPHA_SHAPES_LINE_DISTANCE_TEST_THRESHOLD = 1e-9;//TODO: Maybe adjust threshold
constexpr float ALPHA_SHAPES_IN_SPHERE_THRESHOLD = 0.0f;//FIXME:1e-9;//TODO: Maybe adjust threshold;
constexpr float ALPHA_SHAPES_MIN_SPHERE_THRESHOLD = 0.0f;//FIXME:1e-9;//TODO: Maybe adjust threshold;
constexpr float ALPHA_SHAPES_IN_SPHERE_LAMBDA_THRESHOLD = 0.0f;//FIXME:1e-9;//TODO: Maybe adjust threshold;
constexpr float ALPHA_SHAPES_MIN_SPHERE_RADIUS_THRESHOLD = 0.0f;//FIXME:1e-9;//TODO: Maybe adjust threshold;
constexpr float ALPHA_SHAPES_IN_SPHERE_RADIUS_THRESHOLD = 0.0f;//FIXME:1e-9;//TODO: Maybe adjust threshold;

//NOTE: Actually this should equal the max value of particles as theoretically all particles can lie in the same circumsphere
constexpr __device__ size_t ALPHA_SHAPES_MAX_CIRCUMSPHERE_POINTS = 100;//FIXME:20;//TODO:Set to correct value

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

__forceinline__ __device__ void alpha_shapes_get_circumcircle(const std::array<float, 3>& a, const std::array<float, 3>& b, const std::array<float, 3>& c, std::array<float, 3>& center, float& radius){
	constexpr size_t CG_STEPS = 32;
	constexpr float CG_EPSILON = 1.0f;
	 
	const mn::vec<float, 3> a_vec{a[0], a[1], a[2]};
	const mn::vec<float, 3> b_vec{b[0], b[1], b[2]};
	const mn::vec<float, 3> c_vec{c[0], c[1], c[2]};
	
	//Iteratively recalculating radius and center descending towards smaller radii
	//Actually Feasible direction approach from https://www.jstor.org/stable/pdf/170443.pdf?refreqid=excelsior%3A91bcaa67290275ae8fabb8cb0c805d21&ab_segments=&origin=&initiator=&acceptTC=1 / An algorithm for the minimax Weber problem
	float r;
	float r_a;
	float r_b;
	float r_c;
	mn::vec<float, 3> center_vec = (a_vec + b_vec + c_vec) * (1.0f / 3.0f);
	for(size_t i = 0; i < CG_STEPS; ++i){
		//Recalculate radius
		r_a = std::sqrt((center_vec - a_vec).dot(center_vec - a_vec));
		r_b = std::sqrt((center_vec - b_vec).dot(center_vec - b_vec));
		r_c = std::sqrt((center_vec - c_vec).dot(center_vec - c_vec));
		
		const float r_min = std::min(std::min(r_a, r_b), r_c);
		const float r_max = std::max(std::max(r_a, r_b), r_c);
		
		float step_size = std::numeric_limits<float>::max();
		mn::vec<float, 3> direction {0.0f, 0.0f, 0.0f};
		if(r_max - r_a <= CG_EPSILON){
			if(r_max - r_a > 0.0f){
				step_size = std::min(step_size, r_max - r_a);
			}
			direction += (a_vec - center_vec) / std::sqrt(r_a);
		}
		if(r_max - r_b <= CG_EPSILON){
			if(r_max - r_b > 0.0f){
				step_size = std::min(step_size, r_max - r_b);
			}
			direction += (b_vec - center_vec) / std::sqrt(r_b);
		}
		if(r_max - r_c <= CG_EPSILON){
			if(r_max - r_c > 0.0f){
				step_size = std::min(step_size, r_max - r_c);
			}
			direction += (c_vec - center_vec) / std::sqrt(r_c);
		}
		
		//Convergated (only r_max == r_a)
		if(step_size == std::numeric_limits<float>::max()){
			break;
		}
		
		//Recalculate center
		center_vec += direction * step_size;
	}
	
	//Recalculate radius
	r_a = std::sqrt((center_vec - a_vec).dot(center_vec - a_vec));
	r_b = std::sqrt((center_vec - b_vec).dot(center_vec - b_vec));
	r_c = std::sqrt((center_vec - c_vec).dot(center_vec - c_vec));
	
	r = (r_a + r_b + r_c) * (1.0f / 3.0f);
	//r = r_max;
	
	center = center_vec.data_arr();
	radius = r;
}

__forceinline__ __device__ void alpha_shapes_get_circumsphere(const std::array<float, 3>& a, const std::array<float, 3>& b, const std::array<float, 3>& c, const std::array<float, 3>& d, std::array<float, 3>& center, float& radius){
	constexpr size_t CG_STEPS = 32;
	constexpr float CG_EPSILON = 1.0f;
	 
	const mn::vec<float, 3> a_vec{a[0], a[1], a[2]};
	const mn::vec<float, 3> b_vec{b[0], b[1], b[2]};
	const mn::vec<float, 3> c_vec{c[0], c[1], c[2]};
	const mn::vec<float, 3> d_vec{d[0], d[1], d[2]};
	
	//Iteratively recalculating radius and center descending towards smaller radii
	//Actually Feasible direction approach from https://www.jstor.org/stable/pdf/170443.pdf?refreqid=excelsior%3A91bcaa67290275ae8fabb8cb0c805d21&ab_segments=&origin=&initiator=&acceptTC=1 / An algorithm for the minimax Weber problem
	float r;
	float r_a;
	float r_b;
	float r_c;
	float r_d;
	mn::vec<float, 3> center_vec = (a_vec + b_vec + c_vec + d_vec) * 0.25f;
	for(size_t i = 0; i < CG_STEPS; ++i){
		//Recalculate radius
		r_a = std::sqrt((center_vec - a_vec).dot(center_vec - a_vec));
		r_b = std::sqrt((center_vec - b_vec).dot(center_vec - b_vec));
		r_c = std::sqrt((center_vec - c_vec).dot(center_vec - c_vec));
		r_d = std::sqrt((center_vec - d_vec).dot(center_vec - d_vec));
		
		const float r_min = std::min(std::min(r_a, r_b), std::min(r_c, r_d));
		const float r_max = std::max(std::max(r_a, r_b), std::max(r_c, r_d));
		
		float step_size = std::numeric_limits<float>::max();
		mn::vec<float, 3> direction {0.0f, 0.0f, 0.0f};
		if(r_max - r_a <= CG_EPSILON){
			if(r_max - r_a > 0.0f){
				step_size = std::min(step_size, r_max - r_a);
			}
			direction += (a_vec - center_vec) / std::sqrt(r_a);
		}
		if(r_max - r_b <= CG_EPSILON){
			if(r_max - r_b > 0.0f){
				step_size = std::min(step_size, r_max - r_b);
			}
			direction += (b_vec - center_vec) / std::sqrt(r_b);
		}
		if(r_max - r_c <= CG_EPSILON){
			if(r_max - r_c > 0.0f){
				step_size = std::min(step_size, r_max - r_c);
			}
			direction += (c_vec - center_vec) / std::sqrt(r_c);
		}
		if(r_max - r_d <= CG_EPSILON){
			if(r_max - r_d > 0.0f){
				step_size = std::min(step_size, r_max - r_d);
			}
			direction += (d_vec - center_vec) / std::sqrt(r_d);
		}
		
		//Convergated (only r_max == r_a)
		if(step_size == std::numeric_limits<float>::max()){
			break;
		}
		
		//Recalculate center
		center_vec += direction * step_size;
		
		//TODO: Use inverse squareroot
		//center_vec = (r * ((1.0f / r_a) + (1.0f / r_b) + (1.0f / r_c) + (1.0f / r_d)) - 3.0f) * center_vec + summed - r * ((1.0f / r_a) * a_vec + (1.0f / r_b) * b_vec  + (1.0f / r_c) * c_vec  + (1.0f / r_d) * d_vec);
		//center_vec += (1.0f - (r / r_a0)) * ca + (1.0f - (r / r_b0)) * cb + (1.0f - (r / r_c0)) * cc + (1.0f - (r / r_d0)) * cd;
	}
	
	//Recalculate radius
	r_a = std::sqrt((center_vec - a_vec).dot(center_vec - a_vec));
	r_b = std::sqrt((center_vec - b_vec).dot(center_vec - b_vec));
	r_c = std::sqrt((center_vec - c_vec).dot(center_vec - c_vec));
	r_d = std::sqrt((center_vec - d_vec).dot(center_vec - d_vec));
	
	r = (r_a + r_b + r_c + r_d) * 0.25f;
	//r = r_max;
	
	center = center_vec.data_arr();
	radius = r;
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
		if(particle_position[0] != p0_position[0] || particle_position[1] != p0_position[1] || particle_position[2] != p0_position[2]){
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
		
		const vec3 normal = (p1_position - p0_position) / std::sqrt((p1_position - p0_position).dot(p1_position - p0_position));
		
		//Find smallest meridian sphere
		float current_smallest_meridian_sphere_radius = std::numeric_limits<float>::max();
		int p2;
		for(int particle_id = 1; particle_id < count; particle_id++) {
			const std::array<float, 3> particle_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, particle_indices[particle_id], blockid);
			const vec3 particle_position {particle_position_arr[0], particle_position_arr[1], particle_position_arr[2]};
			
			const vec3 diff_to_line = (particle_position - p0_position) - (normal).dot(particle_position - p0_position) * normal;
			const float squared_distance_to_line = diff_to_line[0] * diff_to_line[0] + diff_to_line[1] * diff_to_line[1] + diff_to_line[2] * diff_to_line[2];
			
			//Skip particles on same line
			if(squared_distance_to_line > ALPHA_SHAPES_LINE_DISTANCE_TEST_THRESHOLD){
				//Calculate meridian sphere radius
				vec3 circle_center;
				float circle_radius;
				alpha_shapes_get_circumcircle(p0_position.data_arr(), p1_position.data_arr(), particle_position.data_arr(), circle_center.data_arr(), circle_radius);
				
				if(circle_radius < current_smallest_meridian_sphere_radius){
					current_smallest_meridian_sphere_radius = circle_radius;
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
__forceinline__ __device__ bool alpha_shapes_get_fourth_point(const ParticleBuffer<MaterialType> particle_buffer, const Partition prev_partition, AlphaShapesParticleBuffer alpha_shapes_particle_buffer, const int tmpabc, const int test_index, const ForwardIterator& begin, const ForwardIterator& end, const ivec3 blockid, const std::array<std::array<float, 3>, 3>& triangle_positions, int& point_id, const bool do_halfspace_test){
	const std::array<float, 3>& triangle_normal = alpha_shapes_calculate_triangle_normal(triangle_positions);
	
	const vec3 triangle_normal_vec {triangle_normal[0], triangle_normal[1], triangle_normal[2]};
	
	const vec3 p0_position {triangle_positions[0][0], triangle_positions[0][1], triangle_positions[0][2]};
	
	//Only search in normal direction
	ForwardIterator p3_iter = thrust::min_element(thrust::seq, begin, end, [&particle_buffer, &prev_partition, &blockid, &triangle_positions, &do_halfspace_test, &triangle_normal_vec, &p0_position](const int& a, const int& b){
		const std::array<float, 3> particle_position_arr_a = alpha_shapes_get_particle_position(particle_buffer, prev_partition, a, blockid);
		const std::array<float, 3> particle_position_arr_b = alpha_shapes_get_particle_position(particle_buffer, prev_partition, b, blockid);
		
		const vec3 particle_position_a {particle_position_arr_a[0], particle_position_arr_a[1], particle_position_arr_a[2]};
		const vec3 particle_position_b {particle_position_arr_b[0], particle_position_arr_b[1], particle_position_arr_b[2]};
		
		//Test if in half_space; Also sorts out particles that lie in a plane with the triangle
		bool in_halfspace_a = triangle_normal_vec.dot(particle_position_a - p0_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
		bool in_halfspace_b = triangle_normal_vec.dot(particle_position_b - p0_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
		
		bool ret = false;
		//Triangle positions are always bigger
		if(
			   (particle_position_arr_a[0] == triangle_positions[0][0] && particle_position_arr_a[1] == triangle_positions[0][1] && particle_position_arr_a[2] == triangle_positions[0][2])
			|| (particle_position_arr_a[0] == triangle_positions[1][0] && particle_position_arr_a[1] == triangle_positions[1][1] && particle_position_arr_a[2] == triangle_positions[1][2])
			|| (particle_position_arr_a[0] == triangle_positions[2][0] && particle_position_arr_a[1] == triangle_positions[2][1] && particle_position_arr_a[2] == triangle_positions[2][2])
		){
			ret = false;
		}else if(
			   (particle_position_arr_b[0] == triangle_positions[0][0] && particle_position_arr_b[1] == triangle_positions[0][1] && particle_position_arr_b[2] == triangle_positions[0][2])
			|| (particle_position_arr_b[0] == triangle_positions[1][0] && particle_position_arr_b[1] == triangle_positions[1][1] && particle_position_arr_b[2] == triangle_positions[1][2])
			|| (particle_position_arr_b[0] == triangle_positions[2][0] && particle_position_arr_b[1] == triangle_positions[2][1] && particle_position_arr_b[2] == triangle_positions[2][2])
		){
			ret = true;
		} else if(!do_halfspace_test || (in_halfspace_a && in_halfspace_b)){
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
			
			//a lies in sphere_b, but b does not lie in sphere_a
			ret = ((squared_distance_b - (radius_a * radius_a)) > ALPHA_SHAPES_IN_SPHERE_THRESHOLD && (squared_distance_a - (radius_b * radius_b)) <= ALPHA_SHAPES_IN_SPHERE_THRESHOLD);
		}else if(in_halfspace_a){
			/*//Calculate delaunay spheres
			vec3 sphere_center_a;
			float radius_a;
			alpha_shapes_get_circumsphere(triangle_positions[0], triangle_positions[1], triangle_positions[2], particle_position_arr_a, sphere_center_a.data_arr(), radius_a);
			
			//Smaller delaunay sphere is the on that does not contain the other point;
			const vec3 diff_b = sphere_center_a - particle_position_b;
			const float squared_distance_b = diff_b[0] * diff_b[0] + diff_b[1] * diff_b[1] + diff_b[2] * diff_b[2];
			
			//If at least one is not in halfspace, consider a smaller if it is in halfspace (and b is not) and if the circumsphere around a does not contain b; otherwise a is not smaller (but equal)
			//FIXME:ret = ((squared_distance_b - (radius_a * radius_a)) > ALPHA_SHAPES_IN_SPHERE_THRESHOLD);*/
			ret = true;
		}
		return ret;
	});
	
	point_id = *p3_iter;
	
	const std::array<float, 3> p3_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, point_id, blockid);
	const vec3 p3_position {p3_position_arr[0], p3_position_arr[1], p3_position_arr[2]};
	
	//Test if in half_space; Also sorts out particles that lie in a plane with the triangle
	bool found = !do_halfspace_test || (triangle_normal_vec.dot(p3_position - p0_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD);
	
	//TODO: Do some smarter test, maybe at sorting time (currently we do not check that correctly)
	//Check if other point lies in circumsphere
	if(found){
		vec3 sphere_center;
		float radius;
		alpha_shapes_get_circumsphere(triangle_positions[0], triangle_positions[1], triangle_positions[2], p3_position_arr, sphere_center.data_arr(), radius);
		
		for(ForwardIterator i = begin; i != end; ++i){
			const std::array<float, 3> particle_position_arr_a = alpha_shapes_get_particle_position(particle_buffer, prev_partition, *i, blockid);
			
			//Don't test tetrahedron corner points
			if(
				   (particle_position_arr_a[0] != triangle_positions[0][0] || particle_position_arr_a[1] != triangle_positions[0][1] || particle_position_arr_a[2] != triangle_positions[0][2])
				&& (particle_position_arr_a[0] != triangle_positions[1][0] || particle_position_arr_a[1] != triangle_positions[1][1] || particle_position_arr_a[2] != triangle_positions[1][2])
				&& (particle_position_arr_a[0] != triangle_positions[2][0] || particle_position_arr_a[1] != triangle_positions[2][1] || particle_position_arr_a[2] != triangle_positions[2][2])
				&& (particle_position_arr_a[0] != p3_position_arr[0] || particle_position_arr_a[1] != p3_position_arr[1] || particle_position_arr_a[2] != p3_position_arr[2])
			){
			
				const vec3 particle_position_a {particle_position_arr_a[0], particle_position_arr_a[1], particle_position_arr_a[2]};
				
				const vec3 diff_a = sphere_center - particle_position_a;
				const float squared_distance_a = diff_a[0] * diff_a[0] + diff_a[1] * diff_a[1] + diff_a[2] * diff_a[2];
				
				if(tmpabc == test_index){
					//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, *i, blockid, 5, (diff_a * 100.0f).data_arr(), 0.0f, squared_distance_a);
				}
				
				if((squared_distance_a - (radius * radius)) <= ALPHA_SHAPES_IN_SPHERE_THRESHOLD){
					vec3 sphere_center_a;
					float radius_a;
					alpha_shapes_get_circumsphere(triangle_positions[0], triangle_positions[1], triangle_positions[2], particle_position_arr_a, sphere_center_a.data_arr(), radius_a);
					
					const vec3 diff = sphere_center_a - p3_position;
					const float squared_distance = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
					
					bool in_halfspace_a = triangle_normal_vec.dot(particle_position_a - p0_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
					bool in_halfspace = triangle_normal_vec.dot(p3_position - p0_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
					
					//printf("%d %d # %d %d # %.28f %.28f %.28f %.28f - ", point_id, *i, (in_halfspace ? 1 : 0), (in_halfspace_a ? 1 : 0), (radius * radius), (radius_a * radius_a), squared_distance, squared_distance_a);
					//found = false;
					//found = found && !in_halfspace_a;//Only test points in halfspace
					//break;
				}
			}
		}
		
		if(tmpabc == test_index){
			const vec3 diff_a = sphere_center - p3_position;
			const float squared_distance_a = diff_a[0] * diff_a[0] + diff_a[1] * diff_a[1] + diff_a[2] * diff_a[2];
			//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, point_id, blockid, 4, (diff_a * 100.0f).data_arr(), 0.0f, squared_distance_a);
		}
	}
	
	
		
	
	return found;
}

template<typename Partition, MaterialE MaterialType>
__forceinline__ __device__ void alpha_shapes_check_contact_condition(const ParticleBuffer<MaterialType> particle_buffer, const Partition prev_partition, AlphaShapesParticleBuffer alpha_shapes_particle_buffer, const int* own_particle_indices, std::array<int, 3>* triangles, bool* triangles_is_alpha, int* temporary_hull_triangles, int& current_triangle_count, int& next_triangle_count, int& temporary_hull_triangles_count, int& next_temporary_hull_triangles_count, const ivec3 blockid, const int triangle_index, const int p3_id, const bool is_alpha, const int finalize_particles_start, const int finalize_particles_end){
	const std::array<int, 3> current_triangle = triangles[triangle_index];
	
	std::array<int, 4> contact_indices = {};
	int face_contacts = 0;
	
	//printf("ABC %d %d\r\n", current_triangle_count, next_triangle_count);
	
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
	
	int swap_mapping_count = 0;
	std::array<std::pair<int, int>, 12> swap_mapping = {};
	
	std::array<int, 3> triangle_counts_per_vertex = {};
	for(int contact_triangle_index = face_contacts - 1; contact_triangle_index >= 0; contact_triangle_index--) {//NOTE: Loop goes backwards to handle big indices first which allows easier swapping
		//If the new tetrahedron is in alpha and the current face is not, then the current face is a boundary face that has to be kept; Same the other way round
		if(triangles_is_alpha[contact_indices[contact_triangle_index]] != is_alpha){
			//FIXME:alpha_shapes_finalize_triangle(particle_buffer, prev_partition, alpha_shapes_particle_buffer, own_particle_indices, triangles[contact_indices[contact_triangle_index]], blockid, finalize_particles_start, finalize_particles_end);
		}
		
		//Check which vertices are contacting
		for(int vertex_index = 0; vertex_index < 3; vertex_index++) {
			const int particle_index = triangles[contact_indices[contact_triangle_index]][vertex_index];
			if(particle_index == current_triangle[0]){
				triangle_counts_per_vertex[0]++;
			}else if(particle_index == current_triangle[1]){
				triangle_counts_per_vertex[1]++;
			}else if(particle_index == current_triangle[2]){
				triangle_counts_per_vertex[2]++;
			}//particle_index == p3_id
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

		//Update mappings
		int first_swap_mapping = -1;
		int second_swap_mapping = -1;
		for(int j = 0; j < swap_mapping_count; ++j){
			if(swap_mapping[j].first == contact_indices[contact_triangle_index]){
				first_swap_mapping = j;
			}
			if(swap_mapping[j].first == swap_index){
				second_swap_mapping = j;
			}
		}
		if(first_swap_mapping >= 0 && second_swap_mapping >= 0){
			thrust::swap(swap_mapping[first_swap_mapping].second, swap_mapping[second_swap_mapping].second);
		}else if(first_swap_mapping >= 0){
			swap_mapping[swap_mapping_count].first = swap_index;
			swap_mapping[swap_mapping_count].second = swap_mapping[first_swap_mapping].second;
			swap_mapping_count++;
			swap_mapping[first_swap_mapping].second = swap_index;
		}else if(second_swap_mapping >= 0){
			swap_mapping[swap_mapping_count].first = contact_indices[contact_triangle_index];
			swap_mapping[swap_mapping_count].second = swap_mapping[second_swap_mapping].second;
			swap_mapping_count++;
			swap_mapping[second_swap_mapping].second = contact_indices[contact_triangle_index];
		}else{
			swap_mapping[swap_mapping_count].first = contact_indices[contact_triangle_index];
			swap_mapping[swap_mapping_count].second = swap_index;
			swap_mapping_count++;
			if(contact_indices[contact_triangle_index] != swap_index){
				swap_mapping[swap_mapping_count].first = swap_index;
				swap_mapping[swap_mapping_count].second = contact_indices[contact_triangle_index];
				swap_mapping_count++;
			}
		}
		
	}
	
	//Fill gap between current list and next list
	for(int i = 0; i < next_triangle_start - current_triangle_count; ++i) {
		thrust::swap(triangles[next_triangle_start - 1 - i], triangles[next_triangle_start + next_triangle_count - 1 - i]);
		thrust::swap(triangles_is_alpha[next_triangle_start - 1 - i], triangles_is_alpha[next_triangle_start + next_triangle_count - 1 - i]);
		
		int first_swap_mapping = -1;
		int second_swap_mapping = -1;
		for(int j = 0; j < swap_mapping_count; ++j){
			if(swap_mapping[j].first == next_triangle_start - 1 - i){
				first_swap_mapping = j;
			}
			if(swap_mapping[j].first == next_triangle_start + next_triangle_count - 1 - i){
				second_swap_mapping = j;
			}
		}
		if(first_swap_mapping >= 0 && second_swap_mapping >= 0){
			thrust::swap(swap_mapping[first_swap_mapping].second, swap_mapping[second_swap_mapping].second);
		}else if(first_swap_mapping >= 0){
			swap_mapping[swap_mapping_count].first = next_triangle_start + next_triangle_count - 1 - i;
			swap_mapping[swap_mapping_count].second = swap_mapping[first_swap_mapping].second;
			swap_mapping_count++;
			swap_mapping[first_swap_mapping].second = next_triangle_start + next_triangle_count - 1 - i;
		}
		//Other cases cannot appear, cause we are only swapping indices previously already swapped
	}
	
	//Remove all contacting triangles from temporary hull
	int next_temporary_hull_triangles_start = temporary_hull_triangles_count;
	for(int i = temporary_hull_triangles_count + next_temporary_hull_triangles_count - 1; i >= 0; i--) {//NOTE: Loop goes backwards to handle big indices first which allows easier swapping
		//Override swapped indices
		for(int j = 0; j < swap_mapping_count; ++j){
			//Search entry thatcontains our current value and retrieve its new index
			if(temporary_hull_triangles[i] == swap_mapping[j].second){
				temporary_hull_triangles[i] = swap_mapping[j].first;
				break;//Don't apply further swaps, otherwise we override previous swaps
			}
		}
		//If index is out of bounds it was swapped out and can be removed
		if(temporary_hull_triangles[i] >= current_triangle_count + next_triangle_count){
			int swap_index;
			if(i < temporary_hull_triangles_count){
				swap_index = temporary_hull_triangles_count - 1;//Swap with last active triangle
				
				//Decrease triangle count
				temporary_hull_triangles_count--;
			}else{
				swap_index = next_temporary_hull_triangles_start + next_temporary_hull_triangles_count - 1;//Swap with first next triangle
				
				//Decrease next triangle count
				next_temporary_hull_triangles_count--;
			}
			
			//Swap contacting triangle to the end
			thrust::swap(temporary_hull_triangles[i], temporary_hull_triangles[swap_index]);
		}
	}
	
	//Fill gap between current list and next list
	for(int i = 0; i < next_temporary_hull_triangles_start - temporary_hull_triangles_count; ++i) {
		thrust::swap(temporary_hull_triangles[next_temporary_hull_triangles_start - 1 - i], temporary_hull_triangles[next_temporary_hull_triangles_start + next_temporary_hull_triangles_count - 1 - i]);
	}
	
	int next_triangle_end = current_triangle_count + next_triangle_count;
	//printf("ABC %d %d %d\r\n", current_triangle_count, next_triangle_count, next_triangle_end);
	next_triangle_count += (4 - face_contacts);
	next_temporary_hull_triangles_count += (4 - face_contacts);
	
	if(current_triangle_count + next_triangle_count > static_cast<int>(ALPHA_SHAPES_MAX_TRIANGLE_COUNT)){
		printf("Too much triangles: May not be more than %d, but is %d\n", static_cast<int>(ALPHA_SHAPES_MAX_TRIANGLE_COUNT), current_triangle_count + next_triangle_count);
	}
	
	//All new added triangles are in next temporary convex hull
	for(int i = 0; i < (4 - face_contacts); ++i) {
		temporary_hull_triangles[temporary_hull_triangles_count + next_temporary_hull_triangles_count - 1 - i] = next_triangle_end + i;
	}
	
	printf("ABC %d %d %d # %d\r\n", triangle_counts_per_vertex[0], triangle_counts_per_vertex[1], triangle_counts_per_vertex[2], face_contacts);
	
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
	
	//printf("DEF %d %d %d %d # %d %d %d\r\n", face_contacts, next_triangle_end, next_triangle_count, triangle_index, current_triangle[0], current_triangle[1], current_triangle[2]);
	if(triangles[0][0] == 0 && triangles[0][1] == 0 && triangles[0][2] == 0){
		//printf("TEST3\r\n");
	}
	
	for(int i = 0; i < (4 - face_contacts); ++i){
		//printf("O %d %d %d\r\n", triangles[next_triangle_end + i][0], triangles[next_triangle_end + i][1], triangles[next_triangle_end + i][2]);
	}
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

template<typename Partition, MaterialE MaterialType>
__forceinline__ __device__ void tmp_write_particle_data(const ParticleBuffer<MaterialType> particle_buffer, const Partition prev_partition, AlphaShapesParticleBuffer alpha_shapes_particle_buffer, const int id, const ivec3 blockid, const int point_type, const std::array<float, 3>& normal, float mean_curvature, float gauss_curvature){
	int advection_source_blockno;
	int source_pidib;
	alpha_shapes_fetch_id(particle_buffer, prev_partition, id, blockid, advection_source_blockno, source_pidib);
	
	auto particle_bin													= alpha_shapes_particle_buffer.ch(_0, particle_buffer.bin_offsets[advection_source_blockno] + source_pidib / config::G_BIN_CAPACITY);
	const int particle_id_in_bin = source_pidib  % config::G_BIN_CAPACITY;
	
	particle_bin.val(_0, particle_id_in_bin) = point_type;
	particle_bin.val(_1, particle_id_in_bin) = normal[0];
	particle_bin.val(_2, particle_id_in_bin) = normal[1];
	particle_bin.val(_3, particle_id_in_bin) = normal[2];
	particle_bin.val(_4, particle_id_in_bin) = mean_curvature;
	particle_bin.val(_5, particle_id_in_bin) = gauss_curvature;
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

//Creates triangulation for contact_triangles and additional_contact_particles lying on a (approximate) circumsphere
//Based on Mesh Generation for Technology CAD in Three Dimensions / https://www.iue.tuwien.ac.at/phd/fleischmann/node64.html#sec:cosphericaldesign
//FIXME: Maybe ensure global ordering so that different cells return the same triangulation for a subset of particles; Maybe not possible only in this functiosn (depends on initial triangles)
template<typename Partition, MaterialE MaterialType>
__forceinline__ __device__ void alpha_shapes_build_tetrahedra(const ParticleBuffer<MaterialType> particle_buffer, const Partition prev_partition, AlphaShapesParticleBuffer alpha_shapes_particle_buffer, const int* particle_indices, const int particle_indices_count, const int* own_particle_indices, std::array<int, 3>* triangles, bool* triangles_is_alpha, int& current_triangle_count, int& next_triangle_count, const ivec3 blockid, const float alpha, const int finalize_particles_start, const int finalize_particles_end, const int* contact_triangles, const int* additional_contact_particles, const int contact_triangles_count, const int additional_contact_particles_count, float& minimum_x, float& maximum_x, int& tmpabc, const int test_index){
	constexpr size_t ALPHA_SHAPES_MAX_TEMPORARY_CONVEX_HULL_TRIANGLES = 2 * (ALPHA_SHAPES_MAX_CIRCUMSPHERE_POINTS) - 4;
	
	int temporary_convex_hull_triangles_count = 0;
	int next_temporary_convex_hull_triangles_count = 0;
	std::array<int, ALPHA_SHAPES_MAX_TEMPORARY_CONVEX_HULL_TRIANGLES> temporary_convex_hull = {};
	
	//Add contact triangles to convex hull
	for(int i = 0; i < contact_triangles_count; ++i){
		temporary_convex_hull[i] = contact_triangles[i];
	}
	temporary_convex_hull_triangles_count = contact_triangles_count;
	
	//Process additional particles
	for(int i = 0; i < additional_contact_particles_count; ++i){
		i = additional_contact_particles_count - 1;//FIXME:
		
		const std::array<float, 3> particle_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, additional_contact_particles[i], blockid);
		const vec3 particle_position {particle_position_arr[0], particle_position_arr[1], particle_position_arr[2]};
		
		//if(tmpabc == test_index){
			//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, additional_contact_particles[i], blockid, 2, {0.0f, 0.0f, 0.0f}, 0.0f, 0.0f);
		//}
		
		//printf("M %d # %d\n", tmpabc, additional_contact_particles[i]);
		
		//For each triangle of the convex hull facing towards the point (point is in triangle halfspace) try to add a tetrahedron formed by this triangle and this point
		for(int j = 0; j < temporary_convex_hull_triangles_count; ++j){
			for(int k = 0; k < current_triangle_count; ++k){
				//printf("X %d # %d %d %d\n", tmpabc, triangles[k][0], triangles[k][1], triangles[k][2]);
			}
			for(int k = current_triangle_count; k < current_triangle_count + next_triangle_count; ++k){
				//printf("Y %d # %d %d %d\n", tmpabc, triangles[k][0], triangles[k][1], triangles[k][2]);
			}
			for(int k = 0; k < temporary_convex_hull_triangles_count; ++k){
				//printf("X1 %d # %d \n", tmpabc, temporary_convex_hull[k]);
			}
			for(int k = temporary_convex_hull_triangles_count; k < temporary_convex_hull_triangles_count + next_temporary_convex_hull_triangles_count; ++k){
				//printf("Y1 %d # %d \n", tmpabc, temporary_convex_hull[k]);
			}
			
			const std::array<float, 3> p0_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[temporary_convex_hull[j]][0], blockid);
			const std::array<float, 3> p1_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[temporary_convex_hull[j]][1], blockid);
			const std::array<float, 3> p2_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[temporary_convex_hull[j]][2], blockid);
			
			const vec3 p0_position {p0_position_arr[0], p0_position_arr[1], p0_position_arr[2]};
			const vec3 p1_position {p1_position_arr[0], p1_position_arr[1], p1_position_arr[2]};
			const vec3 p2_position {p2_position_arr[0], p2_position_arr[1], p2_position_arr[2]};
			
			const std::array<std::array<float, 3>, 3> triangle_positions {
				  p0_position.data_arr()
				, p1_position.data_arr()
				, p2_position.data_arr()
			};
			
			const std::array<float, 3>& triangle_normal0 = alpha_shapes_calculate_triangle_normal(triangle_positions);
			const vec3 triangle_normal_vec0 {triangle_normal0[0], triangle_normal0[1], triangle_normal0[2]};
			
			if(tmpabc == test_index){
				//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, triangles[temporary_convex_hull[j]][0], blockid, 1, triangle_normal0, 0.0f, 0.0f);
				//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, triangles[temporary_convex_hull[j]][1], blockid, 1, triangle_normal0, 0.0f, 0.0f);
				//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, triangles[temporary_convex_hull[j]][2], blockid, 1, triangle_normal0, 0.0f, 0.0f);
				//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, additional_contact_particles[i], blockid, 2, {0.0f, 0.0f, 0.0f}, 0.0f, 0.0f);
			}
			
			//Halfspace test
			const bool in_halfspace = triangle_normal_vec0.dot(particle_position - p0_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
			//printf("Z0 %d - ", (in_halfspace ? 1 : 0));
			if(in_halfspace){
				//printf("N %d # %d # %d %d %d\n", tmpabc, j, triangles[temporary_convex_hull[j]][0], triangles[temporary_convex_hull[j]][1], triangles[temporary_convex_hull[j]][2]);
				
				vec3 sphere_center;
				float radius;
				alpha_shapes_get_circumsphere(triangle_positions[0], triangle_positions[1], triangle_positions[2], particle_position_arr, sphere_center.data_arr(), radius);
				
				const float lambda = triangle_normal_vec0.dot(sphere_center - p0_position);
				
				//Convex test
				bool convex = true;
				/*for(int iter = 0; iter < particle_indices_count; ++iter){
					const std::array<float, 3> particle_position_arr_a = alpha_shapes_get_particle_position(particle_buffer, prev_partition, particle_indices[iter], blockid);
					const vec3 particle_position_a {particle_position_arr_a[0], particle_position_arr_a[1], particle_position_arr_a[2]};
					
					//Don't test tetrahedron corner points
					if(
						   (particle_position_arr_a[0] != triangle_positions[0][0] || particle_position_arr_a[1] != triangle_positions[0][1] || particle_position_arr_a[2] != triangle_positions[0][2])
						&& (particle_position_arr_a[0] != triangle_positions[1][0] || particle_position_arr_a[1] != triangle_positions[1][1] || particle_position_arr_a[2] != triangle_positions[1][2])
						&& (particle_position_arr_a[0] != triangle_positions[2][0] || particle_position_arr_a[1] != triangle_positions[2][1] || particle_position_arr_a[2] != triangle_positions[2][2])
						&& (particle_position_arr_a[0] != particle_position[0] || particle_position_arr_a[1] != particle_position[1] || particle_position_arr_a[2] != particle_position[2])
					){
					
						const bool in_halfspace_a0 = triangle_normal_vec0.dot(particle_position_a - p0_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
						
						//Only test points in halfspace
						if(in_halfspace_a0){
							//NOTE: Testing all 4 spheres of resulting tetrahedron. If one is more near to one side and on the inner side of the tetrahedron we cannot creater a new tetrahedron with the current point cause at least one side needs to create a tetrahedron with another vertex
							vec3 sphere_center_a0;
							vec3 sphere_center_a1;
							vec3 sphere_center_a2;
							vec3 sphere_center_a3;
							float radius_a0;
							float radius_a1;
							float radius_a2;
							float radius_a3;
							alpha_shapes_get_circumsphere(triangle_positions[0], triangle_positions[1], triangle_positions[2], particle_position_arr_a, sphere_center_a0.data_arr(), radius_a0);
							alpha_shapes_get_circumsphere(triangle_positions[1], triangle_positions[0], particle_position_arr, particle_position_arr_a, sphere_center_a1.data_arr(), radius_a1);
							alpha_shapes_get_circumsphere(triangle_positions[2], triangle_positions[1], particle_position_arr, particle_position_arr_a, sphere_center_a2.data_arr(), radius_a2);
							alpha_shapes_get_circumsphere(triangle_positions[0], triangle_positions[2], particle_position_arr, particle_position_arr_a, sphere_center_a3.data_arr(), radius_a3);
							
							const std::array<float, 3>& triangle_normal1 = alpha_shapes_calculate_triangle_normal({
								triangle_positions[1],
								triangle_positions[0],
								particle_position_arr
							});
							const std::array<float, 3>& triangle_normal2 = alpha_shapes_calculate_triangle_normal({
								triangle_positions[2],
								triangle_positions[1],
								particle_position_arr
							});
							const std::array<float, 3>& triangle_normal3 = alpha_shapes_calculate_triangle_normal({
								triangle_positions[0],
								triangle_positions[2],
								particle_position_arr
							});
							const vec3 triangle_normal_vec1 {triangle_normal1[0], triangle_normal1[1], triangle_normal1[2]};
							const vec3 triangle_normal_vec2 {triangle_normal2[0], triangle_normal2[1], triangle_normal2[2]};
							const vec3 triangle_normal_vec3 {triangle_normal3[0], triangle_normal3[1], triangle_normal3[2]};
							
							const bool in_halfspace_a1 = triangle_normal_vec1.dot(particle_position_a - particle_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
							const bool in_halfspace_a2 = triangle_normal_vec2.dot(particle_position_a - particle_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
							const bool in_halfspace_a3 = triangle_normal_vec3.dot(particle_position_a - particle_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
							
							
							//const float lambda_a = triangle_normal_vec.dot(sphere_center_a - p0_position);
							
							//Smaller delaunay sphere is the on that does not contain the other point
							//const vec3 diff = sphere_center_a - particle_position;
							//const float squared_distance = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
							//const vec3 diff_a = sphere_center - particle_position_a;
							//const float squared_distance_a = diff_a[0] * diff_a[0] + diff_a[1] * diff_a[1] + diff_a[2] * diff_a[2];
							
							//a lies in sphere_b, but b does not lie in sphere_a
							//TODO: Maybe use threshold here?
							//bool in_sphere = (squared_distance > (radius_a * radius_a) && squared_distance_a <= (radius * radius));
							
							//If both points lie in each others sphere or none lies in each others sphere use lambda to determine smaller
							//if(!in_sphere){
							//	in_sphere = (lambda_a - lambda) <= ALPHA_SHAPES_MIN_SPHERE_THRESHOLD;
							//}
							
							bool in_sphere = ((radius_a0 - radius) <= ALPHA_SHAPES_MIN_SPHERE_RADIUS_THRESHOLD);
							in_sphere = in_sphere || (in_halfspace_a1 && ((radius_a1 - radius) <= ALPHA_SHAPES_MIN_SPHERE_RADIUS_THRESHOLD));
							in_sphere = in_sphere || (in_halfspace_a2 && ((radius_a2 - radius) <= ALPHA_SHAPES_MIN_SPHERE_RADIUS_THRESHOLD));
							in_sphere = in_sphere || (in_halfspace_a3 && ((radius_a3 - radius) <= ALPHA_SHAPES_MIN_SPHERE_RADIUS_THRESHOLD));
							
							//If a is in circumsphere the tetrahedron is invalid
							if(in_sphere){
								convex = false;
								printf("INTERSECT %d # %d %d %d # %d\n", tmpabc, triangles[temporary_convex_hull[j]][0], triangles[temporary_convex_hull[j]][1], triangles[temporary_convex_hull[j]][2], particle_indices[iter]);
								if(tmpabc == test_index){
									//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, triangles[temporary_convex_hull[j]][0], blockid, 1, triangle_normal0, 0.0f, 0.0f);
									//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, triangles[temporary_convex_hull[j]][1], blockid, 1, triangle_normal0, 0.0f, 0.0f);
									//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, triangles[temporary_convex_hull[j]][2], blockid, 1, triangle_normal0, 0.0f, 0.0f);
									//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, additional_contact_particles[i], blockid, 2, {0.0f, 0.0f, 0.0f}, 0.0f, 0.0f);
									//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, particle_indices[iter], blockid, 3, {0.0f, 0.0f, 0.0f}, 0.0f, 0.0f);
								}
								//FIXME:break;
							}
						}
					}
				}*/
				
				//printf("Convex %d # %d %d %d # %d\n", tmpabc, triangles[temporary_convex_hull[j]][0], triangles[temporary_convex_hull[j]][1], triangles[temporary_convex_hull[j]][2], (convex ? 1 : 0));
				
				//If the tetrahedron is valid, evaluate contact conditions; Otherwise mark the conecting triangle as non-convex and don't generate the tetrahedron
				//A non-convex triangle is finalized if it is alpha
				if(convex){
					//Check alpha shape condition and mark faces accordingly
					const bool is_alpha = (radius * radius <= alpha);
					
					//Check contact conditions and update triangle list
					//NOTE: This changes the triangle counts
					alpha_shapes_check_contact_condition(particle_buffer, prev_partition, alpha_shapes_particle_buffer, own_particle_indices, triangles, triangles_is_alpha, temporary_convex_hull.data(), current_triangle_count, next_triangle_count, temporary_convex_hull_triangles_count, next_temporary_convex_hull_triangles_count, blockid, temporary_convex_hull[j], additional_contact_particles[i], is_alpha,finalize_particles_start, finalize_particles_end);
				
					for(int k = 0; k < current_triangle_count; ++k){
						//printf("X %d # %d %d %d\n", tmpabc, triangles[k][0], triangles[k][1], triangles[k][2]);
					}
					for(int k = current_triangle_count; k < current_triangle_count + next_triangle_count; ++k){
						//printf("Y %d # %d %d %d\n", tmpabc, triangles[k][0], triangles[k][1], triangles[k][2]);
					}
				
					//Update bounds
					//FIXME: Correct distance (e.g. intersection of triangle halfspace with xy and xz boundaries
					minimum_x = std::numeric_limits<float>::min();//std::min(minimum_x, sphere_center[0] - radius);
					maximum_x = std::numeric_limits<float>::max();//std::max(maximum_x, sphere_center[0] + radius);
				}else{
					//Remove the face and move it to alpha if it is alpha
					
					if(triangles_is_alpha[temporary_convex_hull[0]]){
						//FIXME:alpha_shapes_finalize_triangle(particle_buffer, prev_partition, alpha_shapes_particle_buffer, own_particle_indices.data(), temporary_convex_hull[j], blockid, own_last_active_particles_start, own_active_particles_start + own_active_particles_count);
					}
		
					//Swap contact triangles to end of list to remove them
					int swap_index;
					bool from_current_triangles;
					if(temporary_convex_hull[j] < current_triangle_count){
						from_current_triangles = true;
						swap_index = current_triangle_count - 1;//Swap with last active triangle
						
						//Decrease triangle count
						current_triangle_count--;
					}else{
						from_current_triangles = false;
						swap_index = current_triangle_count + next_triangle_count - 1;//Swap with first next triangle
						
						//Decrease next triangle count
						next_triangle_count--;
					}
					
					//Swap contacting triangle to the end
					thrust::swap(triangles[temporary_convex_hull[j]], triangles[swap_index]);
					thrust::swap(triangles_is_alpha[temporary_convex_hull[j]], triangles_is_alpha[swap_index]);
					thrust::swap(temporary_convex_hull[j], temporary_convex_hull[temporary_convex_hull_triangles_count - 1]);
					
					//Fill gap between current list and next list
					if(from_current_triangles){
						thrust::swap(triangles[current_triangle_count], triangles[current_triangle_count + next_triangle_count - 1]);
						thrust::swap(triangles_is_alpha[current_triangle_count], triangles_is_alpha[current_triangle_count + next_triangle_count - 1]);
					}
					thrust::swap(temporary_convex_hull[temporary_convex_hull_triangles_count - 1], temporary_convex_hull[temporary_convex_hull_triangles_count + next_temporary_convex_hull_triangles_count - 1]);
					
					//Decrease triangle count
					temporary_convex_hull_triangles_count--;
					
					
					//tmpabc++;
				}
				
				//NOTE: No index before j can be swapped cause they all were out of halfspace for this point and the point is not part of them (we only add additional points not already handled); Actually only j can be swapped, but the contact handle algorithm is designed to support swaps at any position
				//Reduce j to revisit current, new swapped index
				//Only swap if j is not last, cause if so we swapped j with itself
				if(j != temporary_convex_hull_triangles_count){
					j--;
				}
			}
			//tmpabc++;
		}
		
		
		//Swap lists
		temporary_convex_hull_triangles_count += next_temporary_convex_hull_triangles_count;
		next_temporary_convex_hull_triangles_count = 0;
	}
}

template<typename Partition, MaterialE MaterialType>
__forceinline__ __device__ bool alpha_shapes_handle_triangle_compare_func(const ParticleBuffer<MaterialType> particle_buffer, const Partition prev_partition, const ivec3 blockid, const std::array<std::array<float, 3>, 3> triangle_positions, const std::array<float, 3> triangle_normal_arr, const int& a, const int& b){
	const vec3 p0_position {triangle_positions[0][0], triangle_positions[0][1], triangle_positions[0][2]};
	const vec3 triangle_normal_vec {triangle_normal_arr[0], triangle_normal_arr[1], triangle_normal_arr[2]};
		
	const std::array<float, 3> particle_position_arr_a = alpha_shapes_get_particle_position(particle_buffer, prev_partition, a, blockid);
	const std::array<float, 3> particle_position_arr_b = alpha_shapes_get_particle_position(particle_buffer, prev_partition, b, blockid);
	
	const vec3 particle_position_a {particle_position_arr_a[0], particle_position_arr_a[1], particle_position_arr_a[2]};
	const vec3 particle_position_b {particle_position_arr_b[0], particle_position_arr_b[1], particle_position_arr_b[2]};
	
	//Test if in half_space; Also sorts out particles that lie in a plane with the triangle
	const bool in_halfspace_a = triangle_normal_vec.dot(particle_position_a - p0_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
	const bool in_halfspace_b = triangle_normal_vec.dot(particle_position_b - p0_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
	
	bool ret = false;
	//Triangle positions are always bigger
	if(
		   (particle_position_arr_a[0] == triangle_positions[0][0] && particle_position_arr_a[1] == triangle_positions[0][1] && particle_position_arr_a[2] == triangle_positions[0][2])
		|| (particle_position_arr_a[0] == triangle_positions[1][0] && particle_position_arr_a[1] == triangle_positions[1][1] && particle_position_arr_a[2] == triangle_positions[1][2])
		|| (particle_position_arr_a[0] == triangle_positions[2][0] && particle_position_arr_a[1] == triangle_positions[2][1] && particle_position_arr_a[2] == triangle_positions[2][2])
	){
		ret = false;
	}else if(
		   (particle_position_arr_b[0] == triangle_positions[0][0] && particle_position_arr_b[1] == triangle_positions[0][1] && particle_position_arr_b[2] == triangle_positions[0][2])
		|| (particle_position_arr_b[0] == triangle_positions[1][0] && particle_position_arr_b[1] == triangle_positions[1][1] && particle_position_arr_b[2] == triangle_positions[1][2])
		|| (particle_position_arr_b[0] == triangle_positions[2][0] && particle_position_arr_b[1] == triangle_positions[2][1] && particle_position_arr_b[2] == triangle_positions[2][2])
	){
		ret = in_halfspace_a;
	} else if(in_halfspace_a && in_halfspace_b){
		//Calculate delaunay spheres
		vec3 sphere_center_a;
		vec3 sphere_center_b;
		float radius_a;
		float radius_b;
		alpha_shapes_get_circumsphere(triangle_positions[0], triangle_positions[1], triangle_positions[2], particle_position_arr_a, sphere_center_a.data_arr(), radius_a);
		alpha_shapes_get_circumsphere(triangle_positions[0], triangle_positions[1], triangle_positions[2], particle_position_arr_b, sphere_center_b.data_arr(), radius_b);
		
		/*
		const float lambda_a = triangle_normal_vec.dot(sphere_center_a - p0_position);
		const float lambda_b = triangle_normal_vec.dot(sphere_center_b - p0_position);
		
		//Smaller delaunay sphere is the on that does not contain the other point
		const vec3 diff_a = sphere_center_b - particle_position_a;
		const float squared_distance_a = diff_a[0] * diff_a[0] + diff_a[1] * diff_a[1] + diff_a[2] * diff_a[2];
		const vec3 diff_b = sphere_center_a - particle_position_b;
		const float squared_distance_b = diff_b[0] * diff_b[0] + diff_b[1] * diff_b[1] + diff_b[2] * diff_b[2];
		
		//a lies in sphere_b, but b does not lie in sphere_a
		ret = (squared_distance_b > (radius_a * radius_a) && squared_distance_a <= (radius_b * radius_b));
		
		//If both points lie in each others sphere or none lies in each others sphere use lambda to determine smaller
		if(!ret){
			ret = (lambda_a < lambda_b);
		}
		*/
		ret = (radius_a < radius_b);
	}else if(in_halfspace_a){
		ret = true;
	}
	return ret;
};

//Based on Mesh Generation for Technology CAD in Three Dimensions / https://www.iue.tuwien.ac.at/phd/fleischmann/node64.html#sec:cosphericaldesign
template<typename Partition, MaterialE MaterialType>
__forceinline__ __device__ void alpha_shapes_handle_triangle(const ParticleBuffer<MaterialType> particle_buffer, const Partition prev_partition, AlphaShapesParticleBuffer alpha_shapes_particle_buffer, const int* particle_indices, const int particle_indices_count, const int* own_particle_indices, std::array<int, 3>* triangles, bool* triangles_is_alpha, int& current_triangle_count, int& next_triangle_count, const ivec3 blockid, const float alpha, const int finalize_particles_start, const int finalize_particles_end, bool is_first_triangle, const int triangle_index, float& minimum_x, float& maximum_x, int& tmpabc, const int test_index){
	const std::array<int, 3> current_triangle = triangles[triangle_index];
	
	const std::array<float, 3> p0_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, current_triangle[0], blockid);
	const std::array<float, 3> p1_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, current_triangle[1], blockid);
	const std::array<float, 3> p2_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, current_triangle[2], blockid);
	
	const vec3 p0_position {p0_position_arr[0], p0_position_arr[1], p0_position_arr[2]};
	const vec3 p1_position {p1_position_arr[0], p1_position_arr[1], p1_position_arr[2]};
	const vec3 p2_position {p2_position_arr[0], p2_position_arr[1], p2_position_arr[2]};
	
	const std::array<std::array<float, 3>, 3> triangle_positions {
		  p0_position_arr
		, p1_position_arr
		, p2_position_arr
	};
	
	std::array<float, 3> triangle_normal = alpha_shapes_calculate_triangle_normal(triangle_positions);
	vec3 triangle_normal_vec {triangle_normal[0], triangle_normal[1], triangle_normal[2]};
	
	//If we handle the first triangle we first have to determine normal direction
	bool flipped_normal = false;
	if(is_first_triangle){
		const int* p3_iter = thrust::min_element(thrust::seq, particle_indices, particle_indices + particle_indices_count, [&particle_buffer, &prev_partition, &blockid, &triangle_positions, &triangle_normal_vec, &p0_position](const int& a, const int& b){
			const std::array<float, 3> particle_position_arr_a = alpha_shapes_get_particle_position(particle_buffer, prev_partition, a, blockid);
			const std::array<float, 3> particle_position_arr_b = alpha_shapes_get_particle_position(particle_buffer, prev_partition, b, blockid);
			
			const vec3 particle_position_a {particle_position_arr_a[0], particle_position_arr_a[1], particle_position_arr_a[2]};
			const vec3 particle_position_b {particle_position_arr_b[0], particle_position_arr_b[1], particle_position_arr_b[2]};
			
			bool ret = false;
			//Triangle positions are always bigger
			if(
				   (particle_position_arr_a[0] == triangle_positions[0][0] && particle_position_arr_a[1] == triangle_positions[0][1] && particle_position_arr_a[2] == triangle_positions[0][2])
				|| (particle_position_arr_a[0] == triangle_positions[1][0] && particle_position_arr_a[1] == triangle_positions[1][1] && particle_position_arr_a[2] == triangle_positions[1][2])
				|| (particle_position_arr_a[0] == triangle_positions[2][0] && particle_position_arr_a[1] == triangle_positions[2][1] && particle_position_arr_a[2] == triangle_positions[2][2])
			){
				ret = false;
			}else if(
				   (particle_position_arr_b[0] == triangle_positions[0][0] && particle_position_arr_b[1] == triangle_positions[0][1] && particle_position_arr_b[2] == triangle_positions[0][2])
				|| (particle_position_arr_b[0] == triangle_positions[1][0] && particle_position_arr_b[1] == triangle_positions[1][1] && particle_position_arr_b[2] == triangle_positions[1][2])
				|| (particle_position_arr_b[0] == triangle_positions[2][0] && particle_position_arr_b[1] == triangle_positions[2][1] && particle_position_arr_b[2] == triangle_positions[2][2])
			){
				ret = true;
			} else {
				//Calculate delaunay spheres
				vec3 sphere_center_a;
				vec3 sphere_center_b;
				float radius_a;
				float radius_b;
				alpha_shapes_get_circumsphere(triangle_positions[0], triangle_positions[1], triangle_positions[2], particle_position_arr_a, sphere_center_a.data_arr(), radius_a);
				alpha_shapes_get_circumsphere(triangle_positions[0], triangle_positions[1], triangle_positions[2], particle_position_arr_b, sphere_center_b.data_arr(), radius_b);
				
				/*
				const float lambda_a = triangle_normal_vec.dot(sphere_center_a - p0_position);
				const float lambda_b = triangle_normal_vec.dot(sphere_center_b - p0_position);
				
				//Smaller delaunay sphere is the on that does not contain the other point
				const vec3 diff_a = sphere_center_b - particle_position_a;
				const float squared_distance_a = diff_a[0] * diff_a[0] + diff_a[1] * diff_a[1] + diff_a[2] * diff_a[2];
				const vec3 diff_b = sphere_center_a - particle_position_b;
				const float squared_distance_b = diff_b[0] * diff_b[0] + diff_b[1] * diff_b[1] + diff_b[2] * diff_b[2];
				
				//a lies in sphere_b, but b does not lie in sphere_a
				ret = (squared_distance_b > (radius_a * radius_a) && squared_distance_a <= (radius_b * radius_b));
				
				//If both points lie in each others sphere or none lies in each others sphere use lambda to determine smaller
				if(!ret){
					ret = (std::abs(lambda_a) < std::abs(lambda_b));
				}
				*/
				ret = (radius_a < radius_b);
			}
			return ret;
		});
		
		const std::array<float, 3> particle_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, *p3_iter, blockid);
		const vec3 particle_position {particle_position_arr[0], particle_position_arr[1], particle_position_arr[2]};
		
		//If nearest point is not in halfspace of the normal, flip the normal
		const bool in_halfspace = triangle_normal_vec.dot(particle_position - p0_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
		if(!in_halfspace){
			flipped_normal = true;
			triangle_normal[0] = -triangle_normal[0];
			triangle_normal[1] = -triangle_normal[1];
			triangle_normal[2] = -triangle_normal[2];
			triangle_normal_vec = -triangle_normal_vec;
			
			thrust::swap(triangles[triangle_index][1], triangles[triangle_index][2]);
		}
	}
	
	int circumsphere_points_count = 0;
	int circumsphere_triangles_count = 0;
	std::array<int, ALPHA_SHAPES_MAX_CIRCUMSPHERE_POINTS> circumsphere_points = {};
	std::array<int, ALPHA_SHAPES_MAX_CIRCUMSPHERE_POINTS / 3> circumsphere_triangles = {};
	
	//Find smallest point for triangle
	const int* p3_iter = thrust::min_element(thrust::seq, particle_indices, particle_indices + particle_indices_count, [&particle_buffer, &prev_partition, &blockid, &triangle_positions, &triangle_normal](const int& a, const int& b){
		return alpha_shapes_handle_triangle_compare_func(particle_buffer, prev_partition, blockid, triangle_positions, triangle_normal, a, b);
	});
	
	circumsphere_points[circumsphere_points_count++] = *p3_iter;
	
	//Add all triangles in halfspace of point
	bool in_halfspace_of_current_triangle = false;
	{
		const std::array<float, 3> current_particle_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, *p3_iter, blockid);
		const vec3 current_particle_position {current_particle_position_arr[0], current_particle_position_arr[1], current_particle_position_arr[2]};
		
		for(int j = 0; j < current_triangle_count + next_triangle_count; ++j){
			const std::array<float, 3> current_p0_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[j][0], blockid);
			const std::array<float, 3> current_p1_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[j][1], blockid);
			const std::array<float, 3> current_p2_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[j][2], blockid);
			
			const vec3 current_p0_position {current_p0_position_arr[0], current_p0_position_arr[1], current_p0_position_arr[2]};
			
			const std::array<float, 3> current_triangle_normal = alpha_shapes_calculate_triangle_normal({
				current_p0_position_arr,
				current_p1_position_arr,
				current_p2_position_arr
			});
			
			const vec3 current_triangle_normal_vec {current_triangle_normal[0], current_triangle_normal[1], current_triangle_normal[2]};
				
			//Perform halfspace test
			const bool current_in_halfspace = current_triangle_normal_vec.dot(current_particle_position - current_p0_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
			
			//If point is in an outer halfspace of the triangle, add the triangle to active set
			if(current_in_halfspace){
				circumsphere_triangles[circumsphere_triangles_count++] = j;
				if(j == triangle_index){
					in_halfspace_of_current_triangle = true;
				}
			}
		}
	}
	
	//If the point is not in the halfspace of the currnt triangle we have no smallest point so we set the count to 0 what will cause the triangle to be finalized
	if(!in_halfspace_of_current_triangle){
		if(tmpabc == test_index){
			//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, triangles[triangle_index][0], blockid, 1, triangle_normal, 0.0f, 0.0f);
			//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, triangles[triangle_index][1], blockid, 1, triangle_normal, 0.0f, 0.0f);
			//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, triangles[triangle_index][2], blockid, 1, triangle_normal, 0.0f, 0.0f);
			//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, circumsphere_points[0], blockid, 3, {0.0f, 0.0f, 0.0f}, 0.0f, 0.0f);
		}
		circumsphere_points_count = 0;
	}
	
	//Only proceed if we have at least one point
	if(circumsphere_points_count > 0){
		//Extend convex hull, adding all points in current new convex hull and withing threshold distance to new convex hull till converged
		//Also maintaining a list of triangles activated by convex hull extension
		for(int m = 0; m < circumsphere_points_count; ++m){
			const std::array<float, 3> current_outest_point_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, circumsphere_points[m], blockid);
			const vec3 current_outest_point_position {current_outest_point_position_arr[0], current_outest_point_position_arr[1], current_outest_point_position_arr[2]};

			float smallest_outsider = std::numeric_limits<float>::max();
			for(int i = 0; i < particle_indices_count; ++i){
				const std::array<float, 3> current_particle_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, particle_indices[i], blockid);
				const vec3 current_particle_position {current_particle_position_arr[0], current_particle_position_arr[1], current_particle_position_arr[2]};
				
				//Test if point is already in set
				bool already_in_set = false;
				for(int j = 0; j < circumsphere_points_count; ++j){
					if(particle_indices[i] == circumsphere_points[j]){
						already_in_set = true;
						break;
					}
				}

				//If point is already in set skip it
				if(already_in_set){
					continue;
				}
				
				//Test if point is on correct side; This also means that it is not in current convex hull
				bool in_convex_hull = true;
				bool in_new_convex_hull = true;
				for(int j = 0; j < current_triangle_count + next_triangle_count; ++j){
					//Test if triangle is active
					bool already_in_set = false;
					for(int k = 0; k < circumsphere_points_count; ++k){
						if(j == circumsphere_triangles[k]){
							already_in_set = true;
							break;
						}
					}
					
					const std::array<float, 3> current_p0_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[j][0], blockid);
					const std::array<float, 3> current_p1_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[j][1], blockid);
					const std::array<float, 3> current_p2_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[j][2], blockid);
					
					const vec3 current_p0_position {current_p0_position_arr[0], current_p0_position_arr[1], current_p0_position_arr[2]};
					
					const std::array<float, 3> current_triangle_normal = alpha_shapes_calculate_triangle_normal({
						current_p0_position_arr,
						current_p1_position_arr,
						current_p2_position_arr
					});
					
					const vec3 current_triangle_normal_vec {current_triangle_normal[0], current_triangle_normal[1], current_triangle_normal[2]};
						
					//Perform halfspace test
					const bool current_in_halfspace = current_triangle_normal_vec.dot(current_particle_position - current_p0_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
					
					//If point is in an outer halfspace it is not in convex hull
					if(current_in_halfspace){
						in_convex_hull = false;
						
						//If point is in an outer halfspace of an triangle not active it is not in new convex hull
						if(!already_in_set){
							in_new_convex_hull = false;
							break;
						}
					}
				}
				
				//We are only searching for points on the correct side of our active triangle set
				if(!in_convex_hull){
					
					if(tmpabc == test_index){
						//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, circumsphere_points[m], blockid, 3, {0.0f, 0.0f, 0.0f}, 0.0f, 0.0f);
						//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, particle_indices[i], blockid, 4, {0.0f, 0.0f, 0.0f}, 0.0f, 0.0f);
					}
					
					//bool in_new_convex_hull = true;
					for(int j = 0; j < circumsphere_triangles_count; ++j){
						const std::array<float, 3> current_p0_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[circumsphere_triangles[j]][0], blockid);
						const std::array<float, 3> current_p1_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[circumsphere_triangles[j]][1], blockid);
						const std::array<float, 3> current_p2_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[circumsphere_triangles[j]][2], blockid);
						
						//Find outer edge (the one not being shared by another triangle in this set)
						std::array<int, 3> edge_count = {};//Count for oposite edge
						for(int k = 0; k < circumsphere_triangles_count; ++k){
							std::array<int, 3> vertex_count = {};
							if(
								(triangles[circumsphere_triangles[j]][0] == triangles[circumsphere_triangles[k]][0] || triangles[circumsphere_triangles[j]][0] == triangles[circumsphere_triangles[k]][1] || triangles[circumsphere_triangles[j]][0] == triangles[circumsphere_triangles[k]][2])
							){
								vertex_count[0]++;
							}
							if(
								(triangles[circumsphere_triangles[j]][1] == triangles[circumsphere_triangles[k]][0] || triangles[circumsphere_triangles[j]][1] == triangles[circumsphere_triangles[k]][1] || triangles[circumsphere_triangles[j]][1] == triangles[circumsphere_triangles[k]][2])
							){
								vertex_count[1]++;
							}
							if(
								(triangles[circumsphere_triangles[j]][2] == triangles[circumsphere_triangles[k]][0] || triangles[circumsphere_triangles[j]][2] == triangles[circumsphere_triangles[k]][1] || triangles[circumsphere_triangles[j]][2] == triangles[circumsphere_triangles[k]][2])
							){
								vertex_count[2]++;
							}
							
							//All edges that have a neighbour face have count 2 (one for current face, one for the other face)
							if(vertex_count[1] == 2 && vertex_count[2] == 2){
								edge_count[0]++;
							}else if(vertex_count[2] == 2 && vertex_count[0] == 2){
								edge_count[1]++;
							}else if(vertex_count[0] == 2 && vertex_count[1] == 2){
								edge_count[2]++;
							}//Otherwise no edge connection
						}
						
						//Test all outer faces
						if(edge_count[0] == 0){
							const std::array<float, 3> current_triangle_normal = alpha_shapes_calculate_triangle_normal({
								current_p1_position_arr,
								current_p2_position_arr,
								current_outest_point_position_arr
							});
							
							const vec3 current_triangle_normal_vec {current_triangle_normal[0], current_triangle_normal[1], current_triangle_normal[2]};
							
							//Perform halfspace test
							const bool current_in_halfspace = current_triangle_normal_vec.dot(current_particle_position - current_outest_point_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
							
							/*printf("ABC0 %d # %.28f %.28f # %.28f %.28f %.28f # %.28f %.28f %.28f # %.28f %.28f %.28f # %.28f %.28f %.28f\n"
								, (current_in_halfspace ? 1 : 0)
								, current_triangle_normal_vec.dot(current_particle_position - current_outest_point_position)
								, ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD
								, current_triangle_normal_vec[0]
								, current_triangle_normal_vec[1]
								, current_triangle_normal_vec[2]
								, current_particle_position[0]
								, current_particle_position[1]
								, current_particle_position[2]
								, current_outest_point_position[0]
								, current_outest_point_position[1]
								, current_outest_point_position[2]
								, (current_particle_position - current_outest_point_position)[0]
								, (current_particle_position - current_outest_point_position)[1]
								, (current_particle_position - current_outest_point_position)[2]
							);*/
							
							if(tmpabc == test_index){
								//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, triangles[circumsphere_triangles[j]][2], blockid, 1, current_triangle_normal, 0.0f, 0.0f);
							}
							
							//If point is in an outer halfspace it is not in convex hull
							if(current_in_halfspace){
								smallest_outsider = std::min(smallest_outsider, current_triangle_normal_vec.dot(current_particle_position - current_outest_point_position));
								in_new_convex_hull = false;
								//FIXME:break;
							}
						}
						if(edge_count[1] == 0){
							const std::array<float, 3> current_triangle_normal = alpha_shapes_calculate_triangle_normal({
								current_p2_position_arr,
								current_p0_position_arr,
								current_outest_point_position_arr
							});
							
							const vec3 current_triangle_normal_vec {current_triangle_normal[0], current_triangle_normal[1], current_triangle_normal[2]};
							
							//Perform halfspace test
							const bool current_in_halfspace = current_triangle_normal_vec.dot(current_particle_position - current_outest_point_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
							
							/*printf("ABC1 %d # %.28f %.28f # %.28f %.28f %.28f # %.28f %.28f %.28f # %.28f %.28f %.28f # %.28f %.28f %.28f\n"
								, (current_in_halfspace ? 1 : 0)
								, current_triangle_normal_vec.dot(current_particle_position - current_outest_point_position)
								, ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD
								, current_triangle_normal_vec[0]
								, current_triangle_normal_vec[1]
								, current_triangle_normal_vec[2]
								, current_particle_position[0]
								, current_particle_position[1]
								, current_particle_position[2]
								, current_outest_point_position[0]
								, current_outest_point_position[1]
								, current_outest_point_position[2]
								, (current_particle_position - current_outest_point_position)[0]
								, (current_particle_position - current_outest_point_position)[1]
								, (current_particle_position - current_outest_point_position)[2]
							);*/
							
							if(tmpabc == test_index){
								//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, triangles[circumsphere_triangles[j]][0], blockid, 1, current_triangle_normal, 0.0f, 0.0f);
							}
							
							//If point is in an outer halfspace it is not in convex hull
							if(current_in_halfspace){
								smallest_outsider = std::min(smallest_outsider, current_triangle_normal_vec.dot(current_particle_position - current_outest_point_position));
								in_new_convex_hull = false;
								//FIXME:break;
							}
						}
						if(edge_count[2] == 0){
							const std::array<float, 3> current_triangle_normal = alpha_shapes_calculate_triangle_normal({
								current_p0_position_arr,
								current_p1_position_arr,
								current_outest_point_position_arr
							});
							
							const vec3 current_triangle_normal_vec {current_triangle_normal[0], current_triangle_normal[1], current_triangle_normal[2]};
							
							//Perform halfspace test
							const bool current_in_halfspace = current_triangle_normal_vec.dot(current_particle_position - current_outest_point_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
							
							/*printf("ABC2 %d # %.28f %.28f # %.28f %.28f %.28f # %.28f %.28f %.28f # %.28f %.28f %.28f # %.28f %.28f %.28f\n"
								, (current_in_halfspace ? 1 : 0)
								, current_triangle_normal_vec.dot(current_particle_position - current_outest_point_position)
								, ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD
								, current_triangle_normal_vec[0]
								, current_triangle_normal_vec[1]
								, current_triangle_normal_vec[2]
								, current_particle_position[0]
								, current_particle_position[1]
								, current_particle_position[2]
								, current_outest_point_position[0]
								, current_outest_point_position[1]
								, current_outest_point_position[2]
								, (current_particle_position - current_outest_point_position)[0]
								, (current_particle_position - current_outest_point_position)[1]
								, (current_particle_position - current_outest_point_position)[2]
							);*/
							
							if(tmpabc == test_index){
								//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, triangles[circumsphere_triangles[j]][1], blockid, 1, current_triangle_normal, 0.0f, 0.0f);
							}
							
							//If point is in an outer halfspace it is not in convex hull
							if(current_in_halfspace){
								smallest_outsider = std::min(smallest_outsider, current_triangle_normal_vec.dot(current_particle_position - current_outest_point_position));
								in_new_convex_hull = false;
								//FIXME:break;
							}
						}//Otherwise triangle is no border triangle
					}
					
					//If point is in new convex hull, add it to our set
					if(in_new_convex_hull){
						printf("IN HULL\n");
						
						if(tmpabc <= test_index){
							//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, particle_indices[i], blockid, 2, {0.0f, 0.0f, 0.0f}, 0.0f, 0.0f);
						}
						
						if(tmpabc == test_index){
							for(int abc = 0; abc < circumsphere_triangles_count; ++abc){
								const std::array<float, 3> current_p0_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[circumsphere_triangles[abc]][0], blockid);
								const std::array<float, 3> current_p1_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[circumsphere_triangles[abc]][1], blockid);
								const std::array<float, 3> current_p2_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[circumsphere_triangles[abc]][2], blockid);
								
								const vec3 current_p0_position {current_p0_position_arr[0], current_p0_position_arr[1], current_p0_position_arr[2]};
								
								const std::array<float, 3> current_triangle_normal = alpha_shapes_calculate_triangle_normal({
									current_p0_position_arr,
									current_p1_position_arr,
									current_p2_position_arr
								});
								//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, triangles[circumsphere_triangles[abc]][0], blockid, 1, current_triangle_normal, 0.0f, 0.0f);
								//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, triangles[circumsphere_triangles[abc]][1], blockid, 1, current_triangle_normal, 0.0f, 0.0f);
								//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, triangles[circumsphere_triangles[abc]][2], blockid, 1, current_triangle_normal, 0.0f, 0.0f);
							}
							
							//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, circumsphere_points[m], blockid, 3, {0.0f, 0.0f, 0.0f}, 0.0f, 0.0f);
							//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, particle_indices[i], blockid, 4, {0.0f, 0.0f, 0.0f}, 0.0f, 0.0f);
						}
						//tmpabc++;
						
						if(tmpabc == test_index){
							tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, particle_indices[i], blockid, 2, {0.0f, 0.0f, 0.0f}, 0.0f, 0.0f);
						}
						
						
						//If we are out of space print a warning
						if(circumsphere_points_count >= ALPHA_SHAPES_MAX_CIRCUMSPHERE_POINTS){
							printf("More points in circumsphere than we have memory for.\r\n");
						}else{
							circumsphere_points[circumsphere_points_count++] = particle_indices[i];
							
							//Add all triangles in halfspace of point if not already active
							for(int j = 0; j < current_triangle_count + next_triangle_count; ++j){
								//Test if triangle is already in set
								bool already_in_set = false;
								for(int l = 0; l < circumsphere_triangles_count; ++l){
									if(j == circumsphere_triangles[l]){
										already_in_set = true;
										break;
									}
								}

								//If triangle is already in set skip it
								if(already_in_set){
									continue;
								}
								
								const std::array<float, 3> current_p0_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[j][0], blockid);
								const std::array<float, 3> current_p1_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[j][1], blockid);
								const std::array<float, 3> current_p2_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[j][2], blockid);
								
								const vec3 current_p0_position {current_p0_position_arr[0], current_p0_position_arr[1], current_p0_position_arr[2]};
								
								const std::array<float, 3> current_triangle_normal = alpha_shapes_calculate_triangle_normal({
									current_p0_position_arr,
									current_p1_position_arr,
									current_p2_position_arr
								});
								
								const vec3 current_triangle_normal_vec {current_triangle_normal[0], current_triangle_normal[1], current_triangle_normal[2]};
									
								//Perform halfspace test
								const bool current_in_halfspace = current_triangle_normal_vec.dot(current_particle_position - current_p0_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
								
								//If point is in an outer halfspace of the triangle, add the triangle to active set
								if(current_in_halfspace){
									if(circumsphere_triangles_count >= ALPHA_SHAPES_MAX_CIRCUMSPHERE_POINTS / 3){
										printf("More triangles in circumsphere than we have memory for.\r\n");
									}else{
										circumsphere_triangles[circumsphere_triangles_count++] = j;
										
										//Reset outest point index so that convex hull is retested with new active triangle for all points
										m = 0;
									}
								}
							}
						}
					}
				}
			}
			//tmpabc++;
			
			printf("DEF %.28f %.28f\n", smallest_outsider, ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD);
		}
		
		//Find outest point. That is the point with the most other points in its convex hull. If several, than the one with better delaunay criterion for the current triangle.
		int max_in_hull_count = -1;
		int outest_point = -1;
		for(int m = 0; m < circumsphere_points_count; ++m){
			const std::array<float, 3> current_outest_point_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, circumsphere_points[m], blockid);
			const vec3 current_outest_point_position {current_outest_point_position_arr[0], current_outest_point_position_arr[1], current_outest_point_position_arr[2]};
			
			int current_in_hull_count = 0;
			for(int j = 0; j < circumsphere_points_count; ++j){
				const std::array<float, 3> current_particle_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, circumsphere_points[j], blockid);
				const vec3 current_particle_position {current_particle_position_arr[0], current_particle_position_arr[1], current_particle_position_arr[2]};
				
				if(m != j){
					bool in_new_convex_hull = true;
					for(int i = 0; i < circumsphere_triangles_count; ++i){
						const std::array<float, 3> current_p0_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[circumsphere_triangles[i]][0], blockid);
						const std::array<float, 3> current_p1_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[circumsphere_triangles[i]][1], blockid);
						const std::array<float, 3> current_p2_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[circumsphere_triangles[i]][2], blockid);
						
						//Find outer edge (the one not being shared by another triangle in this set)
						std::array<int, 3> edge_count = {};//Count for oposite edge
						for(int k = 0; k < circumsphere_triangles_count; ++k){
							std::array<int, 3> vertex_count = {};
							if(
								(triangles[circumsphere_triangles[i]][0] == triangles[circumsphere_triangles[k]][0] || triangles[circumsphere_triangles[i]][0] == triangles[circumsphere_triangles[k]][1] || triangles[circumsphere_triangles[i]][0] == triangles[circumsphere_triangles[k]][2])
							){
								vertex_count[0]++;
							}
							if(
								(triangles[circumsphere_triangles[i]][1] == triangles[circumsphere_triangles[k]][0] || triangles[circumsphere_triangles[i]][1] == triangles[circumsphere_triangles[k]][1] || triangles[circumsphere_triangles[i]][1] == triangles[circumsphere_triangles[k]][2])
							){
								vertex_count[1]++;
							}
							if(
								(triangles[circumsphere_triangles[i]][2] == triangles[circumsphere_triangles[k]][0] || triangles[circumsphere_triangles[i]][2] == triangles[circumsphere_triangles[k]][1] || triangles[circumsphere_triangles[i]][2] == triangles[circumsphere_triangles[k]][2])
							){
								vertex_count[2]++;
							}
							
							//All edges that have a neighbour face have count 2 (one for current face, one for the other face)
							if(vertex_count[1] == 2 && vertex_count[2] == 2){
								edge_count[0]++;
							}else if(vertex_count[2] == 2 && vertex_count[0] == 2){
								edge_count[1]++;
							}else if(vertex_count[0] == 2 && vertex_count[1] == 2){
								edge_count[2]++;
							}//Otherwise no edge connection
						}
						
						//Calculate outwards pointing normal
						std::array<float, 3> current_triangle_normal = {0.0f, 0.0f, 0.0f};
				
						if(edge_count[0] == 0){
							current_triangle_normal = alpha_shapes_calculate_triangle_normal({
								current_p1_position_arr,
								current_p2_position_arr,
								current_outest_point_position_arr
							});
						}else if(edge_count[1] == 0){
							current_triangle_normal = alpha_shapes_calculate_triangle_normal({
								current_p2_position_arr,
								current_p0_position_arr,
								current_outest_point_position_arr
							});
						}else if(edge_count[2] == 0){
							current_triangle_normal = alpha_shapes_calculate_triangle_normal({
								current_p0_position_arr,
								current_p1_position_arr,
								current_outest_point_position_arr
							});
						}//Otherwise triangle is no border triangle
						
						if(current_triangle_normal[0] != 0.0f || current_triangle_normal[1] != 0.0f || current_triangle_normal[2] != 0.0f){
							const vec3 current_triangle_normal_vec {current_triangle_normal[0], current_triangle_normal[1], current_triangle_normal[2]};
							
							//Perform halfspace test
							const bool current_in_halfspace = current_triangle_normal_vec.dot(current_particle_position - current_outest_point_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
							
							//If point is in an outer halfspace it is not in convex hull
							if(current_in_halfspace){
								in_new_convex_hull = false;
								break;
							}
						}
					}
					
					if(in_new_convex_hull){
						current_in_hull_count++;
					}
				}
			}
			
			if(current_in_hull_count >= max_in_hull_count){
				//If several are equal, choose the one with better delaunay criterion
				if(current_in_hull_count > max_in_hull_count || outest_point == -1 || alpha_shapes_handle_triangle_compare_func(particle_buffer, prev_partition, blockid, triangle_positions, triangle_normal, circumsphere_points[m], circumsphere_points[outest_point])){
					max_in_hull_count = current_in_hull_count;
					outest_point = m;
				}
			}
		}
	
		//If the new convex hull does not contain all points we have a problem. Our resulting hull must be convex, what is not the case. But we need the point cause it is in the convex hull of one of the ohter of our points
		//TODO: Solution might be to adjust the threshold? Not really => think of another solution
		if(max_in_hull_count != circumsphere_points_count - 1){
			printf("New convex hull does not contain all new points\r\n");
		}
		
		//Check that paints are not part of any other triangle
		//FIXME: Remove this,c ause the test is unnecessary
		for(int current_triangle_index = 0; current_triangle_index < current_triangle_count + next_triangle_count; ++current_triangle_index){
			for(int i = 0; i < circumsphere_points_count; ++i){
				if(triangles[current_triangle_index][0] == circumsphere_points[i] || triangles[current_triangle_index][1] == circumsphere_points[i] || triangles[current_triangle_index][2] == circumsphere_points[i]){
					printf("New point is already part of a triangle so part of our convex hull\r\n");
				}
			}
		}
		
		//Swap max point to the end of the list
		thrust::swap(circumsphere_points[outest_point], circumsphere_points[circumsphere_points_count - 1]);
	}
	
	if(tmpabc == test_index){
		for(int m = 0; m < circumsphere_triangles_count; ++m){
			const std::array<float, 3> current_p0_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[circumsphere_triangles[m]][0], blockid);
			const std::array<float, 3> current_p1_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[circumsphere_triangles[m]][1], blockid);
			const std::array<float, 3> current_p2_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[circumsphere_triangles[m]][2], blockid);
			
			const vec3 current_p0_position {current_p0_position_arr[0], current_p0_position_arr[1], current_p0_position_arr[2]};
			
			const std::array<float, 3> current_triangle_normal = alpha_shapes_calculate_triangle_normal({
				current_p0_position_arr,
				current_p1_position_arr,
				current_p2_position_arr
			});
			//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, triangles[circumsphere_triangles[m]][0], blockid, 1, current_triangle_normal, 0.0f, 0.0f);
			//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, triangles[circumsphere_triangles[m]][1], blockid, 1, current_triangle_normal, 0.0f, 0.0f);
			//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, triangles[circumsphere_triangles[m]][2], blockid, 1, current_triangle_normal, 0.0f, 0.0f);
		}
		
		
		for(int m = 0; m < circumsphere_points_count; ++m){
			//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, circumsphere_points[m], blockid, 2, {0.0f, 0.0f, 0.0f}, 0.0f, 0.0f);
		}
		if(circumsphere_points_count > 0){
			//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, circumsphere_points[0], blockid, 3, {0.0f, 0.0f, 0.0f}, 0.0f, 0.0f);
		}
	}

	/*
	int circumsphere_points_count = 0;
	//Doubled space for in-place sorting and merging; Keeping one point more do detect if we have more points in epsilon hull than we have space
	std::array<int, 2 * ALPHA_SHAPES_MAX_CIRCUMSPHERE_POINTS + 1> circumsphere_points = {};
	
	size_t already_sorted_count = 0;
	size_t sort_offset = 0;
	for(size_t i = 0; i < (particle_indices_count + ALPHA_SHAPES_MAX_CIRCUMSPHERE_POINTS - 1) / ALPHA_SHAPES_MAX_CIRCUMSPHERE_POINTS; ++i){
		const size_t current_sort_count = std::min(static_cast<size_t>(std::distance(particle_indices + (i * ALPHA_SHAPES_MAX_CIRCUMSPHERE_POINTS), particle_indices_end)), ALPHA_SHAPES_MAX_CIRCUMSPHERE_POINTS);
		
		std::array<int, 2 * ALPHA_SHAPES_MAX_CIRCUMSPHERE_POINTS + 1>::iterator sort_begin = circumsphere_points.begin() + ALPHA_SHAPES_MAX_CIRCUMSPHERE_POINTS + sort_offset;
		std::array<int, 2 * ALPHA_SHAPES_MAX_CIRCUMSPHERE_POINTS + 1>::iterator sort_end = circumsphere_points.begin() + ALPHA_SHAPES_MAX_CIRCUMSPHERE_POINTS + sort_offset + current_sort_count;
		
		//Copy points to sort array
		thrust::copy(thrust::seq, particle_indices + (i * ALPHA_SHAPES_MAX_CIRCUMSPHERE_POINTS), particle_indices + (i * ALPHA_SHAPES_MAX_CIRCUMSPHERE_POINTS) + current_sort_count, sort_begin);
		
		//Sort array
		thrust::sort(thrust::seq, sort_begin, sort_end, [&particle_buffer, &prev_partition, &blockid, &triangle_positions, &triangle_normal](const int& a, const int& b){
			return alpha_shapes_handle_triangle_compare_func(particle_buffer, prev_partition, blockid, triangle_positions, triangle_normal, a, b);
		});
		
		if(already_sorted_count > 0){
			//Merge with existing array keeping te smallest elements
			//Copied from https://www.geeksforgeeks.org/in-place-merge-sort/
			for(int gap = (already_sorted_count + current_sort_count + 1) / 2; gap > 0; gap = (gap == 1 ? 0 : (gap + 1) / 2)){
				for(int i = 0; (i + gap) < already_sorted_count + current_sort_count; i++){
					const int j = i + gap;
					if(alpha_shapes_handle_triangle_compare_func(particle_buffer, prev_partition, blockid, triangle_positions, triangle_normal, circumsphere_points[j], circumsphere_points[i])){
						thrust::swap(circumsphere_points[i], circumsphere_points[j]);
					}
				}
			}
		}else{
			//Just copy first sorted partition to begin.
			thrust::copy(thrust::seq, sort_begin, sort_end, circumsphere_points.begin());
		}
		
		//As soon as we have one more than the maximum possible points add offset to keep one point more
		if(already_sorted_count == ALPHA_SHAPES_MAX_CIRCUMSPHERE_POINTS && current_sort_count > 0){
			sort_offset = 1;
			already_sorted_count++;
			//NOTE: Decrease not needed cause already_sorted_count is bigger than current_sort_count always except in first round
			//current_sort_count--;
		}
		
		already_sorted_count = std::max(already_sorted_count, current_sort_count);
	}
	
	const std::array<float, 3> particle_position_arr_first = alpha_shapes_get_particle_position(particle_buffer, prev_partition, circumsphere_points[0], blockid);
	const vec3 particle_position_first {particle_position_arr_first[0], particle_position_arr_first[1], particle_position_arr_first[2]};
	
	//Test if in half_space; Also sorts out particles that lie in a plane with the triangle
	const bool in_halfspace_first = triangle_normal_vec.dot(particle_position_first - p0_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
	
	const bool part_of_triangle_first = (
		   (particle_position_arr_first[0] == triangle_positions[0][0] && particle_position_arr_first[1] == triangle_positions[0][1] && particle_position_arr_first[2] == triangle_positions[0][2])
		|| (particle_position_arr_first[0] == triangle_positions[1][0] && particle_position_arr_first[1] == triangle_positions[1][1] && particle_position_arr_first[2] == triangle_positions[1][2])
		|| (particle_position_arr_first[0] == triangle_positions[2][0] && particle_position_arr_first[1] == triangle_positions[2][1] && particle_position_arr_first[2] == triangle_positions[2][2])
	);
	
	vec3 sphere_center_first;
	float radius_first;
	alpha_shapes_get_circumsphere(triangle_positions[0], triangle_positions[1], triangle_positions[2], particle_position_arr_first, sphere_center_first.data_arr(), radius_first);
	
	const float lambda_first = triangle_normal_vec.dot(sphere_center_first - p0_position);
	
	//Only add points if they are not in halfspace and not part of the triangle
	if(!part_of_triangle_first && in_halfspace_first){
		//Add all smallest points in epsilon hull not lying behind previous points
		//TODO: Discard points that lie behind other points
		circumsphere_points_count = 1;//Add first point
		//vec3 particle_position_a = particle_position_first;
		//vec3 sphere_center_a = sphere_center_first;
		float radius_a = radius_first;
		//float lambda_a = lambda_first;
		for(size_t i = 0; i < already_sorted_count - 1; ++i){
			const std::array<float, 3> particle_position_arr_b = alpha_shapes_get_particle_position(particle_buffer, prev_partition, circumsphere_points[i + 1], blockid);
			
			const vec3 particle_position_b {particle_position_arr_b[0], particle_position_arr_b[1], particle_position_arr_b[2]};
			
			//Test if in half_space; Also sorts out particles that lie in a plane with the triangle
			const bool in_halfspace_b = triangle_normal_vec.dot(particle_position_b - p0_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
			
			const bool part_of_triangle_b = (
				   (particle_position_arr_b[0] == triangle_positions[0][0] && particle_position_arr_b[1] == triangle_positions[0][1] && particle_position_arr_b[2] == triangle_positions[0][2])
				|| (particle_position_arr_b[0] == triangle_positions[1][0] && particle_position_arr_b[1] == triangle_positions[1][1] && particle_position_arr_b[2] == triangle_positions[1][2])
				|| (particle_position_arr_b[0] == triangle_positions[2][0] && particle_position_arr_b[1] == triangle_positions[2][1] && particle_position_arr_b[2] == triangle_positions[2][2])
			);
			
			//Calculate delaunay spheres
			vec3 sphere_center_b;
			float radius_b;
			alpha_shapes_get_circumsphere(triangle_positions[0], triangle_positions[1], triangle_positions[2], particle_position_arr_b, sphere_center_b.data_arr(), radius_b);
			
			//Smaller delaunay sphere is the on that does not contain the other point
			//const vec3 diff_a = sphere_center_b - particle_position_a;
			//const float squared_distance_a = diff_a[0] * diff_a[0] + diff_a[1] * diff_a[1] + diff_a[2] * diff_a[2];
			//const vec3 diff_b = sphere_center_a - particle_position_b;
			//const float squared_distance_b = diff_b[0] * diff_b[0] + diff_b[1] * diff_b[1] + diff_b[2] * diff_b[2];
			
			//const float lambda_b = triangle_normal_vec.dot(sphere_center_b - p0_position);
			
			//const bool in_sphere = ((squared_distance_b - (radius_a * radius_a)) <= ALPHA_SHAPES_IN_SPHERE_THRESHOLD);
			//const bool in_sphere = ((lambda_b - lambda_a) <= ALPHA_SHAPES_IN_SPHERE_LAMBDA_THRESHOLD);
			
			const bool in_sphere = ((radius_b - radius_a) <= ALPHA_SHAPES_IN_SPHERE_RADIUS_THRESHOLD);
			
			//In epsilon hull and halfspace but nor part of the current triangle
			const bool in_epsilon_hull = !part_of_triangle_b && in_halfspace_b && in_sphere;
			
			if(!in_epsilon_hull){
				break;
			}
			circumsphere_points_count++;
			
			//particle_position_a = particle_position_b;
			//sphere_center_a = sphere_center_b;
			radius_a = radius_b;
			//lambda_a = lambda_b;
		}
	}*/
	
	/*
	if(circumsphere_points_count >= ALPHA_SHAPES_MAX_CIRCUMSPHERE_POINTS){
		printf("More points in circumsphere than we have memory for.\r\n");
		circumsphere_points_count--;
	}*/
	
	/*
	//Gather triangles from current set
	int circumsphere_triangles_count = 1;
	std::array<int, ALPHA_SHAPES_MAX_CIRCUMSPHERE_POINTS / 3> circumsphere_triangles = {};
	circumsphere_triangles[0] = triangle_index;//Add current triangle
	for(int current_triangle_index = 0; current_triangle_index < current_triangle_count + next_triangle_count; ++current_triangle_index){
		//If all points of the triangle are in circumsphere, add the triangle and remove the points
		int contact_count = 0;
		std::array<int, 3> contact_points = {};
		for(int i = 0; i < circumsphere_points_count; ++i){
			if(triangles[current_triangle_index][0] == circumsphere_points[i] || triangles[current_triangle_index][1] == circumsphere_points[i] || triangles[current_triangle_index][2] == circumsphere_points[i]){
				contact_points[contact_count++] = i;
			}
		}
		
		if(contact_count == 3){
			//Remove points
			for(int i = 2; i >=0; --i){//Backward loop to ensure correct swapping
				//Swap to end
				thrust::swap(circumsphere_points[contact_points[i]], circumsphere_points[circumsphere_points_count - 1]);
				
				//Decrease size
				circumsphere_points_count--;
			}
			
			//Add triangle
			circumsphere_triangles[circumsphere_triangles_count++] = current_triangle_index;
		}
	}
	*/
	
	printf("TEST %d\n", circumsphere_points_count);
	
	//FIXME: Handle triangles not convex due to numeric error
	//TODO: Maybe directly finalize all triangles here
	//Only handle tetrahedra if we found at least one point. If not all triangles are convex we finalize the current triangle
	if(circumsphere_points_count > 0){
		//Build tetrahedra
		alpha_shapes_build_tetrahedra(particle_buffer, prev_partition, alpha_shapes_particle_buffer, particle_indices, particle_indices_count, own_particle_indices, triangles, triangles_is_alpha, current_triangle_count, next_triangle_count, blockid, alpha, finalize_particles_start, finalize_particles_end, circumsphere_triangles.data(), circumsphere_points.data(), circumsphere_triangles_count, circumsphere_points_count, minimum_x, maximum_x, tmpabc, test_index);
		
		//If we are handling the first triangle, re-add it with flipped normal as outer triangle
		if(is_first_triangle){
			if(flipped_normal){
				triangles[current_triangle_count + next_triangle_count][0] = current_triangle[0];
				triangles[current_triangle_count + next_triangle_count][1] = current_triangle[1];
				triangles[current_triangle_count + next_triangle_count][2] = current_triangle[2];
			}else{
				triangles[current_triangle_count + next_triangle_count][0] = current_triangle[0];
				triangles[current_triangle_count + next_triangle_count][1] = current_triangle[2];
				triangles[current_triangle_count + next_triangle_count][2] = current_triangle[1];
			}
			next_triangle_count++;
		}
	}else{
		//Remove the face and move it to alpha if it is alpha
		
		//NOTE: Always false for first triangle
		if(triangles_is_alpha[triangle_index]){
			//FIXME:alpha_shapes_finalize_triangle(particle_buffer, prev_partition, alpha_shapes_particle_buffer, own_particle_indices.data(), temporary_convex_hull[0], blockid, own_last_active_particles_start, own_active_particles_start + own_active_particles_count);
		}
		
		//Swap contacting triangle to the end
		thrust::swap(triangles[triangle_index], triangles[current_triangle_count - 1]);
		thrust::swap(triangles_is_alpha[triangle_index], triangles_is_alpha[current_triangle_count - 1]);
		
		//Fill gap between current list and next list
		thrust::swap(triangles[current_triangle_count - 1], triangles[current_triangle_count + next_triangle_count - 1]);
		thrust::swap(triangles_is_alpha[current_triangle_count - 1], triangles_is_alpha[current_triangle_count + next_triangle_count - 1]);
		
		//Decrease triangle count
		current_triangle_count--;
	}
	
	if(tmpabc == test_index){
		for(int i = 0; i < particle_indices_count; ++i){
			const std::array<float, 3> current_particle_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, particle_indices[i], blockid);
			const vec3 current_particle_position {current_particle_position_arr[0], current_particle_position_arr[1], current_particle_position_arr[2]};
			
			//Test if point is on correct side; This also means that it is not in current convex hull
			bool in_convex_hull = true;
			for(int j = 0; j < current_triangle_count + next_triangle_count; ++j){
				const std::array<float, 3> current_p0_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[j][0], blockid);
				const std::array<float, 3> current_p1_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[j][1], blockid);
				const std::array<float, 3> current_p2_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[j][2], blockid);
				
				const vec3 current_p0_position {current_p0_position_arr[0], current_p0_position_arr[1], current_p0_position_arr[2]};
				
				const std::array<float, 3> current_triangle_normal = alpha_shapes_calculate_triangle_normal({
					current_p0_position_arr,
					current_p1_position_arr,
					current_p2_position_arr
				});
				
				const vec3 current_triangle_normal_vec {current_triangle_normal[0], current_triangle_normal[1], current_triangle_normal[2]};
					
				//Perform halfspace test
				const bool current_in_halfspace = current_triangle_normal_vec.dot(current_particle_position - current_p0_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
				
				//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, triangles[j][0], blockid, 1, current_triangle_normal, 0.0f, 0.0f);
				//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, triangles[j][1], blockid, 1, current_triangle_normal, 0.0f, 0.0f);
				//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, triangles[j][2], blockid, 1, current_triangle_normal, 0.0f, 0.0f);
				
				//If point is in an outer halfspace it is not in convex hull
				if(current_in_halfspace){
					in_convex_hull = false;
					break;
				}
			}
			
			if(in_convex_hull){
				tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, particle_indices[i], blockid, 4, {0.0f, 0.0f, 0.0f}, 0.0f, 0.0f);
			}
		}
	}
	if(tmpabc == test_index){
		for(int m = 0; m < circumsphere_points_count; ++m){
			tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, circumsphere_points[m], blockid, 2, {0.0f, 0.0f, 0.0f}, 0.0f, 0.0f);
		}
	}
	tmpabc++;
}

template<typename Partition, typename Grid, MaterialE MaterialType>
__global__ void alpha_shapes(const ParticleBuffer<MaterialType> particle_buffer, const Partition prev_partition, const Partition partition, const Grid grid, AlphaShapesParticleBuffer alpha_shapes_particle_buffer, AlphaShapesGridBuffer alpha_shapes_grid_buffer, const unsigned int start_index, const int test_index) {
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
	std::array<int, ALPHA_SHAPES_MAX_PARTICLE_COUNT> particle_indices = {};
	std::array<int, config::G_MAX_PARTICLES_IN_CELL> own_particle_indices = {};
	
	std::array<std::array<int, 3>, ALPHA_SHAPES_MAX_TRIANGLE_COUNT> triangles = {};
	std::array<bool, ALPHA_SHAPES_MAX_TRIANGLE_COUNT> triangles_is_alpha = {};
	
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
	
	//FIXME:
	if(cellid[0] != 126 || cellid[1] != 106 || cellid[2] != 128){
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
	
	int tmpabc = 1;//FIXME: Remove
	
	int current_triangle_count = 0;
	int next_triangle_count = 0;
	
	bool found_initial_tetrahedron = false;
	float minimum_x;
	float maximum_x;
	if(found_initial_triangle){
		current_triangle_count = 1;
		//Create first tetrahedron
		/*{
			std::array<int, 3> current_triangle = triangles[0];
			
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
			found_initial_tetrahedron = alpha_shapes_get_fourth_point(particle_buffer, prev_partition, alpha_shapes_particle_buffer, tmpabc, test_index, particle_indices.begin(), particle_indices.begin() + particle_bucket_size, blockid, current_triangle_positions, p3_id, false);
			
			if(found_initial_tetrahedron){
				const std::array<float, 3> p3_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, p3_id, blockid);
				const vec3 p3_position {p3_position_arr[0], p3_position_arr[1], p3_position_arr[2]};
				
				const std::array<float, 3>& triangle_normal = alpha_shapes_calculate_triangle_normal(current_triangle_positions);
				const vec3 triangle_normal_vec {triangle_normal[0], triangle_normal[1], triangle_normal[2]};
				
				//If in halfspace, current normal is pointing inwards (towards the forth point), so it has to be flipped. Otherwise we only flip the normal temporarly to find the forth point
				if(triangle_normal_vec.dot(p3_position - p0_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD){
					//Flip triangle order to get correct normal direction pointing outwards
					thrust::swap(triangles[0][1], triangles[0][2]);
					thrust::swap(p1_position_arr, p2_position_arr);
					thrust::swap(p1_position, p2_position);
				}else{
					//Flip face normal
					current_triangle_positions = {
						  p0_position.data_arr()
						, p2_position.data_arr()
						, p1_position.data_arr()
					};
					thrust::swap(current_triangle[1], current_triangle[2]);
				}
				
				//NOTE: current_triangle_positions still has inwards pointing normal here; same goes for current_triangle, which gurantees correct new normal calculation
			
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
				//FIXME: Correct distance (e.g. intersection of triangle halfspace with xy and xz boundaries
				minimum_x = std::numeric_limits<float>::min();//minimum_x = sphere_center[0] - radius;
				maximum_x = std::numeric_limits<float>::max();//maximum_x = sphere_center[0] + radius;
			}
		}*/
		
		alpha_shapes_handle_triangle(particle_buffer, prev_partition, alpha_shapes_particle_buffer, particle_indices.data(), particle_bucket_size, own_particle_indices.data(), triangles.data(), triangles_is_alpha.data(), current_triangle_count, next_triangle_count, blockid, alpha, 0, own_particle_bucket_size, true, 0, minimum_x, maximum_x, tmpabc, test_index);
		//tmpabc++;
		
		for(size_t i = 0; i < current_triangle_count + next_triangle_count; ++i){
			//if(
			//	  (current_triangle[0] == triangles[i][0] || current_triangle[0] == triangles[i][1] || current_triangle[0] == triangles[i][2])
			//	&&(current_triangle[1] == triangles[i][0] || current_triangle[1] == triangles[i][1] || current_triangle[1] == triangles[i][2])
			//	&&(current_triangle[2] == triangles[i][0] || current_triangle[2] == triangles[i][1] || current_triangle[2] == triangles[i][2])
			//){
				//printf("X %d %d %d # %d %d %d \n", tmpabc, current_triangle_count, next_triangle_count, triangles[i][0], triangles[i][1], triangles[i][2]);
			//}
		}
		
		current_triangle_count = next_triangle_count;
		next_triangle_count = 0;
		
		found_initial_tetrahedron = (current_triangle_count > 1);
	}
	
	int active_particles_start = 0;
	int active_particles_count = 0;
	
	int own_active_particles_start = 0;
	int own_active_particles_count = 0;
	int own_last_active_particles_start = 0;
	
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
		
		printf("B %d %d - ", active_particles_start, active_particles_count);
		
		//current_triangle_count = 4;
		
		while(own_active_particles_count > 0){
			minimum_x = std::numeric_limits<float>::max();
			maximum_x = std::numeric_limits<float>::min();
			
			while(current_triangle_count > 0){
				/*const std::array<int, 3> current_triangle = triangles[0];

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
				const bool found = alpha_shapes_get_fourth_point(particle_buffer, prev_partition, alpha_shapes_particle_buffer, tmpabc, test_index, particle_indices.begin() + active_particles_start, particle_indices.begin() + active_particles_start + active_particles_count, blockid, current_triangle_positions, p3_id, true);
				const std::array<float, 3> p3_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, p3_id, blockid);
				const vec3 p3_position {p3_position_arr[0], p3_position_arr[1], p3_position_arr[2]};

				if(tmpabc == test_index){
					const std::array<float, 3>& triangle_normal = alpha_shapes_calculate_triangle_normal(current_triangle_positions);
					tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, current_triangle[0], blockid, 1, triangle_normal, 0.0f, (found ? 1.0f : 0.0f));
					tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, current_triangle[1], blockid, 2, triangle_normal, 0.0f, (found ? 1.0f : 0.0f));
					tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, current_triangle[2], blockid, 3, triangle_normal, 0.0f, (found ? 1.0f : 0.0f));
				}
				
				//If no tetrahedron could be created we have a boundary face of the convex hull
				if(found){
					//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, p3_id, blockid, 5, {0.0f, 0.0f, 0.0f}, 0.0f, 1.0f);
					if(tmpabc == test_index){
						tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, p3_id, blockid, 4, {0.0f, 0.0f, 0.0f}, 0.0f, 1.0f);
					}
					//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, p3_id, blockid, 5, {0.0f, 0.0f, 0.0f}, 0.0f, 1.0f);
					
					//Check alpha shape condition and mark faces accordingly
					std::array<float, 3> sphere_center;
					float radius;
					alpha_shapes_get_circumsphere(p0_position_arr, p1_position_arr, p2_position_arr, p3_position_arr, sphere_center, radius);
					const bool is_alpha = (radius * radius <= alpha);
					
					//Check contact conditions and update triangle list
					//NOTE: This changes the triangle counts
					alpha_shapes_check_contact_condition(particle_buffer, prev_partition, alpha_shapes_particle_buffer, own_particle_indices.data(), triangles.data(), triangles_is_alpha.data(), current_triangle_count, next_triangle_count, blockid, p3_id, is_alpha, own_last_active_particles_start, own_active_particles_start + own_active_particles_count);
					
					//Update bounds
					//FIXME: Correct distance (e.g. intersection of triangle halfspace with xy and xz boundaries
					minimum_x = std::numeric_limits<float>::min();//std::min(minimum_x, sphere_center[0] - radius);
					maximum_x = std::numeric_limits<float>::max();//std::max(maximum_x, sphere_center[0] + radius);
				}else{
					//Remove the face and move it to alpha if it is alpha
					
					if(triangles_is_alpha[0]){
						//FIXME:alpha_shapes_finalize_triangle(particle_buffer, prev_partition, alpha_shapes_particle_buffer, own_particle_indices.data(), current_triangle, blockid, own_last_active_particles_start, own_active_particles_start + own_active_particles_count);
					}
					
					//Swap contacting triangle to the end
					thrust::swap(triangles[0], triangles[current_triangle_count - 1]);
					thrust::swap(triangles_is_alpha[0], triangles_is_alpha[current_triangle_count - 1]);
					
					//Fill gap between current list and next list
					thrust::swap(triangles[current_triangle_count - 1], triangles[current_triangle_count + next_triangle_count - 1]);
					thrust::swap(triangles_is_alpha[current_triangle_count - 1], triangles_is_alpha[current_triangle_count + next_triangle_count - 1]);
					
					//Decrease triangle count
					current_triangle_count--;
				}
				
				for(size_t i = 0; i < current_triangle_count + next_triangle_count; ++i){
					//if(
					//	  (current_triangle[0] == triangles[i][0] || current_triangle[0] == triangles[i][1] || current_triangle[0] == triangles[i][2])
					//	&&(current_triangle[1] == triangles[i][0] || current_triangle[1] == triangles[i][1] || current_triangle[1] == triangles[i][2])
					//	&&(current_triangle[2] == triangles[i][0] || current_triangle[2] == triangles[i][1] || current_triangle[2] == triangles[i][2])
					//){
						printf("X %d %d %d # %d %d %d \n", tmpabc, current_triangle_count, next_triangle_count, triangles[i][0], triangles[i][1], triangles[i][2]);
					//}
				}
				
				
				*/
				alpha_shapes_handle_triangle(particle_buffer, prev_partition, alpha_shapes_particle_buffer, particle_indices.data() + active_particles_start, active_particles_count, own_particle_indices.data(), triangles.data(), triangles_is_alpha.data(), current_triangle_count, next_triangle_count, blockid, alpha, own_last_active_particles_start, own_active_particles_start + own_active_particles_count, false, 0, minimum_x, maximum_x, tmpabc, test_index);
				//tmpabc++;
				
				for(size_t i = 0; i < current_triangle_count + next_triangle_count; ++i){
					//if(
					//	  (current_triangle[0] == triangles[i][0] || current_triangle[0] == triangles[i][1] || current_triangle[0] == triangles[i][2])
					//	&&(current_triangle[1] == triangles[i][0] || current_triangle[1] == triangles[i][1] || current_triangle[1] == triangles[i][2])
					//	&&(current_triangle[2] == triangles[i][0] || current_triangle[2] == triangles[i][1] || current_triangle[2] == triangles[i][2])
					//){
						//printf("X %d %d %d # %d %d %d \n", tmpabc, current_triangle_count, next_triangle_count, triangles[i][0], triangles[i][1], triangles[i][2]);
					//}
				}
			}
			
			printf("A %d - ", next_triangle_count);
			
			//All triangles have been handled, either checked for tetrahedron or removed due to contacting
			
			//Swap triangle lists
			current_triangle_count = next_triangle_count;
			next_triangle_count = 0;
			
			//FIXME:
			if(current_triangle_count > 2 * particle_bucket_size - 4){
				printf("More triangles than convex hull can have\n");
				current_triangle_count = 0;
			}
			
			//Move sweep line
			
			//Move lower bound; Remove particles that are out of range
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
			//FIXME:alpha_shapes_finalize_particles(particle_buffer, prev_partition, alpha_shapes_particle_buffer, own_particle_indices.data(), blockid, own_last_active_particles_start, own_active_particles_start);
			
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
			//FIXME:alpha_shapes_finalize_triangle(particle_buffer, prev_partition, alpha_shapes_particle_buffer, own_particle_indices.data(), triangles[current_triangle_index], blockid, own_last_active_particles_start, own_active_particles_start);
		}
	}
	
	//Finalize all particles left
	//FIXME:alpha_shapes_finalize_particles(particle_buffer, prev_partition, alpha_shapes_particle_buffer, own_particle_indices.data(), blockid, own_last_active_particles_start, own_active_particles_start);
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