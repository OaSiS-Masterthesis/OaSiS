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

//NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables, readability-identifier-naming) Check is buggy and reports variable errors for template arguments
using AlphaShapesParticleBufferData  = Structural<StructuralType::DENSE, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::SUM_POW2_ALIGN>, ParticleBinDomain, attrib_layout::SOA, i32_, f32_, f32_, f32_, f32_, f32_>;//Point type, normal, mean_curvature, gauss_curvature
//NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables, readability-identifier-naming)

struct AlphaShapesParticleBuffer : Instance<particle_buffer_<AlphaShapesParticleBufferData>> {
	using base_t							 = Instance<particle_buffer_<AlphaShapesParticleBufferData>>;

	AlphaShapesParticleBuffer() = default;

	template<typename Allocator>
	AlphaShapesParticleBuffer(Allocator allocator, std::size_t count)
		: base_t {spawn<particle_buffer_<AlphaShapesParticleBufferData>, orphan_signature>(allocator, count)}
		{}
};

//TODO: Make magic numbers to constants where suitable
//TODO: Ensure call dimensions and such are small enough to allow narrowing conversations. Or directly use unsigned where possible
//TODO: Maybe use names instead of formula signs for better understanding
//NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers, readability-identifier-naming, misc-definitions-in-headers) CUDA does not yet support std::span; Common names for physical formulas; Cannot declare __global__ functions inline

enum class AlphaShapePointType {
	OUTER_POINT = 0,
	INNER_POINT,
	ISOLATED_POINT,
	CURVE_1D,
	CURVE_2D,
	TOTAL
};

struct AlphaShapeParticle{
	int id = -1;
	int blockno = -1;
	int src_blockno = -1;
	std::array<float, 3> position = {};
	bool used = false;
};

struct AlphaShapeTriangle{
	std::array<size_t, 3> particle_indices = {};
	bool is_alpha = false;
	
	static __forceinline__ __host__ __device__ std::array<float, 3> calculate_normal(const std::array<std::array<float, 3>, 3>& triangle_positions){
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
};

template<typename Partition, MaterialE MaterialType, size_t NumParticles>
__forceinline__ __device__ void alpha_shapes_fetch_particles(const ParticleBuffer<MaterialType> particle_buffer, const ParticleBuffer<MaterialType> next_particle_buffer, const Partition prev_partition, const Partition partition, std::array<AlphaShapeParticle, NumParticles>& particles, const int particle_bucket_size, const int src_blockno){
	const auto blockid			   = partition.active_keys[blockIdx.x];
	
	for(int particle_id_in_block = 0; particle_id_in_block < particle_bucket_size; particle_id_in_block++) {
		//Fetch index of the advection source
		int advection_source_blockno;
		int source_pidib;
		{
			//Fetch advection (direction at high bits, particle in in cell at low bits)
			const int advect = next_particle_buffer.blockbuckets[src_blockno * config::G_PARTICLE_NUM_PER_BLOCK + particle_id_in_block];

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
		particles[particle_id_in_block].id = source_pidib;
		particles[particle_id_in_block].blockno = advection_source_blockno;
		particles[particle_id_in_block].src_blockno = src_blockno;
		
		FetchParticleBufferDataIntermediate fetch_particle_buffer_tmp = {};
		fetch_particle_buffer_data<MaterialType>(particle_buffer, advection_source_blockno, source_pidib, fetch_particle_buffer_tmp);
		particles[particle_id_in_block].position = fetch_particle_buffer_tmp.pos;
	}
}

template<MaterialE MaterialType>
__forceinline__ __device__ void alpha_shapes_store_particle(const ParticleBuffer<MaterialType> particle_buffer, AlphaShapesParticleBuffer alpha_shapes_particle_buffer, const AlphaShapeParticle& particle, const AlphaShapePointType point_type, const std::array<float, 3>& normal, const float mean_curvature, const float gauss_curvature){
	auto particle_bin													= alpha_shapes_particle_buffer.ch(_0, particle_buffer.bin_offsets[particle.blockno] + particle.id / config::G_BIN_CAPACITY);
	particle_bin.val(_0, particle.id  % config::G_BIN_CAPACITY) = static_cast<int32_t>(point_type);
	particle_bin.val(_1, particle.id  % config::G_BIN_CAPACITY) = normal[0];
	particle_bin.val(_2, particle.id  % config::G_BIN_CAPACITY) = normal[1];
	particle_bin.val(_3, particle.id  % config::G_BIN_CAPACITY) = normal[2];
	particle_bin.val(_4, particle.id  % config::G_BIN_CAPACITY) = mean_curvature;
	particle_bin.val(_5, particle.id  % config::G_BIN_CAPACITY) = gauss_curvature;
}

//Basically copied from https://rodolphe-vaillant.fr/entry/127/find-a-tetrahedron-circumcenter
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
    const float denominator = 0.5 / (ba_x * cross_cd_x + ba_y * cross_cd_y + ba_z * cross_cd_z);
	
	//FIXME: Maybe clamp denominator if near to zero to ensure nearly degenerated tetrahedra are not unstable
 
    // Calculate offset (from 'a') of circumcenter.
    const float circ_x = (len_ba * cross_cd_x + len_ca * cross_db_x + len_da * cross_bc_x);
    const float circ_y = (len_ba * cross_cd_y + len_ca * cross_db_y + len_da * cross_bc_y);
    const float circ_z = (len_ba * cross_cd_z + len_ca * cross_db_z + len_da * cross_bc_z);
 
    center[0] = a[0] + circ_x * denominator;
    center[1] = a[1] + circ_y * denominator;
    center[2] = a[2] + circ_z * denominator;
	radius = std::sqrt(circ_x * circ_x + circ_y * circ_y + circ_z * circ_z) * denominator;
}

template<size_t NumParticles>
__forceinline__ __device__ bool alpha_shapes_get_first_triangle(const std::array<AlphaShapeParticle, NumParticles>& particles, const size_t count, AlphaShapeTriangle& triangle){
	//Pick first point
	const AlphaShapeParticle p0 = particles[0];
	const vec3 p0_position {p0.position[0], p0.position[1], p0.position[2]};
	
	//Find nearest point
	float current_minimum_distance = std::numeric_limits<float>::max();
	AlphaShapeParticle p1;
	size_t p1_index;
	for(int particle_id_in_block = 1; particle_id_in_block < count; particle_id_in_block++) {
		const vec3 particle_position {particles[particle_id_in_block].position[0], particles[particle_id_in_block].position[1], particles[particle_id_in_block].position[2]};
		
		//Skip particles with same position
		if(particle_position[0] != p0_position[0] || particle_position[1] != p0_position[01] || particle_position[2] != p0_position[2]){
			//Calculate distance
			const vec3 diff = p0_position - particle_position;
			const float squared_distance = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
			
			if(squared_distance < current_minimum_distance){
				current_minimum_distance = squared_distance;
				p1 = particles[particle_id_in_block];
				p1_index = particle_id_in_block;
			}
		}
	}
	
	if(current_minimum_distance < std::numeric_limits<float>::max()){
		const vec3 p1_position {p1.position[0], p1.position[1], p1.position[2]};
		
		//Find smallest meridian sphere
		float current_smallest_meridian_sphere_radius = std::numeric_limits<float>::max();
		//AlphaShapeParticle p2;
		size_t p2_index;
		for(int particle_id_in_block = 1; particle_id_in_block < count; particle_id_in_block++) {
			const vec3 particle_position {particles[particle_id_in_block].position[0], particles[particle_id_in_block].position[1], particles[particle_id_in_block].position[2]};
			
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
					//p2 = particles[particle_id_in_block];
					p2_index = particle_id_in_block;
				}
			}
		}
		
		//Return indices
		triangle.particle_indices[0] = 0;
		triangle.particle_indices[1] = p1_index;
		triangle.particle_indices[2] = p2_index;
		
		return (current_smallest_meridian_sphere_radius < std::numeric_limits<float>::max());
	}
}

template<size_t NumParticles, typename ForwardIterator>
__forceinline__ __device__ bool alpha_shapes_get_fourth_point(const ForwardIterator& begin, const ForwardIterator& end, const std::array<std::array<float, 3>, 3>& triangle_positions, size_t& point_id){
	const std::array<float, 3>& triangle_normal = AlphaShapeTriangle::calculate_normal(triangle_positions);
	
	const vec3 triangle_normal_vec {triangle_normal[0], triangle_normal[1], triangle_normal[2]};
	
	//Only search in normal direction
	typename std::array<AlphaShapeParticle, NumParticles>::iterator p3_iter = thrust::min_element(thrust::seq, begin, end, [&triangle_positions, &triangle_normal_vec](const AlphaShapeParticle& a, const AlphaShapeParticle& b){
		const vec3 particle_position_a {a.position[0], a.position[1], a.position[2]};
		const vec3 particle_position_b {b.position[0], b.position[1], b.position[2]};
		
		//Test if in half_space; Also sorts out particles that lie in a plane with the triangle
		bool in_halfspace_a = triangle_normal_vec.dot(particle_position_a) > 0;
		bool in_halfspace_b = triangle_normal_vec.dot(particle_position_b) > 0;
		
		if(!in_halfspace_a && !in_halfspace_b){
			//Calculate delaunay spheres
			vec3 sphere_center_a;
			vec3 sphere_center_b;
			float radius_a;
			float radius_b;
			alpha_shapes_get_circumsphere(triangle_positions[0], triangle_positions[1], triangle_positions[2], a.position, sphere_center_a.data_arr(), radius_a);
			alpha_shapes_get_circumsphere(triangle_positions[0], triangle_positions[1], triangle_positions[2], b.position, sphere_center_b.data_arr(), radius_b);
			
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
	
	point_id = std::distance(begin, p3_iter);
	
	const vec3 p3_position {p3_iter->position[0], p3_iter->position[1], p3_iter->position[2]};
	
	//FIXME: Do we have to recheck that no other point lies in circumsphere or is this guranteed?
		
	//Test if in half_space; Also sorts out particles that lie in a plane with the triangle
	return (triangle_normal_vec.dot(p3_position) > 0);
}

template<size_t NumParticles, size_t NumTriangles>
__forceinline__ __device__ void alpha_shapes_check_contact_condition(std::array<AlphaShapeParticle, NumParticles>& particles, std::array<AlphaShapeTriangle, NumTriangles>& triangles, std::array<AlphaShapeTriangle, NumTriangles>& alpha_triangles, size_t& alpha_triangle_count, size_t& current_triangle_count, size_t& next_triangle_count, int& current_triangle_index, const size_t p3_id, const bool is_alpha){
	const AlphaShapeTriangle& current_triangle = triangles[current_triangle_index];
	
	std::array<size_t, 4> contact_indices;
	size_t face_contacts = 1;
	
	contact_indices[0] = current_triangle_index;
	
	//Check contact conditions and update triangle list
	if(particles[p3_id].used){
		//Find triangles in touch
		for(int contact_triangle_index = 0; contact_triangle_index < current_triangle_count; contact_triangle_index++) {
			const AlphaShapeTriangle& contact_triangle = triangles[current_triangle_index];
			
			//Test for face contact
			if(
				   (contact_triangle.particle_indices[0] == current_triangle.particle_indices[0] || contact_triangle.particle_indices[0] == current_triangle.particle_indices[1] || contact_triangle.particle_indices[0] == current_triangle.particle_indices[2] || contact_triangle.particle_indices[0] == p3_id)
				&& (contact_triangle.particle_indices[1] == current_triangle.particle_indices[0] || contact_triangle.particle_indices[1] == current_triangle.particle_indices[1] || contact_triangle.particle_indices[1] == current_triangle.particle_indices[2] || contact_triangle.particle_indices[1] == p3_id)
				&& (contact_triangle.particle_indices[2] == current_triangle.particle_indices[0] || contact_triangle.particle_indices[2] == current_triangle.particle_indices[1] || contact_triangle.particle_indices[2] == current_triangle.particle_indices[2] || contact_triangle.particle_indices[2] == p3_id)
			){
				contact_indices[face_contacts++] = contact_triangle_index;
			}
		}
	}else{
		//Mark as used
		particles[p3_id].used = true;
	}
	
	//Next triangle list starts at end of active triangle list
	size_t next_triangle_start = current_triangle_count;
	
	for(int contact_triangle_index = 0; contact_triangle_index < face_contacts; contact_triangle_index++) {
		//If the new tetrahedron is in alpha and the current face is not, then the current face is a boundary face that has to be kept; Same the other way round
		if(triangles[contact_indices[contact_triangle_index]].is_alpha != is_alpha){
			alpha_triangles[alpha_triangle_count++] = triangles[contact_indices[contact_triangle_index]];
		}
		
		//Swap contact triangles to end of list to remove them
		size_t swap_index;
		if(contact_indices[contact_triangle_index] < current_triangle_count){
			swap_index = current_triangle_count - 1;//Swap with last active triangle
			
			//Decrease triangle count
			current_triangle_count--;
		}else{
			swap_index = next_triangle_start;//Swap with first next triangle
			
			//Increase start
			next_triangle_start++;
			
			//Decrease next triangle count
			next_triangle_count--;
		}
		
		//Swap contacting triangle to the end
		thrust::swap(triangles[contact_indices[contact_triangle_index]], triangles[swap_index]);
	}
	
	//Empty space is now from [current_triangle_count; next_triangle_start[
	
	//Add new triangles
	//Ensure correct order (current_triangle cw normal points outwards)
	if(face_contacts == 1){
		triangles[next_triangle_start - 3].particle_indices[0] = current_triangle.particle_indices[0];
		triangles[next_triangle_start - 3].particle_indices[1] = current_triangle.particle_indices[1];
		triangles[next_triangle_start - 3].particle_indices[2] = p3_id;
		triangles[next_triangle_start - 2].particle_indices[0] = current_triangle.particle_indices[1];
		triangles[next_triangle_start - 2].particle_indices[1] = current_triangle.particle_indices[2];
		triangles[next_triangle_start - 2].particle_indices[2] = p3_id;
		triangles[next_triangle_start - 1].particle_indices[0] = current_triangle.particle_indices[2];
		triangles[next_triangle_start - 1].particle_indices[1] = current_triangle.particle_indices[0];
		triangles[next_triangle_start - 1].particle_indices[2] = p3_id;
		
		triangles[next_triangle_start - 3].is_alpha = is_alpha;
		triangles[next_triangle_start - 2].is_alpha = is_alpha;
		triangles[next_triangle_start - 1].is_alpha = is_alpha;
	}else if(face_contacts == 2){
		std::array<size_t, 3> triangle_counts_per_vertex;
		for(int contact_triangle_index = 1; contact_triangle_index < face_contacts; contact_triangle_index++) {
			//Check which vertices are contacting
			for(int vertex_index = 0; vertex_index < 3; vertex_index++) {
				const size_t particle_index = triangles[contact_indices[contact_triangle_index]].particle_indices[vertex_index];
				if(particle_index == current_triangle.particle_indices[0]){
					triangle_counts_per_vertex[0]++;
				}else if(particle_index == current_triangle.particle_indices[1]){
					triangle_counts_per_vertex[1]++;
				}else{//particle_index == current_triangle.particle_indices[2]
					triangle_counts_per_vertex[2]++;
				}
			}
		}
		
		if(triangle_counts_per_vertex[0] == 0){
			triangles[next_triangle_start - 2].particle_indices[0] = current_triangle.particle_indices[0];
			triangles[next_triangle_start - 2].particle_indices[1] = current_triangle.particle_indices[1];
			triangles[next_triangle_start - 2].particle_indices[2] = p3_id;
			triangles[next_triangle_start - 1].particle_indices[0] = current_triangle.particle_indices[2];
			triangles[next_triangle_start - 1].particle_indices[1] = current_triangle.particle_indices[0];
			triangles[next_triangle_start - 1].particle_indices[2] = p3_id;
		}else if(triangle_counts_per_vertex[1] == 0){
			triangles[next_triangle_start - 2].particle_indices[0] = current_triangle.particle_indices[0];
			triangles[next_triangle_start - 2].particle_indices[1] = current_triangle.particle_indices[1];
			triangles[next_triangle_start - 2].particle_indices[2] = p3_id;
			triangles[next_triangle_start - 1].particle_indices[0] = current_triangle.particle_indices[1];
			triangles[next_triangle_start - 1].particle_indices[1] = current_triangle.particle_indices[2];
			triangles[next_triangle_start - 1].particle_indices[2] = p3_id;
		}else {//triangle_counts_per_vertex[2] == 0
			triangles[next_triangle_start - 2].particle_indices[0] = current_triangle.particle_indices[1];
			triangles[next_triangle_start - 2].particle_indices[1] = current_triangle.particle_indices[2];
			triangles[next_triangle_start - 2].particle_indices[2] = p3_id;
			triangles[next_triangle_start - 1].particle_indices[0] = current_triangle.particle_indices[2];
			triangles[next_triangle_start - 1].particle_indices[1] = current_triangle.particle_indices[0];
			triangles[next_triangle_start - 1].particle_indices[2] = p3_id;
		}
		
		triangles[next_triangle_start - 2].is_alpha = is_alpha;
		triangles[next_triangle_start - 1].is_alpha = is_alpha;
	}else if(face_contacts == 3){
		std::array<size_t, 3> triangle_counts_per_vertex;
		for(int contact_triangle_index = 1; contact_triangle_index < face_contacts; contact_triangle_index++) {
			//Check which vertices are contacting
			for(int vertex_index = 0; vertex_index < 3; vertex_index++) {
				const size_t particle_index = triangles[contact_indices[contact_triangle_index]].particle_indices[vertex_index];
				if(particle_index == current_triangle.particle_indices[0]){
					triangle_counts_per_vertex[0]++;
				}else if(particle_index == current_triangle.particle_indices[1]){
					triangle_counts_per_vertex[1]++;
				}else{//particle_index == current_triangle.particle_indices[2]
					triangle_counts_per_vertex[2]++;
				}
			}
		}
		
		if(triangle_counts_per_vertex[0] == 2){
			triangles[next_triangle_start - 1].particle_indices[0] = current_triangle.particle_indices[1];
			triangles[next_triangle_start - 1].particle_indices[1] = current_triangle.particle_indices[2];
			triangles[next_triangle_start - 1].particle_indices[2] = p3_id;
		}else if(triangle_counts_per_vertex[1] == 2){
			triangles[next_triangle_start - 1].particle_indices[0] = current_triangle.particle_indices[2];
			triangles[next_triangle_start - 1].particle_indices[1] = current_triangle.particle_indices[0];
			triangles[next_triangle_start - 1].particle_indices[2] = p3_id;
		}else {//triangle_counts_per_vertex[2] == 2
			triangles[next_triangle_start - 1].particle_indices[0] = current_triangle.particle_indices[0];
			triangles[next_triangle_start - 1].particle_indices[1] = current_triangle.particle_indices[1];
			triangles[next_triangle_start - 1].particle_indices[2] = p3_id;
		}
		
		triangles[next_triangle_start - 1].is_alpha = is_alpha;
	}//Otherwise nothing to do, just faces removed
	
	next_triangle_start -= (4 - face_contacts);
	next_triangle_count += (4 - face_contacts);
	
	//Fill space between lists
	for(size_t swap_index = current_triangle_count; swap_index < next_triangle_start; ++swap_index){
		//Swap triangle
		thrust::swap(triangles[swap_index], triangles[next_triangle_start + next_triangle_count - 1]);
		
		//Decrease start
		next_triangle_start--;
	}
	
	//Decrease index to revisit current triangle that is now the new swapped triangle from the end
	current_triangle_index--;
}

template<MaterialE MaterialType, size_t NumParticles, size_t NumTriangles>
__forceinline__ __device__ void alpha_shapes_finalize_particles(const ParticleBuffer<MaterialType> particle_buffer, AlphaShapesParticleBuffer alpha_shapes_particle_buffer, const std::array<AlphaShapeParticle, NumParticles>& particles, const std::array<AlphaShapeTriangle, NumTriangles>& alpha_triangles, const size_t alpha_triangle_count, const int src_blockno, const size_t range_start, const size_t range_end){
	for(int particle_id_in_block = range_start; particle_id_in_block < range_end; particle_id_in_block++) {
		//Only handle own particles
		if(particles[particle_id_in_block].src_blockno == src_blockno){
			const vec3 particle_position {particles[particle_id_in_block].position[0], particles[particle_id_in_block].position[1], particles[particle_id_in_block].position[2]};
		
			//TODO: Maybe different curvature calculation?
			vec3 particle_normal {0.0f, 0.0f, 0.0f};
			
			float summed_angles = 0.0f;
			float summed_area = 0.0f;
			vec3 summed_laplacians {0.0f, 0.0f, 0.0f};
		
			std::array<AlphaShapeParticle, NumParticles> neighbour_points;
			size_t contact_triangles_count = 0;
			for(int current_triangle_index = 0; current_triangle_index < alpha_triangle_count; current_triangle_index++) {
				const std::array<vec3, 3> triangle_positions {
					  vec3(particles[alpha_triangles[current_triangle_index].particle_indices[0]].position[0], particles[alpha_triangles[current_triangle_index].particle_indices[0]].position[1], particles[alpha_triangles[current_triangle_index].particle_indices[0]].position[2])
					, vec3(particles[alpha_triangles[current_triangle_index].particle_indices[1]].position[0], particles[alpha_triangles[current_triangle_index].particle_indices[1]].position[1], particles[alpha_triangles[current_triangle_index].particle_indices[1]].position[2])
					, vec3(particles[alpha_triangles[current_triangle_index].particle_indices[2]].position[0], particles[alpha_triangles[current_triangle_index].particle_indices[2]].position[1], particles[alpha_triangles[current_triangle_index].particle_indices[2]].position[2])
				};
				
				vec3 face_normal;
				vec_cross_vec_3d(face_normal.data_arr(), (triangle_positions[1] - triangle_positions[0]).data_arr(), (triangle_positions[2] - triangle_positions[0]).data_arr());
				
				const float face_normal_length = sqrt(face_normal[0] * face_normal[0] + face_normal[1] * face_normal[1] + face_normal[2] * face_normal[2]);
				
				//Normalize
				face_normal = face_normal / face_normal_length;
				
				const float face_area = 0.5f * face_normal_length;
				
				size_t contact_index;
				size_t neighbour_index;
				vec3 contact_barycentric {0.0f, 0.0f, 0.0f};
				bool contact = false;
				if(alpha_triangles[current_triangle_index].particle_indices[0] == particle_id_in_block){
					contact_index = 0;
					neighbour_index = 1;
					contact_barycentric[0] = 1.0f;
					contact = true;
				}else if(alpha_triangles[current_triangle_index].particle_indices[1] == particle_id_in_block){
					contact_index = 1;
					neighbour_index = 2;
					contact_barycentric[1] = 1.0f;
					contact = true;
				}else if(alpha_triangles[current_triangle_index].particle_indices[2] == particle_id_in_block){
					contact_index = 2;
					neighbour_index = 0;
					contact_barycentric[2] = 1.0f;
					contact = true;
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
						contact = true;
					}
				}
				
				if(contact){
					//Accumulate values
					if(contact_barycentric[0] > 0.0f && contact_barycentric[1] > 0.0f && contact_barycentric[0] > 0.0f){//Point somewhere in triangle
						//Use face normal
						particle_normal += face_normal;
						
						//Gauss curvature
						summed_angles += 2 * M_PI;
						summed_area += 1.0f;//Just ensure this is not zero
						
						//Break cause a point can only lie on one triangle if not on their edges
						break;
					}else if(contact_barycentric[0] == 1.0f || contact_barycentric[1] == 1.0f || contact_barycentric[0] == 1.0f){//Point on vertex
						float cosine = (triangle_positions[(contact_index + 1) % 3] - triangle_positions[contact_index]).dot(triangle_positions[(contact_index + 2) % 3] - triangle_positions[contact_index]) / sqrt((triangle_positions[(contact_index + 1) % 3] - triangle_positions[contact_index]).dot(triangle_positions[(contact_index + 1) % 3] - triangle_positions[contact_index]) * (triangle_positions[(contact_index + 2) % 3] - triangle_positions[contact_index]).dot(triangle_positions[(contact_index + 2) % 3] - triangle_positions[contact_index]));
						cosine = std::min(std::max(cosine, -1.0f), 1.0f);
						const float angle = std::acos(cosine);
						
						//Normal
						particle_normal += angle * face_normal;
						
						//Gauss curvature
						summed_angles += angle;
						summed_area += face_area * (1.0f / 3.0f);
						
						//Store neighbour
					neighbour_points[contact_triangles_count] = particles[alpha_triangles[current_triangle_index].particle_indices[neighbour_index]];
					}else{//Point on edge
						//Use half normal
						particle_normal += 0.5f * face_normal;
						
						//Gauss curvature
						summed_angles += M_PI;
						summed_area += face_area * (1.0f / 3.0f);
						
						size_t opposite_index;
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
					
					contact_triangles_count++;
				}
			}
			
			AlphaShapePointType point_type = AlphaShapePointType::ISOLATED_POINT;
			
			//Calculate particle states
			if(contact_triangles_count == 0){
				//Isolated point or point in shell
				point_type = AlphaShapePointType::ISOLATED_POINT;//FIXME: All are currently treated as isolated points
				
				//TODO: Or might be part of curve or thin surface
				//TODO: Decide wheter interior or exterior point
			}else if(contact_triangles_count > 2){
				point_type = AlphaShapePointType::OUTER_POINT;
				
				//Sort points by angle so that incident faces are near each other

				//Copied from https://stackoverflow.com/questions/47949485/sorting-a-list-of-3d-points-in-clockwise-order
				//FIXME: Verify this is correct!
				
				const AlphaShapeParticle first_neighbour = neighbour_points[0];
				const vec3 first_neighbour_position {first_neighbour.position[0], first_neighbour.position[1], first_neighbour.position[2]};
				
				const vec3 first_diff = first_neighbour_position - particle_position;
				
				vec3 first_cross;
				vec_cross_vec_3d(first_cross.data_arr(), first_diff.data_arr(), particle_position.data_arr());
				
				thrust::sort(thrust::seq, neighbour_points.begin(), neighbour_points.begin() + contact_triangles_count, [&particles, &particle_id_in_block, &particle_position, &first_diff, &first_cross](const AlphaShapeParticle& a, const AlphaShapeParticle& b){
					const vec3 neighbour_a_position {a.position[0], a.position[1], a.position[2]};
					const vec3 neighbour_b_position {b.position[0], b.position[1], b.position[2]};
					
					const vec3 diff_a = neighbour_a_position - particle_position;
					const vec3 diff_b = neighbour_b_position - particle_position;
					
					const float dot_a = first_cross.dot(diff_a);	
					const float dot_b = first_cross.dot(diff_b);
					
					
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
					
					const AlphaShapeParticle prev_neighbour = neighbour_points[prev_neighbour_index];
					const AlphaShapeParticle current_neighbour = neighbour_points[current_neighbour_index];
					const AlphaShapeParticle next_neighbour = neighbour_points[next_neighbour_index];
					
					const vec3 prev_position {prev_neighbour.position[0], prev_neighbour.position[1], prev_neighbour.position[2]};
					const vec3 current_position {current_neighbour.position[0], current_neighbour.position[1], current_neighbour.position[2]};
					const vec3 next_position {next_neighbour.position[0], next_neighbour.position[1], next_neighbour.position[2]};
					
					
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
			
			alpha_shapes_store_particle(particle_buffer, alpha_shapes_particle_buffer, particles[particle_id_in_block], point_type, particle_normal.data_arr(), mean_curvature, gauss_curvature);
		}
	}
}

//FIXME: if n+2 (=5) or more points lie on a sphere there are several possible triangulations; Can we ignore this or do we have to handle this and if so how can we do this; Maybe seep line ensures consistency?
template<typename Partition, MaterialE MaterialType>
__global__ void alpha_shapes(const ParticleBuffer<MaterialType> particle_buffer, ParticleBuffer<MaterialType> next_particle_buffer, const Partition prev_partition, const Partition partition, AlphaShapesParticleBuffer alpha_shapes_particle_buffer) {
	//Big enough to cover all cells near the current cell that can contain particles near enough to make a face an alpha face
	constexpr size_t KERNEL_SIZE = static_cast<size_t>(config::G_DX / const_sqrt(config::MAX_ALPHA)) + 1;//NOTE:Static cast required for expression being const
	constexpr size_t KERNEL_LENGTH = 2 * KERNEL_SIZE + 1;//Sidelength of the kernel cube
	constexpr size_t NUMBER_OF_BLOCKS = KERNEL_LENGTH * KERNEL_LENGTH;
	
	constexpr size_t MAX_TRIANGLE_COUNT = NUMBER_OF_BLOCKS * config::ALPHA_SHAPES_TRIANGLES_PER_BLOCK;
	
	const int src_blockno		   = static_cast<int>(blockIdx.x);
	const ivec3 blockid			   = partition.active_keys[blockIdx.x];
	
	
	//TODO: Smaller alpha based on density or cohesion maybe
	const float alpha = config::MAX_ALPHA;
	
	//FIXME: Actually we only need to handle blocks in radius of dx + sqrt(alpha)
	//Fetch particles
	int particle_bucket_size = 0;
	std::array<AlphaShapeParticle, NUMBER_OF_BLOCKS * config::G_PARTICLE_NUM_PER_BLOCK> particles;
	for(int i = -static_cast<int>(KERNEL_SIZE); i <= static_cast<int>(KERNEL_SIZE); ++i){
		for(int j = -static_cast<int>(KERNEL_SIZE); j <= static_cast<int>(KERNEL_SIZE); ++j){
			for(int k = -static_cast<int>(KERNEL_SIZE); k <= static_cast<int>(KERNEL_SIZE); ++k){
				const ivec3 current_blockid {blockid[0] + i, blockid[1] + j, blockid[2] + k};
				const int current_blockno = partition.query(current_blockid);
				
				//Only handle active blocks
				if(current_blockno != -1){
					const int current_bucket_size = next_particle_buffer.particle_bucket_sizes[current_blockno];
					
					particle_bucket_size += current_bucket_size;
					alpha_shapes_fetch_particles(particle_buffer, next_particle_buffer, prev_partition, partition, particles, current_bucket_size, current_blockno);
				}
			}
		}
	}
	
	//Filter by max distance
	const vec3 block_center = ((blockid * static_cast<int>(config::G_BLOCKSIZE)).cast<float>() + 3.5f) * config::G_DX;//0->3.5 1->7.5 ...; 1.5 is lower bound of block 0, 5.5 is lower bound of block 1, ...
	for(int particle_id_in_block = 0; particle_id_in_block < particle_bucket_size; particle_id_in_block++) {
		const vec3 particle_position {particles[particle_id_in_block].position[0], particles[particle_id_in_block].position[1], particles[particle_id_in_block].position[2]};
		
		//Calculate distance to center
		const vec3 diff = block_center - particle_position;
		const float squared_distance = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
		
		if(squared_distance > (config::G_DX * config::G_DX + alpha)){
			//Remove by exchanging with last element
			thrust::swap(particles[particle_id_in_block], particles[particle_bucket_size - 1]);
			
			//Decrease count
			particle_bucket_size--;
			
			//Revisit swapped particle
			particle_id_in_block--;
		}
	}
	
	//If we have no particles in the bucket return
	if(particle_bucket_size == 0) {
		return;
	}
	
	std::array<AlphaShapeTriangle, MAX_TRIANGLE_COUNT> triangles;
	std::array<AlphaShapeTriangle, MAX_TRIANGLE_COUNT> alpha_triangles;
		
	//Sort by ascending x
	thrust::sort(thrust::seq, particles.begin(), particles.begin() + particle_bucket_size, [](const AlphaShapeParticle& a, const AlphaShapeParticle& b){
		return a.position[0] < b.position[0];
	});
	
	//Build delaunay triangulation with this points and keep all these intersecting the node
	
	//Create first triangle
	const bool found_initial_triangle = alpha_shapes_get_first_triangle(particles, particle_bucket_size, triangles[0]);
	
	bool found_initial_tetrahedron = false;
	float minimum_x;
	float maximum_x;
	if(found_initial_triangle){
		//Mark as used
		for(int i = 0; i < 3; i++) {
			particles[triangles[0].particle_indices[i]].used = true;
		}
		
		//Create first tetrahedron
		{
			const AlphaShapeTriangle current_triangle = triangles[0];
			
			const vec3 p0_position {particles[current_triangle.particle_indices[0]].position[0], particles[current_triangle.particle_indices[0]].position[1], particles[current_triangle.particle_indices[0]].position[2]};
			const vec3 p1_position {particles[current_triangle.particle_indices[1]].position[0], particles[current_triangle.particle_indices[1]].position[1], particles[current_triangle.particle_indices[1]].position[2]};
			const vec3 p2_position {particles[current_triangle.particle_indices[2]].position[0], particles[current_triangle.particle_indices[2]].position[1], particles[current_triangle.particle_indices[2]].position[2]};
			
			//Find smallest delaunay tetrahedron
			std::array<std::array<float, 3>, 3> current_triangle_positions {
				  p0_position.data_arr()
				, p1_position.data_arr()
				, p2_position.data_arr()
			};
			
			size_t p3_id;
			found_initial_tetrahedron = alpha_shapes_get_fourth_point<NUMBER_OF_BLOCKS * config::G_PARTICLE_NUM_PER_BLOCK>(particles.begin(), particles.end(), current_triangle_positions, p3_id);
			
			if(!found_initial_tetrahedron){
				//Flip face normal
				current_triangle_positions = {
					  p0_position.data_arr()
					, p2_position.data_arr()
					, p1_position.data_arr()
				};
				
				//Flip triangle order to get correct normal direction pointing outwards
				thrust::swap(triangles[0].particle_indices[1], triangles[0].particle_indices[2]);
				
				alpha_shapes_get_fourth_point<NUMBER_OF_BLOCKS * config::G_PARTICLE_NUM_PER_BLOCK>(particles.begin(), particles.end(), current_triangle_positions, p3_id);
			}
			
			if(found_initial_tetrahedron){
				const vec3 p3_position {particles[p3_id].position[0], particles[p3_id].position[1], particles[p3_id].position[2]};
			
				//Check alpha shape condition and mark faces accordingly
				std::array<float, 3> sphere_center;
				float radius;
				alpha_shapes_get_circumsphere(particles[current_triangle.particle_indices[0]].position, particles[current_triangle.particle_indices[1]].position, particles[current_triangle.particle_indices[2]].position, particles[p3_id].position, sphere_center, radius);
				const bool is_alpha = (radius * radius <= alpha);
				
				//Mark as used
				particles[p3_id].used = true;
				
				//Add triangles
				//Ensure correct order (current_triangle cw normal points outwards)
				triangles[1].particle_indices[0] = current_triangle.particle_indices[0];
				triangles[1].particle_indices[1] = current_triangle.particle_indices[1];
				triangles[1].particle_indices[2] = p3_id;
				triangles[2].particle_indices[0] = current_triangle.particle_indices[1];
				triangles[2].particle_indices[1] = current_triangle.particle_indices[2];
				triangles[2].particle_indices[2] = p3_id;
				triangles[3].particle_indices[0] = current_triangle.particle_indices[2];
				triangles[3].particle_indices[1] = current_triangle.particle_indices[0];
				triangles[3].particle_indices[2] = p3_id;
				
				triangles[0].is_alpha = is_alpha;
				triangles[1].is_alpha = is_alpha;
				triangles[2].is_alpha = is_alpha;
				triangles[3].is_alpha = is_alpha;
				
				//Init bounds
				minimum_x = sphere_center[0] - radius;
				maximum_x = sphere_center[0] + radius;
			}
		}
	}
	
	size_t active_particles_start = 0;
	size_t active_particles_count = 0;
	size_t last_active_particles_start = 0;
	
	size_t current_triangle_count = 0;
	size_t alpha_triangle_count = 0;
	
	if(found_initial_tetrahedron){
		//Init sweep line
		
		//Move upper bound; Activate additional particles based on range to new triangles
		for(int particle_id_in_block = active_particles_start + active_particles_count; particle_id_in_block < particle_bucket_size; particle_id_in_block++) {
			if(particles[particle_id_in_block].position[0] > maximum_x){
				break;
			}
			active_particles_count++;
		}
		
		current_triangle_count = 4;
		size_t next_triangle_count = 0;
		
		while(active_particles_count > 0){
			minimum_x = std::numeric_limits<float>::max();
			maximum_x = std::numeric_limits<float>::min();
			
			for(int current_triangle_index = 0; current_triangle_index < current_triangle_count; current_triangle_index++) {
				const AlphaShapeTriangle current_triangle = triangles[current_triangle_index];
				
				const vec3 p0_position {particles[current_triangle.particle_indices[0]].position[0], particles[current_triangle.particle_indices[0]].position[1], particles[current_triangle.particle_indices[0]].position[2]};
				const vec3 p1_position {particles[current_triangle.particle_indices[1]].position[0], particles[current_triangle.particle_indices[1]].position[1], particles[current_triangle.particle_indices[1]].position[2]};
				const vec3 p2_position {particles[current_triangle.particle_indices[2]].position[0], particles[current_triangle.particle_indices[2]].position[1], particles[current_triangle.particle_indices[2]].position[2]};
				
				//Find smallest delaunay tetrahedron
				const std::array<std::array<float, 3>, 3> current_triangle_positions {
					  p0_position.data_arr()
					, p1_position.data_arr()
					, p2_position.data_arr()
				};
				
				size_t p3_id;
				const bool found = alpha_shapes_get_fourth_point<NUMBER_OF_BLOCKS * config::G_PARTICLE_NUM_PER_BLOCK>(particles.begin(), particles.end(), current_triangle_positions, p3_id);
				const vec3 p3_position {particles[p3_id].position[0], particles[p3_id].position[1], particles[p3_id].position[2]};
				
				//If no tetrahedron could be created we have a boundary face of the convex hull
				if(found){
					//Remove the face and move it to alpha if it is alpha
					
					//Swap contacting triangle to the end
					thrust::swap(triangles[current_triangle_index], triangles[current_triangle_count - 1]);
					
					//Decrease triangle count
					current_triangle_count--;
					
					//Revisit swaped triangle
					current_triangle_index--;
					
					if(current_triangle.is_alpha){
						alpha_triangles[alpha_triangle_count++] = current_triangle;
					}
				}else{
				
					//Check alpha shape condition and mark faces accordingly
					std::array<float, 3> sphere_center;
					float radius;
					alpha_shapes_get_circumsphere(particles[current_triangle.particle_indices[0]].position, particles[current_triangle.particle_indices[1]].position, particles[current_triangle.particle_indices[2]].position, particles[p3_id].position, sphere_center, radius);
					const bool is_alpha = (radius * radius <= alpha);
					
					//Check contact conditions and update triangle list
					//NOTE: This decreases current_triangle_index and changes the triangle counts
					alpha_shapes_check_contact_condition(particles, triangles, alpha_triangles, alpha_triangle_count, current_triangle_count, next_triangle_count, current_triangle_index, p3_id, is_alpha);
					
					//Update bounds
					minimum_x = std::min(minimum_x, sphere_center[0] - radius);
					maximum_x = std::min(minimum_x, sphere_center[0] + radius);
				}
			}
			
			//All triangles have been handled, either checked for tetrahedron or removed due to contacting
			
			//Swap triangle lists
			current_triangle_count = next_triangle_count;
			next_triangle_count = 0;
			
			//Move sweep line
			
			//Move lower bound; Remopve particles that are out of range
			for(int particle_id_in_block = active_particles_start; particle_id_in_block < active_particles_start + active_particles_count; particle_id_in_block++) {
				if(particles[particle_id_in_block].position[0] >= minimum_x){
					break;
				}
				active_particles_start++;
				active_particles_count--;
			}
			
			//Move upper bound; Activate additional particles based on range to new triangles
			for(int particle_id_in_block = active_particles_start + active_particles_count; particle_id_in_block < particle_bucket_size; particle_id_in_block++) {
				if(particles[particle_id_in_block].position[0] > maximum_x){
					break;
				}
				active_particles_count++;
			}
			
			//Finalize particles and faces if possible (based on sweep line)
			alpha_shapes_finalize_particles(particle_buffer, alpha_shapes_particle_buffer, particles, alpha_triangles, alpha_triangle_count, src_blockno, last_active_particles_start, active_particles_start);
			
			//Remove faces from all lists if all of their particles were handled
			for(int current_triangle_index = 0; current_triangle_index < alpha_triangle_count; current_triangle_index++) {
				if(
					   (alpha_triangles[current_triangle_index].particle_indices[0] < active_particles_start)
					&& (alpha_triangles[current_triangle_index].particle_indices[1] < active_particles_start)
					&& (alpha_triangles[current_triangle_index].particle_indices[2] < active_particles_start)
				){
					//Swap contacting triangle to the end
					thrust::swap(alpha_triangles[current_triangle_index], alpha_triangles[alpha_triangle_count - 1]);
					
					//Decrease triangle count
					alpha_triangle_count--;
					
					//Revisit swaped triangle
					current_triangle_index--;
				}
			}
			
			//Save last bounds
			last_active_particles_start = active_particles_start;
		}
	}else{
		active_particles_start = particle_bucket_size;//Handle all particles
	}
	
	//Add all faces left over at the end to alpha_triangles when no more points are found if they are marked as alpha.
	for(int current_triangle_index = 0; current_triangle_index < current_triangle_count; current_triangle_index++) {
		if(triangles[current_triangle_index].is_alpha){
			alpha_triangles[alpha_triangle_count++] = triangles[current_triangle_index];
		}
	}
	
	//Finalize all particles left
	alpha_shapes_finalize_particles(particle_buffer, alpha_shapes_particle_buffer, particles, alpha_triangles, alpha_triangle_count, src_blockno, last_active_particles_start, active_particles_start);
}

//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers, readability-identifier-naming, misc-definitions-in-headers)
}// namespace mn

#endif