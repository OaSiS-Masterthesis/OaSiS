#ifndef SURFACE_RECONSTRUCTION_CUH
#define SURFACE_RECONSTRUCTION_CUH

#include "particle_buffer.cuh"

namespace mn {
	
constexpr float SURFACE_HALFSPACE_TEST_THRESHOLD = 0.0f;//FIXME:1e-7;//TODO: Maybe adjust threshold
	
//NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables, readability-identifier-naming) Check is buggy and reports variable errors for template arguments
using SurfaceParticleBufferData  = Structural<StructuralType::DENSE, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::SUM_POW2_ALIGN>, ParticleBinDomain, attrib_layout::SOA, f32_, f32_, f32_, f32_, f32_, f32_, f32_, f32_>;//Point type (integer bytes as float, needs to be casted accordingly), normal, mean_curvature, gauss_curvature ; temporary: summed_area, normal, summed_angles, summed_laplacians
//NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables, readability-identifier-naming)
	
struct SurfaceParticleBuffer : Instance<particle_buffer_<SurfaceParticleBufferData>> {
	using base_t							 = Instance<particle_buffer_<SurfaceParticleBufferData>>;
	
	managed_memory_type* managed_memory;

	SurfaceParticleBuffer() = default;

	template<typename Allocator>
	SurfaceParticleBuffer(Allocator allocator, managed_memory_type* managed_memory, std::size_t count)
		: base_t {spawn<particle_buffer_<SurfaceParticleBufferData>, orphan_signature>(allocator, count)}
		, managed_memory(managed_memory)
		{}
};

enum class SurfacePointType {
	OUTER_POINT = 0,
	INNER_POINT,
	ISOLATED_POINT,
	CURVE_1D,
	CURVE_2D,
	TOTAL
};

__forceinline__ __host__ __device__ std::array<float, 3> surface_calculate_triangle_normal(const std::array<std::array<float, 3>, 3>& triangle_positions){
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

//TODO: Actually we'd also need a threshold for normal (halfspace spanning more than one direction) but this can cause problems (convex hull becoming infinite if ends of two adjacent halfspaces are parallel)
__forceinline__ __host__ __device__ bool surface_test_in_halfspace(const std::array<std::array<float, 3>, 3>& triangle_positions, const std::array<float, 3> particle_position){
	const std::array<vec3, 3> triangle_positions_vec {
		  vec3(triangle_positions[0][0], triangle_positions[0][1], triangle_positions[0][2])
		, vec3(triangle_positions[1][0], triangle_positions[1][1], triangle_positions[1][2])
		, vec3(triangle_positions[2][0], triangle_positions[2][1], triangle_positions[2][2])
	};
	
	const std::array<float, 3> triangle_normal = surface_calculate_triangle_normal(triangle_positions);
	
	const vec3 triangle_normal_vec {triangle_normal[0], triangle_normal[1], triangle_normal[2]};
	const vec3 particle_position_vec {particle_position[0], particle_position[1], particle_position[2]};
	
	//Find nearest point
	//TODO: Maybe ensure that we have an ordering if several have same distance; Or use average;
	const std::array<vec3, 3>::const_iterator nearest_point = thrust::min_element(thrust::seq, triangle_positions_vec.begin(), triangle_positions_vec.end(), [&particle_position_vec](const vec3& a, const vec3& b){
		return (particle_position_vec - a).dot(particle_position_vec - a) < (particle_position_vec - b).dot(particle_position_vec - b);
	});
		
	//Perform halfspace test
	const bool current_in_halfspace = triangle_normal_vec.dot(particle_position_vec - *nearest_point) > SURFACE_HALFSPACE_TEST_THRESHOLD;
	
	return current_in_halfspace;
}
	
}// namespace mn

#endif