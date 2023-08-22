#ifndef SURFACE_RECONSTRUCTION_CUH
#define SURFACE_RECONSTRUCTION_CUH

#include "particle_buffer.cuh"

namespace mn {
	
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
	
}// namespace mn

#endif