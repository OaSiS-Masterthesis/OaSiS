#ifndef GRID_BUFFER_CUH
#define GRID_BUFFER_CUH
#include <MnSystem/Cuda/HostUtils.hpp>

#include "kernels.cuh"
#include "settings.h"

namespace mn {
using namespace placeholder;//NOLINT(google-build-using-namespace) Allow placeholders to be included generally for simplification

//NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables, readability-identifier-naming) Check is buggy and reports variable errors from template arguments
using grid_block_  = Structural<StructuralType::DENSE, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::SUM_POW2_ALIGN>, BlockDomain, attrib_layout::SOA, f32_, f32_, f32_, f32_>;//mass, momentum/velocity
using grid_		   = Structural<StructuralType::DENSE, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::COMPACT>, GridDomain, attrib_layout::AOS, grid_block_>;
using grid_buffer_ = Structural<StructuralType::DYNAMIC, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::COMPACT>, GridBufferDomain, attrib_layout::AOS, grid_block_>;

using TemporaryGridBlockData  = Structural<StructuralType::DENSE, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::SUM_POW2_ALIGN>, BlockDomain, attrib_layout::SOA, f32_, f32_, f32_, f32_>;
using TemporaryGridBufferData = Structural<StructuralType::DYNAMIC, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::COMPACT>, GridBufferDomain, attrib_layout::AOS, TemporaryGridBlockData>;
//NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables, readability-identifier-naming)

struct GridBuffer : Instance<grid_buffer_> {
	using base_t = Instance<grid_buffer_>;
	
	//TODO: check that this is small enough (smaller than 1.0, bigger than 0.0)?
	const std::array<float, 3> relative_offset;

	template<typename Allocator>
	explicit GridBuffer(Allocator allocator, const std::array<float, 3>& relative_offset = {0.0f, 0.0f, 0.0f})
		: base_t {spawn<grid_buffer_, orphan_signature>(allocator)}
		, relative_offset(relative_offset){}

	template<typename Allocator>
	void check_capacity(Allocator allocator, std::size_t capacity) {
		if(capacity > this->capacity) {
			this->resize(allocator, capacity);
		}
	}

	template<typename CudaContext>
	void reset(int block_count, CudaContext& cu_dev) {
		//check_cuda_errors(cudaMemsetAsync((void *)&this->val_1d(_0, 0), 0, grid_block_::size * block_count, cu_dev.stream_compute()));
		cu_dev.compute_launch({block_count, config::G_BLOCKVOLUME}, clear_grid, *this);
	}
	
	__forceinline__ __host__ __device__ const std::array<float, 3>& get_offset() const{
		return relative_offset;
	}
};


struct TemporaryGridBuffer : Instance<TemporaryGridBufferData> {
	using base_t = Instance<TemporaryGridBufferData>;

	template<typename Allocator>
	explicit TemporaryGridBuffer(Allocator allocator)
		: base_t {spawn<TemporaryGridBufferData, orphan_signature>(allocator)} {}

	template<typename Allocator>
	void check_capacity(Allocator allocator, std::size_t capacity) {
		if(capacity > this->capacity) {
			this->resize(allocator, capacity);
		}
	}

	template<typename CudaContext>
	void reset(int block_count, CudaContext& cu_dev) {
		check_cuda_errors(cudaMemsetAsync((void *)&this->val_1d(_0, 0), 0, TemporaryGridBlockData::size * block_count, cu_dev.stream_compute()));
	}
};

}// namespace mn

#endif