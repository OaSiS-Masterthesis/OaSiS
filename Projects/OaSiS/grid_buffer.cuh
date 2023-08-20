#ifndef GRID_BUFFER_CUH
#define GRID_BUFFER_CUH
#include <MnSystem/Cuda/HostUtils.hpp>

#include "kernels.cuh"
#include "settings.h"
#include "managed_memory.hpp"

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
	
	managed_memory_type* managed_memory;
	
	//TODO: check that this is small enough (smaller than 1.0, bigger than 0.0)?
	const std::array<float, 3> relative_offset;

	template<typename Allocator>
	explicit GridBuffer(Allocator allocator, managed_memory_type* managed_memory, const std::array<float, 3>& relative_offset = {0.0f, 0.0f, 0.0f})
		: base_t {spawn<grid_buffer_, orphan_signature>(allocator)}
		, relative_offset(relative_offset), managed_memory(managed_memory){}

	template<typename Allocator>
	void check_capacity(Allocator allocator, std::size_t capacity) {
		if(capacity > this->capacity) {
			this->resize(allocator, capacity);
		}
	}

	template<typename CudaContext>
	void reset(int block_count, CudaContext& cu_dev) {
		bool this_is_locked = this->is_locked();
		
		//check_cuda_errors(cudaMemsetAsync((void *)&this->val_1d(_0, 0), 0, grid_block_::size * block_count, cu_dev.stream_compute()));
		managed_memory->managed_memory_type::acquire<MemoryType::DEVICE>(this->acquire());
		cu_dev.compute_launch({block_count, config::G_BLOCKVOLUME}, clear_grid, *this);
		managed_memory->release(
			(this_is_locked ? nullptr : this->release())
		);
	}
	
	__forceinline__ __host__ __device__ const std::array<float, 3>& get_offset() const{
		return relative_offset;
	}
};

struct TemporaryGridBuffer : Instance<TemporaryGridBufferData> {
	using base_t = Instance<TemporaryGridBufferData>;
	
	managed_memory_type* managed_memory;

	template<typename Allocator>
	explicit TemporaryGridBuffer(Allocator allocator, managed_memory_type* managed_memory)
		: base_t {spawn<TemporaryGridBufferData, orphan_signature>(allocator)} 
		, managed_memory(managed_memory){}

	template<typename Allocator>
	void check_capacity(Allocator allocator, std::size_t capacity) {
		if(capacity > this->capacity) {
			this->resize(allocator, capacity);
		}
	}

	template<typename CudaContext>
	void reset(int block_count, CudaContext& cu_dev) {
		bool this_is_locked = this->is_locked();
		
		managed_memory->managed_memory_type::acquire<MemoryType::DEVICE>(this->acquire());
		check_cuda_errors(cudaMemsetAsync((void *)&this->val_1d(_0, 0), 0, TemporaryGridBlockData::size * block_count, cu_dev.stream_compute()));
		managed_memory->release(
			(this_is_locked ? nullptr : this->release())
		);
	}
};

}// namespace mn

#endif