#ifndef HASH_TABLE_CUH
#define HASH_TABLE_CUH
#include <MnBase/Object/Structural.h>

#include <MnSystem/Cuda/HostUtils.hpp>

#include "settings.h"
#include "managed_memory.hpp"

namespace mn {

template<int>
struct HaloPartition {
	template<typename Allocator>
	HaloPartition(Allocator allocator, managed_memory_type* managed_memory, int max_block_count) {
		(void) allocator;
		(void) managed_memory;
		(void) max_block_count;
	}

	template<typename Allocator>
	void resize_partition(Allocator allocator, std::size_t prev_capacity, std::size_t capacity) {}
	void copy_to(HaloPartition& other, std::size_t block_count, cudaStream_t stream) {}
};

template<>
struct HaloPartition<1> {
	managed_memory_type* managed_memory;
	
	int* halo_count;
	int h_count;
	char* halo_marks;///< halo particle blocks
	int* overlap_marks;
	ivec3* halo_blocks;

	template<typename Allocator>
	HaloPartition(Allocator allocator, managed_memory_type* managed_memory, int max_block_count)
		: h_count(0) 
		, managed_memory(managed_memory){
		halo_count	  = static_cast<int*>(allocator.allocate(sizeof(char) * max_block_count));
		halo_marks	  = static_cast<char*>(allocator.allocate(sizeof(char) * max_block_count));
		overlap_marks = static_cast<int*>(allocator.allocate(sizeof(int) * max_block_count));
		halo_blocks	  = static_cast<ivec3*>(allocator.allocate(sizeof(ivec3) * max_block_count));
	}

	void copy_to(HaloPartition& other, std::size_t block_count, cudaStream_t stream) const {
		other.h_count = h_count;
		
		void* halo_marks_ptr = halo_marks;
		void* other_halo_marks_ptr = other.halo_marks;
		void* overlap_marks_ptr = overlap_marks;
		void* other_overlap_marks_ptr = other.overlap_marks;
		void* halo_blocks_ptr = halo_blocks;
		void* other_halo_blocks_ptr = other.halo_blocks;
		managed_memory->acquire_any(
			  reinterpret_cast<void**>(&halo_marks_ptr)
			, reinterpret_cast<void**>(&other_halo_marks_ptr)
			, reinterpret_cast<void**>(&overlap_marks_ptr)
			, reinterpret_cast<void**>(&other_overlap_marks_ptr)
			, reinterpret_cast<void**>(&halo_blocks_ptr)
			, reinterpret_cast<void**>(&other_halo_blocks_ptr)
		);
		check_cuda_errors(cudaMemcpyAsync(other_halo_marks_ptr, halo_marks_ptr, sizeof(char) * block_count, cudaMemcpyDefault, stream));
		check_cuda_errors(cudaMemcpyAsync(other_overlap_marks_ptr, overlap_marks_ptr, sizeof(int) * block_count, cudaMemcpyDefault, stream));
		check_cuda_errors(cudaMemcpyAsync(other_halo_blocks_ptr, halo_blocks_ptr, sizeof(ivec3) * block_count, cudaMemcpyDefault, stream));
		managed_memory->release(halo_marks, other.halo_marks, overlap_marks, other.overlap_marks, halo_blocks, other.halo_blocks);
	}

	template<typename Allocator>
	void resize_partition(Allocator allocator, std::size_t prev_capacity, std::size_t capacity) {
		allocator.deallocate(halo_marks, sizeof(char) * prev_capacity);
		allocator.deallocate(overlap_marks, sizeof(int) * prev_capacity);
		allocator.deallocate(halo_blocks, sizeof(ivec3) * prev_capacity);
		halo_marks	  = static_cast<char*>(allocator.allocate(sizeof(char) * capacity));
		overlap_marks = static_cast<int*>(allocator.allocate(sizeof(int) * capacity));
		halo_blocks	  = static_cast<ivec3*>(allocator.allocate(sizeof(ivec3) * capacity));
	}

	void reset_halo_count(cudaStream_t stream) const {
		void* halo_count_ptr = halo_count;
		if(managed_memory->get_memory_type(halo_count_ptr) == MemoryType::HOST){
			managed_memory->managed_memory_type::acquire<MemoryType::HOST>(reinterpret_cast<void**>(&halo_count_ptr));
			memset(halo_count_ptr, 0, sizeof(int));
		}else{
			managed_memory->managed_memory_type::acquire<MemoryType::DEVICE>(reinterpret_cast<void**>(&halo_count_ptr));
			check_cuda_errors(cudaMemsetAsync(halo_count_ptr, 0, sizeof(int), stream));
		}
		managed_memory->release(halo_count);
	}

	void reset_overlap_marks(uint32_t neighbor_block_count, cudaStream_t stream) const {
		void* overlap_marks_ptr = overlap_marks;
		if(managed_memory->get_memory_type(overlap_marks_ptr) == MemoryType::HOST){
			managed_memory->managed_memory_type::acquire<MemoryType::HOST>(reinterpret_cast<void**>(&overlap_marks_ptr));
			memset(overlap_marks_ptr, 0, sizeof(int) * neighbor_block_count);
		}else{
			managed_memory->managed_memory_type::acquire<MemoryType::DEVICE>(reinterpret_cast<void**>(&overlap_marks_ptr));
			check_cuda_errors(cudaMemsetAsync(overlap_marks_ptr, 0, sizeof(int) * neighbor_block_count, stream));
		}
		managed_memory->release(overlap_marks);
	}

	void retrieve_halo_count(cudaStream_t stream) {
		void* halo_count_ptr = halo_count;
		managed_memory->acquire_any(reinterpret_cast<void**>(&halo_count_ptr));
		check_cuda_errors(cudaMemcpyAsync(&h_count, halo_count_ptr, sizeof(int), cudaMemcpyDefault, stream));
		managed_memory->release(halo_count);
	}
};

//NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables, readability-identifier-naming) Check is buggy and reporst variable errors fro template arguments
using block_partition_ = Structural<StructuralType::HASH, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::COMPACT>, GridDomain, attrib_layout::AOS, empty_>;
//NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables, readability-identifier-naming)

//NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic) Using pointer arithmetics cause library and allocators do so.
template<int Opt = 1>
struct Partition
	: Instance<block_partition_>
	, HaloPartition<Opt> {
	using base_t	  = Instance<block_partition_>;
	using halo_base_t = HaloPartition<Opt>;
	using block_partition_::key_t;
	using block_partition_::value_t;
	static_assert(sentinel_v == (value_t) (-1), "sentinel value not full 1s\n");

	template<typename Allocator>
	Partition(Allocator allocator, managed_memory_type* managed_memory, int max_block_count)
		: halo_base_t {allocator, managed_memory, max_block_count} {
		allocate_table(allocator, max_block_count);
		/// init
		reset();
	}

	~Partition() = default;

	Partition(const Partition& other)				 = default;
	Partition(Partition&& other) noexcept			 = default;
	Partition& operator=(const Partition& other)	 = default;
	Partition& operator=(Partition&& other) noexcept = default;

	template<typename Allocator>
	void resize_partition(Allocator allocator, std::size_t capacity) {
		halo_base_t::resize_partition(allocator, this->capacity, capacity);
		resize_table(allocator, capacity);
	}

	void reset() {
		this->Instance<block_partition_>::count = this->Instance<block_partition_>::count_virtual;
		this->index_table = this->index_table_virtual;
		if(managed_memory->get_memory_type(this->Instance<block_partition_>::count) == MemoryType::HOST){
			managed_memory->managed_memory_type::acquire<MemoryType::HOST>(reinterpret_cast<void**>(&this->Instance<block_partition_>::count));
			memset(this->Instance<block_partition_>::count, 0, sizeof(value_t));
		}else{
			managed_memory->managed_memory_type::acquire<MemoryType::DEVICE>(reinterpret_cast<void**>(&this->Instance<block_partition_>::count));
			check_cuda_errors(cudaMemset(this->Instance<block_partition_>::count, 0, sizeof(value_t)));
		}
		if(managed_memory->get_memory_type(this->index_table) == MemoryType::HOST){
			managed_memory->managed_memory_type::acquire<MemoryType::HOST>(reinterpret_cast<void**>(&this->index_table));
			memset(this->index_table, 0xff, sizeof(value_t) * domain::extent);
		}else{
			managed_memory->managed_memory_type::acquire<MemoryType::DEVICE>(reinterpret_cast<void**>(&this->index_table));
			check_cuda_errors(cudaMemset(this->index_table, 0xff, sizeof(value_t) * domain::extent));
		}
		managed_memory->release(this->Instance<block_partition_>::count_virtual, this->index_table_virtual);
	}
	void reset_table(cudaStream_t stream) {
		this->index_table = this->index_table_virtual;
		if(managed_memory->get_memory_type(this->index_table) == MemoryType::HOST){
			managed_memory->managed_memory_type::acquire<MemoryType::HOST>(reinterpret_cast<void**>(&this->index_table));
			memset(this->index_table, 0xff, sizeof(value_t) * domain::extent);
		}else{
			managed_memory->managed_memory_type::acquire<MemoryType::DEVICE>(reinterpret_cast<void**>(&this->index_table));
			check_cuda_errors(cudaMemset(this->index_table, 0xff, sizeof(value_t) * domain::extent));
		}
		managed_memory->release(this->index_table_virtual);
	}
	void copy_to(Partition& other, std::size_t block_count, cudaStream_t stream) {
		halo_base_t::copy_to(other, block_count, stream);
		
		this->index_table = this->index_table_virtual;
		other.index_table = other.index_table_virtual;
		managed_memory->acquire_any(
			  reinterpret_cast<void**>(&this->index_table)
			, reinterpret_cast<void**>(&other.index_table)
		);
		check_cuda_errors(cudaMemcpyAsync(other.index_table, this->index_table, sizeof(value_t) * domain::extent, cudaMemcpyDefault, stream));
		managed_memory->release(this->index_table_virtual, other.index_table_virtual);
	}
	//FIXME: passing kjey_t here might cause problems because cuda is buggy
	__forceinline__ __device__ value_t insert(key_t key) noexcept {
		value_t tag = atomicCAS(&this->index(key), sentinel_v, 0);
		if(tag == sentinel_v) {
			value_t idx			   = atomicAdd(this->Instance<block_partition_>::count, 1);
			this->index(key)	   = idx;
			this->active_keys[idx] = key;///< created a record
			return idx;
		}
		return -1;
	}
	//FIXME: passing kjey_t here might cause problems because cuda is buggy
	__forceinline__ __device__ value_t query(key_t key) const noexcept {
		return this->index(key);
	}
	__forceinline__ __device__ void reinsert(value_t index) {
		this->index(this->active_keys[index]) = index;
	}
};
//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

}// namespace mn

#endif