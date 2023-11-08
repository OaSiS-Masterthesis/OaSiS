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
	
	int* halo_count_virtual;
	char* halo_marks_virtual;
	int* overlap_marks_virtual;
	ivec3* halo_blocks_virtual;

	template<typename Allocator>
	HaloPartition(Allocator allocator, managed_memory_type* managed_memory, int max_block_count)
		: h_count(0) 
		, managed_memory(managed_memory){
		halo_count_virtual	  = static_cast<int*>(allocator.allocate(sizeof(char) * max_block_count));
		halo_marks_virtual	  = static_cast<char*>(allocator.allocate(sizeof(char) * max_block_count));
		overlap_marks_virtual = static_cast<int*>(allocator.allocate(sizeof(int) * max_block_count));
		halo_blocks_virtual	  = static_cast<ivec3*>(allocator.allocate(sizeof(ivec3) * max_block_count));
	}

	void copy_to(HaloPartition& other, std::size_t block_count, cudaStream_t stream) {
		other.h_count = h_count;
		
		bool halo_marks_is_locked = managed_memory->is_locked(halo_marks_virtual);
		bool other_halo_marks_is_locked = managed_memory->is_locked(other.halo_marks_virtual);
		bool overlap_marks_is_locked = managed_memory->is_locked(overlap_marks_virtual);
		bool other_overlap_marks_is_locked = managed_memory->is_locked(other.overlap_marks_virtual);
		bool halo_blocks_is_locked = managed_memory->is_locked(halo_blocks_virtual);
		bool other_halo_blocks_is_locked = managed_memory->is_locked(other.halo_blocks_virtual);
		
		this->halo_marks = halo_marks_virtual;
		other.halo_marks = other.halo_marks_virtual;
		this->overlap_marks = overlap_marks_virtual;
		other.overlap_marks = other.overlap_marks_virtual;
		this->halo_blocks = halo_blocks_virtual;
		other.halo_blocks = other.halo_blocks_virtual;
		managed_memory->acquire_any(
			  reinterpret_cast<void**>(&this->halo_marks)
			, reinterpret_cast<void**>(&other.halo_marks)
			, reinterpret_cast<void**>(&this->overlap_marks)
			, reinterpret_cast<void**>(&other.overlap_marks)
			, reinterpret_cast<void**>(&this->halo_blocks)
			, reinterpret_cast<void**>(&other.halo_blocks)
		);
		check_cuda_errors(cudaMemcpyAsync(other.halo_marks, this->halo_marks, sizeof(char) * block_count, cudaMemcpyDefault, stream));
		check_cuda_errors(cudaMemcpyAsync(other.overlap_marks, this->overlap_marks, sizeof(int) * block_count, cudaMemcpyDefault, stream));
		check_cuda_errors(cudaMemcpyAsync(other.halo_blocks, this->halo_blocks, sizeof(ivec3) * block_count, cudaMemcpyDefault, stream));
		managed_memory->release(
			  (halo_marks_is_locked ? nullptr : halo_marks_virtual)
			, (other_halo_marks_is_locked ? nullptr : other.halo_marks_virtual)
			, (overlap_marks_is_locked ? nullptr : overlap_marks_virtual)
			, (other_overlap_marks_is_locked ? nullptr : other.overlap_marks_virtual)
			, (halo_blocks_is_locked ? nullptr : halo_blocks_virtual)
			, (other_halo_blocks_is_locked ? nullptr : other.halo_blocks_virtual)
		);
	}

	template<typename Allocator>
	void resize_partition(Allocator allocator, std::size_t prev_capacity, std::size_t capacity) {
		allocator.deallocate(halo_marks_virtual, sizeof(char) * prev_capacity);
		allocator.deallocate(overlap_marks_virtual, sizeof(int) * prev_capacity);
		allocator.deallocate(halo_blocks_virtual, sizeof(ivec3) * prev_capacity);
		halo_marks_virtual	  = static_cast<char*>(allocator.allocate(sizeof(char) * capacity));
		overlap_marks_virtual = static_cast<int*>(allocator.allocate(sizeof(int) * capacity));
		halo_blocks_virtual	  = static_cast<ivec3*>(allocator.allocate(sizeof(ivec3) * capacity));
	}

	void reset_halo_count(cudaStream_t stream) {
		bool halo_count_is_locked = managed_memory->is_locked(halo_count_virtual);
		
		this->halo_count = halo_count_virtual;
		if(managed_memory->get_memory_type(this->halo_count) == MemoryType::HOST){
			managed_memory->managed_memory_type::acquire<MemoryType::HOST>(reinterpret_cast<void**>(&this->halo_count));
			memset(this->halo_count, 0, sizeof(int));
		}else{
			managed_memory->managed_memory_type::acquire<MemoryType::DEVICE>(reinterpret_cast<void**>(&this->halo_count));
			check_cuda_errors(cudaMemsetAsync(this->halo_count, 0, sizeof(int), stream));
		}
		managed_memory->release(
			(halo_count_is_locked ? nullptr : halo_count_virtual)
		);
	}

	void reset_overlap_marks(uint32_t neighbor_block_count, cudaStream_t stream) {
		bool overlap_marks_is_locked = managed_memory->is_locked(overlap_marks_virtual);
		
		this->overlap_marks = overlap_marks_virtual;
		if(managed_memory->get_memory_type(this->overlap_marks) == MemoryType::HOST){
			managed_memory->managed_memory_type::acquire<MemoryType::HOST>(reinterpret_cast<void**>(&this->overlap_marks));
			memset(this->overlap_marks, 0, sizeof(int) * neighbor_block_count);
		}else{
			managed_memory->managed_memory_type::acquire<MemoryType::DEVICE>(reinterpret_cast<void**>(&this->overlap_marks));
			check_cuda_errors(cudaMemsetAsync(this->overlap_marks, 0, sizeof(int) * neighbor_block_count, stream));
		}
		managed_memory->release(
			(overlap_marks_is_locked ? nullptr : overlap_marks_virtual)
		);
	}

	void retrieve_halo_count(cudaStream_t stream) {
		bool halo_count_is_locked = managed_memory->is_locked(halo_count_virtual);
		
		this->halo_count = halo_count_virtual;
		managed_memory->acquire_any(reinterpret_cast<void**>(&this->halo_count));
		check_cuda_errors(cudaMemcpyAsync(&h_count, this->halo_count, sizeof(int), cudaMemcpyDefault, stream));
		managed_memory->release(
			(halo_count_is_locked ? nullptr : halo_count_virtual)
		);
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
		bool count_is_locked = managed_memory->is_locked(this->Instance<block_partition_>::count_virtual);
		bool index_table_is_locked = managed_memory->is_locked(this->index_table_virtual);
		
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
		managed_memory->release(
			  (count_is_locked ? nullptr : this->Instance<block_partition_>::count_virtual)
			, (index_table_is_locked ? nullptr : this->index_table_virtual)
		);
	}
	void reset_table(cudaStream_t stream) {
		bool index_table_is_locked = managed_memory->is_locked(this->index_table_virtual);
		
		this->index_table = this->index_table_virtual;
		if(managed_memory->get_memory_type(this->index_table) == MemoryType::HOST){
			managed_memory->managed_memory_type::acquire<MemoryType::HOST>(reinterpret_cast<void**>(&this->index_table));
			memset(this->index_table, 0xff, sizeof(value_t) * domain::extent);
		}else{
			managed_memory->managed_memory_type::acquire<MemoryType::DEVICE>(reinterpret_cast<void**>(&this->index_table));
			check_cuda_errors(cudaMemset(this->index_table, 0xff, sizeof(value_t) * domain::extent));
		}
		managed_memory->release(
			(index_table_is_locked ? nullptr : this->index_table_virtual)
		);
	}
	void copy_to(Partition& other, std::size_t block_count, cudaStream_t stream) {
		halo_base_t::copy_to(other, block_count, stream);
		
		bool index_table_is_locked = managed_memory->is_locked(this->index_table_virtual);
		bool other_index_table_is_locked = managed_memory->is_locked(other.index_table_virtual);
		
		this->index_table = this->index_table_virtual;
		other.index_table = other.index_table_virtual;
		managed_memory->acquire_any(
			  reinterpret_cast<void**>(&this->index_table)
			, reinterpret_cast<void**>(&other.index_table)
		);
		check_cuda_errors(cudaMemcpyAsync(other.index_table, this->index_table, sizeof(value_t) * domain::extent, cudaMemcpyDefault, stream));
		managed_memory->release(
			  (index_table_is_locked ? nullptr : this->index_table_virtual)
			, (other_index_table_is_locked ? nullptr : other.index_table_virtual)
		);
	}
	//FIXME: passing key_t here might cause problems because cuda is buggy
	__forceinline__ __device__ value_t insert(key_t key) noexcept {
		//FIXME: Put to another place. Maybe allow negative entries
		if(key[0] < 0 || key[1] < 0 || key[2] < 0){
			return -1;
		}
		value_t tag = atomicCAS(&this->index(key), sentinel_v, 0);
		if(tag == sentinel_v) {
			value_t idx			   = atomicAdd(this->Instance<block_partition_>::count, 1);
			this->index(key)	   = idx;
			this->active_keys[idx] = key;///< created a record
			return idx;
		}
		return -1;
	}
	//FIXME: passing key_t here might cause problems because cuda is buggy
	__forceinline__ __device__ value_t query(key_t key) const noexcept {
		//FIXME: Put to another place. Maybe allow negative entries
		if(key[0] < 0 || key[1] < 0 || key[2] < 0){
			return -1;
		}
		return this->index(key);
	}
	__forceinline__ __device__ void reinsert(value_t index) {
		this->index(this->active_keys[index]) = index;
	}
};
//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

}// namespace mn

#endif