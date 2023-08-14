#include <MnSystem/Cuda/Cuda.h>

#include <map>
#include <set>
#include <stdexcept>
#include <cstdlib>

namespace mn{
	
__global__ void memmove(char* dst, char* src, size_t size){
	for(size_t offset = threadIdx.x; offset < size; offset += blockDim.x){
		const char data = *(src + offset);
		__syncthreads();
		*(dst + offset) = data;
	}
}
	
class SpaceManager {
private:
	struct Space{
		Space(const size_t size, const size_t alignment, const bool is_free)
		: size(size), alignment(alignment), is_free(is_free)
		{}
		size_t size;
		size_t alignment;
		bool is_free;
	};

	void* memory;
	size_t total_size;
	
	std::map<uintptr_t, Space> spaces;
	
	void merge(const std::map<uintptr_t, Space>::iterator iter){
		std::map<uintptr_t, Space>::iterator next = std::next(iter);
		//Merge if possible
		if(next != spaces.end()){
			
			if(next->second.is_free && next->first == iter->first + iter->second.size){
				iter->second.size += next->second.size;
				spaces.erase(next);
			}
		}
		if(iter != spaces.begin()){
			std::map<uintptr_t, Space>::iterator prev = std::prev(iter);
			if(prev->second.is_free && iter->first == prev->first + prev->second.size){
				prev->second.size += iter->second.size;
				spaces.erase(iter);
			}
		}
	}
	
	constexpr uintptr_t get_aligned_address(const uintptr_t address, const size_t alignment){
		const size_t offset = (alignment - address % alignment);
		return address + offset - ((offset / alignment) * alignment);
	}
	
public:	

	SpaceManager()
	: SpaceManager(nullptr, 0) {}

	SpaceManager(void* memory, size_t size)
	: memory(memory)
	  , total_size(size){
		spaces.emplace(reinterpret_cast<uintptr_t>(memory), Space(size, 1, true));
	}
	
	void* allocate(std::size_t bytes, std::size_t alignment) {
		void* ret = nullptr;
		
		for(std::map<uintptr_t, Space>::iterator iter = spaces.begin(); iter != spaces.end(); ++iter){
			if(iter->second.is_free){
				const uintptr_t next_aligned_address = get_aligned_address(iter->first, alignment);
				const size_t aligned_size = iter->second.size - (next_aligned_address - iter->first);
				if(aligned_size >= bytes){
					ret = reinterpret_cast<void*>(next_aligned_address);
					const size_t new_size = bytes + (next_aligned_address - iter->first);
					if(new_size < iter->second.size){
						spaces.emplace(iter->first + new_size, Space(iter->second.size - new_size, 1, true));
					}
					
					iter->second.size = new_size;
					iter->second.alignment = alignment;
					iter->second.is_free = false;
					break;
				}
			}
		}
		
		return ret;
	}
	
	void deallocate(void* p, std::size_t size) {
		if(p != nullptr){
			bool found = false;
			for(std::map<uintptr_t, Space>::iterator iter = spaces.begin(); iter != spaces.end(); ++iter){
				const uintptr_t next_aligned_address = get_aligned_address(iter->first, iter->second.alignment);
				if(next_aligned_address == reinterpret_cast<uintptr_t>(p)){
					found = true;
					iter->second.alignment = 1;
					iter->second.is_free = true;
					merge(iter);
					break;
				}
			}
			if(!found){
				throw std::invalid_argument("Pointer was not allocated using this allocator");
			}
		}
	}
	
	//Compresses memory, providing as much connected free space as possible. Not moving locked locations
	void defragmentate(const std::set<void*>& locked_locations, std::map<uintptr_t, std::pair<uintptr_t, size_t>>& moves, std::map<uintptr_t, std::pair<uintptr_t, size_t>>& space_changes){
		/*
		bool move_happened = true;
		while(move_happened){
			move_happened = false;
			for(std::map<uintptr_t, Space>::iterator iter_i = spaces.begin(); iter_i != spaces.end(); ++iter_i){
				if(locked_locations.find(reinterpret_cast<void*>(iter_i->first)) == locked_locations.end()){
					if(iter_i->second.is_free){
						//Find best fit
						size_t smallest_diff = std::numeric_limits<size_t>::max();;
						std::map<uintptr_t, Space>::iterator iter_best_fit;
						bool found = false;
						
						//If next is movable, move it into the empty space
						if(std::next(iter_i) != spaces.end() && locked_locations.find(reinterpret_cast<void*>(std::next(iter_i)->first)) == locked_locations.end()){
							iter_best_fit = std::next(iter_i);
							found = true;
						}else{
							for(std::map<uintptr_t, Space>::iterator iter_j = std::next(iter_i); iter_j != spaces.end(); ++iter_j){
								if(locked_locations.find(reinterpret_cast<void*>(iter_j->first)) == locked_locations.end()){
									const uintptr_t next_aligned_address_current_free = get_aligned_address(iter_i->first, iter_j->second.alignment);
									const uintptr_t next_aligned_address_current_full = get_aligned_address(iter_j->first, iter_j->second.alignment);
									const size_t aligned_size_full = iter_j->second.size - (next_aligned_address_current_full - iter_j->first) + (next_aligned_address_current_free - iter_i->first);
									if(!iter_j->second.is_free && aligned_size_full <= iter_i->second.size){
										const size_t size_diff = iter_i->second.size - aligned_size_full;
										if(size_diff < smallest_diff){
											smallest_diff = size_diff;
											iter_best_fit = iter_j;
											found = true;
										}
									}
								}
							}
						}
						if(found){
							//Move
							const uintptr_t next_aligned_address_current_free = get_aligned_address(iter_i->first, iter_best_fit->second.alignment);
							const uintptr_t next_aligned_address_current_full = get_aligned_address(iter_best_fit->first, iter_best_fit->second.alignment);
							const size_t aligned_size_full = iter_best_fit->second.size - (next_aligned_address_current_full - iter_best_fit->first) + (next_aligned_address_current_free - iter_i->first);
							
							const size_t prev_size_free = iter_i->second.size;
							
							const std::pair<uintptr_t, std::pair<uintptr_t, size_t>> new_move = std::make_pair(next_aligned_address_current_full, std::make_pair(next_aligned_address_current_free, iter_best_fit->second.size));
							
							//Store chane in spaces
							bool replaced_existing = false;
							for(std::map<uintptr_t, std::pair<uintptr_t, size_t>>::iterator iter_space_change = space_changes.begin(); iter_space_change != space_changes.end(); ++iter_space_change){
								if(iter_space_change->second.first == next_aligned_address_current_full){
									iter_space_change->second.first = next_aligned_address_current_free;
									replaced_existing = true;
									break;
								}
							}
							if(!replaced_existing){
								space_changes.emplace(next_aligned_address_current_full, std::make_pair(next_aligned_address_current_free, iter_best_fit->second.size));
							}
							
							
							//Store moves
							
							//Remap overlaping ranges
							{
								bool found_overlap = false;
								std::map<uintptr_t, size_t> mapped_ranges_of_new_move;
								std::map<uintptr_t, std::pair<uintptr_t, size_t>>::iterator iter_move = moves.begin();
								while(iter_move != moves.end()){
									if(iter_move->second.first <= new_move.first && (iter_move->second.first + iter_move->second.second) > new_move.first){//Begin in mapped range
										const uintptr_t diff_start = (new_move.first - iter_move->second.first);
										if(diff_start > 0){
											moves.emplace(std::make_pair(iter_move->first, std::make_pair(iter_move->second.first, diff_start)));
										}
										found_overlap = true;
									}
									if(iter_move->second.first < (new_move.first + new_move.second.second) && (iter_move->second.first + iter_move->second.second) >= (new_move.first + new_move.second.second)){//End in mapped range
										const uintptr_t diff_end = ((new_move.first + new_move.second.second) - iter_move->second.first);
										if(diff_end > 0){
											moves.emplace(std::make_pair(iter_move->first + diff_end, std::make_pair(iter_move->second.first + diff_end, iter_move->second.second - diff_end)));
										}
										found_overlap = true;
									}
									
									
									if(new_move.first <= iter_move->second.first && (new_move.first + new_move.second.second) > iter_move->first){//Mapped begin in range, maybe also end
										const uintptr_t diff_start = (iter_move->second.first - new_move.first);
										moves.emplace(std::make_pair(iter_move->second.first, std::make_pair(new_move.second.first + diff_start, std::min(iter_move->second.second, new_move.second.second - diff_start))));
										mapped_ranges_of_new_move.emplace(std::make_pair(new_move.second.first + diff_start, std::min(iter_move->second.second, new_move.second.second - diff_start)));
										found_overlap = true;
									}else if(new_move.first < (iter_move->second.first + iter_move->second.second) && (new_move.first + new_move.second.second) >= (iter_move->second.first + iter_move->second.second)){//Only mapped end in range
										const uintptr_t diff_end = ((iter_move->second.first + iter_move->second.second) - new_move.first);
										if(diff_end > 0){
											moves.emplace(std::make_pair(new_move.first, std::make_pair(new_move.second.first, new_move.second.second - diff_end)));
											mapped_ranges_of_new_move.emplace(std::make_pair(new_move.second.first, new_move.second.second - diff_end));
										}
										found_overlap = true;
									}
									
									std::map<uintptr_t, std::pair<uintptr_t, size_t>>::iterator iter_prev = iter_move;
									iter_move = std::next(iter_prev);
									if(found_overlap){
										moves.erase(iter_prev);
									}
								}
								//Add non-overlapping parts
								size_t last_end = new_move.first;
								for(const std::pair<uintptr_t, size_t>& range : mapped_ranges_of_new_move){
									const uintptr_t diff_start = last_end - new_move.first;
									const size_t range_size = range.first - last_end;
									if(range_size > 0){
										moves.emplace(std::make_pair(new_move.first + diff_start, std::make_pair(new_move.second.first + diff_start, range_size)));
									}
									last_end = range.first + range.second;
								}
								//Last
								{
									const uintptr_t diff_start = last_end - new_move.first;
									const size_t range_size = (new_move.first + new_move.second.second) - last_end;
									if(range_size > 0){
										moves.emplace(std::make_pair(new_move.first + diff_start, std::make_pair(new_move.second.first + diff_start, range_size)));
									}
								}
							}
							
							//Merge moves
							{
								std::map<uintptr_t, std::pair<uintptr_t, size_t>>::iterator last_move = moves.begin();
								std::map<uintptr_t, std::pair<uintptr_t, size_t>>::iterator iter_move = std::next(moves.begin());
								while(iter_move != moves.end()){
									if(last_move->first + last_move->second.second == iter_move->first && last_move->second.first + last_move->second.second == iter_move->second.first){
										last_move->second.second += iter_move->second.second;
										std::map<uintptr_t, std::pair<uintptr_t, size_t>>::iterator iter_prev = iter_move;
										iter_move = std::next(iter_move);
										moves.erase(iter_prev);
									}else{
										last_move = iter_move;
										iter_move = std::next(iter_move);
									}
								}
							}
							
							
							iter_i->second.size = aligned_size_full;
							iter_i->second.alignment = iter_best_fit->second.alignment;
							iter_i->second.is_free = false;
							
							//Overlap
							if(next_aligned_address_current_free + aligned_size_full > next_aligned_address_current_full){
								if(aligned_size_full < prev_size_free){
									spaces.emplace(iter_i->first + aligned_size_full, Space(prev_size_free - aligned_size_full, 1, true));
									//Cannot be merged. Previous is not free (we just moved a full memory there) and next cannot be free (otherwise it would already have been merged)
								}
								
								iter_best_fit->second.alignment = 1;
								iter_best_fit->second.is_free = true;
								merge(iter_best_fit);
							}else{
								std::pair<std::map<uintptr_t, Space>::iterator, bool> emplacement = spaces.emplace(iter_i->first + aligned_size_full, Space((iter_best_fit->second.size + prev_size_free) - aligned_size_full, 1, true));
								spaces.erase(iter_best_fit);
								merge(emplacement.first);
							}
							move_happened = true;
						}
					}
				}
			}
		}
		*/
	}
	
	//Find free space, also consideringmemory that is currently in use. Returns wheter space was found and what locations to preempt
	bool find_place(const std::set<void*>& locked_locations, std::size_t bytes, std::size_t alignment, std::vector<std::tuple<void*, size_t, size_t>>& preempted_locations){
		if(bytes == 0){
			return true;
		}
		
		size_t accumulated_size = 0;
		std::map<uintptr_t, Space>::iterator iter_first;
		for(std::map<uintptr_t, Space>::iterator iter_i = spaces.begin(); iter_i != spaces.end(); ++iter_i){
			if(locked_locations.find(reinterpret_cast<void*>(iter_i->first)) == locked_locations.end()){
				if(accumulated_size == 0){
					const uintptr_t next_aligned_address = get_aligned_address(iter_i->first, alignment);
					const size_t aligned_size = iter_i->second.size - (next_aligned_address - iter_i->first);
					accumulated_size += aligned_size;
					iter_first = iter_i;
				}else{
					accumulated_size += iter_i->second.size;
				}
			}else{
				accumulated_size = 0;
			}
			
			if(accumulated_size >= bytes){
				for(std::map<uintptr_t, Space>::iterator iter_j = iter_first; iter_j != std::next(iter_i); ++iter_j){
					if(!iter_j->second.is_free){
						uintptr_t next_aligned_address = get_aligned_address(iter_j->first, iter_j->second.alignment);
						void* next_aligned_address_ptr = reinterpret_cast<void*>(next_aligned_address);
						preempted_locations.push_back(std::make_tuple(next_aligned_address_ptr, iter_j->second.size, iter_j->second.alignment));
					}
				}
				return true;
			}
		}
		
		return false;
	}
	
	void print_data(){
		std::cout << "Spaces:" << std::endl;
		std::size_t used_size = 0;
		for(std::map<uintptr_t, Space>::iterator iter = spaces.begin(); iter != spaces.end(); ++iter){
			uintptr_t next_aligned_address = get_aligned_address(iter->first, iter->second.alignment);
			std::cout << reinterpret_cast<void*>(iter->first) << "/" << reinterpret_cast<void*>(next_aligned_address)  << ": (" << iter->second.size << ", " << iter->second.alignment << ", " << iter->second.is_free << ")" << std::endl;
			if(!iter->second.is_free){
				used_size += iter->second.size;
			}
		}
		std::cout << "Memory Usage: " << used_size << "/" << total_size << std::endl;
	}
};

enum MemoryType{
	NONE = 0,
	DEVICE,
	SWAP,
	HOST
};

struct MemoryLocation{
	MemoryLocation()
	: address(nullptr), size(0), alignment(1), memory_type(MemoryType::NONE)
	{}
	
	MemoryLocation(void* address, const size_t size, const size_t alignment, const MemoryType memory_type)
	: address(address), size(size), alignment(alignment), memory_type(memory_type)
	{}
	
	void* address;
	size_t size;
	size_t alignment;
	MemoryType memory_type;
};

template<typename DeviceAllocator, size_t DeviceMemorySize, size_t SwapMemorySize>
class ManagedMemory {
	private:
	
	intptr_t pointers = 1;
	
	int gpuid;
	DeviceAllocator device_allocator;
	
	void* device_memory;
	void* swap_memory;
	
	SpaceManager device_space_manager;
	SpaceManager swap_space_manager;
	
	std::map<uintptr_t, MemoryLocation> memory_locations;
	std::set<void*> locks_device;
	std::set<void*> locks_swap;
	std::set<void*> locks_host;
	
	MemoryLocation preempt(std::size_t bytes, std::size_t alignment){
		void* ret = nullptr;
		MemoryType ret_memory_type;
		
		//Look for free space on device
		ret = device_space_manager.allocate(bytes, alignment);
		ret_memory_type = MemoryType::DEVICE;
		
		//If not found, search in swap space
		if(ret == nullptr){
			//Look for free space in swap memory
			ret = swap_space_manager.allocate(bytes, alignment);
			ret_memory_type = MemoryType::SWAP;
			
			//If not found, move pages from device to swap space or to host so that we have enough memory
			if(ret == nullptr){
				
				//Preempt from device. If not possible, preemt from swap space
				std::vector<std::tuple<void*, size_t, size_t>> removed_allocations_device;
				if(device_space_manager.find_place(locks_device, bytes, alignment, removed_allocations_device)){
					
					//Move memory to host/swap
					for(const std::tuple<void*, size_t, size_t>& removed_allocation : removed_allocations_device){
						
						//Find memory location
						std::map<uintptr_t, MemoryLocation>::iterator removed_memory_location_iter = memory_locations.end();
						for(std::map<uintptr_t, MemoryLocation>::iterator memory_location_iter = memory_locations.begin(); memory_location_iter != memory_locations.end(); ++memory_location_iter){
							if(memory_location_iter->second.address == std::get<0>(removed_allocation)){
								removed_memory_location_iter = memory_location_iter;
								break;
							}
						}
						
						//Swap memory location. First try to preemt from swap, otherwise move directly tp host
						void* new_address_device;
						MemoryType memory_type_device;
						
						std::vector<std::tuple<void*, size_t, size_t>> removed_allocations_swap;
						if(swap_space_manager.find_place(locks_swap, std::get<1>(removed_allocation), std::get<2>(removed_allocation), removed_allocations_swap)){
							
							//Move memory to host
							for(const std::tuple<void*, size_t, size_t>& removed_allocation_swap : removed_allocations_swap){
								//Find memory location
								std::map<uintptr_t, MemoryLocation>::iterator removed_memory_location_swap_iter = memory_locations.end();
								for(std::map<uintptr_t, MemoryLocation>::iterator memory_location_iter = memory_locations.begin(); memory_location_iter != memory_locations.end(); ++memory_location_iter){
									if(memory_location_iter->second.address == std::get<0>(removed_allocation_swap)){
										removed_memory_location_swap_iter = memory_location_iter;
										break;
									}
								}
								
								//Move to host
								void* new_address_swap = std::malloc(std::get<1>(removed_allocation_swap));
							
								if(new_address_swap == nullptr){
									throw std::bad_alloc();
								}else{
									move_memory(removed_memory_location_swap_iter, MemoryLocation(new_address_swap, std::get<1>(removed_allocation_swap), std::get<2>(removed_allocation_swap), MemoryType::HOST));
								}
							}
							
							new_address_device = swap_space_manager.allocate(std::get<1>(removed_allocation), std::get<2>(removed_allocation));
							memory_type_device = MemoryType::SWAP;
							
							assert(new_address_device != nullptr && "Could not allocate mem though we just freed enough");
						}else{
							new_address_device = std::malloc(std::get<1>(removed_allocation));
							memory_type_device = MemoryType::HOST;

							if(new_address_device == nullptr){
								throw std::bad_alloc();
							}
						}
						
						move_memory(removed_memory_location_iter, MemoryLocation(new_address_device, std::get<1>(removed_allocation), std::get<2>(removed_allocation), memory_type_device));
					}
					
					//Get pointer
					ret = device_space_manager.allocate(bytes, alignment);
					ret_memory_type = MemoryType::DEVICE;
					
					assert(ret != nullptr && "Could not allocate mem though we just freed enough");
				}else{
					
					std::vector<std::tuple<void*, size_t, size_t>> removed_allocations_swap;
					if(swap_space_manager.find_place(locks_swap, bytes, alignment, removed_allocations_swap)){
						
						//Move memory to host
						for(const std::tuple<void*, size_t, size_t>& removed_allocation : removed_allocations_swap){
							
							//Find memory location
							std::map<uintptr_t, MemoryLocation>::iterator removed_memory_location_iter = memory_locations.end();
							for(std::map<uintptr_t, MemoryLocation>::iterator memory_location_iter = memory_locations.begin(); memory_location_iter != memory_locations.end(); ++memory_location_iter){
								if(memory_location_iter->second.address == std::get<0>(removed_allocation)){
									removed_memory_location_iter = memory_location_iter;
									break;
								}
							}
							
							//Move to host
							void* new_address = std::malloc(std::get<1>(removed_allocation));
						
							if(new_address == nullptr){
								throw std::bad_alloc();
							}else{
								move_memory(removed_memory_location_iter, MemoryLocation(new_address, std::get<1>(removed_allocation), std::get<2>(removed_allocation), MemoryType::HOST));
							}
						}
						
						//Get pointer
						ret = swap_space_manager.allocate(bytes, alignment);
						ret_memory_type = MemoryType::SWAP;
						
						assert(ret != nullptr && "Could not allocate mem though we just freed enough");
					}//Otherwise no place was found
				}
			}
		}
		
		return MemoryLocation(ret, bytes, alignment, ret_memory_type);
	}
	
	void move_memory(std::map<const uintptr_t, MemoryLocation>::iterator old_mem_location_iter, const MemoryLocation new_mem_location){
		auto& cu_dev = Cuda::ref_cuda_context(gpuid);
		check_cuda_errors(cudaMemcpyAsync(new_mem_location.address, old_mem_location_iter->second.address, old_mem_location_iter->second.size, cudaMemcpyDefault, cu_dev.stream_compute()));
		cu_dev.syncStream<Cuda::StreamIndex::COMPUTE>();
		switch(old_mem_location_iter->second.memory_type){
			case DEVICE:
				device_space_manager.deallocate(old_mem_location_iter->second.address, old_mem_location_iter->second.size);
				break;
			case SWAP:
				swap_space_manager.deallocate(old_mem_location_iter->second.address, old_mem_location_iter->second.size);
				break;
			case HOST:
				free(old_mem_location_iter->second.address);
				break;
			default:
				assert(false && "Reached unreachable state");
		}
		old_mem_location_iter->second = new_mem_location;
	}
	
	public:
		ManagedMemory(DeviceAllocator& device_allocator, int gpuid)
		: device_allocator(device_allocator), gpuid(gpuid)
		{
			device_memory = device_allocator.allocate(DeviceMemorySize);
			check_cuda_errors(cudaMallocHost(&swap_memory, SwapMemorySize));
			
			device_space_manager = SpaceManager(device_memory, DeviceMemorySize);
			swap_space_manager = SpaceManager(swap_memory, SwapMemorySize);
		}
		
		~ManagedMemory(){
			device_allocator.deallocate(device_memory, DeviceMemorySize);
			check_cuda_errors(cudaFreeHost(swap_memory));
			
			//Free host memory
			for(const std::pair<uintptr_t, MemoryLocation>& memory_location : memory_locations){
				if(memory_location.second.memory_type == MemoryType::HOST){
					free(memory_location.second.address);
				}
			}
		}
		
		void defragmentate(){
			auto& cu_dev = Cuda::ref_cuda_context(gpuid);
			
			std::map<uintptr_t, std::pair<uintptr_t, size_t>> space_changes_device;
			std::map<uintptr_t, std::pair<uintptr_t, size_t>> space_changes_swap;
			std::map<uintptr_t, std::pair<uintptr_t, size_t>> moves_device;
			std::map<uintptr_t, std::pair<uintptr_t, size_t>> moves_swap;
			device_space_manager.defragmentate(locks_device, moves_device, space_changes_device);
			swap_space_manager.defragmentate(locks_swap, moves_swap, space_changes_swap);
			
			//TODO:Apply moves
			
			for(const std::pair<uintptr_t, std::pair<uintptr_t, size_t>> & move : moves_device){
				if(move.first != move.second.first){
					//Overlap => use memmove
					if(
						   (move.first < move.second.first && move.first + move.second.second > move.second.first)
						|| (move.second.first < move.second.first && move.second.first + move.second.second > move.first)
					){
						cu_dev.compute_launch({1, 32}, memmove, reinterpret_cast<char*>(move.second.first), reinterpret_cast<char*>(move.first), move.second.second);
					}else{
						check_cuda_errors(cudaMemcpyAsync(reinterpret_cast<void*>(move.second.first), reinterpret_cast<void*>(move.first), move.second.second, cudaMemcpyDefault, cu_dev.stream_compute()));
					}
				}
			}
			
			for(const std::pair<uintptr_t, std::pair<uintptr_t, size_t>> & move : moves_swap){
				if(move.first != move.second.first){
					//Overlap => use memmove
					if(
						   (move.first < move.second.first && move.first + move.second.second > move.second.first)
						|| (move.second.first < move.second.first && move.second.first + move.second.second > move.first)
					){
						cu_dev.compute_launch({1, 32}, memmove, reinterpret_cast<char*>(move.second.first), reinterpret_cast<char*>(move.first), move.second.second);
					}else{
						check_cuda_errors(cudaMemcpyAsync(reinterpret_cast<void*>(move.second.first), reinterpret_cast<void*>(move.first), move.second.second, cudaMemcpyDefault, cu_dev.stream_compute()));
					}
				}
			}
			
			cu_dev.syncStream<Cuda::StreamIndex::COMPUTE>();
			
			//Update locations
			for(std::map<uintptr_t, MemoryLocation>::iterator memory_location_iter = memory_locations.begin(); memory_location_iter != memory_locations.end(); ++memory_location_iter){
				if(memory_location_iter->second.memory_type == MemoryType::DEVICE){
					std::map<uintptr_t, std::pair<uintptr_t, size_t>>::iterator space_change_iter = space_changes_device.find(reinterpret_cast<uintptr_t>(memory_location_iter->second.address));
					if(space_change_iter != space_changes_device.end()){
						memory_location_iter->second.address = reinterpret_cast<void*>(space_change_iter->second.first);
					}
				}else if(memory_location_iter->second.memory_type == MemoryType::SWAP){
					std::map<uintptr_t, std::pair<uintptr_t, size_t>>::iterator space_change_iter = space_changes_swap.find(reinterpret_cast<uintptr_t>(memory_location_iter->second.address));
					if(space_change_iter != space_changes_swap.end()){
						memory_location_iter->second.address = reinterpret_cast<void*>(space_change_iter->second.first);
					}
				}
			}
		}
	
		void* allocate(std::size_t bytes, std::size_t alignment) {//NOLINT(readability-convert-member-functions-to-static) Method is designed to be a non-static class member
			uintptr_t ret = 0;
			if(bytes != 0){
				void* data = device_space_manager.allocate(bytes, alignment);
				if(data == nullptr){
					data = swap_space_manager.allocate(bytes, alignment);
					if(data == nullptr){
						data = std::malloc(bytes);
						
						if(data == nullptr){
							throw std::bad_alloc();
						}else{
							ret = pointers++;
							memory_locations.emplace(ret, MemoryLocation(data, bytes, alignment, MemoryType::HOST));
						}
					}else{
						ret = pointers++;
						memory_locations.emplace(ret, MemoryLocation(data, bytes, alignment, MemoryType::SWAP));
					}
				}else{
					ret = pointers++;
					memory_locations.emplace(ret, MemoryLocation(data, bytes, alignment, MemoryType::DEVICE));
				}
			}
			
			return reinterpret_cast<void*>(ret);
		}

		void deallocate(void* p, std::size_t size) {//NOLINT(readability-convert-member-functions-to-static) Method is designed to be a non-static class member
			(void) size;
			
			if(p != nullptr){
				std::map<uintptr_t, MemoryLocation>::iterator mem_location_iter = memory_locations.find(reinterpret_cast<uintptr_t>(p));
				if(mem_location_iter != memory_locations.end()){
					std::set<void*>::iterator lock_iter;
					switch(mem_location_iter->second.memory_type){
						case DEVICE:
							lock_iter = locks_device.find(p);
							if(lock_iter != locks_device.end()){
								throw std::runtime_error("Data is currently locked");
							}else{
								device_space_manager.deallocate(mem_location_iter->second.address, size);
							}
							break;
						case SWAP:
							lock_iter = locks_swap.find(p);
							if(lock_iter != locks_swap.end()){
								throw std::runtime_error("Data is currently locked");
							}else{
								swap_space_manager.deallocate(mem_location_iter->second.address, size);
							}
							break;
						case HOST:
							lock_iter = locks_host.find(p);
							if(lock_iter != locks_host.end()){
								throw std::runtime_error("Data is currently locked");
							}else{
								free(mem_location_iter->second.address);
							}
							break;
						default:
							assert(false && "Reached unreachable state");
					}
					memory_locations.erase(mem_location_iter);
				}else{
					throw std::invalid_argument("Pointer was not allocated using this allocator");
				}
			}
		}
		
		template<MemoryType memory_type, typename... Args, typename std::enable_if<std::conjunction<std::is_convertible<Args, void**>...>::value>::type* = nullptr>
		void acquire(Args... args){
			std::vector<void**> ptrs = {args...};
			
			bool already_defragmentated = false;
			for(void** ptr : ptrs){
				if(ptr != nullptr && *ptr != nullptr){
					std::map<uintptr_t, MemoryLocation>::iterator mem_location_iter = memory_locations.find(reinterpret_cast<uintptr_t>(*ptr));
					if(mem_location_iter != memory_locations.end()){
						const size_t size = mem_location_iter->second.size;
						const size_t alignment = mem_location_iter->second.alignment;
						
						void* ret;
						MemoryLocation new_mem_location;
						std::set<void*>::iterator lock_iter_device = locks_device.find(*ptr);
						std::set<void*>::iterator lock_iter_swap = locks_swap.find(*ptr);
						std::set<void*>::iterator lock_iter_host = locks_host.find(*ptr);
						switch(memory_type){
							case DEVICE:
								if(lock_iter_host != locks_host.end()){
									throw std::runtime_error("Data is currently locked to another device");
								}else{
									if(mem_location_iter->second.memory_type == MemoryType::HOST){
										//First try to directly move data to device
										ret = device_space_manager.allocate(size, alignment);
										if(ret == nullptr){
											//If that failed try to move to swap memory
											ret = swap_space_manager.allocate(size, alignment);
											
											if(ret == nullptr){
												//Defragmentate memory
												if(!already_defragmentated){
													this->defragmentate();
													already_defragmentated = true;
												}
												
												//If that failed, try to preemt pages
												new_mem_location = preempt(size, alignment);
												ret = new_mem_location.address;
												
												if(ret == nullptr){
													throw std::runtime_error("Out of memory.");
												}
											}else{
												new_mem_location = MemoryLocation(ret, size, alignment, MemoryType::SWAP);
											}
										}else{
											new_mem_location = MemoryLocation(ret, size, alignment, MemoryType::DEVICE);
											
										}
										move_memory(mem_location_iter, new_mem_location);
									}else{
										//TODO: Maybe try to copy data from device or swap to host?
										new_mem_location = mem_location_iter->second;
										ret = mem_location_iter->second.address;
									}
								}
								break;
							case SWAP:
								if(false){
									throw std::runtime_error("Data is currently locked to another device");
								}else{
									ret = mem_location_iter->second.address;
									locks_swap.insert(*ptr);
								}
								break;
							case HOST:
								if(lock_iter_device != locks_device.end()){
									throw std::runtime_error("Data is currently locked to another device");
								}else{
									if(mem_location_iter->second.memory_type == MemoryType::DEVICE){
										//First try to move data to swap memory
										void* ret = swap_space_manager.allocate(size, alignment);
											
										if(ret == nullptr){
											//If that failed try to move directly to host
											ret = std::malloc(size);
							
											if(ret == nullptr){
												throw std::bad_alloc();
											}else{
												new_mem_location = MemoryLocation(ret, size, alignment, MemoryType::HOST);
											}
										}else{
											new_mem_location = MemoryLocation(ret, size, alignment, MemoryType::SWAP);
										}
										move_memory(mem_location_iter, new_mem_location);
									}else{
										new_mem_location = mem_location_iter->second;
										ret = mem_location_iter->second.address;
									}
								}
								break;
							default:
								assert(false && "Reached unreachable state");
						}
						switch(new_mem_location.memory_type){
							case DEVICE:
								locks_device.insert(*ptr);
								break;
							case SWAP:
								locks_swap.insert(*ptr);
								break;
							case HOST:
								locks_host.insert(*ptr);
								break;
							default:
								assert(false && "Reached unreachable state");
						}
						*ptr = ret;
					}else{
						throw std::invalid_argument("Pointer was not allocated using this allocator");
					}
				}
			}
		}
		
		template<typename... Args, typename std::enable_if<std::conjunction<std::is_convertible<Args, void*>...>::value>::type* = nullptr>
		void release(Args... args){
			std::vector<void*> ptrs = {args...};
			
			for(void* ptr : ptrs){
				if(ptr != nullptr){
					std::map<uintptr_t, MemoryLocation>::iterator mem_location_iter = memory_locations.find(reinterpret_cast<uintptr_t>(ptr));
					if(mem_location_iter != memory_locations.end()){
						std::set<void*>::iterator lock_iter_device = locks_device.find(ptr);
						std::set<void*>::iterator lock_iter_swap = locks_swap.find(ptr);
						std::set<void*>::iterator lock_iter_host = locks_host.find(ptr);
						
						if(lock_iter_device != locks_device.end()){
							locks_device.erase(lock_iter_device);
						}
						if(lock_iter_swap != locks_swap.end()){
							locks_swap.erase(lock_iter_swap);
						}
						if(lock_iter_host != locks_host.end()){
							locks_host.erase(lock_iter_host);
						}

					}
				}
			}
		}
	
		void print_data(){
			std::cout << "Locations:" << std::endl;
			for(const std::pair<uintptr_t, MemoryLocation>& memory_location : memory_locations){
				std::cout << reinterpret_cast<void*>(memory_location.first) << ": (" << memory_location.second.address << ", " << memory_location.second.size << ", " << memory_location.second.alignment << ", " << memory_location.second.memory_type << ")" << std::endl;
			}
			std::cout << "Locks(Device):" << std::endl;
			for(const void* lock : locks_device){
				std::cout << lock << std::endl;
			}
			std::cout << "Locks(Swap):" << std::endl;
			for(const void* lock : locks_swap){
				std::cout << lock << std::endl;
			}
			std::cout << "Locks(Host):" << std::endl;
			for(const void* lock : locks_host){
				std::cout << lock << std::endl;
			}
			device_space_manager.print_data();
			swap_space_manager.print_data();
		}
};

}