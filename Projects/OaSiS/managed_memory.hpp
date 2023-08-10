

namespace mn{
	
class SpaceManager {
private:
	struct Space{
		size_t size;
		size_t alignment;
		bool is_free;
	};

	void* memory;
	size_t total_size;
	
	std::map<uintptr_t, Space> spaces;
	
public:	

	SpaceManager(void* memory, size_t size)
	: memory(memory)
	  , total_size(size){
		spaces.emplace(reinterpret_cast<uintptr_t>(memory), {size, 0, true});
	}
	
	void* allocate(std::size_t bytes, std::size_t alignment) {
		void* ret = nullptr;
		
		for(std::pair<uintptr_t, Space>& space_entry : spaces){
			if(space_entry.second.is_free){
				const uintptr_t next_aligned_address = space_entry.first + (alignment - space_entry.first % alignment);
				const size_t aligned_size = space_entry.second.size - (next_aligned_address - space_entry.first);
				if(aligned_size > bytes){
					ret = reinterpret_cast<void*>(next_aligned_address);
					const size_t new_size = bytes + (next_aligned_address - space_entry.first);
					if(new_size < space_entry.second.size){
						spaces.emplace(space_entry.first + new_size, {space_entry.second.size - new_size, 0, true});
					}
					
					space_entry.second.size = new_size;
					space_entry.second.alignment = alignment;
					space_entry.second.is_free = false;
					break;
				}
			}
		}
		
		return ret;
	}
	
	void deallocate(void* p, std::size_t size) {
		if(p != nullptr){
			bool found = false;
			for(size_t i = 0; i < spaces.count(); ++i){
				std::pair<uintptr_t, Space>& space_entry = spaces[i];
				const uintptr_t next_aligned_address = space_entry.first + (space_entry.second.alignment - space_entry.first % space_entry.second.alignment);
				if(next_aligned_address == reinterpret_cast<uintptr_t>(p) && space_entry.first){
					space_entry.second.alignment = 0;
					space_entry.second.is_free = true;
					//Merge if possible
					if((i + 1) < spaces.count()){
						if(spaces[i+1].second.is_free && spaces[i+1].first == space_entry.first + space_entry.second.size){
							space_entry.second.size += spaces[i+1].second.size;
							spaces.erase(i+1);
						}
					}
					if(i > 0){
						if(spaces[i-1].second.is_free && space_entry.first == spaces[i-1].first + spaces[i-1].second.size){
							spaces[i-1].second.size += space_entry.second.size;
							spaces.erase(i);
						}
					}
					break;
				}
			}
			if(!found){
				throw std::invalid_argument("Pointer was not allocated using this allocator");
			}
		}
	}
	
	//Compresses memory, providing as much connected free space as possible. Not moving locked locations
	void compress(const std::vector<void*>& locked_locations){
		
	}
	
	//Find free space, also consideringmemory that is currently in use. Returns wheter space was found and what locations to preempt
	bool find_place(const std::set<void*>& locked_locations, std::size_t bytes, std::size_t alignment, std::vector<std::tuple<void*, size_t, size_t>>& preempted_locations){
		if(size == 0){
			return true;
		}
		
		size_t accumulates_size;
		size_t first_index;
		for(size_t i = 0; i < spaces.count(); ++i){
			std::pair<uintptr_t, Space>& space_entry = spaces[i];
			
			if(locked_locations.find(reinterpret_cast<void*>(space_entry.first) == locked_locations.end()){
				if(accumulates_size == 0){
					const uintptr_t next_aligned_address = space_entry.first + (alignment - space_entry.first % alignment);
					const size_t aligned_size = space_entry.second.size - (next_aligned_address - space_entry.first);
					accumulates_size += aligned_size;
					first_index = i;
				}else{
					accumulates_size += space_entry.second.size;
				}
			}else{
				accumulates_size = 0;
			}
			
			if(accumulates_size >= bytes){
				for(size_t j = first_index; i <= i; ++j){
					const uintptr_t next_aligned_address = spaces[j].first + (spaces[j].second.alignment - spaces[j].first % spaces[j].second.alignment);
					preempted_locations.emplace(reinterpret_cast<void*>(next_aligned_address), spaces[j].second.size, spaces[j].second.alignment);
				}
				return true;
			}
		}
		
		return false;
	}
	
	//Allocate space, freeing given memory locations beforehand if possible
	void* override_place(std::size_t bytes, std::size_t alignment, const std::vector<std::tuple<void*, size_t, size_t>>& preempted_locations){
		for(const std::tuple<void*, size_t, size_t>& preempted_location : preempted_locations){
			deallocate(std::get<0>(preempted_location), std::get<1>(preempted_location));
		}
		
		return allocate(bytes, alignment);
	}
};

template<typename DeviceAllocator, size_t DeviceMemorySize, size_t SwapMemorySize>
class ManagedMemory {
	private:
	
	enum MemoryType{
		DEVICE = 0,
		SWAP,
		HOST
	};
	
	intptr_t pointers = 1;
	
	DeviceAllocator device_allocator;
	
	void* device_memory;
	void* swap_memory;
	
	SpaceManager device_space_manager;
	SpaceManager swap_space_manager;
	
	std::map<uintptr_t, std::pair<void*, MemoryType>> memory_locations;
	std::map<uintptr_t, MemoryType> locks;
	
	void* preempt(std::size_t size, std::size_t alignment){
		void* ret = nullptr;
		
		//Compress memory
		std::vector<void*> locks_device;
		std::vector<void*> locks_swap;
		for(std::pair<uintptr_t, MemoryType>& lock : locks){
			std::map<uintptr_t, std::pair<void*, MemoryType>>::iterator memory_location_iter = memory_locations.find(lock.first);
			if(memory_location_iter->second.second == MemoryType::DEVICE){
				locks_device.push_back(lock_iter->second.first);
			}else if(memory_location_iter->second.second == MemoryType::SWAP){
				locks_swap.push_back(lock_iter->second.first);
			}
		}
		device_space_manager.compress(locks_device);
		swap_space_manager.compress(locks_swap);
		
		//Look for free space
		ret = device_space_manager.allocate(bytes, alignment);
		
		//If not found, search in swap space
		if(ret == nullptr){
			//Look for free space
			ret = swap_space_manager.allocate(bytes, alignment);
			
			//If not found, move pages from swap space too host so that we have enough memory
			if(ret == nullptr){
				std::vector<std::tuple<void*, size_t, size_t>> removed_removed_allocations;
				if(swap_space_manager.find_place(locks_swap, bytes, alignment, removed_removed_allocations)){
					//Move memory to host
					for(const std::tuple<void*, size_t, size_t>& removed_allocation : removed_removed_allocations){
						std::pair<uintptr_t, std::pair<void*, MemoryType>> removed_memory_location;
						for(std::pair<uintptr_t, std::pair<void*, MemoryType>>& ){
							if(memory_location.second.first == std::get<0>(removed_allocation)){
								removed_memory_location = memory_location;
								break;
							}
						}
						void* new_address = aligned_alloc(std::get<2>(removed_allocation), std::get<1>(removed_allocation));
					
						if(new_address == nullptr){
							throw std::bad_alloc("Failed to allocate memory");
						}else{
							move_memory(removed_memory_location, std::make_pair<void*, MemoryType>(new_address, MemoryType::HOST));
						}
					}
					ret = swap_space_manager.override_place(bytes, alignment, removed_removed_allocations);
					
					//Preempt pages from device memory if possible
					removed_removed_allocations.clear();
					const void* swap_memory_location = ret;
					if(device_space_manager.find_place(locks_swap, bytes, alignment, removed_removed_allocations)){
						//Move memory to swap
						for(const std::tuple<void*, size_t, size_t>& removed_allocation : removed_removed_allocations){
							std::pair<uintptr_t, std::pair<void*, MemoryType>> removed_memory_location;
							for(std::pair<uintptr_t, std::pair<void*, MemoryType>>& ){
								if(memory_location.second.first == std::get<0>(removed_allocation)){
									removed_memory_location = memory_location;
									break;
								}
							}
							void* new_address = swap_space_manager(std::get<1>(removed_allocation), std::get<2>(removed_allocation));
						
							if(new_address == nullptr){
								throw std::bad_alloc("Failed to allocate memory");
							}else{
								move_memory(removed_memory_location, std::make_pair<void*, MemoryType>(new_address, MemoryType::SWAP));
							}
						}
						ret = device_space_manager.override_place(bytes, alignment, removed_removed_allocations);
					}
					//Return the address for the data
				}else{//Maybe we have a place in device memory although we did not have a place in swap memory
					if(device_space_manager.find_place(locks_swap, bytes, alignment, removed_removed_allocations)){
						//Move memory to host
						for(const std::tuple<void*, size_t, size_t>& removed_allocation : removed_removed_allocations){
							std::pair<uintptr_t, std::pair<void*, MemoryType>> removed_memory_location;
							for(std::pair<uintptr_t, std::pair<void*, MemoryType>>& ){
								if(memory_location.second.first == std::get<0>(removed_allocation)){
									removed_memory_location = memory_location;
									break;
								}
							}
							void* new_address = aligned_alloc(std::get<2>(removed_allocation), std::get<1>(removed_allocation));
						
							if(new_address == nullptr){
								throw std::bad_alloc("Failed to allocate memory");
							}else{
								move_memory(removed_memory_location, std::make_pair<void*, MemoryType>(new_address, MemoryType::HOST));
							}
						}
						ret = device_space_manager.override_place(bytes, alignment, removed_removed_allocations);
					}//Otherwise no place was found
				}
			}
		}
		return ret;
	}
	
	void move_memory(std::pair<uintptr_t, std::pair<void*, MemoryType>>::iterator mem_location_iter, const size_t size, const std::pair<void*, MemoryType> new_mem_location){
		memCpy();
		switch(mem_location_iter->second.second){
			case DEVICE:
				device_space_manager.deallocate(mem_location_iter->second.first, size);
				break;
			case SWAP:
				swap_space_manager.deallocate(mem_location_iter->second.first, size);
				break;
			case HOST:
				free(mem_location_iter->second.first);
				break;
			default:
				assert(false && "Reached unreachable state");
		}
		mem_location_iter->second = new_mem_location;
	}
	
	public:
		ManagedMemory(DeviceAllocator& device_allocator)
		: device_allocator(device_allocator)
		{
			device_memory = device_allocator.allocate(DeviceMemorySize);
			check_cuda_errors(cudaMallocHost(&swap_memory, SwapMemorySize);
			
			device_space_manager = SpaceManager(device_memory, DeviceMemorySize);
			swap_space_manager = SpaceManager(swap_memory, SwapMemorySize);
		}
		
		~ManagedMemory(){
			device_allocator.deallocate(device_memory, DeviceMemorySize);
			check_cuda_errors(cudaFreeHost(swap_memory);
			
			//Free host memory
			for(const std::pair<uintptr_t, std::pair<void*, MemoryType>>& memory_location : memory_locations){
				if(memory_location.second.second == MemoryType::HOST){
					free(memory_location->second.first);
				}
			}
		}
		
		void compress(){
			std::vector<void*> locks_device;
			std::vector<void*> locks_swap;
			for(std::pair<uintptr_t, MemoryType>& lock : locks){
				std::map<uintptr_t, std::pair<void*, MemoryType>>::iterator memory_location_iter = memory_locations.find(lock.first);
				if(memory_location_iter->second.second == MemoryType::DEVICE){
					locks_device.push_back(lock_iter->second.first);
				}else if(memory_location_iter->second.second == MemoryType::SWAP){
					locks_swap.push_back(lock_iter->second.first);
				}
			}
			device_space_manager.compress(locks_device);
			swap_space_manager.compress(locks_swap);
		}
	
		void* allocate(std::size_t bytes, std::size_t alignment) {//NOLINT(readability-convert-member-functions-to-static) Method is designed to be a non-static class member
			void* ret = device_space_manager.allocate(bytes, alignment);
			if(ret == nullptr){
				ret = swap_space_manager.allocate(bytes, alignment);
				if(ret == nullptr){
					ret = aligned_alloc(alignment, bytes);
					
					if(ret == nullptr){
						throw std::bad_alloc("Failed to allocate memory");
					}else{
						memory_locations.emplace(pointers++, std::make_pair<void*, MemoryType>(ret, MemoryType::HOST));
					}
				}else{
					memory_locations.emplace(pointers++, std::make_pair<void*, MemoryType>(ret, MemoryType::SWAP));
				}
			}else{
				memory_locations.emplace(pointers++, std::make_pair<void*, MemoryType>(ret, MemoryType::DEVICE));
			}
			
			return ret;
		}

		void deallocate(void* p, std::size_t size) {//NOLINT(readability-convert-member-functions-to-static) Method is designed to be a non-static class member
			(void) size;
			
			if(p != nullptr){
				std::map<uintptr_t, std::pair<void*, MemoryType>>::iterator mem_location_iter = memory_locations.find(reinterpret_cast<uintptr_t>(p));
				std::map<uintptr_t, MemoryType>::iterator lock_iter = locks.find(reinterpret_cast<uintptr_t>(p));
				if(lock_iter != locks.end() && lock_iter->second != memory_type){
					throw std::runtime_error("Data is currently locked to another device");
				}else if(mem_location_iter != memory_locations.end()){
					switch(mem_location_iter->second.second){
						case DEVICE:
							device_space_manager.deallocate(mem_location_iter->second.first, size);
							break;
						case SWAP:
							swap_space_manager.deallocate(mem_location_iter->second.first, size);
							break;
						case HOST:
							free(mem_location_iter->second.first);
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
		
		void* acquire(void* p, std::size_t size, std::size_t alignment, MemoryType memory_type){
			if(p != nullptr){
				std::map<uintptr_t, std::pair<void*, MemoryType>>::iterator mem_location_iter = memory_locations.find(reinterpret_cast<uintptr_t>(p));
				std::map<uintptr_t, MemoryType>::iterator lock_iter = locks.find(reinterpret_cast<uintptr_t>(p));
				if(lock_iter != locks.end() && lock_iter->second != memory_type){
					throw std::runtime_error("Data is currently locked to another device");
				}else if(mem_location_iter != memory_locations.end()){
					void* ret;
					switch(memory_type){
						case DEVICE:
							if(mem_location_iter->second.second == MemoryType::HOST){
								//First try to directly move data to device
								ret = device_space_manager.allocate(bytes, alignment);
								if(ret == nullptr){
									//If that failed try to move to swap memory
									ret = swap_space_manager.allocate(bytes, alignment);
									
									if(ret == nullptr){
										//If that failed, try to preemt pages
										std::pair<void*, MemoryType> new_mem_location = preempt(size, alignment);
										ret = new_mem_location.first;
										
										if(ret == nullptr){
											throw std::runtime_error("Out of memory.");
										}else{
											move_memory(mem_location_iter, size, new_mem_location);
										}
								}else{
									}else{
										move_memory(mem_location_iter, size, std::make_pair<void*, MemoryType>(ret, MemoryType::SWAP));
									}
								}else{
									move_memory(mem_location_iter, size, std::make_pair<void*, MemoryType>(ret, MemoryType::DEVICE));
								}
							}else{
								//TODO: Maybe try to copy data from device or swap to host?
								ret = mem_location_iter->second.first;
							}
							break;
						case SWAP:
							ret = mem_location_iter->second.first;
							break;
						case HOST:
							if(mem_location_iter->second.second == MemoryType::DEVICE){
								//First try to move data to swap memory
								void* ret = swap_space_manager.allocate(bytes, alignment);
									
								if(ret == nullptr){
									//If that failed try to move directly to host
									ret = aligned_alloc(alignment, bytes);
					
									if(ret == nullptr){
										throw std::bad_alloc("Failed to allocate memory");
									}else{
										move_memory(mem_location_iter, size, std::make_pair<void*, MemoryType>(ret, MemoryType::HOST));
									}
								}else{
									move_memory(mem_location_iter, size, std::make_pair<void*, MemoryType>(ret, MemoryType::SWAP));
								}
							}else{
								ret = mem_location_iter->second.first;
							}
							break;
						default:
							assert(false && "Reached unreachable state");
					}
					locks.emplace(reinterpret_cast<uintptr_t>(p), memory_type);
					return ret;
					//TODO: Mark memory as locked, not being removable form memory till it was released 
				}else{
					throw std::invalid_argument("Pointer was not allocated using this allocator");
				}
			}
			
		}
		
		void release(void* p){
			if(p != nullptr){
				std::map<uintptr_t, MemoryType>::iterator lock_iter = locks.find(reinterpret_cast<uintptr_t>(p));
				if(lock_iter != locks.end()){
					locks.erase(lock_iter);
				}
			}
		}
	
};

}