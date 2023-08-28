#ifndef MARCHING_CUBES_CUH
#define MARCHING_CUBES_CUH

#include "particle_buffer.cuh"
#include "settings.h"
#include "utility_funcs.hpp"

//Mainly copied from https://github.com/NVlabs/instant-ngp/blob/master/src/marching_cubes.cu
namespace mn {
//NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers, readability-identifier-naming, misc-definitions-in-headers) CUDA does not yet support std::span; Common names for physical formulas; Cannot declare __global__ functions inline

constexpr float MARCHING_CUBES_GRID_BLOCK_SPACING_INV = 30.0f;
constexpr float MARCHING_CUBES_DX_INV = (MARCHING_CUBES_GRID_BLOCK_SPACING_INV * (1 << config::DOMAIN_BITS));
constexpr float MARCHING_CUBES_DX = 1.f / MARCHING_CUBES_DX_INV;
constexpr size_t MARCHING_CUBES_GRID_SCALING = const_ceil<size_t, float>(config::G_DX / MARCHING_CUBES_DX);
constexpr size_t MARCHING_CUBES_MAX_ACTIVE_BLOCK = config::G_MAX_ACTIVE_BLOCK * MARCHING_CUBES_GRID_SCALING;

//TODO: Currently looks wrong with degress bigger 1. Maybe wrong threshold? Maybe other issues?
//NOTE: Values bigger than 0 will only have effect on cells being classified full/empty if they have particles or are fully enclosed by full cells
constexpr size_t MARCHING_CUBES_INTERPOLATION_DEGREE = 1;//MARCHING_CUBES_GRID_SCALING;

constexpr float MARCHING_CUBES_HALFSPACE_TEST_THRESHOLD = 0.0f;//FIXME: Set to corect value

using MarchingCubesGridBufferDomain  = CompactDomain<int, MARCHING_CUBES_MAX_ACTIVE_BLOCK>;

//NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables, readability-identifier-naming) Check is buggy and reports variable errors for template arguments
using MarchingCubesGridBlockData = Structural<StructuralType::DYNAMIC, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::COMPACT>, MarchingCubesGridBufferDomain, attrib_layout::AOS, f32_, u32_, u32_, u32_>;//density, index_x, index_y, index_z
//NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables, readability-identifier-naming)

struct MarchingCubesGridBuffer : Instance<MarchingCubesGridBlockData> {
	using base_t = Instance<MarchingCubesGridBlockData>;
	
	managed_memory_type* managed_memory;

	template<typename Allocator>
	explicit MarchingCubesGridBuffer(Allocator allocator, managed_memory_type* managed_memory)
		: base_t {spawn<MarchingCubesGridBlockData, orphan_signature>(allocator)} 
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
		cu_dev.compute_launch({(block_count + config::CUDA_WARP_SIZE - 1) / config::CUDA_WARP_SIZE, config::CUDA_WARP_SIZE}, marching_cubes_clear_grid, static_cast<size_t>(block_count), *this);
		managed_memory->release(
			(this_is_locked ? nullptr : this->release())
		);
	}
};

//TODO: Change for more spatial packing
constexpr int marching_cubes_calculate_offset(const uint32_t x, const uint32_t y, const uint32_t z, const std::array<int, 3>& grid_size){
	return x + y * grid_size[0] + z * grid_size[0] * grid_size[1];
}

constexpr std::array<int, 3> marching_cubes_calculate_id(const int offset, const std::array<int, 3>& grid_size){
	return {
		  offset % grid_size[0]
		, (offset / grid_size[0]) % grid_size[1]
		, (offset / (grid_size[0] * grid_size[1]))
	};
}

template<typename MarchingCubesGrid>
__global__ void marching_cubes_clear_grid(size_t block_count, MarchingCubesGrid marching_cubes_grid) {
	uint32_t blockno = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(blockno >= block_count) {
		return;
	}
	
	marching_cubes_grid.val_1d(_0, blockno) = 0.0f;
	marching_cubes_grid.val_1d(_1, blockno) = 0;
	marching_cubes_grid.val_1d(_2, blockno) = 0;
	marching_cubes_grid.val_1d(_3, blockno) = 0;
}

template<size_t KernelLength = 3, typename Partition>
__forceinline__ __device__ void marching_cubes_fetch_id(const Partition prev_partition, const int particle_id, const std::array<int, 3> blockid_arr, int& advection_source_blockno, int& source_pidib){
	const ivec3 blockid {blockid_arr[0], blockid_arr[1], blockid_arr[2]};
	
	//Fetch index of the advection source
	{
		//Fetch advection (direction at high bits, particle in in cell at low bits)
		const int advect = particle_id;

		//Retrieve the direction (first stripping the particle id by division)
		ivec3 offset;
		dir_components<KernelLength>(advect / config::G_PARTICLE_NUM_PER_BLOCK, offset.data_arr());

		//Retrieve the particle id by AND for lower bits
		source_pidib = advect % config::G_PARTICLE_NUM_PER_BLOCK;

		//Get global index by adding blockid and offset
		const ivec3 global_advection_index = blockid + offset;

		//Get block_no from partition
		advection_source_blockno = prev_partition.query(global_advection_index);
	}
}

template<typename Partition>
__forceinline__ __device__ int marching_cubes_convert_id(const Partition prev_partition, const int particle_id, const std::array<int, 3> blockid_offset_arr){
	const ivec3 blockid_offset {blockid_offset_arr[0], blockid_offset_arr[1], blockid_offset_arr[2]};
	
	//Fetch index of the advection source
	{
		//Fetch advection (direction at high bits, particle in in cell at low bits)
		const int advect = particle_id;

		//Retrieve the direction (first stripping the particle id by division)
		ivec3 offset;
		dir_components<3>(advect / config::G_PARTICLE_NUM_PER_BLOCK, offset.data_arr());

		//Retrieve the particle id by AND for lower bits
		const int source_pidib = advect % config::G_PARTICLE_NUM_PER_BLOCK;

		//Calculate offset from current blockid
		const int dirtag = dir_offset<2 + 3>((blockid_offset + offset).data_arr());
		
		return (dirtag * config::G_PARTICLE_NUM_PER_BLOCK) | source_pidib;
	}
}


//TODO: Maybe consider mass when calculationg nearest
//minima: x-, x+, y-, y+, z-, z+
template<typename Partition, typename ParticleBuffer>
__forceinline__ __device__ int marching_cubes_get_cell_minima(const Partition prev_partition, const ParticleBuffer particle_buffer, const std::array<float, 3> bounding_box_offset_arr, const std::array<int, 3> global_grid_blockid_arr, const std::array<int, 3> global_grid_cellid_arr, const std::array<int, 3> marching_cubes_cell_id_arr, std::array<int, 6>* minima){
	const vec3 bounding_box_offset {bounding_box_offset_arr[0], bounding_box_offset_arr[1], bounding_box_offset_arr[2]};
	const ivec3 global_grid_blockid {global_grid_blockid_arr[0], global_grid_blockid_arr[1], global_grid_blockid_arr[2]};
	const ivec3 global_grid_cellid {global_grid_cellid_arr[0], global_grid_cellid_arr[1], global_grid_cellid_arr[2]};
	const ivec3 marching_cubes_cell_id {marching_cubes_cell_id_arr[0], marching_cubes_cell_id_arr[1], marching_cubes_cell_id_arr[2]};
	
	//Init maxima and minima
	#pragma unroll 6
	for(int i = 0; i < 6; ++i) {
		(*minima)[i] = -1;
	}
	
	//Iterate through all particles of grid_cell
	//NOTE: Neighbour cells needed cause particle are stored with (get_cell_id<2> - 1)
	for(int grid_x = -2; grid_x <= -1; ++grid_x){
		for(int grid_y = -2; grid_y <= -1; ++grid_y){
			for(int grid_z = -2; grid_z <= -1; ++grid_z){
				const ivec3 cellid_offset {grid_x, grid_y, grid_z};
				const ivec3 current_cellid = global_grid_cellid + cellid_offset;
				const int current_cellno = ((current_cellid[0] & config::G_BLOCKMASK) << (config::G_BLOCKBITS << 1)) | ((current_cellid[1] & config::G_BLOCKMASK) << config::G_BLOCKBITS) | (current_cellid[2] & config::G_BLOCKMASK);
				
				const ivec3 current_blockid = current_cellid / static_cast<int>(config::G_BLOCKSIZE);
				const int current_blockno = prev_partition.query(current_blockid);
				const ivec3 blockid_offset = current_blockid - global_grid_blockid;
				
				//For empty blocks (blockno = -1) current_bucket_size will be zero
				const int current_bucket_size = particle_buffer.cell_particle_counts[(std::max(current_blockno, 0) * config::G_BLOCKVOLUME + current_cellno)] * std::min(current_blockno + 1, 1);
				
				for(int particle_id_in_block = 0; particle_id_in_block < current_bucket_size; ++particle_id_in_block) {
					const int particle_id = particle_buffer.cellbuckets[current_blockno * config::G_PARTICLE_NUM_PER_BLOCK + current_cellno * config::G_MAX_PARTICLES_IN_CELL + particle_id_in_block];
					
					int advection_source_blockno;
					int source_pidib;
					marching_cubes_fetch_id(prev_partition, particle_id, current_blockid.data_arr(), advection_source_blockno, source_pidib);

					//Get bin from particle buffer
					const auto source_bin = particle_buffer.ch(_0, particle_buffer.bin_offsets[advection_source_blockno] + source_pidib / config::G_BIN_CAPACITY);
					
					//Get mass
					const float mass = source_bin.val(_0, source_pidib % config::G_BIN_CAPACITY);//mass
					
					//Get particle position
					vec3 pos;
					pos[0] = source_bin.val(_1, source_pidib % config::G_BIN_CAPACITY);
					pos[1] = source_bin.val(_2, source_pidib % config::G_BIN_CAPACITY);
					pos[2] = source_bin.val(_3, source_pidib % config::G_BIN_CAPACITY);
					
					pos -= bounding_box_offset;
					
					//Get position of grid cell
					const ivec3 global_base_index_0 = get_cell_id<0>(pos.data_arr(), {0.0f, 0.0f, 0.0f}, {MARCHING_CUBES_DX_INV, MARCHING_CUBES_DX_INV, MARCHING_CUBES_DX_INV});
					
					//Get position relative to grid cell
					const vec3 local_pos_0 = pos - global_base_index_0 * MARCHING_CUBES_DX;
					
					//Calculate distance to box walls and store minima
					const float distances[6] {
						  local_pos_0[0]
						, MARCHING_CUBES_DX - local_pos_0[0]
						, local_pos_0[1]
						, MARCHING_CUBES_DX - local_pos_0[1]
						, local_pos_0[2]
						, MARCHING_CUBES_DX - local_pos_0[2]
					};
					
					//If particle is in cell, test if it is smaller than our minima
					if(
						   (global_base_index_0[0] == marching_cubes_cell_id[0])
						&& (global_base_index_0[1] == marching_cubes_cell_id[1])
						&& (global_base_index_0[2] == marching_cubes_cell_id[2])
					){
						//Store minima
						#pragma unroll 6
						for(int dd = 0; dd < 6; ++dd) {
							/*
							bool next = false;
							int last_minimum;
							do{
								last_minimum = minima[marching_cubes_cellno][dd];
								
								if(last_minimum != -1){
									int advection_source_blockno_minimum;
									int source_pidib_minimum;
									marching_cubes_fetch_id<2 + 3>(prev_partition, last_minimum, (global_grid_blockid - 2).data_arr(), advection_source_blockno_minimum, source_pidib_minimum);

									//Get bin from particle buffer
									const auto source_bin_minimum = particle_buffer.ch(_0, particle_buffer.bin_offsets[advection_source_blockno_minimum] + source_pidib_minimum / config::G_BIN_CAPACITY);
									
									//Get particle position
									vec3 pos_minimum;
									pos_minimum[0] = source_bin_minimum.val(_1, source_pidib_minimum % config::G_BIN_CAPACITY);
									pos_minimum[1] = source_bin_minimum.val(_2, source_pidib_minimum % config::G_BIN_CAPACITY);
									pos_minimum[2] = source_bin_minimum.val(_3, source_pidib_minimum % config::G_BIN_CAPACITY);
									
									pos_minimum -= bounding_box_offset;
									
									//Get position of grid cell
									const ivec3 global_base_index_minimum = get_cell_id<0>(pos_minimum.data_arr(), {0.0f, 0.0f, 0.0f}, {MARCHING_CUBES_DX_INV, MARCHING_CUBES_DX_INV, MARCHING_CUBES_DX_INV});
									
									//Get position relative to grid cell
									const vec3 local_pos_minimum = pos_minimum - global_base_index_minimum * MARCHING_CUBES_DX;
									
									const float distances_minimum[6] {
										  local_pos_minimum[0]
										, MARCHING_CUBES_DX - local_pos_minimum[0]
										, local_pos_minimum[1]
										, MARCHING_CUBES_DX - local_pos_minimum[1]
										, local_pos_minimum[2]
										, MARCHING_CUBES_DX - local_pos_minimum[2]
									};
									
									if(distances_minimum[dd] <= distances[dd]){
										next = true;
									}
								}
							}while(!next || (last_minimum != atomicCAS(&(minima[marching_cubes_cellno][dd]), last_minimum, particle_id_in_block)));
							*/
							const int last_minimum = (*minima)[dd];
							
							if(last_minimum != -1){
								int advection_source_blockno_minimum;
								int source_pidib_minimum;
								marching_cubes_fetch_id<2 + 3>(prev_partition, last_minimum, (global_grid_blockid - 2).data_arr(), advection_source_blockno_minimum, source_pidib_minimum);
								
								//Get bin from particle buffer
								const auto source_bin_minimum = particle_buffer.ch(_0, particle_buffer.bin_offsets[advection_source_blockno_minimum] + source_pidib_minimum / config::G_BIN_CAPACITY);
								
								//Get particle position
								vec3 pos_minimum;
								pos_minimum[0] = source_bin_minimum.val(_1, source_pidib_minimum % config::G_BIN_CAPACITY);
								pos_minimum[1] = source_bin_minimum.val(_2, source_pidib_minimum % config::G_BIN_CAPACITY);
								pos_minimum[2] = source_bin_minimum.val(_3, source_pidib_minimum % config::G_BIN_CAPACITY);
								
								pos_minimum -= bounding_box_offset;
								
								//Get position of grid cell
								const ivec3 global_base_index_minimum = get_cell_id<0>(pos_minimum.data_arr(), {0.0f, 0.0f, 0.0f}, {MARCHING_CUBES_DX_INV, MARCHING_CUBES_DX_INV, MARCHING_CUBES_DX_INV});
								
								//Get position relative to grid cell
								const vec3 local_pos_minimum = pos_minimum - global_base_index_minimum * MARCHING_CUBES_DX;
								
								const float distances_minimum[6] {
									  local_pos_minimum[0]
									, MARCHING_CUBES_DX - local_pos_minimum[0]
									, local_pos_minimum[1]
									, MARCHING_CUBES_DX - local_pos_minimum[1]
									, local_pos_minimum[2]
									, MARCHING_CUBES_DX - local_pos_minimum[2]
								};
								
								if(distances_minimum[dd] <= distances[dd]){
									continue;
								}
							}
							
							(*minima)[dd] = marching_cubes_convert_id(prev_partition, particle_id, (blockid_offset + 2).data_arr());
						}
					}
				}
			}
		}
	}
}

/*
vertex indices with z=0
0 -> 1   <-- first edge is 0->1, then 1->2 etc
^    |
|	 v
3 <- 2

with z=1
4 -> 5   <-- fourth edge is 4->5 etc
^    |
|	 v
7 <- 6

edges 8-11 go in +z direction from vertex 0-3
*/
template<typename Partition, typename ParticleBuffer, typename MarchingCubesGrid>
__global__ void marching_cubes_gen_vertices(const Partition prev_partition, const ParticleBuffer particle_buffer, const std::array<int, 3> bounding_box_min_arr, const std::array<float, 3> bounding_box_offset_arr, const std::array<int, 3> grid_size, const float thresh, MarchingCubesGrid marching_cubes_grid, uint32_t* __restrict__ vertex_count) {
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
	
	if (x>=grid_size[0] || y>=grid_size[1] || z>=grid_size[2]) return;
	
	const ivec3 bounding_box_min {bounding_box_min_arr[0], bounding_box_min_arr[1], bounding_box_min_arr[2]};
	const vec3 bounding_box_offset {bounding_box_offset_arr[0], bounding_box_offset_arr[1], bounding_box_offset_arr[2]};

	float f0 = marching_cubes_grid.val(_0, marching_cubes_calculate_offset(x, y, z, grid_size));//density
	bool inside=(f0>thresh);
	
	if(inside){
		printf("A %d %d %d # %d %d %d\n", static_cast<int>(x), static_cast<int>(y), static_cast<int>(z), bounding_box_min[0] + static_cast<int>(x / MARCHING_CUBES_GRID_SCALING), bounding_box_min[1] + static_cast<int>(y / MARCHING_CUBES_GRID_SCALING), bounding_box_min[2] + static_cast<int>(z / MARCHING_CUBES_GRID_SCALING));
	}
	
	if (x<grid_size[0]-1) {
		float f1 = marching_cubes_grid.val(_0, marching_cubes_calculate_offset(x + 1, y, z, grid_size));//density
		if (inside != (f1>thresh)) {
			uint32_t vidx = atomicAdd(vertex_count,1);
			
			ivec3 marching_cubes_cell_id;
			int minima_index;
			if(inside){
				marching_cubes_cell_id[0] = x;
				marching_cubes_cell_id[1] = y;
				marching_cubes_cell_id[2] = z;
				minima_index = 1;
			}else{//(f1>thresh)
				marching_cubes_cell_id[0] = x + 1;
				marching_cubes_cell_id[1] = y;
				marching_cubes_cell_id[2] = z;
				minima_index = 0;
			}
			
			const ivec3 global_grid_cellid = (marching_cubes_cell_id / MARCHING_CUBES_GRID_SCALING).cast<int>() + bounding_box_min;
			const ivec3 global_grid_blockid = global_grid_cellid / config::G_BLOCKSIZE;
			
			int minima[6];
			marching_cubes_get_cell_minima(prev_partition, particle_buffer, bounding_box_offset_arr, global_grid_blockid.data_arr(), global_grid_cellid.data_arr(), marching_cubes_cell_id.data_arr(), reinterpret_cast<std::array<int, 6>*>(&minima));

			marching_cubes_grid.val(_1, marching_cubes_calculate_offset(x, y, z, grid_size)) = minima[minima_index] + 1;//index_x
			
			printf("X %d %d %d # %d %d %d # %d %d # %d %d %d\n", static_cast<int>(x), static_cast<int>(y), static_cast<int>(z), marching_cubes_cell_id[0], marching_cubes_cell_id[1], marching_cubes_cell_id[2], minima_index, minima[minima_index], (particle_id_mapping_buffer[particle_buffer.bin_offsets[advection_source_blockno] * config::G_BIN_CAPACITY + source_pidib]), advection_source_blockno, source_pidib);
			
			/*
			marching_cubes_grid.val(_1, marching_cubes_calculate_offset(x, y, z, grid_size))=vidx+1;//index_x
			float prevf=f0,nextf=f1;
			float dt=((thresh-prevf)/(nextf-prevf));
			const vec3 pos = bounding_box_offset + (vec3{float(x)+dt, float(y), float(z)} * MARCHING_CUBES_DX);
			verts_out[3 * vidx] = pos[0];
			verts_out[3 * vidx + 1] = pos[1];
			verts_out[3 * vidx + 2] = pos[2];
			*/
		}
	}
	if (y<grid_size[1]-1) {
		float f1 = marching_cubes_grid.val(_0, marching_cubes_calculate_offset(x, y + 1, z, grid_size));//density
		if (inside != (f1>thresh)) {
			uint32_t vidx = atomicAdd(vertex_count,1);//index_y
			
			ivec3 marching_cubes_cell_id;
			int minima_index;
			if(inside){
				marching_cubes_cell_id[0] = x;
				marching_cubes_cell_id[1] = y;
				marching_cubes_cell_id[2] = z;
				minima_index = 3;
			}else{//(f1>thresh)
				marching_cubes_cell_id[0] = x;
				marching_cubes_cell_id[1] = y + 1;
				marching_cubes_cell_id[2] = z;
				minima_index = 2;
			}
			
			const ivec3 global_grid_cellid = (marching_cubes_cell_id / MARCHING_CUBES_GRID_SCALING).cast<int>() + bounding_box_min;
			const ivec3 global_grid_blockid = global_grid_cellid / config::G_BLOCKSIZE;
			
			int minima[6];
			marching_cubes_get_cell_minima(prev_partition, particle_buffer, bounding_box_offset_arr, global_grid_blockid.data_arr(), global_grid_cellid.data_arr(), marching_cubes_cell_id.data_arr(), reinterpret_cast<std::array<int, 6>*>(&minima));
			
			marching_cubes_grid.val(_2, marching_cubes_calculate_offset(x, y, z, grid_size)) = minima[minima_index] + 1;//index_y
			
			printf("Y %d %d %d # %d %d %d # %d %d # %d %d %d\n", static_cast<int>(x), static_cast<int>(y), static_cast<int>(z), marching_cubes_cell_id[0], marching_cubes_cell_id[1], marching_cubes_cell_id[2], minima_index, minima[minima_index], (particle_id_mapping_buffer[particle_buffer.bin_offsets[advection_source_blockno] * config::G_BIN_CAPACITY + source_pidib]), advection_source_blockno, source_pidib);
			
			/*
			marching_cubes_grid.val(_2, marching_cubes_calculate_offset(x, y, z, grid_size))=vidx+1;
			float prevf=f0,nextf=f1;
			float dt=((thresh-prevf)/(nextf-prevf));
			const vec3 pos = bounding_box_offset + (vec3{float(x), float(y)+dt, float(z)} * MARCHING_CUBES_DX);
			verts_out[3 * vidx] = pos[0];
			verts_out[3 * vidx + 1] = pos[1];
			verts_out[3 * vidx + 2] = pos[2];
			*/
		}
	}
	if (z<grid_size[2]-1) {
		float f1 = marching_cubes_grid.val(_0, marching_cubes_calculate_offset(x, y, z + 1, grid_size));//density
		if (inside != (f1>thresh)) {
			uint32_t vidx = atomicAdd(vertex_count,1);
			
			ivec3 marching_cubes_cell_id;
			int minima_index;
			if(inside){
				marching_cubes_cell_id[0] = x;
				marching_cubes_cell_id[1] = y;
				marching_cubes_cell_id[2] = z;
				minima_index = 5;
			}else{//(f1>thresh)
				marching_cubes_cell_id[0] = x;
				marching_cubes_cell_id[1] = y;
				marching_cubes_cell_id[2] = z + 1;
				minima_index = 4;
			}
			
			const ivec3 global_grid_cellid = (marching_cubes_cell_id / MARCHING_CUBES_GRID_SCALING).cast<int>() + bounding_box_min;
			const ivec3 global_grid_blockid = global_grid_cellid / config::G_BLOCKSIZE;
			
			int minima[6];
			marching_cubes_get_cell_minima(prev_partition, particle_buffer, bounding_box_offset_arr, global_grid_blockid.data_arr(), global_grid_cellid.data_arr(), marching_cubes_cell_id.data_arr(), reinterpret_cast<std::array<int, 6>*>(&minima));
			
			marching_cubes_grid.val(_3, marching_cubes_calculate_offset(x, y, z, grid_size)) = minima[minima_index] + 1;//index_z
			
			printf("Z %d %d %d # %d %d %d # %d %d # %d %d %d\n", static_cast<int>(x), static_cast<int>(y), static_cast<int>(z), marching_cubes_cell_id[0], marching_cubes_cell_id[1], marching_cubes_cell_id[2], minima_index, minima[minima_index], (particle_id_mapping_buffer[particle_buffer.bin_offsets[advection_source_blockno] * config::G_BIN_CAPACITY + source_pidib]), advection_source_blockno, source_pidib);
			
			/*
			marching_cubes_grid.val(_3, marching_cubes_calculate_offset(x, y, z, grid_size))=vidx+1;//index_z
			float prevf=f0,nextf=f1;
			float dt=((thresh-prevf)/(nextf-prevf));
			const vec3 pos = bounding_box_offset + (vec3{float(x), float(y), float(z)+dt} * MARCHING_CUBES_DX);
			verts_out[3 * vidx] = pos[0];
			verts_out[3 * vidx + 1] = pos[1];
			verts_out[3 * vidx + 2] = pos[2];
			*/
		}
	}
}

template<typename Partition, typename ParticleBuffer>
__forceinline__ __device__ uint32_t marching_cubes_fetch_edge_data(const Partition prev_partition, const ParticleBuffer particle_buffer, const unsigned int* particle_id_mapping_buffer, const std::array<int, 3> bounding_box_min_arr, const int particle_id, const std::array<int, 3> id, std::pair<int, int>& ids){
	if(particle_id > 0){
		const ivec3 current_global_grid_cellid = (ivec3(id[0], id[1], id[2]) / MARCHING_CUBES_GRID_SCALING).cast<int>() + bounding_box_min;
		const ivec3 current_global_grid_blockid = current_global_grid_cellid / config::G_BLOCKSIZE;
		
		int advection_source_blockno;
		int source_pidib;
		marching_cubes_fetch_id<2 + 3>(prev_partition, particle_id, (current_global_grid_blockid - 2).data_arr(), advection_source_blockno, source_pidib);
		
		ids.first = advection_source_blockno;
		ids.second = source_pidib;

		return particle_id_mapping_buffer[particle_buffer.bin_offsets[advection_source_blockno] * config::G_BIN_CAPACITY + source_pidib] + 1;
	}else{
		return 0;
	}
}

template<typename MarchingCubesGrid>
__global__ void marching_cubes_gen_faces(const std::array<int, 3> grid_size, const float thresh, const MarchingCubesGrid marching_cubes_grid, uint32_t* indices_out, uint32_t *__restrict__ triangle_count) {
	// marching cubes tables from https://github.com/pmneila/PyMCubes/blob/master/mcubes/src/marchingcubes.cpp which in turn seems to be from https://web.archive.org/web/20181127124338/http://paulbourke.net/geometry/polygonise/
	// License is BSD 3-clause, which can be found here: https://github.com/pmneila/PyMCubes/blob/master/LICENSE
	/*
	static constexpr uint16_t edge_table[256] =
	{
		0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
		0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
		0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
		0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
		0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
		0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
		0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
		0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
		0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
		0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
		0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
		0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
		0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
		0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
		0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
		0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000
	};
	*/
	static constexpr int8_t triangle_table[256][16] =
	{
		{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
		{3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
		{3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
		{3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
		{9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
		{9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
		{2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
		{8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
		{9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
		{4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
		{3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
		{1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
		{4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
		{4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
		{5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
		{2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
		{9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
		{0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
		{2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
		{10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
		{5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
		{5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
		{9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
		{0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
		{1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
		{10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
		{8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
		{2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
		{7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
		{2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
		{11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
		{5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
		{11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
		{11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
		{1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
		{9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
		{5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
		{2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
		{5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
		{6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
		{3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
		{6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
		{5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
		{1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
		{10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
		{6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
		{8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
		{7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
		{3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
		{5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
		{0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
		{9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
		{8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
		{5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
		{0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
		{6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
		{10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
		{10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
		{8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
		{1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
		{0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
		{10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
		{3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
		{6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
		{9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
		{8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
		{3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
		{6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
		{0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
		{10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
		{10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
		{2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
		{7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
		{7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
		{2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
		{1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
		{11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
		{8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
		{0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
		{7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
		{10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
		{2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
		{6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
		{7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
		{2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
		{1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
		{10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
		{10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
		{0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
		{7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
		{6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
		{8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
		{9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
		{6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
		{4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
		{10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
		{8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
		{0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
		{1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
		{8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
		{10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
		{4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
		{10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
		{5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
		{11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
		{9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
		{6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
		{7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
		{3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
		{7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
		{3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
		{6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
		{9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
		{1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
		{4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
		{7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
		{6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
		{3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
		{0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
		{6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
		{0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
		{11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
		{6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
		{5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
		{9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
		{1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
		{1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
		{10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
		{0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
		{5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
		{10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
		{11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
		{9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
		{7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
		{2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
		{8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
		{9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
		{9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
		{1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
		{9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
		{9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
		{5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
		{0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
		{10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
		{2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
		{0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
		{0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
		{9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
		{5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
		{3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
		{5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
		{8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
		{0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
		{9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
		{1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
		{3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
		{4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
		{9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
		{11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
		{11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
		{2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
		{9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
		{3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
		{1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
		{4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
		{3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
		{0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
		{9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
		{1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
	};

	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
	
	const ivec3 bounding_box_min {bounding_box_min_arr[0], bounding_box_min_arr[1], bounding_box_min_arr[2]};
	const vec3 bounding_box_offset {bounding_box_offset_arr[0], bounding_box_offset_arr[1], bounding_box_offset_arr[2]};
	
	const ivec3 marching_cubes_cell_id {static_cast<int>(x), static_cast<int>(y), static_cast<int>(z)};
	
	const ivec3 global_grid_cellid = (marching_cubes_cell_id / MARCHING_CUBES_GRID_SCALING).cast<int>() + bounding_box_min;
	const ivec3 global_grid_blockid = global_grid_cellid / config::G_BLOCKSIZE;
	
	size_t cell_triangles_count = 0;
	std::array<std::array<uint32_t, 3>, 90> cell_triangles;
	std::array<std::array<std::pair<int, int>, 3>, 90> cell_triangle_ids;
	for(size_t grid_x = std::min(static_cast<int>(x) - 1, 0); grid_x <= 0; ++grid_x){
		for(size_t grid_y = std::min(static_cast<int>(y) - 1, 0); grid_y <= 0; ++grid_y){
			for(size_t grid_z = std::min(static_cast<int>(z) - 1, 0); grid_z <= 0; ++grid_z){
				
				if (grid_x>=grid_size[0]-1 || grid_y>=grid_size[1]-1 || grid_z>=grid_size[2]-1) continue;
				
				int mask=0;
				if (marching_cubes_grid.val(_0, marching_cubes_calculate_offset(grid_x, grid_y, grid_z, grid_size))>thresh) mask|=1;
				if (marching_cubes_grid.val(_0, marching_cubes_calculate_offset(grid_x + 1, grid_y, grid_z, grid_size))>thresh) mask|=2;
				if (marching_cubes_grid.val(_0, marching_cubes_calculate_offset(grid_x + 1, grid_y + 1, grid_z, grid_size))>thresh) mask|=4;
				if (marching_cubes_grid.val(_0, marching_cubes_calculate_offset(grid_x, grid_y + 1, grid_z, grid_size))>thresh) mask|=8;
				if (marching_cubes_grid.val(_0, marching_cubes_calculate_offset(grid_x, grid_y, grid_z + 1, grid_size))>thresh) mask|=16;
				if (marching_cubes_grid.val(_0, marching_cubes_calculate_offset(grid_x + 1, grid_y, grid_z + 1, grid_size))>thresh) mask|=32;
				if (marching_cubes_grid.val(_0, marching_cubes_calculate_offset(grid_x + 1, grid_y + 1, grid_z + 1, grid_size))>thresh) mask|=64;
				if (marching_cubes_grid.val(_0, marching_cubes_calculate_offset(grid_x, grid_y + 1, grid_z + 1, grid_size))>thresh) mask|=128;

				if (!mask || mask==255) return;
				int local_edges[12];
				std::pair<int, int> local_edges_ids[12]
				local_edges[0] = marching_cubes_fetch_edge_data(prev_partition, particle_buffer, particle_id_mapping_buffer, bounding_box_min_arr, marching_cubes_grid.val(_1, marching_cubes_calculate_offset(grid_x, grid_y, grid_z, grid_size)), {grid_x, grid_y, grid_z}, local_edges_ids[0]);
				local_edges[1] = marching_cubes_fetch_edge_data(prev_partition, particle_buffer, particle_id_mapping_buffer, bounding_box_min_arr, marching_cubes_grid.val(_2, marching_cubes_calculate_offset(grid_x + 1, grid_y, grid_z, grid_size)), {grid_x + 1, grid_y, grid_z}, local_edges_ids[1]);
				local_edges[2] = marching_cubes_fetch_edge_data(prev_partition, particle_buffer, particle_id_mapping_buffer, bounding_box_min_arr, marching_cubes_grid.val(_1, marching_cubes_calculate_offset(grid_x, grid_y + 1, grid_z, grid_size)), {grid_x, grid_y + 1, grid_z}, local_edges_ids[2]);
				local_edges[3] = marching_cubes_fetch_edge_data(prev_partition, particle_buffer, particle_id_mapping_buffer, bounding_box_min_arr, marching_cubes_grid.val(_2, marching_cubes_calculate_offset(grid_x, grid_y, grid_z, grid_size)), {grid_x, grid_y, grid_z}, local_edges_ids[3]);
				
				local_edges[4] = marching_cubes_fetch_edge_data(prev_partition, particle_buffer, particle_id_mapping_buffer, bounding_box_min_arr, marching_cubes_grid.val(_1, marching_cubes_calculate_offset(grid_x, grid_y, grid_z + 1, grid_size)), {grid_x, grid_y, grid_z + 1}, local_edges_ids[4]);
				local_edges[5] = marching_cubes_fetch_edge_data(prev_partition, particle_buffer, particle_id_mapping_buffer, bounding_box_min_arr, marching_cubes_grid.val(_2, marching_cubes_calculate_offset(grid_x + 1, grid_y, grid_z + 1, grid_size)), {grid_x + 1, grid_y, grid_z + 1}, local_edges_ids[5]);
				local_edges[6] = marching_cubes_fetch_edge_data(prev_partition, particle_buffer, particle_id_mapping_buffer, bounding_box_min_arr, marching_cubes_grid.val(_1, marching_cubes_calculate_offset(grid_x, grid_y + 1, grid_z + 1, grid_size)), {grid_x, grid_y + 1, grid_z + 1}, local_edges_ids[6]);
				local_edges[7] = marching_cubes_fetch_edge_data(prev_partition, particle_buffer, particle_id_mapping_buffer, bounding_box_min_arr, marching_cubes_grid.val(_2, marching_cubes_calculate_offset(grid_x, grid_y, grid_z + 1, grid_size)), {grid_x, grid_y, grid_z + 1}, local_edges_ids[7]);
				
				local_edges[8] = marching_cubes_fetch_edge_data(prev_partition, particle_buffer, particle_id_mapping_buffer, bounding_box_min_arr, marching_cubes_grid.val(_3, marching_cubes_calculate_offset(grid_x, grid_y, grid_z, grid_size)), {grid_x, grid_y, grid_z}, local_edges_ids[8]);
				local_edges[9] = marching_cubes_fetch_edge_data(prev_partition, particle_buffer, particle_id_mapping_buffer, bounding_box_min_arr, marching_cubes_grid.val(_3, marching_cubes_calculate_offset(grid_x + 1, grid_y, grid_z, grid_size)), {grid_x + 1, grid_y, grid_z}, local_edges_ids[9]);
				local_edges[10] = marching_cubes_fetch_edge_data(prev_partition, particle_buffer, particle_id_mapping_buffer, bounding_box_min_arr, marching_cubes_grid.val(_3, marching_cubes_calculate_offset(grid_x + 1, grid_y + 1, grid_z, grid_size)), {grid_x + 1, grid_y + 1, grid_z}, local_edges_ids[10]);
				local_edges[11] = marching_cubes_fetch_edge_data(prev_partition, particle_buffer, particle_id_mapping_buffer, bounding_box_min_arr, marching_cubes_grid.val(_3, marching_cubes_calculate_offset(grid_x, grid_y + 1, grid_z, grid_size)), {grid_x, grid_y + 1, grid_z}, local_edges_ids[11]);
				uint32_t tricount=0;
				const int8_t *triangles=triangle_table[mask];
				for (;tricount<15;tricount+=3) if (triangles[tricount]<0) break;
				
				for (int i=0;i<5;++i) {
					bool adjacent_triangle = false;
					
					uint32_t indices[3];
					std::array<std::pair<int, int>, 3> ids;
					for (int k=0;k<3;++k) {
						int j = triangles[3 * i + k];
						if (j<0) break;
						if (!local_edges[j]) {
							printf("at %d %d %d, mask is %d, j is %d, local_edges is 0\n", grid_x,grid_y,grid_z,mask,j);
						}
						if(j == 0 || j == 3 || j == 8){
							adjacent_triangle = true;
						}
						indices[k]=local_edges[j]-1;
						ids[k] = local_edges_ids[j];
					}
					
					const bool all_unique = (
							  (indices[0] != indices[1])
						   && (indices[1] != indices[2])
						   && (indices[2] != indices[0])
					);
					
					//Only output triangle if it is not degenerated
					if(all_unique){
						if(grid_x == x && grid_y == y && grid_z == z){
							uint32_t tidx = atomicAdd(triangle_count,3);
							if (indices_out) {
								//NOTE: Flipping order to make ccw
								indices_out[tidx] = indices[1];
								indices_out[tidx + 1] = indices[0];
								indices_out[tidx + 2] = indices[2];
							}
							printf("B %d %d %d # %d %d %d\n", static_cast<int>(grid_x), static_cast<int>(grid_y), static_cast<int>(grid_z), indices[0], indices[1], indices[2]);
						}
						if(adjacent_triangle){
							cell_triangles[cell_triangles_count] = indices;
							cell_triangle_ids[cell_triangles_count] = ids;
							cell_triangles_count++;
						}
					}else{
						if(grid_x == x && grid_y == y && grid_z == z){
							printf("C %d %d %d # %d %d %d\n", static_cast<int>(grid_x), static_cast<int>(grid_y), static_cast<int>(grid_z), indices[0], indices[1], indices[2]);
						}
					}
				}
			}
		}
	}
	
	const float cotan_clamp_min_rad = 1.0f / std::tan(3.0f * (180.0f / static_cast<float>(M_PI)));
	const float cotan_clamp_max_rad = 1.0f / std::tan(177.0f * (180.0f / static_cast<float>(M_PI)));
	
	//Iterate through all particles of grid_cell
	//NOTE: Neighbour cells needed cause particle are stored with (get_cell_id<2> - 1)
	for(int grid_x = -2; grid_x <= -1; ++grid_x){
		for(int grid_y = -2; grid_y <= -1; ++grid_y){
			for(int grid_z = -2; grid_z <= -1; ++grid_z){
				const ivec3 cellid_offset {grid_x, grid_y, grid_z};
				const ivec3 current_cellid = global_grid_cellid + cellid_offset;
				const int current_cellno = ((current_cellid[0] & config::G_BLOCKMASK) << (config::G_BLOCKBITS << 1)) | ((current_cellid[1] & config::G_BLOCKMASK) << config::G_BLOCKBITS) | (current_cellid[2] & config::G_BLOCKMASK);
				
				const ivec3 current_blockid = current_cellid / static_cast<int>(config::G_BLOCKSIZE);
				const int current_blockno = prev_partition.query(current_blockid);
				const ivec3 blockid_offset = current_blockid - global_grid_blockid;
				
				//For empty blocks (blockno = -1) current_bucket_size will be zero
				const int current_bucket_size = particle_buffer.cell_particle_counts[(std::max(current_blockno, 0) * config::G_BLOCKVOLUME + current_cellno)] * std::min(current_blockno + 1, 1);
				
				for(int particle_id_in_block = 0; particle_id_in_block < current_bucket_size; ++particle_id_in_block) {
					const int particle_id = particle_buffer.cellbuckets[current_blockno * config::G_PARTICLE_NUM_PER_BLOCK + current_cellno * config::G_MAX_PARTICLES_IN_CELL + particle_id_in_block];
					
					int advection_source_blockno;
					int source_pidib;
					marching_cubes_fetch_id(prev_partition, particle_id, current_blockid.data_arr(), advection_source_blockno, source_pidib);
					
					const int global_particle_id = particle_id_mapping_buffer[particle_buffer.bin_offsets[advection_source_blockno] * config::G_BIN_CAPACITY + source_pidib]

					//Get bin from particle buffer
					const auto source_bin = particle_buffer.ch(_0, particle_buffer.bin_offsets[advection_source_blockno] + source_pidib / config::G_BIN_CAPACITY);
					
					//Get particle position
					vec3 pos;
					pos[0] = source_bin.val(_1, source_pidib % config::G_BIN_CAPACITY);
					pos[1] = source_bin.val(_2, source_pidib % config::G_BIN_CAPACITY);
					pos[2] = source_bin.val(_3, source_pidib % config::G_BIN_CAPACITY);
					
					//Get position of grid cell
					const ivec3 global_base_index_0 = get_cell_id<0>((pos - bounding_box_offset).data_arr(), {0.0f, 0.0f, 0.0f}, {MARCHING_CUBES_DX_INV, MARCHING_CUBES_DX_INV, MARCHING_CUBES_DX_INV});
					
					//If particle is in cell, classify it
					if(
						   (global_base_index_0[0] == marching_cubes_cell_id[0])
						&& (global_base_index_0[1] == marching_cubes_cell_id[1])
						&& (global_base_index_0[2] == marching_cubes_cell_id[2])
					){
						AlphaShapesPointType point_type = AlphaShapesPointType::ISOLATED_POINT;
						
						vec3 summed_normal {0.0f, 0.0f, 0.0f};
						vec3 summed_laplacians {0.0f, 0.0f, 0.0f};
						float summed_face_area = 0.0f;
						float summed_angles = 0.0f;
						
						//If triangles were generated near the current cell, classify particles based on it.
						if(cell_triangles_count > 0){
							bool already_finalized = false;
							bool outer = false;
							for(size_t triangle_index = 0; triangle_index < cell_triangles_count; ++triangle_index){
								const std::array<uint32_t, 3> current_triangle = cell_triangles[triangle_index];
								
								const auto source_bin_0 = particle_buffer.ch(_0, particle_buffer.bin_offsets[cell_triangle_ids[triangle_index][0].first] + cell_triangle_ids[triangle_index][0].second / config::G_BIN_CAPACITY);
								const auto source_bin_1 = particle_buffer.ch(_0, particle_buffer.bin_offsets[cell_triangle_ids[triangle_index][1].first] + cell_triangle_ids[triangle_index][1].second / config::G_BIN_CAPACITY);
								const auto source_bin_2 = particle_buffer.ch(_0, particle_buffer.bin_offsets[cell_triangle_ids[triangle_index][2].first] + cell_triangle_ids[triangle_index][2].second / config::G_BIN_CAPACITY);
		
								position[0] = source_bin.val(_1, source_pidib % config::G_BIN_CAPACITY);
								position[1] = source_bin.val(_2, source_pidib % config::G_BIN_CAPACITY);
								position[2] = source_bin.val(_3, source_pidib % config::G_BIN_CAPACITY);
								
								const std::array<vec3, 3> triangle_positions {
									  vec3(source_bin.val(_1, cell_triangle_ids[triangle_index][0].second % config::G_BIN_CAPACITY), source_bin.val(_2, cell_triangle_ids[triangle_index][0].second % config::G_BIN_CAPACITY), source_bin.val(_3, cell_triangle_ids[triangle_index][0].second % config::G_BIN_CAPACITY))
									, vec3(source_bin.val(_1, cell_triangle_ids[triangle_index][1].second % config::G_BIN_CAPACITY), source_bin.val(_2, cell_triangle_ids[triangle_index][1].second % config::G_BIN_CAPACITY), source_bin.val(_3, cell_triangle_ids[triangle_index][1].second % config::G_BIN_CAPACITY))
									, vec3(source_bin.val(_1, cell_triangle_ids[triangle_index][2].second % config::G_BIN_CAPACITY), source_bin.val(_2, cell_triangle_ids[triangle_index][2].second % config::G_BIN_CAPACITY), source_bin.val(_3, cell_triangle_ids[triangle_index][2].second % config::G_BIN_CAPACITY))
								};
								
								vec3 face_normal;
								vec_cross_vec_3d(face_normal.data_arr(), (triangle_positions[1] - triangle_positions[0]).data_arr(), (triangle_positions[2] - triangle_positions[0]).data_arr());
								
								const float face_normal_length = sqrt(face_normal[0] * face_normal[0] + face_normal[1] * face_normal[1] + face_normal[2] * face_normal[2]);
								
								//Normalize
								face_normal = face_normal / face_normal_length;
								
								const float face_area = 0.5f * face_normal_length;
												
								//Perform halfspace test
								const bool in_halfspace = alpha_shapes_test_in_halfspace(
									  {
										  triangle_positions[0].data_arr()
										, triangle_positions[1].data_arr()
										, triangle_positions[2].data_arr()
									  }
									, pos.data_arr()
								);
								
								const bool in_halfspace_without_threshold = face_normal.dot(pos - triangle_positions[0]) > 0.0f;
								
								//If particle is a triangle vertex it is part of the triangle
								if(current_triangle[0] == global_particle_id || current_triangle[1] == global_particle_id || current_triangle[2] == global_particle_id){
									int contact_index;
									if(current_triangle[0] == global_particle_id){
										contact_index = 0;
									}else if(current_triangle[1] == global_particle_id){
										contact_index = 1;
									}else{//current_triangle[2] == global_particle_id
										contact_index = 2;
									}
									
									auto particle_bin													= surface_particle_buffer.ch(_0, particle_buffer.bin_offsets[advection_source_blockno] + source_pidib / config::G_BIN_CAPACITY);
									const int particle_id_in_bin = source_pidib  % config::G_BIN_CAPACITY;
									
									float cosine = (triangle_positions[(contact_index + 1) % 3] - triangle_positions[contact_index]).dot(triangle_positions[(contact_index + 2) % 3] - triangle_positions[contact_index]) / sqrt((triangle_positions[(contact_index + 1) % 3] - triangle_positions[contact_index]).dot(triangle_positions[(contact_index + 1) % 3] - triangle_positions[contact_index]) * (triangle_positions[(contact_index + 2) % 3] - triangle_positions[contact_index]).dot(triangle_positions[(contact_index + 2) % 3] - triangle_positions[contact_index]));
									cosine = std::min(std::max(cosine, -1.0f), 1.0f);
									const float angle = std::acos(cosine);
									
									//Normal
									summed_normal += angle * face_normal;
									
									//Gauss curvature
									summed_face_area += face_area * (1.0f / 3.0f);
									summed_angles += angle;
									
									//Mean curvature
									float current_cotan;
									float next_cotan;
									{
										const vec3 a = pos - triangle_positions[(contact_index + 2) % 3];
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
									summed_laplacians += laplacian;
									
									already_finalized = true;
									break;
								}else if(!in_halfspace && in_halfspace_without_threshold){ //If particle lie within threshold from triangle they are on the triangle
									
									const vec9 barycentric_projection_matrix {
											  triangle_positions[0][0] - triangle_positions[2][0], triangle_positions[0][1] - triangle_positions[2][1], triangle_positions[0][2] - triangle_positions[2][2]
											, triangle_positions[1][0] - triangle_positions[2][0], triangle_positions[1][1] - triangle_positions[2][1], triangle_positions[1][2] - triangle_positions[2][2]
											, -face_normal[0], -face_normal[1], -face_normal[2]
									};
									
									vec3 contact_barycentric;
									solve_linear_system(barycentric_projection_matrix.data_arr(), contact_barycentric.data_arr(),  (particle_position - triangle_positions[2]).data_arr());
								
									//Calculate last barycentric coordinate
									contact_barycentric[2] = 1.0f - contact_barycentric[0] - contact_barycentric[1];
									
									//Accumulate values
									if(contact_barycentric[0] > 0.0f){//Point somewhere in triangle
										//Use face normal
										summed_normal += face_normal;
										
										//Gauss curvature
										summed_face_area += 1.0f;//Just ensure this is not zero
										summed_angles += 2.0f * static_cast<float>(M_PI);
									}else if(contact_barycentric[0] == 1.0f || contact_barycentric[1] == 1.0f || contact_barycentric[0] == 1.0f){//Point on vertex
										int contact_index;
										if(contact_barycentric[0] == 1.0f){
											contact_index = 0;
										}else if(contact_barycentric[1] == 1.0f){
											contact_index = 1;
										}else {//contact_barycentric[2] == 1.0f
											contact_index = 2;
										}
									
										auto particle_bin													= surface_particle_buffer.ch(_0, particle_buffer.bin_offsets[advection_source_blockno] + source_pidib / config::G_BIN_CAPACITY);
										const int particle_id_in_bin = source_pidib  % config::G_BIN_CAPACITY;
										
										float cosine = (triangle_positions[(contact_index + 1) % 3] - triangle_positions[contact_index]).dot(triangle_positions[(contact_index + 2) % 3] - triangle_positions[contact_index]) / sqrt((triangle_positions[(contact_index + 1) % 3] - triangle_positions[contact_index]).dot(triangle_positions[(contact_index + 1) % 3] - triangle_positions[contact_index]) * (triangle_positions[(contact_index + 2) % 3] - triangle_positions[contact_index]).dot(triangle_positions[(contact_index + 2) % 3] - triangle_positions[contact_index]));
										cosine = std::min(std::max(cosine, -1.0f), 1.0f);
										const float angle = std::acos(cosine);
										
										//Normal
										summed_normal += angle * face_normal;
										
										//Gauss curvature
										summed_face_area += face_area * (1.0f / 3.0f);
										summed_angles += angle;
										
										//Mean curvature
										float current_cotan;
										float next_cotan;
										{
											const vec3 a = pos - triangle_positions[(contact_index + 2) % 3];
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
										summed_laplacians += laplacian;
									}else{//Point on edge
										//Use half normal
										summed_normal += 0.5f * face_normal;
										
										//Gauss curvature
										summed_face_area += face_area * (1.0f / 3.0f);//Just ensure this is not zero
										summed_angles += static_cast<float>(M_PI);
										
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
										summed_laplacians += laplacian;
									}
									
									already_finalized = true;
									break;
								}else if(in_halfspace){ //All particles that are outside of one of the halfspaces spanned by the triangles are outer
									outer = true;
									break;
								}
							}
							
							if(!outer && !already_finalized){
								point_type = AlphaShapesPointType::INNER_POINT;
							}
							
						}else{
							//If no triangles were generated, all particles are inner if the cell is full, otherwise outer (default).
							if(marching_cubes_grid.val(_0, marching_cubes_calculate_offset(grid_x, grid_y, grid_z, grid_size))>thresh){
								point_type = AlphaShapesPointType::INNER_POINT;
							}
						}
						
						//Finalize particle
						auto particle_bin													= surface_particle_buffer.ch(_0, particle_buffer.bin_offsets[advection_source_blockno] + source_pidib / config::G_BIN_CAPACITY);
						const int particle_id_in_bin = source_pidib  % config::G_BIN_CAPACITY;
						
						summed_laplacians /= 2.0f * summed_area;
						const float laplacian_norm = sqrt(summed_laplacians[0] * summed_laplacians[0] + summed_laplacians[1] * summed_laplacians[1] + summed_laplacians[2] * summed_laplacians[2]);

						const float gauss_curvature = (2.0f * static_cast<float>(M_PI) - summed_angles) / summed_area;
						const float mean_curvature = 0.5f * laplacian_norm;
						
						const float normal_length = std::sqrt(summed_normal[0] * summed_normal[0] + summed_normal[1] * summed_normal[1] + summed_normal[2] * summed_normal[2]);
						
						particle_bin.val(_0, particle_id_in_bin) = *reinterpret_cast<float*>(&point_type);
						particle_bin.val(_1, particle_id_in_bin) = summed_normal[0] / normal_length;
						particle_bin.val(_2, particle_id_in_bin) = summed_normal[1] / normal_length;
						particle_bin.val(_3, particle_id_in_bin) = summed_normal[2] / normal_length;
						particle_bin.val(_4, particle_id_in_bin) = mean_curvature;
						particle_bin.val(_5, particle_id_in_bin) = gauss_curvature;
					}
				}
			}
		}
	}
}

template<typename Partition, typename ParticleBuffer, typename MarchingCubesGrid>
__global__ void marching_cubes_calculate_density(Partition partition, Partition prev_partition, ParticleBuffer particle_buffer, ParticleBuffer next_particle_buffer, MarchingCubesGrid marching_cubes_grid, const std::array<float, 3> bounding_box_offset_arr, const std::array<int, 3> grid_size) {
	const int particle_counts	= next_particle_buffer.particle_bucket_sizes[blockIdx.x];
	const ivec3 blockid			= partition.active_keys[blockIdx.x];
	const auto advection_bucket = next_particle_buffer.blockbuckets + blockIdx.x * config::G_PARTICLE_NUM_PER_BLOCK;
	
	const vec3 bounding_box_offset {bounding_box_offset_arr[0], bounding_box_offset_arr[1], bounding_box_offset_arr[2]};
	
	// auto particle_offset = particle_buffer.bin_offsets[blockIdx.x];
	for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < particle_counts; particle_id_in_block += static_cast<int>(blockDim.x)) {
		int advection_source_blockno;
		int source_pidib;
		marching_cubes_fetch_id(prev_partition, advection_bucket[particle_id_in_block], blockid.data_arr(), advection_source_blockno, source_pidib);

		//Get bin from particle buffer
		const auto source_bin = particle_buffer.ch(_0, particle_buffer.bin_offsets[advection_source_blockno] + source_pidib / config::G_BIN_CAPACITY);
		
		//Get mass
		const float mass = source_bin.val(_0, source_pidib % config::G_BIN_CAPACITY);//mass
		
		//Get particle position
		vec3 pos;
		pos[0] = source_bin.val(_1, source_pidib % config::G_BIN_CAPACITY);
		pos[1] = source_bin.val(_2, source_pidib % config::G_BIN_CAPACITY);
		pos[2] = source_bin.val(_3, source_pidib % config::G_BIN_CAPACITY);
		
		pos -= bounding_box_offset;
		
		//Get position of grid cell
		const ivec3 global_base_index_0 = get_cell_id<0>(pos.data_arr(), {0.0f, 0.0f, 0.0f}, {MARCHING_CUBES_DX_INV, MARCHING_CUBES_DX_INV, MARCHING_CUBES_DX_INV});
		const ivec3 global_base_index = get_cell_id<MARCHING_CUBES_INTERPOLATION_DEGREE>(pos.data_arr(), {0.0f, 0.0f, 0.0f}, {MARCHING_CUBES_DX_INV, MARCHING_CUBES_DX_INV, MARCHING_CUBES_DX_INV});
		
		//printf("D %d %d %d # %d %d %d\n", global_base_index_0[0], global_base_index_0[1], global_base_index_0[2], blockid[0], blockid[1], blockid[2]);
		
		//Get position relative to grid cell
		const vec3 local_pos = pos - global_base_index * MARCHING_CUBES_DX;
		
		//Calculate weights
		vec<float, 3, MARCHING_CUBES_INTERPOLATION_DEGREE + 1> weight;
		
		#pragma unroll 3
		for(int dd = 0; dd < 3; ++dd) {
			const std::array<float, MARCHING_CUBES_INTERPOLATION_DEGREE + 1> current_weight = bspline_weight<float, MARCHING_CUBES_INTERPOLATION_DEGREE>(local_pos[dd]);
			for(int i = 0; i < MARCHING_CUBES_INTERPOLATION_DEGREE + 1; ++i){
				weight(dd, i)		  = current_weight[i];
			}
			for(int i = MARCHING_CUBES_INTERPOLATION_DEGREE + 1; i < 3; ++i){
				weight(dd, i)		  = 0.0f;
			}
		}
		
		//Spread mass
		for(char i = -static_cast<char>(MARCHING_CUBES_INTERPOLATION_DEGREE); i < static_cast<char>(MARCHING_CUBES_INTERPOLATION_DEGREE) + 1; i++) {
			for(char j = -static_cast<char>(MARCHING_CUBES_INTERPOLATION_DEGREE); j < static_cast<char>(MARCHING_CUBES_INTERPOLATION_DEGREE) + 1; j++) {
				for(char k = -static_cast<char>(MARCHING_CUBES_INTERPOLATION_DEGREE); k < static_cast<char>(MARCHING_CUBES_INTERPOLATION_DEGREE) + 1; k++) {
					const ivec3 coord = global_base_index + ivec3(i, j, k);
					
					if(
						   (coord[0] >= 0 && coord[1] >= 0 && coord[2] >= 0)
						&& (coord[0] < grid_size[0] && coord[1]< grid_size[1] && coord[2] < grid_size[2])
					){
						//Weight
						const float W = weight(0, std::abs(i)) * weight(1, std::abs(j)) * weight(2, std::abs(k));
						
						atomicAdd(&marching_cubes_grid.val(_0, marching_cubes_calculate_offset(coord[0], coord[1], coord[2], grid_size)), W * mass * (MARCHING_CUBES_DX_INV * MARCHING_CUBES_DX_INV * MARCHING_CUBES_DX_INV));
					}
				}
			}
		}
	}
}

//TODO: Somehow use that several marching cubes cells lie in the same gridcell => several minima in one run, but not too much for shared memory
template<typename Partition, typename ParticleBuffer, typename MarchingCubesGrid>
__global__ void marching_cubes_sort_out_invalid_cells(Partition partition, Partition prev_partition, ParticleBuffer particle_buffer, MarchingCubesGrid marching_cubes_grid, const std::array<int, 3> bounding_box_min_arr, const std::array<float, 3> bounding_box_offset_arr, const std::array<int, 3> grid_size) {
	const ivec3 bounding_box_min {bounding_box_min_arr[0], bounding_box_min_arr[1], bounding_box_min_arr[2]};
	const vec3 bounding_box_offset {bounding_box_offset_arr[0], bounding_box_offset_arr[1], bounding_box_offset_arr[2]};
	
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
	
	if (x>=grid_size[0] || y>=grid_size[1] || z>=grid_size[2]) return;
	
	const ivec3 marching_cubes_cell_id {static_cast<int>(x), static_cast<int>(y), static_cast<int>(z)};
	
	const ivec3 global_grid_cellid = (marching_cubes_cell_id / MARCHING_CUBES_GRID_SCALING).cast<int>() + bounding_box_min;
	const ivec3 global_grid_blockid = global_grid_cellid / config::G_BLOCKSIZE;
	
	//x-, x+, y-, y+, z-, z+
	int minima[6];
	
	marching_cubes_get_cell_minima(prev_partition, particle_buffer, bounding_box_offset_arr, global_grid_blockid.data_arr(), global_grid_cellid.data_arr(), marching_cubes_cell_id.data_arr(), reinterpret_cast<std::array<int, 6>*>(&minima));
	
	const vec3 normals[6] {
		  vec3(1.0f, 0.0f, 0.0f) //xx
		, vec3(sqrt(0.5f), sqrt(0.5f), 0.0f) //xy
		, vec3(sqrt(0.5f), 0.0f, sqrt(0.5f)) //xz
		, vec3(0.0f, 1.0f, 0.0f) //yy
		, vec3(0.0f, sqrt(0.5f), sqrt(0.5f)) //yz
		, vec3(0.0f, 0.0f, 1.0f) //zz
	};
	
	const int max_merge_counts[6] {
		  0 //xx
		, 1 //xy
		, 1 //xz
		, 0 //yy
		, 1 //yz
		, 0 //zz
	};
	
	//Check if cell is valid. If not set density to 0
	
	//If minimum is -1 the cell is empty
	if(minima[0] != -1){
		vec3 positions[6];
		
		//Fetch positions
		#pragma unroll 6
		for(int dd = 0; dd < 6; ++dd) {
			int advection_source_blockno;
			int source_pidib;
			marching_cubes_fetch_id<2 + 3>(prev_partition, minima[dd], (global_grid_blockid - 2).data_arr(), advection_source_blockno, source_pidib);

			//Get bin from particle buffer
			const auto source_bin = particle_buffer.ch(_0, particle_buffer.bin_offsets[advection_source_blockno] + source_pidib / config::G_BIN_CAPACITY);
			
			//Get particle position
			positions[dd][0] = source_bin.val(_1, source_pidib % config::G_BIN_CAPACITY);
			positions[dd][1] = source_bin.val(_2, source_pidib % config::G_BIN_CAPACITY);
			positions[dd][2] = source_bin.val(_3, source_pidib % config::G_BIN_CAPACITY);
		}
		
		//xx, xy, xz, yy, yz, zz
		int merge_counts[6];
		
		//Calculate merges
		for(int i = 0; i < 6; ++i) {
			for(int j = i + 1; j < 6; ++j) {
				const bool axes[3] {
					  (i < 2) || (j < 2)
					, (i > 1 && i < 4) || (j > 1 && j < 4)
					, (i > 3) || (j > 3)
				};
				
				int merge_type;
				if(axes[0] && !axes[1] && !axes[2]){
					merge_type = 0;
				}else if(axes[0] && axes[1] && !axes[2]){
					merge_type = 1;
				}else if(axes[0] && !axes[1] && axes[2]){
					merge_type = 2;
				}else if(!axes[0] && axes[1] && !axes[2]){
					merge_type = 3;
				}else if(!axes[0] && axes[1] && axes[2]){
					merge_type = 4;
				}else{// !axes[0] && !axes[1] && axes[2]
					merge_type = 5;
				}
				
				//Get separating plane
				const vec3 normal = normals[merge_type];
				
				//Calculate distance along separating plane
				const vec3 diff = positions[i] - positions[j];
				const float distance_in_direction = normal.dot(diff);
				
				//If smaller then threshold, the vertices merge
				if(distance_in_direction <= MARCHING_CUBES_HALFSPACE_TEST_THRESHOLD){
					merge_counts[merge_type]++;
					//printf("A %d %d %d # %d # %d %d %d %d %d %d\n", marching_cubes_cell_id[0], marching_cubes_cell_id[1], marching_cubes_cell_id[2], merge_type, minima[0], minima[1], minima[2], minima[3], minima[4], minima[5]);
				}
			}
		}
		
		//If we got more merges than allowed we have only coplanar values in cell (aligned weith separating plane) and so we mark the cell as empty
		for(int i = 0; i < 6; ++i) {
			if(merge_counts[i] > max_merge_counts[i]){
				//FIXME:marching_cubes_grid.val(_0, marching_cubes_calculate_offset(marching_cubes_cell_id[0], marching_cubes_cell_id[1], marching_cubes_cell_id[2], grid_size)) = 0.0f;
				break;
			}
		}
	}else{
		marching_cubes_grid.val(_0, marching_cubes_calculate_offset(marching_cubes_cell_id[0], marching_cubes_cell_id[1], marching_cubes_cell_id[2], grid_size)) = 0.0f;
	}
	
}
	
//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers, readability-identifier-naming, misc-definitions-in-headers)
}// namespace mn

#endif