#ifndef MULTI_GMPM_KERNELS_CUH
#define MULTI_GMPM_KERNELS_CUH

#include <MnBase/Math/Matrix/MatrixUtils.h>

#include <MnBase/Algorithm/MappingKernels.cuh>
#include <MnSystem/Cuda/DeviceUtils.cuh>

#include <curand_kernel.h>

#include "constitutive_models.cuh"
#include "particle_buffer.cuh"
#include "settings.h"
#include "utility_funcs.hpp"
#include "triangle_mesh.cuh"

namespace mn {
using namespace placeholder;//NOLINT(google-build-using-namespace) Allow placeholders to be included generally for simplification

//TODO: Make magic numbers to constants where suitable
//TODO: Ensure call dimensions and such are small enough to allow narrowing conversations. Or directly use unsigned where possible
//TODO: Maybe use names instead of formula signs for better understanding
//NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers, readability-identifier-naming, misc-definitions-in-headers) CUDA does not yet support std::span; Common names for physical formulas; Cannot declare __global__ functions inline
template<typename ParticleArray, typename Partition>
__global__ void activate_blocks(uint32_t particle_count, ParticleArray particle_array, Partition partition) {
	const uint32_t particle_id = blockIdx.x * blockDim.x + threadIdx.x;
	if(particle_id >= particle_count) {
		return;
	}

	//Get block id by particle pos
	const ivec3 coord	= get_block_id({particle_array.val(_0, particle_id), particle_array.val(_1, particle_id), particle_array.val(_2, particle_id)}) - 2;
	const ivec3 blockid = coord / static_cast<int>(config::G_BLOCKSIZE);

	//Create block in partition
	partition.insert(blockid);
}

template<typename Partition>
__global__ void activate_blocks_for_shell(uint32_t particle_count, const TriangleShell triangle_shell, Partition partition) {
	const uint32_t particle_id = blockIdx.x * blockDim.x + threadIdx.x;
	if(particle_id >= particle_count) {
		return;
	}
	
	const auto vertex_data = triangle_shell.ch(_0, 0).ch(_1, 0);
	
	const float mass = vertex_data.val(_0, particle_id);//mass
	vec3 pos;
	pos[0] = vertex_data.val(_1, particle_id);//pos
	pos[1] = vertex_data.val(_2, particle_id);
	pos[2] = vertex_data.val(_3, particle_id);
	
	//Get block id by particle pos
	const ivec3 coord	= get_block_id(pos.data_arr()) - 2;
	const ivec3 blockid = coord / static_cast<int>(config::G_BLOCKSIZE);

	//Create block in partition
	partition.insert(blockid);
}

template<typename ParticleArray, typename ParticleBuffer, typename Partition>
__global__ void build_particle_cell_buckets(uint32_t particle_count, ParticleArray particle_array, ParticleBuffer particle_buffer, Partition partition) {
	const uint32_t particle_id = blockIdx.x * blockDim.x + threadIdx.x;
	if(particle_id >= particle_count) {
		return;
	}

	//Get block id by particle pos
	const ivec3 coord	= get_block_id({particle_array.val(_0, particle_id), particle_array.val(_1, particle_id), particle_array.val(_2, particle_id)}) - 2;
	const ivec3 blockid = coord / static_cast<int>(config::G_BLOCKSIZE);

	//Fetch block number
	auto blockno = partition.query(blockid);

	//Get cell number in block
	int cellno = (coord[0] & config::G_BLOCKMASK) * config::G_BLOCKSIZE * config::G_BLOCKSIZE + (coord[1] & config::G_BLOCKMASK) * config::G_BLOCKSIZE + (coord[2] & config::G_BLOCKMASK);

	//Increase particle count of cell and get id of partzicle in cell
	auto particle_id_in_cell = atomicAdd(particle_buffer.cell_particle_counts + blockno * config::G_BLOCKVOLUME + cellno, 1);

	//If no space is left, don't store the particle
	if(particle_id_in_cell >= config::G_MAX_PARTICLES_IN_CELL) {
		//Reduce count again
		atomicSub(particle_buffer.cell_particle_counts + blockno * config::G_BLOCKVOLUME + cellno, 1);
#if PRINT_CELL_OVERFLOW
		printf("No space left in cell: block(%d), cell(%d)\n", blockno, cellno);
#endif
		return;
	}

	//Insert particle id in cell bucket
	particle_buffer.cellbuckets[blockno * config::G_PARTICLE_NUM_PER_BLOCK + cellno * config::G_MAX_PARTICLES_IN_CELL + particle_id_in_cell] = static_cast<int>(particle_id);//NOTE:Explicit narrowing conversation.
}

__global__ void cell_bucket_to_block(const int* cell_particle_counts, const int* cellbuckets, int* particle_bucket_sizes, int* buckets) {
	const int cellno		  = static_cast<int>(threadIdx.x) & (config::G_BLOCKVOLUME - 1);
	const int particle_counts = cell_particle_counts[blockIdx.x * config::G_BLOCKVOLUME + cellno];

	for(int particle_id_in_cell = 0; particle_id_in_cell < config::G_MAX_PARTICLES_IN_CELL; particle_id_in_cell++) {
		if(particle_id_in_cell < particle_counts) {
			//Each thread gets its index in the blocks bucket
			const int particle_id_in_block = atomic_agg_inc<int>(particle_bucket_sizes + blockIdx.x);

			//Each particle of the source advection buffer (offset + particle id) is assigned to a particle of the current buffer. This should be a 1:1 mapping, though one particle may be mapped to itself or to another particle
			buckets[blockIdx.x * config::G_PARTICLE_NUM_PER_BLOCK + particle_id_in_block] = cellbuckets[blockIdx.x * config::G_PARTICLE_NUM_PER_BLOCK + cellno * config::G_MAX_PARTICLES_IN_CELL + particle_id_in_cell];
		}
		__syncthreads();
	}
}

template<typename Partition>
__global__ void store_triangle_shell_in_bucket(uint32_t particle_count, const TriangleShell triangle_shell, TriangleShellParticleBuffer triangle_shell_particle_buffer, Partition partition) {
	const uint32_t particle_id = blockIdx.x * blockDim.x + threadIdx.x;
	if(particle_id >= particle_count) {
		return;
	}
	
	//Load data
	const auto vertex_data = triangle_shell.ch(_0, 0).ch(_1, 0);
	
	vec3 pos;
	pos[0] = vertex_data.val(_1, particle_id);//pos
	pos[1] = vertex_data.val(_2, particle_id);
	pos[2] = vertex_data.val(_3, particle_id);
	
	//Get block id by particle pos
	const ivec3 coord	= get_block_id(pos.data_arr()) - 2;
	const ivec3 blockid = coord / static_cast<int>(config::G_BLOCKSIZE);

	//Fetch block number
	auto blockno = partition.query(blockid);
	
	//Catch invalid blocks
	if(blockno == -1) {
		//NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg, readability-magic-numbers) Cuda has no other way to print; Numbers are array indices to be printed
		printf("Invalid block: loc(%d, %d, %d)\n", blockid[0], blockid[1], blockid[2]);
		return;
	}

	//Increase particle count of cell and get id of partzicle in cell
	auto particle_id_in_block = atomicAdd(triangle_shell_particle_buffer.particle_bucket_sizes + blockno, 1);

	//If no space is left, don't store the particle
	if(particle_id_in_block >= config::G_PARTICLE_NUM_PER_BLOCK) {
		//Reduce count again
		atomicSub(triangle_shell_particle_buffer.particle_bucket_sizes + blockno, 1);
		printf("No space left in block: block(%d)\n", blockno);
		return;
	}

	//Insert particle id in cell bucket
	triangle_shell_particle_buffer.blockbuckets[blockno * config::G_PARTICLE_NUM_PER_BLOCK + particle_id_in_block] = particle_id;
}

__global__ void compute_bin_capacity(uint32_t block_count, int const* particle_bucket_sizes, int* bin_sizes) {
	const uint32_t blockno = blockIdx.x * blockDim.x + threadIdx.x;
	if(blockno >= block_count) {
		return;
	}

	//Bin capacity is the ceiled bucket size divided by the bin max capacity
	bin_sizes[blockno] = (particle_bucket_sizes[blockno] + config::G_BIN_CAPACITY - 1) / config::G_BIN_CAPACITY;
}

__global__ void init_adv_bucket(const int* particle_bucket_sizes, int* buckets) {
	const int particle_counts = particle_bucket_sizes[blockIdx.x];
	int* bucket				  = buckets + static_cast<size_t>(blockIdx.x) * config::G_PARTICLE_NUM_PER_BLOCK;

	for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < particle_counts; particle_id_in_block += static_cast<int>(blockDim.x)) {
		//Combine offset of 0 with local index in block
		bucket[particle_id_in_block] = (dir_offset({0, 0, 0}) * config::G_PARTICLE_NUM_PER_BLOCK) | particle_id_in_block;
	}
}

template<typename Grid>
__global__ void clear_grid(Grid grid) {
	auto gridblock = grid.ch(_0, blockIdx.x);
	for(int cell_id_in_block = static_cast<int>(threadIdx.x); cell_id_in_block < config::G_BLOCKVOLUME; cell_id_in_block += static_cast<int>(blockDim.x)) {
		gridblock.val_1d(_0, cell_id_in_block) = 0.0f;
		gridblock.val_1d(_1, cell_id_in_block) = 0.0f;
		gridblock.val_1d(_2, cell_id_in_block) = 0.0f;
		gridblock.val_1d(_3, cell_id_in_block) = 0.0f;
	}
}

template<typename Grid>
__global__ void clear_grid_triangle_shell(Grid grid) {
	auto gridblock = grid.ch(_0, blockIdx.x);
	for(int cell_id_in_block = static_cast<int>(threadIdx.x); cell_id_in_block < config::G_BLOCKVOLUME; cell_id_in_block += static_cast<int>(blockDim.x)) {
		gridblock.val_1d(_0, cell_id_in_block) = 0.0f;
	}
}

template<typename Partition>
__global__ void register_neighbor_blocks(uint32_t block_count, Partition partition) {
	const uint32_t blockno = blockIdx.x * blockDim.x + threadIdx.x;
	if(blockno >= block_count) {
		return;
	}

	//Activate neighbour blocks
	const auto blockid = partition.active_keys[blockno];
	for(char i = 0; i < 2; ++i) {
		for(char j = 0; j < 2; ++j) {
			for(char k = 0; k < 2; ++k) {
				partition.insert(ivec3 {blockid[0] + i, blockid[1] + j, blockid[2] + k});
			}
		}
	}
}

template<typename Partition>
__global__ void register_exterior_blocks(uint32_t block_count, Partition partition) {
	const uint32_t blockno = blockIdx.x * blockDim.x + threadIdx.x;
	if(blockno >= block_count) {
		return;
	}

	//Activate exterior blocks
	const auto blockid = partition.active_keys[blockno];
	for(char i = -1; i < 2; ++i) {
		for(char j = -1; j < 2; ++j) {
			for(char k = -1; k < 2; ++k) {
				partition.insert(ivec3 {blockid[0] + i, blockid[1] + j, blockid[2] + k});
			}
		}
	}
}

template<typename Grid, typename Partition>
__global__ void rasterize(uint32_t particle_counts, const ParticleArray particle_array, Grid grid, const Partition partition, Duration dt, float mass, std::array<float, 3> v0) {
	(void) dt;

	const uint32_t particle_id = blockIdx.x * blockDim.x + threadIdx.x;
	if(particle_id >= particle_counts) {
		return;
	}

	//Fetch particle position and velocity
	const vec3 global_pos {particle_array.val(_0, particle_id), particle_array.val(_1, particle_id), particle_array.val(_2, particle_id)};
	const vec3 vel {v0[0], v0[1], v0[2]};

	//vec9 contrib;
	//vec9 c;
	//contrib.set(0.f);
	//c.set(0.f);

	//contrib = (c * mass - contrib * dt.count()) * config::G_D_INV;

	//Calculate grid index
	const ivec3 global_base_index = get_block_id(global_pos.data_arr()) - 1;

	//Calculate position relative to grid cell
	const vec3 local_pos = global_pos - global_base_index * config::G_DX;

	//Calc kernel
	vec<vec3, 3> dws;
	for(int d = 0; d < 3; ++d) {
		dws[d] = bspline_weight(local_pos[d]);
	}

	for(int i = 0; i < 3; ++i) {
		for(int j = 0; j < 3; ++j) {
			for(int k = 0; k < 3; ++k) {
				//Current offset from grid cell
				const ivec3 offset {i, j, k};

				//Global index of grid cell
				const ivec3 global_index = global_base_index + offset;

				ivec3 global_index_masked;
				for(int d = 0; d < 3; ++d) {
					global_index_masked[d] = global_index[d] & config::G_BLOCKMASK;
				}

				//const vec3 xixp		  = offset * config::G_DX - local_pos;

				//Kernel weight
				const float w = dws[0][i] * dws[1][j] * dws[2][k];

				//Mass constribution to the grid cell
				const float wm = mass * w;

				//Fetch block number
				const int blockno = partition.query(ivec3 {static_cast<int>(global_index[0] >> config::G_BLOCKBITS), static_cast<int>(global_index[1] >> config::G_BLOCKBITS), static_cast<int>(global_index[2] >> config::G_BLOCKBITS)});

				//Fetch grid block and initialize values
				auto grid_block = grid.ch(_0, blockno);
				atomicAdd(&grid_block.val(_0, global_index_masked[0], global_index_masked[1], global_index_masked[2]), wm);
				atomicAdd(&grid_block.val(_1, global_index_masked[0], global_index_masked[1], global_index_masked[2]), wm * vel[0]);// + (contrib[0] * xixp[0] + contrib[3] * xixp[1] + contrib[6] * xixp[2]) * w);
				atomicAdd(&grid_block.val(_2, global_index_masked[0], global_index_masked[1], global_index_masked[2]), wm * vel[1]);//  + (contrib[1] * xixp[0] + contrib[4] * xixp[1] + contrib[7] * xixp[2]) * w);
				atomicAdd(&grid_block.val(_3, global_index_masked[0], global_index_masked[1], global_index_masked[2]), wm * vel[2]);//  + (contrib[2] * xixp[0] + contrib[5] * xixp[1] + contrib[8] * xixp[2]) * w);
			}
		}
	}
}

template<typename ParticleArray>
__global__ void array_to_buffer(ParticleArray particle_array, ParticleBuffer<MaterialE::J_FLUID> particle_buffer) {
	const uint32_t blockno	  = blockIdx.x;
	const int particle_counts = particle_buffer.particle_bucket_sizes[blockno];
	const int* bucket		  = particle_buffer.blockbuckets + static_cast<size_t>(blockno) * config::G_PARTICLE_NUM_PER_BLOCK;

	for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < particle_counts; particle_id_in_block += static_cast<int>(blockDim.x)) {
		const int particle_id = bucket[particle_id_in_block];

		auto particle_bin = particle_buffer.ch(_0, particle_buffer.bin_offsets[blockno] + particle_id_in_block / config::G_BIN_CAPACITY);
		/// mass
		particle_bin.val(_0, particle_id_in_block % config::G_BIN_CAPACITY) = particle_buffer.mass;
		/// pos
		particle_bin.val(_1, particle_id_in_block % config::G_BIN_CAPACITY) = particle_array.val(_0, particle_id);
		particle_bin.val(_2, particle_id_in_block % config::G_BIN_CAPACITY) = particle_array.val(_1, particle_id);
		particle_bin.val(_3, particle_id_in_block % config::G_BIN_CAPACITY) = particle_array.val(_2, particle_id);
		/// J
		particle_bin.val(_4, particle_id_in_block % config::G_BIN_CAPACITY) = 1.0f;
	}
}

template<typename ParticleArray>
__global__ void array_to_buffer(ParticleArray particle_array, ParticleBuffer<MaterialE::FIXED_COROTATED> particle_buffer) {
	const uint32_t blockno	  = blockIdx.x;
	const int particle_counts = particle_buffer.particle_bucket_sizes[blockno];
	const int* bucket		  = particle_buffer.blockbuckets + static_cast<size_t>(blockno) * config::G_PARTICLE_NUM_PER_BLOCK;

	for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < particle_counts; particle_id_in_block += static_cast<int>(blockDim.x)) {
		const auto particle_id = bucket[particle_id_in_block];

		auto particle_bin = particle_buffer.ch(_0, particle_buffer.bin_offsets[blockno] + particle_id_in_block / config::G_BIN_CAPACITY);
		/// mass
		particle_bin.val(_0, particle_id_in_block % config::G_BIN_CAPACITY) = particle_buffer.mass;
		/// pos
		particle_bin.val(_1, particle_id_in_block % config::G_BIN_CAPACITY) = particle_array.val(_0, particle_id);
		particle_bin.val(_2, particle_id_in_block % config::G_BIN_CAPACITY) = particle_array.val(_1, particle_id);
		particle_bin.val(_3, particle_id_in_block % config::G_BIN_CAPACITY) = particle_array.val(_2, particle_id);
		/// F
		particle_bin.val(_4, particle_id_in_block % config::G_BIN_CAPACITY)	 = 1.f;
		particle_bin.val(_5, particle_id_in_block % config::G_BIN_CAPACITY)	 = 0.f;
		particle_bin.val(_6, particle_id_in_block % config::G_BIN_CAPACITY)	 = 0.f;
		particle_bin.val(_7, particle_id_in_block % config::G_BIN_CAPACITY)	 = 0.f;
		particle_bin.val(_8, particle_id_in_block % config::G_BIN_CAPACITY)	 = 1.f;
		particle_bin.val(_9, particle_id_in_block % config::G_BIN_CAPACITY)	 = 0.f;
		particle_bin.val(_10, particle_id_in_block % config::G_BIN_CAPACITY)	 = 0.f;
		particle_bin.val(_11, particle_id_in_block % config::G_BIN_CAPACITY) = 0.f;
		particle_bin.val(_12, particle_id_in_block % config::G_BIN_CAPACITY) = 1.f;
	}
}

template<typename ParticleArray>
__global__ void array_to_buffer(ParticleArray particle_array, ParticleBuffer<MaterialE::SAND> particle_buffer) {
	const uint32_t blockno	  = blockIdx.x;
	const int particle_counts = particle_buffer.particle_bucket_sizes[blockno];
	const int* bucket		  = particle_buffer.blockbuckets + static_cast<size_t>(blockno) * config::G_PARTICLE_NUM_PER_BLOCK;

	for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < particle_counts; particle_id_in_block += static_cast<int>(blockDim.x)) {
		const auto particle_id = bucket[particle_id_in_block];

		auto particle_bin = particle_buffer.ch(_0, particle_buffer.bin_offsets[blockno] + particle_id_in_block / config::G_BIN_CAPACITY);
		/// mass
		particle_bin.val(_0, particle_id_in_block % config::G_BIN_CAPACITY) = particle_buffer.mass;
		/// pos
		particle_bin.val(_1, particle_id_in_block % config::G_BIN_CAPACITY) = particle_array.val(_0, particle_id);
		particle_bin.val(_2, particle_id_in_block % config::G_BIN_CAPACITY) = particle_array.val(_1, particle_id);
		particle_bin.val(_3, particle_id_in_block % config::G_BIN_CAPACITY) = particle_array.val(_2, particle_id);
		/// F
		particle_bin.val(_4, particle_id_in_block % config::G_BIN_CAPACITY)	 = 1.f;
		particle_bin.val(_5, particle_id_in_block % config::G_BIN_CAPACITY)	 = 0.f;
		particle_bin.val(_6, particle_id_in_block % config::G_BIN_CAPACITY)	 = 0.f;
		particle_bin.val(_7, particle_id_in_block % config::G_BIN_CAPACITY)	 = 0.f;
		particle_bin.val(_8, particle_id_in_block % config::G_BIN_CAPACITY)	 = 1.f;
		particle_bin.val(_9, particle_id_in_block % config::G_BIN_CAPACITY)	 = 0.f;
		particle_bin.val(_10, particle_id_in_block % config::G_BIN_CAPACITY)	 = 0.f;
		particle_bin.val(_11, particle_id_in_block % config::G_BIN_CAPACITY) = 0.f;
		particle_bin.val(_12, particle_id_in_block % config::G_BIN_CAPACITY) = 1.f;
		/// log_jp
		particle_bin.val(_13, particle_id_in_block % config::G_BIN_CAPACITY) = ParticleBuffer<MaterialE::SAND>::LOG_JP_0;
	}
}

template<typename ParticleArray>
__global__ void array_to_buffer(ParticleArray particle_array, ParticleBuffer<MaterialE::NACC> particle_buffer) {
	const uint32_t blockno	  = blockIdx.x;
	const int particle_counts = particle_buffer.particle_bucket_sizes[blockno];
	const int* bucket		  = particle_buffer.blockbuckets + static_cast<size_t>(blockno) * config::G_PARTICLE_NUM_PER_BLOCK;

	for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < particle_counts; particle_id_in_block += static_cast<int>(blockDim.x)) {
		const auto particle_id = bucket[particle_id_in_block];

		auto particle_bin = particle_buffer.ch(_0, particle_buffer.bin_offsets[blockno] + particle_id_in_block / config::G_BIN_CAPACITY);
		/// mass
		particle_bin.val(_0, particle_id_in_block % config::G_BIN_CAPACITY) = particle_buffer.mass;
		/// pos
		particle_bin.val(_1, particle_id_in_block % config::G_BIN_CAPACITY) = particle_array.val(_0, particle_id);
		particle_bin.val(_2, particle_id_in_block % config::G_BIN_CAPACITY) = particle_array.val(_1, particle_id);
		particle_bin.val(_3, particle_id_in_block % config::G_BIN_CAPACITY) = particle_array.val(_2, particle_id);
		/// F
		particle_bin.val(_4, particle_id_in_block % config::G_BIN_CAPACITY)	 = 1.f;
		particle_bin.val(_5, particle_id_in_block % config::G_BIN_CAPACITY)	 = 0.f;
		particle_bin.val(_6, particle_id_in_block % config::G_BIN_CAPACITY)	 = 0.f;
		particle_bin.val(_7, particle_id_in_block % config::G_BIN_CAPACITY)	 = 0.f;
		particle_bin.val(_8, particle_id_in_block % config::G_BIN_CAPACITY)	 = 1.f;
		particle_bin.val(_9, particle_id_in_block % config::G_BIN_CAPACITY)	 = 0.f;
		particle_bin.val(_10, particle_id_in_block % config::G_BIN_CAPACITY)	 = 0.f;
		particle_bin.val(_11, particle_id_in_block % config::G_BIN_CAPACITY) = 0.f;
		particle_bin.val(_12, particle_id_in_block % config::G_BIN_CAPACITY) = 1.f;
		/// log_jp
		particle_bin.val(_13, particle_id_in_block % config::G_BIN_CAPACITY) = ParticleBuffer<MaterialE::NACC>::LOG_JP_0;
	}
}

template<typename Grid, typename Partition>
__global__ void update_grid_velocity_query_max(uint32_t block_count, Grid grid, Partition partition, Duration dt, float* max_vel) {
	const int boundary_condition   = static_cast<int>(std::floor(config::G_BOUNDARY_CONDITION));
	constexpr int NUM_WARPS		   = config::G_NUM_GRID_BLOCKS_PER_CUDA_BLOCK * config::G_NUM_WARPS_PER_GRID_BLOCK;
	constexpr unsigned ACTIVE_MASK = 0xffffffff;

	static constexpr size_t shared_memory_element_count = NUM_WARPS;
	__shared__ float sh_maxvels[shared_memory_element_count];//NOLINT(modernize-avoid-c-arrays) Cannot declare runtime size shared memory as std::array

	//Fetch block number and id
	const int blockno  = static_cast<int>(blockIdx.x) * config::G_NUM_GRID_BLOCKS_PER_CUDA_BLOCK + static_cast<int>(threadIdx.x) / config::CUDA_WARP_SIZE / config::G_NUM_WARPS_PER_GRID_BLOCK;
	const auto blockid = partition.active_keys[blockno];

	//Check if the block is outside of grid bounds
	const int is_in_bound = ((blockid[0] < boundary_condition || blockid[0] >= config::G_GRID_SIZE - boundary_condition) << 2) | ((blockid[1] < boundary_condition || blockid[1] >= config::G_GRID_SIZE - boundary_condition) << 1) | (blockid[2] < boundary_condition || blockid[2] >= config::G_GRID_SIZE - boundary_condition);

	//Initialize shared memory
	if(threadIdx.x < NUM_WARPS) {
		sh_maxvels[threadIdx.x] = 0.0f;
	}
	__syncthreads();

	/// within-warp computations
	if(blockno < block_count) {
		auto grid_block = grid.ch(_0, blockno);
		for(int cell_id_in_block = static_cast<int>(threadIdx.x % config::CUDA_WARP_SIZE); cell_id_in_block < config::G_BLOCKVOLUME; cell_id_in_block += config::CUDA_WARP_SIZE) {
			const float mass = grid_block.val_1d(_0, cell_id_in_block);
			float vel_sqr	 = 0.0f;
			vec3 vel;
			if(mass > 0.0f) {
				const float mass_inv = 1.f / mass;

				//int i = (cell_id_in_block >> (config::G_BLOCKBITS << 1)) & config::G_BLOCKMASK;
				//int j = (cell_id_in_block >> config::G_BLOCKBITS) & config::G_BLOCKMASK;
				//int k = cell_id_in_block & config::G_BLOCKMASK;

				//Fetch current velocity
				vel[0] = grid_block.val_1d(_1, cell_id_in_block);
				vel[1] = grid_block.val_1d(_2, cell_id_in_block);
				vel[2] = grid_block.val_1d(_3, cell_id_in_block);

				//Update velocity. Set to 0 if outside of bounds
				vel[0] = is_in_bound & 4 ? 0.0f : vel[0] * mass_inv;
				vel[1] = is_in_bound & 2 ? 0.0f : vel[1] * mass_inv;
				vel[1] += config::G_GRAVITY * dt.count();
				vel[2] = is_in_bound & 1 ? 0.0f : vel[2] * mass_inv;
				// if (is_in_bound) ///< sticky
				//  vel.set(0.f);

				//Write back velocity
				grid_block.val_1d(_1, cell_id_in_block) = vel[0];
				grid_block.val_1d(_2, cell_id_in_block) = vel[1];
				grid_block.val_1d(_3, cell_id_in_block) = vel[2];

				//Calculate squared velocity
				vel_sqr += vel[0] * vel[0];
				vel_sqr += vel[1] * vel[1];
				vel_sqr += vel[2] * vel[2];
			}

			//If we have nan values, signalize failure by setting max_vel to inf
			if(isnan(vel_sqr)) {
				vel_sqr = std::numeric_limits<float>::infinity();
			}

			// unsigned activeMask = __ballot_sync(0xffffffff, mv[0] != 0.0f);

			//Calculate max velocity in warp
			for(int iter = 1; iter % config::CUDA_WARP_SIZE; iter <<= 1) {
				float tmp = __shfl_down_sync(ACTIVE_MASK, vel_sqr, iter, config::CUDA_WARP_SIZE);
				if((threadIdx.x % config::CUDA_WARP_SIZE) + iter < config::CUDA_WARP_SIZE) {
					vel_sqr = tmp > vel_sqr ? tmp : vel_sqr;
				}
			}
			//TODO: Ensure threadIdx.x / config::CUDA_WARP_SIZE is smaller than NUM_WARPS
			if(vel_sqr > sh_maxvels[threadIdx.x / config::CUDA_WARP_SIZE] && (threadIdx.x % config::CUDA_WARP_SIZE) == 0) {
				sh_maxvels[threadIdx.x / config::CUDA_WARP_SIZE] = vel_sqr;
			}
		}
	}
	__syncthreads();

	//Calculate global max velocity
	/// various assumptions
	for(int interval = NUM_WARPS >> 1; interval > 0; interval >>= 1) {
		if(threadIdx.x < interval) {
			if(sh_maxvels[static_cast<int>(threadIdx.x) + interval] > sh_maxvels[threadIdx.x]) {
				sh_maxvels[threadIdx.x] = sh_maxvels[static_cast<int>(threadIdx.x) + interval];
			}
		}
		__syncthreads();
	}
	if(threadIdx.x == 0) {
		atomic_max(max_vel, sh_maxvels[0]);
	}
}

//Need this, cause we cannot partially instantiate function templates in current c++ version
struct FetchParticleBufferDataIntermediate {
	float mass;
	std::array<float, 3> pos;
	float J;
};

template<MaterialE MaterialType>
__forceinline__ __device__ void fetch_particle_buffer_data(const ParticleBuffer<MaterialType> particle_buffer, int advection_source_blockno, int source_pidib, FetchParticleBufferDataIntermediate& data);

template<>
__forceinline__ __device__ void fetch_particle_buffer_data<MaterialE::J_FLUID>(const ParticleBuffer<MaterialE::J_FLUID> particle_buffer, int advection_source_blockno, int source_pidib, FetchParticleBufferDataIntermediate& data) {
	auto source_particle_bin = particle_buffer.ch(_0, advection_source_blockno);
	data.mass				 = source_particle_bin.val(_0, source_pidib % config::G_BIN_CAPACITY);
	data.pos[0]				 = source_particle_bin.val(_1, source_pidib % config::G_BIN_CAPACITY);
	data.pos[1]				 = source_particle_bin.val(_2, source_pidib % config::G_BIN_CAPACITY);
	data.pos[2]				 = source_particle_bin.val(_3, source_pidib % config::G_BIN_CAPACITY);
	data.J					 = source_particle_bin.val(_4, source_pidib % config::G_BIN_CAPACITY);
}

template<>
__forceinline__ __device__ void fetch_particle_buffer_data<MaterialE::FIXED_COROTATED>(const ParticleBuffer<MaterialE::FIXED_COROTATED> particle_buffer, int advection_source_blockno, int source_pidib, FetchParticleBufferDataIntermediate& data) {
	auto source_particle_bin = particle_buffer.ch(_0, advection_source_blockno);
	data.mass				 = source_particle_bin.val(_0, source_pidib % config::G_BIN_CAPACITY);
	data.pos[0]				 = source_particle_bin.val(_1, source_pidib % config::G_BIN_CAPACITY);
	data.pos[1]				 = source_particle_bin.val(_2, source_pidib % config::G_BIN_CAPACITY);
	data.pos[2]				 = source_particle_bin.val(_3, source_pidib % config::G_BIN_CAPACITY);
}

template<>
__forceinline__ __device__ void fetch_particle_buffer_data<MaterialE::SAND>(const ParticleBuffer<MaterialE::SAND> particle_buffer, int advection_source_blockno, int source_pidib, FetchParticleBufferDataIntermediate& data) {
	auto source_particle_bin = particle_buffer.ch(_0, advection_source_blockno);
	data.mass				 = source_particle_bin.val(_0, source_pidib % config::G_BIN_CAPACITY);
	data.pos[0]				 = source_particle_bin.val(_1, source_pidib % config::G_BIN_CAPACITY);
	data.pos[1]				 = source_particle_bin.val(_2, source_pidib % config::G_BIN_CAPACITY);
	data.pos[2]				 = source_particle_bin.val(_3, source_pidib % config::G_BIN_CAPACITY);
}

template<>
__forceinline__ __device__ void fetch_particle_buffer_data<MaterialE::NACC>(const ParticleBuffer<MaterialE::NACC> particle_buffer, int advection_source_blockno, int source_pidib, FetchParticleBufferDataIntermediate& data) {
	auto source_particle_bin = particle_buffer.ch(_0, advection_source_blockno);
	data.mass				 = source_particle_bin.val(_0, source_pidib % config::G_BIN_CAPACITY);
	data.pos[0]				 = source_particle_bin.val(_1, source_pidib % config::G_BIN_CAPACITY);
	data.pos[1]				 = source_particle_bin.val(_2, source_pidib % config::G_BIN_CAPACITY);
	data.pos[2]				 = source_particle_bin.val(_3, source_pidib % config::G_BIN_CAPACITY);
}

//Need this, cause we cannot partially instantiate function templates in current c++ version
struct StoreParticleDataIntermediate {
	float mass;
	std::array<float, 3> pos;
	float J;
	std::array<float, 9> F;
	float log_jp;
};

template<MaterialE MaterialType>
__forceinline__ __device__ void store_particle_data(const ParticleBuffer<MaterialType> next_particle_buffer, int src_blockno, int particle_id_in_block, StoreParticleDataIntermediate& data);

template<>
__forceinline__ __device__ void store_particle_data<MaterialE::J_FLUID>(const ParticleBuffer<MaterialE::J_FLUID> next_particle_buffer, int src_blockno, int particle_id_in_block, StoreParticleDataIntermediate& data) {
	//Write back particle data
	{
		auto particle_bin													= next_particle_buffer.ch(_0, next_particle_buffer.bin_offsets[src_blockno] + particle_id_in_block / config::G_BIN_CAPACITY);
		particle_bin.val(_0, particle_id_in_block % config::G_BIN_CAPACITY) = data.mass;
		particle_bin.val(_1, particle_id_in_block % config::G_BIN_CAPACITY) = data.pos[0];
		particle_bin.val(_2, particle_id_in_block % config::G_BIN_CAPACITY) = data.pos[1];
		particle_bin.val(_3, particle_id_in_block % config::G_BIN_CAPACITY) = data.pos[2];
		particle_bin.val(_4, particle_id_in_block % config::G_BIN_CAPACITY) = data.J;
	}
}

template<>
__forceinline__ __device__ void store_particle_data(const ParticleBuffer<MaterialE::FIXED_COROTATED> next_particle_buffer, int src_blockno, int particle_id_in_block, StoreParticleDataIntermediate& data) {
	{
		auto particle_bin													 = next_particle_buffer.ch(_0, next_particle_buffer.bin_offsets[src_blockno] + particle_id_in_block / config::G_BIN_CAPACITY);
		particle_bin.val(_0, particle_id_in_block % config::G_BIN_CAPACITY) = data.mass;
		particle_bin.val(_1, particle_id_in_block % config::G_BIN_CAPACITY)	 = data.pos[0];
		particle_bin.val(_2, particle_id_in_block % config::G_BIN_CAPACITY)	 = data.pos[1];
		particle_bin.val(_3, particle_id_in_block % config::G_BIN_CAPACITY)	 = data.pos[2];
		particle_bin.val(_4, particle_id_in_block % config::G_BIN_CAPACITY)	 = data.F[0];
		particle_bin.val(_5, particle_id_in_block % config::G_BIN_CAPACITY)	 = data.F[1];
		particle_bin.val(_6, particle_id_in_block % config::G_BIN_CAPACITY)	 = data.F[2];
		particle_bin.val(_7, particle_id_in_block % config::G_BIN_CAPACITY)	 = data.F[3];
		particle_bin.val(_8, particle_id_in_block % config::G_BIN_CAPACITY)	 = data.F[4];
		particle_bin.val(_9, particle_id_in_block % config::G_BIN_CAPACITY)	 = data.F[5];
		particle_bin.val(_10, particle_id_in_block % config::G_BIN_CAPACITY)	 = data.F[6];
		particle_bin.val(_11, particle_id_in_block % config::G_BIN_CAPACITY) = data.F[7];
		particle_bin.val(_12, particle_id_in_block % config::G_BIN_CAPACITY) = data.F[8];
	}
}

template<>
__forceinline__ __device__ void store_particle_data(const ParticleBuffer<MaterialE::SAND> next_particle_buffer, int src_blockno, int particle_id_in_block, StoreParticleDataIntermediate& data) {
	{
		auto particle_bin													 = next_particle_buffer.ch(_0, next_particle_buffer.bin_offsets[src_blockno] + particle_id_in_block / config::G_BIN_CAPACITY);
		particle_bin.val(_0, particle_id_in_block % config::G_BIN_CAPACITY) = data.mass;
		particle_bin.val(_1, particle_id_in_block % config::G_BIN_CAPACITY)	 = data.pos[0];
		particle_bin.val(_2, particle_id_in_block % config::G_BIN_CAPACITY)	 = data.pos[1];
		particle_bin.val(_3, particle_id_in_block % config::G_BIN_CAPACITY)	 = data.pos[2];
		particle_bin.val(_4, particle_id_in_block % config::G_BIN_CAPACITY)	 = data.F[0];
		particle_bin.val(_5, particle_id_in_block % config::G_BIN_CAPACITY)	 = data.F[1];
		particle_bin.val(_6, particle_id_in_block % config::G_BIN_CAPACITY)	 = data.F[2];
		particle_bin.val(_7, particle_id_in_block % config::G_BIN_CAPACITY)	 = data.F[3];
		particle_bin.val(_8, particle_id_in_block % config::G_BIN_CAPACITY)	 = data.F[4];
		particle_bin.val(_9, particle_id_in_block % config::G_BIN_CAPACITY)	 = data.F[5];
		particle_bin.val(_10, particle_id_in_block % config::G_BIN_CAPACITY)	 = data.F[6];
		particle_bin.val(_11, particle_id_in_block % config::G_BIN_CAPACITY) = data.F[7];
		particle_bin.val(_12, particle_id_in_block % config::G_BIN_CAPACITY) = data.F[8];
		particle_bin.val(_13, particle_id_in_block % config::G_BIN_CAPACITY) = data.log_jp;
	}
}

template<>
__forceinline__ __device__ void store_particle_data(const ParticleBuffer<MaterialE::NACC> next_particle_buffer, int src_blockno, int particle_id_in_block, StoreParticleDataIntermediate& data) {
	{
		auto particle_bin													 = next_particle_buffer.ch(_0, next_particle_buffer.bin_offsets[src_blockno] + particle_id_in_block / config::G_BIN_CAPACITY);
		particle_bin.val(_0, particle_id_in_block % config::G_BIN_CAPACITY) = data.mass;
		particle_bin.val(_1, particle_id_in_block % config::G_BIN_CAPACITY)	 = data.pos[0];
		particle_bin.val(_2, particle_id_in_block % config::G_BIN_CAPACITY)	 = data.pos[1];
		particle_bin.val(_3, particle_id_in_block % config::G_BIN_CAPACITY)	 = data.pos[2];
		particle_bin.val(_4, particle_id_in_block % config::G_BIN_CAPACITY)	 = data.F[0];
		particle_bin.val(_5, particle_id_in_block % config::G_BIN_CAPACITY)	 = data.F[1];
		particle_bin.val(_6, particle_id_in_block % config::G_BIN_CAPACITY)	 = data.F[2];
		particle_bin.val(_7, particle_id_in_block % config::G_BIN_CAPACITY)	 = data.F[3];
		particle_bin.val(_8, particle_id_in_block % config::G_BIN_CAPACITY)	 = data.F[4];
		particle_bin.val(_9, particle_id_in_block % config::G_BIN_CAPACITY)	 = data.F[5];
		particle_bin.val(_10, particle_id_in_block % config::G_BIN_CAPACITY)	 = data.F[6];
		particle_bin.val(_11, particle_id_in_block % config::G_BIN_CAPACITY) = data.F[7];
		particle_bin.val(_12, particle_id_in_block % config::G_BIN_CAPACITY) = data.F[8];
		particle_bin.val(_13, particle_id_in_block % config::G_BIN_CAPACITY) = data.log_jp;
	}
}

//Need this, cause we cannot partially instantiate function templates in current c++ version
struct CalculateContributionAndStoreParticleDataIntermediate {
	float mass;
	std::array<float, 3> pos;
	float J;
};

template<MaterialE MaterialType>
__forceinline__ __device__ void calculate_contribution_and_store_particle_data(const ParticleBuffer<MaterialType> particle_buffer, const ParticleBuffer<MaterialType> next_particle_buffer, int advection_source_blockno, int source_pidib, int src_blockno, int particle_id_in_block, Duration dt, const std::array<float, 9>& A, std::array<float, 9>& contrib, CalculateContributionAndStoreParticleDataIntermediate& data);

template<>
__forceinline__ __device__ void calculate_contribution_and_store_particle_data<MaterialE::J_FLUID>(const ParticleBuffer<MaterialE::J_FLUID> particle_buffer, const ParticleBuffer<MaterialE::J_FLUID> next_particle_buffer, int advection_source_blockno, int source_pidib, int src_blockno, int particle_id_in_block, Duration dt, const std::array<float, 9>& A, std::array<float, 9>& contrib, CalculateContributionAndStoreParticleDataIntermediate& data) {
	(void) advection_source_blockno;
	(void) source_pidib;

	//Update determinante of deformation gradiant
	//Divergence of velocity multiplied with time and transfered to global space
	data.J += (A[0] + A[4] + A[8]) * dt.count() * config::G_D_INV * data.J;

	//Too low is bad. clamp to 0.1
	//TODO: Maybe make this 0.1 a parameter
	if(data.J < 0.1) {
		data.J = 0.1;
	}

	//TODO: What is calculated here?
	{
		float voln	   = data.J * particle_buffer.volume;
		float pressure = particle_buffer.bulk * (powf(data.J, -particle_buffer.gamma) - 1.f);
		//? - stress; stress = pressure * identity;
		{
			contrib[0] = ((A[0] + A[0]) * config::G_D_INV * particle_buffer.viscosity - pressure) * voln;
			contrib[1] = (A[1] + A[3]) * config::G_D_INV * particle_buffer.viscosity * voln;
			contrib[2] = (A[2] + A[6]) * config::G_D_INV * particle_buffer.viscosity * voln;

			contrib[3] = (A[3] + A[1]) * config::G_D_INV * particle_buffer.viscosity * voln;
			contrib[4] = ((A[4] + A[4]) * config::G_D_INV * particle_buffer.viscosity - pressure) * voln;
			contrib[5] = (A[5] + A[7]) * config::G_D_INV * particle_buffer.viscosity * voln;

			contrib[6] = (A[6] + A[2]) * config::G_D_INV * particle_buffer.viscosity * voln;
			contrib[7] = (A[7] + A[5]) * config::G_D_INV * particle_buffer.viscosity * voln;
			contrib[8] = ((A[8] + A[8]) * config::G_D_INV * particle_buffer.viscosity - pressure) * voln;
		}
	}

	//Write back particle data
	
	StoreParticleDataIntermediate store_particle_data_tmp = {};
	store_particle_data_tmp.mass = data.mass;
	store_particle_data_tmp.pos													= data.pos;
	store_particle_data_tmp.J														= data.J;

	store_particle_data<MaterialE::J_FLUID>(next_particle_buffer, src_blockno, particle_id_in_block, store_particle_data_tmp);
}

template<>
__forceinline__ __device__ void calculate_contribution_and_store_particle_data<MaterialE::FIXED_COROTATED>(const ParticleBuffer<MaterialE::FIXED_COROTATED> particle_buffer, const ParticleBuffer<MaterialE::FIXED_COROTATED> next_particle_buffer, int advection_source_blockno, int source_pidib, int src_blockno, int particle_id_in_block, Duration dt, const std::array<float, 9>& A, std::array<float, 9>& contrib, CalculateContributionAndStoreParticleDataIntermediate& data) {
	vec3x3 dws;
//((d & 0x3) != 0 ? 0.f : 1.f) is identity matrix
#pragma unroll 9
	for(int d = 0; d < 9; ++d) {
		dws.val(d) = A[d] * dt.count() * config::G_D_INV + ((d & 0x3) != 0 ? 0.f : 1.f);
	}

	{
		vec9 F;
		auto source_particle_bin = particle_buffer.ch(_0, advection_source_blockno);
		contrib[0]				 = source_particle_bin.val(_4, source_pidib % config::G_BIN_CAPACITY);
		contrib[1]				 = source_particle_bin.val(_5, source_pidib % config::G_BIN_CAPACITY);
		contrib[2]				 = source_particle_bin.val(_6, source_pidib % config::G_BIN_CAPACITY);
		contrib[3]				 = source_particle_bin.val(_7, source_pidib % config::G_BIN_CAPACITY);
		contrib[4]				 = source_particle_bin.val(_8, source_pidib % config::G_BIN_CAPACITY);
		contrib[5]				 = source_particle_bin.val(_9, source_pidib % config::G_BIN_CAPACITY);
		contrib[6]				 = source_particle_bin.val(_10, source_pidib % config::G_BIN_CAPACITY);
		contrib[7]				 = source_particle_bin.val(_11, source_pidib % config::G_BIN_CAPACITY);
		contrib[8]				 = source_particle_bin.val(_12, source_pidib % config::G_BIN_CAPACITY);
		matrix_matrix_multiplication_3d(dws.data_arr(), contrib, F.data_arr());
		
		StoreParticleDataIntermediate store_particle_data_tmp = {};
		store_particle_data_tmp.mass = data.mass;
		store_particle_data_tmp.pos													= data.pos;
		store_particle_data_tmp.F														= F.data_arr();

		store_particle_data<MaterialE::FIXED_COROTATED>(next_particle_buffer, src_blockno, particle_id_in_block, store_particle_data_tmp);

		ComputeStressIntermediate compute_stress_tmp = {};
		compute_stress<float, MaterialE::FIXED_COROTATED>(particle_buffer.volume, particle_buffer.mu, particle_buffer.lambda, F.data_arr(), contrib, compute_stress_tmp);
	}
}

template<>
__forceinline__ __device__ void calculate_contribution_and_store_particle_data<MaterialE::SAND>(const ParticleBuffer<MaterialE::SAND> particle_buffer, const ParticleBuffer<MaterialE::SAND> next_particle_buffer, int advection_source_blockno, int source_pidib, int src_blockno, int particle_id_in_block, Duration dt, const std::array<float, 9>& A, std::array<float, 9>& contrib, CalculateContributionAndStoreParticleDataIntermediate& data) {
	vec3x3 dws;
//((d & 0x3) != 0 ? 0.f : 1.f) is identity matrix
#pragma unroll 9
	for(int d = 0; d < 9; ++d) {
		dws.val(d) = A[d] * dt.count() * config::G_D_INV + ((d & 0x3) != 0 ? 0.f : 1.f);
	}

	{
		vec9 F;
		float log_jp;
		auto source_particle_bin = particle_buffer.ch(_0, advection_source_blockno);
		contrib[0]				 = source_particle_bin.val(_4, source_pidib % config::G_BIN_CAPACITY);
		contrib[1]				 = source_particle_bin.val(_5, source_pidib % config::G_BIN_CAPACITY);
		contrib[2]				 = source_particle_bin.val(_6, source_pidib % config::G_BIN_CAPACITY);
		contrib[3]				 = source_particle_bin.val(_7, source_pidib % config::G_BIN_CAPACITY);
		contrib[4]				 = source_particle_bin.val(_8, source_pidib % config::G_BIN_CAPACITY);
		contrib[5]				 = source_particle_bin.val(_9, source_pidib % config::G_BIN_CAPACITY);
		contrib[6]				 = source_particle_bin.val(_10, source_pidib % config::G_BIN_CAPACITY);
		contrib[7]				 = source_particle_bin.val(_11, source_pidib % config::G_BIN_CAPACITY);
		contrib[8]				 = source_particle_bin.val(_12, source_pidib % config::G_BIN_CAPACITY);
		log_jp					 = source_particle_bin.val(_13, source_pidib % config::G_BIN_CAPACITY);

		matrix_matrix_multiplication_3d(dws.data_arr(), contrib, F.data_arr());
		ComputeStressIntermediate compute_stress_tmp = {};
		compute_stress_tmp.cohesion					 = particle_buffer.cohesion;
		compute_stress_tmp.beta						 = particle_buffer.beta;
		compute_stress_tmp.yield_surface			 = particle_buffer.yield_surface;
		compute_stress_tmp.volume_correction		 = particle_buffer.volume_correction;
		compute_stress_tmp.log_jp					 = log_jp;
		compute_stress<float, MaterialE::SAND>(particle_buffer.volume, particle_buffer.mu, particle_buffer.lambda, F.data_arr(), contrib, compute_stress_tmp);
		log_jp = compute_stress_tmp.log_jp;
		
		StoreParticleDataIntermediate store_particle_data_tmp = {};
		store_particle_data_tmp.mass = data.mass;
		store_particle_data_tmp.pos													= data.pos;
		store_particle_data_tmp.F														= F.data_arr();
		store_particle_data_tmp.log_jp														= log_jp;

		store_particle_data<MaterialE::SAND>(next_particle_buffer, src_blockno, particle_id_in_block, store_particle_data_tmp);
	}
}

template<>
__forceinline__ __device__ void calculate_contribution_and_store_particle_data<MaterialE::NACC>(const ParticleBuffer<MaterialE::NACC> particle_buffer, const ParticleBuffer<MaterialE::NACC> next_particle_buffer, int advection_source_blockno, int source_pidib, int src_blockno, int particle_id_in_block, Duration dt, const std::array<float, 9>& A, std::array<float, 9>& contrib, CalculateContributionAndStoreParticleDataIntermediate& data) {
	vec3x3 dws;
//((d & 0x3) != 0 ? 0.f : 1.f) is identity matrix
#pragma unroll 9
	for(int d = 0; d < 9; ++d) {
		dws.val(d) = A[d] * dt.count() * config::G_D_INV + ((d & 0x3) != 0 ? 0.f : 1.f);
	}

	{
		vec9 F;
		float log_jp;
		auto source_particle_bin = particle_buffer.ch(_0, advection_source_blockno);
		contrib[0]				 = source_particle_bin.val(_4, source_pidib % config::G_BIN_CAPACITY);
		contrib[1]				 = source_particle_bin.val(_5, source_pidib % config::G_BIN_CAPACITY);
		contrib[2]				 = source_particle_bin.val(_6, source_pidib % config::G_BIN_CAPACITY);
		contrib[3]				 = source_particle_bin.val(_7, source_pidib % config::G_BIN_CAPACITY);
		contrib[4]				 = source_particle_bin.val(_8, source_pidib % config::G_BIN_CAPACITY);
		contrib[5]				 = source_particle_bin.val(_9, source_pidib % config::G_BIN_CAPACITY);
		contrib[6]				 = source_particle_bin.val(_10, source_pidib % config::G_BIN_CAPACITY);
		contrib[7]				 = source_particle_bin.val(_11, source_pidib % config::G_BIN_CAPACITY);
		contrib[8]				 = source_particle_bin.val(_12, source_pidib % config::G_BIN_CAPACITY);
		log_jp					 = source_particle_bin.val(_13, source_pidib % config::G_BIN_CAPACITY);

		matrix_matrix_multiplication_3d(dws.data_arr(), contrib, F.data_arr());
		ComputeStressIntermediate compute_stress_tmp = {};
		compute_stress_tmp.bm						 = particle_buffer.bm;
		compute_stress_tmp.xi						 = particle_buffer.xi;
		compute_stress_tmp.beta						 = particle_buffer.beta;
		compute_stress_tmp.msqr						 = particle_buffer.msqr;
		compute_stress_tmp.hardening_on				 = particle_buffer.hardening_on;
		compute_stress_tmp.log_jp					 = log_jp;
		compute_stress<float, MaterialE::NACC>(particle_buffer.volume, particle_buffer.mu, particle_buffer.lambda, F.data_arr(), contrib, compute_stress_tmp);
		log_jp = compute_stress_tmp.log_jp;
		
		StoreParticleDataIntermediate store_particle_data_tmp = {};
		store_particle_data_tmp.mass = data.mass;
		store_particle_data_tmp.pos													= data.pos;
		store_particle_data_tmp.F														= F.data_arr();
		store_particle_data_tmp.log_jp														= log_jp;

		store_particle_data<MaterialE::NACC>(next_particle_buffer, src_blockno, particle_id_in_block, store_particle_data_tmp);
	}
}

template<typename Partition, typename Grid, MaterialE MaterialType>
__global__ void g2p2g(Duration dt, Duration new_dt, const ParticleBuffer<MaterialType> particle_buffer, ParticleBuffer<MaterialType> next_particle_buffer, const Partition prev_partition, Partition partition, const Grid grid, Grid next_grid) {
	static constexpr uint64_t NUM_VI_PER_BLOCK = static_cast<uint64_t>(config::G_BLOCKVOLUME) * 3;
	static constexpr uint64_t NUM_VI_IN_ARENA  = NUM_VI_PER_BLOCK << 3;

	static constexpr uint64_t NUM_M_VI_PER_BLOCK = static_cast<uint64_t>(config::G_BLOCKVOLUME) * 4;
	static constexpr uint64_t NUM_M_VI_IN_ARENA	 = NUM_M_VI_PER_BLOCK << 3;

	static constexpr unsigned ARENAMASK = (config::G_BLOCKSIZE << 1) - 1;
	static constexpr unsigned ARENABITS = config::G_BLOCKBITS + 1;

	using ViArena	  = std::array<std::array<std::array<std::array<float, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, 3>*;
	using ViArenaRef  = std::array<std::array<std::array<std::array<float, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, 3>&;
	using MViArena	  = std::array<std::array<std::array<std::array<float, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, 4>*;
	using MViArenaRef = std::array<std::array<std::array<std::array<float, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, 4>&;

	static constexpr size_t shared_memory_size = static_cast<size_t>((3 + 4) * (config::G_BLOCKSIZE << 1) * (config::G_BLOCKSIZE << 1) * (config::G_BLOCKSIZE << 1)) * sizeof(float);
	__shared__ char shmem[shared_memory_size];//NOLINT(modernize-avoid-c-arrays, readability-redundant-declaration) Cannot declare runtime size shared memory as std::array; extern has different meaning here

	ViArenaRef __restrict__ g2pbuffer  = *static_cast<ViArena>(static_cast<void*>(static_cast<char*>(shmem)));
	MViArenaRef __restrict__ p2gbuffer = *static_cast<MViArena>(static_cast<void*>(static_cast<char*>(shmem) + NUM_VI_IN_ARENA * sizeof(float)));

	//Layout of buffers: 1 dimension is channel. other dimensions range [0, 3] contains current block, range [4, 7] contains next block
	//The first cell of the next block is handled by the current block only (but may receive values of particles traveling out of next block in negative direction)

	const int src_blockno		   = static_cast<int>(blockIdx.x);
	const auto blockid			   = partition.active_keys[blockIdx.x];
	const int particle_bucket_size = next_particle_buffer.particle_bucket_sizes[src_blockno];
	
	//If we have no particles in the bucket return
	if(particle_bucket_size == 0) {
		return;
	}

	//Load data from grid to shared memory
	for(int base = static_cast<int>(threadIdx.x); base < NUM_VI_IN_ARENA; base += static_cast<int>(blockDim.x)) {
		const char local_block_id = static_cast<char>(base / NUM_VI_PER_BLOCK);
		const auto blockno		  = partition.query(ivec3 {blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0), blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0), blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
		const auto grid_block	  = grid.ch(_0, blockno);
		int channelid			  = static_cast<int>(base % NUM_VI_PER_BLOCK);
		const char c			  = static_cast<char>(channelid & 0x3f);

		const char cz = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		const char cy = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		const char cx = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		float val;
		if(channelid == 0) {
			val = grid_block.val_1d(_1, c);
		} else if(channelid == 1) {
			val = grid_block.val_1d(_2, c);
		} else {
			val = grid_block.val_1d(_3, c);
		}

		g2pbuffer[channelid][static_cast<size_t>(static_cast<size_t>(cx) + (local_block_id & 4 ? config::G_BLOCKSIZE : 0))][static_cast<size_t>(static_cast<size_t>(cy) + (local_block_id & 2 ? config::G_BLOCKSIZE : 0))][static_cast<size_t>(static_cast<size_t>(cz) + (local_block_id & 1 ? config::G_BLOCKSIZE : 0))] = val;
	}
	__syncthreads();

	//Clear return buffer
	for(int base = static_cast<int>(threadIdx.x); base < NUM_M_VI_IN_ARENA; base += static_cast<int>(blockDim.x)) {
		int loc		 = base;
		const char z = static_cast<char>(loc & ARENAMASK);
		loc >>= ARENABITS;

		const char y = static_cast<char>(loc & ARENAMASK);
		loc >>= ARENABITS;

		const char x = static_cast<char>(loc & ARENAMASK);

		p2gbuffer[loc >> ARENABITS][x][y][z] = 0.0f;
	}
	__syncthreads();

	//Perform update
	for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < particle_bucket_size; particle_id_in_block += static_cast<int>(blockDim.x)) {
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
			const int advection_source_blockno_from_partition = prev_partition.query(global_advection_index);

			//Get block number in particle bins
			advection_source_blockno = particle_buffer.bin_offsets[advection_source_blockno_from_partition] + source_pidib / config::G_BIN_CAPACITY;
		}

		//Fetch position and determinant of deformation gradient
		FetchParticleBufferDataIntermediate fetch_particle_buffer_tmp = {};
		fetch_particle_buffer_data<MaterialType>(particle_buffer, advection_source_blockno, source_pidib, fetch_particle_buffer_tmp);
		const float mass = fetch_particle_buffer_tmp.mass;
		vec3 pos {fetch_particle_buffer_tmp.pos[0], fetch_particle_buffer_tmp.pos[1], fetch_particle_buffer_tmp.pos[2]};
		float J	 = fetch_particle_buffer_tmp.J;

		//Get position of grid cell
		ivec3 global_base_index = get_block_id(pos.data_arr()) - 1;

		//Get position relative to grid cell
		vec3 local_pos = pos - global_base_index * config::G_DX;

		//Save global_base_index
		ivec3 base_index = global_base_index;

		//Calculate weights and mask global index
		vec3x3 dws;
#pragma unroll 3
		for(int dd = 0; dd < 3; ++dd) {
			const vec3 weight = bspline_weight(local_pos[dd]);
			dws(dd, 0)		  = weight[0];
			dws(dd, 1)		  = weight[1];
			dws(dd, 2)		  = weight[2];

			//Calculate (modulo (config::G_BLOCKMASK + 1)) + 1 of index (== (..., -4, -3, -2, -1, 0, 1, 2, 3, 4, ...) -> (..., 4, 1, 2, 3, 4, 1, 2, 3, 4, ...))
			global_base_index[dd] = ((base_index[dd] - 1) & config::G_BLOCKMASK) + 1;
		}

		//Calculate particle velocity and APIC affine matrix
		//v_p = sum(i, weight_i_p * v_i)
		//A = sum(i, weight_i_p * v_i * (x_i - x_p))
		vec3 vel;
		vel.set(0.f);
		vec9 A;//affine state
		A.set(0.f);
#pragma unroll 3
		for(char i = 0; i < 3; i++) {
#pragma unroll 3
			for(char j = 0; j < 3; j++) {
#pragma unroll 3
				for(char k = 0; k < 3; k++) {
					//(x_i - x_p)
					const vec3 xixp = vec3 {static_cast<float>(i), static_cast<float>(j), static_cast<float>(k)} * config::G_DX - local_pos;

					//Weight
					const float W = dws(0, i) * dws(1, j) * dws(2, k);

					//Velocity of grid cell
					const vec3 vi {g2pbuffer[0][static_cast<size_t>(static_cast<size_t>(global_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(global_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(global_base_index[2]) + k)], g2pbuffer[1][static_cast<size_t>(static_cast<size_t>(global_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(global_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(global_base_index[2]) + k)], g2pbuffer[2][static_cast<size_t>(static_cast<size_t>(global_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(global_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(global_base_index[2]) + k)]};

					//Calculate velocity
					vel += W * vi;

					//Calculate APIC affine matrix
					A[0] += W * vi[0] * xixp[0];
					A[1] += W * vi[1] * xixp[0];
					A[2] += W * vi[2] * xixp[0];
					A[3] += W * vi[0] * xixp[1];
					A[4] += W * vi[1] * xixp[1];
					A[5] += W * vi[2] * xixp[1];
					A[6] += W * vi[0] * xixp[2];
					A[7] += W * vi[1] * xixp[2];
					A[8] += W * vi[2] * xixp[2];
				}
			}
		}
		
		if(
			   (blockid[0] == 32 && blockid[1] == 24 && blockid[2] == 32)
		){
			printf("C %d %d %d # %d %d # %f %f %f - ", (base_index[0] - 1) / static_cast<int>(config::G_BLOCKSIZE), (base_index[1] - 1) / static_cast<int>(config::G_BLOCKSIZE), (base_index[2] - 1) / static_cast<int>(config::G_BLOCKSIZE), advection_source_blockno, source_pidib, pos[0], pos[1], pos[2]);
		}
		
		//Update particle position
		pos += vel * dt.count();

		CalculateContributionAndStoreParticleDataIntermediate store_particle_buffer_tmp = {};
		store_particle_buffer_tmp.mass = mass;
		store_particle_buffer_tmp.pos													= pos.data_arr();
		store_particle_buffer_tmp.J														= J;

		vec9 contrib;
		calculate_contribution_and_store_particle_data<MaterialType>(particle_buffer, next_particle_buffer, advection_source_blockno, source_pidib, src_blockno, particle_id_in_block, dt, A.data_arr(), contrib.data_arr(), store_particle_buffer_tmp);

		//Update momentum?
		//Multiply A with mass to complete it. Then subtract current momentum?
		//C = A * D^-1
		contrib = (A * mass - contrib * new_dt.count()) * config::G_D_INV;

		//Calculate grid index after movement
		ivec3 new_global_base_index = get_block_id(pos.data_arr()) - 1;

		//Update local position
		local_pos = pos - new_global_base_index * config::G_DX;

		//Store index and movement direction
		{
			//Calculate direction offset
			const int dirtag = dir_offset(((base_index - 1) / static_cast<int>(config::G_BLOCKSIZE) - (new_global_base_index - 1) / static_cast<int>(config::G_BLOCKSIZE)).data_arr());

			//Store particle in new block
			next_particle_buffer.add_advection(partition, new_global_base_index - 1, dirtag, particle_id_in_block);
			// partition.add_advection(new_global_base_index - 1, dirtag, particle_id_in_block);
		}

		//Calculate weights and mask global index
#pragma unroll 3
		for(char dd = 0; dd < 3; ++dd) {
			const vec3 weight = bspline_weight(local_pos[dd]);
			dws(dd, 0)		  = weight[0];
			dws(dd, 1)		  = weight[1];
			dws(dd, 2)		  = weight[2];

			//Calculate (modulo (config::G_BLOCKMASK + 1)) + 1 of index (== (..., -4, -3, -2, -1, 0, 1, 2, 3, 4, ...) -> (..., 4, 1, 2, 3, 4, 1, 2, 3, 4, ...)) and add offset (may not be out of range max [-4, 4] min [-1, 1])
			new_global_base_index[dd] = (((base_index[dd] - 1) & config::G_BLOCKMASK) + 1) + (new_global_base_index[dd] - base_index[dd]);
		}

		//Dim of p2gbuffer is (4, 8, 8, 8). So if values are too big, discard whole particle
		if(new_global_base_index[0] < 0 || new_global_base_index[1] < 0 || new_global_base_index[2] < 0 || new_global_base_index[0] + 2 >= 8 || new_global_base_index[1] + 2 >= 8 || new_global_base_index[2] + 2 >= 8) {
			//NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg) Cuda has no other way to print; Numbers are array indices to be printed
			printf("new_global_base_index out of range: %d %d %d\n", new_global_base_index[0], new_global_base_index[1], new_global_base_index[2]);
			return;
		}

		//Calculate new gird momentum and mass
		//m_i * v_i = sum(p, w_i_p * m_p * vel_p + ?)
#pragma unroll 3
		for(char i = 0; i < 3; i++) {
#pragma unroll 3
			for(char j = 0; j < 3; j++) {
#pragma unroll 3
				for(char k = 0; k < 3; k++) {
					pos			  = vec3 {static_cast<float>(i), static_cast<float>(j), static_cast<float>(k)} * config::G_DX - local_pos;
					const float W = dws(0, i) * dws(1, j) * dws(2, k);
					const auto wm = mass * W;

					atomicAdd(&p2gbuffer[0][static_cast<size_t>(static_cast<size_t>(new_global_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(new_global_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(new_global_base_index[2]) + k)], wm);
					atomicAdd(&p2gbuffer[1][static_cast<size_t>(static_cast<size_t>(new_global_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(new_global_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(new_global_base_index[2]) + k)], wm * vel[0] + (contrib[0] * pos[0] + contrib[3] * pos[1] + contrib[6] * pos[2]) * W);
					atomicAdd(&p2gbuffer[2][static_cast<size_t>(static_cast<size_t>(new_global_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(new_global_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(new_global_base_index[2]) + k)], wm * vel[1] + (contrib[1] * pos[0] + contrib[4] * pos[1] + contrib[7] * pos[2]) * W);
					atomicAdd(&p2gbuffer[3][static_cast<size_t>(static_cast<size_t>(new_global_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(new_global_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(new_global_base_index[2]) + k)], wm * vel[2] + (contrib[2] * pos[0] + contrib[5] * pos[1] + contrib[8] * pos[2]) * W);
				}
			}
		}
	}
	__syncthreads();

	//Store data from shared memory to grid
	for(int base = static_cast<int>(threadIdx.x); base < NUM_M_VI_IN_ARENA; base += static_cast<int>(blockDim.x)) {
		const char local_block_id = static_cast<char>(base / NUM_M_VI_PER_BLOCK);
		const auto blockno		  = partition.query(ivec3 {blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0), blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0), blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
		// auto grid_block = next_grid.template ch<0>(blockno);
		int channelid = static_cast<int>(base & (NUM_M_VI_PER_BLOCK - 1));
		const char c  = static_cast<char>(channelid % config::G_BLOCKVOLUME);

		const char cz = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		const char cy = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		const char cx = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		float val = p2gbuffer[channelid][static_cast<size_t>(static_cast<size_t>(cx) + (local_block_id & 4 ? config::G_BLOCKSIZE : 0))][static_cast<size_t>(static_cast<size_t>(cy) + (local_block_id & 2 ? config::G_BLOCKSIZE : 0))][static_cast<size_t>(static_cast<size_t>(cz) + (local_block_id & 1 ? config::G_BLOCKSIZE : 0))];
		if(channelid == 0) {
			atomicAdd(&next_grid.ch(_0, blockno).val_1d(_0, c), val);
		} else if(channelid == 1) {
			atomicAdd(&next_grid.ch(_0, blockno).val_1d(_1, c), val);
		} else if(channelid == 2) {
			atomicAdd(&next_grid.ch(_0, blockno).val_1d(_2, c), val);
		} else {
			atomicAdd(&next_grid.ch(_0, blockno).val_1d(_3, c), val);
		}
	}
}

//Need this, cause we cannot partially instantiate function templates in current c++ version
struct CalculateContributionIntermediate {
	std::array<float, 3> pos;
	float J;
	std::array<float, 9> deformation_gradient;
	float log_jp;
};

template<MaterialE MaterialType>
__forceinline__ __device__ void calculate_contribution(const ParticleBuffer<MaterialType> particle_buffer, Duration dt, const std::array<float, 9>& A, std::array<float, 9>& contrib, CalculateContributionIntermediate& data);

template<>
__forceinline__ __device__ void calculate_contribution<MaterialE::J_FLUID>(const ParticleBuffer<MaterialE::J_FLUID> particle_buffer, Duration dt, const std::array<float, 9>& A, std::array<float, 9>& contrib, CalculateContributionIntermediate& data) {
	//Update determinante of deformation gradiant
	//Divergence of velocity multiplied with time and transfered to global space
	data.J += (A[0] + A[4] + A[8]) * dt.count() * config::G_D_INV * data.J;

	//Too low is bad. clamp to 0.1
	//TODO: Maybe make this 0.1 a parameter
	if(data.J < 0.1) {
		data.J = 0.1;
	}

	//TODO: What is calculated here?
	{
		float voln	   = data.J * particle_buffer.volume;
		float pressure = particle_buffer.bulk * (powf(data.J, -particle_buffer.gamma) - 1.f);
		//? - stress; stress = pressure * identity;
		{
			contrib[0] = ((A[0] + A[0]) * config::G_D_INV * particle_buffer.viscosity - pressure) * voln;
			contrib[1] = (A[1] + A[3]) * config::G_D_INV * particle_buffer.viscosity * voln;
			contrib[2] = (A[2] + A[6]) * config::G_D_INV * particle_buffer.viscosity * voln;

			contrib[3] = (A[3] + A[1]) * config::G_D_INV * particle_buffer.viscosity * voln;
			contrib[4] = ((A[4] + A[4]) * config::G_D_INV * particle_buffer.viscosity - pressure) * voln;
			contrib[5] = (A[5] + A[7]) * config::G_D_INV * particle_buffer.viscosity * voln;

			contrib[6] = (A[6] + A[2]) * config::G_D_INV * particle_buffer.viscosity * voln;
			contrib[7] = (A[7] + A[5]) * config::G_D_INV * particle_buffer.viscosity * voln;
			contrib[8] = ((A[8] + A[8]) * config::G_D_INV * particle_buffer.viscosity - pressure) * voln;
		}
	}
}

template<>
__forceinline__ __device__ void calculate_contribution<MaterialE::FIXED_COROTATED>(const ParticleBuffer<MaterialE::FIXED_COROTATED> particle_buffer, Duration dt, const std::array<float, 9>& A, std::array<float, 9>& contrib, CalculateContributionIntermediate& data) {
	vec3x3 dws;
//((d & 0x3) != 0 ? 0.f : 1.f) is identity matrix
#pragma unroll 9
	for(int d = 0; d < 9; ++d) {
		dws.val(d) = A[d] * dt.count() * config::G_D_INV + ((d & 0x3) != 0 ? 0.f : 1.f);
	}

	{
		vec9 F;
		contrib[0]				 = data.deformation_gradient[0];
		contrib[1]				 = data.deformation_gradient[1];
		contrib[2]				 = data.deformation_gradient[2];
		contrib[3]				 = data.deformation_gradient[3];
		contrib[4]				 = data.deformation_gradient[4];
		contrib[5]				 = data.deformation_gradient[5];
		contrib[6]				 = data.deformation_gradient[6];
		contrib[7]				 = data.deformation_gradient[7];
		contrib[8]				 = data.deformation_gradient[8];
		matrix_matrix_multiplication_3d(dws.data_arr(), contrib, F.data_arr());
		ComputeStressIntermediate compute_stress_tmp = {};
		compute_stress<float, MaterialE::FIXED_COROTATED>(particle_buffer.volume, particle_buffer.mu, particle_buffer.lambda, F.data_arr(), contrib, compute_stress_tmp);
	}
}

template<>
__forceinline__ __device__ void calculate_contribution<MaterialE::SAND>(const ParticleBuffer<MaterialE::SAND> particle_buffer, Duration dt, const std::array<float, 9>& A, std::array<float, 9>& contrib, CalculateContributionIntermediate& data) {
	vec3x3 dws;
//((d & 0x3) != 0 ? 0.f : 1.f) is identity matrix
#pragma unroll 9
	for(int d = 0; d < 9; ++d) {
		dws.val(d) = A[d] * dt.count() * config::G_D_INV + ((d & 0x3) != 0 ? 0.f : 1.f);
	}

	{
		vec9 F;
		float log_jp;
		contrib[0]				 = data.deformation_gradient[0];
		contrib[1]				 = data.deformation_gradient[1];
		contrib[2]				 = data.deformation_gradient[2];
		contrib[3]				 = data.deformation_gradient[3];
		contrib[4]				 = data.deformation_gradient[4];
		contrib[5]				 = data.deformation_gradient[5];
		contrib[6]				 = data.deformation_gradient[6];
		contrib[7]				 = data.deformation_gradient[7];
		contrib[8]				 = data.deformation_gradient[8];
		log_jp					 = data.log_jp;

		matrix_matrix_multiplication_3d(dws.data_arr(), contrib, F.data_arr());
		ComputeStressIntermediate compute_stress_tmp = {};
		compute_stress_tmp.cohesion					 = particle_buffer.cohesion;
		compute_stress_tmp.beta						 = particle_buffer.beta;
		compute_stress_tmp.yield_surface			 = particle_buffer.yield_surface;
		compute_stress_tmp.volume_correction		 = particle_buffer.volume_correction;
		compute_stress_tmp.log_jp					 = log_jp;
		compute_stress<float, MaterialE::SAND>(particle_buffer.volume, particle_buffer.mu, particle_buffer.lambda, F.data_arr(), contrib, compute_stress_tmp);
		log_jp = compute_stress_tmp.log_jp;
	}
}

template<>
__forceinline__ __device__ void calculate_contribution<MaterialE::NACC>(const ParticleBuffer<MaterialE::NACC> particle_buffer, Duration dt, const std::array<float, 9>& A, std::array<float, 9>& contrib, CalculateContributionIntermediate& data) {
	vec3x3 dws;
//((d & 0x3) != 0 ? 0.f : 1.f) is identity matrix
#pragma unroll 9
	for(int d = 0; d < 9; ++d) {
		dws.val(d) = A[d] * dt.count() * config::G_D_INV + ((d & 0x3) != 0 ? 0.f : 1.f);
	}

	{
		vec9 F;
		float log_jp;
		contrib[0]				 = data.deformation_gradient[0];
		contrib[1]				 = data.deformation_gradient[1];
		contrib[2]				 = data.deformation_gradient[2];
		contrib[3]				 = data.deformation_gradient[3];
		contrib[4]				 = data.deformation_gradient[4];
		contrib[5]				 = data.deformation_gradient[5];
		contrib[6]				 = data.deformation_gradient[6];
		contrib[7]				 = data.deformation_gradient[7];
		contrib[8]				 = data.deformation_gradient[8];
		log_jp					 = data.log_jp;

		matrix_matrix_multiplication_3d(dws.data_arr(), contrib, F.data_arr());
		ComputeStressIntermediate compute_stress_tmp = {};
		compute_stress_tmp.bm						 = particle_buffer.bm;
		compute_stress_tmp.xi						 = particle_buffer.xi;
		compute_stress_tmp.beta						 = particle_buffer.beta;
		compute_stress_tmp.msqr						 = particle_buffer.msqr;
		compute_stress_tmp.hardening_on				 = particle_buffer.hardening_on;
		compute_stress_tmp.log_jp					 = log_jp;
		compute_stress<float, MaterialE::NACC>(particle_buffer.volume, particle_buffer.mu, particle_buffer.lambda, F.data_arr(), contrib, compute_stress_tmp);
		log_jp = compute_stress_tmp.log_jp;
	}
}

//grid.vel and grid.mass are transfered to shell; shell.mass is transfered to grid
//TODO: Somehow transfer shell.vel to grid!
template<typename Partition, typename Grid, MaterialE MaterialType>
__global__ void grid_to_shell(Duration dt, Duration new_dt, const ParticleBuffer<MaterialType> particle_buffer, TriangleShell triangle_shell, TriangleShellParticleBuffer triangle_shell_particle_buffer, Partition partition, const Grid grid) {
	static constexpr uint64_t NUM_M_VI_PER_BLOCK = static_cast<uint64_t>(config::G_BLOCKVOLUME) * 4;
	static constexpr uint64_t NUM_M_VI_IN_ARENA	 = NUM_M_VI_PER_BLOCK << 3;

	using MViArena	  = std::array<std::array<std::array<std::array<float, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, 4>*;
	using MViArenaRef = std::array<std::array<std::array<std::array<float, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, 4>&;
	
	using ShellBufferArena	  = std::array<std::array<std::array<std::array<float, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, 4>*;
	using ShellBufferArenaRef = std::array<std::array<std::array<std::array<float, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, 4>&;

	static constexpr size_t shared_memory_size = static_cast<size_t>((4) * (config::G_BLOCKSIZE << 1) * (config::G_BLOCKSIZE << 1) * (config::G_BLOCKSIZE << 1)) * sizeof(float);
	__shared__ char shmem[shared_memory_size];//NOLINT(modernize-avoid-c-arrays, readability-redundant-declaration) Cannot declare runtime size shared memory as std::array; extern has different meaning here

	MViArenaRef __restrict__ g2pbuffer = *static_cast<MViArena>(static_cast<void*>(static_cast<char*>(shmem)));

	//Layout of buffers: 1 dimension is channel. other dimensions range [0, 3] contains current block, range [4, 7] contains next block
	//The first cell of the next block is handled by the current block only (but may receive values of particles traveling out of next block in negative direction)

	const int src_blockno		   = static_cast<int>(blockIdx.x);
	const auto blockid			   = partition.active_keys[blockIdx.x];
	const int particle_bucket_size = triangle_shell_particle_buffer.particle_bucket_sizes[src_blockno];

	//If we have no particles in the bucket return
	if(particle_bucket_size == 0) {
		return;
	}

	//Load data from grid to shared memory
	for(int base = static_cast<int>(threadIdx.x); base < NUM_M_VI_IN_ARENA; base += static_cast<int>(blockDim.x)) {
		const char local_block_id = static_cast<char>(base / NUM_M_VI_PER_BLOCK);
		const auto blockno		  = partition.query(ivec3 {blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0), blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0), blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
		const auto grid_block	  = grid.ch(_0, blockno);
		int channelid = static_cast<int>(base & (NUM_M_VI_PER_BLOCK - 1));
		const char c  = static_cast<char>(channelid % config::G_BLOCKVOLUME);

		const char cz = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		const char cy = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		const char cx = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		float val;
		if(channelid == 0) {
			val = grid_block.val_1d(_0, c);
		} else if(channelid == 1) {
			val = grid_block.val_1d(_1, c);
		} else if(channelid == 2) {
			val = grid_block.val_1d(_2, c);
		} else {
			val = grid_block.val_1d(_3, c);
		}

		g2pbuffer[channelid][static_cast<size_t>(static_cast<size_t>(cx) + (local_block_id & 4 ? config::G_BLOCKSIZE : 0))][static_cast<size_t>(static_cast<size_t>(cy) + (local_block_id & 2 ? config::G_BLOCKSIZE : 0))][static_cast<size_t>(static_cast<size_t>(cz) + (local_block_id & 1 ? config::G_BLOCKSIZE : 0))] = val;
	}
	__syncthreads();

	//Perform update
	for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < particle_bucket_size; particle_id_in_block += static_cast<int>(blockDim.x)) {
		//Fetch index of the advection source
		const uint32_t vertex_id = triangle_shell_particle_buffer.blockbuckets[src_blockno * config::G_PARTICLE_NUM_PER_BLOCK + particle_id_in_block];
		
		//TODO:Fetch pos and other data
		auto vertex_data = triangle_shell.ch(_0, 0).ch(_1, 0);
	
		const float mass = vertex_data.val(_0, vertex_id);
		vec3 pos;
		pos[0] = vertex_data.val(_1, vertex_id);//pos
		pos[1] = vertex_data.val(_2, vertex_id);
		pos[2] = vertex_data.val(_3, vertex_id);
		
		vec3 momentum;
		momentum[0] = vertex_data.val(_4, vertex_id);//pos
		momentum[1] = vertex_data.val(_5, vertex_id);
		momentum[2] = vertex_data.val(_6, vertex_id);
		
		//Get position of grid cell
		ivec3 global_base_index = get_block_id(pos.data_arr()) - 1;

		//Get position relative to grid cell
		vec3 local_pos = pos - global_base_index * config::G_DX;

		//Save global_base_index
		ivec3 base_index = global_base_index;

		//Calculate weights and mask global index
		vec3x3 dws;
#pragma unroll 3
		for(int dd = 0; dd < 3; ++dd) {
			const vec3 weight = bspline_weight(local_pos[dd]);
			dws(dd, 0)		  = weight[0];
			dws(dd, 1)		  = weight[1];
			dws(dd, 2)		  = weight[2];

			//Calculate (modulo (config::G_BLOCKMASK + 1)) + 1 of index (== (..., -4, -3, -2, -1, 0, 1, 2, 3, 4, ...) -> (..., 4, 1, 2, 3, 4, 1, 2, 3, 4, ...))
			global_base_index[dd] = ((base_index[dd] - 1) & config::G_BLOCKMASK) + 1;
		}

		//Calculate particle velocity and APIC affine matrix
		//v_p = sum(i, weight_i_p * v_i)
		//A = sum(i, weight_i_p * v_i * (x_i - x_p))
		vec3 vel;
		vel.set(0.f);
		vec9 A;//affine state
		A.set(0.f);
		vec3 new_momentum;
		new_momentum.set(0.f);
#pragma unroll 3
		for(char i = 0; i < 3; i++) {
#pragma unroll 3
			for(char j = 0; j < 3; j++) {
#pragma unroll 3
				for(char k = 0; k < 3; k++) {
					//(x_i - x_p)
					const vec3 xixp = vec3 {static_cast<float>(i), static_cast<float>(j), static_cast<float>(k)} * config::G_DX - local_pos;

					//Weight
					const float W = dws(0, i) * dws(1, j) * dws(2, k);
					
					const float grid_mass = g2pbuffer[0][static_cast<size_t>(static_cast<size_t>(global_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(global_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(global_base_index[2]) + k)];
					//Velocity of grid cell
					const vec3 vi {g2pbuffer[1][static_cast<size_t>(static_cast<size_t>(global_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(global_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(global_base_index[2]) + k)], g2pbuffer[2][static_cast<size_t>(static_cast<size_t>(global_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(global_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(global_base_index[2]) + k)], g2pbuffer[3][static_cast<size_t>(static_cast<size_t>(global_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(global_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(global_base_index[2]) + k)]};

					new_momentum += W * vi * grid_mass;
					
					//Calculate velocity
					vel += W * vi;
	
					//Calculate APIC affine matrix
					A[0] += W * vi[0] * xixp[0];
					A[1] += W * vi[1] * xixp[0];
					A[2] += W * vi[2] * xixp[0];
					A[3] += W * vi[0] * xixp[1];
					A[4] += W * vi[1] * xixp[1];
					A[5] += W * vi[2] * xixp[1];
					A[6] += W * vi[0] * xixp[2];
					A[7] += W * vi[1] * xixp[2];
					A[8] += W * vi[2] * xixp[2];
				}
			}
		}
		
		//Store meomentum and new mass(angular part already contained in velocity)
		//vertex_data.val(_0, vertex_id) = new_mass;
		
		//FIXME: Don't use grid for mass redistribution. Instead use particle/shell vertices (outer shell may act exactly the same as domain, except for surface tension and connection to inner shell)
		//FIXME: Remove shell_grid?!
		vertex_data.val(_0, vertex_id) = mass;
		
		//FIXME: Transfering grid_momentum does not work. Why?
		//vertex_data.val(_4, vertex_id) = new_momentum[0];
		//vertex_data.val(_5, vertex_id) = new_momentum[1];
		//vertex_data.val(_6, vertex_id) = new_momentum[2];
		
		vertex_data.val(_4, vertex_id) = vel[0] * mass;
		vertex_data.val(_5, vertex_id) = vel[1] * mass;
		vertex_data.val(_6, vertex_id) = vel[2] * mass;
	}
}

template<typename Partition, MaterialE MaterialType>
__forceinline__ __device__ void spawn_new_particles(ParticleBuffer<MaterialType> next_particle_buffer, Partition partition, int partition_block_count, const float drop_out_mass, const std::array<float, 3> prev_pos, const std::array<float, 3> pos, int blockno){
	int bin_count;
	if(blockno == partition_block_count - 1){
		bin_count = next_particle_buffer.bin_offsets[blockno + 1] - next_particle_buffer.bin_offsets[blockno];
	}else{
		bin_count = partition_block_count - next_particle_buffer.bin_offsets[blockno];
	}

	//Calculate grid index before and after movement
	const ivec3 global_base_index = get_block_id(prev_pos) - 1;
	const ivec3 new_global_base_index = get_block_id(pos) - 1;
	
	const ivec3 cellid = new_global_base_index - 1;
	
	const int cellno = ((cellid[0] & config::G_BLOCKMASK) << (config::G_BLOCKBITS << 1)) | ((cellid[1] & config::G_BLOCKMASK) << config::G_BLOCKBITS) | (cellid[2] & config::G_BLOCKMASK);

	const size_t space_left = std::min(config::G_MAX_PARTICLES_IN_CELL - next_particle_buffer.cell_particle_counts[blockno * config::G_BLOCKVOLUME + cellno], bin_count * config::G_BIN_CAPACITY - next_particle_buffer.particle_bucket_sizes[blockno]);

	//TODO: More variation/different algorithm
	size_t count_particles = std::ceil(drop_out_mass / next_particle_buffer.mass);
	count_particles = std::min(count_particles, std::min(space_left, (space_left + 3) / 4));
	
	const float mass = drop_out_mass / static_cast<float>(count_particles);
	
	ivec3 blockid = cellid / static_cast<int>(config::G_BLOCKSIZE);
	printf("B %d # %d %d %d - ", (int)count_particles, blockid[0], blockid[1], blockid[2]);
	
	//TODO: Correct parameters
	curandState random_state;
	curand_init(0, blockno, 0, &random_state);
	for(size_t i = 0; i < count_particles; ++i){
		
		//TODO: Different way to calculate this?
		//NOTE: 0.0 is excluded by curand_uniform so we take 0.5 - curand_uniform to get range [-0.5; 0.5[;
		const vec3 particle_pos {
			  (static_cast<float>(new_global_base_index[0]) + (0.5f - curand_uniform(&random_state))) * config::G_DX
			, (static_cast<float>(new_global_base_index[1]) + (0.5f - curand_uniform(&random_state))) * config::G_DX
			, (static_cast<float>(new_global_base_index[2]) + (0.5f - curand_uniform(&random_state))) * config::G_DX
		};

		//TODO: Ensure particle_pos is in the right cell. Or calc cell by particle pos and store momentum in the right place
		//const ivec3 new_global_base_index1 = get_block_id(particle_pos.data_arr());
		//const ivec3 blockid0 = (new_global_base_index - 1) / static_cast<int>(config::G_BLOCKSIZE);
		//const ivec3 blockid1 = (new_global_base_index1 - 1) / static_cast<int>(config::G_BLOCKSIZE);
		//printf("%d %d %d # %d %d %d - ", blockid0[0], blockid0[1], blockid0[2], blockid1[0], blockid1[1], blockid1[2]);
		
		int particle_id_in_block = atomicAdd(next_particle_buffer.particle_bucket_sizes + blockno, 1);
		
		//If no space is left, don't store the particle
		if(particle_id_in_block >= config::G_PARTICLE_NUM_PER_BLOCK) {
			//Reduce count again
			atomicSub(next_particle_buffer.particle_bucket_sizes + blockno, 1);
			printf("No space left in block: block(%d)\n", blockno);
			return;
		}
		
		const int bin_no = next_particle_buffer.bin_offsets[blockno] + particle_id_in_block / config::G_BIN_CAPACITY;
		
		printf("G %d %d %d # %d %d # %f %f %f- ", blockid[0], blockid[1], blockid[2], bin_no, particle_id_in_block, particle_pos[0], particle_pos[1], particle_pos[2]);
		
		//TODO: Fill in values
		StoreParticleDataIntermediate store_particle_data_tmp = {};
		store_particle_data_tmp.mass = mass;
		store_particle_data_tmp.pos													= particle_pos.data_arr();
		store_particle_data_tmp.J													= 1.0f;
		//store_particle_data_tmp.F														= F;
		//store_particle_data_tmp.log_jp														= log_jp;

		//Create new particle
		store_particle_data<MaterialType>(next_particle_buffer, bin_no, particle_id_in_block, store_particle_data_tmp);

		//Store index and movement direction
		{
			//Calculate direction offset
			const int dirtag = dir_offset(((global_base_index - 1) / static_cast<int>(config::G_BLOCKSIZE) - (new_global_base_index - 1) / static_cast<int>(config::G_BLOCKSIZE)).data_arr());
			
			//Store particle in new block
			next_particle_buffer.add_advection(partition, cellid, dirtag, particle_id_in_block);
		}
	}
}

template<typename Partition, typename Grid, MaterialE MaterialType>
__global__ void shell_to_grid(Duration dt, Duration new_dt, int partition_block_count, const ParticleBuffer<MaterialType> particle_buffer, ParticleBuffer<MaterialType> next_particle_buffer, TriangleMesh triangle_mesh, TriangleShell prev_triangle_shell, TriangleShell triangle_shell, TriangleShellParticleBuffer triangle_shell_particle_buffer, Partition partition, Grid next_grid) {
	static constexpr uint64_t NUM_M_VI_PER_BLOCK = static_cast<uint64_t>(config::G_BLOCKVOLUME) * 4;
	static constexpr uint64_t NUM_M_VI_IN_ARENA	 = NUM_M_VI_PER_BLOCK << 3;

	static constexpr unsigned ARENAMASK = (config::G_BLOCKSIZE << 1) - 1;
	static constexpr unsigned ARENABITS = config::G_BLOCKBITS + 1;

	using MViArena	  = std::array<std::array<std::array<std::array<float, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, 4>*;
	using MViArenaRef = std::array<std::array<std::array<std::array<float, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, 4>&;
	
	using ShellBufferArena	  = std::array<std::array<std::array<std::array<float, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, 4>*;
	using ShellBufferArenaRef = std::array<std::array<std::array<std::array<float, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, 4>&;

	static constexpr size_t shared_memory_size = static_cast<size_t>((4) * (config::G_BLOCKSIZE << 1) * (config::G_BLOCKSIZE << 1) * (config::G_BLOCKSIZE << 1)) * sizeof(float);
	__shared__ char shmem[shared_memory_size];//NOLINT(modernize-avoid-c-arrays, readability-redundant-declaration) Cannot declare runtime size shared memory as std::array; extern has different meaning here

	MViArenaRef __restrict__ p2gbuffer = *static_cast<MViArena>(static_cast<void*>(static_cast<char*>(shmem)));

	//Layout of buffers: 1 dimension is channel. other dimensions range [0, 3] contains current block, range [4, 7] contains next block
	//The first cell of the next block is handled by the current block only (but may receive values of particles traveling out of next block in negative direction)

	const int src_blockno		   = static_cast<int>(blockIdx.x);
	const auto blockid			   = partition.active_keys[blockIdx.x];
	const int particle_bucket_size = triangle_shell_particle_buffer.particle_bucket_sizes[src_blockno];

	//If we have no particles in the bucket return
	if(particle_bucket_size == 0) {
		return;
	}

	//Clear return buffer
	for(int base = static_cast<int>(threadIdx.x); base < NUM_M_VI_IN_ARENA; base += static_cast<int>(blockDim.x)) {
		int loc		 = base;
		const char z = static_cast<char>(loc & ARENAMASK);
		loc >>= ARENABITS;

		const char y = static_cast<char>(loc & ARENAMASK);
		loc >>= ARENABITS;

		const char x = static_cast<char>(loc & ARENAMASK);

		p2gbuffer[loc >> ARENABITS][x][y][z] = 0.0f;
	}
	__syncthreads();

	//Perform update
	for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < particle_bucket_size; particle_id_in_block += static_cast<int>(blockDim.x)) {
		//Fetch index of the advection source
		const uint32_t vertex_id = triangle_shell_particle_buffer.blockbuckets[src_blockno * config::G_PARTICLE_NUM_PER_BLOCK + particle_id_in_block];
		
		//Load data
		auto mesh_data = triangle_mesh.ch(_0, 0).ch(_0, 0);
		const auto prev_shell_data_outer = prev_triangle_shell.ch(_0, 0).ch(_1, 0);
		auto shell_data_outer = triangle_shell.ch(_0, 0).ch(_1, 0);
		auto vertex_data = triangle_shell.ch(_0, 0).ch(_1, 0);
		
		const float mass_outer = shell_data_outer.val(_0, vertex_id);
		
		vec3 mesh_pos;
		mesh_pos[0] = mesh_data.val(_3, vertex_id);//global_pos
		mesh_pos[1] = mesh_data.val(_4, vertex_id);
		mesh_pos[2] = mesh_data.val(_5, vertex_id);
		
		vec3 normal;
		normal[0] = mesh_data.val(_9, vertex_id);//normal
		normal[1] = mesh_data.val(_10, vertex_id);
		normal[2] = mesh_data.val(_11, vertex_id);
		
		vec3 shell_pos;
		shell_pos[0] = prev_shell_data_outer.val(_1, vertex_id);//pos
		shell_pos[1] = prev_shell_data_outer.val(_2, vertex_id);
		shell_pos[2] = prev_shell_data_outer.val(_3, vertex_id);
		
		vec3 momentum;
		momentum[0] = shell_data_outer.val(_4, vertex_id);//momentum
		momentum[1] = shell_data_outer.val(_5, vertex_id);
		momentum[2] = shell_data_outer.val(_6, vertex_id);
		
		//If we have no mass at the point, the height at the point is 0.0
		vec3 extrapolated_pos = mesh_pos;
		vec3 new_pos = mesh_pos;
		if(mass_outer > 0.0f){
			const vec3 vel = momentum / mass_outer;
			
			extrapolated_pos = shell_pos + vel * dt.count();
		
			//TODO: Currently we just act by distance to inner
			vec3 diff = shell_pos - mesh_pos;
			const float distance = sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
			//If too much stress, separate and regenerate pos (Done at different stage, we just mark buffer)
			if(distance > config::G_DX * 0.1f) {
				//TODO: How much mass drops out?
				//TODO: Too small mass causes system to explode. Investigate why!
				const float drop_out_mass = (mass_outer > 0.0001f) ? std::min(mass_outer, 0.01f) : 0.0f;

				//TODO: Momentum transfer happens via grid?!
				//TODO: How correctly calculate new momentum
				//const vec3 drop_out_momentum = momentum * 0.5f;
				
				//Only drop out if mass is not 0.0
				if(drop_out_mass > 0.0f){
					//Create new particles
					spawn_new_particles(next_particle_buffer, partition, partition_block_count, drop_out_mass, shell_pos.data_arr(), extrapolated_pos.data_arr(), src_blockno);
				}
				
				//Recalculate pos and velocity
				const float new_mass = mass_outer - drop_out_mass;
				
				//const vec3 new_momentum = momentum - drop_out_momentum;
				//const vec3 new_velocity = new_momentum / new_mass;

				//Calculate new outer vertex pos
				//TODO: Correct algorithm. Currently we just update the position based on the new veloicity, but actually it should be based on current fluid distribution
				//TODO: Ensure that new pos is not farer away than one grid cell from previous pos
				new_pos = mesh_pos + normal * std::min(new_mass, config::G_DX * 0.9f);
				
				//Store data
				shell_data_outer.val(_0, vertex_id) = new_mass;
				//shell_data_outer.val(_4, vertex_id) = new_velocity[0];
				//shell_data_outer.val(_5, vertex_id) = new_velocity[1];
				//shell_data_outer.val(_6, vertex_id) = new_velocity[2];
			}
			
		}
		
		//Store pos
		shell_data_outer.val(_1, vertex_id) = new_pos[0];
		shell_data_outer.val(_2, vertex_id) = new_pos[1];
		shell_data_outer.val(_3, vertex_id) = new_pos[2];
		
		//Calculate other parameters
		//TODO: Correct algorithm.
		vec9 A = {};
			
		//Get position of grid cell
		ivec3 base_index = get_block_id(shell_pos.data_arr()) - 1;

		//TODO: Fill in values
		//TODO: Advection?
		CalculateContributionIntermediate calculate_contribution_tmp = {};
		calculate_contribution_tmp.pos													= extrapolated_pos.data_arr();
		calculate_contribution_tmp.J														= 1.0f;
		//calculate_contribution_tmp.deformation_gradient														= deformation_gradient;
		//calculate_contribution_tmp.log_jp														= log_jp;

		vec9 contrib;
		calculate_contribution(particle_buffer, dt, A.data_arr(), contrib.data_arr(), calculate_contribution_tmp);
		
		//Update momentum?
		//Multiply A with mass to complete it. Then subtract current momentum?
		//C = A * D^-1
		contrib = (A * mass_outer - contrib * new_dt.count()) * config::G_D_INV;

		//Calculate grid index after movement
		ivec3 new_global_base_index = get_block_id(extrapolated_pos.data_arr()) - 1;

		//Get position relative to grid cell
		const vec3 local_pos = extrapolated_pos - new_global_base_index * config::G_DX;

		//Calculate weights and mask global index
		vec3x3 dws;
#pragma unroll 3
		for(char dd = 0; dd < 3; ++dd) {
			const vec3 weight = bspline_weight(local_pos[dd]);
			dws(dd, 0)		  = weight[0];
			dws(dd, 1)		  = weight[1];
			dws(dd, 2)		  = weight[2];

			//Calculate (modulo (config::G_BLOCKMASK + 1)) + 1 of index (== (..., -4, -3, -2, -1, 0, 1, 2, 3, 4, ...) -> (..., 4, 1, 2, 3, 4, 1, 2, 3, 4, ...)) and add offset (may not be out of range max [-4, 4] min [-1, 1])
			new_global_base_index[dd] = (((base_index[dd] - 1) & config::G_BLOCKMASK) + 1) + (new_global_base_index[dd] - base_index[dd]);
		}

		//Dim of p2gbuffer is (4, 8, 8, 8). So if values are too big, discard whole particle
		if(new_global_base_index[0] < 0 || new_global_base_index[1] < 0 || new_global_base_index[2] < 0 || new_global_base_index[0] + 2 >= 8 || new_global_base_index[1] + 2 >= 8 || new_global_base_index[2] + 2 >= 8) {
			//NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg) Cuda has no other way to print; Numbers are array indices to be printed
			printf("new_global_base_index out of range: %d %d %d\n", new_global_base_index[0], new_global_base_index[1], new_global_base_index[2]);
			return;
		}
		
		//Calculate new gird momentum and mass
		//m_i * v_i = sum(p, w_i_p * m_p * vel_p + ?)
#pragma unroll 3
		for(char i = 0; i < 3; i++) {
#pragma unroll 3
			for(char j = 0; j < 3; j++) {
#pragma unroll 3
				for(char k = 0; k < 3; k++) {
					const vec3 pos			  = vec3 {static_cast<float>(i), static_cast<float>(j), static_cast<float>(k)} * config::G_DX - local_pos;
					const float W = dws(0, i) * dws(1, j) * dws(2, k);
					const auto wm = mass_outer * W;

					atomicAdd(&p2gbuffer[0][static_cast<size_t>(static_cast<size_t>(new_global_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(new_global_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(new_global_base_index[2]) + k)], wm);
					atomicAdd(&p2gbuffer[1][static_cast<size_t>(static_cast<size_t>(new_global_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(new_global_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(new_global_base_index[2]) + k)], W * momentum[0] + (contrib[0] * pos[0] + contrib[3] * pos[1] + contrib[6] * pos[2]) * W);
					atomicAdd(&p2gbuffer[2][static_cast<size_t>(static_cast<size_t>(new_global_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(new_global_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(new_global_base_index[2]) + k)], W * momentum[1] + (contrib[1] * pos[0] + contrib[4] * pos[1] + contrib[7] * pos[2]) * W);
					atomicAdd(&p2gbuffer[3][static_cast<size_t>(static_cast<size_t>(new_global_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(new_global_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(new_global_base_index[2]) + k)], W * momentum[2] + (contrib[2] * pos[0] + contrib[5] * pos[1] + contrib[8] * pos[2]) * W);
				}
			}
		}
	}
	__syncthreads();

	//Store data from shared memory to grid
	for(int base = static_cast<int>(threadIdx.x); base < NUM_M_VI_IN_ARENA; base += static_cast<int>(blockDim.x)) {
		const char local_block_id = static_cast<char>(base / NUM_M_VI_PER_BLOCK);
		const auto blockno		  = partition.query(ivec3 {blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0), blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0), blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
		// auto grid_block = next_grid.template ch<0>(blockno);
		int channelid = static_cast<int>(base & (NUM_M_VI_PER_BLOCK - 1));
		const char c  = static_cast<char>(channelid % config::G_BLOCKVOLUME);

		const char cz = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		const char cy = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		const char cx = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		float val = p2gbuffer[channelid][static_cast<size_t>(static_cast<size_t>(cx) + (local_block_id & 4 ? config::G_BLOCKSIZE : 0))][static_cast<size_t>(static_cast<size_t>(cy) + (local_block_id & 2 ? config::G_BLOCKSIZE : 0))][static_cast<size_t>(static_cast<size_t>(cz) + (local_block_id & 1 ? config::G_BLOCKSIZE : 0))];
		if(channelid == 0) {
			atomicAdd(&next_grid.ch(_0, blockno).val_1d(_0, c), val);
		} else if(channelid == 1) {
			atomicAdd(&next_grid.ch(_0, blockno).val_1d(_1, c), val);
		} else if(channelid == 2) {
			atomicAdd(&next_grid.ch(_0, blockno).val_1d(_2, c), val);
		} else {
			atomicAdd(&next_grid.ch(_0, blockno).val_1d(_3, c), val);
		}
	}
}

template<typename Grid>
__global__ void mark_active_grid_blocks(uint32_t block_count, const Grid grid, int* marks) {
	const auto idx	  = blockIdx.x * blockDim.x + threadIdx.x;
	const int blockno = static_cast<int>(idx / config::G_BLOCKVOLUME);
	const int cellno  = static_cast<int>(idx % config::G_BLOCKVOLUME);
	if(blockno >= block_count) {
		return;
	}

	//If the grid cell has a mass mark it as active
	if(grid.ch(_0, blockno).val_1d(_0, cellno) != 0.0f) {
		marks[blockno] = 1;
	}
}

__global__ void mark_active_particle_blocks(uint32_t block_count, const int* __restrict__ particle_bucket_sizes, int* marks) {
	const std::size_t blockno = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if(blockno >= block_count) {
		return;
	}

	//If the particle bucket has particles marke it as active
	if(particle_bucket_sizes[blockno] > 0) {
		marks[blockno] = 1;
	}
}

template<typename Partition>
__global__ void update_partition(uint32_t block_count, const int* __restrict__ source_nos, const Partition partition, Partition next_partition) {
	const std::size_t blockno = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if(blockno >= block_count) {
		return;
	}

	const uint32_t source_no			= source_nos[blockno];
	const auto source_blockid			= partition.active_keys[source_no];
	next_partition.active_keys[blockno] = source_blockid;
	next_partition.reinsert(static_cast<int>(blockno));
}

template<typename ParticleBuffer>
__global__ void update_buckets(uint32_t block_count, const int* __restrict__ source_nos, const ParticleBuffer particle_buffer, ParticleBuffer next_particle_buffer) {
	__shared__ std::size_t source_no[1];//NOLINT(modernize-avoid-c-arrays) Cannot declare shared memory as std::array?
	const std::size_t blockno = blockIdx.x;
	if(blockno >= block_count) {
		return;
	}

	//Copy size and set source_no
	if(threadIdx.x == 0) {
		//Index of block in old layout
		source_no[0]										= source_nos[blockno];
		next_particle_buffer.particle_bucket_sizes[blockno] = particle_buffer.particle_bucket_sizes[source_no[0]];
	}
	__syncthreads();

	//Copy buckets
	const auto particle_counts = next_particle_buffer.particle_bucket_sizes[blockno];
	for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < particle_counts; particle_id_in_block += static_cast<int>(blockDim.x)) {
		next_particle_buffer.blockbuckets[blockno * config::G_PARTICLE_NUM_PER_BLOCK + particle_id_in_block] = particle_buffer.blockbuckets[source_no[0] * config::G_PARTICLE_NUM_PER_BLOCK + particle_id_in_block];
	}
}

template<typename Partition, typename Grid>
__global__ void copy_selected_grid_blocks(const ivec3* __restrict__ prev_blockids, const Partition partition, const int* __restrict__ marks, Grid prev_grid, Grid grid) {
	const auto blockid = prev_blockids[blockIdx.x];

	//If the block is marked as active and if it is found in the new partition, copy the values from the old grid
	if(marks[blockIdx.x]) {
		const auto blockno = partition.query(blockid);
		if(blockno == -1) {
			return;
		}

		const auto sourceblock				= prev_grid.ch(_0, blockIdx.x);
		auto targetblock					= grid.ch(_0, blockno);
		targetblock.val_1d(_0, threadIdx.x) = sourceblock.val_1d(_0, threadIdx.x);
		targetblock.val_1d(_1, threadIdx.x) = sourceblock.val_1d(_1, threadIdx.x);
		targetblock.val_1d(_2, threadIdx.x) = sourceblock.val_1d(_2, threadIdx.x);
		targetblock.val_1d(_3, threadIdx.x) = sourceblock.val_1d(_3, threadIdx.x);
	}
}

template<typename Partition, typename Grid>
__global__ void copy_selected_grid_blocks_triangle_shell(const ivec3* __restrict__ prev_blockids, const Partition partition, const int* __restrict__ marks, Grid prev_grid, Grid grid) {
	const auto blockid = prev_blockids[blockIdx.x];

	//If the block is marked as active and if it is found in the new partition, copy the values from the old grid
	//if(marks[blockIdx.x]) {
		//FIXME: Copy correct blocks!
		const auto blockno = partition.query(blockid);
		if(blockno == -1) {
			return;
		}

		const auto sourceblock				= prev_grid.ch(_0, blockIdx.x);
		auto targetblock					= grid.ch(_0, blockno);
		targetblock.val_1d(_0, threadIdx.x) = sourceblock.val_1d(_0, threadIdx.x);
		targetblock.val_1d(_1, threadIdx.x) = sourceblock.val_1d(_1, threadIdx.x);
		targetblock.val_1d(_2, threadIdx.x) = sourceblock.val_1d(_2, threadIdx.x);
		targetblock.val_1d(_3, threadIdx.x) = sourceblock.val_1d(_3, threadIdx.x);
	//}
}

template<typename Partition>
__global__ void check_table(uint32_t block_count, Partition partition) {
	uint32_t blockno = blockIdx.x * blockDim.x + threadIdx.x;
	if(blockno >= block_count) {
		return;
	}
	auto blockid = partition.active_keys[blockno];
	if(partition.query(blockid) != blockno) {
		printf("FUCK, partition table is wrong!\n");
	}
}

template<typename Grid>
__global__ void sum_grid_mass(Grid grid, float* sum) {
	atomicAdd(sum, grid.ch(_0, blockIdx.x).val_1d(_0, threadIdx.x));
}

__global__ void sum_particle_counts(uint32_t count, int* __restrict__ counts, int* sum) {
	auto idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= count) {
		return;
	}
	atomicAdd(sum, counts[idx]);
}

template<typename Partition>
__global__ void check_partition(uint32_t block_count, Partition partition) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= block_count) {
		return;
	}
	ivec3 blockid = partition.active_keys[idx];
	if(blockid[0] == 0 || blockid[1] == 0 || blockid[2] == 0) {
		printf("\tDAMN, encountered zero block record\n");
	}
	if(partition.query(blockid) != idx) {
		int id	  = partition.query(blockid);
		ivec3 bid = partition.active_keys[id];
		printf(
			"\t\tcheck partition %d, (%d, %d, %d), feedback index %d, (%d, %d, "
			"%d)\n",
			idx,
			(int) blockid[0],
			(int) blockid[1],
			(int) blockid[2],
			id,
			bid[0],
			bid[1],
			bid[2]
		);
	}
}

template<typename Partition, typename Domain>
__global__ void check_partition_domain(uint32_t block_count, int did, Domain const domain, Partition partition) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= block_count) {
		return;
	}
	ivec3 blockid = partition.active_keys[idx];
	if(domain.inside(blockid)) {
		printf("%d-th block (%d, %d, %d) is in domain[%d] (%d, %d, %d)-(%d, %d, %d)\n", idx, blockid[0], blockid[1], blockid[2], did, domain._min[0], domain._min[1], domain._min[2], domain._max[0], domain._max[1], domain._max[2]);
	}
}

template<typename Partition, typename ParticleBuffer, typename ParticleArray>
__global__ void retrieve_particle_buffer(Partition partition, Partition prev_partition, ParticleBuffer particle_buffer, ParticleBuffer next_particle_buffer, ParticleArray particle_array, int* parcount) {
	const int particle_counts	= next_particle_buffer.particle_bucket_sizes[blockIdx.x];
	const ivec3 blockid			= partition.active_keys[blockIdx.x];
	const auto advection_bucket = next_particle_buffer.blockbuckets + blockIdx.x * config::G_PARTICLE_NUM_PER_BLOCK;

	// auto particle_offset = particle_buffer.bin_offsets[blockIdx.x];
	for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < particle_counts; particle_id_in_block += static_cast<int>(blockDim.x)) {
		//Fetch advection (direction at high bits, particle in in cell at low bits)
		const auto advect = advection_bucket[particle_id_in_block];

		//Retrieve the direction (first stripping the particle id by division)
		ivec3 source_blockid;
		dir_components(advect / config::G_PARTICLE_NUM_PER_BLOCK, source_blockid.data_arr());

		//Retrieve the particle id by AND for lower bits
		const auto source_pidib = advect % config::G_PARTICLE_NUM_PER_BLOCK;

		//Get global index by adding the blockid
		source_blockid += blockid;

		//Get block from partition
		const auto advection_source_blockno_from_partition = prev_partition.query(source_blockid);

		//Get bin from particle buffer
		const auto source_bin = particle_buffer.ch(_0, particle_buffer.bin_offsets[advection_source_blockno_from_partition] + source_pidib / config::G_BIN_CAPACITY);

		//Calculate particle id in destination buffer
		const auto particle_id = atomicAdd(parcount, 1);

		//Copy position to destination buffer
		particle_array.val(_0, particle_id) = source_bin.val(_1, source_pidib % config::G_BIN_CAPACITY);
		particle_array.val(_1, particle_id) = source_bin.val(_2, source_pidib % config::G_BIN_CAPACITY);
		particle_array.val(_2, particle_id) = source_bin.val(_3, source_pidib % config::G_BIN_CAPACITY);
	}
}

__global__ void calculate_center_of_mass(TriangleMesh triangle_mesh, uint32_t vertex_count, float* center_of_mass){
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	constexpr int NUM_WARPS		   = (config::DEFAULT_CUDA_BLOCK_SIZE + config::CUDA_WARP_SIZE - 1) / config::CUDA_WARP_SIZE;
	constexpr unsigned ACTIVE_MASK = 0xffffffff;
	
	static constexpr size_t shared_memory_element_count = NUM_WARPS;
	__shared__ vec3 shared_tmp[shared_memory_element_count];//NOLINT(modernize-avoid-c-arrays) Cannot declare runtime size shared memory as std::array

	//Initialize shared memory
	if(threadIdx.x < NUM_WARPS) {
		shared_tmp[threadIdx.x] = {};
	}
	__syncthreads();

	/// within-warp computations
	vec3 weight_pos;
	weight_pos.set(0.0f);
	if(idx < vertex_count) {
		const float mass = triangle_mesh.mass / static_cast<float>(vertex_count);
		
		//Load position
		const auto vertex_data = triangle_mesh.ch(_0, 0).ch(_0, 0);
		vec3 pos;
		pos[0] = vertex_data.val(_3, idx);//pos
		pos[1] = vertex_data.val(_4, idx);
		pos[2] = vertex_data.val(_5, idx);

		weight_pos = pos * mass;
	}
	//Calculate in warp
	for(size_t i = 0; i < 3; ++i){
		for(int iter = 1; iter % config::CUDA_WARP_SIZE; iter <<= 1) {
			float tmp = __shfl_down_sync(ACTIVE_MASK, weight_pos[i], iter, config::CUDA_WARP_SIZE);
			if((threadIdx.x % config::CUDA_WARP_SIZE) + iter < config::CUDA_WARP_SIZE) {
				weight_pos[i] += tmp;
			}
		}
	}
	
	//TODO: Ensure threadIdx.x / config::CUDA_WARP_SIZE is smaller than NUM_WARPS
	if(threadIdx.x % config::CUDA_WARP_SIZE == 0) {
		shared_tmp[threadIdx.x / config::CUDA_WARP_SIZE] += weight_pos;
	}
	__syncthreads();

	//Add values
	for(int interval = NUM_WARPS >> 1; interval > 0; interval >>= 1) {
		if(threadIdx.x < interval) {
			for(size_t i = 0; i < 3; ++i){
				atomicAdd(&(shared_tmp[threadIdx.x][i]), shared_tmp[static_cast<int>(threadIdx.x) + interval][i]);
			}
		}
		__syncthreads();
	}
	if(threadIdx.x == 0) {
		for(size_t i = 0; i < 3; ++i){
			atomicAdd(&(center_of_mass[i]), shared_tmp[0][i]);
		}
	}
};

__global__ void calculate_inertia_and_relative_pos(TriangleMesh triangle_mesh, uint32_t vertex_count, std::array<float, 3> center_of_mass, float* inertia_sum){
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	constexpr int NUM_WARPS		   = (config::DEFAULT_CUDA_BLOCK_SIZE + config::CUDA_WARP_SIZE - 1) / config::CUDA_WARP_SIZE;
	constexpr unsigned ACTIVE_MASK = 0xffffffff;
	
	static constexpr size_t shared_memory_element_count = NUM_WARPS;
	__shared__ vec9 shared_tmp[shared_memory_element_count];//NOLINT(modernize-avoid-c-arrays) Cannot declare runtime size shared memory as std::array

	//Initialize shared memory
	if(threadIdx.x < NUM_WARPS) {
		shared_tmp[threadIdx.x] = {};
	}
	__syncthreads();

	/// within-warp computations
	vec9 weight_inertia;
	weight_inertia.set(0.0f);
	if(idx < vertex_count) {
		const float mass = triangle_mesh.mass / static_cast<float>(vertex_count);
		
		//Load position
		auto vertex_data = triangle_mesh.ch(_0, 0).ch(_0, 0);
		vec3 pos;
		pos[0] = vertex_data.val(_3, idx);//pos
		pos[1] = vertex_data.val(_4, idx);
		pos[2] = vertex_data.val(_5, idx);
		
		//Calculate and store relative position
		const vec3 center_of_mass_vec {center_of_mass[0], center_of_mass[1], center_of_mass[2]};
		const vec3 relative_pos = pos - center_of_mass_vec;
		
		vertex_data.val(_0, idx) = relative_pos[0];//relative_pos
		vertex_data.val(_1, idx) = relative_pos[1];
		vertex_data.val(_2, idx) = relative_pos[2];
		
		//Calculate inertia
		const std::array<float, 9> relative_pos_cross {
			 0.0f, relative_pos[2], -relative_pos[1]
			,-relative_pos[2], 0.0f, relative_pos[0]
			,relative_pos[1], -relative_pos[0], 0.0f
		};
		
		vec9 inertia;
		inertia.set(0.0f);
		matrix_matrix_multiplication_3d(relative_pos_cross, relative_pos_cross, inertia.data_arr());
		

		weight_inertia = -inertia * mass;
	}
	//Calculate in warp
	for(size_t i = 0; i < 9; ++i){
		for(int iter = 1; iter % config::CUDA_WARP_SIZE; iter <<= 1) {
			float tmp = __shfl_down_sync(ACTIVE_MASK, weight_inertia[i], iter, config::CUDA_WARP_SIZE);
			if((threadIdx.x % config::CUDA_WARP_SIZE) + iter < config::CUDA_WARP_SIZE) {
				weight_inertia[i] += tmp;
			}
		}
	}
	
	//TODO: Ensure threadIdx.x / config::CUDA_WARP_SIZE is smaller than NUM_WARPS
	if(threadIdx.x % config::CUDA_WARP_SIZE == 0) {
		shared_tmp[threadIdx.x / config::CUDA_WARP_SIZE] += weight_inertia;
	}
	__syncthreads();

	//Add values
	for(int interval = NUM_WARPS >> 1; interval > 0; interval >>= 1) {
		if(threadIdx.x < interval) {
			for(size_t i = 0; i < 9; ++i){
				atomicAdd(&(shared_tmp[threadIdx.x][i]), shared_tmp[static_cast<int>(threadIdx.x) + interval][i]);
			}
		}
		__syncthreads();
	}
	if(threadIdx.x == 0) {
		for(size_t i = 0; i < 9; ++i){
			atomicAdd(&(inertia_sum[i]), shared_tmp[0][i]);
		}
	}
};

__global__ void calculate_count_faces(TriangleMesh triangle_mesh, uint32_t face_count){
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= face_count) {
		return;
	}
	
	auto vertex_data = triangle_mesh.ch(_0, 0).ch(_0, 0);
	const auto face_data = triangle_mesh.ch(_0, 0).ch(_1, 0);
	
	vec<uint32_t, 3> vertex_indices;
	vertex_indices[0] = face_data.val(_0, idx);
	vertex_indices[1] = face_data.val(_1, idx);
	vertex_indices[2] = face_data.val(_2, idx);
	
	for(size_t i = 0; i < 3; ++i){
		const int current_vertex_index = vertex_indices[i];
		
		//Add one per face
		atomicAdd(&vertex_data.val(_9, current_vertex_index), 1u);
	}
}

__global__ void calculate_normals(TriangleMesh triangle_mesh, uint32_t face_count){
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= face_count) {
		return;
	}
	
	auto vertex_data = triangle_mesh.ch(_0, 0).ch(_0, 0);
	const auto face_data = triangle_mesh.ch(_0, 0).ch(_1, 0);
	
	vec<uint32_t, 3> vertex_indices;
	vertex_indices[0] = face_data.val(_0, idx);
	vertex_indices[1] = face_data.val(_1, idx);
	vertex_indices[2] = face_data.val(_2, idx);
	
	vec3 positions[3];
	
	for(size_t i = 0; i < 3; ++i){
		const int current_vertex_index = vertex_indices[i];
		
		positions[i][0] = vertex_data.val(_3, current_vertex_index);//global_pos
		positions[i][1] = vertex_data.val(_4, current_vertex_index);
		positions[i][2] = vertex_data.val(_5, current_vertex_index);
	}
	
	vec3 face_normal;
	vec_cross_vec_3d(face_normal.data_arr(), (positions[1] - positions[0]).data_arr(), (positions[2] - positions[0]).data_arr());
	
	//Normalize
	face_normal = face_normal / sqrt(face_normal[0] * face_normal[0] + face_normal[1] * face_normal[1] + face_normal[2] * face_normal[2]);
	
	for(size_t i = 0; i < 3; ++i){
		const int current_vertex_index = vertex_indices[i];
		
		float cosine = (positions[(i + 1) % 3] - positions[i]).dot(positions[(i + 2) % 3] - positions[i]) / sqrt((positions[(i + 1) % 3] - positions[i]).dot(positions[(i + 1) % 3] - positions[i]) * (positions[(i + 2) % 3] - positions[i]).dot(positions[(i + 2) % 3] - positions[i]));
		cosine = std::min(std::max(cosine, -1.0f), 1.0f);
		const float angle = std::acos(cosine);
		
		//Add one per face
		atomicAdd(&vertex_data.val(_9, current_vertex_index), face_normal[0] * angle);//normal
		atomicAdd(&vertex_data.val(_10, current_vertex_index), face_normal[1] * angle);
		atomicAdd(&vertex_data.val(_11, current_vertex_index), face_normal[2] * angle);
	}
}

__global__ void normalize_normals(TriangleMesh triangle_mesh, uint32_t vertex_count){
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= vertex_count) {
		return;
	}
	
	//Load data
	auto vertex_data = triangle_mesh.ch(_0, 0).ch(_0, 0);
	
	vec3 normal;
	normal[0] = vertex_data.val(_9, idx);//normal
	normal[1] = vertex_data.val(_10, idx);
	normal[2] = vertex_data.val(_11, idx);
	
	//Normalize
	normal = normal / sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
	
	//Store data
	vertex_data.val(_9, idx) = normal[0];//normal
	vertex_data.val(_10, idx) = normal[1];
	vertex_data.val(_11, idx) = normal[2];
}

__global__ void copy_triangle_mesh_data_to_device(TriangleMesh triangle_mesh, uint32_t vertex_count, uint32_t face_count, float* positions, uint32_t* faces){
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < vertex_count) {
		//Load data
		auto vertex_data = triangle_mesh.ch(_0, 0).ch(_0, 0);
		
		//Store data
		vertex_data.val(_3, idx) = positions[idx * 3];//global_pos
		vertex_data.val(_4, idx) = positions[idx * 3 + 1];
		vertex_data.val(_5, idx) = positions[idx * 3 + 2];
		
		//Clear other data
		//TODO: Maybe put this somewhere else?
		vertex_data.val(_6, idx) = 0.0f;
		vertex_data.val(_7, idx) = 0.0f;
		vertex_data.val(_8, idx) = 0.0f;
		vertex_data.val(_9, idx) = 0.0f;
		vertex_data.val(_10, idx) = 0.0f;
		vertex_data.val(_11, idx) = 0.0f;
	}
	
	if(idx < face_count) {
		//Load data
		auto face_data = triangle_mesh.ch(_0, 0).ch(_1, 0);
		
		//Store data
		face_data.val(_0, idx) = faces[idx * 3];//face indices
		face_data.val(_1, idx) = faces[idx * 3 + 1];
		face_data.val(_2, idx) = faces[idx * 3 + 2];
	}
}

__global__ void copy_triangle_mesh_data_to_host(TriangleMesh triangle_mesh, uint32_t vertex_count, float* positions){
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= vertex_count) {
		return;
	}
	
	//Load data
	auto vertex_data = triangle_mesh.ch(_0, 0).ch(_0, 0);
	
	//Store data
	positions[idx * 3] = vertex_data.val(_3, idx);//global_pos
	positions[idx * 3 + 1] = vertex_data.val(_4, idx);
	positions[idx * 3 + 2] = vertex_data.val(_5, idx);
}

__global__ void copy_triangle_shell_data_to_host(TriangleShell triangle_shell, uint32_t vertex_count, float* positions){
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= vertex_count) {
		return;
	}
	
	//Load data
	auto vertex_data = triangle_shell.ch(_0, 0).ch(_1, 0);
	
	//Store data
	positions[idx * 3] = vertex_data.val(_1, idx);//global_pos
	positions[idx * 3 + 1] = vertex_data.val(_2, idx);
	positions[idx * 3 + 2] = vertex_data.val(_3, idx);
}

__global__ void init_triangle_shell(TriangleMesh triangle_mesh, TriangleShell triangle_shell, uint32_t vertex_count){
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= vertex_count) {
		return;
	}
	
	//Load data
	const auto mesh_data = triangle_mesh.ch(_0, 0).ch(_0, 0);
	auto vertex_data_inner = triangle_shell.ch(_0, 0).ch(_0, 0);
	auto vertex_data_outer = triangle_shell.ch(_0, 0).ch(_1, 0);
	
	//Store data
	vertex_data_inner.val(_0, idx) = 0.0f;//mass
	vertex_data_outer.val(_0, idx) = 0.01f;//FIXME:0.0f;//mass
	
	//TODO:Do we have to clear this or is it reset at each step?
	vertex_data_inner.val(_1, idx) = 0.0f;
	vertex_data_inner.val(_2, idx) = 0.0f;
	vertex_data_inner.val(_3, idx) = 0.0f;
	
	vertex_data_outer.val(_4, idx) = 0.0f;
	vertex_data_outer.val(_5, idx) = 0.0f;
	vertex_data_outer.val(_6, idx) = 0.0f;
	
	//FIXME:vertex_data_outer.val(_1, idx) = mesh_data.val(_3, idx);//pos
	//FIXME:vertex_data_outer.val(_2, idx) = mesh_data.val(_4, idx);
	//FIXME:vertex_data_outer.val(_3, idx) = mesh_data.val(_5, idx);
	
	//FIXME: Remove when we don't have initial mass anymore
	{
		vec3 mesh_pos;
		mesh_pos[0] = mesh_data.val(_3, idx);//global_pos
		mesh_pos[1] = mesh_data.val(_4, idx);
		mesh_pos[2] = mesh_data.val(_5, idx);
		
		vec3 normal;
		normal[0] = mesh_data.val(_9, idx);//normal
		normal[1] = mesh_data.val(_10, idx);
		normal[2] = mesh_data.val(_11, idx);
		
		const float mass_outer = vertex_data_outer.val(_0, idx);

		//Calculate new outer vertex pos
		//TODO: Correct algorithm. Currently we just update the position based on the new veloicity, but actually it should be based on current fluid distribution
		vec3 new_pos = mesh_pos + normal * std::min(mass_outer, config::G_DX * 0.9f);
		
		//Store pos
		vertex_data_outer.val(_1, idx) = new_pos[0];
		vertex_data_outer.val(_2, idx) = new_pos[1];
		vertex_data_outer.val(_3, idx) = new_pos[2];
	}
	
	
}

__global__ void clear_triangle_shell(TriangleShell triangle_shell, uint32_t vertex_count){
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= vertex_count) {
		return;
	}
	
	//Load data
	auto vertex_data_inner = triangle_shell.ch(_0, 0).ch(_0, 0);
	auto vertex_data_outer = triangle_shell.ch(_0, 0).ch(_1, 0);
	
	//Store data
	vertex_data_inner.val(_0, idx) = 0.0f;//mass
	vertex_data_outer.val(_0, idx) = 0.0f;//mass
	
	vertex_data_inner.val(_1, idx) = 0.0f;//momentum
	vertex_data_inner.val(_2, idx) = 0.0f;
	vertex_data_inner.val(_3, idx) = 0.0f;
	
	vertex_data_outer.val(_4, idx) = 0.0f;//momentum
	vertex_data_outer.val(_5, idx) = 0.0f;
	vertex_data_outer.val(_6, idx) = 0.0f;
}

__global__ void update_triangle_mesh(TriangleMesh triangle_mesh, uint32_t vertex_count, std::array<float, 3> center_of_mass, std::array<float, 3> linear_velocity, std::array<float, 4> rotation, std::array<float, 3> angular_velocity){
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= vertex_count) {
		return;
	}
	
	const vec3 center_of_mass_vec {center_of_mass[0], center_of_mass[1], center_of_mass[2]};
	const vec3 linear_velocity_vec {linear_velocity[0], linear_velocity[1], linear_velocity[2]};
	
	//Load data
	auto vertex_data = triangle_mesh.ch(_0, 0).ch(_0, 0);
	
	vec3 relative_pos;
	relative_pos[0] = vertex_data.val(_0, idx);//relative_pos
	relative_pos[1] = vertex_data.val(_1, idx);
	relative_pos[2] = vertex_data.val(_2, idx);

	//Calculate new values
	vec3 new_relative_pos;
	rotate_by_quat(relative_pos.data_arr(), rotation, new_relative_pos.data_arr());
	
	const vec3 new_pos = center_of_mass_vec + new_relative_pos;
	
	const std::array<float, 9> angular_velocity_cross {
		 0.0f, angular_velocity[2], -angular_velocity[1]
		,-angular_velocity[2], 0.0f, angular_velocity[0]
		,angular_velocity[1], -angular_velocity[0], 0.0f
	};
	
	vec3 angular_velocity_tmp;
	matrix_vector_multiplication_3d(angular_velocity_cross, new_relative_pos.data_arr(), angular_velocity_tmp.data_arr());
	const vec3 new_velocity = linear_velocity_vec + angular_velocity_tmp;
	
	//Store data
	vertex_data.val(_3, idx) = new_pos[0];//global_pos
	vertex_data.val(_4, idx) = new_pos[1];
	vertex_data.val(_5, idx) = new_pos[2];
	vertex_data.val(_6, idx) = new_velocity[0];//velocity
	vertex_data.val(_7, idx) = new_velocity[1];
	vertex_data.val(_8, idx) = new_velocity[2];
}

//TODO: Face based?
//TODO: Navier stokes, face based, but keeping in mind that fores are 3d?
//TODO:Distribute mass and velocity on inner site
__global__ void update_triangle_shell_inner(TriangleMesh triangle_mesh, TriangleShell triangle_shell, uint32_t vertex_count, Duration dt){
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= vertex_count) {
		return;
	}
	
	//Load data
	const auto mesh_data = triangle_mesh.ch(_0, 0).ch(_0, 0);
	auto shell_data = triangle_shell.ch(_0, 0).ch(_0, 0);
	
	const float mass = shell_data.val(_0, idx);///mass
	
	vec3 current_momentum;
	current_momentum[0] = shell_data.val(_1, idx);//velocity
	current_momentum[1] = shell_data.val(_2, idx);
	current_momentum[2] = shell_data.val(_3, idx);
	
	vec3 mesh_velocity;
	mesh_velocity[0] = mesh_data.val(_6, idx);//velocity
	mesh_velocity[1] = mesh_data.val(_7, idx);
	mesh_velocity[2] = mesh_data.val(_8, idx);
	
	//Transfer speed of triangle mesh (using adhesion)
	//TODO: Actually material specific
	//current_velocity = (mesh_velocity + current_velocity) * 0.5f;
	
	//Currently just transfer mass
	const vec3 momentum = mesh_velocity * triangle_mesh.mass;
	
	
	//TODO: Also apply friction somehow (either here or in later step!)
	
	//TODO: Other forces (fluid affected by triangle mesh surfaces, not just vertices => other force directions?; Also no centripetal force => verify, that fluid moves away from rotation center of rotationg body
	
	//Add gravity
	//FIXME: Only apply if not already applied on triangle mesh; Or actually it is model depended if this should be applied.
	//current_velocity[1] += config::G_GRAVITY * dt.count();
	
	//Store data
	shell_data.val(_1, idx) = momentum[0];
	shell_data.val(_2, idx) = momentum[1];
	shell_data.val(_3, idx) = momentum[2];
}
/*
__global__ void update_triangle_shell_subdomain(TriangleMesh triangle_mesh, TriangleShell triangle_shell, TriangleShell next_triangle_shell, uint32_t face_count){
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= face_count) {
		return;
	}
	
	const auto vertex_data = triangle_mesh.ch(_0, 0).ch(_0, 0);
	const auto face_data = triangle_mesh.ch(_0, 0).ch(_1, 0);
	const auto shell_data_inner = triangle_shell.ch(_0, 0).ch(_0, 0);
	const auto shell_data_outer = triangle_shell.ch(_0, 0).ch(_1, 0);
	
	auto next_shell_data_inner = next_triangle_shell.ch(_0, 0).ch(_0, 0);
	auto next_shell_data_outer = next_triangle_shell.ch(_0, 0).ch(_1, 0);
	
	vec<uint32_t, 3> vertex_indices;
	vertex_indices[0] = face_data.val(_0, idx);
	vertex_indices[1] = face_data.val(_1, idx);
	vertex_indices[2] = face_data.val(_2, idx);
	
	
	float cell_mass = 0.0f;
	vec3 cell_velocity = {};
	
	//TODO: Correct algorithm. Currently we do not more than just smoothing mass and velocity
	for(size_t i = 0; i < 3; ++i){
		const uint32_t current_vertex_index = vertex_indices[i];
		
		const uint32_t current_count_faces = vertex_data.val(_9, current_vertex_index);//count_faces
		
		const float weight = 1.0f / static_cast<float>(current_count_faces);
		
		const float mass_inner = shell_data_inner.val(_0, current_vertex_index);///mass
		const float mass_outer = shell_data_outer.val(_0, current_vertex_index);///mass
		
		vec3 velocity_inner;
		velocity_inner[0] = shell_data_inner.val(_1, current_vertex_index);//velocity
		velocity_inner[1] = shell_data_inner.val(_2, current_vertex_index);
		velocity_inner[2] = shell_data_inner.val(_3, current_vertex_index);
		
		vec3 velocity_outer;
		velocity_outer[0] = shell_data_outer.val(_4, current_vertex_index);//velocity
		velocity_outer[1] = shell_data_outer.val(_5, current_vertex_index);
		velocity_outer[2] = shell_data_outer.val(_6, current_vertex_index);
		
		cell_mass += mass_inner * weight;
		cell_mass += mass_outer * weight;
		
		cell_velocity += velocity_inner * weight;
		cell_velocity += velocity_outer * weight;
	}
	
	for(size_t i = 0; i < 3; ++i){
		const int current_vertex_index = vertex_indices[i];
		
		//Store new data, being 1/6  of cell data
		atomicAdd(&next_shell_data_inner.val(_0, current_vertex_index), cell_mass/6.0f);
		atomicAdd(&next_shell_data_outer.val(_0, current_vertex_index), cell_mass/6.0f);
		
		atomicAdd(&next_shell_data_inner.val(_1, current_vertex_index), cell_velocity[0]/6.0f);
		atomicAdd(&next_shell_data_inner.val(_2, current_vertex_index), cell_velocity[1]/6.0f);
		atomicAdd(&next_shell_data_inner.val(_3, current_vertex_index), cell_velocity[2]/6.0f);
		
		atomicAdd(&next_shell_data_outer.val(_4, current_vertex_index), cell_velocity[0]/6.0f);
		atomicAdd(&next_shell_data_outer.val(_5, current_vertex_index), cell_velocity[1]/6.0f);
		atomicAdd(&next_shell_data_outer.val(_6, current_vertex_index), cell_velocity[2]/6.0f);
	}
}*/

__global__ void update_triangle_shell_subdomain(TriangleShell triangle_shell, TriangleShell next_triangle_shell, uint32_t vertex_count){
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= vertex_count) {
		return;
	}
	
	const auto shell_data_inner = triangle_shell.ch(_0, 0).ch(_0, 0);
	const auto shell_data_outer = triangle_shell.ch(_0, 0).ch(_1, 0);
	auto next_shell_data_inner = next_triangle_shell.ch(_0, 0).ch(_0, 0);
	auto next_shell_data_outer = next_triangle_shell.ch(_0, 0).ch(_1, 0);
	
	const float mass_inner = shell_data_inner.val(_0, idx);//mass
	const float mass_outer = shell_data_outer.val(_0, idx);//mass
	
	vec3 momentum_inner;
	momentum_inner[0] = shell_data_inner.val(_1, idx);//velocity
	momentum_inner[1] = shell_data_inner.val(_2, idx);
	momentum_inner[2] = shell_data_inner.val(_3, idx);
	
	vec3 momentum_outer;
	momentum_outer[0] = shell_data_outer.val(_4, idx);//momentum
	momentum_outer[1] = shell_data_outer.val(_5, idx);
	momentum_outer[2] = shell_data_outer.val(_6, idx);
	
	/*
	//TODO: Where do mass redistribution for outer shell? (also including transfer between domain and shell) => Not at all/outer shelöl behaves very similiar to domain; Inner shell redistribution can happen in update of inner shell (and may include transfer between mesh and domain)
	
	//Smooth mass and momentum
	const float smoothed_mass = (mass_inner + mass_outer) * 0.5f;
	
	if(smoothed_mass > 0.0f){
		const vec3 smoothed_velocity = ((velocity_inner * mass_inner + momentum_outer) * 0.5f) / smoothed_mass;
	
		next_shell_data_inner.val(_0, idx) = smoothed_mass;
		next_shell_data_outer.val(_0, idx) = smoothed_mass;
	
		next_shell_data_inner.val(_1, idx) = smoothed_velocity[0];
		next_shell_data_inner.val(_2, idx) = smoothed_velocity[1];
		next_shell_data_inner.val(_3, idx) = smoothed_velocity[2];
		next_shell_data_outer.val(_4, idx) = smoothed_velocity[0];
		next_shell_data_outer.val(_5, idx) = smoothed_velocity[1];
		next_shell_data_outer.val(_6, idx) = smoothed_velocity[2];
	}
	*/
	
	const vec3 total_momentum = momentum_inner + momentum_outer;
	
	next_shell_data_inner.val(_0, idx) = mass_inner;
	next_shell_data_outer.val(_0, idx) = mass_outer;
	
	next_shell_data_inner.val(_1, idx) = 0.0f;
	next_shell_data_inner.val(_2, idx) = 0.0f;
	next_shell_data_inner.val(_3, idx) = 0.0f;
	next_shell_data_outer.val(_4, idx) = total_momentum[0];
	next_shell_data_outer.val(_5, idx) = total_momentum[1];
	next_shell_data_outer.val(_6, idx) = total_momentum[2];
}

//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers, readability-identifier-naming, misc-definitions-in-headers)
}// namespace mn

#endif