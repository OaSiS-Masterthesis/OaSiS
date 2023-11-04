#ifndef IQ_CUH
#define IQ_CUH

#include "kernels.cuh"

namespace mn {
using namespace placeholder;//NOLINT(google-build-using-namespace) Allow placeholders to be included generally for simplification

//TODO: Make magic numbers to constants where suitable
//TODO: Ensure call dimensions and such are small enough to allow narrowing conversations. Or directly use unsigned where possible
//TODO: Maybe use names instead of formula signs for better understanding
//NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers, readability-identifier-naming, misc-definitions-in-headers) CUDA does not yet support std::span; Common names for physical formulas; Cannot declare __global__ functions inline
namespace iq {
	
//NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables, readability-identifier-naming) Check is buggy and reports variable errors for template arguments
using FluidParticleBufferData  = Structural<StructuralType::DENSE, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::SUM_POW2_ALIGN>, ParticleBinDomain, attrib_layout::SOA, f32_, f32_, f32_, f32_, f32_, f32_, f32_, f32_, f32_, f32_, f32_, f32_, i32_>;//velocity, C, count_neighbours
//NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables, readability-identifier-naming)
	
struct FluidParticleBuffer : Instance<particle_buffer_<FluidParticleBufferData>> {
	using base_t							 = Instance<particle_buffer_<FluidParticleBufferData>>;
	
	managed_memory_type* managed_memory;

	FluidParticleBuffer() = default;

	template<typename Allocator>
	FluidParticleBuffer(Allocator allocator, managed_memory_type* managed_memory, std::size_t count)
		: base_t {spawn<particle_buffer_<FluidParticleBufferData>, orphan_signature>(allocator, count)}
		, managed_memory(managed_memory)
		{}
};

template<MaterialE MaterialType>
__global__ void init_fluid_particle_buffer(const ParticleBuffer<MaterialType> particle_buffer, FluidParticleBuffer iq_fluid_particle_buffer, std::array<float, 3> v0){
	const int src_blockno		   = static_cast<int>(blockIdx.x);
	const int particle_bucket_size = particle_buffer.particle_bucket_sizes[src_blockno];
	
	//If we have no particles in the bucket return
	if(particle_bucket_size == 0) {
		return;
	}
	
	for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < particle_bucket_size; particle_id_in_block += static_cast<int>(blockDim.x)) {
		auto fluid_particle_bin													= iq_fluid_particle_buffer.ch(_0, particle_buffer.bin_offsets[src_blockno] + particle_id_in_block / config::G_BIN_CAPACITY);
		//velocity
		fluid_particle_bin.val(_0, particle_id_in_block % config::G_BIN_CAPACITY) = v0[0];
		fluid_particle_bin.val(_1, particle_id_in_block % config::G_BIN_CAPACITY) = v0[1];
		fluid_particle_bin.val(_2, particle_id_in_block % config::G_BIN_CAPACITY) = v0[2];
		//C
		fluid_particle_bin.val(_3, particle_id_in_block % config::G_BIN_CAPACITY) = 0.0f;
		fluid_particle_bin.val(_4, particle_id_in_block % config::G_BIN_CAPACITY) = 0.0f;
		fluid_particle_bin.val(_5, particle_id_in_block % config::G_BIN_CAPACITY) = 0.0f;
		fluid_particle_bin.val(_6, particle_id_in_block % config::G_BIN_CAPACITY) = 0.0f;
		fluid_particle_bin.val(_7, particle_id_in_block % config::G_BIN_CAPACITY) = 0.0f;
		fluid_particle_bin.val(_8, particle_id_in_block % config::G_BIN_CAPACITY) = 0.0f;
		fluid_particle_bin.val(_9, particle_id_in_block % config::G_BIN_CAPACITY) = 0.0f;
		fluid_particle_bin.val(_10, particle_id_in_block % config::G_BIN_CAPACITY) = 0.0f;
		fluid_particle_bin.val(_11, particle_id_in_block % config::G_BIN_CAPACITY) = 0.0f;
	}
}
	
constexpr size_t BLOCK_SIZE = config::G_BLOCKVOLUME;//config::G_PARTICLE_BATCH_CAPACITY;


//TODO: Make adjustable by extern paraameter (like changing if we use constant or linear kernels)
//Kernel degrees
constexpr size_t INTERPOLATION_DEGREE_SOLID_VELOCITY = 2;
constexpr size_t INTERPOLATION_DEGREE_SOLID_PRESSURE = 0;
constexpr size_t INTERPOLATION_DEGREE_FLUID_VELOCITY = 2;
constexpr size_t INTERPOLATION_DEGREE_FLUID_PRESSURE = 0;
constexpr size_t INTERPOLATION_DEGREE_INTERFACE_PRESSURE = 0;
constexpr size_t INTERPOLATION_DEGREE_MAX = std::max(std::max(std::max(INTERPOLATION_DEGREE_SOLID_VELOCITY, INTERPOLATION_DEGREE_SOLID_PRESSURE), std::max(INTERPOLATION_DEGREE_FLUID_VELOCITY, INTERPOLATION_DEGREE_FLUID_PRESSURE)), INTERPOLATION_DEGREE_INTERFACE_PRESSURE);

constexpr size_t MAX_SHARED_PARTICLE_SOLID = config::G_BLOCKVOLUME * config::G_MAX_PARTICLES_IN_CELL >> 4;
constexpr size_t MAX_SHARED_PARTICLE_FLUID = config::G_BLOCKVOLUME * config::G_MAX_PARTICLES_IN_CELL >> 5;

static_assert(MAX_SHARED_PARTICLE_SOLID > 0 && MAX_SHARED_PARTICLE_FLUID > 0 && "Shared count must be at least 1");

//Matrix layout
constexpr size_t NUM_ROWS_PER_BLOCK = config::G_BLOCKVOLUME;
constexpr size_t NUM_COLUMNS_PER_BLOCK = (2 * INTERPOLATION_DEGREE_MAX + 1) * (2 * INTERPOLATION_DEGREE_MAX + 1) * (2 * INTERPOLATION_DEGREE_MAX + 1);
constexpr size_t LHS_MATRIX_SIZE_Y = 4;
constexpr size_t LHS_MATRIX_SIZE_X = 4;

constexpr size_t LHS_MATRIX_TOTAL_BLOCK_COUNT = 12;
__device__ const std::array<size_t, LHS_MATRIX_SIZE_Y> lhs_num_blocks_per_row = {
	  2
	, 3
	, 3
	, 4
};
__device__ const std::array<std::array<size_t, LHS_MATRIX_SIZE_X>, LHS_MATRIX_SIZE_Y> lhs_block_offsets_per_row = {{
	  {0, 3, std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()}
	, {1, 2, 3, std::numeric_limits<size_t>::max()}
	, {1, 2, 3, std::numeric_limits<size_t>::max()}
	, {0, 1, 2, 3}
}};

constexpr size_t SOLVE_VELOCITY_MATRIX_SIZE_Y = 2;
constexpr size_t SOLVE_VELOCITY_MATRIX_SIZE_X = 4;

constexpr size_t SOLVE_VELOCITY_MATRIX_TOTAL_BLOCK_COUNT = 5;
__device__ const std::array<size_t, SOLVE_VELOCITY_MATRIX_SIZE_Y> solve_velocity_num_blocks_per_row = {
	  2
	, 3
};
__device__ const std::array<std::array<size_t, SOLVE_VELOCITY_MATRIX_SIZE_X>, SOLVE_VELOCITY_MATRIX_SIZE_Y> solve_velocity_block_offsets_per_row = {{
	  {0, 3, std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()}
	, {1, 2, 3, std::numeric_limits<size_t>::max()}
}};

__device__ const std::array<size_t, 1> temporary_num_blocks_per_row = {
	1
};

__device__ const std::array<std::array<size_t, 1>, 1> temporary_block_offsets_per_row = {{
	{0}
}};

//FIXME: If INTERPOLATION_DEGREE_MAX is too big neighbour blocks were not activated
static_assert((INTERPOLATION_DEGREE_MAX / config::G_BLOCKSIZE) <= 1 && "Neighbour blocks not activated");

struct IQCreatePointers {
	float* scaling_solid;
	float* scaling_fluid;
	
	float* mass_solid;
	float* mass_fluid;
	
	const int* gradient_solid_rows;
	const int* gradient_solid_columns;
	float* gradient_solid_values;
	const int* gradient_fluid_rows;
	const int* gradient_fluid_columns;
	float* gradient_fluid_values;
	
	const int* coupling_solid_rows;
	const int* coupling_solid_columns;
	float* coupling_solid_values;
	const int* coupling_fluid_rows;
	const int* coupling_fluid_columns;
	float* coupling_fluid_values;
	
	const int* boundary_fluid_rows;
	const int* boundary_fluid_columns;
	float* boundary_fluid_values;
	
	float* iq_rhs;
	float* iq_solve_velocity_result;
};

//Sliced
template<size_t BLOCK_SIZE, size_t ITEMS_PER_THREAD>
constexpr size_t get_thread_index(const size_t index){
	return index % BLOCK_SIZE;
}

template<size_t BLOCK_SIZE, size_t ITEMS_PER_THREAD>
constexpr size_t get_thread_offset(const size_t index){
	return index / BLOCK_SIZE;
}

template<size_t BLOCK_SIZE, size_t ITEMS_PER_THREAD>
constexpr size_t get_thread_count(const size_t thread_id, const size_t global_count){
	return (global_count / BLOCK_SIZE) + (global_count % BLOCK_SIZE > thread_id ? 1 : 0);
}

template<size_t BLOCK_SIZE, size_t ITEMS_PER_THREAD>
constexpr size_t get_global_index(const size_t thread_id, const size_t offset){
	return offset * BLOCK_SIZE + thread_id;
}

//TODO: Maybe activate on cell level (but data is on block level); Maybe recheck that blocks are not empty
template<size_t MatrixSizeX, size_t MatrixSizeY, size_t NumRowsPerBlock, size_t NumColumnsPerBlock, size_t NumDimensionsPerRow, typename Partition>
__global__ void clear_iq_system(const size_t* num_blocks_per_row, const std::array<size_t, MatrixSizeX>* block_offsets_per_row, const uint32_t num_active_blocks, const uint32_t num_blocks, const Partition partition, int* iq_lhs_rows, int* iq_lhs_columns, float* iq_lhs_values) {
	//const int src_blockno		   = static_cast<int>(blockIdx.x);
	const auto blockid			   = partition.active_keys[blockIdx.x];
	
	//Handle own rows and add column data for all neighbour cells of each cell (fixed amount)
	//We can calculate the offset in the column array by our id and the amount of neighbour cells (=columns)
	const size_t base_row = NumRowsPerBlock * blockIdx.x;
	
	//Accumulate blocks per row
	std::array<size_t, MatrixSizeY> accumulated_blocks_per_row;
	accumulated_blocks_per_row[0] = 0;
	for(size_t i = 1; i < MatrixSizeY; ++i){
		accumulated_blocks_per_row[i] = accumulated_blocks_per_row[i - 1] + num_blocks_per_row[i - 1];
	}
	
	//Fill matrix
	for(size_t row = static_cast<int>(threadIdx.x); row < NumRowsPerBlock; row += static_cast<int>(blockDim.x)){
		const ivec3 cellid = blockid * static_cast<int>(config::G_BLOCKSIZE) + ivec3((static_cast<int>(row) / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE, (static_cast<int>(row) / config::G_BLOCKSIZE) % config::G_BLOCKSIZE, static_cast<int>(row) % config::G_BLOCKSIZE);
		
		for(size_t row_offset = 0; row_offset < MatrixSizeY; ++row_offset){
			for(size_t dimension = 0; dimension < NumDimensionsPerRow; ++dimension){
				iq_lhs_rows[row_offset * NumDimensionsPerRow * NumRowsPerBlock * num_blocks + NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension] = accumulated_blocks_per_row[row_offset] * NumDimensionsPerRow * NumRowsPerBlock * num_active_blocks * NumColumnsPerBlock + num_blocks_per_row[row_offset] * (NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension) * NumColumnsPerBlock;
			}
		}
		
		int neighbour_cellnos[NumColumnsPerBlock];
		for(size_t column = 0; column < NumColumnsPerBlock; ++column){
			const ivec3 neighbour_local_id = ivec3(static_cast<int>((column / ((2 * INTERPOLATION_DEGREE_MAX + 1) * (2 * INTERPOLATION_DEGREE_MAX + 1))) % (2 * INTERPOLATION_DEGREE_MAX + 1)), static_cast<int>((column / (2 * INTERPOLATION_DEGREE_MAX + 1)) % (2 * INTERPOLATION_DEGREE_MAX + 1)), static_cast<int>(column % (2 * INTERPOLATION_DEGREE_MAX + 1))) - ivec3(static_cast<int>(INTERPOLATION_DEGREE_MAX), static_cast<int>(INTERPOLATION_DEGREE_MAX), static_cast<int>(INTERPOLATION_DEGREE_MAX));
			const ivec3 neighbour_cellid = cellid + neighbour_local_id;
			const ivec3 neighbour_blockid = neighbour_cellid / static_cast<int>(config::G_BLOCKSIZE);
			
			const ivec3 neighbour_base_cellid = neighbour_blockid * static_cast<int>(config::G_BLOCKSIZE);
			const ivec3 neighbour_celloffset = neighbour_cellid - neighbour_base_cellid;
			
			const int neighbour_blockno = partition.query(neighbour_blockid);
			const int neighbour_cellno = NumRowsPerBlock * neighbour_blockno + (config::G_BLOCKSIZE * config::G_BLOCKSIZE) * neighbour_celloffset[0] + config::G_BLOCKSIZE * neighbour_celloffset[1] + neighbour_celloffset[2];
			
			neighbour_cellnos[column] = neighbour_cellno;
			//if(neighbour_cellno < 0){
			//	printf("ERROR %d %d %d # %d %d %d # %d %d %d # %d # %d\n", static_cast<int>(blockIdx.x), static_cast<int>(row), static_cast<int>(column), cellid[0], cellid[1], cellid[2], neighbour_cellid[0], neighbour_cellid[1], neighbour_cellid[2], neighbour_blockno, neighbour_cellno);
			//}
		}
		
		//Sort columns
		thrust::sort(thrust::seq, neighbour_cellnos, neighbour_cellnos + NumColumnsPerBlock);
		
		for(size_t column = 0; column < NumColumnsPerBlock; ++column){
			for(size_t row_offset = 0; row_offset < MatrixSizeY; ++row_offset){
				for(size_t column_offset_index = 0; column_offset_index < num_blocks_per_row[row_offset]; ++column_offset_index){
					for(size_t dimension = 0; dimension < NumDimensionsPerRow; ++dimension){
						iq_lhs_columns[iq_lhs_rows[row_offset * NumDimensionsPerRow * NumRowsPerBlock * num_blocks + NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension] + column_offset_index * NumColumnsPerBlock + column] = block_offsets_per_row[row_offset][column_offset_index] * num_blocks * NumRowsPerBlock + neighbour_cellnos[column];
						iq_lhs_values[iq_lhs_rows[row_offset * NumDimensionsPerRow * NumRowsPerBlock * num_blocks + NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension] + column_offset_index * NumColumnsPerBlock + column] = 0.0f;
					}
				}
			}
		}
	}
}

template<size_t MatrixSizeY, size_t NumRowsPerBlock, size_t NumDimensionsPerRow, typename Partition>
__global__ void fill_empty_rows(const uint32_t num_blocks, const Partition partition, int* iq_lhs_rows) {
	//const int src_blockno		   = static_cast<int>(blockIdx.x);
	const auto blockid			   = partition.active_keys[blockIdx.x];
	
	//Handle own rows and add column data for all neighbour cells of each cell (fixed amount)
	//We can calculate the offset in the column array by our id and the amount of neighbour cells (=columns)
	const size_t base_row = NumRowsPerBlock * blockIdx.x;
	
	//Fill empty rows
	for(size_t row = static_cast<int>(threadIdx.x); row < NumRowsPerBlock; row += static_cast<int>(blockDim.x)){
		const ivec3 cellid = blockid * static_cast<int>(config::G_BLOCKSIZE) + ivec3((static_cast<int>(row) / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE, (static_cast<int>(row) / config::G_BLOCKSIZE) % config::G_BLOCKSIZE, static_cast<int>(row) % config::G_BLOCKSIZE);

		for(size_t dimension = 0; dimension < NumDimensionsPerRow; ++dimension){
			for(size_t row_offset = 0; row_offset < MatrixSizeY; ++row_offset){
				const int own_value = atomicAdd(&(iq_lhs_rows[row_offset * NumDimensionsPerRow * NumRowsPerBlock * num_blocks + NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension]), 0);
				
				if(own_value == 0){
					//TODO: Use more efficient way, like maybe binary search
					//NOTE: We are using atomic load/store to ensure operations are atomic. It does not matter wether they are correctly ordered as if we don't see the single write that might occure we are heading to the next memory locations till we either find a write or a original value
					int value;
					for(int next_row_offset = 0; next_row_offset <= (row_offset * NumDimensionsPerRow * NumRowsPerBlock * num_blocks + NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension); ++next_row_offset){
						//Atomic load
						value = atomicAdd(&(iq_lhs_rows[row_offset * NumDimensionsPerRow * NumRowsPerBlock * num_blocks + NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension - next_row_offset]), 0);
						if(value != 0){
							break;
						}
					}
				
					/*
					printf("ABC0 %d %d %d %d # %d # %d %d\n"
						, static_cast<int>(base_row)
						, static_cast<int>(row)
						, static_cast<int>(row_offset)
						, static_cast<int>(dimension)
						, static_cast<int>(row_offset * NumDimensionsPerRow * NumRowsPerBlock * num_blocks + NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension)
						, own_value
						, value
					);*/
					
					//Atomic store, including non-atomic load of own value. This is okay, cause only this thread may modify this value
					atomicAdd(&(iq_lhs_rows[row_offset * NumDimensionsPerRow * NumRowsPerBlock * num_blocks + NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension]), (value - own_value));
				}
			}
		}
	}
}

template<size_t NumRowsPerBlock, size_t NumDimensionsPerRow, typename Partition>
__global__ void add_block_rows(const size_t matrix_size_y, size_t* num_blocks_per_row, const uint32_t num_blocks, const Partition partition, const int** iq_parts_rows, int* iq_rows) {
	//const int src_blockno		   = static_cast<int>(blockIdx.x);
	//const auto blockid			   = partition.active_keys[blockIdx.x];
	
	//Handle own rows and add column data for all neighbour cells of each cell (fixed amount)
	//We can calculate the offset in the column array by our id and the amount of neighbour cells (=columns)
	const size_t base_row = NumRowsPerBlock * blockIdx.x;
	
	for(size_t row = static_cast<int>(threadIdx.x); row < NumRowsPerBlock; row += static_cast<int>(blockDim.x)){
		for(size_t dimension = 0; dimension < NumDimensionsPerRow; ++dimension){
			for(size_t row_offset = 0; row_offset < matrix_size_y; ++row_offset){
				//Accumulate blocks per row
				size_t accumulated_blocks_per_row = 0;
				for(size_t i = 0; i < row_offset; ++i){
					accumulated_blocks_per_row += num_blocks_per_row[i];
				}
				
				size_t elements_in_row = 0;
				for(size_t column_offset_index = 0; column_offset_index < num_blocks_per_row[row_offset]; ++column_offset_index){
					const size_t block_index = accumulated_blocks_per_row + column_offset_index;
					
					elements_in_row += (iq_parts_rows[block_index][NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension + 1] - iq_parts_rows[block_index][NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension]);
				}
				
				atomicAdd(&(iq_rows[row_offset * NumDimensionsPerRow * NumRowsPerBlock * num_blocks + NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension]), elements_in_row);
			}
		}
	}
}

template<size_t NumRowsPerBlock, size_t NumDimensionsPerRow, typename Partition>
__global__ void copy_values(const size_t matrix_size_x, const size_t matrix_size_y, size_t* num_blocks_per_row, size_t* block_offsets_per_row, const uint32_t num_blocks, const Partition partition, const int** iq_parts_rows, const int** iq_parts_columns, const float** iq_parts_values, const int* iq_rows, int* iq_columns, float* iq_values) {
	//const int src_blockno		   = static_cast<int>(blockIdx.x);
	//const auto blockid			   = partition.active_keys[blockIdx.x];
	
	//Handle own rows and add column data for all neighbour cells of each cell (fixed amount)
	//We can calculate the offset in the column array by our id and the amount of neighbour cells (=columns)
	const size_t base_row = NumRowsPerBlock * blockIdx.x;

	for(size_t row = static_cast<int>(threadIdx.x); row < NumRowsPerBlock; row += static_cast<int>(blockDim.x)){
		for(size_t dimension = 0; dimension < NumDimensionsPerRow; ++dimension){
			for(size_t row_offset = 0; row_offset < matrix_size_y; ++row_offset){
				//Accumulate blocks per row
				size_t accumulated_blocks_per_row = 0;
				for(size_t i = 0; i < row_offset; ++i){
					accumulated_blocks_per_row += num_blocks_per_row[i];
				}
				
				size_t offset_in_row = 0;
				for(size_t column_offset_index = 0; column_offset_index < num_blocks_per_row[row_offset]; ++column_offset_index){
					const size_t block_index = accumulated_blocks_per_row + column_offset_index;
					const int values_offset = iq_rows[row_offset * NumDimensionsPerRow * NumRowsPerBlock * num_blocks + NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension] + offset_in_row;
					
					const size_t values_in_row = iq_parts_rows[block_index][NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension + 1] - iq_parts_rows[block_index][NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension];
					
					if(values_in_row > 0){
						const int* columns_ptr_src = &(iq_parts_columns[block_index][iq_parts_rows[block_index][NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension]]);
						const float* values_ptr_src = &(iq_parts_values[block_index][iq_parts_rows[block_index][NumDimensionsPerRow * base_row + NumDimensionsPerRow * row + dimension]]);
					
						thrust::transform(thrust::seq, columns_ptr_src, columns_ptr_src + values_in_row, &(iq_columns[values_offset]), [&matrix_size_x, &block_offsets_per_row, &num_blocks, &row_offset, &column_offset_index](const int& column){
							return block_offsets_per_row[row_offset * matrix_size_x + column_offset_index] * num_blocks * NumRowsPerBlock + column;
						});
						thrust::copy(thrust::seq, values_ptr_src, values_ptr_src + values_in_row, &(iq_values[values_offset]));
						
						offset_in_row += values_in_row;
					}
				}
			}
		}
	}
}

template<MaterialE MaterialTypeSolid>
__forceinline__ __device__ void store_data_solid(const ParticleBuffer<MaterialTypeSolid> particle_buffer_solid, float* __restrict__ scaling_solid, float* __restrict__ pressure_solid_nominator, float* __restrict__ pressure_solid_denominator, const float W_pressure, const float W_velocity, const float mass, const float J);

template<MaterialE MaterialTypeSolid>
__forceinline__ __device__ void store_data_neigbours_solid(const ParticleBuffer<MaterialTypeSolid> particle_buffer_solid, float* __restrict__ gradient_solid, const float W1_pressure, const float delta_W_velocity, const float mass, const float J);

template<MaterialE MaterialTypeSolid>
__forceinline__ __device__ void store_data_neigbours_coupling_solid(const ParticleBuffer<MaterialTypeSolid> particle_buffer_solid, float* __restrict__ coupling_solid, const float W_velocity, const float W1_pressure, const float contact_area, const float normal);

template<MaterialE MaterialTypeFluid>
__forceinline__ __device__ void store_data_fluid(const ParticleBuffer<MaterialTypeFluid> particle_buffer_fluid, float* __restrict__ scaling_fluid, float* __restrict__ pressure_fluid_nominator, float* __restrict__ pressure_fluid_denominator, const float W_pressure, const float W_velocity, const float mass, const float J);

template<MaterialE MaterialTypeFluid>
__forceinline__ __device__ void store_data_neigbours_fluid(const ParticleBuffer<MaterialTypeFluid> particle_buffer_fluid, float* __restrict__ gradient_fluid, float* __restrict__ boundary_fluid, const float W1_pressure, const float delta_W_velocity, const float mass);

template<MaterialE MaterialTypeFluid>
__forceinline__ __device__ void store_data_neigbours_coupling_fluid(const ParticleBuffer<MaterialTypeFluid> particle_buffer_fluid, float* __restrict__ coupling_fluid, const float W_velocity, const float W1_pressure, const float contact_area, const float normal);


template<>
__forceinline__ __device__ void store_data_solid<MaterialE::J_FLUID>(const ParticleBuffer<MaterialE::J_FLUID> particle_buffer_solid, float* __restrict__ scaling_solid, float* __restrict__ pressure_solid_nominator, float* __restrict__ pressure_solid_denominator, const float W_pressure, const float W_velocity, const float mass, const float J){
	printf("Material type not supported for coupling as solid.");
}

template<>
__forceinline__ __device__ void store_data_solid<MaterialE::FIXED_COROTATED>(const ParticleBuffer<MaterialE::FIXED_COROTATED> particle_buffer_solid, float* __restrict__ scaling_solid, float* __restrict__ pressure_solid_nominator, float* __restrict__ pressure_solid_denominator, const float W_pressure, const float W_velocity, const float mass, const float J){
	printf("Material type not supported for coupling as solid.");
}

template<>
__forceinline__ __device__ void store_data_solid<MaterialE::SAND>(const ParticleBuffer<MaterialE::SAND> particle_buffer_solid, float* __restrict__ scaling_solid, float* __restrict__ pressure_solid_nominator, float* __restrict__ pressure_solid_denominator, const float W_pressure, const float W_velocity, const float mass, const float J){
	printf("Material type not supported for coupling as solid.");
}

template<>
__forceinline__ __device__ void store_data_solid<MaterialE::NACC>(const ParticleBuffer<MaterialE::NACC> particle_buffer_solid, float* __restrict__ scaling_solid, float* __restrict__ pressure_solid_nominator, float* __restrict__ pressure_solid_denominator, const float W_pressure, const float W_velocity, const float mass, const float J){
	printf("Material type not supported for coupling as solid.");
}

template<>
__forceinline__ __device__ void store_data_solid<MaterialE::FIXED_COROTATED_GHOST>(const ParticleBuffer<MaterialE::FIXED_COROTATED_GHOST> particle_buffer_solid, float* __restrict__ scaling_solid, float* __restrict__ pressure_solid_nominator, float* __restrict__ pressure_solid_denominator, const float W_pressure, const float W_velocity, const float mass, const float J){			
	const float volume_0 = (mass / particle_buffer_solid.rho);
	const float pressure = -particle_buffer_solid.lambda * (J - 1.0f);
	
	(*scaling_solid) += (volume_0 / particle_buffer_solid.lambda) * W_pressure;
	(*pressure_solid_nominator) += volume_0 * J * pressure * W_pressure;
	(*pressure_solid_denominator) += volume_0 * J * W_pressure;
}

template<>
__forceinline__ __device__ void store_data_neigbours_solid<MaterialE::J_FLUID>(const ParticleBuffer<MaterialE::J_FLUID> particle_buffer_solid, float* __restrict__ gradient_solid, const float W1_pressure, const float delta_W_velocity, const float mass, const float J){
	printf("Material type not supported for coupling as solid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_solid<MaterialE::FIXED_COROTATED>(const ParticleBuffer<MaterialE::FIXED_COROTATED> particle_buffer_solid, float* __restrict__ gradient_solid, const float W1_pressure, const float delta_W_velocity, const float mass, const float J){
	printf("Material type not supported for coupling as solid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_solid<MaterialE::SAND>(const ParticleBuffer<MaterialE::SAND> particle_buffer_solid, float* __restrict__ gradient_solid, const float W1_pressure, const float delta_W_velocity, const float mass, const float J){
	printf("Material type not supported for coupling as solid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_solid<MaterialE::NACC>(const ParticleBuffer<MaterialE::NACC> particle_buffer_solid, float* __restrict__ gradient_solid, const float W1_pressure, const float delta_W_velocity, const float mass, const float J){
	printf("Material type not supported for coupling as solid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_solid<MaterialE::FIXED_COROTATED_GHOST>(const ParticleBuffer<MaterialE::FIXED_COROTATED_GHOST> particle_buffer_solid, float* __restrict__ gradient_solid, const float W1_pressure, const float delta_W_velocity, const float mass, const float J){
	(*gradient_solid) += -(mass / particle_buffer_solid.rho) * J * W1_pressure * delta_W_velocity;
}

template<>
__forceinline__ __device__ void store_data_neigbours_coupling_solid<MaterialE::J_FLUID>(const ParticleBuffer<MaterialE::J_FLUID> particle_buffer_solid, float* __restrict__ coupling_solid, const float W_velocity, const float W1_pressure, const float contact_area, const float normal){
	printf("Material type not supported for coupling as solid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_coupling_solid<MaterialE::FIXED_COROTATED>(const ParticleBuffer<MaterialE::FIXED_COROTATED> particle_buffer_solid, float* __restrict__ coupling_solid, const float W_velocity, const float W1_pressure, const float contact_area, const float normal){
	printf("Material type not supported for coupling as solid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_coupling_solid<MaterialE::SAND>(const ParticleBuffer<MaterialE::SAND> particle_buffer_solid, float* __restrict__ coupling_solid, const float W_velocity, const float W1_pressure, const float contact_area, const float normal){
	printf("Material type not supported for coupling as solid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_coupling_solid<MaterialE::NACC>(const ParticleBuffer<MaterialE::NACC> particle_buffer_solid, float* __restrict__ coupling_solid, const float W_velocity, const float W1_pressure, const float contact_area, const float normal){
	printf("Material type not supported for coupling as solid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_coupling_solid<MaterialE::FIXED_COROTATED_GHOST>(const ParticleBuffer<MaterialE::FIXED_COROTATED_GHOST> particle_buffer_solid, float* __restrict__ coupling_solid, const float W_velocity, const float W1_pressure, const float contact_area, const float normal){
	(*coupling_solid) += contact_area * W_velocity * W1_pressure * normal;
}

template<>
__forceinline__ __device__ void store_data_fluid<MaterialE::J_FLUID>(const ParticleBuffer<MaterialE::J_FLUID> particle_buffer_fluid, float* __restrict__ scaling_fluid, float* __restrict__ pressure_fluid_nominator, float* __restrict__ pressure_fluid_denominator, const float W_pressure, const float W_velocity, const float mass, const float J){			
	const float volume_0 = (mass / particle_buffer_fluid.rho);
	const float lambda = (particle_buffer_fluid.bulk - (2.0f / 3.0f) * particle_buffer_fluid.viscosity);
	//const float pressure = lambda * (powf(J, -particle_buffer_fluid.gamma) - 1.0f);
	const float pressure = lambda * (J - 1.0f);
	
	//FIXME: Why does solid use volume_0/lambda for scaling? How does that corralate to 1/(lambda * J)?
	//Volume weighted average of pressure;
	(*scaling_fluid) += (volume_0 / lambda) * W_pressure;
	//(*scaling_fluid) += -(1.0f / (lambda * particle_buffer_fluid.gamma * powf(J, -particle_buffer_fluid.gamma))) * W_pressure;
	(*pressure_fluid_nominator) += volume_0 * J * pressure * W_pressure;
	(*pressure_fluid_denominator) += volume_0 * J * W_pressure;
}

template<>
__forceinline__ __device__ void store_data_fluid<MaterialE::FIXED_COROTATED>(const ParticleBuffer<MaterialE::FIXED_COROTATED> particle_buffer_fluid, float* __restrict__ scaling_fluid, float* __restrict__ pressure_fluid_nominator, float* __restrict__ pressure_fluid_denominator, const float W_pressure, const float W_velocity, const float mass, const float J){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_fluid<MaterialE::SAND>(const ParticleBuffer<MaterialE::SAND> particle_buffer_fluid, float* __restrict__ scaling_fluid, float* __restrict__ pressure_fluid_nominator, float* __restrict__ pressure_fluid_denominator, const float W_pressure, const float W_velocity, const float mass, const float J){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_fluid<MaterialE::NACC>(const ParticleBuffer<MaterialE::NACC> particle_buffer_fluid, float* __restrict__ scaling_fluid, float* __restrict__ pressure_fluid_nominator, float* __restrict__ pressure_fluid_denominator, const float W_pressure, const float W_velocity, const float mass, const float J){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_fluid<MaterialE::FIXED_COROTATED_GHOST>(const ParticleBuffer<MaterialE::FIXED_COROTATED_GHOST> particle_buffer_fluid, float* __restrict__ scaling_fluid, float* __restrict__ pressure_fluid_nominator, float* __restrict__ pressure_fluid_denominator, const float W_pressure, const float W_velocity, const float mass, const float J){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_fluid<MaterialE::J_FLUID>(const ParticleBuffer<MaterialE::J_FLUID> particle_buffer_fluid, float* __restrict__ gradient_fluid, float* __restrict__ boundary_fluid, const float W1_pressure, const float delta_W_velocity, const float mass){
	(*gradient_fluid) += -(mass / particle_buffer_fluid.rho) * W1_pressure * delta_W_velocity;
	
	//FIXME: Is that correct?  Actually also not particle based maybe? And just add once?
	//if(boundary_fluid != nullptr){
	//(*boundary_fluid) += W_velocity * W1_pressure * boundary_normal[alpha];
	//}
}

template<>
__forceinline__ __device__ void store_data_neigbours_fluid<MaterialE::FIXED_COROTATED>(const ParticleBuffer<MaterialE::FIXED_COROTATED> particle_buffer_fluid, float* __restrict__ gradient_fluid, float* __restrict__ boundary_fluid, const float W1_pressure, const float delta_W_velocity, const float mass){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_fluid<MaterialE::SAND>(const ParticleBuffer<MaterialE::SAND> particle_buffer_fluid, float* __restrict__ gradient_fluid, float* __restrict__ boundary_fluid, const float W1_pressure, const float delta_W_velocity, const float mass){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_fluid<MaterialE::NACC>(const ParticleBuffer<MaterialE::NACC> particle_buffer_fluid, float* __restrict__ gradient_fluid, float* __restrict__ boundary_fluid, const float W1_pressure, const float delta_W_velocity, const float mass){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_fluid<MaterialE::FIXED_COROTATED_GHOST>(const ParticleBuffer<MaterialE::FIXED_COROTATED_GHOST> particle_buffer_fluid, float* __restrict__ gradient_fluid, float* __restrict__ boundary_fluid, const float W1_pressure, const float delta_W_velocity, const float mass){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_coupling_fluid<MaterialE::J_FLUID>(const ParticleBuffer<MaterialE::J_FLUID> particle_buffer_fluid, float* __restrict__ coupling_fluid, const float W_velocity, const float W1_pressure, const float contact_area, const float normal){
	(*coupling_fluid) += contact_area * W_velocity * W1_pressure * normal;
}

template<>
__forceinline__ __device__ void store_data_neigbours_coupling_fluid<MaterialE::FIXED_COROTATED>(const ParticleBuffer<MaterialE::FIXED_COROTATED> particle_buffer_fluid, float* __restrict__ coupling_fluid, const float W_velocity, const float W1_pressure, const float contact_area, const float normal){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_coupling_fluid<MaterialE::SAND>(const ParticleBuffer<MaterialE::SAND> particle_buffer_fluid, float* __restrict__ coupling_fluid, const float W_velocity, const float W1_pressure, const float contact_area, const float normal){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_coupling_fluid<MaterialE::NACC>(const ParticleBuffer<MaterialE::NACC> particle_buffer_fluid, float* __restrict__ coupling_fluid, const float W_velocity, const float W1_pressure, const float contact_area, const float normal){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_coupling_fluid<MaterialE::FIXED_COROTATED_GHOST>(const ParticleBuffer<MaterialE::FIXED_COROTATED_GHOST> particle_buffer_fluid, float* __restrict__ coupling_fluid, const float W_velocity, const float W1_pressure, const float contact_area, const float normal){
	printf("Material type not supported for coupling as fluid.");
}

template<MaterialE MaterialType>
__forceinline__ __device__ void update_strain_solid(const ParticleBuffer<MaterialType> particle_buffer, int src_blockno, int particle_id_in_block, const float weighted_pressure);

template<>
__forceinline__ __device__ void update_strain_solid<MaterialE::J_FLUID>(const ParticleBuffer<MaterialE::J_FLUID> particle_buffer, int src_blockno, int particle_id_in_block, const float weighted_pressure) {
	printf("Material type not supported for updating strain as solid.");
}

template<>
__forceinline__ __device__ void update_strain_solid(const ParticleBuffer<MaterialE::FIXED_COROTATED> particle_buffer, int src_blockno, int particle_id_in_block, const float weighted_pressure) {
	printf("Material type not supported for updating strain as solid.");
}

template<>
__forceinline__ __device__ void update_strain_solid(const ParticleBuffer<MaterialE::SAND> particle_buffer, int src_blockno, int particle_id_in_block, const float weighted_pressure) {
	printf("Material type not supported for updating strain as solid.");
}

template<>
__forceinline__ __device__ void update_strain_solid(const ParticleBuffer<MaterialE::NACC> particle_buffer, int src_blockno, int particle_id_in_block, const float weighted_pressure) {
	printf("Material type not supported for updating strain as solid.");
}

template<>
__forceinline__ __device__ void update_strain_solid(const ParticleBuffer<MaterialE::FIXED_COROTATED_GHOST> particle_buffer, int src_blockno, int particle_id_in_block, const float weighted_pressure) {
	float J = 1.0f - (weighted_pressure / particle_buffer.lambda);
	
	//Too low is bad. clamp to 0.1
	//TODO: Maybe make this 0.1 a parameter
	if(J < 0.1) {
		J = 0.1;
	}
	
	{
		auto particle_bin													 = particle_buffer.ch(_0, particle_buffer.bin_offsets[src_blockno] + particle_id_in_block / config::G_BIN_CAPACITY);
		particle_bin.val(_13, particle_id_in_block % config::G_BIN_CAPACITY) = J;
	}
}

template<MaterialE MaterialType>
__forceinline__ __device__ void update_strain_fluid(const ParticleBuffer<MaterialType> particle_buffer, int src_blockno, int particle_id_in_block, const float weighted_pressure);

template<>
__forceinline__ __device__ void update_strain_fluid<MaterialE::J_FLUID>(const ParticleBuffer<MaterialE::J_FLUID> particle_buffer, int src_blockno, int particle_id_in_block, const float weighted_pressure) {
	const float lambda = (particle_buffer.bulk - (2.0f / 3.0f) * particle_buffer.viscosity);
	
	//float J = pow(1.0f + (weighted_pressure / lambda), -1.0f/particle_buffer.gamma);
	float J = 1.0f + (weighted_pressure / lambda);
	
	//Too low is bad. clamp to 0.1
	//TODO: Maybe make this 0.1 a parameter
	if(J < 0.1) {
		J = 0.1;
	}
	
	{
		auto particle_bin													 = particle_buffer.ch(_0, particle_buffer.bin_offsets[src_blockno] + particle_id_in_block / config::G_BIN_CAPACITY);
		particle_bin.val(_4, particle_id_in_block % config::G_BIN_CAPACITY) = J;
	}
}

template<>
__forceinline__ __device__ void update_strain_fluid(const ParticleBuffer<MaterialE::FIXED_COROTATED> particle_buffer, int src_blockno, int particle_id_in_block, const float weighted_pressure) {
	printf("Material type not supported for updating strain as fluid.");
}

template<>
__forceinline__ __device__ void update_strain_fluid(const ParticleBuffer<MaterialE::SAND> particle_buffer, int src_blockno, int particle_id_in_block, const float weighted_pressure) {
	printf("Material type not supported for updating strain as fluid.");
}

template<>
__forceinline__ __device__ void update_strain_fluid(const ParticleBuffer<MaterialE::NACC> particle_buffer, int src_blockno, int particle_id_in_block, const float weighted_pressure) {
	printf("Material type not supported for updating strain as fluid.");
}

template<>
__forceinline__ __device__ void update_strain_fluid(const ParticleBuffer<MaterialE::FIXED_COROTATED_GHOST> particle_buffer, int src_blockno, int particle_id_in_block, const float weighted_pressure) {
	printf("Material type not supported for updating strain as fluid.");
}

template<typename Partition, typename Grid, MaterialE MaterialTypeSolid, MaterialE MaterialTypeFluid>
__forceinline__ __device__ void aggregate_data_solid(const ParticleBuffer<MaterialTypeSolid> particle_buffer_solid, const ParticleBuffer<MaterialTypeFluid> particle_buffer_fluid, const ParticleBuffer<MaterialTypeSolid> next_particle_buffer_solid, const ParticleBuffer<MaterialTypeFluid> next_particle_buffer_fluid, const Partition prev_partition, const Grid grid_solid, const Grid grid_fluid, FluidParticleBuffer iq_fluid_particle_buffer, const std::array<float, 3>* __restrict__ position_shared, const float* __restrict__ mass_shared, const float* __restrict__ J_shared, const std::array<float, 3>* __restrict__ normal_shared, const SurfacePointType* __restrict__ point_type_shared, const float* __restrict__ contact_area_shared, const int particle_offset, const int current_blockno, const ivec3 current_blockid, const ivec3 block_cellid, const int particle_id_in_block, float* __restrict__ scaling_solid, float* __restrict__ pressure_solid_nominator, float* __restrict__ pressure_solid_denominator, float* __restrict__ mass_solid, float* __restrict__ gradient_solid, float* __restrict__ coupling_solid, float* __restrict__ scaling_fluid, float* __restrict__ pressure_fluid_nominator, float* __restrict__ pressure_fluid_denominator, float* __restrict__ mass_fluid, float* __restrict__ gradient_fluid, float* __restrict__ boundary_fluid, float* __restrict__ velocity_fluid, float* __restrict__ coupling_fluid) {
	const vec3 normal {normal_shared[particle_id_in_block - particle_offset][0], normal_shared[particle_id_in_block - particle_offset][1], normal_shared[particle_id_in_block - particle_offset][2]};
	//const SurfacePointType point_type = point_type_shared[particle_id_in_block - particle_offset];
	const float contact_area = contact_area_shared[particle_id_in_block - particle_offset];
	
	const vec3 pos {position_shared[particle_id_in_block - particle_offset][0], position_shared[particle_id_in_block - particle_offset][1], position_shared[particle_id_in_block - particle_offset][2]};
	const float mass = mass_shared[particle_id_in_block - particle_offset];
	const float J  = J_shared[particle_id_in_block - particle_offset];
	
	//Get position of grid cell
	const ivec3 global_base_index_solid_pressure = get_cell_id<INTERPOLATION_DEGREE_SOLID_PRESSURE>(pos.data_arr(), grid_solid.get_offset());
	const ivec3 global_base_index_solid_velocity = get_cell_id<INTERPOLATION_DEGREE_SOLID_VELOCITY>(pos.data_arr(), grid_solid.get_offset());
	const ivec3 global_base_index_solid_2 = get_cell_id<2>(pos.data_arr(), grid_solid.get_offset());
	
	const ivec3 global_base_index_fluid_velocity = get_cell_id<INTERPOLATION_DEGREE_FLUID_VELOCITY>(pos.data_arr(), grid_fluid.get_offset());//NOTE: Using solid/interface quadrature position
	
	const ivec3 global_base_index_interface_pressure = get_cell_id<INTERPOLATION_DEGREE_INTERFACE_PRESSURE>(pos.data_arr(), grid_solid.get_offset());
	

	//Get position relative to grid cell
	const vec3 local_pos_solid_pressure = pos - (global_base_index_solid_pressure + vec3(grid_solid.get_offset()[0], grid_solid.get_offset()[1], grid_solid.get_offset()[2])) * config::G_DX;
	const vec3 local_pos_solid_velocity = pos - (global_base_index_solid_velocity + vec3(grid_solid.get_offset()[0], grid_solid.get_offset()[1], grid_solid.get_offset()[2])) * config::G_DX;
	
	const vec3 local_pos_fluid_velocity = pos - (global_base_index_fluid_velocity + vec3(grid_fluid.get_offset()[0], grid_fluid.get_offset()[1], grid_fluid.get_offset()[2])) * config::G_DX;
	
	const vec3 local_pos_interface_pressure = pos - (global_base_index_interface_pressure + vec3(grid_solid.get_offset()[0], grid_solid.get_offset()[1], grid_solid.get_offset()[2])) * config::G_DX;

	//Calculate weights
	vec3x3 weight_solid_pressure;
	vec3x3 weight_solid_velocity;
	vec3x3 gradient_weight_solid_velocity;
	
	vec3x3 weight_fluid_velocity;
	vec3x3 gradient_weight_fluid_velocity;
	
	vec3x3 weight_interface_pressure;
	
	#pragma unroll 3
	for(int dd = 0; dd < 3; ++dd) {
		const std::array<float, INTERPOLATION_DEGREE_SOLID_PRESSURE + 1> current_weight_solid_pressure = bspline_weight<float, INTERPOLATION_DEGREE_SOLID_PRESSURE>(local_pos_solid_pressure[dd]);
		for(int i = 0; i < INTERPOLATION_DEGREE_SOLID_PRESSURE + 1; ++i){
			weight_solid_pressure(dd, i)		  = current_weight_solid_pressure[i];
		}
		for(int i = INTERPOLATION_DEGREE_SOLID_PRESSURE + 1; i < 3; ++i){
			weight_solid_pressure(dd, i)		  = 0.0f;
		}
		
		const std::array<float, INTERPOLATION_DEGREE_SOLID_VELOCITY + 1> current_weight_solid_velocity = bspline_weight<float, INTERPOLATION_DEGREE_SOLID_VELOCITY>(local_pos_solid_velocity[dd]);
		for(int i = 0; i < INTERPOLATION_DEGREE_SOLID_VELOCITY + 1; ++i){
			weight_solid_velocity(dd, i)		  = current_weight_solid_velocity[i];
		}
		for(int i = INTERPOLATION_DEGREE_SOLID_VELOCITY + 1; i < 3; ++i){
			weight_solid_velocity(dd, i)		  = 0.0f;
		}
		
		const std::array<float, INTERPOLATION_DEGREE_SOLID_VELOCITY + 1> current_gradient_weight_solid_velocity = bspline_gradient_weight<float, INTERPOLATION_DEGREE_SOLID_VELOCITY>(local_pos_solid_velocity[dd]);
		for(int i = 0; i < INTERPOLATION_DEGREE_SOLID_VELOCITY + 1; ++i){
			gradient_weight_solid_velocity(dd, i)		  = current_gradient_weight_solid_velocity[i];
		}
		for(int i = INTERPOLATION_DEGREE_SOLID_VELOCITY + 1; i < 3; ++i){
			gradient_weight_solid_velocity(dd, i)		  = 0.0f;
		}
		
		const std::array<float, INTERPOLATION_DEGREE_FLUID_VELOCITY + 1> current_weight_fluid_velocity = bspline_weight<float, INTERPOLATION_DEGREE_FLUID_VELOCITY>(local_pos_fluid_velocity[dd]);
		for(int i = 0; i < INTERPOLATION_DEGREE_FLUID_VELOCITY + 1; ++i){
			weight_fluid_velocity(dd, i)		  = current_weight_fluid_velocity[i];
		}
		for(int i = INTERPOLATION_DEGREE_FLUID_VELOCITY + 1; i < 3; ++i){
			weight_fluid_velocity(dd, i)		  = 0.0f;
		}
		
		const std::array<float, INTERPOLATION_DEGREE_FLUID_VELOCITY + 1> current_gradient_weight_fluid_velocity = bspline_gradient_weight<float, INTERPOLATION_DEGREE_FLUID_VELOCITY>(local_pos_fluid_velocity[dd]);
		for(int i = 0; i < INTERPOLATION_DEGREE_FLUID_VELOCITY + 1; ++i){
			gradient_weight_fluid_velocity(dd, i)		  = current_gradient_weight_fluid_velocity[i];
		}
		for(int i = INTERPOLATION_DEGREE_FLUID_VELOCITY + 1; i < 3; ++i){
			gradient_weight_fluid_velocity(dd, i)		  = 0.0f;
		}
		
		const std::array<float, INTERPOLATION_DEGREE_INTERFACE_PRESSURE + 1> current_weight_interface_pressure = bspline_weight<float, INTERPOLATION_DEGREE_INTERFACE_PRESSURE>(local_pos_interface_pressure[dd]);
		for(int i = 0; i < INTERPOLATION_DEGREE_INTERFACE_PRESSURE + 1; ++i){
			weight_interface_pressure(dd, i)		  = current_weight_interface_pressure[i];
		}
		for(int i = INTERPOLATION_DEGREE_INTERFACE_PRESSURE + 1; i < 3; ++i){
			weight_interface_pressure(dd, i)		  = 0.0f;
		}
	}
	
	//Get near fluid particles
	__shared__ float mass_fluid_total;
	__shared__ std::array<float, 3> momentum_fluid_local;
	
	if(threadIdx.x == 0){
		mass_fluid_total = 0.0f;
		momentum_fluid_local[0] = 0.0f;
		momentum_fluid_local[1] = 0.0f;
		momentum_fluid_local[2] = 0.0f;
	}
	
	__syncthreads();
	
	for(int grid_x = -4; grid_x <= 1; ++grid_x){
		for(int grid_y = -4; grid_y <= 1; ++grid_y){
			for(int grid_z = -4; grid_z <= 1; ++grid_z){
				const ivec3 cell_offset {grid_x, grid_y, grid_z};
				const ivec3 current_cellid = global_base_index_solid_2 + cell_offset;
				const ivec3 current_blockid = current_cellid / static_cast<int>(config::G_BLOCKSIZE);
				const int current_blockno_fluid = prev_partition.query(current_blockid);
				
				//Skip empty blocks
				if(current_blockno_fluid == -1){
					continue;
				}
				
				for(int particle_id_in_block_fluid = static_cast<int>(threadIdx.x); particle_id_in_block_fluid <  next_particle_buffer_fluid.particle_bucket_sizes[current_blockno_fluid]; particle_id_in_block_fluid += static_cast<int>(blockDim.x)) {
					//Fetch index of the advection source
					int advection_source_blockno_fluid;
					int source_pidib_fluid;
					{
						//Fetch advection (direction at high bits, particle in in cell at low bits)
						const int advect = next_particle_buffer_fluid.blockbuckets[current_blockno_fluid * config::G_PARTICLE_NUM_PER_BLOCK + particle_id_in_block_fluid];

						//Retrieve the direction (first stripping the particle id by division)
						ivec3 offset;
						dir_components<3>(advect / config::G_PARTICLE_NUM_PER_BLOCK, offset.data_arr());

						//Retrieve the particle id by AND for lower bits
						source_pidib_fluid = advect & (config::G_PARTICLE_NUM_PER_BLOCK - 1);

						//Get global index by adding blockid and offset
						const ivec3 global_advection_index = current_blockid + offset;

						//Get block_no from partition
						advection_source_blockno_fluid = prev_partition.query(global_advection_index);
					}
					
					auto fluid_particle_bin = iq_fluid_particle_buffer.ch(_0, particle_buffer_fluid.bin_offsets[advection_source_blockno_fluid] + source_pidib_fluid / config::G_BIN_CAPACITY);
					const int particle_id_in_bin = source_pidib_fluid  % config::G_BIN_CAPACITY;

					//Fetch position and determinant of deformation gradient
					FetchParticleBufferDataIntermediate fetch_particle_buffer_tmp = {};
					fetch_particle_buffer_data<MaterialTypeFluid>(particle_buffer_fluid, advection_source_blockno_fluid, source_pidib_fluid, fetch_particle_buffer_tmp);
					const float mass_fluid = fetch_particle_buffer_tmp.mass;
					vec3 pos_fluid {fetch_particle_buffer_tmp.pos[0], fetch_particle_buffer_tmp.pos[1], fetch_particle_buffer_tmp.pos[2]};
					//float J_fluid	 = fetch_particle_buffer_tmp.J;
					
					
					vec3 velocity_fluid;
					velocity_fluid[0] = fluid_particle_bin.val(_0, particle_id_in_bin);
					velocity_fluid[1] = fluid_particle_bin.val(_1, particle_id_in_bin);
					velocity_fluid[2] = fluid_particle_bin.val(_2, particle_id_in_bin);
					
					const int count_neighbours = fluid_particle_bin.val(_12, particle_id_in_bin);
					
					const vec3 diff = pos_fluid - pos;//NOTE: Same order as in other neighbour check to ensure same results
					const float distance = std::sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
					
					if(distance <= 0.5f * config::G_DX){
						atomicAdd(&mass_fluid_total, mass_fluid / static_cast<float>(count_neighbours));
						atomicAdd(&(momentum_fluid_local[0]), velocity_fluid[0] * mass_fluid / static_cast<float>(count_neighbours));
						atomicAdd(&(momentum_fluid_local[1]), velocity_fluid[1] * mass_fluid / static_cast<float>(count_neighbours));
						atomicAdd(&(momentum_fluid_local[2]), velocity_fluid[2] * mass_fluid / static_cast<float>(count_neighbours));
					}
				}
			}
		}
	}
	
	__syncthreads();
	
	//Store data
	//Note: Weights are 0 if outside of interpolation degree/radius around particles cell
	//Foreach node in the block we add values accoring to particle kernel, also handling all neighbours of the particles cell
	
	for(size_t local_cell_index = 0; local_cell_index < get_thread_count<BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, config::G_BLOCKVOLUME); local_cell_index++){
		const size_t cell_index = get_global_index<BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_cell_index);
		const ivec3 local_id {static_cast<int>((cell_index / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE), static_cast<int>((cell_index / config::G_BLOCKSIZE) % config::G_BLOCKSIZE), static_cast<int>(cell_index % config::G_BLOCKSIZE)};
		
		const ivec3 local_offset_pressure = global_base_index_solid_pressure - (block_cellid + local_id);
		const ivec3 local_offset_velocity_solid = global_base_index_solid_velocity - (block_cellid + local_id);
		const ivec3 local_offset_velocity_fluid = global_base_index_fluid_velocity - (block_cellid + local_id);
			
		const ivec3 absolute_local_offset_pressure {std::abs(local_offset_pressure[0]), std::abs(local_offset_pressure[1]), std::abs(local_offset_pressure[2])};
		const ivec3 absolute_local_offset_velocity_solid {std::abs(local_offset_velocity_solid[0]), std::abs(local_offset_velocity_solid[1]), std::abs(local_offset_velocity_solid[2])};
		const ivec3 absolute_local_offset_velocity_fluid {std::abs(local_offset_velocity_fluid[0]), std::abs(local_offset_velocity_fluid[1]), std::abs(local_offset_velocity_fluid[2])};

		//Weight
		const float W_pressure = (absolute_local_offset_pressure[0] < 3 ? weight_solid_pressure(0, absolute_local_offset_pressure[0]) : 0.0f) * (absolute_local_offset_pressure[1] < 3 ? weight_solid_pressure(1, absolute_local_offset_pressure[1]) : 0.0f) * (absolute_local_offset_pressure[2] < 3 ? weight_solid_pressure(2, absolute_local_offset_pressure[2]) : 0.0f);
		const float W_velocity_solid = (absolute_local_offset_velocity_solid[0] < 3 ? weight_solid_velocity(0, absolute_local_offset_velocity_solid[0]) : 0.0f) * (absolute_local_offset_velocity_solid[1] < 3 ? weight_solid_velocity(1, absolute_local_offset_velocity_solid[1]) : 0.0f) * (absolute_local_offset_velocity_solid[2] < 3 ? weight_solid_velocity(2, absolute_local_offset_velocity_solid[2]) : 0.0f);
		const float W_velocity_fluid = (absolute_local_offset_velocity_fluid[0] < 3 ? weight_fluid_velocity(0, absolute_local_offset_velocity_fluid[0]) : 0.0f) * (absolute_local_offset_velocity_fluid[1] < 3 ? weight_fluid_velocity(1, absolute_local_offset_velocity_fluid[1]) : 0.0f) * (absolute_local_offset_velocity_fluid[2] < 3 ? weight_fluid_velocity(2, absolute_local_offset_velocity_fluid[2]) : 0.0f);
		
		float* current_scaling_solid = &(scaling_solid[local_cell_index]);
		float* current_pressure_solid_nominator = &(pressure_solid_nominator[local_cell_index]);
		float* current_pressure_solid_denominator = &(pressure_solid_denominator[local_cell_index]);
		
		float* current_scaling_fluid = &(scaling_fluid[local_cell_index]);
		float* current_pressure_fluid_nominator = &(pressure_fluid_nominator[local_cell_index]);
		float* current_pressure_fluid_denominator = &(pressure_fluid_denominator[local_cell_index]);
		
		store_data_solid(particle_buffer_solid, current_scaling_solid, current_pressure_solid_nominator, current_pressure_solid_denominator, W_pressure, W_velocity_solid, mass, J);
		
		if(mass_fluid_total > 0.0f){
			store_data_fluid(particle_buffer_fluid, current_scaling_fluid, current_pressure_fluid_nominator, current_pressure_fluid_denominator, W_pressure, W_velocity_fluid, mass_fluid_total, J);
		}
	}
	
	for(size_t local_cell_index = 0; local_cell_index < get_thread_count<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, 3 * config::G_BLOCKVOLUME); local_cell_index++){
		const size_t cell_index = get_global_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_cell_index) / 3;
		const size_t alpha = get_global_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_cell_index) % 3;
		const ivec3 local_id {static_cast<int>((cell_index / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE), static_cast<int>((cell_index / config::G_BLOCKSIZE) % config::G_BLOCKSIZE), static_cast<int>(cell_index % config::G_BLOCKSIZE)};
		
		const ivec3 local_offset_velocity_solid = global_base_index_solid_velocity - (block_cellid + local_id);
		const ivec3 local_offset_velocity_fluid = global_base_index_fluid_velocity - (block_cellid + local_id);
		
		const ivec3 absolute_local_offset_velocity_solid {std::abs(local_offset_velocity_solid[0]), std::abs(local_offset_velocity_solid[1]), std::abs(local_offset_velocity_solid[2])};
		const ivec3 absolute_local_offset_velocity_fluid {std::abs(local_offset_velocity_fluid[0]), std::abs(local_offset_velocity_fluid[1]), std::abs(local_offset_velocity_fluid[2])};

		//Weight
		const float W_velocity_solid = (absolute_local_offset_velocity_solid[0] < 3 ? weight_solid_velocity(0, absolute_local_offset_velocity_solid[0]) : 0.0f) * (absolute_local_offset_velocity_solid[1] < 3 ? weight_solid_velocity(1, absolute_local_offset_velocity_solid[1]) : 0.0f) * (absolute_local_offset_velocity_solid[2] < 3 ? weight_solid_velocity(2, absolute_local_offset_velocity_solid[2]) : 0.0f);
		const float W_velocity_fluid = (absolute_local_offset_velocity_fluid[0] < 3 ? weight_fluid_velocity(0, absolute_local_offset_velocity_fluid[0]) : 0.0f) * (absolute_local_offset_velocity_fluid[1] < 3 ? weight_fluid_velocity(1, absolute_local_offset_velocity_fluid[1]) : 0.0f) * (absolute_local_offset_velocity_fluid[2] < 3 ? weight_fluid_velocity(2, absolute_local_offset_velocity_fluid[2]) : 0.0f);
		
		mass_solid[local_cell_index] += mass * W_velocity_solid;
		
		if(mass_fluid_total > 0.0f){
			mass_fluid[local_cell_index] += mass_fluid_total * W_velocity_fluid;
			
			//printf("TMP1 %d %d # %d # %.28f # %.28f\n", static_cast<int>(blockIdx.x), static_cast<int>(cell_index), static_cast<int>(alpha), mass_fluid_total, momentum_fluid_local[alpha]);
			
			//Increase grid momentum by particle momentum
			velocity_fluid[local_cell_index] += W_velocity_fluid * momentum_fluid_local[alpha];
		}
	}
	
	for(size_t local_cell_index = 0; local_cell_index < get_thread_count<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, 3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME); local_cell_index++){
		const size_t cell_index = get_global_index<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_cell_index) / (3 * NUM_COLUMNS_PER_BLOCK);
		const size_t column = (get_global_index<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_cell_index) / 3) % NUM_COLUMNS_PER_BLOCK;
		const size_t alpha = get_global_index<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_cell_index) % 3;
		const ivec3 local_id {static_cast<int>((cell_index / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE), static_cast<int>((cell_index / config::G_BLOCKSIZE) % config::G_BLOCKSIZE), static_cast<int>(cell_index % config::G_BLOCKSIZE)};
		const ivec3 neighbour_local_id = ivec3(static_cast<int>((column / ((2 * INTERPOLATION_DEGREE_MAX + 1) * (2 * INTERPOLATION_DEGREE_MAX + 1))) % (2 * INTERPOLATION_DEGREE_MAX + 1)), static_cast<int>((column / (2 * INTERPOLATION_DEGREE_MAX + 1)) % (2 * INTERPOLATION_DEGREE_MAX + 1)), static_cast<int>(column % (2 * INTERPOLATION_DEGREE_MAX + 1))) - ivec3(static_cast<int>(INTERPOLATION_DEGREE_MAX), static_cast<int>(INTERPOLATION_DEGREE_MAX), static_cast<int>(INTERPOLATION_DEGREE_MAX));
			
		const ivec3 local_offset_velocity_solid = global_base_index_solid_velocity - (block_cellid + local_id);
		const ivec3 local_offset_velocity_fluid = global_base_index_fluid_velocity - (block_cellid + local_id);
		const ivec3 neighbour_local_offset_pressure = global_base_index_solid_pressure - (block_cellid + local_id + neighbour_local_id);
		const ivec3 neighbour_local_offset_pressure_interface = global_base_index_interface_pressure - (block_cellid + local_id + neighbour_local_id);
		
		const ivec3 absolute_local_offset_velocity_solid {std::abs(local_offset_velocity_solid[0]), std::abs(local_offset_velocity_solid[1]), std::abs(local_offset_velocity_solid[2])};
		const ivec3 absolute_local_offset_velocity_fluid {std::abs(local_offset_velocity_fluid[0]), std::abs(local_offset_velocity_fluid[1]), std::abs(local_offset_velocity_fluid[2])};
		const ivec3 neighbour_absolute_local_offset {std::abs(neighbour_local_offset_pressure[0]), std::abs(neighbour_local_offset_pressure[1]), std::abs(neighbour_local_offset_pressure[2])};
		const ivec3 neighbour_absolute_local_offset_pressure_interface {std::abs(neighbour_local_offset_pressure_interface[0]), std::abs(neighbour_local_offset_pressure_interface[1]), std::abs(neighbour_local_offset_pressure_interface[2])};			

		//Weight
		const float delta_W_velocity_solid = ((alpha == 0 ? (absolute_local_offset_velocity_solid[0] < 3 ? gradient_weight_solid_velocity(0, absolute_local_offset_velocity_solid[0]) : 0.0f) : (absolute_local_offset_velocity_solid[0] < 3 ? weight_solid_velocity(0, absolute_local_offset_velocity_solid[0]) : 0.0f)) * (alpha == 1 ? (absolute_local_offset_velocity_solid[1] < 3 ? gradient_weight_solid_velocity(1, absolute_local_offset_velocity_solid[1]) : 0.0f) : (absolute_local_offset_velocity_solid[1] < 3 ? weight_solid_velocity(1, absolute_local_offset_velocity_solid[1]) : 0.0f)) * (alpha == 2 ? (absolute_local_offset_velocity_solid[2] < 3 ? gradient_weight_solid_velocity(2, absolute_local_offset_velocity_solid[2]) : 0.0f) : (absolute_local_offset_velocity_solid[2] < 3 ? weight_solid_velocity(2, absolute_local_offset_velocity_solid[2]) : 0.0f))) * config::G_DX_INV;
		const float delta_W_velocity_fluid = ((alpha == 0 ? (absolute_local_offset_velocity_fluid[0] < 3 ? gradient_weight_fluid_velocity(0, absolute_local_offset_velocity_fluid[0]) : 0.0f) : (absolute_local_offset_velocity_fluid[0] < 3 ? weight_fluid_velocity(0, absolute_local_offset_velocity_fluid[0]) : 0.0f)) * (alpha == 1 ? (absolute_local_offset_velocity_fluid[1] < 3 ? gradient_weight_fluid_velocity(1, absolute_local_offset_velocity_fluid[1]) : 0.0f) : (absolute_local_offset_velocity_fluid[1] < 3 ? weight_fluid_velocity(1, absolute_local_offset_velocity_fluid[1]) : 0.0f)) * (alpha == 2 ? (absolute_local_offset_velocity_fluid[2] < 3 ? gradient_weight_fluid_velocity(2, absolute_local_offset_velocity_fluid[2]) : 0.0f) : (absolute_local_offset_velocity_fluid[2] < 3 ? weight_fluid_velocity(2, absolute_local_offset_velocity_fluid[2]) : 0.0f))) * config::G_DX_INV;
		const float W_velocity_solid = (absolute_local_offset_velocity_solid[0] < 3 ? weight_solid_velocity(0, absolute_local_offset_velocity_solid[0]) : 0.0f) * (absolute_local_offset_velocity_solid[1] < 3 ? weight_solid_velocity(1, absolute_local_offset_velocity_solid[1]) : 0.0f) * (absolute_local_offset_velocity_solid[2] < 3 ? weight_solid_velocity(2, absolute_local_offset_velocity_solid[2]) : 0.0f);	
		const float W_velocity_fluid = (absolute_local_offset_velocity_fluid[0] < 3 ? weight_fluid_velocity(0, absolute_local_offset_velocity_fluid[0]) : 0.0f) * (absolute_local_offset_velocity_fluid[1] < 3 ? weight_fluid_velocity(1, absolute_local_offset_velocity_fluid[1]) : 0.0f) * (absolute_local_offset_velocity_fluid[2] < 3 ? weight_fluid_velocity(2, absolute_local_offset_velocity_fluid[2]) : 0.0f);
		const float W1_pressure = (neighbour_absolute_local_offset[0] < 3 ? weight_solid_pressure(0, neighbour_absolute_local_offset[0]) : 0.0f) * (neighbour_absolute_local_offset[1] < 3 ? weight_solid_pressure(1, neighbour_absolute_local_offset[1]) : 0.0f) * (neighbour_absolute_local_offset[2] < 3 ? weight_solid_pressure(2, neighbour_absolute_local_offset[2]) : 0.0f);
		const float W1_pressure_interface = (neighbour_absolute_local_offset_pressure_interface[0] < 3 ? weight_interface_pressure(0, neighbour_absolute_local_offset_pressure_interface[0]) : 0.0f) * (neighbour_absolute_local_offset_pressure_interface[1] < 3 ? weight_interface_pressure(1, neighbour_absolute_local_offset_pressure_interface[1]) : 0.0f) * (neighbour_absolute_local_offset_pressure_interface[2] < 3 ? weight_interface_pressure(2, neighbour_absolute_local_offset_pressure_interface[2]) : 0.0f);					
				
		float* current_gradient_solid = &(gradient_solid[local_cell_index]);
		
		float* current_gradient_fluid = &(gradient_fluid[local_cell_index]);
		float* current_boundary_fluid = &(boundary_fluid[local_cell_index]);
		
		float* current_coupling_solid = &(coupling_solid[local_cell_index]);
		float* current_coupling_fluid = &(coupling_fluid[local_cell_index]);

		store_data_neigbours_solid(particle_buffer_solid, current_gradient_solid, W1_pressure, delta_W_velocity_solid, mass, J);
		
		if(mass_fluid_total > 0.0f){
			store_data_neigbours_fluid(particle_buffer_fluid, current_gradient_fluid, current_boundary_fluid, W1_pressure, delta_W_velocity_fluid, mass_fluid_total);

			//Coupling
			store_data_neigbours_coupling_solid(particle_buffer_solid, current_coupling_solid, W_velocity_solid, W1_pressure_interface, contact_area, normal[alpha]);
			store_data_neigbours_coupling_fluid(particle_buffer_fluid, current_coupling_fluid, W_velocity_fluid, W1_pressure_interface, contact_area, normal[alpha]);
		}
	}
}

template<typename Partition, typename Grid, MaterialE MaterialTypeSolid, MaterialE MaterialTypeFluid>
__forceinline__ __device__ void aggregate_data_fluid(const ParticleBuffer<MaterialTypeSolid> particle_buffer_solid, const ParticleBuffer<MaterialTypeFluid> particle_buffer_fluid, const ParticleBuffer<MaterialTypeSolid> next_particle_buffer_solid, const ParticleBuffer<MaterialTypeFluid> next_particle_buffer_fluid, const Partition prev_partition, const Grid grid_solid, const Grid grid_fluid, const std::array<float, 3>* __restrict__ position_shared, const float* __restrict__ mass_shared, const float* __restrict__ J_shared, const std::array<float, 3>* __restrict__ velocity_shared, const std::array<float, 9>* __restrict__ C_shared, int* __restrict__ count_neighbours_shared, const int particle_offset, const int current_blockno, const ivec3 current_blockid, const ivec3 block_cellid, const int particle_id_in_block, float* __restrict__ scaling_fluid, float* __restrict__ pressure_fluid_nominator, float* __restrict__ pressure_fluid_denominator, float* __restrict__ mass_fluid, float* __restrict__ gradient_fluid, float* __restrict__ boundary_fluid, float* __restrict__ velocity_fluid) {
	const vec3 pos {position_shared[particle_id_in_block - particle_offset][0], position_shared[particle_id_in_block - particle_offset][1], position_shared[particle_id_in_block - particle_offset][2]};
	const float mass = mass_shared[particle_id_in_block - particle_offset];
	const float J  = J_shared[particle_id_in_block - particle_offset];
	
	const vec3 velocity {velocity_shared[particle_id_in_block - particle_offset][0], velocity_shared[particle_id_in_block - particle_offset][1], velocity_shared[particle_id_in_block - particle_offset][2]};
	const std::array<float, 9> C = C_shared[particle_id_in_block - particle_offset];
	
	//Get position of grid cell
	const ivec3 global_base_index_solid_pressure = get_cell_id<INTERPOLATION_DEGREE_FLUID_PRESSURE>(pos.data_arr(), grid_solid.get_offset());
	const ivec3 global_base_index_fluid_velocity = get_cell_id<INTERPOLATION_DEGREE_FLUID_VELOCITY>(pos.data_arr(), grid_fluid.get_offset());
	
	const ivec3 global_base_index_fluid_2 = get_cell_id<2>(pos.data_arr(), grid_fluid.get_offset());
	
	//Get position relative to grid cell
	const vec3 local_pos_solid_pressure = pos - (global_base_index_solid_pressure + vec3(grid_solid.get_offset()[0], grid_solid.get_offset()[1], grid_solid.get_offset()[2])) * config::G_DX;
	const vec3 local_pos_fluid_velocity = pos - (global_base_index_fluid_velocity + vec3(grid_fluid.get_offset()[0], grid_fluid.get_offset()[1], grid_fluid.get_offset()[2])) * config::G_DX;

	//Calculate weights
	vec3x3 weight_solid_pressure;
	vec3x3 weight_fluid_velocity;
	vec3x3 gradient_weight_fluid_velocity;
	
	#pragma unroll 3
	for(int dd = 0; dd < 3; ++dd) {
		const std::array<float, INTERPOLATION_DEGREE_FLUID_PRESSURE + 1> current_weight_solid_pressure = bspline_weight<float, INTERPOLATION_DEGREE_FLUID_PRESSURE>(local_pos_solid_pressure[dd]);
		for(int i = 0; i < INTERPOLATION_DEGREE_FLUID_PRESSURE + 1; ++i){
			weight_solid_pressure(dd, i)		  = current_weight_solid_pressure[i];
		}
		for(int i = INTERPOLATION_DEGREE_FLUID_PRESSURE + 1; i < 3; ++i){
			weight_solid_pressure(dd, i)		  = 0.0f;
		}
		
		const std::array<float, INTERPOLATION_DEGREE_FLUID_VELOCITY + 1> current_weight_fluid_velocity = bspline_weight<float, INTERPOLATION_DEGREE_FLUID_VELOCITY>(local_pos_fluid_velocity[dd]);
		for(int i = 0; i < INTERPOLATION_DEGREE_FLUID_VELOCITY + 1; ++i){
			weight_fluid_velocity(dd, i)		  = current_weight_fluid_velocity[i];
		}
		for(int i = INTERPOLATION_DEGREE_FLUID_VELOCITY + 1; i < 3; ++i){
			weight_fluid_velocity(dd, i)		  = 0.0f;
		}
		
		const std::array<float, INTERPOLATION_DEGREE_FLUID_VELOCITY + 1> current_gradient_weight_fluid_velocity = bspline_gradient_weight<float, INTERPOLATION_DEGREE_FLUID_VELOCITY>(local_pos_fluid_velocity[dd]);
		for(int i = 0; i < INTERPOLATION_DEGREE_FLUID_VELOCITY + 1; ++i){
			gradient_weight_fluid_velocity(dd, i)		  = current_gradient_weight_fluid_velocity[i];
		}
		for(int i = INTERPOLATION_DEGREE_FLUID_VELOCITY + 1; i < 3; ++i){
			gradient_weight_fluid_velocity(dd, i)		  = 0.0f;
		}
	}
	
	//Get near fluid particles
	for(int grid_x = -4; grid_x <= 1; ++grid_x){
		for(int grid_y = -4; grid_y <= 1; ++grid_y){
			for(int grid_z = -4; grid_z <= 1; ++grid_z){
				const ivec3 cell_offset {grid_x, grid_y, grid_z};
				const ivec3 current_cellid = global_base_index_fluid_2 + cell_offset;
				const ivec3 current_blockid = current_cellid / static_cast<int>(config::G_BLOCKSIZE);
				const int current_blockno_solid = prev_partition.query(current_blockid);
				
				//Skip empty blocks
				if(current_blockno_solid == -1){
					continue;
				}
				
				for(int particle_id_in_block_solid = static_cast<int>(threadIdx.x); particle_id_in_block_solid <  next_particle_buffer_solid.particle_bucket_sizes[current_blockno_solid]; particle_id_in_block_solid += static_cast<int>(blockDim.x)) {
					//Fetch index of the advection source
					int advection_source_blockno_solid;
					int source_pidib_solid;
					{
						//Fetch advection (direction at high bits, particle in in cell at low bits)
						const int advect = next_particle_buffer_solid.blockbuckets[current_blockno_solid * config::G_PARTICLE_NUM_PER_BLOCK + particle_id_in_block_solid];

						//Retrieve the direction (first stripping the particle id by division)
						ivec3 offset;
						dir_components<3>(advect / config::G_PARTICLE_NUM_PER_BLOCK, offset.data_arr());

						//Retrieve the particle id by AND for lower bits
						source_pidib_solid = advect & (config::G_PARTICLE_NUM_PER_BLOCK - 1);

						//Get global index by adding blockid and offset
						const ivec3 global_advection_index = current_blockid + offset;

						//Get block_no from partition
						advection_source_blockno_solid = prev_partition.query(global_advection_index);
					}

					//Fetch position and determinant of deformation gradient
					FetchParticleBufferDataIntermediate fetch_particle_buffer_tmp = {};
					fetch_particle_buffer_data<MaterialTypeSolid>(particle_buffer_solid, advection_source_blockno_solid, source_pidib_solid, fetch_particle_buffer_tmp);
					//const float mass_solid = fetch_particle_buffer_tmp.mass;
					vec3 pos_solid {fetch_particle_buffer_tmp.pos[0], fetch_particle_buffer_tmp.pos[1], fetch_particle_buffer_tmp.pos[2]};
					//float J_solid	 = fetch_particle_buffer_tmp.J;
					
					const vec3 diff = pos - pos_solid;//NOTE: Same order as in other neighbour check to ensure same results
					const float distance = std::sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
					
					if(distance <= 0.5f * config::G_DX){
						atomicAdd(&(count_neighbours_shared[particle_id_in_block - particle_offset]), 1);
					}
				}
			}
		}
	}
	
	__syncthreads();
	
	const float reduced_mass = mass / static_cast<float>(count_neighbours_shared[particle_id_in_block - particle_offset]);
	
	//Store data
	//Note: Weights are 0 if outside of interpolation degree/radius around particles cell
	//Foreach node in the block we add values accoring to particle kernel, also handling all neighbours of the particles cell
	
	for(size_t local_cell_index = 0; local_cell_index < get_thread_count<BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, config::G_BLOCKVOLUME); local_cell_index++){
		const size_t cell_index = get_global_index<BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_cell_index);
		const ivec3 local_id {static_cast<int>((cell_index / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE), static_cast<int>((cell_index / config::G_BLOCKSIZE) % config::G_BLOCKSIZE), static_cast<int>(cell_index % config::G_BLOCKSIZE)};
		
		const ivec3 local_offset_pressure = global_base_index_solid_pressure - (block_cellid + local_id);
		const ivec3 local_offset_velocity = global_base_index_fluid_velocity - (block_cellid + local_id);
		
		const ivec3 absolute_local_offset_pressure {std::abs(local_offset_pressure[0]), std::abs(local_offset_pressure[1]), std::abs(local_offset_pressure[2])};
		const ivec3 absolute_local_offset_velocity {std::abs(local_offset_velocity[0]), std::abs(local_offset_velocity[1]), std::abs(local_offset_velocity[2])};

		//Weight
		const float W_pressure = (absolute_local_offset_pressure[0] < 3 ? weight_solid_pressure(0, absolute_local_offset_pressure[0]) : 0.0f) * (absolute_local_offset_pressure[1] < 3 ? weight_solid_pressure(1, absolute_local_offset_pressure[1]) : 0.0f) * (absolute_local_offset_pressure[2] < 3 ? weight_solid_pressure(2, absolute_local_offset_pressure[2]) : 0.0f);
		const float W_velocity = (absolute_local_offset_velocity[0] < 3 ? weight_fluid_velocity(0, absolute_local_offset_velocity[0]) : 0.0f) * (absolute_local_offset_velocity[1] < 3 ? weight_fluid_velocity(1, absolute_local_offset_velocity[1]) : 0.0f) * (absolute_local_offset_velocity[2] < 3 ? weight_fluid_velocity(2, absolute_local_offset_velocity[2]) : 0.0f);
		
		float* current_scaling_fluid = &(scaling_fluid[local_cell_index]);
		float* current_pressure_fluid_nominator = &(pressure_fluid_nominator[local_cell_index]);
		float* current_pressure_fluid_denominator = &(pressure_fluid_denominator[local_cell_index]);
		
		store_data_fluid(particle_buffer_fluid, current_scaling_fluid, current_pressure_fluid_nominator, current_pressure_fluid_denominator, W_pressure, W_velocity, reduced_mass, J);
	}
	
	for(size_t local_cell_index = 0; local_cell_index < get_thread_count<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, 3 * config::G_BLOCKVOLUME); local_cell_index++){
		const size_t cell_index = get_global_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_cell_index) / 3;
		const size_t alpha = get_global_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_cell_index) % 3;
		const ivec3 local_id {static_cast<int>((cell_index / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE), static_cast<int>((cell_index / config::G_BLOCKSIZE) % config::G_BLOCKSIZE), static_cast<int>(cell_index % config::G_BLOCKSIZE)};
		
		const ivec3 local_offset_velocity_fluid = global_base_index_fluid_velocity - (block_cellid + local_id);
		
		const ivec3 absolute_local_offset_velocity_fluid {std::abs(local_offset_velocity_fluid[0]), std::abs(local_offset_velocity_fluid[1]), std::abs(local_offset_velocity_fluid[2])};

		const vec3 xixp = local_id * config::G_DX - local_pos_fluid_velocity;
			
		//Weight
		const float W_velocity = (absolute_local_offset_velocity_fluid[0] < 3 ? weight_fluid_velocity(0, absolute_local_offset_velocity_fluid[0]) : 0.0f) * (absolute_local_offset_velocity_fluid[1] < 3 ? weight_fluid_velocity(1, absolute_local_offset_velocity_fluid[1]) : 0.0f) * (absolute_local_offset_velocity_fluid[2] < 3 ? weight_fluid_velocity(2, absolute_local_offset_velocity_fluid[2]) : 0.0f);
				
		mass_fluid[local_cell_index] += reduced_mass * W_velocity;
		
		if(count_neighbours_shared[particle_id_in_block - particle_offset] > 1){
			//printf("TMP0 %d %d # %d # %.28f # %.28f\n", static_cast<int>(blockIdx.x), static_cast<int>(cell_index), static_cast<int>(alpha), reduced_mass, (reduced_mass * velocity[alpha]));
			
			//Increase grid momentum by particle momentum
			velocity_fluid[local_cell_index] += W_velocity * (reduced_mass * velocity[alpha] - (C[alpha] * xixp[0] + C[alpha + 1] * xixp[1] + C[alpha + 2] * xixp[2]));
		}
	}
	
	for(size_t local_cell_index = 0; local_cell_index < get_thread_count<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, 3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME); local_cell_index++){
		const size_t cell_index = get_global_index<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_cell_index) / (3 * NUM_COLUMNS_PER_BLOCK);
		const size_t column = (get_global_index<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_cell_index) / 3) % NUM_COLUMNS_PER_BLOCK;
		const size_t alpha = get_global_index<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_cell_index) % 3;
		const ivec3 local_id {static_cast<int>((cell_index / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE), static_cast<int>((cell_index / config::G_BLOCKSIZE) % config::G_BLOCKSIZE), static_cast<int>(cell_index % config::G_BLOCKSIZE)};
		const ivec3 neighbour_local_id = ivec3(static_cast<int>((column / ((2 * INTERPOLATION_DEGREE_MAX + 1) * (2 * INTERPOLATION_DEGREE_MAX + 1))) % (2 * INTERPOLATION_DEGREE_MAX + 1)), static_cast<int>((column / (2 * INTERPOLATION_DEGREE_MAX + 1)) % (2 * INTERPOLATION_DEGREE_MAX + 1)), static_cast<int>(column % (2 * INTERPOLATION_DEGREE_MAX + 1))) - ivec3(static_cast<int>(INTERPOLATION_DEGREE_MAX), static_cast<int>(INTERPOLATION_DEGREE_MAX), static_cast<int>(INTERPOLATION_DEGREE_MAX));
			
		const ivec3 local_offset_velocity_fluid = global_base_index_fluid_velocity - (block_cellid + local_id);
		const ivec3 neighbour_local_offset_pressure = global_base_index_solid_pressure - (block_cellid + local_id + neighbour_local_id);
		
		const ivec3 absolute_local_offset_velocity_fluid {std::abs(local_offset_velocity_fluid[0]), std::abs(local_offset_velocity_fluid[1]), std::abs(local_offset_velocity_fluid[2])};
		const ivec3 neighbour_absolute_local_offset {std::abs(neighbour_local_offset_pressure[0]), std::abs(neighbour_local_offset_pressure[1]), std::abs(neighbour_local_offset_pressure[2])};												

		//Weight
		const float delta_W_velocity = ((alpha == 0 ? (absolute_local_offset_velocity_fluid[0] < 3 ? gradient_weight_fluid_velocity(0, absolute_local_offset_velocity_fluid[0]) : 0.0f) : (absolute_local_offset_velocity_fluid[0] < 3 ? weight_fluid_velocity(0, absolute_local_offset_velocity_fluid[0]) : 0.0f)) * (alpha == 1 ? (absolute_local_offset_velocity_fluid[1] < 3 ? gradient_weight_fluid_velocity(1, absolute_local_offset_velocity_fluid[1]) : 0.0f) : (absolute_local_offset_velocity_fluid[1] < 3 ? weight_fluid_velocity(1, absolute_local_offset_velocity_fluid[1]) : 0.0f)) * (alpha == 2 ? (absolute_local_offset_velocity_fluid[2] < 3 ? gradient_weight_fluid_velocity(2, absolute_local_offset_velocity_fluid[2]) : 0.0f) : (absolute_local_offset_velocity_fluid[2] < 3 ? weight_fluid_velocity(2, absolute_local_offset_velocity_fluid[2]) : 0.0f))) * config::G_DX_INV;
		const float W1_pressure = (neighbour_absolute_local_offset[0] < 3 ? weight_solid_pressure(0, neighbour_absolute_local_offset[0]) : 0.0f) * (neighbour_absolute_local_offset[1] < 3 ? weight_solid_pressure(1, neighbour_absolute_local_offset[1]) : 0.0f) * (neighbour_absolute_local_offset[2] < 3 ? weight_solid_pressure(2, neighbour_absolute_local_offset[2]) : 0.0f);
									
		float* current_gradient_fluid = &(gradient_fluid[local_cell_index]);
		float* current_boundary_fluid = &(boundary_fluid[local_cell_index]);
		
		store_data_neigbours_fluid(particle_buffer_fluid, current_gradient_fluid, current_boundary_fluid, W1_pressure, delta_W_velocity, reduced_mass);
	}
}

//TODO: Directly store into matrices, not into local memory
template<typename Partition, typename Grid, MaterialE MaterialTypeSolid, MaterialE MaterialTypeFluid>
__global__ void create_iq_system(const uint32_t num_blocks, Duration dt, const ParticleBuffer<MaterialTypeSolid> particle_buffer_solid, const ParticleBuffer<MaterialTypeFluid> particle_buffer_fluid, const ParticleBuffer<MaterialTypeSolid> next_particle_buffer_solid, const ParticleBuffer<MaterialTypeFluid> next_particle_buffer_fluid, const Partition prev_partition, const Partition partition, const Grid grid_solid, const Grid grid_fluid, FluidParticleBuffer iq_fluid_particle_buffer, const SurfaceParticleBuffer surface_particle_buffer_solid, const SurfaceParticleBuffer surface_particle_buffer_fluid, IQCreatePointers iq_pointers) {
	//Particles with offset [-2, 0] can lie within cell (due to storing with interpolation degree 2 wich results in offset of 2); Interolation degree may offset positions so we need [-2, 2] for all interpolation positions in our cell. Then wee also need neighbour positions so we get [-4, 4];
	constexpr size_t KERNEL_SIZE = 2 * INTERPOLATION_DEGREE_MAX + 5 + 1;//Plus one for both sides being inclusive
	constexpr size_t KERNEL_OFFSET = INTERPOLATION_DEGREE_MAX + 2;
	
	//Both positive, both rounded up. Start will later be negated
	constexpr size_t KERNEL_START_BLOCK = (KERNEL_SIZE - KERNEL_OFFSET - 1 + config::G_BLOCKSIZE - 1) / config::G_BLOCKSIZE;
	constexpr size_t KERNEL_END_BLOCK = (KERNEL_OFFSET + config::G_BLOCKSIZE - 1) / config::G_BLOCKSIZE;
	
	const size_t base_row = NUM_ROWS_PER_BLOCK * blockIdx.x;
	const int boundary_condition   = static_cast<int>(std::floor(config::G_BOUNDARY_CONDITION));

	float mass_solid_local[(3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE];
	float gradient_solid_local[(3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE];
	float scaling_solid_local[(1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE];
	float velocity_solid_local[(3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE];
	float pressure_solid_nominator_local[(1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE];
	float pressure_solid_denominator_local[(1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE];
	
	float mass_fluid_local[(3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE];
	float gradient_fluid_local[(3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE];
	float scaling_fluid_local[(1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE];
	float boundary_fluid_local[(3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE];
	float velocity_fluid_local[(3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE];
	float pressure_fluid_nominator_local[(1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE];
	float pressure_fluid_denominator_local[(1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE];
	
	float coupling_solid_local[(3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE];
	float coupling_fluid_local[(3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE];

	const int src_blockno		   = static_cast<int>(blockIdx.x);
	const auto blockid			   = partition.active_keys[blockIdx.x];
	const ivec3 block_cellid = blockid * static_cast<int>(config::G_BLOCKSIZE);
	//const int particle_bucket_size_solid = next_particle_buffer_solid.particle_bucket_sizes[src_blockno];
	//const int particle_bucket_size_fluid = next_particle_buffer_fluid.particle_bucket_sizes[src_blockno];
	
	//Check if the block is outside of grid bounds
	const int is_in_bound = ((blockid[0] < boundary_condition || blockid[0] >= config::G_GRID_SIZE - boundary_condition) << 2) | ((blockid[1] < boundary_condition || blockid[1] >= config::G_GRID_SIZE - boundary_condition) << 1) | (blockid[2] < boundary_condition || blockid[2] >= config::G_GRID_SIZE - boundary_condition);

	
	//If we have no particles in the bucket return
	//TODO: If both are zero then all mass will be zero and our equations are all 0; But maybe we have equations without mass?
	//if(particle_bucket_size_solid == 0 && particle_bucket_size_fluid == 0) {
	//	return;
	//}
	
	//Init memory/Load velocity
	const auto grid_block_solid	  = grid_solid.ch(_0, src_blockno);
	const auto grid_block_fluid	  = grid_fluid.ch(_0, src_blockno);
	for(size_t i = 0; i < (1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i){
		scaling_solid_local[i] = 0.0f;
		pressure_solid_nominator_local[i] = 0.0f;
		pressure_solid_denominator_local[i] = 0.0f;
		
		scaling_fluid_local[i] = 0.0f;
		pressure_fluid_nominator_local[i] = 0.0f;
		pressure_fluid_denominator_local[i] = 0.0f;
	}
	for(size_t i = 0; i < (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i){
		mass_solid_local[i] = 0.0f;
		mass_fluid_local[i] = 0.0f;
		if(get_global_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, i) < 3 * config::G_BLOCKVOLUME){
			if((get_global_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, i) % 3) == 0) {
				velocity_solid_local[i] = grid_block_solid.val_1d(_1, get_global_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, i) / 3);
				velocity_fluid_local[i] = 0.0f;//grid_block_fluid.val_1d(_1, get_global_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, i) / 3);
			} else if((get_global_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, i) % 3) == 1) {
				velocity_solid_local[i] = grid_block_solid.val_1d(_2, get_global_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, i) / 3);
				velocity_fluid_local[i] = 0.0f;//grid_block_fluid.val_1d(_2, get_global_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, i) / 3);
			} else {
				velocity_solid_local[i] = grid_block_solid.val_1d(_3, get_global_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, i) / 3);
				velocity_fluid_local[i] = 0.0f;//grid_block_fluid.val_1d(_3, get_global_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, i) / 3);
			}
		}
	}		
	for(size_t i = 0; i < (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i){				
		gradient_solid_local[i] = 0.0f;
		gradient_fluid_local[i] = 0.0f;
		
		boundary_fluid_local[i] = 0.0f;
		coupling_solid_local[i] = 0.0f;
		coupling_fluid_local[i] = 0.0f;
	}
	
	//Aggregate data		
	{
		
		//TODO: Maybe only load neighbour cells not neighbour blocks
		for(int grid_x = -static_cast<int>(KERNEL_START_BLOCK); grid_x <= static_cast<int>(KERNEL_END_BLOCK); ++grid_x){
			for(int grid_y = -static_cast<int>(KERNEL_START_BLOCK); grid_y <= static_cast<int>(KERNEL_END_BLOCK); ++grid_y){
				for(int grid_z = -static_cast<int>(KERNEL_START_BLOCK); grid_z <= static_cast<int>(KERNEL_END_BLOCK); ++grid_z){
					const ivec3 block_offset {grid_x, grid_y, grid_z};
					const ivec3 current_blockid = blockid + block_offset;
					const int current_blockno = prev_partition.query(current_blockid);
					
					//Skip empty blocks
					if(current_blockno == -1){
						continue;
					}
					
					for(int particle_offset = 0; particle_offset < next_particle_buffer_fluid.particle_bucket_sizes[current_blockno]; particle_offset += static_cast<int>(MAX_SHARED_PARTICLE_FLUID)){
						__shared__ std::array<float, 3> position_shared[MAX_SHARED_PARTICLE_FLUID];
						__shared__ float mass_shared[MAX_SHARED_PARTICLE_FLUID];
						__shared__ float J_shared[MAX_SHARED_PARTICLE_FLUID];
						
						__shared__ std::array<float, 3> velocity_shared[MAX_SHARED_PARTICLE_FLUID];
						__shared__ std::array<float, 9> C_shared[MAX_SHARED_PARTICLE_FLUID];
						
						__shared__ int count_neighbours_shared[MAX_SHARED_PARTICLE_FLUID];
						
						for(int particle_id_in_block = particle_offset + static_cast<int>(threadIdx.x); particle_id_in_block < next_particle_buffer_fluid.particle_bucket_sizes[current_blockno] && (particle_id_in_block - particle_offset) < MAX_SHARED_PARTICLE_FLUID; particle_id_in_block += static_cast<int>(blockDim.x)) {
							//Fetch index of the advection source
							int advection_source_blockno;
							int source_pidib;
							{
								//Fetch advection (direction at high bits, particle in in cell at low bits)
								const int advect = next_particle_buffer_fluid.blockbuckets[current_blockno * config::G_PARTICLE_NUM_PER_BLOCK + particle_id_in_block];

								//Retrieve the direction (first stripping the particle id by division)
								ivec3 offset;
								dir_components<3>(advect / config::G_PARTICLE_NUM_PER_BLOCK, offset.data_arr());

								//Retrieve the particle id by AND for lower bits
								source_pidib = advect & (config::G_PARTICLE_NUM_PER_BLOCK - 1);

								//Get global index by adding blockid and offset
								const ivec3 global_advection_index = current_blockid + offset;

								//Get block_no from partition
								advection_source_blockno = prev_partition.query(global_advection_index);
							}
							
							auto fluid_particle_bin = iq_fluid_particle_buffer.ch(_0, particle_buffer_fluid.bin_offsets[advection_source_blockno] + source_pidib / config::G_BIN_CAPACITY);
							const int particle_id_in_bin = source_pidib  % config::G_BIN_CAPACITY;
							
							velocity_shared[particle_id_in_block - particle_offset][0] = fluid_particle_bin.val(_0, particle_id_in_bin);
							velocity_shared[particle_id_in_block - particle_offset][1] = fluid_particle_bin.val(_1, particle_id_in_bin);
							velocity_shared[particle_id_in_block - particle_offset][2] = fluid_particle_bin.val(_2, particle_id_in_bin);
							
							C_shared[particle_id_in_block - particle_offset][0] = fluid_particle_bin.val(_3, particle_id_in_bin);
							C_shared[particle_id_in_block - particle_offset][1] = fluid_particle_bin.val(_4, particle_id_in_bin);
							C_shared[particle_id_in_block - particle_offset][2] = fluid_particle_bin.val(_5, particle_id_in_bin);
							C_shared[particle_id_in_block - particle_offset][3] = fluid_particle_bin.val(_6, particle_id_in_bin);
							C_shared[particle_id_in_block - particle_offset][4] = fluid_particle_bin.val(_7, particle_id_in_bin);
							C_shared[particle_id_in_block - particle_offset][5] = fluid_particle_bin.val(_8, particle_id_in_bin);
							C_shared[particle_id_in_block - particle_offset][6] = fluid_particle_bin.val(_9, particle_id_in_bin);
							C_shared[particle_id_in_block - particle_offset][7] = fluid_particle_bin.val(_10, particle_id_in_bin);
							C_shared[particle_id_in_block - particle_offset][8] = fluid_particle_bin.val(_11, particle_id_in_bin);

							//Fetch position and determinant of deformation gradient
							FetchParticleBufferDataIntermediate fetch_particle_buffer_tmp = {};
							fetch_particle_buffer_data<MaterialTypeFluid>(particle_buffer_fluid, advection_source_blockno, source_pidib, fetch_particle_buffer_tmp);
							position_shared[particle_id_in_block - particle_offset] = {fetch_particle_buffer_tmp.pos[0], fetch_particle_buffer_tmp.pos[1], fetch_particle_buffer_tmp.pos[2]};
							mass_shared[particle_id_in_block - particle_offset] = fetch_particle_buffer_tmp.mass;
							J_shared[particle_id_in_block - particle_offset] = fetch_particle_buffer_tmp.J;
							
							//Init count_neighbours_shared
							count_neighbours_shared[particle_id_in_block - particle_offset] = 1;
						}
						
						__syncthreads();
						
						for(int particle_id_in_block = particle_offset; particle_id_in_block < next_particle_buffer_fluid.particle_bucket_sizes[current_blockno] && (particle_id_in_block - particle_offset) < MAX_SHARED_PARTICLE_FLUID; ++particle_id_in_block) {
							aggregate_data_fluid(
								  particle_buffer_solid
								, particle_buffer_fluid
								, next_particle_buffer_solid
								, next_particle_buffer_fluid
								, prev_partition
								, grid_solid
								, grid_fluid
								, &(position_shared[0])
								, &(mass_shared[0])
								, &(J_shared[0])
								, &(velocity_shared[0])
								, &(C_shared[0])
								, &(count_neighbours_shared[0])
								, particle_offset
								, current_blockno
								, current_blockid
								, block_cellid
								, particle_id_in_block
								, &(scaling_fluid_local[0])
								, &(pressure_fluid_nominator_local[0])
								, &(pressure_fluid_denominator_local[0])
								, &(mass_fluid_local[0])
								, &(gradient_fluid_local[0])
								, &(boundary_fluid_local[0])
								, &(velocity_fluid_local[0])
							);
						}
						
						__syncthreads();
						
						//Store count neighbours
						for(int particle_id_in_block = particle_offset + static_cast<int>(threadIdx.x); particle_id_in_block < next_particle_buffer_fluid.particle_bucket_sizes[current_blockno] && (particle_id_in_block - particle_offset) < MAX_SHARED_PARTICLE_FLUID; particle_id_in_block += static_cast<int>(blockDim.x)) {
							//Fetch index of the advection source
							int advection_source_blockno;
							int source_pidib;
							{
								//Fetch advection (direction at high bits, particle in in cell at low bits)
								const int advect = next_particle_buffer_fluid.blockbuckets[current_blockno * config::G_PARTICLE_NUM_PER_BLOCK + particle_id_in_block];

								//Retrieve the direction (first stripping the particle id by division)
								ivec3 offset;
								dir_components<3>(advect / config::G_PARTICLE_NUM_PER_BLOCK, offset.data_arr());

								//Retrieve the particle id by AND for lower bits
								source_pidib = advect & (config::G_PARTICLE_NUM_PER_BLOCK - 1);

								//Get global index by adding blockid and offset
								const ivec3 global_advection_index = current_blockid + offset;

								//Get block_no from partition
								advection_source_blockno = prev_partition.query(global_advection_index);
							}
							
							auto fluid_particle_bin = iq_fluid_particle_buffer.ch(_0, particle_buffer_fluid.bin_offsets[advection_source_blockno] + source_pidib / config::G_BIN_CAPACITY);
							const int particle_id_in_bin = source_pidib  % config::G_BIN_CAPACITY;
							
							fluid_particle_bin.val(_12, particle_id_in_bin) = count_neighbours_shared[particle_id_in_block - particle_offset];
						}
						
						__syncthreads();
					}
					
					for(int particle_offset = 0; particle_offset < next_particle_buffer_solid.particle_bucket_sizes[current_blockno]; particle_offset += static_cast<int>(MAX_SHARED_PARTICLE_SOLID)){
						__shared__ std::array<float, 3> position_shared[MAX_SHARED_PARTICLE_SOLID];
						__shared__ float mass_shared[MAX_SHARED_PARTICLE_SOLID];
						__shared__ float J_shared[MAX_SHARED_PARTICLE_SOLID];
						
						__shared__ std::array<float, 3> normal_shared[MAX_SHARED_PARTICLE_SOLID];
						__shared__ SurfacePointType point_type_shared[MAX_SHARED_PARTICLE_SOLID];
						__shared__ float contact_area_shared[MAX_SHARED_PARTICLE_SOLID];
						
						for(int particle_id_in_block = particle_offset + static_cast<int>(threadIdx.x); particle_id_in_block < next_particle_buffer_solid.particle_bucket_sizes[current_blockno] && (particle_id_in_block - particle_offset) < MAX_SHARED_PARTICLE_SOLID; particle_id_in_block += static_cast<int>(blockDim.x)) {
							//Fetch index of the advection source
							int advection_source_blockno;
							int source_pidib;
							{
								//Fetch advection (direction at high bits, particle in in cell at low bits)
								const int advect = next_particle_buffer_solid.blockbuckets[current_blockno * config::G_PARTICLE_NUM_PER_BLOCK + particle_id_in_block];

								//Retrieve the direction (first stripping the particle id by division)
								ivec3 offset;
								dir_components<3>(advect / config::G_PARTICLE_NUM_PER_BLOCK, offset.data_arr());

								//Retrieve the particle id by AND for lower bits
								source_pidib = advect & (config::G_PARTICLE_NUM_PER_BLOCK - 1);

								//Get global index by adding blockid and offset
								const ivec3 global_advection_index = current_blockid + offset;

								//Get block_no from partition
								advection_source_blockno = prev_partition.query(global_advection_index);
							}
							
							auto surface_particle_bin = surface_particle_buffer_solid.ch(_0, particle_buffer_solid.bin_offsets[advection_source_blockno] + source_pidib / config::G_BIN_CAPACITY);
							const int surface_particle_id_in_bin = source_pidib  % config::G_BIN_CAPACITY;
							
							normal_shared[particle_id_in_block - particle_offset][0] = surface_particle_bin.val(_1, surface_particle_id_in_bin);
							normal_shared[particle_id_in_block - particle_offset][1] = surface_particle_bin.val(_2, surface_particle_id_in_bin);
							normal_shared[particle_id_in_block - particle_offset][2] = surface_particle_bin.val(_3, surface_particle_id_in_bin);
							point_type_shared[particle_id_in_block - particle_offset] = *reinterpret_cast<SurfacePointType*>(&surface_particle_bin.val(_0, surface_particle_id_in_bin));
							contact_area_shared[particle_id_in_block - particle_offset] = surface_particle_bin.val(_6, surface_particle_id_in_bin);

							//Fetch position and determinant of deformation gradient
							FetchParticleBufferDataIntermediate fetch_particle_buffer_tmp = {};
							fetch_particle_buffer_data<MaterialTypeSolid>(particle_buffer_solid, advection_source_blockno, source_pidib, fetch_particle_buffer_tmp);
							position_shared[particle_id_in_block - particle_offset] = {fetch_particle_buffer_tmp.pos[0], fetch_particle_buffer_tmp.pos[1], fetch_particle_buffer_tmp.pos[2]};
							mass_shared[particle_id_in_block - particle_offset] = fetch_particle_buffer_tmp.mass;
							J_shared[particle_id_in_block - particle_offset] = fetch_particle_buffer_tmp.J;
						}
						
						__syncthreads();
						
						for(int particle_id_in_block = particle_offset; particle_id_in_block < next_particle_buffer_solid.particle_bucket_sizes[current_blockno] && (particle_id_in_block - particle_offset) < MAX_SHARED_PARTICLE_SOLID; ++particle_id_in_block) {
							aggregate_data_solid(
								  particle_buffer_solid
								, particle_buffer_fluid
								, next_particle_buffer_solid
								, next_particle_buffer_fluid
								, prev_partition
								, grid_solid
								, grid_fluid
								, iq_fluid_particle_buffer
								, &(position_shared[0])
								, &(mass_shared[0])
								, &(J_shared[0])
								, &(normal_shared[0])
								, &(point_type_shared[0])
								, &(contact_area_shared[0])
								, particle_offset
								, current_blockno
								, current_blockid
								, block_cellid
								, particle_id_in_block
								, &(scaling_solid_local[0])
								, &(pressure_solid_nominator_local[0])
								, &(pressure_solid_denominator_local[0])
								, &(mass_solid_local[0])
								, &(gradient_solid_local[0])
								, &(coupling_solid_local[0])
								, &(scaling_fluid_local[0])
								, &(pressure_fluid_nominator_local[0])
								, &(pressure_fluid_denominator_local[0])
								, &(mass_fluid_local[0])
								, &(gradient_fluid_local[0])
								, &(boundary_fluid_local[0])
								, &(velocity_fluid_local[0])
								, &(coupling_fluid_local[0])
								
							);
						}
						
						__syncthreads();
					}
				}
			}
		}
	}	
	
	//Update fluid grid
	for(size_t local_cell_index = 0; local_cell_index < get_thread_count<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, 3 * config::G_BLOCKVOLUME); local_cell_index++){
		const size_t row = get_global_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_cell_index) / 3;
		const size_t alpha = get_global_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_cell_index) % 3;
		const ivec3 local_id {static_cast<int>((row / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE), static_cast<int>((row / config::G_BLOCKSIZE) % config::G_BLOCKSIZE), static_cast<int>(row % config::G_BLOCKSIZE)};

		const float mass = mass_fluid_local[local_cell_index];
		if(mass > 0.0f) {
			const float mass_inv = 1.f / mass;

			//int i = (cell_id_in_block >> (config::G_BLOCKBITS << 1)) & config::G_BLOCKMASK;
			//int j = (cell_id_in_block >> config::G_BLOCKBITS) & config::G_BLOCKMASK;
			//int k = cell_id_in_block & config::G_BLOCKMASK;

			//Fetch current velocity
			float vel = velocity_fluid_local[local_cell_index];

			//Update velocity. Set to 0 if outside of bounds
			if(alpha == 0){
				vel = is_in_bound & 4 ? 0.0f : vel * mass_inv;
			}else if(alpha == 1){
				vel = is_in_bound & 2 ? 0.0f : vel * mass_inv;
				vel += config::G_GRAVITY * dt.count();
			}else{//alpha == 2
				vel = is_in_bound & 1 ? 0.0f : vel * mass_inv;
			}
			// if (is_in_bound) ///< sticky
			//  vel = 0.0f;

			//Write back velocity
			//if(alpha == 0){
			//	grid_block_fluid.val_1d(_1, row) = vel;
			//}else if(alpha == 1){
			//	grid_block_fluid.val_1d(_2, row) = vel;
			//}else{//alpha == 2
			//	grid_block_fluid.val_1d(_3, row) = vel;
			//}
			
			//printf("TMP2 %d %d # %d # %.28f # %.28f # %.28f\n", static_cast<int>(blockIdx.x), static_cast<int>(row), static_cast<int>(alpha), mass, vel, velocity_fluid_local[local_cell_index]);
			
			
			velocity_fluid_local[local_cell_index] = vel;
		}
		
		//Store mass to grid
		//if(alpha == 0){
		//	grid_block_fluid.val_1d(_0, row) = mass;
		//}
		
		//FIXME: Recalculate max_vel. This should actually happen before we create IQ-System
	}
	
	//Column that represents (row, row)
	//constexpr size_t IDENTIITY_NEIGHBOUR_INDEX = (INTERPOLATION_DEGREE_MAX * ((2 * INTERPOLATION_DEGREE_MAX + 1) * (2 * INTERPOLATION_DEGREE_MAX + 1)) + INTERPOLATION_DEGREE_MAX * (2 * INTERPOLATION_DEGREE_MAX + 1) + INTERPOLATION_DEGREE_MAX);
	
	//Store data in matrix
	//NOTE: Coupling was stored in transposed form
	
	/*
		RHS = {
			p^s
			p^f
			-
			-
		}
		SOLVE_VELOCITY_RESULT = {
			v^s
			v^f
			-
			-
		}
	*/
	for(size_t local_cell_index = 0; local_cell_index < get_thread_count<BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, config::G_BLOCKVOLUME); local_cell_index++){
		const size_t row = get_global_index<BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_cell_index);
		const ivec3 local_id {static_cast<int>((row / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE), static_cast<int>((row / config::G_BLOCKSIZE) % config::G_BLOCKSIZE), static_cast<int>(row % config::G_BLOCKSIZE)};

		float pressure_solid;
		float pressure_fluid;
		//Only calculate for particles with pressure_denominator bigger than 0 (otherwise we will divide by 0)
		if(pressure_solid_denominator_local[local_cell_index] > 0.0f){
			pressure_solid = pressure_solid_nominator_local[local_cell_index] / pressure_solid_denominator_local[local_cell_index];
		}else{
			pressure_solid = 0.0f;
		}
		if(pressure_fluid_denominator_local[local_cell_index] > 0.0f){
			pressure_fluid = pressure_fluid_nominator_local[local_cell_index] / pressure_fluid_denominator_local[local_cell_index];
		}else{
			pressure_fluid = 0.0f;
		}
		
		const int row_index_pressure_solid = base_row + row;
		const int row_index_pressure_fluid = NUM_ROWS_PER_BLOCK * num_blocks + base_row + row;
		
		atomicAdd(&(iq_pointers.iq_rhs[row_index_pressure_solid]), pressure_solid);
		atomicAdd(&(iq_pointers.iq_rhs[row_index_pressure_fluid]), pressure_fluid);
		
		//NOTE: Storing S/dt
		const int row_index_scaling = base_row + row;
		atomicAdd(&(iq_pointers.scaling_solid[row_index_scaling]), scaling_solid_local[local_cell_index] / dt.count());
		atomicAdd(&(iq_pointers.scaling_fluid[row_index_scaling]), scaling_fluid_local[local_cell_index] / dt.count());
		
		//Add slight scaling to make matrix definite
		//TODO: Only add to columns with gradient?
		//const float scaling_fluid = (std::pow(config::G_DX, 3.0f) / particle_buffer_fluid.rho) / 1e-8f;
		//atomicAdd(&(iq_pointers.scaling_fluid[row_index_scaling]), scaling_fluid / dt.count());
	}
	
	for(size_t local_cell_index = 0; local_cell_index < get_thread_count<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, 3 * config::G_BLOCKVOLUME); local_cell_index++){
		const size_t row = get_global_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_cell_index) / 3;
		const size_t alpha = get_global_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_cell_index) % 3;
		const ivec3 local_id {static_cast<int>((row / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE), static_cast<int>((row / config::G_BLOCKSIZE) % config::G_BLOCKSIZE), static_cast<int>(row % config::G_BLOCKSIZE)};
		
		float mass_solid;
		float mass_fluid;
		//Only calculate for particles with mass bigger than 0 (otherwise we will divide by 0)
		if(mass_solid_local[local_cell_index] > 0.0f){
			mass_solid = dt.count() / mass_solid_local[local_cell_index];
		}else{
			mass_solid = 0.0f;
		}
		if(mass_fluid_local[local_cell_index] > 0.0f){
			mass_fluid = dt.count() / mass_fluid_local[local_cell_index];
		}else{
			mass_fluid = 0.0f;
		}
		
		const int row_index_velocity_solid = 3 * (base_row + row) + alpha;
		const int row_index_velocity_fluid = 3 * NUM_ROWS_PER_BLOCK * num_blocks + 3 * (base_row + row) + alpha;
		
		atomicAdd(&(iq_pointers.iq_solve_velocity_result[row_index_velocity_solid]), velocity_solid_local[local_cell_index]);
		atomicAdd(&(iq_pointers.iq_solve_velocity_result[row_index_velocity_fluid]), velocity_fluid_local[local_cell_index]);
		
		//NOTE: Storing dt * M^-1
		const int row_index_mass = 3 * (base_row + row) + alpha;
		atomicAdd(&(iq_pointers.mass_solid[row_index_mass]), mass_solid);
		atomicAdd(&(iq_pointers.mass_fluid[row_index_mass]), mass_fluid);
	}
	
	for(size_t local_cell_index = 0; local_cell_index < get_thread_count<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, 3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME); local_cell_index++){
		const size_t row = get_global_index<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_cell_index) / (3 * NUM_COLUMNS_PER_BLOCK);
		const size_t column = (get_global_index<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_cell_index) / 3) % NUM_COLUMNS_PER_BLOCK;
		const size_t alpha = get_global_index<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_cell_index) % 3;
		const ivec3 local_id {static_cast<int>((row / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE), static_cast<int>((row / config::G_BLOCKSIZE) % config::G_BLOCKSIZE), static_cast<int>(row % config::G_BLOCKSIZE)};
		const ivec3 neighbour_local_id = ivec3(static_cast<int>((column / ((2 * INTERPOLATION_DEGREE_MAX + 1) * (2 * INTERPOLATION_DEGREE_MAX + 1))) % (2 * INTERPOLATION_DEGREE_MAX + 1)), static_cast<int>((column / (2 * INTERPOLATION_DEGREE_MAX + 1)) % (2 * INTERPOLATION_DEGREE_MAX + 1)), static_cast<int>(column % (2 * INTERPOLATION_DEGREE_MAX + 1))) - ivec3(static_cast<int>(INTERPOLATION_DEGREE_MAX), static_cast<int>(INTERPOLATION_DEGREE_MAX), static_cast<int>(INTERPOLATION_DEGREE_MAX));
			
		const ivec3 neighbour_cellid = block_cellid + local_id + neighbour_local_id;
		const ivec3 neighbour_blockid = neighbour_cellid / static_cast<int>(config::G_BLOCKSIZE);
		
		const ivec3 neighbour_base_cellid = neighbour_blockid * static_cast<int>(config::G_BLOCKSIZE);
		const ivec3 neighbour_celloffset = neighbour_cellid - neighbour_base_cellid;
		
		const int neighbour_blockno = partition.query(neighbour_blockid);
		const int neighbour_cellno = NUM_ROWS_PER_BLOCK * neighbour_blockno + (config::G_BLOCKSIZE * config::G_BLOCKSIZE) * neighbour_celloffset[0] + config::G_BLOCKSIZE * neighbour_celloffset[1] + neighbour_celloffset[2];
		
		const int row_index = 3 * (base_row + row) + alpha;
		
		int local_column_index = -1;
		for(size_t lhs_column = 0; lhs_column < NUM_COLUMNS_PER_BLOCK; ++lhs_column){
			if(neighbour_cellno == iq_pointers.gradient_solid_columns[iq_pointers.gradient_solid_rows[row_index] + lhs_column]){
				local_column_index = lhs_column;
				break;
			}
		}
		
		const int column_index = iq_pointers.gradient_solid_rows[row_index] + local_column_index;
		
		atomicAdd(&(iq_pointers.gradient_solid_values[column_index]), gradient_solid_local[local_cell_index]);
		atomicAdd(&(iq_pointers.gradient_fluid_values[column_index]), gradient_fluid_local[local_cell_index]);
		
		//NOTE: Storing H^T
		atomicAdd(&(iq_pointers.coupling_solid_values[column_index]), coupling_solid_local[local_cell_index]);
		atomicAdd(&(iq_pointers.coupling_fluid_values[column_index]), coupling_fluid_local[local_cell_index]);
		
		atomicAdd(&(iq_pointers.boundary_fluid_values[column_index]), boundary_fluid_local[local_cell_index]);
	}
}

template<typename Partition, typename Grid, MaterialE MaterialTypeSolid, MaterialE MaterialTypeFluid>
__global__ void update_velocity_and_strain(const ParticleBuffer<MaterialTypeSolid> particle_buffer_solid, const ParticleBuffer<MaterialTypeFluid> particle_buffer_fluid, const ParticleBuffer<MaterialTypeSolid> next_particle_buffer_solid, const ParticleBuffer<MaterialTypeFluid> next_particle_buffer_fluid, const Partition prev_partition, Partition partition, Grid grid_solid, Grid grid_fluid, const float* delta_v_solid, const float* delta_v_fluid, const float* pressure_solid, const float* pressure_fluid) {
	const int src_blockno		   = static_cast<int>(blockIdx.x);
	const auto blockid			   = partition.active_keys[blockIdx.x];
	const ivec3 block_cellid = blockid * static_cast<int>(config::G_BLOCKSIZE);
	
	//Check if the block is outside of grid bounds
	const int boundary_condition   = static_cast<int>(std::floor(config::G_BOUNDARY_CONDITION));
	const int is_in_bound = ((blockid[0] < boundary_condition || blockid[0] >= config::G_GRID_SIZE - boundary_condition) << 2) | ((blockid[1] < boundary_condition || blockid[1] >= config::G_GRID_SIZE - boundary_condition) << 1) | (blockid[2] < boundary_condition || blockid[2] >= config::G_GRID_SIZE - boundary_condition);

	
	//Update velocity
	auto grid_block_solid = grid_solid.ch(_0, src_blockno);
	auto grid_block_fluid = grid_fluid.ch(_0, src_blockno);
	for(int cell_id_in_block = threadIdx.x; cell_id_in_block < config::G_BLOCKVOLUME; cell_id_in_block += blockDim.x) {
		grid_block_solid.val_1d(_1, cell_id_in_block) += is_in_bound & 4 ? 0.0f : delta_v_solid[3 * config::G_BLOCKVOLUME * src_blockno + 3 * cell_id_in_block];
		grid_block_solid.val_1d(_2, cell_id_in_block) += is_in_bound & 2 ? 0.0f : delta_v_solid[3 * config::G_BLOCKVOLUME * src_blockno + 3 * cell_id_in_block + 1];
		grid_block_solid.val_1d(_3, cell_id_in_block) += is_in_bound & 1 ? 0.0f : delta_v_solid[3 * config::G_BLOCKVOLUME * src_blockno + 3 * cell_id_in_block + 2];
		
		//printf("ABC2 %.28f %.28f %.28f\n", grid_block_solid.val_1d(_1, cell_id_in_block), grid_block_solid.val_1d(_2, cell_id_in_block), grid_block_solid.val_1d(_3, cell_id_in_block));
		
		grid_block_fluid.val_1d(_1, cell_id_in_block) += is_in_bound & 4 ? 0.0f : delta_v_fluid[3 * config::G_BLOCKVOLUME * src_blockno + 3 * cell_id_in_block];
		grid_block_fluid.val_1d(_2, cell_id_in_block) += is_in_bound & 2 ? 0.0f : delta_v_fluid[3 * config::G_BLOCKVOLUME * src_blockno + 3 * cell_id_in_block + 1];
		grid_block_fluid.val_1d(_3, cell_id_in_block) += is_in_bound & 1 ? 0.0f : delta_v_fluid[3 * config::G_BLOCKVOLUME * src_blockno + 3 * cell_id_in_block + 2];
		
		/*
		if(
			(grid_block_solid.val_1d(_1, cell_id_in_block) > 30.0f)
			|| (grid_block_solid.val_1d(_2, cell_id_in_block) > 30.0f)
			|| (grid_block_solid.val_1d(_3, cell_id_in_block) > 30.0f)
			|| (grid_block_fluid.val_1d(_1, cell_id_in_block) > 30.0f)
			|| (grid_block_fluid.val_1d(_2, cell_id_in_block) > 30.0f)
			|| (grid_block_fluid.val_1d(_3, cell_id_in_block) > 30.0f)
		){
			printf("TMP0 %d # %.28f %.28f %.28f # %.28f %.28f %.28f # %.28f %.28f %.28f # %.28f %.28f %.28f\n"
				, (src_blockno * config::G_BLOCKVOLUME + cell_id_in_block)
				, grid_block_solid.val_1d(_1, cell_id_in_block)
				, grid_block_solid.val_1d(_2, cell_id_in_block)
				, grid_block_solid.val_1d(_3, cell_id_in_block)
				, grid_block_fluid.val_1d(_1, cell_id_in_block)
				, grid_block_fluid.val_1d(_2, cell_id_in_block)
				, grid_block_fluid.val_1d(_3, cell_id_in_block)
				, delta_v_solid[3 * config::G_BLOCKVOLUME * src_blockno + 3 * cell_id_in_block]
				, delta_v_solid[3 * config::G_BLOCKVOLUME * src_blockno + 3 * cell_id_in_block + 1]
				, delta_v_solid[3 * config::G_BLOCKVOLUME * src_blockno + 3 * cell_id_in_block + 2]
				, delta_v_fluid[3 * config::G_BLOCKVOLUME * src_blockno + 3 * cell_id_in_block]
				, delta_v_fluid[3 * config::G_BLOCKVOLUME * src_blockno + 3 * cell_id_in_block + 1]
				, delta_v_fluid[3 * config::G_BLOCKVOLUME * src_blockno + 3 * cell_id_in_block + 2]
			);
		}*/
	}
	
	//Particles in cell can have offset of [0, 5] ([0, 3] current block, 2 for offset caused by kernel 2 in storing); Then additional 2 are added in both directions for max kernel degree => [-2, 7] or absolute [0, 9] with offset 2
	constexpr size_t KERNEL_SIZE = 2 * INTERPOLATION_DEGREE_MAX + 5 + 1;//Plus one for both sides being inclusive
	constexpr size_t KERNEL_OFFSET = INTERPOLATION_DEGREE_MAX;
	
	constexpr size_t CELL_COUNT = KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE;

#if (FIXED_COROTATED_GHOST_ENABLE_STRAIN_UPDATE == 0)
	const int particle_bucket_size_solid = next_particle_buffer_solid.particle_bucket_sizes[src_blockno];
	
	__shared__ float pressure_solid_shared[KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE];
	
	//If we have no particles in the bucket return
	if(particle_bucket_size_solid == 0) {
		return;
	}
	
	//Load data from grid to shared memory
	for(int base = static_cast<int>(threadIdx.x); base < CELL_COUNT; base += static_cast<int>(blockDim.x)) {
		const ivec3 absolute_local_cellid = ivec3(static_cast<int>((base / (KERNEL_SIZE * KERNEL_SIZE)) % KERNEL_SIZE), static_cast<int>((base / KERNEL_SIZE) % KERNEL_SIZE), static_cast<int>(base % KERNEL_SIZE));
		const ivec3 local_cellid = absolute_local_cellid - ivec3(static_cast<int>(KERNEL_OFFSET), static_cast<int>(KERNEL_OFFSET), static_cast<int>(KERNEL_OFFSET));
		const ivec3 current_blockid = (block_cellid + local_cellid) / config::G_BLOCKSIZE;
		const auto blockno = partition.query(current_blockid);
	
		const ivec3 cellid_in_block = (block_cellid + local_cellid) - current_blockid * config::G_BLOCKSIZE;
		const int cellno_in_block = (config::G_BLOCKSIZE * config::G_BLOCKSIZE) * cellid_in_block[0] + config::G_BLOCKSIZE * cellid_in_block[1] + cellid_in_block[2];

		const float val = pressure_solid[config::G_BLOCKVOLUME * blockno + cellno_in_block];

		pressure_solid_shared[absolute_local_cellid[0]][absolute_local_cellid[1]][absolute_local_cellid[2]] = val;
	}
	__syncthreads();

	//Update strain
	for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < particle_bucket_size_solid; particle_id_in_block += static_cast<int>(blockDim.x)) {
		//Fetch index of the advection source
		int advection_source_blockno;
		int source_pidib;
		{
			//Fetch advection (direction at high bits, particle in in cell at low bits)
			const int advect = next_particle_buffer_solid.blockbuckets[src_blockno * config::G_PARTICLE_NUM_PER_BLOCK + particle_id_in_block];

			//Retrieve the direction (first stripping the particle id by division)
			ivec3 offset;
			dir_components<3>(advect / config::G_PARTICLE_NUM_PER_BLOCK, offset.data_arr());

			//Retrieve the particle id by AND for lower bits
			source_pidib = advect & (config::G_PARTICLE_NUM_PER_BLOCK - 1);

			//Get global index by adding blockid and offset
			const ivec3 global_advection_index = blockid + offset;

			//Get block_no from partition
			advection_source_blockno = prev_partition.query(global_advection_index);
		}

		//Fetch position and determinant of deformation gradient
		FetchParticleBufferDataIntermediate fetch_particle_buffer_tmp = {};
		fetch_particle_buffer_data<MaterialTypeSolid>(particle_buffer_solid, advection_source_blockno, source_pidib, fetch_particle_buffer_tmp);
		vec3 pos {fetch_particle_buffer_tmp.pos[0], fetch_particle_buffer_tmp.pos[1], fetch_particle_buffer_tmp.pos[2]};
		
		//Get position of grid cell
		const ivec3 global_base_index_solid_pressure = get_cell_id<INTERPOLATION_DEGREE_SOLID_PRESSURE>(pos.data_arr(), grid_solid.get_offset());
		
		//Get position relative to grid cell
		const vec3 local_pos_solid_pressure = pos - (global_base_index_solid_pressure + vec3(grid_solid.get_offset()[0], grid_solid.get_offset()[1], grid_solid.get_offset()[2])) * config::G_DX;

		//Calculate weights
		vec3x3 weight_solid_pressure;
		
		#pragma unroll 3
		for(int dd = 0; dd < 3; ++dd) {
			const std::array<float, INTERPOLATION_DEGREE_SOLID_PRESSURE + 1> current_weight_solid_pressure = bspline_weight<float, INTERPOLATION_DEGREE_SOLID_PRESSURE>(local_pos_solid_pressure[dd]);
			for(int i = 0; i < INTERPOLATION_DEGREE_SOLID_PRESSURE + 1; ++i){
				weight_solid_pressure(dd, i)		  = current_weight_solid_pressure[i];
			}
			for(int i = INTERPOLATION_DEGREE_SOLID_PRESSURE + 1; i < 3; ++i){
				weight_solid_pressure(dd, i)		  = 0.0f;
			}
		}
		
		float weighted_pressure = 0.0f;
		//Load data
		//Note: Weights are 0 if outside of interpolation degree/radius around particles cell
		//Foreach node in the block we add values accoring to particle kernel, also handling all neighbours of the particles cell
		for(char i = -static_cast<char>(INTERPOLATION_DEGREE_MAX); i < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; i++) {
			for(char j = -static_cast<char>(INTERPOLATION_DEGREE_MAX); j < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; j++) {
				for(char k = -static_cast<char>(INTERPOLATION_DEGREE_MAX); k < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; k++) {
					const ivec3 local_id = (global_base_index_solid_pressure - block_cellid) + ivec3(i, j, k);
					const ivec3 absolute_local_id = local_id + ivec3(static_cast<int>(KERNEL_OFFSET), static_cast<int>(KERNEL_OFFSET), static_cast<int>(KERNEL_OFFSET));
					
					if(
						   (absolute_local_id[0] < 0 || absolute_local_id[0] >= KERNEL_SIZE)
						|| (absolute_local_id[1] < 0 || absolute_local_id[1] >= KERNEL_SIZE)
						|| (absolute_local_id[2] < 0 || absolute_local_id[2] >= KERNEL_SIZE)
					){
						//printf("ERROR4 %d %d %d # %d %d %d # %.28f %.28f %.28f\n", local_id[0], local_id[1], local_id[2], absolute_local_id[0], absolute_local_id[1], absolute_local_id[2], pos[0], pos[1], pos[2]);
					}
					
					//Weight
					const float W_pressure = weight_solid_pressure(0, std::abs(i)) * weight_solid_pressure(1, std::abs(j)) * weight_solid_pressure(2, std::abs(k));
					
					weighted_pressure += pressure_solid_shared[absolute_local_id[0]][absolute_local_id[1]][absolute_local_id[2]] * W_pressure;
				}
			}
		}
		
		update_strain_solid<MaterialTypeSolid>(particle_buffer_solid, advection_source_blockno, source_pidib, weighted_pressure);
	}
	
#endif

#if (J_FLUID_ENABLE_STRAIN_UPDATE == 0)
	const int particle_bucket_size_fluid = next_particle_buffer_fluid.particle_bucket_sizes[src_blockno];
	
	__shared__ float pressure_fluid_shared[KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE];
	
	//If we have no particles in the bucket return
	if(particle_bucket_size_fluid == 0) {
		return;
	}
	
	//Load data from grid to shared memory
	for(int base = static_cast<int>(threadIdx.x); base < CELL_COUNT; base += static_cast<int>(blockDim.x)) {
		const ivec3 absolute_local_cellid = ivec3(static_cast<int>((base / (KERNEL_SIZE * KERNEL_SIZE)) % KERNEL_SIZE), static_cast<int>((base / KERNEL_SIZE) % KERNEL_SIZE), static_cast<int>(base % KERNEL_SIZE));
		const ivec3 local_cellid = absolute_local_cellid - ivec3(static_cast<int>(KERNEL_OFFSET), static_cast<int>(KERNEL_OFFSET), static_cast<int>(KERNEL_OFFSET));
		const ivec3 current_blockid = (block_cellid + local_cellid) / config::G_BLOCKSIZE;
		const auto blockno = partition.query(current_blockid);
	
		const ivec3 cellid_in_block = (block_cellid + local_cellid) - current_blockid * config::G_BLOCKSIZE;
		const int cellno_in_block = (config::G_BLOCKSIZE * config::G_BLOCKSIZE) * cellid_in_block[0] + config::G_BLOCKSIZE * cellid_in_block[1] + cellid_in_block[2];

		const float val = pressure_fluid[config::G_BLOCKVOLUME * blockno + cellno_in_block];

		pressure_fluid_shared[absolute_local_cellid[0]][absolute_local_cellid[1]][absolute_local_cellid[2]] = val;
	}
	__syncthreads();

	//Update strain
	for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < particle_bucket_size_fluid; particle_id_in_block += static_cast<int>(blockDim.x)) {
		//Fetch index of the advection source
		int advection_source_blockno;
		int source_pidib;
		{
			//Fetch advection (direction at high bits, particle in in cell at low bits)
			const int advect = next_particle_buffer_fluid.blockbuckets[src_blockno * config::G_PARTICLE_NUM_PER_BLOCK + particle_id_in_block];

			//Retrieve the direction (first stripping the particle id by division)
			ivec3 offset;
			dir_components<3>(advect / config::G_PARTICLE_NUM_PER_BLOCK, offset.data_arr());

			//Retrieve the particle id by AND for lower bits
			source_pidib = advect & (config::G_PARTICLE_NUM_PER_BLOCK - 1);

			//Get global index by adding blockid and offset
			const ivec3 global_advection_index = blockid + offset;

			//Get block_no from partition
			advection_source_blockno = prev_partition.query(global_advection_index);
		}

		//Fetch position and determinant of deformation gradient
		FetchParticleBufferDataIntermediate fetch_particle_buffer_tmp = {};
		fetch_particle_buffer_data<MaterialTypeFluid>(particle_buffer_fluid, advection_source_blockno, source_pidib, fetch_particle_buffer_tmp);
		vec3 pos {fetch_particle_buffer_tmp.pos[0], fetch_particle_buffer_tmp.pos[1], fetch_particle_buffer_tmp.pos[2]};
		
		//Get position of grid cell
		const ivec3 global_base_index_fluid_pressure = get_cell_id<INTERPOLATION_DEGREE_FLUID_PRESSURE>(pos.data_arr(), grid_fluid.get_offset());
		
		//Get position relative to grid cell
		const vec3 local_pos_fluid_pressure = pos - (global_base_index_fluid_pressure + vec3(grid_fluid.get_offset()[0], grid_fluid.get_offset()[1], grid_fluid.get_offset()[2])) * config::G_DX;

		//Calculate weights
		vec3x3 weight_fluid_pressure;
		
		#pragma unroll 3
		for(int dd = 0; dd < 3; ++dd) {
			const std::array<float, INTERPOLATION_DEGREE_FLUID_PRESSURE + 1> current_weight_fluid_pressure = bspline_weight<float, INTERPOLATION_DEGREE_FLUID_PRESSURE>(local_pos_fluid_pressure[dd]);
			for(int i = 0; i < INTERPOLATION_DEGREE_FLUID_PRESSURE + 1; ++i){
				weight_fluid_pressure(dd, i)		  = current_weight_fluid_pressure[i];
			}
			for(int i = INTERPOLATION_DEGREE_FLUID_PRESSURE + 1; i < 3; ++i){
				weight_fluid_pressure(dd, i)		  = 0.0f;
			}
		}
		
		float weighted_pressure = 0.0f;
		//Load data
		//Note: Weights are 0 if outside of interpolation degree/radius around particles cell
		//Foreach node in the block we add values accoring to particle kernel, also handling all neighbours of the particles cell
		for(char i = -static_cast<char>(INTERPOLATION_DEGREE_MAX); i < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; i++) {
			for(char j = -static_cast<char>(INTERPOLATION_DEGREE_MAX); j < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; j++) {
				for(char k = -static_cast<char>(INTERPOLATION_DEGREE_MAX); k < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; k++) {
					const ivec3 local_id = (global_base_index_fluid_pressure - block_cellid) + ivec3(i, j, k);
					const ivec3 absolute_local_id = local_id + ivec3(static_cast<int>(KERNEL_OFFSET), static_cast<int>(KERNEL_OFFSET), static_cast<int>(KERNEL_OFFSET));
					
					if(
						   (absolute_local_id[0] < 0 || absolute_local_id[0] >= KERNEL_SIZE)
						|| (absolute_local_id[1] < 0 || absolute_local_id[1] >= KERNEL_SIZE)
						|| (absolute_local_id[2] < 0 || absolute_local_id[2] >= KERNEL_SIZE)
					){
						//printf("ERROR4 %d %d %d # %d %d %d # %.28f %.28f %.28f\n", local_id[0], local_id[1], local_id[2], absolute_local_id[0], absolute_local_id[1], absolute_local_id[2], pos[0], pos[1], pos[2]);
					}
					
					//Weight
					const float W_pressure = weight_fluid_pressure(0, std::abs(i)) * weight_fluid_pressure(1, std::abs(j)) * weight_fluid_pressure(2, std::abs(k));
					
					weighted_pressure += pressure_fluid_shared[absolute_local_id[0]][absolute_local_id[1]][absolute_local_id[2]] * W_pressure;
				}
			}
		}
		
		update_strain_fluid<MaterialTypeFluid>(particle_buffer_fluid, advection_source_blockno, source_pidib, weighted_pressure);
	}
#endif
}

}// namespace iq

//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers, readability-identifier-naming, misc-definitions-in-headers)
}// namespace mn

#endif