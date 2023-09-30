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
	
constexpr size_t BLOCK_SIZE = config::G_PARTICLE_BATCH_CAPACITY;


//TODO: Make adjustable by extern paraameter (like changing if we use constant or linear kernels)
//Kernel degrees
constexpr size_t INTERPOLATION_DEGREE_SOLID_VELOCITY = 2;
constexpr size_t INTERPOLATION_DEGREE_SOLID_PRESSURE = 0;
constexpr size_t INTERPOLATION_DEGREE_FLUID_VELOCITY = 1;
constexpr size_t INTERPOLATION_DEGREE_FLUID_PRESSURE = 0;
constexpr size_t INTERPOLATION_DEGREE_INTERFACE_PRESSURE = 0;
constexpr size_t INTERPOLATION_DEGREE_MAX = std::max(std::max(std::max(INTERPOLATION_DEGREE_SOLID_VELOCITY, INTERPOLATION_DEGREE_SOLID_PRESSURE), std::max(INTERPOLATION_DEGREE_FLUID_VELOCITY, INTERPOLATION_DEGREE_FLUID_PRESSURE)), INTERPOLATION_DEGREE_INTERFACE_PRESSURE);

constexpr size_t KERNEL_SIZE = 2 * INTERPOLATION_DEGREE_MAX + 1 + 4;//Neighbour cells by interpolation kernel plus offset due to particles being stored with offset
constexpr size_t KERNEL_OFFSET = INTERPOLATION_DEGREE_MAX;

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

//FIXME: If INTERPOLATION_DEGREE_MAX is too big neighbour blocks were not activated
static_assert((INTERPOLATION_DEGREE_MAX / config::G_BLOCKSIZE) <= 1 && "Neighbour blocks not activated");

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

template<MaterialE MaterialTypeSolid>
__forceinline__ __device__ void store_data_solid(const ParticleBuffer<MaterialTypeSolid> particle_buffer_solid, float* scaling_solid, float* pressure_solid_nominator, float* pressure_solid_denominator, const float W_0, const float W_2, const FetchParticleBufferDataIntermediate& data);

template<MaterialE MaterialTypeSolid>
__forceinline__ __device__ void store_data_neigbours_solid(const ParticleBuffer<MaterialTypeSolid> particle_buffer_solid, float* gradient_solid, const float W1_0, const float delta_w_2, const FetchParticleBufferDataIntermediate& data);

template<MaterialE MaterialTypeSolid>
__forceinline__ __device__ void store_data_neigbours_coupling_solid(const ParticleBuffer<MaterialTypeSolid> particle_buffer_solid, float* coupling_solid, const float W_2, const float W1_0, const float contact_area, const float normal);

template<MaterialE MaterialTypeFluid>
__forceinline__ __device__ void store_data_fluid(const ParticleBuffer<MaterialTypeFluid> particle_buffer_solid, const FetchParticleBufferDataIntermediate& data);

template<MaterialE MaterialTypeFluid>
__forceinline__ __device__ void store_data_neigbours_fluid(const ParticleBuffer<MaterialTypeFluid> particle_buffer_solid, float* gradient_fluid, float* boundary_fluid, const float W_1, const float W1_0, const float delta_w_1, const FetchParticleBufferDataIntermediate& data);

template<MaterialE MaterialTypeFluid>
__forceinline__ __device__ void store_data_neigbours_coupling_fluid(const ParticleBuffer<MaterialTypeFluid> particle_buffer_fluid, float* coupling_fluid, const float W_1, const float W1_0, const float contact_area, const float normal);


template<>
__forceinline__ __device__ void store_data_solid<MaterialE::J_FLUID>(const ParticleBuffer<MaterialE::J_FLUID> particle_buffer_solid, float* scaling_solid, float* pressure_solid_nominator, float* pressure_solid_denominator, const float W_0, const float W_2, const FetchParticleBufferDataIntermediate& data){
	printf("Material type not supported for coupling as solid.");
}

template<>
__forceinline__ __device__ void store_data_solid<MaterialE::FIXED_COROTATED>(const ParticleBuffer<MaterialE::FIXED_COROTATED> particle_buffer_solid, float* scaling_solid, float* pressure_solid_nominator, float* pressure_solid_denominator, const float W_0, const float W_2, const FetchParticleBufferDataIntermediate& data){
	printf("Material type not supported for coupling as solid.");
}

template<>
__forceinline__ __device__ void store_data_solid<MaterialE::SAND>(const ParticleBuffer<MaterialE::SAND> particle_buffer_solid, float* scaling_solid, float* pressure_solid_nominator, float* pressure_solid_denominator, const float W_0, const float W_2, const FetchParticleBufferDataIntermediate& data){
	printf("Material type not supported for coupling as solid.");
}

template<>
__forceinline__ __device__ void store_data_solid<MaterialE::NACC>(const ParticleBuffer<MaterialE::NACC> particle_buffer_solid, float* scaling_solid, float* pressure_solid_nominator, float* pressure_solid_denominator, const float W_0, const float W_2, const FetchParticleBufferDataIntermediate& data){
	printf("Material type not supported for coupling as solid.");
}

template<>
__forceinline__ __device__ void store_data_solid<MaterialE::FIXED_COROTATED_GHOST>(const ParticleBuffer<MaterialE::FIXED_COROTATED_GHOST> particle_buffer_solid, float* scaling_solid, float* pressure_solid_nominator, float* pressure_solid_denominator, const float W_0, const float W_2, const FetchParticleBufferDataIntermediate& data){			
	const float volume = (data.mass / particle_buffer_solid.rho);
	
	if(scaling_solid != nullptr){
		atomicAdd(scaling_solid, (volume / particle_buffer_solid.lambda) * W_0);
	}
	if(pressure_solid_nominator != nullptr){
		atomicAdd(pressure_solid_nominator, volume * data.J * (-particle_buffer_solid.lambda * (data.J - 1.0f)) * W_0);
	}
	if(pressure_solid_denominator != nullptr){
		atomicAdd(pressure_solid_denominator, volume * data.J * W_0);
	}
}

template<>
__forceinline__ __device__ void store_data_neigbours_solid<MaterialE::J_FLUID>(const ParticleBuffer<MaterialE::J_FLUID> particle_buffer_solid, float* gradient_solid, const float W1_0, const float delta_w_2, const FetchParticleBufferDataIntermediate& data){
	printf("Material type not supported for coupling as solid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_solid<MaterialE::FIXED_COROTATED>(const ParticleBuffer<MaterialE::FIXED_COROTATED> particle_buffer_solid, float* gradient_solid, const float W1_0, const float delta_w_2, const FetchParticleBufferDataIntermediate& data){
	printf("Material type not supported for coupling as solid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_solid<MaterialE::SAND>(const ParticleBuffer<MaterialE::SAND> particle_buffer_solid, float* gradient_solid, const float W1_0, const float delta_w_2, const FetchParticleBufferDataIntermediate& data){
	printf("Material type not supported for coupling as solid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_solid<MaterialE::NACC>(const ParticleBuffer<MaterialE::NACC> particle_buffer_solid, float* gradient_solid, const float W1_0, const float delta_w_2, const FetchParticleBufferDataIntermediate& data){
	printf("Material type not supported for coupling as solid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_solid<MaterialE::FIXED_COROTATED_GHOST>(const ParticleBuffer<MaterialE::FIXED_COROTATED_GHOST> particle_buffer_solid, float* gradient_solid, const float W1_0, const float delta_w_2, const FetchParticleBufferDataIntermediate& data){
	if(gradient_solid != nullptr){
		atomicAdd(gradient_solid, -(data.mass / particle_buffer_solid.rho) * data.J * W1_0 * delta_w_2);
	}
}
								
template<>
__forceinline__ __device__ void store_data_neigbours_coupling_solid<MaterialE::J_FLUID>(const ParticleBuffer<MaterialE::J_FLUID> particle_buffer_solid, float* coupling_solid, const float W_2, const float W1_0, const float contact_area, const float normal){
	printf("Material type not supported for coupling as solid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_coupling_solid<MaterialE::FIXED_COROTATED>(const ParticleBuffer<MaterialE::FIXED_COROTATED> particle_buffer_solid, float* coupling_solid, const float W_2, const float W1_0, const float contact_area, const float normal){
	printf("Material type not supported for coupling as solid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_coupling_solid<MaterialE::SAND>(const ParticleBuffer<MaterialE::SAND> particle_buffer_solid, float* coupling_solid, const float W_2, const float W1_0, const float contact_area, const float normal){
	printf("Material type not supported for coupling as solid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_coupling_solid<MaterialE::NACC>(const ParticleBuffer<MaterialE::NACC> particle_buffer_solid, float* coupling_solid, const float W_2, const float W1_0, const float contact_area, const float normal){
	printf("Material type not supported for coupling as solid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_coupling_solid<MaterialE::FIXED_COROTATED_GHOST>(const ParticleBuffer<MaterialE::FIXED_COROTATED_GHOST> particle_buffer_solid, float* coupling_solid, const float W_2, const float W1_0, const float contact_area, const float normal){
	if(coupling_solid != nullptr){
		atomicAdd(coupling_solid, contact_area * W_2 * W1_0 * normal);
	}
}

template<>
__forceinline__ __device__ void store_data_fluid<MaterialE::J_FLUID>(const ParticleBuffer<MaterialE::J_FLUID> particle_buffer_fluid, const FetchParticleBufferDataIntermediate& data){			
	//Nothing
}

template<>
__forceinline__ __device__ void store_data_fluid<MaterialE::FIXED_COROTATED>(const ParticleBuffer<MaterialE::FIXED_COROTATED> particle_buffer_fluid, const FetchParticleBufferDataIntermediate& data){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_fluid<MaterialE::SAND>(const ParticleBuffer<MaterialE::SAND> particle_buffer_fluid, const FetchParticleBufferDataIntermediate& data){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_fluid<MaterialE::NACC>(const ParticleBuffer<MaterialE::NACC> particle_buffer_fluid, const FetchParticleBufferDataIntermediate& data){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_fluid<MaterialE::FIXED_COROTATED_GHOST>(const ParticleBuffer<MaterialE::FIXED_COROTATED_GHOST> particle_buffer_fluid, const FetchParticleBufferDataIntermediate& data){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_fluid<MaterialE::J_FLUID>(const ParticleBuffer<MaterialE::J_FLUID> particle_buffer_fluid, float* gradient_fluid, float* boundary_fluid, const float W_1, const float W1_0, const float delta_w_1, const FetchParticleBufferDataIntermediate& data){
	if(gradient_fluid != nullptr){
		atomicAdd(gradient_fluid, -(data.mass / particle_buffer_fluid.rho) * data.J * W1_0 * delta_w_1);
	}
	if(boundary_fluid != nullptr){
		//FIXME: Is that correct?  Actually also not particle based maybe? And just add once?
		//atomicAdd(boundary_fluid, W_1 * W1_0 * boundary_normal[alpha]);
	}
}

template<>
__forceinline__ __device__ void store_data_neigbours_fluid<MaterialE::FIXED_COROTATED>(const ParticleBuffer<MaterialE::FIXED_COROTATED> particle_buffer_fluid, float* gradient_fluid, float* boundary_fluid, const float W_1, const float W1_0, const float delta_w_1, const FetchParticleBufferDataIntermediate& data){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_fluid<MaterialE::SAND>(const ParticleBuffer<MaterialE::SAND> particle_buffer_fluid, float* gradient_fluid, float* boundary_fluid, const float W_1, const float W1_0, const float delta_w_1, const FetchParticleBufferDataIntermediate& data){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_fluid<MaterialE::NACC>(const ParticleBuffer<MaterialE::NACC> particle_buffer_fluid, float* gradient_fluid, float* boundary_fluid, const float W_1, const float W1_0, const float delta_w_1, const FetchParticleBufferDataIntermediate& data){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_fluid<MaterialE::FIXED_COROTATED_GHOST>(const ParticleBuffer<MaterialE::FIXED_COROTATED_GHOST> particle_buffer_fluid, float* gradient_fluid, float* boundary_fluid, const float W_1, const float W1_0, const float delta_w_1, const FetchParticleBufferDataIntermediate& data){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_coupling_fluid<MaterialE::J_FLUID>(const ParticleBuffer<MaterialE::J_FLUID> particle_buffer_fluid, float* coupling_fluid, const float W_1, const float W1_0, const float contact_area, const float normal){
	if(coupling_fluid != nullptr){
		atomicAdd(coupling_fluid, contact_area * W_1 * W1_0 * normal);
	}
}

template<>
__forceinline__ __device__ void store_data_neigbours_coupling_fluid<MaterialE::FIXED_COROTATED>(const ParticleBuffer<MaterialE::FIXED_COROTATED> particle_buffer_fluid, float* coupling_fluid, const float W_1, const float W1_0, const float contact_area, const float normal){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_coupling_fluid<MaterialE::SAND>(const ParticleBuffer<MaterialE::SAND> particle_buffer_fluid, float* coupling_fluid, const float W_1, const float W1_0, const float contact_area, const float normal){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_coupling_fluid<MaterialE::NACC>(const ParticleBuffer<MaterialE::NACC> particle_buffer_fluid, float* coupling_fluid, const float W_1, const float W1_0, const float contact_area, const float normal){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_coupling_fluid<MaterialE::FIXED_COROTATED_GHOST>(const ParticleBuffer<MaterialE::FIXED_COROTATED_GHOST> particle_buffer_fluid, float* coupling_fluid, const float W_1, const float W1_0, const float contact_area, const float normal){
	printf("Material type not supported for coupling as fluid.");
}

template<MaterialE MaterialType>
__forceinline__ __device__ void update_strain(const ParticleBuffer<MaterialType> particle_buffer, int src_blockno, int particle_id_in_block, const float weighted_pressure);

template<>
__forceinline__ __device__ void update_strain<MaterialE::J_FLUID>(const ParticleBuffer<MaterialE::J_FLUID> particle_buffer, int src_blockno, int particle_id_in_block, const float weighted_pressure) {
	printf("Material type not supported for updating strain.");
}

template<>
__forceinline__ __device__ void update_strain(const ParticleBuffer<MaterialE::FIXED_COROTATED> particle_buffer, int src_blockno, int particle_id_in_block, const float weighted_pressure) {
	printf("Material type not supported for updating strain.");
}

template<>
__forceinline__ __device__ void update_strain(const ParticleBuffer<MaterialE::SAND> particle_buffer, int src_blockno, int particle_id_in_block, const float weighted_pressure) {
	printf("Material type not supported for updating strain.");
}

template<>
__forceinline__ __device__ void update_strain(const ParticleBuffer<MaterialE::NACC> particle_buffer, int src_blockno, int particle_id_in_block, const float weighted_pressure) {
	printf("Material type not supported for updating strain.");
}

template<>
__forceinline__ __device__ void update_strain(const ParticleBuffer<MaterialE::FIXED_COROTATED_GHOST> particle_buffer, int src_blockno, int particle_id_in_block, const float weighted_pressure) {
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

template<typename Partition, typename Grid, MaterialE MaterialTypeSolid>
__forceinline__ __device__ void aggregate_data_solid(const ParticleBuffer<MaterialTypeSolid> particle_buffer_solid, const ParticleBuffer<MaterialTypeSolid> next_particle_buffer_solid, const Partition prev_partition, const Grid grid_solid, const int current_blockno, const ivec3 current_blockid, const ivec3 block_cellid, const int particle_id_in_block, const int column_start, const int column_end, float* scaling_solid, float* pressure_solid_nominator, float* pressure_solid_denominator, float* mass_solid, float* gradient_solid) {
	const int column_range = column_end - column_start;
	
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

	//Fetch position and determinant of deformation gradient
	FetchParticleBufferDataIntermediate fetch_particle_buffer_tmp = {};
	fetch_particle_buffer_data<MaterialTypeSolid>(particle_buffer_solid, advection_source_blockno, source_pidib, fetch_particle_buffer_tmp);
	const float mass = fetch_particle_buffer_tmp.mass;
	vec3 pos {fetch_particle_buffer_tmp.pos[0], fetch_particle_buffer_tmp.pos[1], fetch_particle_buffer_tmp.pos[2]};
	//float J	 = fetch_particle_buffer_tmp.J;
	
	//Get position of grid cell
	const ivec3 global_base_index_solid_0 = get_cell_id<0>(pos.data_arr(), grid_solid.get_offset());
	const ivec3 global_base_index_solid_2 = get_cell_id<2>(pos.data_arr(), grid_solid.get_offset());

	//Get position relative to grid cell
	const vec3 local_pos_solid_0 = pos - (global_base_index_solid_0 + vec3(grid_solid.get_offset()[0], grid_solid.get_offset()[1], grid_solid.get_offset()[2])) * config::G_DX;
	const vec3 local_pos_solid_2 = pos - (global_base_index_solid_2 + vec3(grid_solid.get_offset()[0], grid_solid.get_offset()[1], grid_solid.get_offset()[2])) * config::G_DX;

	//Calculate weights
	vec3x3 weight_solid_0;
	vec3x3 weight_solid_2;
	vec3x3 gradient_weight_solid_2;
	
	#pragma unroll 3
	for(int dd = 0; dd < 3; ++dd) {
		const std::array<float, INTERPOLATION_DEGREE_SOLID_PRESSURE + 1> current_weight_solid_0 = bspline_weight<float, INTERPOLATION_DEGREE_SOLID_PRESSURE>(local_pos_solid_0[dd]);
		for(int i = 0; i < INTERPOLATION_DEGREE_SOLID_PRESSURE + 1; ++i){
			weight_solid_0(dd, i)		  = current_weight_solid_0[i];
		}
		for(int i = INTERPOLATION_DEGREE_SOLID_PRESSURE + 1; i < 3; ++i){
			weight_solid_0(dd, i)		  = 0.0f;
		}
		
		const std::array<float, INTERPOLATION_DEGREE_SOLID_VELOCITY + 1> current_weight_solid_2 = bspline_weight<float, INTERPOLATION_DEGREE_SOLID_VELOCITY>(local_pos_solid_2[dd]);
		for(int i = 0; i < INTERPOLATION_DEGREE_SOLID_VELOCITY + 1; ++i){
			weight_solid_2(dd, i)		  = current_weight_solid_2[i];
		}
		for(int i = INTERPOLATION_DEGREE_SOLID_VELOCITY + 1; i < 3; ++i){
			weight_solid_2(dd, i)		  = 0.0f;
		}
		
		const std::array<float, INTERPOLATION_DEGREE_SOLID_VELOCITY + 1> current_gradient_weight_solid_2 = bspline_gradient_weight<float, INTERPOLATION_DEGREE_SOLID_VELOCITY>(local_pos_solid_2[dd]);
		for(int i = 0; i < INTERPOLATION_DEGREE_SOLID_VELOCITY + 1; ++i){
			gradient_weight_solid_2(dd, i)		  = current_gradient_weight_solid_2[i];
		}
		for(int i = INTERPOLATION_DEGREE_SOLID_VELOCITY + 1; i < 3; ++i){
			gradient_weight_solid_2(dd, i)		  = 0.0f;
		}
	}
	
	//Store data
	//Note: Weights are 0 if outside of interpolation degree/radius around particles cell
	//Foreach node in the block we add values accoring to particle kernel, also handling all neighbours of the particles cell
	/*for(char i = -static_cast<char>(INTERPOLATION_DEGREE_MAX); i < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; i++) {
		for(char j = -static_cast<char>(INTERPOLATION_DEGREE_MAX); j < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; j++) {
			for(char k = -static_cast<char>(INTERPOLATION_DEGREE_MAX); k < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; k++) {
				const ivec3 local_id = (global_base_index_solid_2 - block_cellid) + ivec3(i, j, k);
				//Only handle for nodes in current block
				if(
					   (local_id[0] >= 0 && local_id[0] < config::G_BLOCKSIZE)
					&& (local_id[1] >= 0 && local_id[1] < config::G_BLOCKSIZE)
					&& (local_id[2] >= 0 && local_id[2] < config::G_BLOCKSIZE)
				){
					//Weight
					const float W_0 = weight_solid_0(0, std::abs(i)) * weight_solid_0(1, std::abs(j)) * weight_solid_0(2, std::abs(k));
					const float W_2 = weight_solid_2(0, std::abs(i)) * weight_solid_2(1, std::abs(j)) * weight_solid_2(2, std::abs(k));
					
					float* current_scaling_solid = (scaling_solid == nullptr ? nullptr : &(scaling_solid[(config::G_BLOCKSIZE * config::G_BLOCKSIZE) * local_id[0] + config::G_BLOCKSIZE * local_id[1] + local_id[2]]));
					float* current_pressure_solid_nominator = (pressure_solid_nominator == nullptr ? nullptr : &(pressure_solid_nominator[(config::G_BLOCKSIZE * config::G_BLOCKSIZE) * local_id[0] + config::G_BLOCKSIZE * local_id[1] + local_id[2]]));
					float* current_pressure_solid_denominator = (pressure_solid_denominator == nullptr ? nullptr : &(pressure_solid_denominator[(config::G_BLOCKSIZE * config::G_BLOCKSIZE) * local_id[0] + config::G_BLOCKSIZE * local_id[1] + local_id[2]]));
					
					store_data_solid(particle_buffer_solid, current_scaling_solid, current_pressure_solid_nominator, current_pressure_solid_denominator, W_0, W_2, fetch_particle_buffer_tmp);
						
					for(size_t alpha = 0; alpha < 3; ++alpha){
						const float delta_w_2 = ((alpha == 0 ? gradient_weight_solid_2(0, std::abs(i)) : weight_solid_2(0, std::abs(i))) * (alpha == 1 ? gradient_weight_solid_2(1, std::abs(j)) : weight_solid_2(1, std::abs(j))) * (alpha == 2 ? gradient_weight_solid_2(2, std::abs(k)) : weight_solid_2(2, std::abs(k)))) * config::G_DX_INV;
						
						if(mass_solid != nullptr){
							atomicAdd(&(mass_solid[(3 * config::G_BLOCKSIZE * config::G_BLOCKSIZE) * local_id[0] + (3 * config::G_BLOCKSIZE) * local_id[1] + 3 * local_id[2] + alpha]), mass * W_2);
						}
						
						//Handle all neighbours
						for(char i1 = -static_cast<char>(INTERPOLATION_DEGREE_MAX); i1 < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; i1++) {
							for(char j1 = -static_cast<char>(INTERPOLATION_DEGREE_MAX); j1 < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; j1++) {
								for(char k1 = -static_cast<char>(INTERPOLATION_DEGREE_MAX); k1 < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; k1++) {
									const ivec3 neighbour_local_offset {i1, j1, k1};
									const ivec3 neighbour_kernel_offset = ivec3(i, j, k) + neighbour_local_offset;
									const ivec3 neighbour_absolute_kernel_offset {std::abs(i1), std::abs(j1), std::abs(k1)};
									const ivec3 neighbour_absolute_local_offset = neighbour_local_offset + ivec3(static_cast<int>(INTERPOLATION_DEGREE_MAX), static_cast<int>(INTERPOLATION_DEGREE_MAX), static_cast<int>(INTERPOLATION_DEGREE_MAX));
									
									const int column = neighbour_absolute_local_offset[0] * ((2 * INTERPOLATION_DEGREE_MAX + 1) * (2 * INTERPOLATION_DEGREE_MAX + 1)) + neighbour_absolute_local_offset[1] * (2 * INTERPOLATION_DEGREE_MAX + 1) + neighbour_absolute_local_offset[2];
									//Only handle for neighbours of neighbour
									if(
										   (column >= column_start && column < column_end)
										&& (neighbour_absolute_kernel_offset[0] < INTERPOLATION_DEGREE_MAX)
										&& (neighbour_absolute_kernel_offset[1] < INTERPOLATION_DEGREE_MAX)
										&& (neighbour_absolute_kernel_offset[2] < INTERPOLATION_DEGREE_MAX)
									){
										const int local_column = column - column_start;
										
										const float W1_0 = weight_solid_0(0, neighbour_absolute_kernel_offset[0]) * weight_solid_0(1, neighbour_absolute_kernel_offset[1]) * weight_solid_0(2, neighbour_absolute_kernel_offset[2]);
										
										float* current_gradient_solid = (gradient_solid == nullptr ? nullptr : &(gradient_solid[(3 * config::G_BLOCKSIZE * config::G_BLOCKSIZE * column_range) * local_id[0] + (3 * config::G_BLOCKSIZE * column_range) * local_id[1] + (3 * column_range) * local_id[2] + 3 * local_column + alpha]));

										store_data_neigbours_solid(particle_buffer_solid, current_gradient_solid, W1_0, delta_w_2, fetch_particle_buffer_tmp);
									}
								}
							}
						}
					}
				}
			}
		}
	}*/
	
	for(char i = 0; i < static_cast<char>(config::G_BLOCKSIZE); i++) {
		for(char j = 0; j < static_cast<char>(config::G_BLOCKSIZE); j++) {
			for(char k = 0; k < static_cast<char>(config::G_BLOCKSIZE); k++) {
				const ivec3 local_offset_0 = global_base_index_solid_0 - (block_cellid + ivec3(i, j, k));
				const ivec3 local_offset_2 = global_base_index_solid_2 - (block_cellid + ivec3(i, j, k));
				
				const ivec3 absolute_local_offset_0 {std::abs(local_offset_0[0]), std::abs(local_offset_0[1]), std::abs(local_offset_0[2])};
				const ivec3 absolute_local_offset_2 {std::abs(local_offset_2[0]), std::abs(local_offset_2[1]), std::abs(local_offset_2[2])};

				//Weight
				const float W_0 = (absolute_local_offset_0[0] < 3 ? weight_solid_0(0, absolute_local_offset_0[0]) : 0.0f) * (absolute_local_offset_0[1] < 3 ? weight_solid_0(1, absolute_local_offset_0[1]) : 0.0f) * (absolute_local_offset_0[2] < 3 ? weight_solid_0(2, absolute_local_offset_0[2]) : 0.0f);
				const float W_2 = (absolute_local_offset_2[0] < 3 ? weight_solid_2(0, absolute_local_offset_2[0]) : 0.0f) * (absolute_local_offset_2[1] < 3 ? weight_solid_2(1, absolute_local_offset_2[1]) : 0.0f) * (absolute_local_offset_2[2] < 3 ? weight_solid_2(2, absolute_local_offset_2[2]) : 0.0f);
				
				float* current_scaling_solid = (scaling_solid == nullptr ? nullptr : &(scaling_solid[(config::G_BLOCKSIZE * config::G_BLOCKSIZE) * i + config::G_BLOCKSIZE * j + k]));
				float* current_pressure_solid_nominator = (pressure_solid_nominator == nullptr ? nullptr : &(pressure_solid_nominator[(config::G_BLOCKSIZE * config::G_BLOCKSIZE) * i + config::G_BLOCKSIZE * j + k]));
				float* current_pressure_solid_denominator = (pressure_solid_denominator == nullptr ? nullptr : &(pressure_solid_denominator[(config::G_BLOCKSIZE * config::G_BLOCKSIZE) * i + config::G_BLOCKSIZE * j + k]));
				
				store_data_solid(particle_buffer_solid, current_scaling_solid, current_pressure_solid_nominator, current_pressure_solid_denominator, W_0, W_2, fetch_particle_buffer_tmp);
					
				for(size_t alpha = 0; alpha < 3; ++alpha){
					const float delta_w_2 = ((alpha == 0 ? (absolute_local_offset_2[0] < 3 ? gradient_weight_solid_2(0, absolute_local_offset_2[0]) : 0.0f) : (absolute_local_offset_2[0] < 3 ? weight_solid_2(0, absolute_local_offset_2[0]) : 0.0f)) * (alpha == 1 ? (absolute_local_offset_2[1] < 3 ? gradient_weight_solid_2(1, absolute_local_offset_2[1]) : 0.0f) : (absolute_local_offset_2[1] < 3 ? weight_solid_2(1, absolute_local_offset_2[1]) : 0.0f)) * (alpha == 2 ? (absolute_local_offset_2[2] < 3 ? gradient_weight_solid_2(2, absolute_local_offset_2[2]) : 0.0f) : (absolute_local_offset_2[2] < 3 ? weight_solid_2(2, absolute_local_offset_2[2]) : 0.0f))) * config::G_DX_INV;
					
					if(mass_solid != nullptr){
						atomicAdd(&(mass_solid[(3 * config::G_BLOCKSIZE * config::G_BLOCKSIZE) * i + (3 * config::G_BLOCKSIZE) * j + 3 * k + alpha]), mass * W_2);
					}
					
					//Handle all neighbours
					for(char i1 = -static_cast<char>(INTERPOLATION_DEGREE_MAX); i1 < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; i1++) {
						for(char j1 = -static_cast<char>(INTERPOLATION_DEGREE_MAX); j1 < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; j1++) {
							for(char k1 = -static_cast<char>(INTERPOLATION_DEGREE_MAX); k1 < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; k1++) {
								const ivec3 neighbour_local_offset_0 = global_base_index_solid_0 - (block_cellid + ivec3(i, j, k) + ivec3(i1, j1, k1));
								const ivec3 neighbour_absolute_local_offset {std::abs(neighbour_local_offset_0[0]), std::abs(neighbour_local_offset_0[1]), std::abs(neighbour_local_offset_0[2])};
								const ivec3 neighbour_array_local_offset = ivec3(i1, j1, k1) + ivec3(static_cast<int>(INTERPOLATION_DEGREE_MAX), static_cast<int>(INTERPOLATION_DEGREE_MAX), static_cast<int>(INTERPOLATION_DEGREE_MAX));
								
								const int column = neighbour_array_local_offset[0] * ((2 * INTERPOLATION_DEGREE_MAX + 1) * (2 * INTERPOLATION_DEGREE_MAX + 1)) + neighbour_array_local_offset[1] * (2 * INTERPOLATION_DEGREE_MAX + 1) + neighbour_array_local_offset[2];
								//Only handle for neighbours of neighbour
								if(
									   (column >= column_start && column < column_end)
								){
									const int local_column = column - column_start;
									
									const float W1_0 = (neighbour_absolute_local_offset[0] < 3 ? weight_solid_0(0, neighbour_absolute_local_offset[0]) : 0.0f) * (neighbour_absolute_local_offset[1] < 3 ? weight_solid_0(1, neighbour_absolute_local_offset[1]) : 0.0f) * (neighbour_absolute_local_offset[2] < 3 ? weight_solid_0(2, neighbour_absolute_local_offset[2]) : 0.0f);
									
									float* current_gradient_solid = (gradient_solid == nullptr ? nullptr : &(gradient_solid[(3 * config::G_BLOCKSIZE * config::G_BLOCKSIZE * column_range) * i + (3 * config::G_BLOCKSIZE * column_range) * j + (3 * column_range) * k + 3 * local_column + alpha]));

									store_data_neigbours_solid(particle_buffer_solid, current_gradient_solid, W1_0, delta_w_2, fetch_particle_buffer_tmp);
								}
							}
						}
					}
				}
			}
		}
	}
}

template<typename Partition, typename Grid, MaterialE MaterialTypeFluid>
__forceinline__ __device__ void aggregate_data_fluid(const ParticleBuffer<MaterialTypeFluid> particle_buffer_fluid, const ParticleBuffer<MaterialTypeFluid> next_particle_buffer_fluid, const Partition prev_partition, const Grid grid_solid, const Grid grid_fluid, const int current_blockno, const ivec3 current_blockid, const ivec3 block_cellid, const int particle_id_in_block, const int column_start, const int column_end, float* mass_fluid, float* gradient_fluid, float* boundary_fluid) {
	const int column_range = column_end - column_start;
	
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

	//Fetch position and determinant of deformation gradient
	FetchParticleBufferDataIntermediate fetch_particle_buffer_tmp = {};
	fetch_particle_buffer_data<MaterialTypeFluid>(particle_buffer_fluid, advection_source_blockno, source_pidib, fetch_particle_buffer_tmp);
	const float mass = fetch_particle_buffer_tmp.mass;
	vec3 pos {fetch_particle_buffer_tmp.pos[0], fetch_particle_buffer_tmp.pos[1], fetch_particle_buffer_tmp.pos[2]};
	//float J	 = fetch_particle_buffer_tmp.J;
	
	//Get position of grid cell
	const ivec3 global_base_index_solid_0 = get_cell_id<0>(pos.data_arr(), grid_solid.get_offset());
	const ivec3 global_base_index_fluid_1 = get_cell_id<1>(pos.data_arr(), grid_fluid.get_offset());
	const ivec3 global_base_index_fluid_2 = get_cell_id<2>(pos.data_arr(), grid_fluid.get_offset());
	
	//Get position relative to grid cell
	const vec3 local_pos_solid_0 = pos - (global_base_index_solid_0 + vec3(grid_solid.get_offset()[0], grid_solid.get_offset()[1], grid_solid.get_offset()[2])) * config::G_DX;
	const vec3 local_pos_fluid_1 = pos - (global_base_index_fluid_1 + vec3(grid_fluid.get_offset()[0], grid_fluid.get_offset()[1], grid_fluid.get_offset()[2])) * config::G_DX;

	//Calculate weights
	vec3x3 weight_solid_0;
	vec3x3 weight_fluid_1;
	vec3x3 gradient_weight_fluid_1;
	
	#pragma unroll 3
	for(int dd = 0; dd < 3; ++dd) {
		const std::array<float, INTERPOLATION_DEGREE_FLUID_PRESSURE + 1> current_weight_solid_0 = bspline_weight<float, INTERPOLATION_DEGREE_FLUID_PRESSURE>(local_pos_solid_0[dd]);
		for(int i = 0; i < INTERPOLATION_DEGREE_FLUID_PRESSURE + 1; ++i){
			weight_solid_0(dd, i)		  = current_weight_solid_0[i];
		}
		for(int i = INTERPOLATION_DEGREE_FLUID_PRESSURE + 1; i < 3; ++i){
			weight_solid_0(dd, i)		  = 0.0f;
		}
		
		const std::array<float, INTERPOLATION_DEGREE_FLUID_VELOCITY + 1> current_weight_fluid_1 = bspline_weight<float, INTERPOLATION_DEGREE_FLUID_VELOCITY>(local_pos_fluid_1[dd]);
		for(int i = 0; i < INTERPOLATION_DEGREE_FLUID_VELOCITY + 1; ++i){
			weight_fluid_1(dd, i)		  = current_weight_fluid_1[i];
		}
		for(int i = INTERPOLATION_DEGREE_FLUID_VELOCITY + 1; i < 3; ++i){
			weight_fluid_1(dd, i)		  = 0.0f;
		}
		
		const std::array<float, INTERPOLATION_DEGREE_FLUID_VELOCITY + 1> current_gradient_weight_fluid_1 = bspline_gradient_weight<float, INTERPOLATION_DEGREE_FLUID_VELOCITY>(local_pos_fluid_1[dd]);
		for(int i = 0; i < INTERPOLATION_DEGREE_FLUID_VELOCITY + 1; ++i){
			gradient_weight_fluid_1(dd, i)		  = current_gradient_weight_fluid_1[i];
		}
		for(int i = INTERPOLATION_DEGREE_FLUID_VELOCITY + 1; i < 3; ++i){
			gradient_weight_fluid_1(dd, i)		  = 0.0f;
		}
	}
	
	//Store data
	//Note: Weights are 0 if outside of interpolation degree/radius around particles cell
	//Foreach node in the block we add values accoring to particle kernel, also handling all neighbours of the particles cell
	/*for(char i = -static_cast<char>(INTERPOLATION_DEGREE_MAX); i < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; i++) {
		for(char j = -static_cast<char>(INTERPOLATION_DEGREE_MAX); j < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; j++) {
			for(char k = -static_cast<char>(INTERPOLATION_DEGREE_MAX); k < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; k++) {
				const ivec3 local_id = (global_base_index_fluid_2 - block_cellid) + ivec3(i, j, k);
				//Only handle for nodes in current block
				if(
					   (local_id[0] >= 0 && local_id[0] < config::G_BLOCKSIZE)
					&& (local_id[1] >= 0 && local_id[1] < config::G_BLOCKSIZE)
					&& (local_id[2] >= 0 && local_id[2] < config::G_BLOCKSIZE)
				){
					//Weight
					const float W_1 = weight_fluid_1(0, std::abs(i)) * weight_fluid_1(1, std::abs(j)) * weight_fluid_1(2, std::abs(k));
					
					store_data_fluid(particle_buffer_fluid, fetch_particle_buffer_tmp);
						
					for(size_t alpha = 0; alpha < 3; ++alpha){
						const float delta_w_1 = ((alpha == 0 ? gradient_weight_fluid_1(0, std::abs(i)) : weight_fluid_1(0, std::abs(i))) * (alpha == 1 ? gradient_weight_fluid_1(1, std::abs(j)) : weight_fluid_1(1, std::abs(j))) * (alpha == 2 ? gradient_weight_fluid_1(2, std::abs(k)) : weight_fluid_1(2, std::abs(k)))) * config::G_DX_INV;

						if(mass_fluid != nullptr){
							atomicAdd(&(mass_fluid[(3 * config::G_BLOCKSIZE * config::G_BLOCKSIZE) * local_id[0] + (3 * config::G_BLOCKSIZE) * local_id[1] + 3 * local_id[2] + alpha]), mass * W_1);
						}
						
						//Handle all neighbours
						for(char i1 = -static_cast<char>(INTERPOLATION_DEGREE_MAX); i1 < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; i1++) {
							for(char j1 = -static_cast<char>(INTERPOLATION_DEGREE_MAX); j1 < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; j1++) {
								for(char k1 = -static_cast<char>(INTERPOLATION_DEGREE_MAX); k1 < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; k1++) {
									const ivec3 neighbour_local_offset {i1, j1, k1};
									const ivec3 neighbour_kernel_offset = ivec3(i, j, k) + neighbour_local_offset;
									const ivec3 neighbour_absolute_kernel_offset {std::abs(i1), std::abs(j1), std::abs(k1)};
									const ivec3 neighbour_absolute_local_offset = neighbour_local_offset + ivec3(static_cast<int>(INTERPOLATION_DEGREE_MAX), static_cast<int>(INTERPOLATION_DEGREE_MAX), static_cast<int>(INTERPOLATION_DEGREE_MAX));
									
									const int column = neighbour_absolute_local_offset[0] * ((2 * INTERPOLATION_DEGREE_MAX + 1) * (2 * INTERPOLATION_DEGREE_MAX + 1)) + neighbour_absolute_local_offset[1] * (2 * INTERPOLATION_DEGREE_MAX + 1) + neighbour_absolute_local_offset[2];
									//Only handle for neighbours of neighbour
									if(
										   (column >= column_start && column < column_end)
										&& (neighbour_absolute_kernel_offset[0] < INTERPOLATION_DEGREE_MAX)
										&& (neighbour_absolute_kernel_offset[1] < INTERPOLATION_DEGREE_MAX)
										&& (neighbour_absolute_kernel_offset[2] < INTERPOLATION_DEGREE_MAX)
									){
										const int local_column = column - column_start;
										
										const float W1_0 = weight_solid_0(0, neighbour_absolute_kernel_offset[0]) * weight_solid_0(1, neighbour_absolute_kernel_offset[1]) * weight_solid_0(2, neighbour_absolute_kernel_offset[2]);
										
										float* current_gradient_fluid = (gradient_fluid == nullptr ? nullptr : &(gradient_fluid[(3 * config::G_BLOCKSIZE * config::G_BLOCKSIZE * column_range) * local_id[0] + (3 * config::G_BLOCKSIZE * column_range) * local_id[1] + (3 * column_range) * local_id[2] + 3 * local_column + alpha]));
										float* current_boundary_fluid = (boundary_fluid == nullptr ? nullptr : &(boundary_fluid[(3 * config::G_BLOCKSIZE * config::G_BLOCKSIZE * column_range) * local_id[0] + (3 * config::G_BLOCKSIZE * column_range) * local_id[1] + (3 * column_range) * local_id[2] + 3 * local_column + alpha]));
										
										store_data_neigbours_fluid(particle_buffer_fluid, current_gradient_fluid, current_boundary_fluid, W_1, W1_0, delta_w_1, fetch_particle_buffer_tmp);
									}
								}
							}
						}
					}
				}
			}
		}
	}*/
	
	for(char i = 0; i < static_cast<char>(config::G_BLOCKSIZE); i++) {
		for(char j = 0; j < static_cast<char>(config::G_BLOCKSIZE); j++) {
			for(char k = 0; k < static_cast<char>(config::G_BLOCKSIZE); k++) {
				const ivec3 local_offset_0_solid = global_base_index_solid_0 - (block_cellid + ivec3(i, j, k));
				const ivec3 local_offset_1_fluid = global_base_index_fluid_1 - (block_cellid + ivec3(i, j, k));
				
				const ivec3 absolute_local_offset_0_solid {std::abs(local_offset_0_solid[0]), std::abs(local_offset_0_solid[1]), std::abs(local_offset_0_solid[2])};
				const ivec3 absolute_local_offset_1_fluid {std::abs(local_offset_1_fluid[0]), std::abs(local_offset_1_fluid[1]), std::abs(local_offset_1_fluid[2])};

				//Weight
				const float W_1 = (absolute_local_offset_1_fluid[0] < 3 ? weight_fluid_1(0, absolute_local_offset_1_fluid[0]) : 0.0f) * (absolute_local_offset_1_fluid[1] < 3 ? weight_fluid_1(1, absolute_local_offset_1_fluid[1]) : 0.0f) * (absolute_local_offset_1_fluid[2] < 3 ? weight_fluid_1(2, absolute_local_offset_1_fluid[2]) : 0.0f);
				
				store_data_fluid(particle_buffer_fluid, fetch_particle_buffer_tmp);
					
				for(size_t alpha = 0; alpha < 3; ++alpha){
					const float delta_w_1 = ((alpha == 0 ? (absolute_local_offset_1_fluid[0] < 3 ? gradient_weight_fluid_1(0, absolute_local_offset_1_fluid[0]) : 0.0f) : (absolute_local_offset_1_fluid[0] < 3 ? weight_fluid_1(0, absolute_local_offset_1_fluid[0]) : 0.0f)) * (alpha == 1 ? (absolute_local_offset_1_fluid[1] < 3 ? gradient_weight_fluid_1(1, absolute_local_offset_1_fluid[1]) : 0.0f) : (absolute_local_offset_1_fluid[1] < 3 ? weight_fluid_1(1, absolute_local_offset_1_fluid[1]) : 0.0f)) * (alpha == 2 ? (absolute_local_offset_1_fluid[2] < 3 ? gradient_weight_fluid_1(2, absolute_local_offset_1_fluid[2]) : 0.0f) : (absolute_local_offset_1_fluid[2] < 3 ? weight_fluid_1(2, absolute_local_offset_1_fluid[2]) : 0.0f))) * config::G_DX_INV;
					
					if(mass_fluid != nullptr){
						atomicAdd(&(mass_fluid[(3 * config::G_BLOCKSIZE * config::G_BLOCKSIZE) * i + (3 * config::G_BLOCKSIZE) * j + 3 * k + alpha]), mass * W_1);
					}
					
					//Handle all neighbours
					for(char i1 = -static_cast<char>(INTERPOLATION_DEGREE_MAX); i1 < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; i1++) {
						for(char j1 = -static_cast<char>(INTERPOLATION_DEGREE_MAX); j1 < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; j1++) {
							for(char k1 = -static_cast<char>(INTERPOLATION_DEGREE_MAX); k1 < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; k1++) {
								const ivec3 neighbour_local_offset_0 = global_base_index_solid_0 - (block_cellid + ivec3(i, j, k) + ivec3(i1, j1, k1));
								const ivec3 neighbour_absolute_local_offset {std::abs(neighbour_local_offset_0[0]), std::abs(neighbour_local_offset_0[1]), std::abs(neighbour_local_offset_0[2])};
								const ivec3 neighbour_array_local_offset = ivec3(i1, j1, k1) + ivec3(static_cast<int>(INTERPOLATION_DEGREE_MAX), static_cast<int>(INTERPOLATION_DEGREE_MAX), static_cast<int>(INTERPOLATION_DEGREE_MAX));
								
								const int column = neighbour_array_local_offset[0] * ((2 * INTERPOLATION_DEGREE_MAX + 1) * (2 * INTERPOLATION_DEGREE_MAX + 1)) + neighbour_array_local_offset[1] * (2 * INTERPOLATION_DEGREE_MAX + 1) + neighbour_array_local_offset[2];
								//Only handle for neighbours of neighbour
								if(
									   (column >= column_start && column < column_end)
								){
									const int local_column = column - column_start;
									
									const float W1_0 = (neighbour_absolute_local_offset[0] < 3 ? weight_solid_0(0, neighbour_absolute_local_offset[0]) : 0.0f) * (neighbour_absolute_local_offset[1] < 3 ? weight_solid_0(1, neighbour_absolute_local_offset[1]) : 0.0f) * (neighbour_absolute_local_offset[2] < 3 ? weight_solid_0(2, neighbour_absolute_local_offset[2]) : 0.0f);
									
									float* current_gradient_fluid = (gradient_fluid == nullptr ? nullptr : &(gradient_fluid[(3 * config::G_BLOCKSIZE * config::G_BLOCKSIZE * column_range) * i + (3 * config::G_BLOCKSIZE * column_range) * j + (3 * column_range) * k + 3 * local_column + alpha]));
									float* current_boundary_fluid = (boundary_fluid == nullptr ? nullptr : &(boundary_fluid[(3 * config::G_BLOCKSIZE * config::G_BLOCKSIZE * column_range) * i + (3 * config::G_BLOCKSIZE * column_range) * j + (3 * column_range) * k + 3 * local_column + alpha]));
									
									store_data_neigbours_fluid(particle_buffer_fluid, current_gradient_fluid, current_boundary_fluid, W_1, W1_0, delta_w_1, fetch_particle_buffer_tmp);
								}
							}
						}
					}
				}
			}
		}
	}
}

template<typename Partition, typename Grid, MaterialE MaterialTypeSolid, MaterialE MaterialTypeFluid>
__forceinline__ __device__ void aggregate_data_coupling(const ParticleBuffer<MaterialTypeSolid> particle_buffer_solid, const ParticleBuffer<MaterialTypeFluid> particle_buffer_fluid, const ParticleBuffer<MaterialTypeSolid> next_particle_buffer_solid, const ParticleBuffer<MaterialTypeFluid> next_particle_buffer_fluid, const Partition prev_partition, const Grid grid_solid, const SurfaceParticleBuffer surface_particle_buffer, const int current_blockno, const ivec3 current_blockid, const ivec3 block_cellid, const int particle_id_in_block, const int column_start, const int column_end, float* coupling_solid, float* coupling_fluid) {
	const int column_range = column_end - column_start;
	
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

	auto surface_particle_bin = surface_particle_buffer.ch(_0, particle_buffer_solid.bin_offsets[advection_source_blockno] + source_pidib / config::G_BIN_CAPACITY);
	const int surface_particle_id_in_bin = source_pidib  % config::G_BIN_CAPACITY;

	//Fetch position and determinant of deformation gradient
	FetchParticleBufferDataIntermediate fetch_particle_buffer_tmp = {};
	fetch_particle_buffer_data<MaterialTypeSolid>(particle_buffer_solid, advection_source_blockno, source_pidib, fetch_particle_buffer_tmp);
	//const float mass = fetch_particle_buffer_tmp.mass;
	vec3 pos {fetch_particle_buffer_tmp.pos[0], fetch_particle_buffer_tmp.pos[1], fetch_particle_buffer_tmp.pos[2]};
	//float J	 = fetch_particle_buffer_tmp.J;
	
	vec3 normal;
	const SurfacePointType point_type = *reinterpret_cast<SurfacePointType*>(&surface_particle_bin.val(_0, surface_particle_id_in_bin));
	normal[0] = surface_particle_bin.val(_1, surface_particle_id_in_bin);
	normal[1] = surface_particle_bin.val(_2, surface_particle_id_in_bin);
	normal[2] = surface_particle_bin.val(_3, surface_particle_id_in_bin);
	const float contact_area = surface_particle_bin.val(_6, surface_particle_id_in_bin);
	
	//Get position of grid cell
	const ivec3 global_base_index_solid_0 = get_cell_id<0>(pos.data_arr(), grid_solid.get_offset());
	const ivec3 global_base_index_solid_1 = get_cell_id<1>(pos.data_arr(), grid_solid.get_offset());
	const ivec3 global_base_index_solid_2 = get_cell_id<2>(pos.data_arr(), grid_solid.get_offset());

	//Get position relative to grid cell
	const vec3 local_pos_solid_0 = pos - (global_base_index_solid_0 + vec3(grid_solid.get_offset()[0], grid_solid.get_offset()[1], grid_solid.get_offset()[2])) * config::G_DX;
	const vec3 local_pos_solid_1 = pos - (global_base_index_solid_1 + vec3(grid_solid.get_offset()[0], grid_solid.get_offset()[1], grid_solid.get_offset()[2])) * config::G_DX;
	const vec3 local_pos_solid_2 = pos - (global_base_index_solid_2 + vec3(grid_solid.get_offset()[0], grid_solid.get_offset()[1], grid_solid.get_offset()[2])) * config::G_DX;

	//Calculate weights
	vec3x3 weight_solid_0;
	vec3x3 weight_solid_2;
	
	vec3x3 weight_fluid_0;
	vec3x3 weight_fluid_1;
	
	#pragma unroll 3
	for(int dd = 0; dd < 3; ++dd) {
		const std::array<float, INTERPOLATION_DEGREE_SOLID_PRESSURE + 1> current_weight_solid_0 = bspline_weight<float, INTERPOLATION_DEGREE_SOLID_PRESSURE>(local_pos_solid_0[dd]);
		for(int i = 0; i < INTERPOLATION_DEGREE_SOLID_PRESSURE + 1; ++i){
			weight_solid_0(dd, i)		  = current_weight_solid_0[i];
		}
		for(int i = INTERPOLATION_DEGREE_SOLID_PRESSURE + 1; i < 3; ++i){
			weight_solid_0(dd, i)		  = 0.0f;
		}
		
		const std::array<float, INTERPOLATION_DEGREE_SOLID_VELOCITY + 1> current_weight_solid_2 = bspline_weight<float, INTERPOLATION_DEGREE_SOLID_VELOCITY>(local_pos_solid_2[dd]);
		for(int i = 0; i < INTERPOLATION_DEGREE_SOLID_VELOCITY + 1; ++i){
			weight_solid_2(dd, i)		  = current_weight_solid_2[i];
		}
		for(int i = INTERPOLATION_DEGREE_SOLID_VELOCITY + 1; i < 3; ++i){
			weight_solid_2(dd, i)		  = 0.0f;
		}
		
		const std::array<float, INTERPOLATION_DEGREE_FLUID_PRESSURE + 1> current_weight_fluid_0 = bspline_weight<float, INTERPOLATION_DEGREE_FLUID_PRESSURE>(local_pos_solid_0[dd]);
		for(int i = 0; i < INTERPOLATION_DEGREE_FLUID_PRESSURE + 1; ++i){
			weight_fluid_0(dd, i)		  = current_weight_fluid_0[i];
		}
		for(int i = INTERPOLATION_DEGREE_FLUID_PRESSURE + 1; i < 3; ++i){
			weight_fluid_0(dd, i)		  = 0.0f;
		}
		
		const std::array<float, INTERPOLATION_DEGREE_FLUID_VELOCITY + 1> current_weight_fluid_1 = bspline_weight<float, INTERPOLATION_DEGREE_FLUID_VELOCITY>(local_pos_solid_1[dd]);
		for(int i = 0; i < INTERPOLATION_DEGREE_FLUID_VELOCITY + 1; ++i){
			weight_fluid_1(dd, i)		  = current_weight_fluid_1[i];
		}
		for(int i = INTERPOLATION_DEGREE_FLUID_VELOCITY + 1; i < 3; ++i){
			weight_fluid_1(dd, i)		  = 0.0f;
		}
	}
	
	//Get near fluid particles
	bool has_neighbours = false;
	for(int grid_x = -1; grid_x <= 1; ++grid_x){
		for(int grid_y = -1; grid_y <= 1; ++grid_y){
			for(int grid_z = -1; grid_z <= 1; ++grid_z){
				const ivec3 cell_offset {grid_x, grid_y, grid_z};
				const ivec3 current_cellid = global_base_index_solid_2 + cell_offset;
				const ivec3 current_blockid = current_cellid / static_cast<int>(config::G_BLOCKSIZE);
				const int current_blockno_fluid = prev_partition.query(current_blockid);
				
				//Skip empty blocks
				if(current_blockno_fluid == -1){
					continue;
				}
				
				for(int particle_id_in_block_fluid = 0; particle_id_in_block_fluid <  next_particle_buffer_fluid.particle_bucket_sizes[current_blockno_fluid]; particle_id_in_block_fluid++) {
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

					//Fetch position and determinant of deformation gradient
					FetchParticleBufferDataIntermediate fetch_particle_buffer_tmp = {};
					fetch_particle_buffer_data<MaterialTypeFluid>(particle_buffer_fluid, advection_source_blockno_fluid, source_pidib_fluid, fetch_particle_buffer_tmp);
					//const float mass_fluid = fetch_particle_buffer_tmp.mass;
					vec3 pos_fluid {fetch_particle_buffer_tmp.pos[0], fetch_particle_buffer_tmp.pos[1], fetch_particle_buffer_tmp.pos[2]};
					//float J_fluid	 = fetch_particle_buffer_tmp.J;
					
					const vec3 diff = pos - pos_fluid;
					const float distance = std::sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
					
					if(distance <= 0.5f * config::G_DX){
						has_neighbours = true;
						break;
					}
				}
			}
		}
	}
	
	//Only percede if we have an interface
	//FIXME: Currently only handling outer points
	if(has_neighbours && point_type == SurfacePointType::OUTER_POINT){
	
		//Store data
		//Note: Weights are 0 if outside of interpolation degree/radius around particles cell
		//Foreach node in the block we add values accoring to particle kernel, also handling all neighbours of the particles cell
		/*for(char i = -static_cast<char>(INTERPOLATION_DEGREE_MAX); i < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; i++) {
			for(char j = -static_cast<char>(INTERPOLATION_DEGREE_MAX); j < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; j++) {
				for(char k = -static_cast<char>(INTERPOLATION_DEGREE_MAX); k < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; k++) {
					const ivec3 local_id = (global_base_index_solid_2 - block_cellid) + ivec3(i, j, k);
					//Only handle for nodes in current block
					if(
						   (local_id[0] >= 0 && local_id[0] < config::G_BLOCKSIZE)
						&& (local_id[1] >= 0 && local_id[1] < config::G_BLOCKSIZE)
						&& (local_id[2] >= 0 && local_id[2] < config::G_BLOCKSIZE)
					){
						//Weight
						const float W_2 = weight_solid_2(0, std::abs(i)) * weight_solid_2(1, std::abs(j)) * weight_solid_2(2, std::abs(k));
						
						const float W_1 = weight_fluid_1(0, std::abs(i)) * weight_fluid_1(1, std::abs(j)) * weight_fluid_1(2, std::abs(k));
						
						for(size_t alpha = 0; alpha < 3; ++alpha){
							//Handle all neighbours
							for(char i1 = -static_cast<char>(INTERPOLATION_DEGREE_MAX); i1 < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; i1++) {
								for(char j1 = -static_cast<char>(INTERPOLATION_DEGREE_MAX); j1 < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; j1++) {
									for(char k1 = -static_cast<char>(INTERPOLATION_DEGREE_MAX); k1 < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; k1++) {
										const ivec3 neighbour_local_offset {i1, j1, k1};
										const ivec3 neighbour_kernel_offset = ivec3(i, j, k) + neighbour_local_offset;
										const ivec3 neighbour_absolute_kernel_offset {std::abs(i1), std::abs(j1), std::abs(k1)};
										const ivec3 neighbour_absolute_local_offset = neighbour_local_offset + ivec3(static_cast<int>(INTERPOLATION_DEGREE_MAX), static_cast<int>(INTERPOLATION_DEGREE_MAX), static_cast<int>(INTERPOLATION_DEGREE_MAX));
										
										const int column = neighbour_absolute_local_offset[0] * ((2 * INTERPOLATION_DEGREE_MAX + 1) * (2 * INTERPOLATION_DEGREE_MAX + 1)) + neighbour_absolute_local_offset[1] * (2 * INTERPOLATION_DEGREE_MAX + 1) + neighbour_absolute_local_offset[2];
										//Only handle for neighbours of neighbour
										if(
											   (column >= column_start && column < column_end)
											&& (neighbour_absolute_kernel_offset[0] < INTERPOLATION_DEGREE_MAX)
											&& (neighbour_absolute_kernel_offset[1] < INTERPOLATION_DEGREE_MAX)
											&& (neighbour_absolute_kernel_offset[2] < INTERPOLATION_DEGREE_MAX)
										){
											const int local_column = column - column_start;
											
											const float W1_0_solid = weight_solid_0(0, neighbour_absolute_kernel_offset[0]) * weight_solid_0(1, neighbour_absolute_kernel_offset[1]) * weight_solid_0(2, neighbour_absolute_kernel_offset[2]);
											
											const float W1_0_fluid = weight_fluid_0(0, neighbour_absolute_kernel_offset[0]) * weight_fluid_0(1, neighbour_absolute_kernel_offset[1]) * weight_fluid_0(2, neighbour_absolute_kernel_offset[2]);
											
											
											float* current_coupling_solid = (coupling_solid == nullptr ? nullptr : &(coupling_solid[(3 * config::G_BLOCKSIZE * config::G_BLOCKSIZE * column_range) * local_id[0] + (3 * config::G_BLOCKSIZE * column_range) * local_id[1] + (3 * column_range) * local_id[2] + 3 * local_column + alpha]));
											float* current_coupling_fluid = (coupling_fluid == nullptr ? nullptr : &(coupling_fluid[(3 * config::G_BLOCKSIZE * config::G_BLOCKSIZE * column_range) * local_id[0] + (3 * config::G_BLOCKSIZE * column_range) * local_id[1] + (3 * column_range) * local_id[2] + 3 * local_column + alpha]));

											store_data_neigbours_coupling_solid(particle_buffer_solid, current_coupling_solid, W_2, W1_0_solid, contact_area, normal[alpha]);
											store_data_neigbours_coupling_fluid(particle_buffer_fluid, current_coupling_fluid, W_1, W1_0_fluid, contact_area, normal[alpha]);
										}
									}
								}
							}
						}
					}
				}
			}
		}*/
		
		for(char i = 0; i < static_cast<char>(config::G_BLOCKSIZE); i++) {
			for(char j = 0; j < static_cast<char>(config::G_BLOCKSIZE); j++) {
				for(char k = 0; k < static_cast<char>(config::G_BLOCKSIZE); k++) {
					const ivec3 local_offset_2_solid = global_base_index_solid_2 - (block_cellid + ivec3(i, j, k));
					const ivec3 local_offset_1_fluid = global_base_index_solid_1 - (block_cellid + ivec3(i, j, k));//NOTE: Using solid pos
					
					const ivec3 absolute_local_offset_2_solid {std::abs(local_offset_2_solid[0]), std::abs(local_offset_2_solid[1]), std::abs(local_offset_2_solid[2])};
					
					const ivec3 absolute_local_offset_1_fluid {std::abs(local_offset_1_fluid[0]), std::abs(local_offset_1_fluid[1]), std::abs(local_offset_1_fluid[2])};

					//Weight
					const float W_2 = (absolute_local_offset_2_solid[0] < 3 ? weight_solid_2(0, absolute_local_offset_2_solid[0]) : 0.0f) * (absolute_local_offset_2_solid[1] < 3 ? weight_solid_2(1, absolute_local_offset_2_solid[1]) : 0.0f) * (absolute_local_offset_2_solid[2] < 3 ? weight_solid_2(2, absolute_local_offset_2_solid[2]) : 0.0f);
					
					const float W_1 = (absolute_local_offset_1_fluid[0] < 3 ? weight_solid_2(0, absolute_local_offset_1_fluid[0]) : 0.0f) * (absolute_local_offset_1_fluid[1] < 3 ? weight_solid_2(1, absolute_local_offset_1_fluid[1]) : 0.0f) * (absolute_local_offset_1_fluid[2] < 3 ? weight_solid_2(2, absolute_local_offset_1_fluid[2]) : 0.0f);
					
					for(size_t alpha = 0; alpha < 3; ++alpha){
						//Handle all neighbours
						for(char i1 = -static_cast<char>(INTERPOLATION_DEGREE_MAX); i1 < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; i1++) {
							for(char j1 = -static_cast<char>(INTERPOLATION_DEGREE_MAX); j1 < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; j1++) {
								for(char k1 = -static_cast<char>(INTERPOLATION_DEGREE_MAX); k1 < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; k1++) {
									const ivec3 neighbour_local_offset_0_solid = global_base_index_solid_0 - (block_cellid + ivec3(i, j, k) + ivec3(i1, j1, k1));
									const ivec3 neighbour_local_offset_0_fluid = global_base_index_solid_0 - (block_cellid + ivec3(i, j, k) + ivec3(i1, j1, k1));//NOTE: Using solid pos
									
									const ivec3 neighbour_absolute_local_offset_solid {std::abs(neighbour_local_offset_0_solid[0]), std::abs(neighbour_local_offset_0_solid[1]), std::abs(neighbour_local_offset_0_solid[2])};
									const ivec3 neighbour_absolute_local_offset_fluid {std::abs(neighbour_local_offset_0_fluid[0]), std::abs(neighbour_local_offset_0_fluid[1]), std::abs(neighbour_local_offset_0_fluid[2])};
									
									
									const ivec3 neighbour_array_local_offset = ivec3(i1, j1, k1) + ivec3(static_cast<int>(INTERPOLATION_DEGREE_MAX), static_cast<int>(INTERPOLATION_DEGREE_MAX), static_cast<int>(INTERPOLATION_DEGREE_MAX));
									
									const int column = neighbour_array_local_offset[0] * ((2 * INTERPOLATION_DEGREE_MAX + 1) * (2 * INTERPOLATION_DEGREE_MAX + 1)) + neighbour_array_local_offset[1] * (2 * INTERPOLATION_DEGREE_MAX + 1) + neighbour_array_local_offset[2];
									//Only handle for neighbours of neighbour
									if(
										   (column >= column_start && column < column_end)
									){
										const int local_column = column - column_start;
										
										const float W1_0_solid = (neighbour_absolute_local_offset_solid[0] < 3 ? weight_solid_0(0, neighbour_absolute_local_offset_solid[0]) : 0.0f) * (neighbour_absolute_local_offset_solid[1] < 3 ? weight_solid_0(1, neighbour_absolute_local_offset_solid[1]) : 0.0f) * (neighbour_absolute_local_offset_solid[2] < 3 ? weight_solid_0(2, neighbour_absolute_local_offset_solid[2]) : 0.0f);
										const float W1_0_fluid = (neighbour_absolute_local_offset_fluid[0] < 3 ? weight_solid_0(0, neighbour_absolute_local_offset_fluid[0]) : 0.0f) * (neighbour_absolute_local_offset_fluid[1] < 3 ? weight_solid_0(1, neighbour_absolute_local_offset_fluid[1]) : 0.0f) * (neighbour_absolute_local_offset_fluid[2] < 3 ? weight_solid_0(2, neighbour_absolute_local_offset_fluid[2]) : 0.0f);
										
										
										float* current_coupling_solid = (coupling_solid == nullptr ? nullptr : &(coupling_solid[(3 * config::G_BLOCKSIZE * config::G_BLOCKSIZE * column_range) * i + (3 * config::G_BLOCKSIZE * column_range) * j + (3 * column_range) * k + 3 * local_column + alpha]));
										float* current_coupling_fluid = (coupling_fluid == nullptr ? nullptr : &(coupling_fluid[(3 * config::G_BLOCKSIZE * config::G_BLOCKSIZE * column_range) * i + (3 * config::G_BLOCKSIZE * column_range) * j + (3 * column_range) * k + 3 * local_column + alpha]));

										store_data_neigbours_coupling_solid(particle_buffer_solid, current_coupling_solid, W_2, W1_0_solid, contact_area, normal[alpha]);
										store_data_neigbours_coupling_fluid(particle_buffer_fluid, current_coupling_fluid, W_1, W1_0_fluid, contact_area, normal[alpha]);
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

template<typename Partition, typename Grid, MaterialE MaterialTypeSolid, MaterialE MaterialTypeFluid>
__global__ void create_iq_system(const uint32_t num_blocks, Duration dt, const ParticleBuffer<MaterialTypeSolid> particle_buffer_solid, const ParticleBuffer<MaterialTypeFluid> particle_buffer_fluid, const ParticleBuffer<MaterialTypeSolid> next_particle_buffer_solid, const ParticleBuffer<MaterialTypeFluid> next_particle_buffer_fluid, const Partition prev_partition, const Partition partition, const Grid grid_solid, const Grid grid_fluid, const SurfaceParticleBuffer surface_particle_buffer_solid, const SurfaceParticleBuffer surface_particle_buffer_fluid, const int* iq_lhs_rows, const int* iq_lhs_columns, float* iq_lhs_values, float* iq_rhs, const int* iq_solve_velocity_rows, const int* iq_solve_velocity_columns, float* iq_solve_velocity_values) {
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
	float boundary_fluid_local[(3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE];
	float velocity_fluid_local[(3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE];
	
	float coupling_solid_local[(3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE];
	float coupling_fluid_local[(3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE];

	const int src_blockno		   = static_cast<int>(blockIdx.x);
	const auto blockid			   = partition.active_keys[blockIdx.x];
	const ivec3 block_cellid = blockid * static_cast<int>(config::G_BLOCKSIZE);
	//const int particle_bucket_size_solid = next_particle_buffer_solid.particle_bucket_sizes[src_blockno];
	//const int particle_bucket_size_fluid = next_particle_buffer_fluid.particle_bucket_sizes[src_blockno];
	
	//Check if the block is outside of grid bounds
	vec3 boundary_normal;
	boundary_normal[0] = (blockid[0] < boundary_condition) ? 1.0f : ((blockid[0] >= config::G_GRID_SIZE - boundary_condition) ? -1.0f : 0.0f);
	boundary_normal[1] = (blockid[1] < boundary_condition) ? 1.0f : ((blockid[1] >= config::G_GRID_SIZE - boundary_condition) ? -1.0f : 0.0f);
	boundary_normal[2] = (blockid[2] < boundary_condition) ? 1.0f : ((blockid[2] >= config::G_GRID_SIZE - boundary_condition) ? -1.0f : 0.0f);
	
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
	}
	for(size_t i = 0; i < (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i){
		mass_solid_local[i] = 0.0f;
		mass_fluid_local[i] = 0.0f;
		if(get_global_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, i) < 3 * config::G_BLOCKVOLUME){
			if((get_global_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, i) % 3) == 0) {
				velocity_solid_local[i] = grid_block_solid.val_1d(_1, get_global_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, i) / 3);
				velocity_fluid_local[i] = grid_block_fluid.val_1d(_1, get_global_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, i) / 3);
			} else if((get_global_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, i) % 3) == 1) {
				velocity_solid_local[i] = grid_block_solid.val_1d(_2, get_global_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, i) / 3);
				velocity_fluid_local[i] = grid_block_fluid.val_1d(_2, get_global_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, i) / 3);
			} else {
				velocity_solid_local[i] = grid_block_solid.val_1d(_3, get_global_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, i) / 3);
				velocity_fluid_local[i] = grid_block_fluid.val_1d(_3, get_global_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, i) / 3);
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
		__shared__ float mass_solid[3 * config::G_BLOCKVOLUME];
		__shared__ float scaling_solid[1 * config::G_BLOCKVOLUME];
		__shared__ float pressure_solid_nominator[1 * config::G_BLOCKVOLUME];
		__shared__ float pressure_solid_denominator[1 * config::G_BLOCKVOLUME];
		
		__shared__ float mass_fluid[3 * config::G_BLOCKVOLUME];
		
		//Clear memory
		for(size_t i = 0; i < config::G_BLOCKVOLUME; ++i){
			if(get_thread_index<BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(i) == threadIdx.x){
				scaling_solid[i] = 0.0f;
				pressure_solid_nominator[i] = 0.0f;
				pressure_solid_denominator[i] = 0.0f;
			}
			for(size_t j = 0; j < 3; ++j){
				if(get_thread_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * i + j) == threadIdx.x){
					mass_solid[3 * i + j] = 0.0f;
					mass_fluid[3 * i + j] = 0.0f;
				}
			}
		}
		__syncthreads();
		
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
					
					for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < next_particle_buffer_solid.particle_bucket_sizes[current_blockno]; particle_id_in_block += static_cast<int>(blockDim.x)) {
						aggregate_data_solid(
							  particle_buffer_solid
							, next_particle_buffer_solid
							, prev_partition
							, grid_solid
							, current_blockno
							, current_blockid
							, block_cellid
							, particle_id_in_block
							, 0
							, 0 //No gradient data stored, so we can skip the inner loop
							, &(scaling_solid[0])
							, &(pressure_solid_nominator[0])
							, &(pressure_solid_denominator[0])
							, &(mass_solid[0])
							, nullptr
						);
					}
					
					for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block <  next_particle_buffer_fluid.particle_bucket_sizes[current_blockno]; particle_id_in_block += static_cast<int>(blockDim.x)) {
						aggregate_data_fluid(
							  particle_buffer_fluid
							, next_particle_buffer_fluid
							, prev_partition
							, grid_solid
							, grid_fluid
							, current_blockno
							, current_blockid
							, block_cellid
							, particle_id_in_block
							, 0
							, 0 //No gradient data stored, so we can skip the inner loop
							, &(mass_fluid[0])
							, nullptr
							, nullptr
						);
					}
				}
			}
		}
		__syncthreads();
		
		//Spread data
		for(size_t i = 0; i < config::G_BLOCKVOLUME; ++i){
			if(get_thread_index<BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(i) == threadIdx.x){
				scaling_solid_local[get_thread_offset<BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(i)] += scaling_solid[i];
				pressure_solid_nominator_local[get_thread_offset<BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(i)] += pressure_solid_nominator[i];
				pressure_solid_denominator_local[get_thread_offset<BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(i)] += pressure_solid_denominator[i];
			}
			for(size_t j = 0; j < 3; ++j){
				if(get_thread_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * i + j) == threadIdx.x){
					mass_solid_local[get_thread_offset<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * i + j)] += mass_solid[3 * i + j];
					mass_fluid_local[get_thread_offset<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * i + j)] += mass_fluid[3 * i + j];
				}
			}
		}
	}
				
	for(int column = 0; column < NUM_COLUMNS_PER_BLOCK; ++column){
		__shared__ float gradient_solid[3 * config::G_BLOCKVOLUME];
		__shared__ float gradient_fluid[3 * config::G_BLOCKVOLUME];
		__shared__ float boundary_fluid[3 * config::G_BLOCKVOLUME];
		
		//Clear memory
		for(size_t i = 0; i < config::G_BLOCKVOLUME; ++i){
			for(size_t j = 0; j < 3; ++j){
				if(get_thread_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * NUM_COLUMNS_PER_BLOCK * i + 3 * column + j) == threadIdx.x){
					gradient_solid[3 * i + j] = 0.0f;
					gradient_fluid[3 * i + j] = 0.0f;
					boundary_fluid[3 * i + j] = 0.0f;
				}
			}
		}
		__syncthreads();
		
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
					
					
					for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < next_particle_buffer_solid.particle_bucket_sizes[current_blockno]; particle_id_in_block += static_cast<int>(blockDim.x)) {
						aggregate_data_solid(
							  particle_buffer_solid
							, next_particle_buffer_solid
							, prev_partition
							, grid_solid
							, current_blockno
							, current_blockid
							, block_cellid
							, particle_id_in_block
							, column
							, column + 1
							, nullptr
							, nullptr
							, nullptr
							, nullptr
							, &(gradient_solid[0])
						);
					}
		
					for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block <  next_particle_buffer_fluid.particle_bucket_sizes[current_blockno]; particle_id_in_block += static_cast<int>(blockDim.x)) {
						aggregate_data_fluid(
							  particle_buffer_fluid
							, next_particle_buffer_fluid
							, prev_partition
							, grid_solid
							, grid_fluid
							, current_blockno
							, current_blockid
							, block_cellid
							, particle_id_in_block
							, column
							, column + 1
							, nullptr
							, &(gradient_fluid[0])
							, &(boundary_fluid[0])
						);
					}
				}
			}
		}
		__syncthreads();
		
		//Spread data
		for(size_t i = 0; i < config::G_BLOCKVOLUME; ++i){
			for(size_t j = 0; j < 3; ++j){
				if(get_thread_index<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * NUM_COLUMNS_PER_BLOCK * i + 3 * column + j) == threadIdx.x){
					gradient_solid_local[get_thread_offset<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * NUM_COLUMNS_PER_BLOCK * i + 3 * column + j)] += gradient_solid[3 * i + j];
					gradient_fluid_local[get_thread_offset<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * NUM_COLUMNS_PER_BLOCK * i + 3 * column + j)] += gradient_fluid[3 * i + j];
					boundary_fluid_local[get_thread_offset<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * NUM_COLUMNS_PER_BLOCK * i + 3 * column + j)] += boundary_fluid[3 * i + j];
				}
			}
		}
	}
	
	for(int column = 0; column < NUM_COLUMNS_PER_BLOCK; ++column){
		__shared__ float coupling_solid[3 * config::G_BLOCKVOLUME];
		__shared__ float coupling_fluid[3 * config::G_BLOCKVOLUME];
		
		//Clear memory
		for(size_t i = 0; i < config::G_BLOCKVOLUME; ++i){
			for(size_t j = 0; j < 3; ++j){
				if(get_thread_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * NUM_COLUMNS_PER_BLOCK * i + 3 * column + j) == threadIdx.x){
					coupling_solid[3 * i + j] = 0.0f;
					coupling_fluid[3 * i + j] = 0.0f;
				}
			}
		}
		__syncthreads();
		
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
					
					
					for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < next_particle_buffer_solid.particle_bucket_sizes[current_blockno]; particle_id_in_block += static_cast<int>(blockDim.x)) {
						aggregate_data_coupling(
							  particle_buffer_solid
							, particle_buffer_fluid
							, next_particle_buffer_solid
							, next_particle_buffer_fluid
							, prev_partition
							, grid_solid
							, surface_particle_buffer_solid
							, current_blockno
							, current_blockid
							, block_cellid
							, particle_id_in_block
							, column
							, column + 1
							, &(coupling_solid[0])
							, &(coupling_fluid[0])
						);
					}
				}
			}
		}
		__syncthreads();
		
		//Spread data
		for(size_t i = 0; i < config::G_BLOCKVOLUME; ++i){
			for(size_t j = 0; j < 3; ++j){
				if(get_thread_index<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * NUM_COLUMNS_PER_BLOCK * i + 3 * column + j) == threadIdx.x){
					coupling_solid_local[get_thread_offset<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * NUM_COLUMNS_PER_BLOCK * i + 3 * column + j)] += coupling_solid[3 * i + j];
					coupling_fluid_local[get_thread_offset<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * NUM_COLUMNS_PER_BLOCK * i + 3 * column + j)] += coupling_fluid[3 * i + j];
				}
			}
		}
	}
	
	for(int i = 0; i < (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i){
		if(mass_solid_local[i] > 0.0f && mass_fluid_local[i] > 0.0f){
			//printf("ABC3 %d %d # %.28f %.28f\n", static_cast<int>(threadIdx.x), i, mass_solid_local[i], mass_fluid_local[i]);
		}
	}
	
	//Column that represents (row, row)
	constexpr size_t IDENTIITY_NEIGHBOUR_INDEX = (INTERPOLATION_DEGREE_MAX * ((2 * INTERPOLATION_DEGREE_MAX + 1) * (2 * INTERPOLATION_DEGREE_MAX + 1)) + INTERPOLATION_DEGREE_MAX * (2 * INTERPOLATION_DEGREE_MAX + 1) + INTERPOLATION_DEGREE_MAX);
	
	//Store data in matrix
	for(int row = 0; row < NUM_ROWS_PER_BLOCK; ++row) {
		const ivec3 local_id {static_cast<int>((row / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE), static_cast<int>((row / config::G_BLOCKSIZE) % config::G_BLOCKSIZE), static_cast<int>(row % config::G_BLOCKSIZE)};
		
		//Generate cell mapping
		//Local cells are indexed by offset, global cells are sorted by cellno
		int lhs_column_indices[NUM_COLUMNS_PER_BLOCK];
		int solve_velocity_column_indices[NUM_COLUMNS_PER_BLOCK];
		for(size_t column = 0; column < NUM_COLUMNS_PER_BLOCK; ++column){
			const ivec3 row_cellid = block_cellid + local_id;
			const ivec3 neighbour_local_id = ivec3(static_cast<int>((column / ((2 * INTERPOLATION_DEGREE_MAX + 1) * (2 * INTERPOLATION_DEGREE_MAX + 1))) % (2 * INTERPOLATION_DEGREE_MAX + 1)), static_cast<int>((column / (2 * INTERPOLATION_DEGREE_MAX + 1)) % (2 * INTERPOLATION_DEGREE_MAX + 1)), static_cast<int>(column % (2 * INTERPOLATION_DEGREE_MAX + 1))) - ivec3(static_cast<int>(INTERPOLATION_DEGREE_MAX), static_cast<int>(INTERPOLATION_DEGREE_MAX), static_cast<int>(INTERPOLATION_DEGREE_MAX));
			const ivec3 neighbour_cellid = row_cellid + neighbour_local_id;
			const ivec3 neighbour_blockid = neighbour_cellid / static_cast<int>(config::G_BLOCKSIZE);
			
			const ivec3 neighbour_base_cellid = neighbour_blockid * static_cast<int>(config::G_BLOCKSIZE);
			const ivec3 neighbour_celloffset = neighbour_cellid - neighbour_base_cellid;
			
			const int neighbour_blockno = partition.query(neighbour_blockid);
			const int neighbour_cellno = NUM_ROWS_PER_BLOCK * neighbour_blockno + (config::G_BLOCKSIZE * config::G_BLOCKSIZE) * neighbour_celloffset[0] + config::G_BLOCKSIZE * neighbour_celloffset[1] + neighbour_celloffset[2];
			
			solve_velocity_column_indices[column] = -1;
			for(size_t lhs_column = 0; lhs_column < NUM_COLUMNS_PER_BLOCK; ++lhs_column){
				if(neighbour_cellno == iq_lhs_columns[iq_lhs_rows[base_row + row] + lhs_column]){
					lhs_column_indices[column] = lhs_column;
				}
				if(neighbour_cellno == iq_solve_velocity_columns[iq_solve_velocity_rows[3 * base_row + 3 * row] + lhs_column]){
					solve_velocity_column_indices[column] = lhs_column;
				}
			}
			
			if(threadIdx.x == 0 && solve_velocity_column_indices[column] == -1){
				const ivec3 row_cellid = block_cellid + local_id;
				
				printf("ERROR0 %d %d # %d %d # %d %d %d # %d %d %d # %d %d\n", static_cast<int>(3 * base_row + 3 * row), neighbour_cellno, iq_solve_velocity_rows[3 * base_row + 3 * row], iq_solve_velocity_rows[3 * base_row + 3 * row + 1], row_cellid[0], row_cellid[1], row_cellid[2], neighbour_cellid[0], neighbour_cellid[1], neighbour_cellid[2], static_cast<int>(blockIdx.x), neighbour_blockno);
				for(size_t lhs_column = 0; lhs_column < NUM_COLUMNS_PER_BLOCK; ++lhs_column){
					printf("ERROR1 %d %d # %d # %d\n", static_cast<int>(3 * base_row + 3 * row), neighbour_cellno, iq_solve_velocity_columns[iq_solve_velocity_rows[3 * base_row + 3 * row] + lhs_column], iq_solve_velocity_columns[iq_solve_velocity_rows[3 * base_row + 3 * row + 1] + lhs_column]);
				}
			}
		}
		
		__shared__ float current_mass_solid[3];
		__shared__ float current_scaling_solid;
		__shared__ float current_gradient_solid_row[3];
		__shared__ float current_velocity_solid[3];
		__shared__ float current_pressure_solid_nominator;
		__shared__ float current_pressure_solid_denominator;
		
		__shared__ float current_mass_fluid[3];
		__shared__ float current_gradient_fluid_row[3];
		__shared__ float current_velocity_fluid[3];
		__shared__ float current_boundary_fluid_row[3];
		
		__shared__ float current_coupling_solid_row[3];
		__shared__ float current_coupling_fluid_row[3];
		
		if(get_thread_index<BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(row) == threadIdx.x){
			current_scaling_solid = scaling_solid_local[get_thread_offset<BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(row)];
			current_pressure_solid_nominator = pressure_solid_nominator_local[get_thread_offset<BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(row)];
			current_pressure_solid_denominator = pressure_solid_denominator_local[get_thread_offset<BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(row)];
		}
		
		for(int i = 0; i < 3; ++i){
			if(get_thread_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * row + i) == threadIdx.x){
				current_mass_solid[i] = mass_solid_local[get_thread_offset<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * row + i)];
				current_mass_fluid[i] = mass_fluid_local[get_thread_offset<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * row + i)];
				current_velocity_solid[i] = velocity_solid_local[get_thread_offset<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * row + i)];
				current_velocity_fluid[i] = velocity_fluid_local[get_thread_offset<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * row + i)];
			}
			if(get_thread_index<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * NUM_COLUMNS_PER_BLOCK * row + 3 * IDENTIITY_NEIGHBOUR_INDEX + i) == threadIdx.x){
				current_gradient_solid_row[i] = gradient_solid_local[get_thread_offset<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * NUM_COLUMNS_PER_BLOCK * row + 3 * IDENTIITY_NEIGHBOUR_INDEX + i)];
				current_gradient_fluid_row[i] = gradient_fluid_local[get_thread_offset<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * NUM_COLUMNS_PER_BLOCK * row + 3 * IDENTIITY_NEIGHBOUR_INDEX + i)];
				current_boundary_fluid_row[i] = boundary_fluid_local[get_thread_offset<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * NUM_COLUMNS_PER_BLOCK * row + 3 * IDENTIITY_NEIGHBOUR_INDEX + i)];
				current_coupling_solid_row[i] = coupling_solid_local[get_thread_offset<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * NUM_COLUMNS_PER_BLOCK * row + 3 * IDENTIITY_NEIGHBOUR_INDEX + i)];
				current_coupling_fluid_row[i] = coupling_fluid_local[get_thread_offset<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * NUM_COLUMNS_PER_BLOCK * row + 3 * IDENTIITY_NEIGHBOUR_INDEX + i)];
			}
		}
		
		for(size_t column = 0; column < NUM_COLUMNS_PER_BLOCK; ++column){
			__shared__ float current_gradient_solid_column[3];
			
			__shared__ float current_gradient_fluid_column[3];
			__shared__ float current_boundary_fluid_column[3];
			
			__shared__ float current_coupling_solid_column[3];
			__shared__ float current_coupling_fluid_column[3];
			
			for(int i = 0; i < 3; ++i){
				if(get_thread_index<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * NUM_COLUMNS_PER_BLOCK * row + 3 * column + i) == threadIdx.x){
					current_gradient_solid_column[i] = gradient_solid_local[get_thread_offset<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * NUM_COLUMNS_PER_BLOCK * row + 3 * column + i)];
					current_gradient_fluid_column[i] = gradient_fluid_local[get_thread_offset<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * NUM_COLUMNS_PER_BLOCK * row + 3 * column + i)];
					current_boundary_fluid_column[i] = boundary_fluid_local[get_thread_offset<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * NUM_COLUMNS_PER_BLOCK * row + 3 * column + i)];
					current_coupling_solid_column[i] = coupling_solid_local[get_thread_offset<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * NUM_COLUMNS_PER_BLOCK * row + 3 * column + i)];
					current_coupling_fluid_column[i] = coupling_fluid_local[get_thread_offset<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * NUM_COLUMNS_PER_BLOCK * row + 3 * column + i)];
				}
			}
			
			__syncthreads();
			
			//P = O^T * M^-1 * O => P_row_col = sum(all_rows_in_column; current_o_row * current_o_col / current_m^2) => add current_o_row * current_o_col / current_m^2 to entries for current row and for neighbour row (row and col); Other entries are zero
			if(get_thread_index<BLOCK_SIZE, (NUM_ROWS_PER_BLOCK * NUM_COLUMNS_PER_BLOCK + BLOCK_SIZE - 1) / BLOCK_SIZE>(row * NUM_COLUMNS_PER_BLOCK + column) == threadIdx.x){
				const vec3 square_mass_solid{current_mass_solid[0] * current_mass_solid[0], current_mass_solid[1] * current_mass_solid[1], current_mass_solid[2] * current_mass_solid[2]};
				const vec3 square_mass_fluid{current_mass_fluid[0] * current_mass_fluid[0], current_mass_fluid[1] * current_mass_fluid[1], current_mass_fluid[2] * current_mass_fluid[2]};
				
				//Caling only for diagonal element
				const float scaling_solid = (column == IDENTIITY_NEIGHBOUR_INDEX ? current_scaling_solid : 0.0f);
				
				const float gradient_by_mass_solid = (current_gradient_solid_row[0] * current_gradient_solid_column[0] / square_mass_solid[0] + current_gradient_solid_row[1] * current_gradient_solid_column[1] / square_mass_solid[1] + current_gradient_solid_row[2] * current_gradient_solid_column[2] / square_mass_solid[2]);
				const float gradient_by_mass_fluid = (current_gradient_fluid_row[0] * current_gradient_fluid_column[0] / square_mass_fluid[0] + current_gradient_fluid_row[1] * current_gradient_fluid_column[1] / square_mass_fluid[1] + current_gradient_fluid_row[2] * current_gradient_fluid_column[2] / square_mass_fluid[2]);
				
				const float boundary_by_mass = (current_boundary_fluid_row[0] * current_boundary_fluid_column[0] / square_mass_fluid[0] + current_boundary_fluid_row[1] * current_boundary_fluid_column[1] / square_mass_fluid[1] + current_boundary_fluid_row[2] * current_boundary_fluid_column[2] / square_mass_fluid[2]);
				const float gradient_and_boundary_by_mass = (current_gradient_fluid_row[0] * current_boundary_fluid_column[0] / square_mass_fluid[0] + current_gradient_fluid_row[1] * current_boundary_fluid_column[1] / square_mass_fluid[1] + current_gradient_fluid_row[2] * current_boundary_fluid_column[2] / square_mass_fluid[2]);
				
				const float gradient_and_coupling_by_mass_solid = (current_gradient_solid_row[0] * current_coupling_solid_column[0] / square_mass_solid[0] + current_gradient_solid_row[1] * current_coupling_solid_column[1] / square_mass_solid[1] + current_gradient_solid_row[2] * current_coupling_solid_column[2] / square_mass_solid[2]);
				const float gradient_and_coupling_by_mass_fluid = (current_gradient_fluid_row[0] * current_coupling_fluid_column[0] / square_mass_fluid[0] + current_gradient_fluid_row[1] * current_coupling_fluid_column[1] / square_mass_fluid[1] + current_gradient_fluid_row[2] * current_coupling_fluid_column[2] / square_mass_fluid[2]);
				const float boundary_and_coupling_by_mass_fluid = (current_boundary_fluid_row[0] * current_coupling_fluid_column[0] / square_mass_fluid[0] + current_boundary_fluid_row[1] * current_coupling_fluid_column[1] / square_mass_fluid[1] + current_boundary_fluid_row[2] * current_coupling_fluid_column[2] / square_mass_fluid[2]);
				
				const float coupling_by_mass_solid = (current_coupling_solid_row[0] * current_coupling_solid_column[0] / square_mass_solid[0] + current_coupling_solid_row[1] * current_coupling_solid_column[1] / square_mass_solid[1] + current_coupling_solid_row[2] * current_coupling_solid_column[2] / square_mass_solid[2]);
				const float coupling_by_mass_fluid = (current_coupling_fluid_row[0] * current_coupling_fluid_column[0] / square_mass_fluid[0] + current_coupling_fluid_row[1] * current_coupling_fluid_column[1] / square_mass_fluid[1] + current_coupling_fluid_row[2] * current_coupling_fluid_column[2] / square_mass_fluid[2]);
				
				std::array<std::array<float, LHS_MATRIX_SIZE_X>, LHS_MATRIX_SIZE_Y> a;
				
				std::array<std::array<float, LHS_MATRIX_SIZE_X>, LHS_MATRIX_SIZE_Y> a_transposed;
				
				//Clear empty values
				a[0][1] = 0.0f;
				a[0][2] = 0.0f;
				a[1][0] = 0.0f;
				a[2][0] = 0.0f;
				
				a_transposed[0][1] = 0.0f;
				a_transposed[0][2] = 0.0f;
				a_transposed[1][0] = 0.0f;
				a_transposed[2][0] = 0.0f;
				
				//Clear other values
				a[3][3] = 0.0f;
				
				a_transposed[3][3] = 0.0f;
				
				//Only calculate for particles with mass bigger than 0 (otherwise we will divide by 0)
				if(
					   square_mass_solid[0] > 0.0f
					&& square_mass_solid[1] > 0.0f
					&& square_mass_solid[2] > 0.0f
				){
					a[0][0] = scaling_solid / dt.count() + dt.count() * gradient_by_mass_solid;
					a[0][3] = -dt.count() * gradient_and_coupling_by_mass_solid;
					a[3][3] += dt.count() * coupling_by_mass_solid;
					
					//Avoid adding twice to diagonale
					if(column != IDENTIITY_NEIGHBOUR_INDEX){
						a_transposed[0][0] = dt.count() * gradient_by_mass_solid;
						a_transposed[0][3] = -dt.count() * gradient_and_coupling_by_mass_solid;
						a_transposed[3][3] += dt.count() * coupling_by_mass_solid;
					}else{
						a_transposed[0][0] = 0.0f;
						a_transposed[0][3] = 0.0f;
					}
				}else{
					a[0][0] = scaling_solid / dt.count();
					a[0][3] = 0.0f;
					
					a_transposed[0][0] = 0.0f;
					a_transposed[0][3] = 0.0f;
				}
				
				//FIXME:
				/*if(blockIdx.x == 0 && row == 0 && column == 0){
					a[0][0] = 1.0f;
				}else{
					a[0][0] = 0.0f;
				}*/
				
				if(
					   square_mass_fluid[0] > 0.0f
					&& square_mass_fluid[1] > 0.0f
					&& square_mass_fluid[2] > 0.0f
				){
					a[1][1] = dt.count() * gradient_by_mass_fluid;
					a[1][2] = dt.count() * gradient_and_boundary_by_mass;
					a[1][3] = dt.count() * gradient_and_coupling_by_mass_fluid;
					a[2][2] = dt.count() * boundary_by_mass;
					a[2][3] = dt.count() * boundary_and_coupling_by_mass_fluid;
					a[3][3] += dt.count() * coupling_by_mass_fluid;
					
					//Avoid adding twice to diagonale
					if(column != IDENTIITY_NEIGHBOUR_INDEX){
						a_transposed[1][1] = dt.count() * gradient_by_mass_fluid;
						a_transposed[1][2] = dt.count() * gradient_and_boundary_by_mass;
						a_transposed[1][3] = dt.count() * gradient_and_coupling_by_mass_fluid;
						a_transposed[2][2] = dt.count() * boundary_by_mass;
						a_transposed[2][3] = dt.count() * boundary_and_coupling_by_mass_fluid;
						a_transposed[3][3] += dt.count() * coupling_by_mass_fluid;
					}else{
						a_transposed[1][1] = 0.0f;
						a_transposed[1][2] = 0.0f;
						a_transposed[1][3] = 0.0f;
						a_transposed[2][2] = 0.0f;
						a_transposed[2][3] = 0.0f;
					}
				}else{
					a[1][1] = 0.0f;
					a[1][2] = 0.0f;
					a[1][3] = 0.0f;
					a[2][2] = 0.0f;
					a[2][3] = 0.0f;
					
					a_transposed[1][1] = 0.0f;
					a_transposed[1][2] = 0.0f;
					a_transposed[1][3] = 0.0f;
					a_transposed[2][2] = 0.0f;
					a_transposed[2][3] = 0.0f;
				}
				
				//Fill symmetric
				a[3][0] = a[0][3];
				a[2][1] = a[1][2];
				a[3][1] = a[1][3];
				a[3][2] = a[2][3];
				
				a_transposed[3][0] = a_transposed[0][3];
				a_transposed[2][1] = a_transposed[1][2];
				a_transposed[3][1] = a_transposed[1][3];
				a_transposed[3][2] = a_transposed[2][3];
				
				std::array<std::array<std::array<float, SOLVE_VELOCITY_MATRIX_SIZE_X>, SOLVE_VELOCITY_MATRIX_SIZE_Y>, 3> solve_velocity;

				for(int k = 0; k < 3; ++k){
					//Clear empty values
					solve_velocity[k][0][1] = 0.0f;
					solve_velocity[k][0][2] = 0.0f;
					solve_velocity[k][1][0] = 0.0f;
					
					//Only calculate for particles with mass bigger than 0 (otherwise we will divide by 0)
					if(
						   current_mass_solid[k] > 0.0f
					){
						solve_velocity[k][0][0] = -dt.count() * current_gradient_solid_column[k] / current_mass_solid[k];
						solve_velocity[k][0][3] = dt.count() * current_coupling_solid_column[k] / current_mass_solid[k];
					}else{
						solve_velocity[k][0][0] = 0.0f;
						solve_velocity[k][0][3] = 0.0f;
					}
					
					if(
						   current_mass_fluid[k] > 0.0f
					){
						solve_velocity[k][1][1] = -dt.count() * current_gradient_fluid_column[k] / current_mass_fluid[k];
						solve_velocity[k][1][2] = -dt.count() * current_boundary_fluid_row[k] / current_mass_fluid[k];
						solve_velocity[k][1][3] = -dt.count() * current_coupling_fluid_column[k] / current_mass_fluid[k];
					}else{
						solve_velocity[k][1][1] = 0.0f;
						solve_velocity[k][1][2] = 0.0f;
						solve_velocity[k][1][3] = 0.0f;
					}
				}
				
				//Store at index (blockid + row, blockid + column), adding it to existing value
				for(size_t i = 0; i < LHS_MATRIX_SIZE_Y; ++i){
					const int row_index = i * NUM_ROWS_PER_BLOCK * num_blocks + base_row + row;
					for(size_t j = 0; j < lhs_num_blocks_per_row[i]; ++j){
						
						const int column_index = iq_lhs_rows[row_index] + j * NUM_COLUMNS_PER_BLOCK + lhs_column_indices[column];
						
						if(a[i][lhs_block_offsets_per_row[i][j]] != 0.0f){
							//printf("IQ_LHS %d %d # %d %d %d # %.28f # %.28f # %.28f %.28f %.28f # %.28f %.28f %.28f\n", row_index, column_index, static_cast<int>(i), static_cast<int>(j), static_cast<int>(lhs_block_offsets_per_row[i][j]), a[i][lhs_block_offsets_per_row[i][j]], gradient_by_mass_solid, current_gradient_solid_row[0], current_gradient_solid_row[1], current_gradient_solid_row[2], current_gradient_solid_column[0], current_gradient_solid_column[1], current_gradient_solid_column[2]);
							//printf("IQ_LHS %d %d # %d %d # %d # %.28f # %.28f\n", row_index, iq_lhs_columns[column_index], static_cast<int>(i), static_cast<int>(lhs_block_offsets_per_row[i][j]), static_cast<int>(column), a[i][lhs_block_offsets_per_row[i][j]], a_transposed[i][lhs_block_offsets_per_row[i][j]]);
						}
						
						atomicAdd(&(iq_lhs_values[column_index]), a[i][lhs_block_offsets_per_row[i][j]]);
					}
				}
				
				//Skip columns that are empty. These might be neighbours of neighbours which should always have zero values stored.
				{
					const int column_cellno = iq_lhs_columns[iq_lhs_rows[base_row + row] + lhs_column_indices[column]];
					if(column_cellno < NUM_ROWS_PER_BLOCK * num_blocks && (iq_lhs_rows[column_cellno + 1] - iq_lhs_rows[column_cellno]) > 0){
						//Generate cell mapping
						//Local cells are indexed by offset, global cells are sorted by cellno
						int row_column_number = -1;
						for(size_t lhs_column = 0; lhs_column < NUM_COLUMNS_PER_BLOCK; ++lhs_column){
							if((base_row + row) == iq_lhs_columns[iq_lhs_rows[column_cellno] + lhs_column]){
								row_column_number = lhs_column;
								break;
							}
						}
						
						/*if(row_column_number == -1){
							const ivec3 row_cellid = block_cellid + local_id;
							
							const ivec3 column_local_id {static_cast<int>((column_cellno / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE), static_cast<int>((column_cellno / config::G_BLOCKSIZE) % config::G_BLOCKSIZE), static_cast<int>(column_cellno % config::G_BLOCKSIZE)};
							const int column_blockno = column_cellno / static_cast<int>(config::G_BLOCKVOLUME);
							const ivec3 column_blockid = partition.active_keys[column_blockno];
							const ivec3 column_cellid = column_blockid * static_cast<int>(config::G_BLOCKSIZE) + column_local_id;
							
							printf("ERROR1 %d %d # %d %d %d # %d %d %d # %d %d %d\n", static_cast<int>(base_row + row), column_cellno, iq_lhs_rows[column_cellno], iq_lhs_rows[column_cellno + 1], lhs_column_indices[column], row_cellid[0], row_cellid[1], row_cellid[2], column_cellid[0], column_cellid[1], column_cellid[2]);
							for(size_t lhs_column = 0; lhs_column < NUM_COLUMNS_PER_BLOCK; ++lhs_column){
								printf("ERROR1 %d %d # %d\n", static_cast<int>(base_row + row), column_cellno, iq_lhs_columns[iq_lhs_rows[column_cellno] + lhs_column]);
							}
							continue;
						}*/
						
						//Store at index (blockid + column, blockid + row), adding it to existing value
						for(size_t i = 0; i < LHS_MATRIX_SIZE_Y; ++i){
							const int row_index = i * NUM_ROWS_PER_BLOCK * num_blocks + column_cellno;
											
							for(size_t j = 0; j < lhs_num_blocks_per_row[i]; ++j){
								
								const int column_index = iq_lhs_rows[row_index] + j * NUM_COLUMNS_PER_BLOCK + row_column_number;
								
								if(a_transposed[i][lhs_block_offsets_per_row[i][j]] != 0.0f){
									//printf("IQ_LHS %d %d # %d %d %d # %.28f # %.28f # %.28f %.28f %.28f # %.28f %.28f %.28f\n", row_index, column_index, static_cast<int>(i), static_cast<int>(j), static_cast<int>(lhs_block_offsets_per_row[i][j]), a_transposed[i][lhs_block_offsets_per_row[i][j]], gradient_by_mass_solid, current_gradient_solid_row[0], current_gradient_solid_row[1], current_gradient_solid_row[2], current_gradient_solid_column[0], current_gradient_solid_column[1], current_gradient_solid_column[2]);
									//printf("T_IQ_LHS %d %d # %d %d # %d # %.28f # %.28f\n", row_index, iq_lhs_columns[column_index], static_cast<int>(i), static_cast<int>(lhs_block_offsets_per_row[i][j]), static_cast<int>(column), a[i][lhs_block_offsets_per_row[i][j]], a_transposed[i][lhs_block_offsets_per_row[i][j]]);
								}
								
								atomicAdd(&(iq_lhs_values[column_index]), a_transposed[i][lhs_block_offsets_per_row[i][j]]);
							}
						}
					}else{
						//printf("Skipping column: %d\n", column_cellno);
						
						for(size_t i = 0; i < LHS_MATRIX_SIZE_Y; ++i){
							for(size_t j = 0; j < lhs_num_blocks_per_row[i]; ++j){
								if(a_transposed[i][lhs_block_offsets_per_row[i][j]] != 0.0f){
									printf("ERROR - Skipped non_empty column %d %d\n", static_cast<int>(base_row + row), column_cellno);
								}
							}
						}
					}
				}
				
				//Store at index (blockid + row, blockid + column), adding it to existing value
				for(int k = 0; k < 3; ++k){
					for(size_t i = 0; i < SOLVE_VELOCITY_MATRIX_SIZE_Y; ++i){
						const int row_index = i * 3 * NUM_ROWS_PER_BLOCK * num_blocks + 3 * base_row + 3 * row + k;
						for(size_t j = 0; j < solve_velocity_num_blocks_per_row[i]; ++j){
							
							const int column_index = iq_solve_velocity_rows[row_index] + j * NUM_COLUMNS_PER_BLOCK + solve_velocity_column_indices[column];
							
							if(solve_velocity[k][i][solve_velocity_block_offsets_per_row[i][j]] != 0.0f){
								//printf("Solve_LHS %d %d # %d %d %d # %d # %.28f\n", row_index, column_index, static_cast<int>(i), static_cast<int>(j), static_cast<int>(solve_velocity_block_offsets_per_row[i][j]), k, solve_velocity[k][i][solve_velocity_block_offsets_per_row[i][j]]);
							}
							
							atomicAdd(&(iq_solve_velocity_values[column_index]), solve_velocity[k][i][solve_velocity_block_offsets_per_row[i][j]]);
						}
					}
				}
			}
		}
		
		if(get_thread_index<BLOCK_SIZE, (NUM_ROWS_PER_BLOCK + BLOCK_SIZE - 1) / BLOCK_SIZE>(row) == threadIdx.x){
			const float gradient_and_velocity_solid = (current_gradient_solid_row[0] * current_velocity_solid[0] + current_gradient_solid_row[1] * current_velocity_solid[1] + current_gradient_solid_row[2] * current_velocity_solid[2]);
			const float gradient_and_velocity_fluid = (current_gradient_fluid_row[0] * current_velocity_fluid[0] + current_gradient_fluid_row[1] * current_velocity_fluid[1] + current_gradient_fluid_row[2] * current_velocity_fluid[2]);
			const float boundary_and_velocity_fluid = (current_boundary_fluid_row[0] * current_velocity_fluid[0] + current_boundary_fluid_row[1] * current_velocity_fluid[1] + current_boundary_fluid_row[2] * current_velocity_fluid[2]);
			const float coupling_and_velocity_solid = (current_coupling_solid_row[0] * current_velocity_solid[0] + current_coupling_solid_row[1] * current_velocity_solid[1] + current_coupling_solid_row[2] * current_velocity_solid[2]);
			const float coupling_and_velocity_fluid = (current_coupling_fluid_row[0] * current_velocity_fluid[0] + current_coupling_fluid_row[1] * current_velocity_fluid[1] + current_coupling_fluid_row[2] * current_velocity_fluid[2]);
			
			std::array<float, LHS_MATRIX_SIZE_Y> b;
			//Only calculate for particles with pressure_denominator bigger than 0 (otherwise we will divide by 0)
			if(current_pressure_solid_denominator > 0.0f){
				b[0] = (current_scaling_solid * current_pressure_solid_nominator) / (current_pressure_solid_denominator * dt.count()) - gradient_and_velocity_solid;
			}else{
				b[0] = -gradient_and_velocity_solid;
			}
			//FIXME:
			/*if(blockIdx.x == 0 && row == 0){
				b[0] = 1.0f;
			}else{
				b[0] = 0.0f;
			}*/
			b[1] = gradient_and_velocity_fluid;
			b[2] = boundary_and_velocity_fluid - 0.0f;//FIXME: Correct external force from air interface or something with surface tension
			b[3] = coupling_and_velocity_solid - coupling_and_velocity_fluid;
			
			
			for(size_t i = 0; i < LHS_MATRIX_SIZE_Y; ++i){
				const int row_index = i * NUM_ROWS_PER_BLOCK * num_blocks + base_row + row;
				
				if(b[i] != 0.0f){
					//printf("IQ_RHS %d # %d # %.28f # %.28f %.28f %.28f %.28f # %.28f %.28f %.28f # %.28f %.28f %.28f # %.28f %.28f %.28f # %.28f %.28f %.28f\n", row_index, static_cast<int>(i), b[i], current_scaling_solid, current_pressure_solid_nominator, current_pressure_solid_denominator, gradient_and_velocity_solid, current_gradient_solid_row[0], current_gradient_solid_row[1], current_gradient_solid_row[2], current_velocity_solid[0], current_velocity_solid[1], current_velocity_solid[2], current_gradient_fluid_row[0], current_gradient_fluid_row[1], current_gradient_fluid_row[2], current_velocity_fluid[0], current_velocity_fluid[1], current_velocity_fluid[2]);
				}
				
				if(isnan(b[i])){
					printf("ABC1 %.28f %.28f %.28f\n", current_velocity_solid[0], current_velocity_solid[1], current_velocity_solid[2]);
				}
				
				atomicAdd(&(iq_rhs[row_index]), b[i]);
			}
		}
	}
}
template<typename Partition, typename Grid, MaterialE MaterialTypeSolid>
__global__ void update_velocity_and_strain(const ParticleBuffer<MaterialTypeSolid> particle_buffer_solid, ParticleBuffer<MaterialTypeSolid> next_particle_buffer_soild, const Partition prev_partition, Partition partition, Grid grid_solid, Grid grid_fluid, const float* delta_v_solid, const float* delta_v_fluid, const float* pressure_solid) {
	const int src_blockno		   = static_cast<int>(blockIdx.x);
	const auto blockid			   = partition.active_keys[blockIdx.x];
	const ivec3 block_cellid = blockid * static_cast<int>(config::G_BLOCKSIZE);
	
	//Update velocity
	auto grid_block_solid = grid_solid.ch(_0, src_blockno);
	auto grid_block_fluid = grid_fluid.ch(_0, src_blockno);
	for(int cell_id_in_block = threadIdx.x; cell_id_in_block < config::G_BLOCKVOLUME; cell_id_in_block += blockDim.x) {
		grid_block_solid.val_1d(_1, cell_id_in_block) += delta_v_solid[3 * config::G_BLOCKVOLUME * src_blockno + 3 * cell_id_in_block];
		grid_block_solid.val_1d(_2, cell_id_in_block) += delta_v_solid[3 * config::G_BLOCKVOLUME * src_blockno + 3 * cell_id_in_block + 1];
		grid_block_solid.val_1d(_3, cell_id_in_block) += delta_v_solid[3 * config::G_BLOCKVOLUME * src_blockno + 3 * cell_id_in_block + 2];
		
		//printf("ABC2 %.28f %.28f %.28f\n", grid_block_solid.val_1d(_1, cell_id_in_block), grid_block_solid.val_1d(_2, cell_id_in_block), grid_block_solid.val_1d(_3, cell_id_in_block));
		
		grid_block_fluid.val_1d(_1, cell_id_in_block) += delta_v_fluid[3 * config::G_BLOCKVOLUME * src_blockno + 3 * cell_id_in_block];
		grid_block_fluid.val_1d(_2, cell_id_in_block) += delta_v_fluid[3 * config::G_BLOCKVOLUME * src_blockno + 3 * cell_id_in_block + 1];
		grid_block_fluid.val_1d(_3, cell_id_in_block) += delta_v_fluid[3 * config::G_BLOCKVOLUME * src_blockno + 3 * cell_id_in_block + 2];
	}
	
#if (FIXED_COROTATED_GHOST_ENABLE_STRAIN_UPDATE == 0)
	
	constexpr size_t CELL_COUNT = KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE;
	
	const int particle_bucket_size_solid = next_particle_buffer_soild.particle_bucket_sizes[src_blockno];
	
	__shared__ float pressure_solid_shared[KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE];
	
	//If we have no particles in the bucket return
	if(particle_bucket_size_solid == 0) {
		return;
	}
	
	//Load data from grid to shared memory
	/*for(int base = static_cast<int>(threadIdx.x); base < config::G_BLOCKVOLUME; base += static_cast<int>(blockDim.x)) {
		const ivec3 local_id {(static_cast<int>(base) / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE, (static_cast<int>(base) / config::G_BLOCKSIZE) % config::G_BLOCKSIZE, static_cast<int>(base) % config::G_BLOCKSIZE};

		const float val = pressure_solid[config::G_BLOCKVOLUME * src_blockno + base];
		pressure_solid_shared[local_id[0]][local_id[1]][local_id[2]] = val;
	}*/
	for(int base = static_cast<int>(threadIdx.x); base < CELL_COUNT; base += static_cast<int>(blockDim.x)) {
		const ivec3 absolute_local_cellid = ivec3(static_cast<int>((base / (KERNEL_SIZE * KERNEL_SIZE)) % KERNEL_SIZE), static_cast<int>((base / KERNEL_SIZE) % KERNEL_SIZE), static_cast<int>(base % KERNEL_SIZE));
		const ivec3 local_cellid = absolute_local_cellid - ivec3(static_cast<int>(KERNEL_OFFSET), static_cast<int>(KERNEL_OFFSET), static_cast<int>(KERNEL_OFFSET));
		const ivec3 local_blockid = local_cellid / config::G_BLOCKSIZE;
		const auto blockno = partition.query(blockid + local_blockid);
	
		const ivec3 cellid_in_block = local_cellid - local_blockid * config::G_BLOCKSIZE;
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
			const int advect = next_particle_buffer_soild.blockbuckets[src_blockno * config::G_PARTICLE_NUM_PER_BLOCK + particle_id_in_block];

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
		const ivec3 global_base_index_solid_0 = get_cell_id<0>(pos.data_arr(), grid_solid.get_offset());
		const ivec3 global_base_index_solid_2 = get_cell_id<2>(pos.data_arr(), grid_solid.get_offset());

		//Get position relative to grid cell
		const vec3 local_pos_solid_0 = pos - (global_base_index_solid_0 + vec3(grid_solid.get_offset()[0], grid_solid.get_offset()[1], grid_solid.get_offset()[2])) * config::G_DX;

		//Calculate weights
		vec3x3 weight_solid_0;
		
		#pragma unroll 3
		for(int dd = 0; dd < 3; ++dd) {
			const std::array<float, INTERPOLATION_DEGREE_SOLID_PRESSURE + 1> current_weight_solid_0 = bspline_weight<float, INTERPOLATION_DEGREE_SOLID_PRESSURE>(local_pos_solid_0[dd]);
			for(int i = 0; i < INTERPOLATION_DEGREE_SOLID_PRESSURE + 1; ++i){
				weight_solid_0(dd, i)		  = current_weight_solid_0[i];
			}
			for(int i = INTERPOLATION_DEGREE_SOLID_PRESSURE + 1; i < 3; ++i){
				weight_solid_0(dd, i)		  = 0.0f;
			}
		}
		
		float weighted_pressure = 0.0f;
		//Load data
		//Note: Weights are 0 if outside of interpolation degree/radius around particles cell
		//Foreach node in the block we add values accoring to particle kernel, also handling all neighbours of the particles cell
		for(char i = -static_cast<char>(INTERPOLATION_DEGREE_MAX); i < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; i++) {
			for(char j = -static_cast<char>(INTERPOLATION_DEGREE_MAX); j < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; j++) {
				for(char k = -static_cast<char>(INTERPOLATION_DEGREE_MAX); k < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; k++) {
					const ivec3 local_id = (global_base_index_solid_0 - block_cellid) + ivec3(i, j, k);
					const ivec3 absolute_local_id = local_id + ivec3(static_cast<int>(KERNEL_OFFSET), static_cast<int>(KERNEL_OFFSET), static_cast<int>(KERNEL_OFFSET));
					
					if(
						   (absolute_local_id[0] < 0 || absolute_local_id[0] >= KERNEL_SIZE)
						|| (absolute_local_id[1] < 0 || absolute_local_id[1] >= KERNEL_SIZE)
						|| (absolute_local_id[2] < 0 || absolute_local_id[2] >= KERNEL_SIZE)
					){
						//printf("ERROR4 %d %d %d # %d %d %d\n", local_id[0], local_id[1], local_id[2], absolute_local_id[0], absolute_local_id[1], absolute_local_id[2]);
					}
					
					//Weight
					const float W_0 = weight_solid_0(0, std::abs(i)) * weight_solid_0(1, std::abs(j)) * weight_solid_0(2, std::abs(k));
					
					weighted_pressure += pressure_solid_shared[absolute_local_id[0]][absolute_local_id[1]][absolute_local_id[2]] * W_0;
				}
			}
		}
		
		update_strain<MaterialTypeSolid>(particle_buffer_solid, advection_source_blockno, source_pidib, weighted_pressure);
	}
	
#endif
}

}// namespace iq

//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers, readability-identifier-naming, misc-definitions-in-headers)
}// namespace mn

#endif