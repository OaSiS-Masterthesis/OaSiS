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
constexpr size_t MAX_SHARED_PARTICLE_FLUID = config::G_BLOCKVOLUME * config::G_MAX_PARTICLES_IN_CELL >> 3;

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
__forceinline__ __device__ void store_data_solid(const ParticleBuffer<MaterialTypeSolid> particle_buffer_solid, float* __restrict__ scaling_solid, float* __restrict__ pressure_solid_nominator, float* __restrict__ pressure_solid_denominator, const float W_pressure, const float W_velocity, const float mass, const float J);

template<MaterialE MaterialTypeSolid>
__forceinline__ __device__ void store_data_neigbours_solid(const ParticleBuffer<MaterialTypeSolid> particle_buffer_solid, float* __restrict__ gradient_solid, const float W1_pressure, const float delta_W_velocity, const float mass, const float J);

template<MaterialE MaterialTypeSolid>
__forceinline__ __device__ void store_data_neigbours_coupling_solid(const ParticleBuffer<MaterialTypeSolid> particle_buffer_solid, float* __restrict__ coupling_solid, const float W_velocity, const float W1_pressure, const float contact_area, const float normal);

template<MaterialE MaterialTypeFluid>
__forceinline__ __device__ void store_data_fluid(const ParticleBuffer<MaterialTypeFluid> particle_buffer_solid, const float mass, const float J);

template<MaterialE MaterialTypeFluid>
__forceinline__ __device__ void store_data_neigbours_fluid(const ParticleBuffer<MaterialTypeFluid> particle_buffer_solid, float* __restrict__ gradient_fluid, float* __restrict__ boundary_fluid, const float W_velocity, const float W1_pressure, const float delta_W_velocity, const float mass, const float J);

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
	
	(*scaling_solid) += (volume_0 / particle_buffer_solid.lambda) * W_pressure;
	(*pressure_solid_nominator) += volume_0 * J * (-particle_buffer_solid.lambda * (J - 1.0f)) * W_pressure;
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
__forceinline__ __device__ void store_data_fluid<MaterialE::J_FLUID>(const ParticleBuffer<MaterialE::J_FLUID> particle_buffer_fluid, const float mass, const float J){			
	//Nothing
}

template<>
__forceinline__ __device__ void store_data_fluid<MaterialE::FIXED_COROTATED>(const ParticleBuffer<MaterialE::FIXED_COROTATED> particle_buffer_fluid, const float mass, const float J){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_fluid<MaterialE::SAND>(const ParticleBuffer<MaterialE::SAND> particle_buffer_fluid, const float mass, const float J){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_fluid<MaterialE::NACC>(const ParticleBuffer<MaterialE::NACC> particle_buffer_fluid, const float mass, const float J){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_fluid<MaterialE::FIXED_COROTATED_GHOST>(const ParticleBuffer<MaterialE::FIXED_COROTATED_GHOST> particle_buffer_fluid, const float mass, const float J){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_fluid<MaterialE::J_FLUID>(const ParticleBuffer<MaterialE::J_FLUID> particle_buffer_fluid, float* __restrict__ gradient_fluid, float* __restrict__ boundary_fluid, const float W_velocity, const float W1_pressure, const float delta_W_velocity, const float mass, const float J){
	(*gradient_fluid) += -(mass / particle_buffer_fluid.rho) * J * W1_pressure * delta_W_velocity;
	
	//FIXME: Is that correct?  Actually also not particle based maybe? And just add once?
	//(*boundary_fluid) += W_velocity * W1_pressure * boundary_normal[alpha];
}

template<>
__forceinline__ __device__ void store_data_neigbours_fluid<MaterialE::FIXED_COROTATED>(const ParticleBuffer<MaterialE::FIXED_COROTATED> particle_buffer_fluid, float* __restrict__ gradient_fluid, float* __restrict__ boundary_fluid, const float W_velocity, const float W1_pressure, const float delta_W_velocity, const float mass, const float J){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_fluid<MaterialE::SAND>(const ParticleBuffer<MaterialE::SAND> particle_buffer_fluid, float* __restrict__ gradient_fluid, float* __restrict__ boundary_fluid, const float W_velocity, const float W1_pressure, const float delta_W_velocity, const float mass, const float J){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_fluid<MaterialE::NACC>(const ParticleBuffer<MaterialE::NACC> particle_buffer_fluid, float* __restrict__ gradient_fluid, float* __restrict__ boundary_fluid, const float W_velocity, const float W1_pressure, const float delta_W_velocity, const float mass, const float J){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_fluid<MaterialE::FIXED_COROTATED_GHOST>(const ParticleBuffer<MaterialE::FIXED_COROTATED_GHOST> particle_buffer_fluid, float* __restrict__ gradient_fluid, float* __restrict__ boundary_fluid, const float W_velocity, const float W1_pressure, const float delta_W_velocity, const float mass, const float J){
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

template<typename Partition, typename Grid, MaterialE MaterialTypeSolid, MaterialE MaterialTypeFluid>
__forceinline__ __device__ void aggregate_data_solid(const ParticleBuffer<MaterialTypeSolid> particle_buffer_solid, const ParticleBuffer<MaterialTypeFluid> particle_buffer_fluid, const ParticleBuffer<MaterialTypeSolid> next_particle_buffer_solid, const ParticleBuffer<MaterialTypeFluid> next_particle_buffer_fluid, const Partition prev_partition, const Grid grid_solid, const Grid grid_fluid, const SurfaceParticleBuffer surface_particle_buffer, const std::array<float, 3>* __restrict__ position_shared, const float* __restrict__ mass_shared, const float* __restrict__ J_shared, const std::array<float, 3>* __restrict__ normal_shared, const SurfacePointType* __restrict__ point_type_shared, const float* __restrict__ contact_area_shared, const int particle_offset, const int current_blockno, const ivec3 current_blockid, const ivec3 block_cellid, const int particle_id_in_block, float* __restrict__ scaling_solid, float* __restrict__ pressure_solid_nominator, float* __restrict__ pressure_solid_denominator, float* __restrict__ mass_solid, float* __restrict__ gradient_solid, float* __restrict__ coupling_solid, float* __restrict__ coupling_fluid) {
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
	
	const ivec3 global_base_index_fluid_velocity = get_cell_id<INTERPOLATION_DEGREE_FLUID_VELOCITY>(pos.data_arr(), grid_fluid.get_offset());//NOTE: Using solid/intreface quadrature position

	//Get position relative to grid cell
	const vec3 local_pos_solid_pressure = pos - (global_base_index_solid_pressure + vec3(grid_solid.get_offset()[0], grid_solid.get_offset()[1], grid_solid.get_offset()[2])) * config::G_DX;
	const vec3 local_pos_solid_velocity = pos - (global_base_index_solid_velocity + vec3(grid_solid.get_offset()[0], grid_solid.get_offset()[1], grid_solid.get_offset()[2])) * config::G_DX;
	
	const vec3 local_pos_fluid_velocity = pos - (global_base_index_fluid_velocity + vec3(grid_fluid.get_offset()[0], grid_fluid.get_offset()[1], grid_fluid.get_offset()[2])) * config::G_DX;

	//Calculate weights
	vec3x3 weight_solid_pressure;
	vec3x3 weight_solid_velocity;
	vec3x3 gradient_weight_solid_velocity;
	
	vec3x3 weight_fluid_velocity;
	
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
	}
	
	//Get near fluid particles
	bool has_neighbours_local = false;
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

					//Fetch position and determinant of deformation gradient
					FetchParticleBufferDataIntermediate fetch_particle_buffer_tmp = {};
					fetch_particle_buffer_data<MaterialTypeFluid>(particle_buffer_fluid, advection_source_blockno_fluid, source_pidib_fluid, fetch_particle_buffer_tmp);
					//const float mass_fluid = fetch_particle_buffer_tmp.mass;
					vec3 pos_fluid {fetch_particle_buffer_tmp.pos[0], fetch_particle_buffer_tmp.pos[1], fetch_particle_buffer_tmp.pos[2]};
					//float J_fluid	 = fetch_particle_buffer_tmp.J;
					
					const vec3 diff = pos - pos_fluid;
					const float distance = std::sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
					
					if(distance <= 0.5f * config::G_DX){
						has_neighbours_local = true;
						break;
					}
				}
			}
		}
	}
	
	const bool has_neighbours = (__syncthreads_or(has_neighbours_local ? 1 : 0) == 1);
	
	//Store data
	//Note: Weights are 0 if outside of interpolation degree/radius around particles cell
	//Foreach node in the block we add values accoring to particle kernel, also handling all neighbours of the particles cell
	
	for(size_t local_block_index = 0; local_block_index < get_thread_count<BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, config::G_BLOCKVOLUME); local_block_index++){
		const size_t block_index = get_global_index<BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_block_index);
		const ivec3 local_id {static_cast<int>((block_index / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE), static_cast<int>((block_index / config::G_BLOCKSIZE) % config::G_BLOCKSIZE), static_cast<int>(block_index % config::G_BLOCKSIZE)};
		
		const ivec3 local_offset_0 = global_base_index_solid_pressure - (block_cellid + local_id);
		const ivec3 local_offset_2 = global_base_index_solid_velocity - (block_cellid + local_id);
		
		const ivec3 absolute_local_offset_0 {std::abs(local_offset_0[0]), std::abs(local_offset_0[1]), std::abs(local_offset_0[2])};
		const ivec3 absolute_local_offset_2 {std::abs(local_offset_2[0]), std::abs(local_offset_2[1]), std::abs(local_offset_2[2])};

		//Weight
		const float W_pressure = (absolute_local_offset_0[0] < 3 ? weight_solid_pressure(0, absolute_local_offset_0[0]) : 0.0f) * (absolute_local_offset_0[1] < 3 ? weight_solid_pressure(1, absolute_local_offset_0[1]) : 0.0f) * (absolute_local_offset_0[2] < 3 ? weight_solid_pressure(2, absolute_local_offset_0[2]) : 0.0f);
		const float W_velocity = (absolute_local_offset_2[0] < 3 ? weight_solid_velocity(0, absolute_local_offset_2[0]) : 0.0f) * (absolute_local_offset_2[1] < 3 ? weight_solid_velocity(1, absolute_local_offset_2[1]) : 0.0f) * (absolute_local_offset_2[2] < 3 ? weight_solid_velocity(2, absolute_local_offset_2[2]) : 0.0f);
		
		float* current_scaling_solid = &(scaling_solid[local_block_index]);
		float* current_pressure_solid_nominator = &(pressure_solid_nominator[local_block_index]);
		float* current_pressure_solid_denominator = &(pressure_solid_denominator[local_block_index]);
		
		store_data_solid(particle_buffer_solid, current_scaling_solid, current_pressure_solid_nominator, current_pressure_solid_denominator, W_pressure, W_velocity, mass, J);
	}
	
	for(size_t local_block_index = 0; local_block_index < get_thread_count<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, 3 * config::G_BLOCKVOLUME); local_block_index++){
		const size_t block_index = get_global_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_block_index) / 3;
		const size_t alpha = get_global_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_block_index) % 3;
		const ivec3 local_id {static_cast<int>((block_index / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE), static_cast<int>((block_index / config::G_BLOCKSIZE) % config::G_BLOCKSIZE), static_cast<int>(block_index % config::G_BLOCKSIZE)};
		
		const ivec3 local_offset_2 = global_base_index_solid_velocity - (block_cellid + local_id);
		
		const ivec3 absolute_local_offset_2 {std::abs(local_offset_2[0]), std::abs(local_offset_2[1]), std::abs(local_offset_2[2])};

		//Weight
		const float W_velocity = (absolute_local_offset_2[0] < 3 ? weight_solid_velocity(0, absolute_local_offset_2[0]) : 0.0f) * (absolute_local_offset_2[1] < 3 ? weight_solid_velocity(1, absolute_local_offset_2[1]) : 0.0f) * (absolute_local_offset_2[2] < 3 ? weight_solid_velocity(2, absolute_local_offset_2[2]) : 0.0f);
		
		mass_solid[local_block_index] += mass * W_velocity;
	}
	
	for(size_t local_block_index = 0; local_block_index < get_thread_count<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, 3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME); local_block_index++){
		const size_t block_index = get_global_index<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_block_index) / (3 * NUM_COLUMNS_PER_BLOCK);
		const size_t column = (get_global_index<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_block_index) / 3) % NUM_COLUMNS_PER_BLOCK;
		const size_t alpha = get_global_index<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_block_index) % 3;
		const ivec3 local_id {static_cast<int>((block_index / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE), static_cast<int>((block_index / config::G_BLOCKSIZE) % config::G_BLOCKSIZE), static_cast<int>(block_index % config::G_BLOCKSIZE)};
		const ivec3 neighbour_local_id = ivec3(static_cast<int>((column / ((2 * INTERPOLATION_DEGREE_MAX + 1) * (2 * INTERPOLATION_DEGREE_MAX + 1))) % (2 * INTERPOLATION_DEGREE_MAX + 1)), static_cast<int>((column / (2 * INTERPOLATION_DEGREE_MAX + 1)) % (2 * INTERPOLATION_DEGREE_MAX + 1)), static_cast<int>(column % (2 * INTERPOLATION_DEGREE_MAX + 1))) - ivec3(static_cast<int>(INTERPOLATION_DEGREE_MAX), static_cast<int>(INTERPOLATION_DEGREE_MAX), static_cast<int>(INTERPOLATION_DEGREE_MAX));
			
		const ivec3 local_offset_2 = global_base_index_solid_velocity - (block_cellid + local_id);
		const ivec3 neighbour_local_offset_0 = global_base_index_solid_pressure - (block_cellid + local_id + neighbour_local_id);
		
		const ivec3 absolute_local_offset_2 {std::abs(local_offset_2[0]), std::abs(local_offset_2[1]), std::abs(local_offset_2[2])};
		const ivec3 neighbour_absolute_local_offset {std::abs(neighbour_local_offset_0[0]), std::abs(neighbour_local_offset_0[1]), std::abs(neighbour_local_offset_0[2])};
																

		//Weight
		const float delta_W_velocity = ((alpha == 0 ? (absolute_local_offset_2[0] < 3 ? gradient_weight_solid_velocity(0, absolute_local_offset_2[0]) : 0.0f) : (absolute_local_offset_2[0] < 3 ? weight_solid_velocity(0, absolute_local_offset_2[0]) : 0.0f)) * (alpha == 1 ? (absolute_local_offset_2[1] < 3 ? gradient_weight_solid_velocity(1, absolute_local_offset_2[1]) : 0.0f) : (absolute_local_offset_2[1] < 3 ? weight_solid_velocity(1, absolute_local_offset_2[1]) : 0.0f)) * (alpha == 2 ? (absolute_local_offset_2[2] < 3 ? gradient_weight_solid_velocity(2, absolute_local_offset_2[2]) : 0.0f) : (absolute_local_offset_2[2] < 3 ? weight_solid_velocity(2, absolute_local_offset_2[2]) : 0.0f))) * config::G_DX_INV;
		const float W1_pressure = (neighbour_absolute_local_offset[0] < 3 ? weight_solid_pressure(0, neighbour_absolute_local_offset[0]) : 0.0f) * (neighbour_absolute_local_offset[1] < 3 ? weight_solid_pressure(1, neighbour_absolute_local_offset[1]) : 0.0f) * (neighbour_absolute_local_offset[2] < 3 ? weight_solid_pressure(2, neighbour_absolute_local_offset[2]) : 0.0f);
										
		float* current_gradient_solid = &(gradient_solid[local_block_index]);

		store_data_neigbours_solid(particle_buffer_solid, current_gradient_solid, W1_pressure, delta_W_velocity, mass, J);
	}
	
	//Only percede if we have an interface
	//FIXME: Currently only handling outer points
	if(has_neighbours){
	
		//Store data
		//Note: Weights are 0 if outside of interpolation degree/radius around particles cell
		//Foreach node in the block we add values accoring to particle kernel, also handling all neighbours of the particles cell
		
		for(size_t local_block_index = 0; local_block_index < get_thread_count<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, 3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME); local_block_index++){
			const size_t block_index = get_global_index<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_block_index) / (3 * NUM_COLUMNS_PER_BLOCK);
			const size_t column = (get_global_index<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_block_index) / 3) % NUM_COLUMNS_PER_BLOCK;
			const size_t alpha = get_global_index<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_block_index) % 3;
			const ivec3 local_id {static_cast<int>((block_index / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE), static_cast<int>((block_index / config::G_BLOCKSIZE) % config::G_BLOCKSIZE), static_cast<int>(block_index % config::G_BLOCKSIZE)};
			const ivec3 neighbour_local_id = ivec3(static_cast<int>((column / ((2 * INTERPOLATION_DEGREE_MAX + 1) * (2 * INTERPOLATION_DEGREE_MAX + 1))) % (2 * INTERPOLATION_DEGREE_MAX + 1)), static_cast<int>((column / (2 * INTERPOLATION_DEGREE_MAX + 1)) % (2 * INTERPOLATION_DEGREE_MAX + 1)), static_cast<int>(column % (2 * INTERPOLATION_DEGREE_MAX + 1))) - ivec3(static_cast<int>(INTERPOLATION_DEGREE_MAX), static_cast<int>(INTERPOLATION_DEGREE_MAX), static_cast<int>(INTERPOLATION_DEGREE_MAX));
				
			const ivec3 local_offset_2_solid = global_base_index_solid_velocity - (block_cellid + local_id);
			const ivec3 local_offset_1_fluid = global_base_index_fluid_velocity - (block_cellid + local_id);
			const ivec3 neighbour_local_offset_0_solid = global_base_index_solid_pressure - (block_cellid + local_id + neighbour_local_id);
			const ivec3 neighbour_local_offset_0_fluid = global_base_index_solid_pressure - (block_cellid + local_id + neighbour_local_id);//NOTE: Using solid pos
			
			const ivec3 absolute_local_offset_2_solid {std::abs(local_offset_2_solid[0]), std::abs(local_offset_2_solid[1]), std::abs(local_offset_2_solid[2])};
			const ivec3 absolute_local_offset_1_fluid {std::abs(local_offset_1_fluid[0]), std::abs(local_offset_1_fluid[1]), std::abs(local_offset_1_fluid[2])};
			const ivec3 neighbour_absolute_local_offset_solid {std::abs(neighbour_local_offset_0_solid[0]), std::abs(neighbour_local_offset_0_solid[1]), std::abs(neighbour_local_offset_0_solid[2])};
			const ivec3 neighbour_absolute_local_offset_fluid {std::abs(neighbour_local_offset_0_fluid[0]), std::abs(neighbour_local_offset_0_fluid[1]), std::abs(neighbour_local_offset_0_fluid[2])};
																	

			//Weight
			const float W_velocity_solid = (absolute_local_offset_2_solid[0] < 3 ? weight_solid_velocity(0, absolute_local_offset_2_solid[0]) : 0.0f) * (absolute_local_offset_2_solid[1] < 3 ? weight_solid_velocity(1, absolute_local_offset_2_solid[1]) : 0.0f) * (absolute_local_offset_2_solid[2] < 3 ? weight_solid_velocity(2, absolute_local_offset_2_solid[2]) : 0.0f);	
			const float W_velocity_fluid = (absolute_local_offset_1_fluid[0] < 3 ? weight_fluid_velocity(0, absolute_local_offset_1_fluid[0]) : 0.0f) * (absolute_local_offset_1_fluid[1] < 3 ? weight_fluid_velocity(1, absolute_local_offset_1_fluid[1]) : 0.0f) * (absolute_local_offset_1_fluid[2] < 3 ? weight_fluid_velocity(2, absolute_local_offset_1_fluid[2]) : 0.0f);
			const float W1_pressure_solid = (neighbour_absolute_local_offset_solid[0] < 3 ? weight_solid_pressure(0, neighbour_absolute_local_offset_solid[0]) : 0.0f) * (neighbour_absolute_local_offset_solid[1] < 3 ? weight_solid_pressure(1, neighbour_absolute_local_offset_solid[1]) : 0.0f) * (neighbour_absolute_local_offset_solid[2] < 3 ? weight_solid_pressure(2, neighbour_absolute_local_offset_solid[2]) : 0.0f);
			const float W1_pressure_fluid = (neighbour_absolute_local_offset_fluid[0] < 3 ? weight_solid_pressure(0, neighbour_absolute_local_offset_fluid[0]) : 0.0f) * (neighbour_absolute_local_offset_fluid[1] < 3 ? weight_solid_pressure(1, neighbour_absolute_local_offset_fluid[1]) : 0.0f) * (neighbour_absolute_local_offset_fluid[2] < 3 ? weight_solid_pressure(2, neighbour_absolute_local_offset_fluid[2]) : 0.0f);
											
					
			float* current_coupling_solid = &(coupling_solid[local_block_index]);
			float* current_coupling_fluid = &(coupling_fluid[local_block_index]);

			store_data_neigbours_coupling_solid(particle_buffer_solid, current_coupling_solid, W_velocity_solid, W1_pressure_solid, contact_area, normal[alpha]);
			store_data_neigbours_coupling_fluid(particle_buffer_fluid, current_coupling_fluid, W_velocity_fluid, W1_pressure_fluid, contact_area, normal[alpha]);
		}
	}
}

template<typename Partition, typename Grid, MaterialE MaterialTypeFluid>
__forceinline__ __device__ void aggregate_data_fluid(const ParticleBuffer<MaterialTypeFluid> particle_buffer_fluid, const ParticleBuffer<MaterialTypeFluid> next_particle_buffer_fluid, const Partition prev_partition, const Grid grid_solid, const Grid grid_fluid, const std::array<float, 3>* __restrict__ position_shared, const float* __restrict__ mass_shared, const float* __restrict__ J_shared, const int particle_offset, const int current_blockno, const ivec3 current_blockid, const ivec3 block_cellid, const int particle_id_in_block, float* __restrict__ mass_fluid, float* __restrict__ gradient_fluid, float* __restrict__ boundary_fluid) {
	const vec3 pos {position_shared[particle_id_in_block - particle_offset][0], position_shared[particle_id_in_block - particle_offset][1], position_shared[particle_id_in_block - particle_offset][2]};
	const float mass = mass_shared[particle_id_in_block - particle_offset];
	const float J  = J_shared[particle_id_in_block - particle_offset];
	
	//Get position of grid cell
	const ivec3 global_base_index_solid_pressure = get_cell_id<INTERPOLATION_DEGREE_FLUID_PRESSURE>(pos.data_arr(), grid_solid.get_offset());
	const ivec3 global_base_index_fluid_velocity = get_cell_id<INTERPOLATION_DEGREE_FLUID_VELOCITY>(pos.data_arr(), grid_fluid.get_offset());
	
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
	
	//Store data
	//Note: Weights are 0 if outside of interpolation degree/radius around particles cell
	//Foreach node in the block we add values accoring to particle kernel, also handling all neighbours of the particles cell
	for(size_t local_block_index = 0; local_block_index < get_thread_count<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, 3 * config::G_BLOCKVOLUME); local_block_index++){
		const size_t block_index = get_global_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_block_index) / 3;
		const size_t alpha = get_global_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_block_index) % 3;
		const ivec3 local_id {static_cast<int>((block_index / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE), static_cast<int>((block_index / config::G_BLOCKSIZE) % config::G_BLOCKSIZE), static_cast<int>(block_index % config::G_BLOCKSIZE)};
		
		const ivec3 local_offset_1_fluid = global_base_index_fluid_velocity - (block_cellid + local_id);
		
		const ivec3 absolute_local_offset_1_fluid {std::abs(local_offset_1_fluid[0]), std::abs(local_offset_1_fluid[1]), std::abs(local_offset_1_fluid[2])};

		//Weight
		const float W_velocity = (absolute_local_offset_1_fluid[0] < 3 ? weight_fluid_velocity(0, absolute_local_offset_1_fluid[0]) : 0.0f) * (absolute_local_offset_1_fluid[1] < 3 ? weight_fluid_velocity(1, absolute_local_offset_1_fluid[1]) : 0.0f) * (absolute_local_offset_1_fluid[2] < 3 ? weight_fluid_velocity(2, absolute_local_offset_1_fluid[2]) : 0.0f);
				
		
		mass_fluid[local_block_index] += mass * W_velocity;
	}
	
	for(size_t local_block_index = 0; local_block_index < get_thread_count<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, 3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME); local_block_index++){
		const size_t block_index = get_global_index<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_block_index) / (3 * NUM_COLUMNS_PER_BLOCK);
		const size_t column = (get_global_index<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_block_index) / 3) % NUM_COLUMNS_PER_BLOCK;
		const size_t alpha = get_global_index<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, local_block_index) % 3;
		const ivec3 local_id {static_cast<int>((block_index / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE), static_cast<int>((block_index / config::G_BLOCKSIZE) % config::G_BLOCKSIZE), static_cast<int>(block_index % config::G_BLOCKSIZE)};
		const ivec3 neighbour_local_id = ivec3(static_cast<int>((column / ((2 * INTERPOLATION_DEGREE_MAX + 1) * (2 * INTERPOLATION_DEGREE_MAX + 1))) % (2 * INTERPOLATION_DEGREE_MAX + 1)), static_cast<int>((column / (2 * INTERPOLATION_DEGREE_MAX + 1)) % (2 * INTERPOLATION_DEGREE_MAX + 1)), static_cast<int>(column % (2 * INTERPOLATION_DEGREE_MAX + 1))) - ivec3(static_cast<int>(INTERPOLATION_DEGREE_MAX), static_cast<int>(INTERPOLATION_DEGREE_MAX), static_cast<int>(INTERPOLATION_DEGREE_MAX));
			
		const ivec3 local_offset_1_fluid = global_base_index_fluid_velocity - (block_cellid + local_id);
		const ivec3 neighbour_local_offset_0 = global_base_index_solid_pressure - (block_cellid + local_id + neighbour_local_id);
		
		const ivec3 absolute_local_offset_1_fluid {std::abs(local_offset_1_fluid[0]), std::abs(local_offset_1_fluid[1]), std::abs(local_offset_1_fluid[2])};
		const ivec3 neighbour_absolute_local_offset {std::abs(neighbour_local_offset_0[0]), std::abs(neighbour_local_offset_0[1]), std::abs(neighbour_local_offset_0[2])};												

		//Weight
		const float delta_W_velocity = ((alpha == 0 ? (absolute_local_offset_1_fluid[0] < 3 ? gradient_weight_fluid_velocity(0, absolute_local_offset_1_fluid[0]) : 0.0f) : (absolute_local_offset_1_fluid[0] < 3 ? weight_fluid_velocity(0, absolute_local_offset_1_fluid[0]) : 0.0f)) * (alpha == 1 ? (absolute_local_offset_1_fluid[1] < 3 ? gradient_weight_fluid_velocity(1, absolute_local_offset_1_fluid[1]) : 0.0f) : (absolute_local_offset_1_fluid[1] < 3 ? weight_fluid_velocity(1, absolute_local_offset_1_fluid[1]) : 0.0f)) * (alpha == 2 ? (absolute_local_offset_1_fluid[2] < 3 ? gradient_weight_fluid_velocity(2, absolute_local_offset_1_fluid[2]) : 0.0f) : (absolute_local_offset_1_fluid[2] < 3 ? weight_fluid_velocity(2, absolute_local_offset_1_fluid[2]) : 0.0f))) * config::G_DX_INV;
		const float W_velocity = (absolute_local_offset_1_fluid[0] < 3 ? weight_fluid_velocity(0, absolute_local_offset_1_fluid[0]) : 0.0f) * (absolute_local_offset_1_fluid[1] < 3 ? weight_fluid_velocity(1, absolute_local_offset_1_fluid[1]) : 0.0f) * (absolute_local_offset_1_fluid[2] < 3 ? weight_fluid_velocity(2, absolute_local_offset_1_fluid[2]) : 0.0f);
		const float W1_pressure = (neighbour_absolute_local_offset[0] < 3 ? weight_solid_pressure(0, neighbour_absolute_local_offset[0]) : 0.0f) * (neighbour_absolute_local_offset[1] < 3 ? weight_solid_pressure(1, neighbour_absolute_local_offset[1]) : 0.0f) * (neighbour_absolute_local_offset[2] < 3 ? weight_solid_pressure(2, neighbour_absolute_local_offset[2]) : 0.0f);
									
		float* current_gradient_fluid = &(gradient_fluid[local_block_index]);
		float* current_boundary_fluid = &(boundary_fluid[local_block_index]);
		
		store_data_neigbours_fluid(particle_buffer_fluid, current_gradient_fluid, current_boundary_fluid, W_velocity, W1_pressure, delta_W_velocity, mass, J);
	}
}

template<typename Partition, typename Grid, MaterialE MaterialTypeSolid, MaterialE MaterialTypeFluid>
__global__ void create_iq_system(const uint32_t num_blocks, Duration dt, const ParticleBuffer<MaterialTypeSolid> particle_buffer_solid, const ParticleBuffer<MaterialTypeFluid> particle_buffer_fluid, const ParticleBuffer<MaterialTypeSolid> next_particle_buffer_solid, const ParticleBuffer<MaterialTypeFluid> next_particle_buffer_fluid, const Partition prev_partition, const Partition partition, const Grid grid_solid, const Grid grid_fluid, const SurfaceParticleBuffer surface_particle_buffer_solid, const SurfaceParticleBuffer surface_particle_buffer_fluid, const int* iq_lhs_rows, const int* iq_lhs_columns, float* iq_lhs_values, float* iq_rhs, const int* iq_solve_velocity_rows, const int* iq_solve_velocity_columns, float* iq_solve_velocity_values) {
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
	float boundary_fluid_local[(3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE];
	float velocity_fluid_local[(3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE];
	
	float coupling_solid_local[(3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE];
	float coupling_fluid_local[(3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE];

	const int src_blockno		   = static_cast<int>(blockIdx.x);
	const auto blockid			   = partition.active_keys[blockIdx.x];
	const ivec3 block_cellid = blockid * static_cast<int>(config::G_BLOCKSIZE);
	//const int particle_bucket_size_solid = next_particle_buffer_solid.particle_bucket_sizes[src_blockno];
	//const int particle_bucket_size_fluid = next_particle_buffer_fluid.particle_bucket_sizes[src_blockno];
	
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
								, surface_particle_buffer_solid
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
								, &(coupling_fluid_local[0])
							);
						}
						
						__syncthreads();
					}
					
					for(int particle_offset = 0; particle_offset < next_particle_buffer_fluid.particle_bucket_sizes[current_blockno]; particle_offset += static_cast<int>(MAX_SHARED_PARTICLE_FLUID)){
						__shared__ std::array<float, 3> position_shared[MAX_SHARED_PARTICLE_FLUID];
						__shared__ float mass_shared[MAX_SHARED_PARTICLE_FLUID];
						__shared__ float J_shared[MAX_SHARED_PARTICLE_FLUID];
						
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

							//Fetch position and determinant of deformation gradient
							FetchParticleBufferDataIntermediate fetch_particle_buffer_tmp = {};
							fetch_particle_buffer_data<MaterialTypeFluid>(particle_buffer_fluid, advection_source_blockno, source_pidib, fetch_particle_buffer_tmp);
							position_shared[particle_id_in_block - particle_offset] = {fetch_particle_buffer_tmp.pos[0], fetch_particle_buffer_tmp.pos[1], fetch_particle_buffer_tmp.pos[2]};
							mass_shared[particle_id_in_block - particle_offset] = fetch_particle_buffer_tmp.mass;
							J_shared[particle_id_in_block - particle_offset] = fetch_particle_buffer_tmp.J;
						}
						
						__syncthreads();
						
						for(int particle_id_in_block = particle_offset; particle_id_in_block < next_particle_buffer_fluid.particle_bucket_sizes[current_blockno] && (particle_id_in_block - particle_offset) < MAX_SHARED_PARTICLE_FLUID; ++particle_id_in_block) {
							aggregate_data_fluid(
								  particle_buffer_fluid
								, next_particle_buffer_fluid
								, prev_partition
								, grid_solid
								, grid_fluid
								, &(position_shared[0])
								, &(mass_shared[0])
								, &(J_shared[0])
								, particle_offset
								, current_blockno
								, current_blockid
								, block_cellid
								, particle_id_in_block
								, &(mass_fluid_local[0])
								, &(gradient_fluid_local[0])
								, &(boundary_fluid_local[0])
							);
						}
						
						__syncthreads();
					}
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
	//NOTE: Coupling was stored in transposed form
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
			
			//P = O^T * M^-1 * O => P_row_col = sum(all_rows_in_column; current_o_row * current_o_col / current_m) => add current_o_row * current_o_col / current_m to entries for current row and for neighbour row (row and col); Other entries are zero
			if(get_thread_index<BLOCK_SIZE, (NUM_ROWS_PER_BLOCK * NUM_COLUMNS_PER_BLOCK + BLOCK_SIZE - 1) / BLOCK_SIZE>(row * NUM_COLUMNS_PER_BLOCK + column) == threadIdx.x){
				//Caling only for diagonal element
				const float scaling_solid = (column == IDENTIITY_NEIGHBOUR_INDEX ? current_scaling_solid : 0.0f);
				
				const float gradient_by_mass_solid = (current_gradient_solid_row[0] * current_gradient_solid_column[0] / current_mass_solid[0] + current_gradient_solid_row[1] * current_gradient_solid_column[1] / current_mass_solid[1] + current_gradient_solid_row[2] * current_gradient_solid_column[2] / current_mass_solid[2]);
				const float gradient_by_mass_fluid = (current_gradient_fluid_row[0] * current_gradient_fluid_column[0] / current_mass_fluid[0] + current_gradient_fluid_row[1] * current_gradient_fluid_column[1] / current_mass_fluid[1] + current_gradient_fluid_row[2] * current_gradient_fluid_column[2] / current_mass_fluid[2]);
				
				const float boundary_by_mass = (current_boundary_fluid_row[0] * current_boundary_fluid_column[0] / current_mass_fluid[0] + current_boundary_fluid_row[1] * current_boundary_fluid_column[1] / current_mass_fluid[1] + current_boundary_fluid_row[2] * current_boundary_fluid_column[2] / current_mass_fluid[2]);
				const float gradient_and_boundary_by_mass = (current_gradient_fluid_row[0] * current_boundary_fluid_column[0] / current_mass_fluid[0] + current_gradient_fluid_row[1] * current_boundary_fluid_column[1] / current_mass_fluid[1] + current_gradient_fluid_row[2] * current_boundary_fluid_column[2] / current_mass_fluid[2]);
				
				const float gradient_and_coupling_by_mass_solid = (current_gradient_solid_row[0] * current_coupling_solid_column[0] / current_mass_solid[0] + current_gradient_solid_row[1] * current_coupling_solid_column[1] / current_mass_solid[1] + current_gradient_solid_row[2] * current_coupling_solid_column[2] / current_mass_solid[2]);
				const float gradient_and_coupling_by_mass_fluid = (current_gradient_fluid_row[0] * current_coupling_fluid_column[0] / current_mass_fluid[0] + current_gradient_fluid_row[1] * current_coupling_fluid_column[1] / current_mass_fluid[1] + current_gradient_fluid_row[2] * current_coupling_fluid_column[2] / current_mass_fluid[2]);
				const float boundary_and_coupling_by_mass_fluid = (current_boundary_fluid_row[0] * current_coupling_fluid_column[0] / current_mass_fluid[0] + current_boundary_fluid_row[1] * current_coupling_fluid_column[1] / current_mass_fluid[1] + current_boundary_fluid_row[2] * current_coupling_fluid_column[2] / current_mass_fluid[2]);
				
				const float coupling_by_mass_solid = (current_coupling_solid_row[0] * current_coupling_solid_column[0] / current_mass_solid[0] + current_coupling_solid_row[1] * current_coupling_solid_column[1] / current_mass_solid[1] + current_coupling_solid_row[2] * current_coupling_solid_column[2] / current_mass_solid[2]);
				const float coupling_by_mass_fluid = (current_coupling_fluid_row[0] * current_coupling_fluid_column[0] / current_mass_fluid[0] + current_coupling_fluid_row[1] * current_coupling_fluid_column[1] / current_mass_fluid[1] + current_coupling_fluid_row[2] * current_coupling_fluid_column[2] / current_mass_fluid[2]);
				
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
					   current_mass_solid[0] > 0.0f
					&& current_mass_solid[1] > 0.0f
					&& current_mass_solid[2] > 0.0f
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
					   current_mass_fluid[0] > 0.0f
					&& current_mass_fluid[1] > 0.0f
					&& current_mass_fluid[2] > 0.0f
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
							//const int column_cellno = iq_lhs_columns[iq_lhs_rows[base_row + row] + lhs_column_indices[column]];
							//printf("IQ_LHS %d # %d %d # %.28f\n", static_cast<int>(column_cellno), static_cast<int>(i), static_cast<int>(lhs_block_offsets_per_row[i][j]), a[i][lhs_block_offsets_per_row[i][j]]);
						}
						
						if(i == 0 && lhs_block_offsets_per_row[i][j] == 0 && a[i][lhs_block_offsets_per_row[i][j]] != 0.0f){
							/*printf("IQ_LHS00 %.28f # %.28f # %.28f %.28f %.28f # %.28f %.28f %.28f # %.28f %.28f %.28f\n"
								, a[i][lhs_block_offsets_per_row[i][j]]
								, gradient_by_mass_solid
								, current_mass_solid[0]	, current_mass_solid[1], current_mass_solid[2]
								, current_gradient_solid_row[0], current_gradient_solid_row[1], current_gradient_solid_row[2]
								, current_gradient_solid_column[0], current_gradient_solid_column[1], current_gradient_solid_column[2]						
							);*/
						}
						
						if(i == 1 && lhs_block_offsets_per_row[i][j] == 1 && a[i][lhs_block_offsets_per_row[i][j]] != 0.0f){
							/*printf("IQ_LHS11 %.28f # %.28f # %.28f %.28f %.28f # %.28f %.28f %.28f # %.28f %.28f %.28f\n"
								, a[i][lhs_block_offsets_per_row[i][j]]
								, gradient_by_mass_fluid
								, current_mass_fluid[0]	, current_mass_fluid[1], current_mass_fluid[2]
								, current_gradient_fluid_row[0], current_gradient_fluid_row[1], current_gradient_fluid_row[2]
								, current_gradient_fluid_column[0], current_gradient_fluid_column[1], current_gradient_fluid_column[2]						
							);*/
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
									//printf("T_IQ_LHS %d %d # %d %d %d # %.28f # %.28f # %.28f %.28f %.28f # %.28f %.28f %.28f\n", row_index, column_index, static_cast<int>(i), static_cast<int>(j), static_cast<int>(lhs_block_offsets_per_row[i][j]), a_transposed[i][lhs_block_offsets_per_row[i][j]], gradient_by_mass_solid, current_gradient_solid_row[0], current_gradient_solid_row[1], current_gradient_solid_row[2], current_gradient_solid_column[0], current_gradient_solid_column[1], current_gradient_solid_column[2]);
									//printf("T_IQ_LHS %d %d # %d %d # %d # %.28f # %.28f\n", row_index, iq_lhs_columns[column_index], static_cast<int>(i), static_cast<int>(lhs_block_offsets_per_row[i][j]), static_cast<int>(column), a[i][lhs_block_offsets_per_row[i][j]], a_transposed[i][lhs_block_offsets_per_row[i][j]]);
									//printf("T_IQ_LHS %d # %d %d # %.28f\n", static_cast<int>(base_row + row), static_cast<int>(i), static_cast<int>(lhs_block_offsets_per_row[i][j]), a_transposed[i][lhs_block_offsets_per_row[i][j]]);
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
			
			__syncthreads();
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
					//printf("IQ_RHS %d # %d # %.28f\n", static_cast<int>(base_row + row), static_cast<int>(i), b[i]);
					//const ivec3 row_cellid = block_cellid + local_id;
					//printf("IQ_RHS %d # %d # %.28f # %d %d %d\n", row_index, static_cast<int>(i), b[i], row_cellid[0], row_cellid[1], row_cellid[2]);
				}
				
				if(std::abs(b[i]) > 1e-5){
					//printf("IQ_RHS %d # %d # %.28f # %.28f %.28f %.28f # %.28f %.28f %.28f # %.28f %.28f %.28f # %.28f %.28f %.28f\n", row_index, static_cast<int>(i), b[i], current_gradient_solid_row[0], current_gradient_solid_row[1], current_gradient_solid_row[2], current_gradient_fluid_row[0], current_gradient_fluid_row[1], current_gradient_fluid_row[2], current_velocity_solid[0], current_velocity_solid[1], current_velocity_solid[2], current_velocity_fluid[0], current_velocity_fluid[1], current_velocity_fluid[2]);
				}
				
				if(isnan(b[i])){
					printf("ABC1 %d # %.28f %.28f %.28f # %.28f %.28f %.28f\n", static_cast<int>(i), current_gradient_fluid_row[0], current_gradient_fluid_row[1], current_gradient_fluid_row[2], current_velocity_fluid[0], current_velocity_fluid[1], current_velocity_fluid[2]);
				}
				
				atomicAdd(&(iq_rhs[row_index]), b[i]);
			}
		}
		
		__syncthreads();
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
		}
	}

#if (FIXED_COROTATED_GHOST_ENABLE_STRAIN_UPDATE == 0)
	
	//Particles in cell can have offset of [0, 5] ([0, 3] current block, 2 for offset caused by kernel 2 in storing); Then additional 2 are added in both directions for max kernel degree => [-2, 7] or absolute [0, 9] with offset 2
	constexpr size_t KERNEL_SIZE = 2 * INTERPOLATION_DEGREE_MAX + 5 + 1;//Plus one for both sides being inclusive
	constexpr size_t KERNEL_OFFSET = INTERPOLATION_DEGREE_MAX;
	
	constexpr size_t CELL_COUNT = KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE;
	
	const int particle_bucket_size_solid = next_particle_buffer_soild.particle_bucket_sizes[src_blockno];
	
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
		
		//Get position relative to grid cell
		const vec3 local_pos_solid_0 = pos - (global_base_index_solid_0 + vec3(grid_solid.get_offset()[0], grid_solid.get_offset()[1], grid_solid.get_offset()[2])) * config::G_DX;

		//Calculate weights
		vec3x3 weight_solid_pressure;
		
		#pragma unroll 3
		for(int dd = 0; dd < 3; ++dd) {
			const std::array<float, INTERPOLATION_DEGREE_SOLID_PRESSURE + 1> current_weight_solid_pressure = bspline_weight<float, INTERPOLATION_DEGREE_SOLID_PRESSURE>(local_pos_solid_0[dd]);
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
					const ivec3 local_id = (global_base_index_solid_0 - block_cellid) + ivec3(i, j, k);
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
		
		update_strain<MaterialTypeSolid>(particle_buffer_solid, advection_source_blockno, source_pidib, weighted_pressure);
	}
	
#endif
}

}// namespace iq

//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers, readability-identifier-naming, misc-definitions-in-headers)
}// namespace mn

#endif