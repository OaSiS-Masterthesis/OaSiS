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

//Matrix layout
constexpr size_t LHS_NUM_ROWS_PER_BLOCK = config::G_BLOCKVOLUME;
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

constexpr size_t SOLVE_VELOCITY_NUM_ROWS_PER_BLOCK = 3 * config::G_BLOCKVOLUME;

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
template<size_t MatrixSizeX, size_t MatrixSizeY, size_t NumRowsPerBlock, typename Partition>
__global__ void clear_iq_system(const size_t* lhs_num_blocks_per_row, const std::array<size_t, MatrixSizeX>* lhs_block_offsets_per_row, const uint32_t num_active_blocks, const uint32_t num_blocks, const Partition partition, int* iq_lhs_rows, int* iq_lhs_columns, float* iq_lhs_values) {
	//const int src_blockno		   = static_cast<int>(blockIdx.x);
	const auto blockid			   = partition.active_keys[blockIdx.x];
	
	//Handle own rows and add column data for all neighbour cells of each cell (fixed amount)
	//We can calculate the offset in the column array by our id and the amount of neighbour cells (=columns)
	const size_t base_row = NumRowsPerBlock * blockIdx.x;
	
	//Accumulate blocks per row
	std::array<size_t, MatrixSizeY> accumulated_blocks_per_row;
	accumulated_blocks_per_row[0] = 0;
	for(size_t i = 1; i < MatrixSizeY; ++i){
		accumulated_blocks_per_row[i] = accumulated_blocks_per_row[i - 1] + lhs_num_blocks_per_row[i - 1];
	}
	
	//Fill matrix
	for(size_t row = static_cast<int>(threadIdx.x); row < NumRowsPerBlock; row += static_cast<int>(blockDim.x)){
		const ivec3 cellid = blockid * static_cast<int>(config::G_BLOCKSIZE) + ivec3((static_cast<int>(row) / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE, (static_cast<int>(row) / config::G_BLOCKSIZE) % config::G_BLOCKSIZE, static_cast<int>(row) % config::G_BLOCKSIZE);
		
		for(size_t row_offset = 0; row_offset < MatrixSizeY; ++row_offset){
			iq_lhs_rows[row_offset * NumRowsPerBlock * num_blocks + base_row + row] = accumulated_blocks_per_row[row_offset] * NumRowsPerBlock * num_active_blocks * NUM_COLUMNS_PER_BLOCK + lhs_num_blocks_per_row[row_offset] * (base_row + row) * NUM_COLUMNS_PER_BLOCK;
		}
		
		int neighbour_cellnos[NUM_COLUMNS_PER_BLOCK];
		for(size_t column = 0; column < NUM_COLUMNS_PER_BLOCK; ++column){
			const ivec3 neighbour_local_id = ivec3(static_cast<int>((column / ((2 * INTERPOLATION_DEGREE_MAX + 1) * (2 * INTERPOLATION_DEGREE_MAX + 1))) % (2 * INTERPOLATION_DEGREE_MAX + 1)), static_cast<int>((column / (2 * INTERPOLATION_DEGREE_MAX + 1)) % (2 * INTERPOLATION_DEGREE_MAX + 1)), static_cast<int>(column % (2 * INTERPOLATION_DEGREE_MAX + 1))) - ivec3(static_cast<int>(INTERPOLATION_DEGREE_MAX), static_cast<int>(INTERPOLATION_DEGREE_MAX), static_cast<int>(INTERPOLATION_DEGREE_MAX));
			const ivec3 neighbour_cellid = cellid + neighbour_local_id;
			const ivec3 neighbour_blockid = neighbour_cellid / static_cast<int>(config::G_BLOCKSIZE);
			
			const ivec3 neighbour_base_cellid = neighbour_blockid * static_cast<int>(config::G_BLOCKSIZE);
			const ivec3 neighbour_celloffset = neighbour_cellid - neighbour_base_cellid;
			
			const int neighbour_blockno = partition.query(neighbour_blockid);
			const int neighbour_cellno = NumRowsPerBlock * neighbour_blockno + (config::G_BLOCKSIZE * config::G_BLOCKSIZE) * neighbour_celloffset[0] + config::G_BLOCKSIZE * neighbour_celloffset[1] + neighbour_celloffset[2];
			
			neighbour_cellnos[column] = neighbour_cellno;
		}
		
		//Sort columns
		thrust::sort(thrust::seq, neighbour_cellnos, neighbour_cellnos + NUM_COLUMNS_PER_BLOCK);
		
		for(size_t column = 0; column < NUM_COLUMNS_PER_BLOCK; ++column){
			for(size_t row_offset = 0; row_offset < MatrixSizeY; ++row_offset){
				for(size_t column_offset_index = 0; column_offset_index < lhs_num_blocks_per_row[row_offset]; ++column_offset_index){
					iq_lhs_columns[iq_lhs_rows[row_offset * NumRowsPerBlock * num_blocks + base_row + row] + column_offset_index * NUM_COLUMNS_PER_BLOCK + column] = lhs_block_offsets_per_row[row_offset][column_offset_index] * num_blocks * NumRowsPerBlock + neighbour_cellnos[column];
					iq_lhs_values[iq_lhs_rows[row_offset * NumRowsPerBlock * num_blocks + base_row + row] + column_offset_index * NUM_COLUMNS_PER_BLOCK + column] = 0.0f;
				}
			}
		}
	}
}

template<size_t NumRowsPerBlock, typename Partition>
__global__ void fill_empty_rows(const uint32_t num_blocks, const Partition partition, int* iq_lhs_rows) {
	//const int src_blockno		   = static_cast<int>(blockIdx.x);
	const auto blockid			   = partition.active_keys[blockIdx.x];
	
	//Handle own rows and add column data for all neighbour cells of each cell (fixed amount)
	//We can calculate the offset in the column array by our id and the amount of neighbour cells (=columns)
	const size_t base_row = NumRowsPerBlock * blockIdx.x;
	
	//Fill empty rows
	for(size_t row = static_cast<int>(threadIdx.x); row < NumRowsPerBlock; row += static_cast<int>(blockDim.x)){
		const ivec3 cellid = blockid * static_cast<int>(config::G_BLOCKSIZE) + ivec3((static_cast<int>(row) / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE, (static_cast<int>(row) / config::G_BLOCKSIZE) % config::G_BLOCKSIZE, static_cast<int>(row) % config::G_BLOCKSIZE);

		//TODO: Use more efficient way, like maybe binary search
		int next_active_base_row;
		int next_active_row;
		for(int next_row_offset = 0; (base_row + row + next_row_offset) < (NumRowsPerBlock * num_blocks + 1); ++next_row_offset){
			if(iq_lhs_rows[base_row + row + next_row_offset] != 0){
				next_active_base_row = (base_row + row + next_row_offset) / NumRowsPerBlock;
				next_active_row = (base_row + row + next_row_offset) % NumRowsPerBlock;
			}
		}

		for(size_t row_offset = 0; row_offset < LHS_MATRIX_SIZE_Y; ++row_offset){
			iq_lhs_rows[row_offset * NumRowsPerBlock * num_blocks + base_row + row] = iq_lhs_rows[row_offset * NumRowsPerBlock * num_blocks + next_active_base_row + next_active_row];
		}
	}
}

template<MaterialE MaterialTypeSolid>
__forceinline__ __device__ void store_data_solid(const ParticleBuffer<MaterialTypeSolid> particle_buffer_solid, float* scaling_solid, float* pressure_solid_nominator, float* pressure_solid_denominator, const float W_0, const float W_2, const FetchParticleBufferDataIntermediate& data);

template<MaterialE MaterialTypeSolid>
__forceinline__ __device__ void store_data_neigbours_solid(const ParticleBuffer<MaterialTypeSolid> particle_buffer_solid, float* gradient_solid, const float W1_0, const float delta_w_2, const FetchParticleBufferDataIntermediate& data);

template<MaterialE MaterialTypeFluid>
__forceinline__ __device__ void store_data_fluid(const ParticleBuffer<MaterialTypeFluid> particle_buffer_solid, const FetchParticleBufferDataIntermediate& data);

template<MaterialE MaterialTypeFluid>
__forceinline__ __device__ void store_data_neigbours_fluid(const ParticleBuffer<MaterialTypeFluid> particle_buffer_solid, float* gradient_fluid, float* boundary_fluid, const float W_1, const float W1_0, const float delta_w_1, const FetchParticleBufferDataIntermediate& data);

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
	if(scaling_solid != nullptr){
		atomicAdd(scaling_solid, ((data.mass / particle_buffer_solid.rho) / particle_buffer_solid.lambda) * W_0);
	}
	if(pressure_solid_nominator != nullptr){
		atomicAdd(pressure_solid_nominator, (data.mass / particle_buffer_solid.rho) * data.J * (-particle_buffer_solid.lambda * (data.J - 1.0f)) * W_0);
	}
	if(pressure_solid_denominator != nullptr){
		atomicAdd(pressure_solid_denominator, (data.mass / particle_buffer_solid.rho) * data.J * W_0);
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
		dir_components(advect / config::G_PARTICLE_NUM_PER_BLOCK, offset.data_arr());

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
	const vec3 local_pos_solid_0 = pos - (global_base_index_solid_0 + vec3(grid_solid.get_offset()[0], grid_solid.get_offset()[1], grid_solid.get_offset()[2]) * config::G_BLOCKSIZE) * config::G_DX;
	const vec3 local_pos_solid_2 = pos - (global_base_index_solid_2 + vec3(grid_solid.get_offset()[0], grid_solid.get_offset()[1], grid_solid.get_offset()[2]) * config::G_BLOCKSIZE) * config::G_DX;

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
	for(char i = -static_cast<char>(INTERPOLATION_DEGREE_MAX); i < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; i++) {
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
		dir_components(advect / config::G_PARTICLE_NUM_PER_BLOCK, offset.data_arr());

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
	const ivec3 global_base_index_fluid_0 = get_cell_id<0>(pos.data_arr(), grid_fluid.get_offset());
	const ivec3 global_base_index_fluid_1 = get_cell_id<1>(pos.data_arr(), grid_fluid.get_offset());
	const ivec3 global_base_index_fluid_2 = get_cell_id<2>(pos.data_arr(), grid_fluid.get_offset());

	//Get position relative to grid cell
	const vec3 local_pos_solid_0 = pos - (global_base_index_solid_0 + vec3(grid_solid.get_offset()[0], grid_solid.get_offset()[1], grid_solid.get_offset()[2]) * config::G_BLOCKSIZE) * config::G_DX;
	const vec3 local_pos_fluid_1 = pos - (global_base_index_fluid_1 + vec3(grid_fluid.get_offset()[0], grid_fluid.get_offset()[1], grid_fluid.get_offset()[2]) * config::G_BLOCKSIZE) * config::G_DX;

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
	for(char i = -static_cast<char>(INTERPOLATION_DEGREE_MAX); i < static_cast<char>(INTERPOLATION_DEGREE_MAX) + 1; i++) {
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
	}
}

template<typename Partition, typename Grid, MaterialE MaterialTypeSolid, MaterialE MaterialTypeFluid>
__global__ void create_iq_system(const uint32_t num_blocks, Duration dt, const ParticleBuffer<MaterialTypeSolid> particle_buffer_solid, const ParticleBuffer<MaterialTypeFluid> particle_buffer_fluid, const ParticleBuffer<MaterialTypeSolid> next_particle_buffer_solid, const ParticleBuffer<MaterialTypeFluid> next_particle_buffer_fluid, const Partition partition, const Partition prev_partition, const Grid grid_solid, const Grid grid_fluid, const int* iq_lhs_rows, const int* iq_lhs_columns, float* iq_lhs_values, float* iq_rhs, const int* iq_solve_velocity_rows, const int* iq_solve_velocity_columns, float* iq_solve_velocity_values) {
	constexpr size_t KERNEL_SIZE = INTERPOLATION_DEGREE_MAX / config::G_BLOCKSIZE;
	
	const size_t base_row = LHS_NUM_ROWS_PER_BLOCK * blockIdx.x;
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
		if(get_global_index<BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, i / 3) < config::G_BLOCKVOLUME){
			if((i % 3) == 0) {
				velocity_solid_local[i] = grid_block_solid.val_1d(_1, get_global_index<BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, i / 3));
				velocity_fluid_local[i] = grid_block_fluid.val_1d(_1, get_global_index<BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, i / 3));
			} else if((i % 3) == 1) {
				velocity_solid_local[i] = grid_block_solid.val_1d(_2, get_global_index<BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, i / 3));
				velocity_fluid_local[i] = grid_block_fluid.val_1d(_2, get_global_index<BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, i / 3));
			} else {
				velocity_solid_local[i] = grid_block_solid.val_1d(_3, get_global_index<BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, i / 3));
				velocity_fluid_local[i] = grid_block_fluid.val_1d(_3, get_global_index<BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(threadIdx.x, i / 3));
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
		for(int grid_x = -static_cast<int>(KERNEL_SIZE); grid_x <= static_cast<int>(KERNEL_SIZE); ++grid_x){
			for(int grid_y = -static_cast<int>(KERNEL_SIZE); grid_y <= static_cast<int>(KERNEL_SIZE); ++grid_y){
				for(int grid_z = -static_cast<int>(KERNEL_SIZE); grid_z <= static_cast<int>(KERNEL_SIZE); ++grid_z){
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
		for(int grid_x = -static_cast<int>(KERNEL_SIZE); grid_x <= static_cast<int>(KERNEL_SIZE); ++grid_x){
			for(int grid_y = -static_cast<int>(KERNEL_SIZE); grid_y <= static_cast<int>(KERNEL_SIZE); ++grid_y){
				for(int grid_z = -static_cast<int>(KERNEL_SIZE); grid_z <= static_cast<int>(KERNEL_SIZE); ++grid_z){
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
	
	for(int i = 0; i < (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i){
		if(mass_solid_local[i] > 0.0f && mass_fluid_local[i] > 0.0f){
			printf("ABC3 %d %d # %.28f %.28f\n", static_cast<int>(threadIdx.x), i, mass_solid_local[i], mass_fluid_local[i]);
		}
	}
	
	//Store data in matrix
	for(int row = 0; row < LHS_NUM_ROWS_PER_BLOCK; ++row) {
		const ivec3 local_id {static_cast<int>((row / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE), static_cast<int>((row / config::G_BLOCKSIZE) % config::G_BLOCKSIZE), static_cast<int>(row % config::G_BLOCKSIZE)};
		
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
		
		for(int i = 0; i < 3; ++i){
			if(get_thread_index<BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(row) == threadIdx.x){
				current_scaling_solid = scaling_solid_local[get_thread_offset<BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(row)];
				current_pressure_solid_nominator = pressure_solid_nominator_local[get_thread_offset<BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(row)];
				current_pressure_solid_denominator = pressure_solid_denominator_local[get_thread_offset<BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(row)];
			}
			if(get_thread_index<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * row + i) == threadIdx.x){
				current_mass_solid[i] = mass_solid_local[get_thread_offset<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * row + i)];
				current_mass_fluid[i] = mass_fluid_local[get_thread_offset<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * row + i)];
				current_velocity_solid[i] = velocity_solid_local[get_thread_offset<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * row + i)];
				current_velocity_fluid[i] = velocity_fluid_local[get_thread_offset<BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * row + i)];
			}
			if(get_thread_index<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * NUM_COLUMNS_PER_BLOCK * row + 3 * row + i) == threadIdx.x){
				current_gradient_solid_row[i] = gradient_solid_local[get_thread_offset<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * NUM_COLUMNS_PER_BLOCK * row + 3 * row + i)];
				current_gradient_fluid_row[i] = gradient_fluid_local[get_thread_offset<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * NUM_COLUMNS_PER_BLOCK * row + 3 * row + i)];
				current_boundary_fluid_row[i] = boundary_fluid_local[get_thread_offset<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * NUM_COLUMNS_PER_BLOCK * row + 3 * row + i)];
				current_coupling_solid_row[i] = coupling_solid_local[get_thread_offset<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * NUM_COLUMNS_PER_BLOCK * row + 3 * row + i)];
				current_coupling_fluid_row[i] = coupling_fluid_local[get_thread_offset<BLOCK_SIZE, (3 * NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + BLOCK_SIZE - 1) / BLOCK_SIZE>(3 * NUM_COLUMNS_PER_BLOCK * row + 3 * row + i)];
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
			
			if(get_thread_index<BLOCK_SIZE, (LHS_NUM_ROWS_PER_BLOCK * NUM_COLUMNS_PER_BLOCK + BLOCK_SIZE - 1) / BLOCK_SIZE>(row * NUM_COLUMNS_PER_BLOCK + column) == threadIdx.x){
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
				
				//Clear empty values
				a[0][1] = 0.0f;
				a[0][2] = 0.0f;
				a[1][0] = 0.0f;
				a[2][0] = 0.0f;
				
				//Only calculate for particles with mass bigger than 0 (otherwise we will divide by 0)
				if(
					   current_mass_solid[0] > 0.0f
					&& current_mass_solid[1] > 0.0f
					&& current_mass_solid[2] > 0.0f
				){
					a[0][0] = current_scaling_solid / dt.count() + dt.count() * gradient_by_mass_solid;
					a[0][3] = -dt.count() * gradient_and_coupling_by_mass_solid;
				}else{
					a[0][0] = 0.0f;
					a[0][3] = 0.0f;
				}
				
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
				}else{
					a[1][1] = 0.0f;
					a[1][2] = 0.0f;
					a[1][3] = 0.0f;
					a[2][2] = 0.0f;
					a[2][3] = 0.0f;
				}
				
				if(
					   current_mass_solid[0] > 0.0f
					&& current_mass_solid[1] > 0.0f
					&& current_mass_solid[2] > 0.0f
					&& current_mass_fluid[0] > 0.0f
					&& current_mass_fluid[1] > 0.0f
					&& current_mass_fluid[2] > 0.0f
				){
					a[3][3] += dt.count() * (coupling_by_mass_solid + coupling_by_mass_fluid);
				}else{
					a[3][3] = 0.0f;
				}
				
				//Fill symmetric
				a[3][0] = a[0][3];
				a[2][1] = a[1][2];
				a[3][1] = a[1][3];
				a[3][2] = a[2][3];
				
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
						solve_velocity[k][0][0] = dt.count() * current_gradient_solid_column[k] / current_mass_solid[k];
						solve_velocity[k][0][2] = dt.count() * current_coupling_solid_column[k] / current_mass_solid[k];
					}else{
						solve_velocity[k][0][0] = 0.0f;
						solve_velocity[k][0][2] = 0.0f;
					}
					
					if(
						   current_mass_fluid[k] > 0.0f
					){
						solve_velocity[k][1][1] = dt.count() * current_gradient_fluid_column[k] / current_mass_fluid[k];
						solve_velocity[k][1][2] = dt.count() * current_boundary_fluid_row[k] / current_mass_fluid[k];
						solve_velocity[k][1][3] = dt.count() * current_coupling_fluid_column[k] / current_mass_fluid[k];
					}else{
						solve_velocity[k][1][1] = 0.0f;
						solve_velocity[k][1][2] = 0.0f;
						solve_velocity[k][1][3] = 0.0f;
					}
				}
				
				//Store at index (blockid + row, blockid + column), adding it to existing value
				for(size_t i = 0; i < LHS_MATRIX_SIZE_Y; ++i){
					const int row_index = i * LHS_NUM_ROWS_PER_BLOCK * num_blocks + base_row + row;
					for(size_t j = 0; j < lhs_num_blocks_per_row[i]; ++j){
						
						const int column_index = iq_lhs_rows[row_index] + j * NUM_COLUMNS_PER_BLOCK + column;
						
						atomicAdd(&(iq_lhs_values[column_index]), a[i][lhs_block_offsets_per_row[i][j]]);
					}
				}
				
				for(int k = 0; k < 3; ++k){
					for(size_t i = 0; i < SOLVE_VELOCITY_MATRIX_SIZE_Y; ++i){
						const int row_index = i * SOLVE_VELOCITY_NUM_ROWS_PER_BLOCK * num_blocks + base_row + row;
						for(size_t j = 0; j < solve_velocity_num_blocks_per_row[i]; ++j){
							
							const int column_index = iq_solve_velocity_rows[row_index] + j * NUM_COLUMNS_PER_BLOCK + column;
							
							atomicAdd(&(iq_solve_velocity_values[column_index]), solve_velocity[k][i][solve_velocity_block_offsets_per_row[i][j]]);
						}
					}
				}
			}
		}
		
		if(get_thread_index<BLOCK_SIZE, (LHS_NUM_ROWS_PER_BLOCK + BLOCK_SIZE - 1) / BLOCK_SIZE>(row) == threadIdx.x){
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
			b[1] = gradient_and_velocity_fluid;
			b[2] = boundary_and_velocity_fluid - 0.0f;//FIXME: Correct external force from air interface or something with surface tension
			b[3] = coupling_and_velocity_solid - coupling_and_velocity_fluid;
			
			
			for(size_t i = 0; i < LHS_MATRIX_SIZE_Y; ++i){
				const int row_index = i * LHS_NUM_ROWS_PER_BLOCK * num_blocks + base_row + row;
				atomicAdd(&(iq_rhs[row_index]), b[i]);
			}
		}
	}
}
}// namespace iq

//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers, readability-identifier-naming, misc-definitions-in-headers)
}// namespace mn

#endif