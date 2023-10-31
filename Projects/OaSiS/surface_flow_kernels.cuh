#ifndef SURFACE_FLOW_KERNELS_CUH
#define SURFACE_FLOW_KERNELS_CUH

#include "kernels.cuh"
#include "iq.cuh"

namespace mn {
//TODO: Make magic numbers to constants where suitable
//TODO: Ensure call dimensions and such are small enough to allow narrowing conversations. Or directly use unsigned where possible
//TODO: Maybe use names instead of formula signs for better understanding
//NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers, readability-identifier-naming, misc-definitions-in-headers) CUDA does not yet support std::span; Common names for physical formulas; Cannot declare __global__ functions inline

constexpr size_t SIMPLE_SURFACE_FLOW_MAX_SHARED_PARTICLE_SOLID = config::G_BLOCKVOLUME * config::G_MAX_PARTICLES_IN_CELL >> 4;

constexpr size_t SIMPLE_SURFACE_FLOW_LHS_MATRIX_SIZE_Y = 6;
constexpr size_t SIMPLE_SURFACE_FLOW_LHS_MATRIX_SIZE_X = 6;

constexpr size_t SIMPLE_SURFACE_FLOW_LHS_MATRIX_TOTAL_BLOCK_COUNT = 24;
__device__ const std::array<size_t, SIMPLE_SURFACE_FLOW_LHS_MATRIX_SIZE_Y> simple_surface_flow_lhs_num_blocks_per_row = {
	  3
	, 3
	, 3
	, 5
	, 5
	, 5
};

__device__ const std::array<std::array<size_t, SIMPLE_SURFACE_FLOW_LHS_MATRIX_SIZE_X>, SIMPLE_SURFACE_FLOW_LHS_MATRIX_SIZE_Y> simple_surface_flow_lhs_block_offsets_per_row = {{
	  {0, 3, 4, std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()}
	, {1, 3, 5, std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()}
	, {2, 4, 5, std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()}
	, {0, 1, 3, 4, 5, std::numeric_limits<size_t>::max()}
	, {0, 2, 3, 4, 5, std::numeric_limits<size_t>::max()}
	, {1, 2, 3, 4, 5, std::numeric_limits<size_t>::max()}
}};

constexpr size_t SIMPLE_SURFACE_FLOW_SOLVE_VELOCITY_MATRIX_SIZE_Y = 3;
constexpr size_t SIMPLE_SURFACE_FLOW_SOLVE_VELOCITY_MATRIX_SIZE_X = 6;

constexpr size_t SIMPLE_SURFACE_FLOW_SOLVE_VELOCITY_MATRIX_TOTAL_BLOCK_COUNT = 9;
__device__ const std::array<size_t, SIMPLE_SURFACE_FLOW_SOLVE_VELOCITY_MATRIX_SIZE_Y> simple_surface_flow_solve_velocity_num_blocks_per_row = {
	  3
	, 3
	, 3
};
__device__ const std::array<std::array<size_t, SIMPLE_SURFACE_FLOW_SOLVE_VELOCITY_MATRIX_SIZE_X>, SIMPLE_SURFACE_FLOW_SOLVE_VELOCITY_MATRIX_SIZE_Y> simple_surface_flow_solve_velocity_block_offsets_per_row = {{
	  {0, 3, 4, std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()}
	, {1, 3, 5, std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()}
	, {2, 4, 5, std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()}
}};

struct SimpleSurfaceFlowIQCreatePointers {
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
	
	const int* coupling_solid_domain_rows;
	const int* coupling_solid_domain_columns;
	float* coupling_solid_domain_values;
	const int* coupling_fluid_rows;
	const int* coupling_fluid_columns;
	float* coupling_fluid_values;
	
	float* scaling_domain;
	float* scaling_surface;
	
	float* mass_domain;
	float* mass_surface;
	
	const int* gradient_domain_rows;
	const int* gradient_domain_columns;
	float* gradient_domain_values;
	const int* gradient_surface_rows;
	const int* gradient_surface_columns;
	float* gradient_surface_values;
	
	const int* coupling_domain_rows;
	const int* coupling_domain_columns;
	float* coupling_domain_values;
	const int* coupling_surface_rows;
	const int* coupling_surface_columns;
	float* coupling_surface_values;
	
	const int* surface_flow_coupling_domain_rows;
	const int* surface_flow_coupling_domain_columns;
	float* surface_flow_coupling_domain_values;
	const int* surface_flow_coupling_surface_rows;
	const int* surface_flow_coupling_surface_columns;
	float* surface_flow_coupling_surface_values;
	
	const int* coupling_solid_surface_rows;
	const int* coupling_solid_surface_columns;
	float* coupling_solid_surface_values;
	
	float* mass_single_domain;
	float* mass_single_surface;
	
	float* iq_rhs;
	float* iq_solve_velocity_result;
};

template<typename SurfaceFlowParticleBuffer, MaterialE MaterialType>
__global__ void clear_surface_flow_particle_buffer(const ParticleBuffer<MaterialType> particle_buffer, SurfaceFlowParticleBuffer surface_flow_particle_buffer){
	const uint32_t blockno	  = blockIdx.x;
	const int particle_counts = particle_buffer.particle_bucket_sizes[blockno];
	
	//If we have no particles in the bucket return
	if(particle_counts == 0) {
		return;
	}
	
	for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < particle_counts; particle_id_in_block += static_cast<int>(blockDim.x)) {
		auto particle_bin													= surface_flow_particle_buffer.ch(_0, particle_buffer.bin_offsets[blockno] + particle_id_in_block / config::G_BIN_CAPACITY);
		//mass
		particle_bin.val(_0, particle_id_in_block % config::G_BIN_CAPACITY) = 0.0f;
		//J
		particle_bin.val(_1, particle_id_in_block % config::G_BIN_CAPACITY) = 1.0f;
		//velocity
		particle_bin.val(_2, particle_id_in_block % config::G_BIN_CAPACITY) = 0.0f;
		particle_bin.val(_3, particle_id_in_block % config::G_BIN_CAPACITY) = 0.0f;
		particle_bin.val(_4, particle_id_in_block % config::G_BIN_CAPACITY) = 0.0f;
	}
}

template<MaterialE MaterialTypeFluid>
__forceinline__ __device__ void store_data_neigbours_surface_flow_coupling(const ParticleBuffer<MaterialTypeFluid> particle_buffer_fluid, float* __restrict__ surface_flow_coupling, const float W_velocity, const float W1_pressure, const float contact_area);

template<>
__forceinline__ __device__ void store_data_neigbours_surface_flow_coupling<MaterialE::J_FLUID>(const ParticleBuffer<MaterialE::J_FLUID> particle_buffer_fluid, float* __restrict__ surface_flow_coupling, const float W_velocity, const float W1_pressure, const float contact_area){
	(*surface_flow_coupling) += contact_area * W_velocity * W1_pressure;
}

template<>
__forceinline__ __device__ void store_data_neigbours_surface_flow_coupling<MaterialE::FIXED_COROTATED>(const ParticleBuffer<MaterialE::FIXED_COROTATED> particle_buffer_fluid, float* __restrict__ surface_flow_coupling, const float W_velocity, const float W1_pressure, const float contact_area){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_surface_flow_coupling<MaterialE::SAND>(const ParticleBuffer<MaterialE::SAND> particle_buffer_fluid, float* __restrict__ surface_flow_coupling, const float W_velocity, const float W1_pressure, const float contact_area){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_surface_flow_coupling<MaterialE::NACC>(const ParticleBuffer<MaterialE::NACC> particle_buffer_fluid, float* __restrict__ surface_flow_coupling, const float W_velocity, const float W1_pressure, const float contact_area){
	printf("Material type not supported for coupling as fluid.");
}

template<>
__forceinline__ __device__ void store_data_neigbours_surface_flow_coupling<MaterialE::FIXED_COROTATED_GHOST>(const ParticleBuffer<MaterialE::FIXED_COROTATED_GHOST> particle_buffer_fluid, float* __restrict__ surface_flow_coupling, const float W_velocity, const float W1_pressure, const float contact_area){
	printf("Material type not supported for coupling as fluid.");
}

template<typename Partition, typename Grid, MaterialE MaterialTypeSolid, MaterialE MaterialTypeFluid, typename SurfaceFlowParticleBuffer>
__forceinline__ __device__ void simple_surface_flow_aggregate_data_solid(const ParticleBuffer<MaterialTypeSolid> particle_buffer_solid, const ParticleBuffer<MaterialTypeFluid> particle_buffer_fluid, const ParticleBuffer<MaterialTypeSolid> next_particle_buffer_solid, const ParticleBuffer<MaterialTypeFluid> next_particle_buffer_fluid, const Partition prev_partition, const Grid grid_solid, const Grid grid_fluid, const SurfaceFlowParticleBuffer surface_flow_particle_buffer, const std::array<float, 3>* __restrict__ position_shared, const float* __restrict__ mass_shared, const float* __restrict__ J_shared, const std::array<float, 3>* __restrict__ normal_shared, const SurfacePointType* __restrict__ point_type_shared, const float* __restrict__ contact_area_shared, const float* __restrict__ mass_surface_flow_shared, const float* __restrict__ J_surface_flow_shared, const std::array<float, 3>* __restrict__ velocity_surface_flow_shared, const int particle_offset, const int current_blockno, const ivec3 current_blockid, const ivec3 block_cellid, const int particle_id_in_block
, float* __restrict__ scaling_solid, float* __restrict__ pressure_solid_nominator, float* __restrict__ pressure_solid_denominator, float* __restrict__ mass_solid, float* __restrict__ gradient_solid, float* __restrict__ coupling_solid_domain, float* __restrict__ coupling_solid_surface
, float* __restrict__ coupling_fluid
, float* __restrict__ velocity_surface, float* __restrict__ scaling_surface, float* __restrict__ pressure_surface_nominator, float* __restrict__ pressure_surface_denominator, float* __restrict__ mass_surface, float* __restrict__ gradient_surface, float* __restrict__ coupling_surface, float* __restrict__ surface_flow_coupling_surface
, float* __restrict__ coupling_domain, float* __restrict__ surface_flow_coupling_domain
) {
	const vec3 normal {normal_shared[particle_id_in_block - particle_offset][0], normal_shared[particle_id_in_block - particle_offset][1], normal_shared[particle_id_in_block - particle_offset][2]};
	//const SurfacePointType point_type = point_type_shared[particle_id_in_block - particle_offset];
	const float contact_area = contact_area_shared[particle_id_in_block - particle_offset];
	
	const vec3 pos {position_shared[particle_id_in_block - particle_offset][0], position_shared[particle_id_in_block - particle_offset][1], position_shared[particle_id_in_block - particle_offset][2]};
	const float mass = mass_shared[particle_id_in_block - particle_offset];
	const float J  = J_shared[particle_id_in_block - particle_offset];
	
	const float mass_surface_flow = mass_surface_flow_shared[particle_id_in_block - particle_offset];
	const float J_surface_flow = J_surface_flow_shared[particle_id_in_block - particle_offset];
	const vec3 velocity_surface_flow {velocity_surface_flow_shared[particle_id_in_block - particle_offset][0], velocity_surface_flow_shared[particle_id_in_block - particle_offset][1], velocity_surface_flow_shared[particle_id_in_block - particle_offset][2]};
	
	
	//Calculate surface flow contact area
	//FIXME: Does not change if we change how contact_area of solid is calculated
	const float volume_solid = (mass / particle_buffer_solid.rho) * J_surface_flow;
	const float volume_surface_flow = (mass_surface_flow / particle_buffer_fluid.rho) * J;
	const float contact_area_surface_flow = 2.0f * std::sqrt((volume_solid + volume_surface_flow) / static_cast<float>(M_PI));
	
	//Get position of grid cell
	const ivec3 global_base_index_solid_pressure = get_cell_id<iq::INTERPOLATION_DEGREE_SOLID_PRESSURE>(pos.data_arr(), grid_solid.get_offset());
	const ivec3 global_base_index_solid_velocity = get_cell_id<iq::INTERPOLATION_DEGREE_SOLID_VELOCITY>(pos.data_arr(), grid_solid.get_offset());
	const ivec3 global_base_index_solid_2 = get_cell_id<2>(pos.data_arr(), grid_solid.get_offset());
	
	const ivec3 global_base_index_fluid_velocity = get_cell_id<iq::INTERPOLATION_DEGREE_FLUID_VELOCITY>(pos.data_arr(), grid_fluid.get_offset());//NOTE: Using solid/interface quadrature position
	
	const ivec3 global_base_index_interface_pressure = get_cell_id<iq::INTERPOLATION_DEGREE_INTERFACE_PRESSURE>(pos.data_arr(), grid_solid.get_offset());
	

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
		const std::array<float, iq::INTERPOLATION_DEGREE_SOLID_PRESSURE + 1> current_weight_solid_pressure = bspline_weight<float, iq::INTERPOLATION_DEGREE_SOLID_PRESSURE>(local_pos_solid_pressure[dd]);
		for(int i = 0; i < iq::INTERPOLATION_DEGREE_SOLID_PRESSURE + 1; ++i){
			weight_solid_pressure(dd, i)		  = current_weight_solid_pressure[i];
		}
		for(int i = iq::INTERPOLATION_DEGREE_SOLID_PRESSURE + 1; i < 3; ++i){
			weight_solid_pressure(dd, i)		  = 0.0f;
		}
		
		const std::array<float, iq::INTERPOLATION_DEGREE_SOLID_VELOCITY + 1> current_weight_solid_velocity = bspline_weight<float, iq::INTERPOLATION_DEGREE_SOLID_VELOCITY>(local_pos_solid_velocity[dd]);
		for(int i = 0; i < iq::INTERPOLATION_DEGREE_SOLID_VELOCITY + 1; ++i){
			weight_solid_velocity(dd, i)		  = current_weight_solid_velocity[i];
		}
		for(int i = iq::INTERPOLATION_DEGREE_SOLID_VELOCITY + 1; i < 3; ++i){
			weight_solid_velocity(dd, i)		  = 0.0f;
		}
		
		const std::array<float, iq::INTERPOLATION_DEGREE_SOLID_VELOCITY + 1> current_gradient_weight_solid_velocity = bspline_gradient_weight<float, iq::INTERPOLATION_DEGREE_SOLID_VELOCITY>(local_pos_solid_velocity[dd]);
		for(int i = 0; i < iq::INTERPOLATION_DEGREE_SOLID_VELOCITY + 1; ++i){
			gradient_weight_solid_velocity(dd, i)		  = current_gradient_weight_solid_velocity[i];
		}
		for(int i = iq::INTERPOLATION_DEGREE_SOLID_VELOCITY + 1; i < 3; ++i){
			gradient_weight_solid_velocity(dd, i)		  = 0.0f;
		}
		
		const std::array<float, iq::INTERPOLATION_DEGREE_FLUID_VELOCITY + 1> current_weight_fluid_velocity = bspline_weight<float, iq::INTERPOLATION_DEGREE_FLUID_VELOCITY>(local_pos_fluid_velocity[dd]);
		for(int i = 0; i < iq::INTERPOLATION_DEGREE_FLUID_VELOCITY + 1; ++i){
			weight_fluid_velocity(dd, i)		  = current_weight_fluid_velocity[i];
		}
		for(int i = iq::INTERPOLATION_DEGREE_FLUID_VELOCITY + 1; i < 3; ++i){
			weight_fluid_velocity(dd, i)		  = 0.0f;
		}
		
		const std::array<float, iq::INTERPOLATION_DEGREE_FLUID_VELOCITY + 1> current_gradient_fluid_solid_velocity = bspline_gradient_weight<float, iq::INTERPOLATION_DEGREE_FLUID_VELOCITY>(local_pos_fluid_velocity[dd]);
		for(int i = 0; i < iq::INTERPOLATION_DEGREE_FLUID_VELOCITY + 1; ++i){
			gradient_weight_fluid_velocity(dd, i)		  = current_gradient_fluid_solid_velocity[i];
		}
		for(int i = iq::INTERPOLATION_DEGREE_FLUID_VELOCITY + 1; i < 3; ++i){
			gradient_weight_fluid_velocity(dd, i)		  = 0.0f;
		}
		
		const std::array<float, iq::INTERPOLATION_DEGREE_INTERFACE_PRESSURE + 1> current_weight_interface_pressure = bspline_weight<float, iq::INTERPOLATION_DEGREE_INTERFACE_PRESSURE>(local_pos_interface_pressure[dd]);
		for(int i = 0; i < iq::INTERPOLATION_DEGREE_INTERFACE_PRESSURE + 1; ++i){
			weight_interface_pressure(dd, i)		  = current_weight_interface_pressure[i];
		}
		for(int i = iq::INTERPOLATION_DEGREE_INTERFACE_PRESSURE + 1; i < 3; ++i){
			weight_interface_pressure(dd, i)		  = 0.0f;
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
					//const float J_fluid	 = fetch_particle_buffer_tmp.J;
					
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
	
	//Get near surface particles
	bool has_neighbours_surface_local = false;
	for(int grid_x = -4; grid_x <= 1; ++grid_x){
		for(int grid_y = -4; grid_y <= 1; ++grid_y){
			for(int grid_z = -4; grid_z <= 1; ++grid_z){
				const ivec3 cell_offset {grid_x, grid_y, grid_z};
				const ivec3 current_cellid = global_base_index_solid_2 + cell_offset;
				const ivec3 current_blockid = current_cellid / static_cast<int>(config::G_BLOCKSIZE);
				const int current_blockno_surface = prev_partition.query(current_blockid);
				
				//Skip empty blocks
				if(current_blockno_surface == -1){
					continue;
				}
				
				for(int particle_id_in_block_surface = static_cast<int>(threadIdx.x); particle_id_in_block_surface <  next_particle_buffer_solid.particle_bucket_sizes[current_blockno_surface]; particle_id_in_block_surface += static_cast<int>(blockDim.x)) {
					//Fetch index of the advection source
					int advection_source_blockno_surface;
					int source_pidib_surface;
					{
						//Fetch advection (direction at high bits, particle in in cell at low bits)
						const int advect = next_particle_buffer_solid.blockbuckets[current_blockno_surface * config::G_PARTICLE_NUM_PER_BLOCK + particle_id_in_block_surface];

						//Retrieve the direction (first stripping the particle id by division)
						ivec3 offset;
						dir_components<3>(advect / config::G_PARTICLE_NUM_PER_BLOCK, offset.data_arr());

						//Retrieve the particle id by AND for lower bits
						source_pidib_surface = advect & (config::G_PARTICLE_NUM_PER_BLOCK - 1);

						//Get global index by adding blockid and offset
						const ivec3 global_advection_index = current_blockid + offset;

						//Get block_no from partition
						advection_source_blockno_surface = prev_partition.query(global_advection_index);
					}
					
					auto surface_flow_particle_bin = surface_flow_particle_buffer.ch(_0, particle_buffer_solid.bin_offsets[advection_source_blockno_surface] + source_pidib_surface / config::G_BIN_CAPACITY);
					const int particle_id_in_bin = source_pidib_surface  % config::G_BIN_CAPACITY;

					//Fetch position and determinant of deformation gradient
					FetchParticleBufferDataIntermediate fetch_particle_buffer_tmp = {};
					fetch_particle_buffer_data<MaterialTypeSolid>(particle_buffer_solid, advection_source_blockno_surface, source_pidib_surface, fetch_particle_buffer_tmp);
					vec3 pos_fluid {fetch_particle_buffer_tmp.pos[0], fetch_particle_buffer_tmp.pos[1], fetch_particle_buffer_tmp.pos[2]};
					const float mass_surface = surface_flow_particle_bin.val(_0, particle_id_in_bin);
					//const float J_surface = surface_flow_particle_bin.val(_1, particle_id_in_bin);
					
					const vec3 diff = pos - pos_fluid;
					const float distance = std::sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
					
					if(distance <= 0.5f * config::G_DX && mass_surface > 0.0f){
						has_neighbours_surface_local = true;
						break;
					}
				}
			}
		}
	}
	
	const bool has_neighbours_surface = (__syncthreads_or(has_neighbours_surface_local ? 1 : 0) == 1);
	
	//Store data
	//Note: Weights are 0 if outside of interpolation degree/radius around particles cell
	//Foreach node in the block we add values accoring to particle kernel, also handling all neighbours of the particles cell
	
	for(size_t local_cell_index = 0; local_cell_index < iq::get_thread_count<iq::BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, config::G_BLOCKVOLUME); local_cell_index++){
		const size_t cell_index = iq::get_global_index<iq::BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, local_cell_index);
		const ivec3 local_id {static_cast<int>((cell_index / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE), static_cast<int>((cell_index / config::G_BLOCKSIZE) % config::G_BLOCKSIZE), static_cast<int>(cell_index % config::G_BLOCKSIZE)};
		
		const ivec3 local_offset_pressure = global_base_index_solid_pressure - (block_cellid + local_id);
		const ivec3 local_offset_velocity = global_base_index_solid_velocity - (block_cellid + local_id);
		const ivec3 local_offset_velocity_fluid = global_base_index_fluid_velocity - (block_cellid + local_id);
			
		const ivec3 absolute_local_offset_pressure {std::abs(local_offset_pressure[0]), std::abs(local_offset_pressure[1]), std::abs(local_offset_pressure[2])};
		const ivec3 absolute_local_offset_velocity {std::abs(local_offset_velocity[0]), std::abs(local_offset_velocity[1]), std::abs(local_offset_velocity[2])};
		const ivec3 absolute_local_offset_velocity_fluid {std::abs(local_offset_velocity_fluid[0]), std::abs(local_offset_velocity_fluid[1]), std::abs(local_offset_velocity_fluid[2])};

		//Weight
		const float W_pressure = (absolute_local_offset_pressure[0] < 3 ? weight_solid_pressure(0, absolute_local_offset_pressure[0]) : 0.0f) * (absolute_local_offset_pressure[1] < 3 ? weight_solid_pressure(1, absolute_local_offset_pressure[1]) : 0.0f) * (absolute_local_offset_pressure[2] < 3 ? weight_solid_pressure(2, absolute_local_offset_pressure[2]) : 0.0f);
		const float W_velocity = (absolute_local_offset_velocity[0] < 3 ? weight_solid_velocity(0, absolute_local_offset_velocity[0]) : 0.0f) * (absolute_local_offset_velocity[1] < 3 ? weight_solid_velocity(1, absolute_local_offset_velocity[1]) : 0.0f) * (absolute_local_offset_velocity[2] < 3 ? weight_solid_velocity(2, absolute_local_offset_velocity[2]) : 0.0f);
		const float W_velocity_surface = (absolute_local_offset_velocity_fluid[0] < 3 ? weight_fluid_velocity(0, absolute_local_offset_velocity_fluid[0]) : 0.0f) * (absolute_local_offset_velocity_fluid[1] < 3 ? weight_fluid_velocity(1, absolute_local_offset_velocity_fluid[1]) : 0.0f) * (absolute_local_offset_velocity_fluid[2] < 3 ? weight_fluid_velocity(2, absolute_local_offset_velocity_fluid[2]) : 0.0f);
		
		float* current_scaling_solid = &(scaling_solid[local_cell_index]);
		float* current_pressure_solid_nominator = &(pressure_solid_nominator[local_cell_index]);
		float* current_pressure_solid_denominator = &(pressure_solid_denominator[local_cell_index]);
		
		float* current_scaling_surface = &(scaling_surface[local_cell_index]);
		float* current_pressure_surface_nominator = &(pressure_surface_nominator[local_cell_index]);
		float* current_pressure_surface_denominator = &(pressure_surface_denominator[local_cell_index]);
		
		iq::store_data_solid(particle_buffer_solid, current_scaling_solid, current_pressure_solid_nominator, current_pressure_solid_denominator, W_pressure, W_velocity, mass, J);
		if(mass_surface_flow > 0.0f){
			iq::store_data_fluid(particle_buffer_fluid, current_scaling_surface, current_pressure_surface_nominator, current_pressure_surface_denominator, W_pressure, W_velocity_surface, mass_surface_flow, J_surface_flow);
		}
	}
	
	for(size_t local_cell_index = 0; local_cell_index < iq::get_thread_count<iq::BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, 3 * config::G_BLOCKVOLUME); local_cell_index++){
		const size_t cell_index = iq::get_global_index<iq::BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, local_cell_index) / 3;
		const size_t alpha = iq::get_global_index<iq::BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, local_cell_index) % 3;
		const ivec3 local_id {static_cast<int>((cell_index / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE), static_cast<int>((cell_index / config::G_BLOCKSIZE) % config::G_BLOCKSIZE), static_cast<int>(cell_index % config::G_BLOCKSIZE)};
		
		const ivec3 local_offset_velocity = global_base_index_solid_velocity - (block_cellid + local_id);
		const ivec3 local_offset_velocity_fluid = global_base_index_fluid_velocity - (block_cellid + local_id);
		
		const ivec3 absolute_local_offset_velocity {std::abs(local_offset_velocity[0]), std::abs(local_offset_velocity[1]), std::abs(local_offset_velocity[2])};
		const ivec3 absolute_local_offset_velocity_fluid {std::abs(local_offset_velocity_fluid[0]), std::abs(local_offset_velocity_fluid[1]), std::abs(local_offset_velocity_fluid[2])};

		//Weight
		const float W_velocity = (absolute_local_offset_velocity[0] < 3 ? weight_solid_velocity(0, absolute_local_offset_velocity[0]) : 0.0f) * (absolute_local_offset_velocity[1] < 3 ? weight_solid_velocity(1, absolute_local_offset_velocity[1]) : 0.0f) * (absolute_local_offset_velocity[2] < 3 ? weight_solid_velocity(2, absolute_local_offset_velocity[2]) : 0.0f);
		const float W_velocity_surface = (absolute_local_offset_velocity_fluid[0] < 3 ? weight_fluid_velocity(0, absolute_local_offset_velocity_fluid[0]) : 0.0f) * (absolute_local_offset_velocity_fluid[1] < 3 ? weight_fluid_velocity(1, absolute_local_offset_velocity_fluid[1]) : 0.0f) * (absolute_local_offset_velocity_fluid[2] < 3 ? weight_fluid_velocity(2, absolute_local_offset_velocity_fluid[2]) : 0.0f);
		
		mass_solid[local_cell_index] += mass * W_velocity;
		mass_surface[local_cell_index] += mass_surface_flow * W_velocity_surface;
		
		velocity_surface[local_cell_index] += mass_surface_flow * velocity_surface_flow[alpha] * W_velocity_surface;
	}
	
	for(size_t local_cell_index = 0; local_cell_index < iq::get_thread_count<iq::BLOCK_SIZE, (3 * iq::NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, 3 * iq::NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME); local_cell_index++){
		const size_t cell_index = iq::get_global_index<iq::BLOCK_SIZE, (3 * iq::NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, local_cell_index) / (3 * iq::NUM_COLUMNS_PER_BLOCK);
		const size_t column = (iq::get_global_index<iq::BLOCK_SIZE, (3 * iq::NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, local_cell_index) / 3) % iq::NUM_COLUMNS_PER_BLOCK;
		const size_t alpha = iq::get_global_index<iq::BLOCK_SIZE, (3 * iq::NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, local_cell_index) % 3;
		const ivec3 local_id {static_cast<int>((cell_index / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE), static_cast<int>((cell_index / config::G_BLOCKSIZE) % config::G_BLOCKSIZE), static_cast<int>(cell_index % config::G_BLOCKSIZE)};
		const ivec3 neighbour_local_id = ivec3(static_cast<int>((column / ((2 * iq::INTERPOLATION_DEGREE_MAX + 1) * (2 * iq::INTERPOLATION_DEGREE_MAX + 1))) % (2 * iq::INTERPOLATION_DEGREE_MAX + 1)), static_cast<int>((column / (2 * iq::INTERPOLATION_DEGREE_MAX + 1)) % (2 * iq::INTERPOLATION_DEGREE_MAX + 1)), static_cast<int>(column % (2 * iq::INTERPOLATION_DEGREE_MAX + 1))) - ivec3(static_cast<int>(iq::INTERPOLATION_DEGREE_MAX), static_cast<int>(iq::INTERPOLATION_DEGREE_MAX), static_cast<int>(iq::INTERPOLATION_DEGREE_MAX));
			
		const ivec3 local_offset_velocity = global_base_index_solid_velocity - (block_cellid + local_id);
		const ivec3 local_offset_velocity_fluid = global_base_index_fluid_velocity - (block_cellid + local_id);
		const ivec3 neighbour_local_offset_pressure = global_base_index_solid_pressure - (block_cellid + local_id + neighbour_local_id);
		
		const ivec3 absolute_local_offset_velocity {std::abs(local_offset_velocity[0]), std::abs(local_offset_velocity[1]), std::abs(local_offset_velocity[2])};
		const ivec3 absolute_local_offset_velocity_fluid {std::abs(local_offset_velocity_fluid[0]), std::abs(local_offset_velocity_fluid[1]), std::abs(local_offset_velocity_fluid[2])};
		const ivec3 neighbour_absolute_local_offset {std::abs(neighbour_local_offset_pressure[0]), std::abs(neighbour_local_offset_pressure[1]), std::abs(neighbour_local_offset_pressure[2])};
																

		//Weight
		const float delta_W_velocity = ((alpha == 0 ? (absolute_local_offset_velocity[0] < 3 ? gradient_weight_solid_velocity(0, absolute_local_offset_velocity[0]) : 0.0f) : (absolute_local_offset_velocity[0] < 3 ? weight_solid_velocity(0, absolute_local_offset_velocity[0]) : 0.0f)) * (alpha == 1 ? (absolute_local_offset_velocity[1] < 3 ? gradient_weight_solid_velocity(1, absolute_local_offset_velocity[1]) : 0.0f) : (absolute_local_offset_velocity[1] < 3 ? weight_solid_velocity(1, absolute_local_offset_velocity[1]) : 0.0f)) * (alpha == 2 ? (absolute_local_offset_velocity[2] < 3 ? gradient_weight_solid_velocity(2, absolute_local_offset_velocity[2]) : 0.0f) : (absolute_local_offset_velocity[2] < 3 ? weight_solid_velocity(2, absolute_local_offset_velocity[2]) : 0.0f))) * config::G_DX_INV;
		const float delta_W_velocity_surface = ((alpha == 0 ? (absolute_local_offset_velocity_fluid[0] < 3 ? gradient_weight_fluid_velocity(0, absolute_local_offset_velocity_fluid[0]) : 0.0f) : (absolute_local_offset_velocity_fluid[0] < 3 ? weight_fluid_velocity(0, absolute_local_offset_velocity_fluid[0]) : 0.0f)) * (alpha == 1 ? (absolute_local_offset_velocity_fluid[1] < 3 ? gradient_weight_fluid_velocity(1, absolute_local_offset_velocity_fluid[1]) : 0.0f) : (absolute_local_offset_velocity_fluid[1] < 3 ? weight_fluid_velocity(1, absolute_local_offset_velocity_fluid[1]) : 0.0f)) * (alpha == 2 ? (absolute_local_offset_velocity_fluid[2] < 3 ? gradient_weight_solid_velocity(2, absolute_local_offset_velocity_fluid[2]) : 0.0f) : (absolute_local_offset_velocity_fluid[2] < 3 ? weight_fluid_velocity(2, absolute_local_offset_velocity_fluid[2]) : 0.0f))) * config::G_DX_INV;
		const float W1_pressure = (neighbour_absolute_local_offset[0] < 3 ? weight_solid_pressure(0, neighbour_absolute_local_offset[0]) : 0.0f) * (neighbour_absolute_local_offset[1] < 3 ? weight_solid_pressure(1, neighbour_absolute_local_offset[1]) : 0.0f) * (neighbour_absolute_local_offset[2] < 3 ? weight_solid_pressure(2, neighbour_absolute_local_offset[2]) : 0.0f);
										
		float* current_gradient_solid = &(gradient_solid[local_cell_index]);
		
		float* current_gradient_surface = &(gradient_surface[local_cell_index]);

		iq::store_data_neigbours_solid(particle_buffer_solid, current_gradient_solid, W1_pressure, delta_W_velocity, mass, J);
		if(mass_surface_flow > 0.0f){
			iq::store_data_neigbours_fluid(particle_buffer_fluid, current_gradient_surface, nullptr, 0.0f, W1_pressure, delta_W_velocity_surface, mass_surface_flow, J_surface_flow);
		}
	}
	
	
	
	//Store data
	//Note: Weights are 0 if outside of interpolation degree/radius around particles cell
	//Foreach node in the block we add values accoring to particle kernel, also handling all neighbours of the particles cell
	
	for(size_t local_cell_index = 0; local_cell_index < iq::get_thread_count<iq::BLOCK_SIZE, (3 * iq::NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, 3 * iq::NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME); local_cell_index++){
		const size_t cell_index = iq::get_global_index<iq::BLOCK_SIZE, (3 * iq::NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, local_cell_index) / (3 * iq::NUM_COLUMNS_PER_BLOCK);
		const size_t column = (iq::get_global_index<iq::BLOCK_SIZE, (3 * iq::NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, local_cell_index) / 3) % iq::NUM_COLUMNS_PER_BLOCK;
		const size_t alpha = iq::get_global_index<iq::BLOCK_SIZE, (3 * iq::NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, local_cell_index) % 3;
		const ivec3 local_id {static_cast<int>((cell_index / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE), static_cast<int>((cell_index / config::G_BLOCKSIZE) % config::G_BLOCKSIZE), static_cast<int>(cell_index % config::G_BLOCKSIZE)};
		const ivec3 neighbour_local_id = ivec3(static_cast<int>((column / ((2 * iq::INTERPOLATION_DEGREE_MAX + 1) * (2 * iq::INTERPOLATION_DEGREE_MAX + 1))) % (2 * iq::INTERPOLATION_DEGREE_MAX + 1)), static_cast<int>((column / (2 * iq::INTERPOLATION_DEGREE_MAX + 1)) % (2 * iq::INTERPOLATION_DEGREE_MAX + 1)), static_cast<int>(column % (2 * iq::INTERPOLATION_DEGREE_MAX + 1))) - ivec3(static_cast<int>(iq::INTERPOLATION_DEGREE_MAX), static_cast<int>(iq::INTERPOLATION_DEGREE_MAX), static_cast<int>(iq::INTERPOLATION_DEGREE_MAX));
			
		const ivec3 local_offset_velocity_solid = global_base_index_solid_velocity - (block_cellid + local_id);
		const ivec3 local_offset_velocity_fluid = global_base_index_fluid_velocity - (block_cellid + local_id);
		const ivec3 neighbour_local_offset_pressure_interface = global_base_index_interface_pressure - (block_cellid + local_id + neighbour_local_id);
		
		const ivec3 absolute_local_offset_velocity_solid {std::abs(local_offset_velocity_solid[0]), std::abs(local_offset_velocity_solid[1]), std::abs(local_offset_velocity_solid[2])};
		const ivec3 absolute_local_offset_velocity_fluid {std::abs(local_offset_velocity_fluid[0]), std::abs(local_offset_velocity_fluid[1]), std::abs(local_offset_velocity_fluid[2])};
		const ivec3 neighbour_absolute_local_offset_interface {std::abs(neighbour_local_offset_pressure_interface[0]), std::abs(neighbour_local_offset_pressure_interface[1]), std::abs(neighbour_local_offset_pressure_interface[2])};
		
		//Weight
		const float W_velocity_solid = (absolute_local_offset_velocity_solid[0] < 3 ? weight_solid_velocity(0, absolute_local_offset_velocity_solid[0]) : 0.0f) * (absolute_local_offset_velocity_solid[1] < 3 ? weight_solid_velocity(1, absolute_local_offset_velocity_solid[1]) : 0.0f) * (absolute_local_offset_velocity_solid[2] < 3 ? weight_solid_velocity(2, absolute_local_offset_velocity_solid[2]) : 0.0f);	
		const float W_velocity_fluid = (absolute_local_offset_velocity_fluid[0] < 3 ? weight_fluid_velocity(0, absolute_local_offset_velocity_fluid[0]) : 0.0f) * (absolute_local_offset_velocity_fluid[1] < 3 ? weight_fluid_velocity(1, absolute_local_offset_velocity_fluid[1]) : 0.0f) * (absolute_local_offset_velocity_fluid[2] < 3 ? weight_fluid_velocity(2, absolute_local_offset_velocity_fluid[2]) : 0.0f);
		const float W1_pressure_interface = (neighbour_absolute_local_offset_interface[0] < 3 ? weight_interface_pressure(0, neighbour_absolute_local_offset_interface[0]) : 0.0f) * (neighbour_absolute_local_offset_interface[1] < 3 ? weight_interface_pressure(1, neighbour_absolute_local_offset_interface[1]) : 0.0f) * (neighbour_absolute_local_offset_interface[2] < 3 ? weight_interface_pressure(2, neighbour_absolute_local_offset_interface[2]) : 0.0f);					
				
		float* current_coupling_solid_domain = &(coupling_solid_domain[local_cell_index]);
		float* current_coupling_solid_surface = &(coupling_solid_surface[local_cell_index]);
		float* current_coupling_fluid = &(coupling_fluid[local_cell_index]);
		
		float* current_coupling_domain = &(coupling_domain[local_cell_index]);
		float* current_coupling_surface = &(coupling_surface[local_cell_index]);
		
		float* current_surface_flow_coupling_domain = &(surface_flow_coupling_domain[local_cell_index]);
		float* current_surface_flow_coupling_surface = &(surface_flow_coupling_surface[local_cell_index]);

		//Only proceed if we have an interface
		if(has_neighbours){
			iq::store_data_neigbours_coupling_solid(particle_buffer_solid, current_coupling_solid_domain, W_velocity_solid, W1_pressure_interface, contact_area, normal[alpha]);
			iq::store_data_neigbours_coupling_fluid(particle_buffer_fluid, current_coupling_fluid, W_velocity_fluid, W1_pressure_interface, contact_area, normal[alpha]);
			iq::store_data_neigbours_coupling_fluid(particle_buffer_fluid, current_coupling_domain, W_velocity_fluid, W1_pressure_interface, contact_area, normal[alpha]);
		}
		
		if(has_neighbours_surface){
			iq::store_data_neigbours_coupling_solid(particle_buffer_solid, current_coupling_solid_surface, W_velocity_solid, W1_pressure_interface, contact_area, normal[alpha]);
			iq::store_data_neigbours_coupling_fluid(particle_buffer_fluid, current_coupling_surface, W_velocity_fluid, W1_pressure_interface, contact_area, normal[alpha]);
		}
		
		store_data_neigbours_surface_flow_coupling(particle_buffer_fluid, current_surface_flow_coupling_domain, W_velocity_fluid, W1_pressure_interface, contact_area_surface_flow);
		store_data_neigbours_surface_flow_coupling(particle_buffer_fluid, current_surface_flow_coupling_surface, W_velocity_fluid, W1_pressure_interface, contact_area_surface_flow);
	}
}

template<typename Partition, typename Grid, MaterialE MaterialTypeFluid>
__forceinline__ __device__ void simple_surface_flow_aggregate_data_fluid(const ParticleBuffer<MaterialTypeFluid> particle_buffer_fluid, const ParticleBuffer<MaterialTypeFluid> next_particle_buffer_fluid, const Partition prev_partition, const Grid grid_solid, const Grid grid_fluid, const std::array<float, 3>* __restrict__ position_shared, const float* __restrict__ mass_shared, const float* __restrict__ J_shared, const int particle_offset, const int current_blockno, const ivec3 current_blockid, const ivec3 block_cellid, const int particle_id_in_block, float* __restrict__ scaling_fluid, float* __restrict__ pressure_fluid_nominator, float* __restrict__ pressure_fluid_denominator, float* __restrict__ mass_fluid, float* __restrict__ gradient_fluid) {
	const vec3 pos {position_shared[particle_id_in_block - particle_offset][0], position_shared[particle_id_in_block - particle_offset][1], position_shared[particle_id_in_block - particle_offset][2]};
	const float mass = mass_shared[particle_id_in_block - particle_offset];
	const float J  = J_shared[particle_id_in_block - particle_offset];
	
	//Get position of grid cell
	const ivec3 global_base_index_solid_pressure = get_cell_id<iq::INTERPOLATION_DEGREE_FLUID_PRESSURE>(pos.data_arr(), grid_solid.get_offset());
	const ivec3 global_base_index_fluid_velocity = get_cell_id<iq::INTERPOLATION_DEGREE_FLUID_VELOCITY>(pos.data_arr(), grid_fluid.get_offset());
	
	//Get position relative to grid cell
	const vec3 local_pos_solid_pressure = pos - (global_base_index_solid_pressure + vec3(grid_solid.get_offset()[0], grid_solid.get_offset()[1], grid_solid.get_offset()[2])) * config::G_DX;
	const vec3 local_pos_fluid_velocity = pos - (global_base_index_fluid_velocity + vec3(grid_fluid.get_offset()[0], grid_fluid.get_offset()[1], grid_fluid.get_offset()[2])) * config::G_DX;

	//Calculate weights
	vec3x3 weight_solid_pressure;
	vec3x3 weight_fluid_velocity;
	vec3x3 gradient_weight_fluid_velocity;
	
	#pragma unroll 3
	for(int dd = 0; dd < 3; ++dd) {
		const std::array<float, iq::INTERPOLATION_DEGREE_FLUID_PRESSURE + 1> current_weight_solid_pressure = bspline_weight<float, iq::INTERPOLATION_DEGREE_FLUID_PRESSURE>(local_pos_solid_pressure[dd]);
		for(int i = 0; i < iq::INTERPOLATION_DEGREE_FLUID_PRESSURE + 1; ++i){
			weight_solid_pressure(dd, i)		  = current_weight_solid_pressure[i];
		}
		for(int i = iq::INTERPOLATION_DEGREE_FLUID_PRESSURE + 1; i < 3; ++i){
			weight_solid_pressure(dd, i)		  = 0.0f;
		}
		
		const std::array<float, iq::INTERPOLATION_DEGREE_FLUID_VELOCITY + 1> current_weight_fluid_velocity = bspline_weight<float, iq::INTERPOLATION_DEGREE_FLUID_VELOCITY>(local_pos_fluid_velocity[dd]);
		for(int i = 0; i < iq::INTERPOLATION_DEGREE_FLUID_VELOCITY + 1; ++i){
			weight_fluid_velocity(dd, i)		  = current_weight_fluid_velocity[i];
		}
		for(int i = iq::INTERPOLATION_DEGREE_FLUID_VELOCITY + 1; i < 3; ++i){
			weight_fluid_velocity(dd, i)		  = 0.0f;
		}
		
		const std::array<float, iq::INTERPOLATION_DEGREE_FLUID_VELOCITY + 1> current_gradient_weight_fluid_velocity = bspline_gradient_weight<float, iq::INTERPOLATION_DEGREE_FLUID_VELOCITY>(local_pos_fluid_velocity[dd]);
		for(int i = 0; i < iq::INTERPOLATION_DEGREE_FLUID_VELOCITY + 1; ++i){
			gradient_weight_fluid_velocity(dd, i)		  = current_gradient_weight_fluid_velocity[i];
		}
		for(int i = iq::INTERPOLATION_DEGREE_FLUID_VELOCITY + 1; i < 3; ++i){
			gradient_weight_fluid_velocity(dd, i)		  = 0.0f;
		}
	}
	
	//Store data
	//Note: Weights are 0 if outside of interpolation degree/radius around particles cell
	//Foreach node in the block we add values accoring to particle kernel, also handling all neighbours of the particles cell
	
	for(size_t local_cell_index = 0; local_cell_index < iq::get_thread_count<iq::BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, config::G_BLOCKVOLUME); local_cell_index++){
		const size_t cell_index = iq::get_global_index<iq::BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, local_cell_index);
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
		
		iq::store_data_fluid(particle_buffer_fluid, current_scaling_fluid, current_pressure_fluid_nominator, current_pressure_fluid_denominator, W_pressure, W_velocity, mass, J);
	}
	
	for(size_t local_cell_index = 0; local_cell_index < iq::get_thread_count<iq::BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, 3 * config::G_BLOCKVOLUME); local_cell_index++){
		const size_t cell_index = iq::get_global_index<iq::BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, local_cell_index) / 3;
		const size_t alpha = iq::get_global_index<iq::BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, local_cell_index) % 3;
		const ivec3 local_id {static_cast<int>((cell_index / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE), static_cast<int>((cell_index / config::G_BLOCKSIZE) % config::G_BLOCKSIZE), static_cast<int>(cell_index % config::G_BLOCKSIZE)};
		
		const ivec3 local_offset_velocity_fluid = global_base_index_fluid_velocity - (block_cellid + local_id);
		
		const ivec3 absolute_local_offset_velocity_fluid {std::abs(local_offset_velocity_fluid[0]), std::abs(local_offset_velocity_fluid[1]), std::abs(local_offset_velocity_fluid[2])};

		//Weight
		const float W_velocity = (absolute_local_offset_velocity_fluid[0] < 3 ? weight_fluid_velocity(0, absolute_local_offset_velocity_fluid[0]) : 0.0f) * (absolute_local_offset_velocity_fluid[1] < 3 ? weight_fluid_velocity(1, absolute_local_offset_velocity_fluid[1]) : 0.0f) * (absolute_local_offset_velocity_fluid[2] < 3 ? weight_fluid_velocity(2, absolute_local_offset_velocity_fluid[2]) : 0.0f);
				
		
		mass_fluid[local_cell_index] += mass * W_velocity;
	}
	
	for(size_t local_cell_index = 0; local_cell_index < iq::get_thread_count<iq::BLOCK_SIZE, (3 * iq::NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, 3 * iq::NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME); local_cell_index++){
		const size_t cell_index = iq::get_global_index<iq::BLOCK_SIZE, (3 * iq::NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, local_cell_index) / (3 * iq::NUM_COLUMNS_PER_BLOCK);
		const size_t column = (iq::get_global_index<iq::BLOCK_SIZE, (3 * iq::NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, local_cell_index) / 3) % iq::NUM_COLUMNS_PER_BLOCK;
		const size_t alpha = iq::get_global_index<iq::BLOCK_SIZE, (3 * iq::NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, local_cell_index) % 3;
		const ivec3 local_id {static_cast<int>((cell_index / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE), static_cast<int>((cell_index / config::G_BLOCKSIZE) % config::G_BLOCKSIZE), static_cast<int>(cell_index % config::G_BLOCKSIZE)};
		const ivec3 neighbour_local_id = ivec3(static_cast<int>((column / ((2 * iq::INTERPOLATION_DEGREE_MAX + 1) * (2 * iq::INTERPOLATION_DEGREE_MAX + 1))) % (2 * iq::INTERPOLATION_DEGREE_MAX + 1)), static_cast<int>((column / (2 * iq::INTERPOLATION_DEGREE_MAX + 1)) % (2 * iq::INTERPOLATION_DEGREE_MAX + 1)), static_cast<int>(column % (2 * iq::INTERPOLATION_DEGREE_MAX + 1))) - ivec3(static_cast<int>(iq::INTERPOLATION_DEGREE_MAX), static_cast<int>(iq::INTERPOLATION_DEGREE_MAX), static_cast<int>(iq::INTERPOLATION_DEGREE_MAX));
			
		const ivec3 local_offset_velocity_fluid = global_base_index_fluid_velocity - (block_cellid + local_id);
		const ivec3 neighbour_local_offset_pressure = global_base_index_solid_pressure - (block_cellid + local_id + neighbour_local_id);
		
		const ivec3 absolute_local_offset_velocity_fluid {std::abs(local_offset_velocity_fluid[0]), std::abs(local_offset_velocity_fluid[1]), std::abs(local_offset_velocity_fluid[2])};
		const ivec3 neighbour_absolute_local_offset {std::abs(neighbour_local_offset_pressure[0]), std::abs(neighbour_local_offset_pressure[1]), std::abs(neighbour_local_offset_pressure[2])};												

		//Weight
		const float delta_W_velocity = ((alpha == 0 ? (absolute_local_offset_velocity_fluid[0] < 3 ? gradient_weight_fluid_velocity(0, absolute_local_offset_velocity_fluid[0]) : 0.0f) : (absolute_local_offset_velocity_fluid[0] < 3 ? weight_fluid_velocity(0, absolute_local_offset_velocity_fluid[0]) : 0.0f)) * (alpha == 1 ? (absolute_local_offset_velocity_fluid[1] < 3 ? gradient_weight_fluid_velocity(1, absolute_local_offset_velocity_fluid[1]) : 0.0f) : (absolute_local_offset_velocity_fluid[1] < 3 ? weight_fluid_velocity(1, absolute_local_offset_velocity_fluid[1]) : 0.0f)) * (alpha == 2 ? (absolute_local_offset_velocity_fluid[2] < 3 ? gradient_weight_fluid_velocity(2, absolute_local_offset_velocity_fluid[2]) : 0.0f) : (absolute_local_offset_velocity_fluid[2] < 3 ? weight_fluid_velocity(2, absolute_local_offset_velocity_fluid[2]) : 0.0f))) * config::G_DX_INV;
		const float W_velocity = (absolute_local_offset_velocity_fluid[0] < 3 ? weight_fluid_velocity(0, absolute_local_offset_velocity_fluid[0]) : 0.0f) * (absolute_local_offset_velocity_fluid[1] < 3 ? weight_fluid_velocity(1, absolute_local_offset_velocity_fluid[1]) : 0.0f) * (absolute_local_offset_velocity_fluid[2] < 3 ? weight_fluid_velocity(2, absolute_local_offset_velocity_fluid[2]) : 0.0f);
		const float W1_pressure = (neighbour_absolute_local_offset[0] < 3 ? weight_solid_pressure(0, neighbour_absolute_local_offset[0]) : 0.0f) * (neighbour_absolute_local_offset[1] < 3 ? weight_solid_pressure(1, neighbour_absolute_local_offset[1]) : 0.0f) * (neighbour_absolute_local_offset[2] < 3 ? weight_solid_pressure(2, neighbour_absolute_local_offset[2]) : 0.0f);
									
		float* current_gradient_fluid = &(gradient_fluid[local_cell_index]);
		
		iq::store_data_neigbours_fluid(particle_buffer_fluid, current_gradient_fluid, nullptr, W_velocity, W1_pressure, delta_W_velocity, mass, J);
	}
}

//TODO: Directly store into matrices, notinto local memory
template<typename Partition, typename Grid, MaterialE MaterialTypeSolid, MaterialE MaterialTypeFluid, typename SurfaceFlowParticleBuffer>
__global__ void simple_surface_flow_create_iq_system(const uint32_t num_blocks, Duration dt, const ParticleBuffer<MaterialTypeSolid> particle_buffer_solid, const ParticleBuffer<MaterialTypeFluid> particle_buffer_fluid, const ParticleBuffer<MaterialTypeSolid> next_particle_buffer_solid, const ParticleBuffer<MaterialTypeFluid> next_particle_buffer_fluid, const Partition prev_partition, const Partition partition, const Grid grid_solid, const Grid grid_fluid, const SurfaceParticleBuffer surface_particle_buffer_solid, const SurfaceParticleBuffer surface_particle_buffer_fluid, const SurfaceFlowParticleBuffer surface_flow_particle_buffer, SimpleSurfaceFlowIQCreatePointers iq_pointers) {
	//Particles with offset [-2, 0] can lie within cell (due to storing with interpolation degree 2 wich results in offset of 2); Interolation degree may offset positions so we need [-2, 2] for all interpolation positions in our cell. Then wee also need neighbour positions so we get [-4, 4];
	constexpr size_t KERNEL_SIZE = 2 * iq::INTERPOLATION_DEGREE_MAX + 5 + 1;//Plus one for both sides being inclusive
	constexpr size_t KERNEL_OFFSET = iq::INTERPOLATION_DEGREE_MAX + 2;
	
	//Both positive, both rounded up. Start will later be negated
	constexpr size_t KERNEL_START_BLOCK = (KERNEL_SIZE - KERNEL_OFFSET - 1 + config::G_BLOCKSIZE - 1) / config::G_BLOCKSIZE;
	constexpr size_t KERNEL_END_BLOCK = (KERNEL_OFFSET + config::G_BLOCKSIZE - 1) / config::G_BLOCKSIZE;
	
	const size_t base_row = iq::NUM_ROWS_PER_BLOCK * blockIdx.x;

	float mass_solid_local[(3 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE];
	float gradient_solid_local[(3 * iq::NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE];
	float scaling_solid_local[(1 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE];
	float velocity_solid_local[(3 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE];
	float pressure_solid_nominator_local[(1 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE];
	float pressure_solid_denominator_local[(1 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE];
	
	float mass_domain_local[(3 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE];
	float gradient_domain_local[(3 * iq::NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE];
	float scaling_domain_local[(1 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE];
	float velocity_domain_local[(3 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE];
	float pressure_domain_nominator_local[(1 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE];
	float pressure_domain_denominator_local[(1 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE];
	
	float mass_surface_local[(3 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE];
	float gradient_surface_local[(3 * iq::NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE];
	float scaling_surface_local[(1 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE];
	float velocity_surface_local[(3 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE];
	float pressure_surface_nominator_local[(1 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE];
	float pressure_surface_denominator_local[(1 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE];
	
	float coupling_solid_domain_local[(3 * iq::NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE];
	float coupling_fluid_local[(3 * iq::NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE];
	float coupling_domain_local[(3 * iq::NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE];
	float coupling_surface_local[(3 * iq::NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE];
	
	float coupling_solid_surface_local[(3 * iq::NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE];
	
	float surface_flow_coupling_domain_local[(3 * iq::NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE];
	float surface_flow_coupling_surface_local[(3 * iq::NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE];

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
	for(size_t i = 0; i < (1 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE; ++i){
		scaling_solid_local[i] = 0.0f;
		pressure_solid_nominator_local[i] = 0.0f;
		pressure_solid_denominator_local[i] = 0.0f;
		
		scaling_domain_local[i] = 0.0f;
		pressure_domain_nominator_local[i] = 0.0f;
		pressure_domain_denominator_local[i] = 0.0f;
		
		scaling_surface_local[i] = 0.0f;
		pressure_surface_nominator_local[i] = 0.0f;
		pressure_surface_denominator_local[i] = 0.0f;
	}
	for(size_t i = 0; i < (3 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE; ++i){
		mass_solid_local[i] = 0.0f;
		mass_domain_local[i] = 0.0f;
		mass_surface_local[i] = 0.0f;
		if(iq::get_global_index<iq::BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, i) < 3 * config::G_BLOCKVOLUME){
			if((iq::get_global_index<iq::BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, i) % 3) == 0) {
				velocity_solid_local[i] = grid_block_solid.val_1d(_1, iq::get_global_index<iq::BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, i) / 3);
				velocity_domain_local[i] = grid_block_fluid.val_1d(_1, iq::get_global_index<iq::BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, i) / 3);
			} else if((iq::get_global_index<iq::BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, i) % 3) == 1) {
				velocity_solid_local[i] = grid_block_solid.val_1d(_2, iq::get_global_index<iq::BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, i) / 3);
				velocity_domain_local[i] = grid_block_fluid.val_1d(_2, iq::get_global_index<iq::BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, i) / 3);
			} else {
				velocity_solid_local[i] = grid_block_solid.val_1d(_3, iq::get_global_index<iq::BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, i) / 3);
				velocity_domain_local[i] = grid_block_fluid.val_1d(_3, iq::get_global_index<iq::BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, i) / 3);
			}
			
			velocity_surface_local[i] = 0.0f;
		}
	}		
	for(size_t i = 0; i < (3 * iq::NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE; ++i){				
		gradient_solid_local[i] = 0.0f;
		gradient_domain_local[i] = 0.0f;
		gradient_surface_local[i] = 0.0f;
		
		coupling_solid_domain_local[i] = 0.0f;
		coupling_fluid_local[i] = 0.0f;
		coupling_domain_local[i] = 0.0f;
		coupling_surface_local[i] = 0.0f;
		
		coupling_solid_surface_local[i] = 0.0f;
		
		surface_flow_coupling_domain_local[i] = 0.0f;
		surface_flow_coupling_surface_local[i] = 0.0f;
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
					
					for(int particle_offset = 0; particle_offset < next_particle_buffer_solid.particle_bucket_sizes[current_blockno]; particle_offset += static_cast<int>(SIMPLE_SURFACE_FLOW_MAX_SHARED_PARTICLE_SOLID)){
						__shared__ std::array<float, 3> position_shared[SIMPLE_SURFACE_FLOW_MAX_SHARED_PARTICLE_SOLID];
						__shared__ float mass_shared[SIMPLE_SURFACE_FLOW_MAX_SHARED_PARTICLE_SOLID];
						__shared__ float J_shared[SIMPLE_SURFACE_FLOW_MAX_SHARED_PARTICLE_SOLID];
						
						__shared__ std::array<float, 3> normal_shared[SIMPLE_SURFACE_FLOW_MAX_SHARED_PARTICLE_SOLID];
						__shared__ SurfacePointType point_type_shared[SIMPLE_SURFACE_FLOW_MAX_SHARED_PARTICLE_SOLID];
						__shared__ float contact_area_shared[SIMPLE_SURFACE_FLOW_MAX_SHARED_PARTICLE_SOLID];
						
						__shared__ float mass_surface_flow_shared[SIMPLE_SURFACE_FLOW_MAX_SHARED_PARTICLE_SOLID];
						__shared__ float J_surface_flow_shared[SIMPLE_SURFACE_FLOW_MAX_SHARED_PARTICLE_SOLID];
						__shared__ std::array<float, 3> velocity_surface_flow_shared[SIMPLE_SURFACE_FLOW_MAX_SHARED_PARTICLE_SOLID];
						
						for(int particle_id_in_block = particle_offset + static_cast<int>(threadIdx.x); particle_id_in_block < next_particle_buffer_solid.particle_bucket_sizes[current_blockno] && (particle_id_in_block - particle_offset) < SIMPLE_SURFACE_FLOW_MAX_SHARED_PARTICLE_SOLID; particle_id_in_block += static_cast<int>(blockDim.x)) {
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
							auto surface_flow_particle_bin = surface_flow_particle_buffer.ch(_0, particle_buffer_solid.bin_offsets[advection_source_blockno] + source_pidib / config::G_BIN_CAPACITY);
							const int particle_id_in_bin = source_pidib  % config::G_BIN_CAPACITY;
							
							normal_shared[particle_id_in_block - particle_offset][0] = surface_particle_bin.val(_1, particle_id_in_bin);
							normal_shared[particle_id_in_block - particle_offset][1] = surface_particle_bin.val(_2, particle_id_in_bin);
							normal_shared[particle_id_in_block - particle_offset][2] = surface_particle_bin.val(_3, particle_id_in_bin);
							point_type_shared[particle_id_in_block - particle_offset] = *reinterpret_cast<SurfacePointType*>(&surface_particle_bin.val(_0, particle_id_in_bin));
							contact_area_shared[particle_id_in_block - particle_offset] = surface_particle_bin.val(_6, particle_id_in_bin);

							//Fetch position and determinant of deformation gradient
							FetchParticleBufferDataIntermediate fetch_particle_buffer_tmp = {};
							fetch_particle_buffer_data<MaterialTypeSolid>(particle_buffer_solid, advection_source_blockno, source_pidib, fetch_particle_buffer_tmp);
							position_shared[particle_id_in_block - particle_offset] = {fetch_particle_buffer_tmp.pos[0], fetch_particle_buffer_tmp.pos[1], fetch_particle_buffer_tmp.pos[2]};
							mass_shared[particle_id_in_block - particle_offset] = fetch_particle_buffer_tmp.mass;
							J_shared[particle_id_in_block - particle_offset] = fetch_particle_buffer_tmp.J;
							
							mass_surface_flow_shared[particle_id_in_block - particle_offset] = surface_flow_particle_bin.val(_0, particle_id_in_bin);
							J_surface_flow_shared[particle_id_in_block - particle_offset] = surface_flow_particle_bin.val(_1, particle_id_in_bin);
							velocity_surface_flow_shared[particle_id_in_block - particle_offset][0] = surface_flow_particle_bin.val(_2, particle_id_in_bin);
							velocity_surface_flow_shared[particle_id_in_block - particle_offset][1] = surface_flow_particle_bin.val(_3, particle_id_in_bin);
							velocity_surface_flow_shared[particle_id_in_block - particle_offset][2] = surface_flow_particle_bin.val(_4, particle_id_in_bin);
						}
						
						__syncthreads();
						
						for(int particle_id_in_block = particle_offset; particle_id_in_block < next_particle_buffer_solid.particle_bucket_sizes[current_blockno] && (particle_id_in_block - particle_offset) < SIMPLE_SURFACE_FLOW_MAX_SHARED_PARTICLE_SOLID; ++particle_id_in_block) {
							simple_surface_flow_aggregate_data_solid(
								  particle_buffer_solid
								, particle_buffer_fluid
								, next_particle_buffer_solid
								, next_particle_buffer_fluid
								, prev_partition
								, grid_solid
								, grid_fluid
								, surface_flow_particle_buffer
								, &(position_shared[0])
								, &(mass_shared[0])
								, &(J_shared[0])
								, &(normal_shared[0])
								, &(point_type_shared[0])
								, &(contact_area_shared[0])
								, &(mass_surface_flow_shared[0])
								, &(J_surface_flow_shared[0])
								, &(velocity_surface_flow_shared[0])
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
								, &(coupling_solid_domain_local[0])
								, &(coupling_solid_surface_local[0])
								, &(coupling_fluid_local[0])
								, &(velocity_surface_local[0])
								, &(scaling_surface_local[0])
								, &(pressure_surface_nominator_local[0])
								, &(pressure_surface_denominator_local[0])
								, &(mass_surface_local[0])
								, &(gradient_surface_local[0])
								, &(coupling_surface_local[0])
								, &(surface_flow_coupling_surface_local[0])
								, &(coupling_domain_local[0])
								, &(surface_flow_coupling_domain_local[0])
							);
						}
						
						__syncthreads();
					}
					
					for(int particle_offset = 0; particle_offset < next_particle_buffer_fluid.particle_bucket_sizes[current_blockno]; particle_offset += static_cast<int>(iq::MAX_SHARED_PARTICLE_FLUID)){
						__shared__ std::array<float, 3> position_shared[iq::MAX_SHARED_PARTICLE_FLUID];
						__shared__ float mass_shared[iq::MAX_SHARED_PARTICLE_FLUID];
						__shared__ float J_shared[iq::MAX_SHARED_PARTICLE_FLUID];
						
						for(int particle_id_in_block = particle_offset + static_cast<int>(threadIdx.x); particle_id_in_block < next_particle_buffer_fluid.particle_bucket_sizes[current_blockno] && (particle_id_in_block - particle_offset) < iq::MAX_SHARED_PARTICLE_FLUID; particle_id_in_block += static_cast<int>(blockDim.x)) {
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
						
						for(int particle_id_in_block = particle_offset; particle_id_in_block < next_particle_buffer_fluid.particle_bucket_sizes[current_blockno] && (particle_id_in_block - particle_offset) < iq::MAX_SHARED_PARTICLE_FLUID; ++particle_id_in_block) {
							simple_surface_flow_aggregate_data_fluid(
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
								, &(scaling_domain_local[0])
								, &(pressure_domain_nominator_local[0])
								, &(pressure_domain_denominator_local[0])
								, &(mass_domain_local[0])
								, &(gradient_domain_local[0])
							);
						}
						
						__syncthreads();
					}
				}
			}
		}
	}
	
	//Column that represents (row, row)
	//constexpr size_t IDENTIITY_NEIGHBOUR_INDEX = (iq::INTERPOLATION_DEGREE_MAX * ((2 * iq::INTERPOLATION_DEGREE_MAX + 1) * (2 * iq::INTERPOLATION_DEGREE_MAX + 1)) + iq::INTERPOLATION_DEGREE_MAX * (2 * iq::INTERPOLATION_DEGREE_MAX + 1) + iq::INTERPOLATION_DEGREE_MAX);
	
	//Store data in matrix
	//NOTE: Coupling was stored in transposed form
	
	/*
		RHS = {
			p_solid
			p_domain
			p_surface
			-
			-
			-
		}
		SOLVE_VELOCITY_RESULT = {
			v_solid
			v_domain
			v_surface
			-
			-
			-
		}
	*/
	for(size_t local_cell_index = 0; local_cell_index < iq::get_thread_count<iq::BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, config::G_BLOCKVOLUME); local_cell_index++){
		const size_t row = iq::get_global_index<iq::BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, local_cell_index);
		const ivec3 local_id {static_cast<int>((row / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE), static_cast<int>((row / config::G_BLOCKSIZE) % config::G_BLOCKSIZE), static_cast<int>(row % config::G_BLOCKSIZE)};

		float pressure_solid;
		//FIXME:float pressure_fluid;
		float pressure_domain;
		float pressure_surface;
		//Only calculate for particles with pressure_denominator bigger than 0 (otherwise we will divide by 0)
		if(pressure_solid_denominator_local[local_cell_index] > 0.0f){
			pressure_solid = pressure_solid_nominator_local[local_cell_index] / pressure_solid_denominator_local[local_cell_index];
		}else{
			pressure_solid = 0.0f;
		}
		//FIXME:if(pressure_domain_denominator_local[local_cell_index] > 0.0f || pressure_surface_denominator_local[local_cell_index] > 0.0f){
		//FIXME:	pressure_fluid = (pressure_domain_nominator_local[local_cell_index] + pressure_surface_nominator_local[local_cell_index]) / (pressure_domain_denominator_local[local_cell_index] + pressure_surface_denominator_local[local_cell_index]);
		//FIXME:}else{
		//FIXME:	pressure_fluid = 0.0f;
		//FIXME:}
		if(pressure_domain_denominator_local[local_cell_index] > 0.0f){
			pressure_domain = pressure_domain_nominator_local[local_cell_index] / pressure_domain_denominator_local[local_cell_index];
		}else{
			pressure_domain = 0.0f;
		}
		if(pressure_surface_denominator_local[local_cell_index] > 0.0f){
			pressure_surface = pressure_surface_nominator_local[local_cell_index] / pressure_surface_denominator_local[local_cell_index];
		}else{
			pressure_surface = 0.0f;
		}
		
		const int row_index_pressure_solid = base_row + row;
		//const int row_index_pressure_fluid = iq::NUM_ROWS_PER_BLOCK * num_blocks + base_row + row;
		const int row_index_pressure_domain = iq::NUM_ROWS_PER_BLOCK * num_blocks + base_row + row;
		const int row_index_pressure_surface = 2 * iq::NUM_ROWS_PER_BLOCK * num_blocks + base_row + row;
		
		atomicAdd(&(iq_pointers.iq_rhs[row_index_pressure_solid]), pressure_solid);
		//FIXME:atomicAdd(&(iq_pointers.iq_rhs[row_index_pressure_fluid]), pressure_fluid);
		atomicAdd(&(iq_pointers.iq_rhs[row_index_pressure_domain]), pressure_domain);
		atomicAdd(&(iq_pointers.iq_rhs[row_index_pressure_surface]), pressure_surface);
		
		//NOTE: Storing S/dt
		const int row_index_scaling = base_row + row;
		atomicAdd(&(iq_pointers.scaling_solid[row_index_scaling]), scaling_solid_local[local_cell_index] / dt.count());
		//FIXME:atomicAdd(&(iq_pointers.scaling_fluid[row_index_scaling]), (scaling_domain_local[local_cell_index] + scaling_surface_local[local_cell_index]) / dt.count());
		atomicAdd(&(iq_pointers.scaling_domain[row_index_scaling]), scaling_domain_local[local_cell_index] / dt.count());
		atomicAdd(&(iq_pointers.scaling_surface[row_index_scaling]), scaling_surface_local[local_cell_index] / dt.count());
		
		atomicAdd(&(iq_pointers.mass_single_domain[row_index_scaling]), mass_domain_local[3 * local_cell_index]);
		atomicAdd(&(iq_pointers.mass_single_surface[row_index_scaling]), mass_surface_local[3 * local_cell_index]);
	}
	
	for(size_t local_cell_index = 0; local_cell_index < iq::get_thread_count<iq::BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, 3 * config::G_BLOCKVOLUME); local_cell_index++){
		const size_t row = iq::get_global_index<iq::BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, local_cell_index) / 3;
		const size_t alpha = iq::get_global_index<iq::BLOCK_SIZE, (3 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, local_cell_index) % 3;
		const ivec3 local_id {static_cast<int>((row / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE), static_cast<int>((row / config::G_BLOCKSIZE) % config::G_BLOCKSIZE), static_cast<int>(row % config::G_BLOCKSIZE)};
		
		float mass_solid;
		//FIXME:float mass_fluid;
		float mass_domain;
		float mass_surface;
		
		float velocity_surface;
		//Only calculate for particles with mass bigger than 0 (otherwise we will divide by 0)
		if(mass_solid_local[local_cell_index] > 0.0f){
			mass_solid = dt.count() / mass_solid_local[local_cell_index];
		}else{
			mass_solid = 0.0f;
		}
		//FIXME:if(mass_domain_local[local_cell_index] > 0.0f || mass_surface_local[local_cell_index] > 0.0f){
		//FIXME:	mass_fluid = dt.count() / (mass_domain_local[local_cell_index] + mass_surface_local[local_cell_index]);
		//FIXME:}else{
		//FIXME:	mass_fluid = 0.0f;
		//FIXME:}
		if(mass_domain_local[local_cell_index] > 0.0f){
			mass_domain = dt.count() / mass_domain_local[local_cell_index];
		}else{
			mass_domain = 0.0f;
		}
		if(mass_surface_local[local_cell_index] > 0.0f){
			velocity_surface = velocity_surface_local[local_cell_index] / mass_surface_local[local_cell_index];
			mass_surface = dt.count() / mass_surface_local[local_cell_index];
		}else{
			velocity_surface = 0.0f;
			mass_surface = 0.0f;
		}
		
		const int row_index_velocity_solid = 3 * (base_row + row) + alpha;
		//const int row_index_velocity_fluid = 3 * iq::NUM_ROWS_PER_BLOCK * num_blocks + 3 * (base_row + row) + alpha;
		const int row_index_velocity_domain = 3 * iq::NUM_ROWS_PER_BLOCK * num_blocks + 3 * (base_row + row) + alpha;
		const int row_index_velocity_surface = 6 * iq::NUM_ROWS_PER_BLOCK * num_blocks + 3 * (base_row + row) + alpha;
		
		atomicAdd(&(iq_pointers.iq_solve_velocity_result[row_index_velocity_solid]), velocity_solid_local[local_cell_index]);
		//FIXME:atomicAdd(&(iq_pointers.iq_solve_velocity_result[row_index_velocity_fluid]), (velocity_domain_local[local_cell_index] + velocity_surface_local[local_cell_index]));
		atomicAdd(&(iq_pointers.iq_solve_velocity_result[row_index_velocity_domain]), velocity_domain_local[local_cell_index]);
		atomicAdd(&(iq_pointers.iq_solve_velocity_result[row_index_velocity_surface]), velocity_surface);
		
		//NOTE: Storing dt * M^-1
		const int row_index_mass = 3 * (base_row + row) + alpha;
		atomicAdd(&(iq_pointers.mass_solid[row_index_mass]), mass_solid);
		//FIXME:atomicAdd(&(iq_pointers.mass_fluid[row_index_mass]), mass_fluid);
		atomicAdd(&(iq_pointers.mass_domain[row_index_mass]), mass_domain);
		atomicAdd(&(iq_pointers.mass_surface[row_index_mass]), mass_surface);
	}
	
	for(size_t local_cell_index = 0; local_cell_index < iq::get_thread_count<iq::BLOCK_SIZE, (3 * iq::NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, 3 * iq::NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME); local_cell_index++){
		const size_t row = iq::get_global_index<iq::BLOCK_SIZE, (3 * iq::NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, local_cell_index) / (3 * iq::NUM_COLUMNS_PER_BLOCK);
		const size_t column = (iq::get_global_index<iq::BLOCK_SIZE, (3 * iq::NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, local_cell_index) / 3) % iq::NUM_COLUMNS_PER_BLOCK;
		const size_t alpha = iq::get_global_index<iq::BLOCK_SIZE, (3 * iq::NUM_COLUMNS_PER_BLOCK * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, local_cell_index) % 3;
		const ivec3 local_id {static_cast<int>((row / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE), static_cast<int>((row / config::G_BLOCKSIZE) % config::G_BLOCKSIZE), static_cast<int>(row % config::G_BLOCKSIZE)};
		const ivec3 neighbour_local_id = ivec3(static_cast<int>((column / ((2 * iq::INTERPOLATION_DEGREE_MAX + 1) * (2 * iq::INTERPOLATION_DEGREE_MAX + 1))) % (2 * iq::INTERPOLATION_DEGREE_MAX + 1)), static_cast<int>((column / (2 * iq::INTERPOLATION_DEGREE_MAX + 1)) % (2 * iq::INTERPOLATION_DEGREE_MAX + 1)), static_cast<int>(column % (2 * iq::INTERPOLATION_DEGREE_MAX + 1))) - ivec3(static_cast<int>(iq::INTERPOLATION_DEGREE_MAX), static_cast<int>(iq::INTERPOLATION_DEGREE_MAX), static_cast<int>(iq::INTERPOLATION_DEGREE_MAX));
			
		const ivec3 neighbour_cellid = block_cellid + local_id + neighbour_local_id;
		const ivec3 neighbour_blockid = neighbour_cellid / static_cast<int>(config::G_BLOCKSIZE);
		
		const ivec3 neighbour_base_cellid = neighbour_blockid * static_cast<int>(config::G_BLOCKSIZE);
		const ivec3 neighbour_celloffset = neighbour_cellid - neighbour_base_cellid;
		
		const int neighbour_blockno = partition.query(neighbour_blockid);
		const int neighbour_cellno = iq::NUM_ROWS_PER_BLOCK * neighbour_blockno + (config::G_BLOCKSIZE * config::G_BLOCKSIZE) * neighbour_celloffset[0] + config::G_BLOCKSIZE * neighbour_celloffset[1] + neighbour_celloffset[2];
		
		const int row_index = 3 * (base_row + row) + alpha;
		
		int local_column_index = -1;
		for(size_t lhs_column = 0; lhs_column < iq::NUM_COLUMNS_PER_BLOCK; ++lhs_column){
			if(neighbour_cellno == iq_pointers.gradient_solid_columns[iq_pointers.gradient_solid_rows[row_index] + lhs_column]){
				local_column_index = lhs_column;
				break;
			}
		}
		
		const int column_index = iq_pointers.gradient_solid_rows[row_index] + local_column_index;
		
		atomicAdd(&(iq_pointers.gradient_solid_values[column_index]), gradient_solid_local[local_cell_index]);
		//FIXME:atomicAdd(&(iq_pointers.gradient_fluid_values[column_index]), (gradient_domain_local[local_cell_index] + gradient_surface_local[local_cell_index]));
		atomicAdd(&(iq_pointers.gradient_domain_values[column_index]), gradient_domain_local[local_cell_index]);
		atomicAdd(&(iq_pointers.gradient_surface_values[column_index]), gradient_surface_local[local_cell_index]);
		
		//NOTE: Storing H^T
		atomicAdd(&(iq_pointers.coupling_solid_domain_values[column_index]), coupling_solid_domain_local[local_cell_index]);
		//FIXME:atomicAdd(&(iq_pointers.coupling_fluid_values[column_index]), coupling_fluid_local[local_cell_index]);
		atomicAdd(&(iq_pointers.coupling_domain_values[column_index]), coupling_domain_local[local_cell_index]);
		atomicAdd(&(iq_pointers.coupling_surface_values[column_index]), coupling_surface_local[local_cell_index]);
		
		atomicAdd(&(iq_pointers.coupling_solid_surface_values[column_index]), coupling_solid_surface_local[local_cell_index]);
		
		//NOTE: Storing C^T
		atomicAdd(&(iq_pointers.surface_flow_coupling_domain_values[column_index]), surface_flow_coupling_domain_local[local_cell_index]);
		atomicAdd(&(iq_pointers.surface_flow_coupling_surface_values[column_index]), surface_flow_coupling_surface_local[local_cell_index]);
	}
}

template<typename Partition, typename Grid, MaterialE MaterialTypeSolid, MaterialE MaterialTypeFluid, typename SurfaceFlowParticleBuffer>
__global__ void simple_surface_flow_mass_transfer(const ParticleBuffer<MaterialTypeSolid> particle_buffer_solid, ParticleBuffer<MaterialTypeSolid> next_particle_buffer_solid, const ParticleBuffer<MaterialTypeFluid> particle_buffer_fluid, ParticleBuffer<MaterialTypeFluid> next_particle_buffer_fluid, const Partition prev_partition, Partition partition, Grid grid_fluid, const SurfaceFlowParticleBuffer surface_flow_particle_buffer, const float* delta_density_domain) {
	//Particles with offset [-2, 0] can lie within cell (due to storing with interpolation degree 2 wich results in offset of 2); Interolation degree may offset positions so we need [-2, 2] for all interpolation positions in our cell. Then wee also need neighbour positions so we get [-4, 4];
	constexpr size_t KERNEL_SIZE = 2 * iq::INTERPOLATION_DEGREE_MAX + 5 + 1;//Plus one for both sides being inclusive
	constexpr size_t KERNEL_OFFSET = iq::INTERPOLATION_DEGREE_MAX + 2;
	
	//Both positive, both rounded up. Start will later be negated
	constexpr size_t KERNEL_START_BLOCK = (KERNEL_SIZE - KERNEL_OFFSET - 1 + config::G_BLOCKSIZE - 1) / config::G_BLOCKSIZE;
	constexpr size_t KERNEL_END_BLOCK = (KERNEL_OFFSET + config::G_BLOCKSIZE - 1) / config::G_BLOCKSIZE;
	
	const size_t base_row = iq::NUM_ROWS_PER_BLOCK * blockIdx.x;

	float delta_density_domain_local[(config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE];
	bool was_used[(config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE];

	const int src_blockno		   = static_cast<int>(blockIdx.x);
	const auto blockid			   = partition.active_keys[blockIdx.x];
	const ivec3 block_cellid = blockid * static_cast<int>(config::G_BLOCKSIZE);
	
	//Init memory/Load delta_density_domain
	for(size_t i = 0; i < (1 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE; ++i){
		delta_density_domain_local[i] = 0.0f;
		was_used[i] = false;
	}
	
	for(size_t local_cell_index = 0; local_cell_index < iq::get_thread_count<iq::BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, config::G_BLOCKVOLUME); local_cell_index++){
		const size_t row = iq::get_global_index<iq::BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, local_cell_index);
		const ivec3 local_id {static_cast<int>((row / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE), static_cast<int>((row / config::G_BLOCKSIZE) % config::G_BLOCKSIZE), static_cast<int>(row % config::G_BLOCKSIZE)};

		const int row_index_delta_density = base_row + row;
		delta_density_domain_local[local_cell_index] = delta_density_domain[row_index_delta_density];
	}
	
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
					
					auto surface_flow_particle_bin = surface_flow_particle_buffer.ch(_0, particle_buffer_solid.bin_offsets[advection_source_blockno] + source_pidib / config::G_BIN_CAPACITY);
					const int particle_id_in_bin = source_pidib  % config::G_BIN_CAPACITY;
					
					float mass = surface_flow_particle_bin.val(_0, particle_id_in_bin);
					const float J = surface_flow_particle_bin.val(_1, particle_id_in_bin);
					
					//Get position of grid cell
					const ivec3 global_base_index_fluid_velocity = get_cell_id<iq::INTERPOLATION_DEGREE_FLUID_VELOCITY>(pos.data_arr(), grid_fluid.get_offset());
					
					//Get position relative to grid cell
					const vec3 local_pos_fluid_velocity = pos - (global_base_index_fluid_velocity + vec3(grid_fluid.get_offset()[0], grid_fluid.get_offset()[1], grid_fluid.get_offset()[2])) * config::G_DX;

					//Calculate weights
					vec3x3 weight_fluid_velocity;
					
					#pragma unroll 3
					for(int dd = 0; dd < 3; ++dd) {
						const std::array<float, iq::INTERPOLATION_DEGREE_FLUID_VELOCITY + 1> current_weight_fluid_velocity = bspline_weight<float, iq::INTERPOLATION_DEGREE_FLUID_VELOCITY>(local_pos_fluid_velocity[dd]);
						for(int i = 0; i < iq::INTERPOLATION_DEGREE_FLUID_VELOCITY + 1; ++i){
							weight_fluid_velocity(dd, i)		  = current_weight_fluid_velocity[i];
						}
						for(int i = iq::INTERPOLATION_DEGREE_FLUID_VELOCITY + 1; i < 3; ++i){
							weight_fluid_velocity(dd, i)		  = 0.0f;
						}
					}
					
					float delta_mass = 0.0f;
					for(size_t local_cell_index = 0; local_cell_index < iq::get_thread_count<iq::BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, config::G_BLOCKVOLUME); local_cell_index++){
						const size_t cell_index = iq::get_global_index<iq::BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, local_cell_index);
						const ivec3 local_id {static_cast<int>((cell_index / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE), static_cast<int>((cell_index / config::G_BLOCKSIZE) % config::G_BLOCKSIZE), static_cast<int>(cell_index % config::G_BLOCKSIZE)};
						
						const ivec3 local_offset_velocity = global_base_index_fluid_velocity - (block_cellid + local_id);
							
						const ivec3 absolute_local_offset_velocity {std::abs(local_offset_velocity[0]), std::abs(local_offset_velocity[1]), std::abs(local_offset_velocity[2])};
						
						//Weight
						const float W_velocity = (absolute_local_offset_velocity[0] < 3 ? weight_fluid_velocity(0, absolute_local_offset_velocity[0]) : 0.0f) * (absolute_local_offset_velocity[1] < 3 ? weight_fluid_velocity(1, absolute_local_offset_velocity[1]) : 0.0f) * (absolute_local_offset_velocity[2] < 3 ? weight_fluid_velocity(2, absolute_local_offset_velocity[2]) : 0.0f);
						
						//FIXME: Correct sign!
						delta_mass += delta_density_domain_local[local_cell_index] * W_velocity;
					}
					
					mass += delta_mass;
					
					//Store new mass
					atomicAdd(&surface_flow_particle_bin.val(_0, particle_id_in_bin), delta_mass);
				}
				
				for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < next_particle_buffer_fluid.particle_bucket_sizes[current_blockno]; particle_id_in_block += static_cast<int>(blockDim.x)) {
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
					float mass = fetch_particle_buffer_tmp.mass;
					const float J = fetch_particle_buffer_tmp.J;
					
					auto particle_bin													= particle_buffer_fluid.ch(_0, particle_buffer_fluid.bin_offsets[advection_source_blockno] + source_pidib / config::G_BIN_CAPACITY);
					const int particle_id_in_bin = source_pidib  % config::G_BIN_CAPACITY;
					
					
					//Get position of grid cell
					const ivec3 global_base_index_fluid_velocity = get_cell_id<iq::INTERPOLATION_DEGREE_FLUID_VELOCITY>(pos.data_arr(), grid_fluid.get_offset());
					
					//Get position relative to grid cell
					const vec3 local_pos_fluid_velocity = pos - (global_base_index_fluid_velocity + vec3(grid_fluid.get_offset()[0], grid_fluid.get_offset()[1], grid_fluid.get_offset()[2])) * config::G_DX;

					//Calculate weights
					vec3x3 weight_fluid_velocity;
					
					#pragma unroll 3
					for(int dd = 0; dd < 3; ++dd) {
						const std::array<float, iq::INTERPOLATION_DEGREE_FLUID_VELOCITY + 1> current_weight_fluid_velocity = bspline_weight<float, iq::INTERPOLATION_DEGREE_FLUID_VELOCITY>(local_pos_fluid_velocity[dd]);
						for(int i = 0; i < iq::INTERPOLATION_DEGREE_FLUID_VELOCITY + 1; ++i){
							weight_fluid_velocity(dd, i)		  = current_weight_fluid_velocity[i];
						}
						for(int i = iq::INTERPOLATION_DEGREE_FLUID_VELOCITY + 1; i < 3; ++i){
							weight_fluid_velocity(dd, i)		  = 0.0f;
						}
					}
					
					float delta_mass = 0.0f;
					for(size_t local_cell_index = 0; local_cell_index < iq::get_thread_count<iq::BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, config::G_BLOCKVOLUME); local_cell_index++){
						const size_t cell_index = iq::get_global_index<iq::BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, local_cell_index);
						const ivec3 local_id {static_cast<int>((cell_index / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE), static_cast<int>((cell_index / config::G_BLOCKSIZE) % config::G_BLOCKSIZE), static_cast<int>(cell_index % config::G_BLOCKSIZE)};
						
						const ivec3 local_offset_velocity = global_base_index_fluid_velocity - (block_cellid + local_id);
							
						const ivec3 absolute_local_offset_velocity {std::abs(local_offset_velocity[0]), std::abs(local_offset_velocity[1]), std::abs(local_offset_velocity[2])};
						
						//Weight
						const float W_velocity = (absolute_local_offset_velocity[0] < 3 ? weight_fluid_velocity(0, absolute_local_offset_velocity[0]) : 0.0f) * (absolute_local_offset_velocity[1] < 3 ? weight_fluid_velocity(1, absolute_local_offset_velocity[1]) : 0.0f) * (absolute_local_offset_velocity[2] < 3 ? weight_fluid_velocity(2, absolute_local_offset_velocity[2]) : 0.0f);
						
						//Mark as used if it was used
						if(W_velocity != 0.0f){
							was_used[local_cell_index] = true;
						}
						
						//FIXME: Correct sign!
						delta_mass -= delta_density_domain_local[local_cell_index] * W_velocity;
					}
					
					mass += delta_mass;
					
					//Store new mass
					atomicAdd(&particle_bin.val(_0, particle_id_in_bin), delta_mass);
				}
			}
		}
	}
	
	//For all cells that were not used (means no fluid particle received the mass transfer) we generate new particles
	for(size_t local_cell_index = 0; local_cell_index < iq::get_thread_count<iq::BLOCK_SIZE, (1 * config::G_BLOCKVOLUME + iq::BLOCK_SIZE - 1) / iq::BLOCK_SIZE>(threadIdx.x, config::G_BLOCKVOLUME); local_cell_index++){
		//delta_density_domain_local[local_cell_index] positive here in any case
		if(!was_used[local_cell_index] && delta_density_domain_local[local_cell_index] != 0.0f){
			//TODO:dropped_mass = spawn_new_particles(particle_buffer_fluid, partition, next_grid, partition_block_count, drop_out_mass, shell_pos.data_arr(), extrapolated_pos.data_arr(), src_blockno);
		}
	}
}

template<size_t NumRowsPerBlock>
__global__ void simple_surface_flow_get_min_delta_density(float* delta_density_domain, const float* delta_density_surface, const float* density_domain, const float* density_surface) {
	
	//Handle own rows and add column data for all neighbour cells of each cell (fixed amount)
	//We can calculate the offset in the column array by our id and the amount of neighbour cells (=columns)
	const size_t base_row = NumRowsPerBlock * blockIdx.x;
	
	for(size_t row = static_cast<int>(threadIdx.x); row < NumRowsPerBlock; row += static_cast<int>(blockDim.x)){
		delta_density_domain[base_row + row] = -std::min(std::min(density_domain[base_row + row], -delta_density_domain[base_row + row]), std::min(density_surface[base_row + row], -delta_density_surface[base_row + row]));
	}
}

//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers, readability-identifier-naming, misc-definitions-in-headers)
}// namespace mn

#endif