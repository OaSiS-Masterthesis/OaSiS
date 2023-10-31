#ifndef SURFACE_FLOW_CUH
#define SURFACE_FLOW_CUH

#include <MnSystem/Cuda/Cuda.h>

#include "particle_buffer.cuh"
#include "surface_flow_kernels.cuh"

namespace mn {

//NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables, readability-identifier-naming) Check is buggy and reports variable errors for template arguments
using SurfaceFlowParticleBufferData  = Structural<StructuralType::DENSE, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::SUM_POW2_ALIGN>, ParticleBinDomain, attrib_layout::SOA, f32_, f32_, f32_, f32_, f32_>;//mass, J, velocity
//NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables, readability-identifier-naming)
	
struct SurfaceFlowParticleBuffer : Instance<particle_buffer_<SurfaceFlowParticleBufferData>> {
	using base_t							 = Instance<particle_buffer_<SurfaceFlowParticleBufferData>>;
	
	managed_memory_type* managed_memory;

	SurfaceFlowParticleBuffer() = default;

	template<typename Allocator>
	SurfaceFlowParticleBuffer(Allocator allocator, managed_memory_type* managed_memory, std::size_t count)
		: base_t {spawn<particle_buffer_<SurfaceFlowParticleBufferData>, orphan_signature>(allocator, count)}
		, managed_memory(managed_memory)
		{}
};
 
class SurfaceFlowModel {
public:

	virtual [[nodiscard]] size_t get_lhs_matrix_size_x() const = 0;
	
	virtual [[nodiscard]] size_t get_lhs_matrix_size_y() const = 0;
	
	virtual [[nodiscard]] size_t get_lhs_matrix_total_block_count() const = 0;
	
	virtual [[nodiscard]] size_t get_solve_velocity_matrix_size_x() const = 0;
	
	virtual [[nodiscard]] size_t get_solve_velocity_matrix_size_y() const = 0;
	
	virtual [[nodiscard]] size_t get_solve_velocity_matrix_total_block_count() const = 0;
	
	virtual [[nodiscard]] size_t get_num_temporary_matrices() const = 0;
	
	virtual [[nodiscard]] size_t* get_lhs_num_blocks_per_row_ptr() const = 0;
	
	virtual [[nodiscard]] size_t* get_lhs_block_offsets_per_row_ptr() const = 0;
	
	virtual [[nodiscard]] size_t* get_solve_velocity_num_blocks_per_row_ptr() const = 0;
	
	virtual [[nodiscard]] size_t* get_solve_velocity_block_offsets_per_row_ptr() const = 0;
	
	virtual void initialize(
		std::shared_ptr<gko::Executor>& ginkgo_executor
	) = 0;
	
	virtual void resize_and_clear(
		const int exterior_block_count,
		const int coupling_block_count
	) = 0;
	
	virtual void fill_matrices(
		managed_memory_type& managed_memory,
		std::array<std::vector<particle_buffer_t>, BIN_COUNT>& particle_bins,
		std::vector<Partition<1>>& partitions,
		std::array<std::vector<GridBuffer>, BIN_COUNT>& grid_blocks,
		std::vector<SurfaceParticleBuffer>& surface_particle_buffers,
		std::array<std::vector<SurfaceFlowParticleBuffer>, BIN_COUNT>& surface_flow_particle_buffers,
		const int rollid,
		const Duration& dt,
		const int exterior_block_count,
		const int coupling_block_count,
		const int solid_id,
		const int fluid_id,
		const int surface_flow_id,
		gko::array<float>& iq_lhs_scaling_solid_values,
		gko::array<float>& iq_lhs_scaling_fluid_values,
		gko::array<float>& iq_lhs_mass_solid_values,
		gko::array<float>& iq_lhs_mass_fluid_values,
		gko::array<int>& iq_lhs_3_1_rows,
		gko::array<int>& iq_lhs_3_1_columns,
		gko::array<float>& iq_lhs_gradient_solid_values,
		gko::array<float>& iq_lhs_gradient_fluid_values,
		gko::array<float>& iq_lhs_coupling_solid_values,
		gko::array<float>& iq_lhs_coupling_fluid_values,
		gko::array<float>& iq_rhs_array,
		gko::array<float>& iq_solve_velocity_result_array,
		Cuda::CudaContext& cu_dev
	) = 0;
	
	
	virtual void generate_system_matrices(
		std::shared_ptr<gko::Executor>& ginkgo_executor,
		MatrixOperations& matrix_operations,
		const int exterior_block_count,
		gko::array<float>& iq_lhs_scaling_solid_values,
		gko::array<float>& iq_lhs_scaling_fluid_values,
		gko::array<float>& iq_lhs_mass_solid_values,
		gko::array<float>& iq_lhs_mass_fluid_values,
		gko::array<int>& iq_lhs_3_1_rows,
		gko::array<int>& iq_lhs_3_1_columns,
		gko::array<float>& iq_lhs_gradient_solid_values,
		gko::array<float>& iq_lhs_gradient_fluid_values,
		gko::array<float>& iq_lhs_coupling_solid_values,
		gko::array<float>& iq_lhs_coupling_fluid_values,
		gko::array<float>& iq_solve_velocity_result_array,
		gko::array<float>& gko_identity_values,
		std::shared_ptr<gko::matrix::Dense<float>>& gko_neg_one_dense,
		std::shared_ptr<gko::matrix::Dense<float>>& gko_one_dense,
		std::vector<std::shared_ptr<gko::matrix::Csr<float, int>>>& iq_lhs_parts,
		std::vector<std::shared_ptr<gko::matrix::Csr<float, int>>>& iq_solve_velocity_parts,
		std::vector<std::shared_ptr<gko::matrix::Csr<float, int>>>& temporary_matrices,
		std::vector<std::shared_ptr<gko::matrix::Dense<float>>>& iq_rhs_parts,
		Cuda::CudaContext& cu_dev
	) = 0;
	
	virtual void mass_transfer(
		managed_memory_type& managed_memory,
		std::array<std::vector<particle_buffer_t>, BIN_COUNT>& particle_bins,
		std::vector<Partition<1>>& partitions,
		std::array<std::vector<GridBuffer>, BIN_COUNT>& grid_blocks,
		std::array<std::vector<SurfaceFlowParticleBuffer>, BIN_COUNT>& surface_flow_particle_buffers,
		const int rollid,
		const int partition_block_count,
		const int solid_id,
		const int fluid_id,
		const int surface_flow_id,
		std::shared_ptr<gko::Executor>& ginkgo_executor,
		const int exterior_block_count,
		std::shared_ptr<gko::matrix::Dense<float>>& iq_result,
		gko::array<float>& gko_identity_values,
		std::shared_ptr<gko::matrix::Dense<float>>& gko_neg_one_dense,
		std::vector<std::shared_ptr<gko::matrix::Csr<float, int>>>& iq_lhs_parts,
		Cuda::CudaContext& cu_dev
	) = 0;
	
	virtual void update_after_solve(
		managed_memory_type& managed_memory,
		std::array<std::vector<particle_buffer_t>, BIN_COUNT>& particle_bins,
		std::vector<Partition<1>>& partitions,
		std::array<std::vector<GridBuffer>, BIN_COUNT>& grid_blocks,
		const int rollid,
		const int exterior_block_count,
		const int partition_block_count,
		const int solid_id,
		const int fluid_id,
		const std::shared_ptr<gko::matrix::Dense<float>>& iq_solve_velocity_result,
		const std::shared_ptr<gko::matrix::Dense<float>>& iq_result,
		Cuda::CudaContext& cu_dev
	) = 0;
};

//TODO: Rename; Also at other palces
/*
	Having momentum and mass balance equations and coupling for domain only, for surface flow only and for combined surface flow and domain.
	That way surface and domain are continuous together by Navier-Stokes. Also domain and surface are coupled and move continuous on their own
	Resulting we get a pressure difference (p_surface - p_domain = c^n+1_interface_domain_surface) per node which we can use for mass transfer.
	
	We are enforcing:
	(v^n+1_domain  - v^n+1_surface) = 0; at interface domain-surface
	(p^n+1_domain  - p^n+1_surface) = 0; at interface domain-surface
	(v^n+1_solid   - v^n+1_domain ) * n = 0;  at interface solid-fluid
	(v^n+1_solid   - v^n+1_surface) * n = 0;  at interface solid-fluid
	(p^n+1_solid   - p^n+1_domain  - p^n+1_surface) = 0;  at interface solid-fluid
	
	//NOTE: Using -h and -c!
	
	The matrices look like this:
	Non-symmetric version:
	1/dt * M_solid     G_solid             -                  -                   -                  -                   -H^T_solid_domain  -H^T_solid_surface   -                        v^n+1_solid                         1/dt * M_solid   * v^n_solid   
	G^T_solid          -1/dt * S_solid     -                  -                   -                  -                   -                   -                   -                        p^n+1_solid                         -1/dt * S_solid   * p^n_solid  
	-                  -                   1/dt * M_domain    G_domain            -                  -                   H^T_domain          -                   -C^T_domain              v^n+1_domain                        1/dt * M_domain  * v^n_domain  
	-                  -                   G^T_domain         -1/dt * S_domain    -                  -                   -                   -                   -                        p^n+1_domain                        -1/dt * S_domain  * p^n_domain 
	-                  -                   -                  -                   1/dt * M_surface   G_surface           -                   H^T_surface         C^T_surface              v^n+1_surface                       1/dt * M_solid   * v^n_solid   
	-                  -                   -                  -                   G^T_surface        -1/dt * S_surface   -                   -                   -                        p^n+1_surface                       -1/dt * S_surface * p^n_surface
	-H_solid_domain    -                   H_domain           -                   -                  -                   -                   -                   -                        -h^n+1_interface_solid_domain       0                              
	-H_solid_surface   -                   -                  -                   H_surface          -                   -                   -                   -                   	  -h^n+1_interface_solid_surface      0                              
	-                  -                   -C_domain          -                   C_surface          -                   -                   -                   -                   *    -c^n+1_interface_domain_surface  =  0                              
	
	Solve velocity:
	
	(1)': v^n+1_solid   = v^n_solid   + (-dt * M^-1_solid   * G_solid   * p^n+1_solid   - dt * M^-1_solid   * H^T_solid_domain * h^n+1_interface_solid_domain  - dt * M^-1_solid_surface * H^T_solid   * h^n+1_interface_solid_surface)
	(3)': v^n+1_domain  = v^n_domain  + (-dt * M^-1_domain  * G_domain  * p^n+1_domain  + dt * M^-1_domain  * H^T_domain  * h^n+1_interface_solid_domain  - dt * M^-1_domain  * C^T_domain  * c^n+1_interface_domain_surface)
	(5)': v^n+1_surface = v^n_surface + (-dt * M^-1_surface * G_surface * p^n+1_surface + dt * M^-1_surface * H^T_surface * h^n+1_interface_solid_surface + dt * M^-1_surface * C^T_surface * c^n+1_interface_domain_surface)
	
	Symmetric version:
	(1)*: (1)'->(2)';
	(2)*: (3)'->(4)';
	(3)*: (5)'->(6)';
	(4)*: (1)',(3)'->(7)';
	(5)*: (1)',(5)'->(8)';
	(6)*: (3)',(5)'->(9)';
	------------------------------------------------
	A_11    -       -       A_14    A_15    -             p^n+1_solid                                S_solid/dt   * p^n_solid   + G^T_solid   * v^n_solid  
	-       A_22    -       A_24    -       A_26          p^n+1_domain                               S_domain/dt  * p^n_domain  + G^T_domain  * v^n_domain 
	-       -       A_33    -       A_35    A_36          p^n+1_surface                              S_surface/dt * p^n_surface + G^T_surface * v^n_surface
	A^T_14  A^T_24  -       A_44    A_45    A_46          -h^n+1_interface_solid_domain              H_solid_domain  * v^n_solid   - H_domain  * v^n_domain 
	A^T_15  -       A^T_35  A^T_45  A_55    A_56          -h^n+1_interface_solid_surface             H_solid_surface * v^n_solid   - H_surface * v^n_surface
	-       A^T_26  A^T_36  A^T_46  A^T_56  A_66          -c^n+1_domain_interface_domain_surface     C_domain  * v^n_domain  - C_surface * v^n_surface
	
	A_11 = S_solid/dt   + dt * G^T_solid   * M^-1_solid   * G_solid  
	A_14 = A_15 = dt * G^T_solid   M^-1_solid   * H^T_solid  
	A_22 = S_domain/dt  + dt * G^T_domain  * M^-1_domain  * G_domain 
	A_24 = -dt * G^T_domain  * M^-1_domain  * H^T_domain 
	A_26 = dt * G^T_domain  * M^-1_domain  * C^T_domain 
	A_33 = S_surface/dt + dt * G^T_surface * M^-1_surface * G_surface
	A_35 = -dt * G^T_surface * M^-1_surface * H^T_surface
	A_36 = -dt * G^T_surface * M^-1_surface * C^T_surface
	A_44 = dt * H_solid_domain  * M^-1_solid   * H^H_solid_domain  + dt * H_domain  * M^-1_domain  * H^T_domain 
	A_45 = dt * H_solid_domain  * M^-1_solid   * H^T_solid_surface  
	A_46 = -dt * H_domain  * M^-1_domain  * C^T_domain 
	A_55 = dt * H_solid_surface * M^-1_solid   * H^T_solid_surface  + dt * H_surface * M^-1_surface * H^T_surface
	A_56 = dt * H_surface * M^-1_surface * C^T_surface
	A_66 = dt * C_domain  * M^-1_domain  * C^T_domain + dt * C_surface * M^-1_surface * C^T_surface
	
	
	
	Mass transfer:
	dphi/dt = p_0 / (lambda * J) * dp/dt;
	phi = density; p = pressure; For us dp/dt = c;
	
	H = Coupling at surface of solid in normal direction
	C = Coupling at surface of surface flow in all directions
*/
class SimpleSurfaceFlowModel : public SurfaceFlowModel{
public:	

	using streamIdx		 = Cuda::StreamIndex;
	using eventIdx		 = Cuda::EventIndex;

	static constexpr size_t NUM_TEMPORARY_MATRICES = 1;
	
	gko::array<float> iq_lhs_scaling_domain_values;
	gko::array<float> iq_lhs_scaling_surface_values;
	
	gko::array<float> iq_lhs_mass_domain_values;
	gko::array<float> iq_lhs_mass_surface_values;
	
	gko::array<float> iq_lhs_gradient_domain_values;
	gko::array<float> iq_lhs_gradient_surface_values;
	gko::array<float> iq_lhs_coupling_domain_values;
	gko::array<float> iq_lhs_coupling_surface_values;
	
	gko::array<float> iq_lhs_coupling_solid_surface_values;
	
	gko::array<float> iq_lhs_surface_flow_coupling_domain_values;
	gko::array<float> iq_lhs_surface_flow_coupling_surface_values;
	
	//Like mass, but not dt multiplied and only one column for all three dimensions
	gko::array<float> iq_lhs_mass_single_domain_values;
	gko::array<float> iq_lhs_mass_single_surface_values;
	
	SimpleSurfaceFlowModel(std::shared_ptr<gko::Executor>& ginkgo_executor){
		initialize(ginkgo_executor);
	}
	
	[[nodiscard]] size_t get_lhs_matrix_size_x() const override{
		return SIMPLE_SURFACE_FLOW_LHS_MATRIX_SIZE_X;
	}
	
	[[nodiscard]] size_t get_lhs_matrix_size_y() const override{
		return SIMPLE_SURFACE_FLOW_LHS_MATRIX_SIZE_Y;
	}
	
	[[nodiscard]] size_t get_lhs_matrix_total_block_count() const override{
		return SIMPLE_SURFACE_FLOW_LHS_MATRIX_TOTAL_BLOCK_COUNT;
	}
	
	[[nodiscard]] size_t get_solve_velocity_matrix_size_x() const override{
		return SIMPLE_SURFACE_FLOW_SOLVE_VELOCITY_MATRIX_SIZE_X;
	}
	
	[[nodiscard]] size_t get_solve_velocity_matrix_size_y() const override{
		return SIMPLE_SURFACE_FLOW_SOLVE_VELOCITY_MATRIX_SIZE_Y;
	}
	
	[[nodiscard]] size_t get_solve_velocity_matrix_total_block_count() const override{
		return SIMPLE_SURFACE_FLOW_SOLVE_VELOCITY_MATRIX_TOTAL_BLOCK_COUNT;
	}
	
	[[nodiscard]] size_t get_num_temporary_matrices() const override{
		return NUM_TEMPORARY_MATRICES;
	}
	
	[[nodiscard]] size_t* get_lhs_num_blocks_per_row_ptr() const override{
		size_t* lhs_num_blocks_per_row_ptr;
		cudaGetSymbolAddress(reinterpret_cast<void**>(&lhs_num_blocks_per_row_ptr), simple_surface_flow_lhs_num_blocks_per_row);
		
		return lhs_num_blocks_per_row_ptr;
	}
	
	[[nodiscard]] size_t* get_lhs_block_offsets_per_row_ptr() const override{
		size_t* lhs_block_offsets_per_row_ptr;
		cudaGetSymbolAddress(reinterpret_cast<void**>(&lhs_block_offsets_per_row_ptr), simple_surface_flow_lhs_block_offsets_per_row);
		
		return lhs_block_offsets_per_row_ptr;
	}
	
	[[nodiscard]] size_t* get_solve_velocity_num_blocks_per_row_ptr() const override{
		size_t* solve_velocity_num_blocks_per_row_ptr;
		cudaGetSymbolAddress(reinterpret_cast<void**>(&solve_velocity_num_blocks_per_row_ptr), simple_surface_flow_solve_velocity_num_blocks_per_row);
		
		return solve_velocity_num_blocks_per_row_ptr;
	}
	
	[[nodiscard]] size_t* get_solve_velocity_block_offsets_per_row_ptr() const override{
		size_t* solve_velocity_block_offsets_per_row;
		cudaGetSymbolAddress(reinterpret_cast<void**>(&solve_velocity_block_offsets_per_row), simple_surface_flow_solve_velocity_block_offsets_per_row);
		
		return solve_velocity_block_offsets_per_row;
	}
	
	void initialize(
		std::shared_ptr<gko::Executor>& ginkgo_executor
	) override{
		iq_lhs_scaling_domain_values = gko::array<float>(ginkgo_executor, 32 * config::G_BLOCKVOLUME);
		iq_lhs_scaling_surface_values = gko::array<float>(ginkgo_executor, 32 * config::G_BLOCKVOLUME);
		iq_lhs_mass_domain_values = gko::array<float>(ginkgo_executor, 3 * 32 * config::G_BLOCKVOLUME);
		iq_lhs_mass_surface_values = gko::array<float>(ginkgo_executor, 3 * 32 * config::G_BLOCKVOLUME);
		
		iq_lhs_gradient_domain_values = gko::array<float>(ginkgo_executor,  3 * 32 * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
		iq_lhs_gradient_surface_values = gko::array<float>(ginkgo_executor,  3 * 32 * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
		iq_lhs_coupling_domain_values = gko::array<float>(ginkgo_executor,  3 * 32 * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
		iq_lhs_coupling_surface_values = gko::array<float>(ginkgo_executor,  3 * 32 * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
		
		iq_lhs_coupling_solid_surface_values = gko::array<float>(ginkgo_executor,  3 * 32 * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
		
		iq_lhs_surface_flow_coupling_domain_values = gko::array<float>(ginkgo_executor,  3 * 32 * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
		iq_lhs_surface_flow_coupling_surface_values = gko::array<float>(ginkgo_executor,  3 * 32 * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
		
		iq_lhs_mass_single_domain_values = gko::array<float>(ginkgo_executor, 32 * config::G_BLOCKVOLUME);
		iq_lhs_mass_single_surface_values = gko::array<float>(ginkgo_executor, 32 * config::G_BLOCKVOLUME);
	}
	
	void resize_and_clear(
		const int exterior_block_count,
		const int coupling_block_count
	) override{
		iq_lhs_scaling_domain_values.resize_and_reset(exterior_block_count * config::G_BLOCKVOLUME);
		iq_lhs_scaling_surface_values.resize_and_reset(exterior_block_count * config::G_BLOCKVOLUME);
		iq_lhs_mass_domain_values.resize_and_reset(3 * exterior_block_count * config::G_BLOCKVOLUME);
		iq_lhs_mass_surface_values.resize_and_reset(3 * exterior_block_count * config::G_BLOCKVOLUME);
		
		iq_lhs_gradient_domain_values.resize_and_reset(3 * coupling_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
		iq_lhs_gradient_surface_values.resize_and_reset(3 * coupling_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
		iq_lhs_coupling_domain_values.resize_and_reset(3 * coupling_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
		iq_lhs_coupling_surface_values.resize_and_reset(3 * coupling_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
		
		iq_lhs_surface_flow_coupling_domain_values.resize_and_reset(3 * coupling_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
		iq_lhs_surface_flow_coupling_surface_values.resize_and_reset(3 * coupling_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
		
		iq_lhs_coupling_solid_surface_values.resize_and_reset(3 * coupling_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
		
		iq_lhs_mass_single_domain_values.resize_and_reset(3 * exterior_block_count * config::G_BLOCKVOLUME);
		iq_lhs_mass_single_surface_values.resize_and_reset(3 * exterior_block_count * config::G_BLOCKVOLUME);
		
		iq_lhs_scaling_domain_values.fill(0.0f);
		iq_lhs_scaling_surface_values.fill(0.0f);
		iq_lhs_mass_domain_values.fill(0.0f);
		iq_lhs_mass_surface_values.fill(0.0f);
		
		iq_lhs_gradient_domain_values.fill(0.0f);
		iq_lhs_gradient_surface_values.fill(0.0f);
		iq_lhs_coupling_domain_values.fill(0.0f);
		iq_lhs_coupling_surface_values.fill(0.0f);
		
		iq_lhs_surface_flow_coupling_domain_values.fill(0.0f);
		iq_lhs_surface_flow_coupling_surface_values.fill(0.0f);
		
		iq_lhs_coupling_solid_surface_values.fill(0.0f);
		
		iq_lhs_mass_single_domain_values.fill(0.0f);
		iq_lhs_mass_single_surface_values.fill(0.0f);
	}
	
	void fill_matrices(
		managed_memory_type& managed_memory,
		std::array<std::vector<particle_buffer_t>, BIN_COUNT>& particle_bins,
		std::vector<Partition<1>>& partitions,
		std::array<std::vector<GridBuffer>, BIN_COUNT>& grid_blocks,
		std::vector<SurfaceParticleBuffer>& surface_particle_buffers,
		std::array<std::vector<SurfaceFlowParticleBuffer>, BIN_COUNT>& surface_flow_particle_buffers,
		const int rollid,
		const Duration& dt,
		const int exterior_block_count,
		const int coupling_block_count,
		const int solid_id,
		const int fluid_id,
		const int surface_flow_id,
		gko::array<float>& iq_lhs_scaling_solid_values,
		gko::array<float>& iq_lhs_scaling_fluid_values,
		gko::array<float>& iq_lhs_mass_solid_values,
		gko::array<float>& iq_lhs_mass_fluid_values,
		gko::array<int>& iq_lhs_3_1_rows,
		gko::array<int>& iq_lhs_3_1_columns,
		gko::array<float>& iq_lhs_gradient_solid_values,
		gko::array<float>& iq_lhs_gradient_fluid_values,
		gko::array<float>& iq_lhs_coupling_solid_values,
		gko::array<float>& iq_lhs_coupling_fluid_values,
		gko::array<float>& iq_rhs_array,
		gko::array<float>& iq_solve_velocity_result_array,
		Cuda::CudaContext& cu_dev
	) override{
		//IQ-System create
		/*
			Creates single arrays for composition of left side, right side and solve velocity matrix (G_solid, S_solid, M^s, G_fluid, S_fluid, M^f, H_solid, H_fluid, B, p_solid, p_fluid, v_solid, v_fluid).
			//NOTE: Storing H^T instead H, C^T instead C and dt * M^-1 instead M and S/dt instead S.
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
		match(particle_bins[rollid][solid_id], particle_bins[rollid][fluid_id])([
			this,
			&rollid,
			&dt,
			&exterior_block_count,
			&coupling_block_count,
			&solid_id,
			&fluid_id,
			&surface_flow_id,
			&managed_memory,
			&particle_bins,
			&partitions,
			&grid_blocks,
			&surface_particle_buffers,
			&surface_flow_particle_buffers,
			&iq_lhs_scaling_solid_values,
			&iq_lhs_scaling_fluid_values,
			&iq_lhs_mass_solid_values,
			&iq_lhs_mass_fluid_values,
			&iq_lhs_3_1_rows,
			&iq_lhs_3_1_columns,
			&iq_lhs_gradient_solid_values,
			&iq_lhs_gradient_fluid_values,
			&iq_lhs_coupling_solid_values,
			&iq_lhs_coupling_fluid_values,
			&iq_rhs_array,
			&iq_solve_velocity_result_array,
			&cu_dev
		](auto& particle_buffer_solid, auto& particle_buffer_fluid) {
			auto& next_particle_buffer_solid = get<typename std::decay_t<decltype(particle_buffer_solid)>>(particle_bins[(rollid + 1) % BIN_COUNT][solid_id]);
			auto& next_particle_buffer_fluid = get<typename std::decay_t<decltype(particle_buffer_fluid)>>(particle_bins[(rollid + 1) % BIN_COUNT][fluid_id]);
			
			particle_buffer_solid.bin_offsets = particle_buffer_solid.bin_offsets_virtual;
			particle_buffer_fluid.bin_offsets = particle_buffer_fluid.bin_offsets_virtual;
			next_particle_buffer_solid.particle_bucket_sizes = next_particle_buffer_solid.particle_bucket_sizes_virtual;
			next_particle_buffer_fluid.particle_bucket_sizes = next_particle_buffer_fluid.particle_bucket_sizes_virtual;
			next_particle_buffer_solid.blockbuckets = next_particle_buffer_solid.blockbuckets_virtual;
			next_particle_buffer_fluid.blockbuckets = next_particle_buffer_fluid.blockbuckets_virtual;
			managed_memory.acquire<MemoryType::DEVICE>(
				  particle_buffer_solid.acquire()
				, particle_buffer_fluid.acquire()
				, next_particle_buffer_solid.acquire()
				, next_particle_buffer_fluid.acquire()
				, grid_blocks[0][solid_id].acquire()
				, grid_blocks[0][fluid_id].acquire()
				, surface_particle_buffers[solid_id].acquire()
				, surface_particle_buffers[fluid_id].acquire()
				, surface_flow_particle_buffers[rollid][surface_flow_id].acquire()
				, reinterpret_cast<void**>(&particle_buffer_solid.bin_offsets)
				, reinterpret_cast<void**>(&particle_buffer_fluid.bin_offsets)
				, reinterpret_cast<void**>(&next_particle_buffer_solid.particle_bucket_sizes)
				, reinterpret_cast<void**>(&next_particle_buffer_fluid.particle_bucket_sizes)
				, reinterpret_cast<void**>(&next_particle_buffer_solid.blockbuckets)
				, reinterpret_cast<void**>(&next_particle_buffer_fluid.blockbuckets)
			);
			
			SimpleSurfaceFlowIQCreatePointers pointers;
			pointers.scaling_solid = iq_lhs_scaling_solid_values.get_data();
			pointers.scaling_fluid = iq_lhs_scaling_fluid_values.get_data();
			
			pointers.mass_solid = iq_lhs_mass_solid_values.get_data();
			pointers.mass_fluid = iq_lhs_mass_fluid_values.get_data();
			
			pointers.gradient_solid_rows = iq_lhs_3_1_rows.get_const_data();
			pointers.gradient_solid_columns = iq_lhs_3_1_columns.get_const_data();
			pointers.gradient_solid_values = iq_lhs_gradient_solid_values.get_data();
			pointers.gradient_fluid_rows = iq_lhs_3_1_rows.get_const_data();
			pointers.gradient_fluid_columns = iq_lhs_3_1_columns.get_const_data();
			pointers.gradient_fluid_values = iq_lhs_gradient_fluid_values.get_data();
			
			pointers.coupling_solid_domain_rows = iq_lhs_3_1_rows.get_const_data();
			pointers.coupling_solid_domain_columns = iq_lhs_3_1_columns.get_const_data();
			pointers.coupling_solid_domain_values = iq_lhs_coupling_solid_values.get_data();
			pointers.coupling_fluid_rows = iq_lhs_3_1_rows.get_const_data();
			pointers.coupling_fluid_columns = iq_lhs_3_1_columns.get_const_data();
			pointers.coupling_fluid_values = iq_lhs_coupling_fluid_values.get_data();
			
			pointers.scaling_domain = iq_lhs_scaling_domain_values.get_data();
			pointers.scaling_surface = iq_lhs_scaling_surface_values.get_data();
			
			pointers.mass_domain = iq_lhs_mass_domain_values.get_data();
			pointers.mass_surface = iq_lhs_mass_surface_values.get_data();
			
			pointers.gradient_domain_rows = iq_lhs_3_1_rows.get_const_data();
			pointers.gradient_domain_columns = iq_lhs_3_1_columns.get_const_data();
			pointers.gradient_domain_values = iq_lhs_gradient_domain_values.get_data();
			pointers.gradient_surface_rows = iq_lhs_3_1_rows.get_const_data();
			pointers.gradient_surface_columns = iq_lhs_3_1_columns.get_const_data();
			pointers.gradient_surface_values = iq_lhs_gradient_surface_values.get_data();
			
			pointers.coupling_domain_rows = iq_lhs_3_1_rows.get_const_data();
			pointers.coupling_domain_columns = iq_lhs_3_1_columns.get_const_data();
			pointers.coupling_domain_values = iq_lhs_coupling_domain_values.get_data();
			pointers.coupling_surface_rows = iq_lhs_3_1_rows.get_const_data();
			pointers.coupling_surface_columns = iq_lhs_3_1_columns.get_const_data();
			pointers.coupling_surface_values = iq_lhs_coupling_surface_values.get_data();
			
			pointers.coupling_solid_surface_rows = iq_lhs_3_1_rows.get_const_data();
			pointers.coupling_solid_surface_columns = iq_lhs_3_1_columns.get_const_data();
			pointers.coupling_solid_surface_values = iq_lhs_coupling_solid_surface_values.get_data();
			
			pointers.surface_flow_coupling_domain_rows = iq_lhs_3_1_rows.get_const_data();
			pointers.surface_flow_coupling_domain_columns = iq_lhs_3_1_columns.get_const_data();
			pointers.surface_flow_coupling_domain_values = iq_lhs_surface_flow_coupling_domain_values.get_data();
			pointers.surface_flow_coupling_surface_rows = iq_lhs_3_1_rows.get_const_data();
			pointers.surface_flow_coupling_surface_columns = iq_lhs_3_1_columns.get_const_data();
			pointers.surface_flow_coupling_surface_values = iq_lhs_surface_flow_coupling_surface_values.get_data();
			
			pointers.mass_single_domain = iq_lhs_mass_single_domain_values.get_data();
			pointers.mass_single_surface = iq_lhs_mass_single_surface_values.get_data();
			
			pointers.iq_rhs = iq_rhs_array.get_data();
			pointers.iq_solve_velocity_result = iq_solve_velocity_result_array.get_data();
			
			cu_dev.compute_launch({coupling_block_count, iq::BLOCK_SIZE}, simple_surface_flow_create_iq_system, static_cast<uint32_t>(exterior_block_count), dt, particle_buffer_solid, particle_buffer_fluid, next_particle_buffer_solid, next_particle_buffer_fluid, partitions[(rollid + 1) % BIN_COUNT], partitions[rollid], grid_blocks[0][solid_id], grid_blocks[0][fluid_id], surface_particle_buffers[solid_id], surface_particle_buffers[fluid_id], surface_flow_particle_buffers[rollid][surface_flow_id], pointers);
			
			managed_memory.release(
				  particle_buffer_solid.release()
				, particle_buffer_fluid.release()
				, next_particle_buffer_solid.release()
				, next_particle_buffer_fluid.release()
				, grid_blocks[0][solid_id].release()
				, grid_blocks[0][fluid_id].release()
				, surface_particle_buffers[solid_id].release()
				, surface_particle_buffers[fluid_id].release()
				, surface_flow_particle_buffers[rollid][surface_flow_id].release()
				, particle_buffer_solid.bin_offsets_virtual
				, particle_buffer_fluid.bin_offsets_virtual
				, next_particle_buffer_solid.particle_bucket_sizes_virtual
				, next_particle_buffer_fluid.particle_bucket_sizes_virtual
				, next_particle_buffer_solid.blockbuckets_virtual
				, next_particle_buffer_fluid.blockbuckets_virtual
			);
		});
		
		cu_dev.syncStream<streamIdx::COMPUTE>();
	}

	void generate_system_matrices(
		std::shared_ptr<gko::Executor>& ginkgo_executor,
		MatrixOperations& matrix_operations,
		const int exterior_block_count,
		gko::array<float>& iq_lhs_scaling_solid_values,
		gko::array<float>& iq_lhs_scaling_fluid_values,
		gko::array<float>& iq_lhs_mass_solid_values,
		gko::array<float>& iq_lhs_mass_fluid_values,
		gko::array<int>& iq_lhs_3_1_rows,
		gko::array<int>& iq_lhs_3_1_columns,
		gko::array<float>& iq_lhs_gradient_solid_values,
		gko::array<float>& iq_lhs_gradient_fluid_values,
		gko::array<float>& iq_lhs_coupling_solid_values,
		gko::array<float>& iq_lhs_coupling_fluid_values,
		gko::array<float>& iq_solve_velocity_result_array,
		gko::array<float>& gko_identity_values,
		std::shared_ptr<gko::matrix::Dense<float>>& gko_neg_one_dense,
		std::shared_ptr<gko::matrix::Dense<float>>& gko_one_dense,
		std::vector<std::shared_ptr<gko::matrix::Csr<float, int>>>& iq_lhs_parts,
		std::vector<std::shared_ptr<gko::matrix::Csr<float, int>>>& iq_solve_velocity_parts,
		std::vector<std::shared_ptr<gko::matrix::Csr<float, int>>>& temporary_matrices,
		std::vector<std::shared_ptr<gko::matrix::Dense<float>>>& iq_rhs_parts,
		Cuda::CudaContext& cu_dev
	) override{
		const std::shared_ptr<const gko::matrix::Diagonal<float>> scaling_solid = gko::share(
			gko::matrix::Diagonal<float>::create_const(
				  ginkgo_executor
				, exterior_block_count * config::G_BLOCKVOLUME
				, iq_lhs_scaling_solid_values.as_const_view()
			)
		);
		
		const std::shared_ptr<const gko::matrix::Diagonal<float>> mass_solid = gko::share(
			gko::matrix::Diagonal<float>::create_const(
				  ginkgo_executor
				, 3 * exterior_block_count * config::G_BLOCKVOLUME
				, iq_lhs_mass_solid_values.as_const_view()
			)
		);
		
		std::shared_ptr<gko::matrix::Csr<float, int>> gradient_solid = gko::share(
			gko::matrix::Csr<float, int>::create(
				  ginkgo_executor
				, gko::dim<2>(3 * exterior_block_count * config::G_BLOCKVOLUME, exterior_block_count * config::G_BLOCKVOLUME)
				, iq_lhs_gradient_solid_values.as_view()
				, iq_lhs_3_1_columns.as_view()
				, iq_lhs_3_1_rows.as_view()
			)
		);
		
		std::shared_ptr<gko::matrix::Csr<float, int>> coupling_solid_domain = gko::share(
			gko::matrix::Csr<float, int>::create(
				  ginkgo_executor
				, gko::dim<2>(3 * exterior_block_count * config::G_BLOCKVOLUME, exterior_block_count * config::G_BLOCKVOLUME)
				, iq_lhs_coupling_solid_values.as_view()
				, iq_lhs_3_1_columns.as_view()
				, iq_lhs_3_1_rows.as_view()
			)
		);
		
		std::shared_ptr<gko::matrix::Dense<float>> velocity_solid = gko::share(
			gko::matrix::Dense<float>::create(
				  ginkgo_executor
				, gko::dim<2>(3 * exterior_block_count * config::G_BLOCKVOLUME, 1)
				, std::move(gko::array<float>::view(ginkgo_executor, 3 * exterior_block_count * config::G_BLOCKVOLUME, iq_solve_velocity_result_array.get_data()))
				, 1
			)
		);
		
		const std::shared_ptr<const gko::matrix::Diagonal<float>> gko_identity = gko::share(
			gko::matrix::Diagonal<float>::create_const(
				  ginkgo_executor
				, exterior_block_count * config::G_BLOCKVOLUME
				, gko_identity_values.as_const_view()
			)
		);
		
		const std::shared_ptr<const gko::matrix::Diagonal<float>> scaling_domain = gko::share(
			gko::matrix::Diagonal<float>::create_const(
				  ginkgo_executor
				, exterior_block_count * config::G_BLOCKVOLUME
				, iq_lhs_scaling_domain_values.as_const_view()
			)
		);
		const std::shared_ptr<const gko::matrix::Diagonal<float>> scaling_surface = gko::share(
			gko::matrix::Diagonal<float>::create_const(
				  ginkgo_executor
				, exterior_block_count * config::G_BLOCKVOLUME
				, iq_lhs_scaling_surface_values.as_const_view()
			)
		);
		
		const std::shared_ptr<const gko::matrix::Diagonal<float>> mass_domain = gko::share(
			gko::matrix::Diagonal<float>::create_const(
				  ginkgo_executor
				, 3 * exterior_block_count * config::G_BLOCKVOLUME
				, iq_lhs_mass_domain_values.as_const_view()
			)
		);
		const std::shared_ptr<const gko::matrix::Diagonal<float>> mass_surface = gko::share(
			gko::matrix::Diagonal<float>::create_const(
				  ginkgo_executor
				, 3 * exterior_block_count * config::G_BLOCKVOLUME
				, iq_lhs_mass_surface_values.as_const_view()
			)
		);
		
		std::shared_ptr<gko::matrix::Csr<float, int>> gradient_domain = gko::share(
			gko::matrix::Csr<float, int>::create(
				  ginkgo_executor
				, gko::dim<2>(3 * exterior_block_count * config::G_BLOCKVOLUME, exterior_block_count * config::G_BLOCKVOLUME)
				, iq_lhs_gradient_domain_values.as_view()
				, iq_lhs_3_1_columns.as_view()
				, iq_lhs_3_1_rows.as_view()
			)
		);
		std::shared_ptr<gko::matrix::Csr<float, int>> gradient_surface = gko::share(
			gko::matrix::Csr<float, int>::create(
				  ginkgo_executor
				, gko::dim<2>(3 * exterior_block_count * config::G_BLOCKVOLUME, exterior_block_count * config::G_BLOCKVOLUME)
				, iq_lhs_gradient_surface_values.as_view()
				, iq_lhs_3_1_columns.as_view()
				, iq_lhs_3_1_rows.as_view()
			)
		);
		
		std::shared_ptr<gko::matrix::Csr<float, int>> coupling_domain = gko::share(
			gko::matrix::Csr<float, int>::create(
				  ginkgo_executor
				, gko::dim<2>(3 * exterior_block_count * config::G_BLOCKVOLUME, exterior_block_count * config::G_BLOCKVOLUME)
				, iq_lhs_coupling_domain_values.as_view()
				, iq_lhs_3_1_columns.as_view()
				, iq_lhs_3_1_rows.as_view()
			)
		);
		std::shared_ptr<gko::matrix::Csr<float, int>> coupling_surface = gko::share(
			gko::matrix::Csr<float, int>::create(
				  ginkgo_executor
				, gko::dim<2>(3 * exterior_block_count * config::G_BLOCKVOLUME, exterior_block_count * config::G_BLOCKVOLUME)
				, iq_lhs_coupling_surface_values.as_view()
				, iq_lhs_3_1_columns.as_view()
				, iq_lhs_3_1_rows.as_view()
			)
		);
		
		std::shared_ptr<gko::matrix::Csr<float, int>> coupling_solid_surface = gko::share(
			gko::matrix::Csr<float, int>::create(
				  ginkgo_executor
				, gko::dim<2>(3 * exterior_block_count * config::G_BLOCKVOLUME, exterior_block_count * config::G_BLOCKVOLUME)
				, iq_lhs_coupling_solid_surface_values.as_view()
				, iq_lhs_3_1_columns.as_view()
				, iq_lhs_3_1_rows.as_view()
			)
		);
		
		std::shared_ptr<gko::matrix::Csr<float, int>> surface_flow_coupling_domain = gko::share(
			gko::matrix::Csr<float, int>::create(
				  ginkgo_executor
				, gko::dim<2>(3 * exterior_block_count * config::G_BLOCKVOLUME, exterior_block_count * config::G_BLOCKVOLUME)
				, iq_lhs_surface_flow_coupling_domain_values.as_view()
				, iq_lhs_3_1_columns.as_view()
				, iq_lhs_3_1_rows.as_view()
			)
		);
		std::shared_ptr<gko::matrix::Csr<float, int>> surface_flow_coupling_surface = gko::share(
			gko::matrix::Csr<float, int>::create(
				  ginkgo_executor
				, gko::dim<2>(3 * exterior_block_count * config::G_BLOCKVOLUME, exterior_block_count * config::G_BLOCKVOLUME)
				, iq_lhs_surface_flow_coupling_surface_values.as_view()
				, iq_lhs_3_1_columns.as_view()
				, iq_lhs_3_1_rows.as_view()
			)
		);
		
		std::shared_ptr<gko::matrix::Dense<float>> velocity_domain = gko::share(
			gko::matrix::Dense<float>::create(
				  ginkgo_executor
				, gko::dim<2>(3 * exterior_block_count * config::G_BLOCKVOLUME, 1)
				, std::move(gko::array<float>::view(ginkgo_executor, 3 * exterior_block_count * config::G_BLOCKVOLUME, iq_solve_velocity_result_array.get_data() + 3 * exterior_block_count * config::G_BLOCKVOLUME))
				, 1
			)
		);
		
		std::shared_ptr<gko::matrix::Dense<float>> velocity_surface = gko::share(
			gko::matrix::Dense<float>::create(
				  ginkgo_executor
				, gko::dim<2>(3 * exterior_block_count * config::G_BLOCKVOLUME, 1)
				, std::move(gko::array<float>::view(ginkgo_executor, 3 * exterior_block_count * config::G_BLOCKVOLUME, iq_solve_velocity_result_array.get_data() + 6 * exterior_block_count * config::G_BLOCKVOLUME))
				, 1
			)
		);
		
		//Create solve velocity and rhs
		
		//NOTE: As coupling equations (4)*, (5)* and (6)* in our system is multiplied with -1 we have to swap the signs of the coupling part in solve velocities
		/*
			solve_velocity[0][0] = -dt * M^-1_solid * G_solid
			solve_velocity[0][3] = -dt * M^-1_solid * H^T_solid_domain
			solve_velocity[0][4] = -dt * M^-1_solid * H^T_solid_surface
			
			solve_velocity[1][1] = -dt * M^-1_domain * G_domain
			solve_velocity[1][3] = dt * M^-1_domain * H^T_domain
			solve_velocity[1][5] = -dt * M^-1_domain * C^T_domain
			
			solve_velocity[2][2] = -dt * M^-1_surface * G_surface
			solve_velocity[2][4] = dt * M^-1_surface * H^T_surface
			solve_velocity[2][5] = dt * M^-1_surface * C^T_surface
		*/
		mass_solid->apply(gradient_solid, iq_solve_velocity_parts[0]);
		iq_solve_velocity_parts[0]->scale(gko_neg_one_dense);
		mass_solid->apply(coupling_solid_domain, iq_solve_velocity_parts[1]);
		iq_solve_velocity_parts[1]->scale(gko_neg_one_dense);
		mass_solid->apply(coupling_solid_surface, iq_solve_velocity_parts[2]);
		iq_solve_velocity_parts[2]->scale(gko_neg_one_dense);
		
		mass_domain->apply(gradient_domain, iq_solve_velocity_parts[3]);
		iq_solve_velocity_parts[3]->scale(gko_neg_one_dense);
		mass_domain->apply(coupling_domain, iq_solve_velocity_parts[4]);
		mass_domain->apply(surface_flow_coupling_domain, iq_solve_velocity_parts[5]);
		iq_solve_velocity_parts[5]->scale(gko_neg_one_dense);
		
		mass_surface->apply(gradient_surface, iq_solve_velocity_parts[6]);
		iq_solve_velocity_parts[6]->scale(gko_neg_one_dense);
		mass_surface->apply(coupling_surface, iq_solve_velocity_parts[7]);
		mass_surface->apply(surface_flow_coupling_surface, iq_solve_velocity_parts[8]);
		
		/*
			rhs[0](p_solid) = S_solid/dt * p_solid + G_solid^T * v_solid
			rhs[1](p_domain) = S_domain/dt * p_domain + G_domain^T * v_domain
			rhs[2](p_surface) = S_surface/dt * p_surface + G_surface^T * v_surface
			
			rhs[3] = H_solid_domain * v_solid  - H_domain  * v_domain 
			rhs[4] = H_solid_surface * v_solid  - H_surface * v_surface
			
			rhs[5] = C_domain  * v_domain  - C_surface * v_surface
		*/
		scaling_solid->apply(iq_rhs_parts[0], iq_rhs_parts[0]);
		temporary_matrices[0]->copy_from(std::move(gradient_solid->transpose()));
		temporary_matrices[0]->apply(gko_one_dense, velocity_solid, gko_one_dense, iq_rhs_parts[0]);
		
		scaling_domain->apply(iq_rhs_parts[1], iq_rhs_parts[1]);
		temporary_matrices[0]->copy_from(std::move(gradient_domain->transpose()));
		temporary_matrices[0]->apply(gko_one_dense, velocity_domain, gko_one_dense, iq_rhs_parts[1]);
		
		scaling_surface->apply(iq_rhs_parts[2], iq_rhs_parts[2]);
		temporary_matrices[0]->copy_from(std::move(gradient_surface->transpose()));
		temporary_matrices[0]->apply(gko_one_dense, velocity_surface, gko_one_dense, iq_rhs_parts[2]);
		
		temporary_matrices[0]->copy_from(std::move(coupling_solid_domain->transpose()));
		temporary_matrices[0]->apply(velocity_solid, iq_rhs_parts[3]);
		
		temporary_matrices[0]->copy_from(std::move(coupling_solid_surface->transpose()));
		temporary_matrices[0]->apply(velocity_solid, iq_rhs_parts[4]);
		
		temporary_matrices[0]->copy_from(std::move(coupling_domain->transpose()));
		temporary_matrices[0]->apply(gko_neg_one_dense, velocity_domain, gko_one_dense, iq_rhs_parts[3]);
		
		temporary_matrices[0]->copy_from(std::move(coupling_surface->transpose()));
		temporary_matrices[0]->apply(gko_neg_one_dense, velocity_surface, gko_one_dense, iq_rhs_parts[4]);
		
		temporary_matrices[0]->copy_from(std::move(surface_flow_coupling_domain->transpose()));
		temporary_matrices[0]->apply(velocity_domain, iq_rhs_parts[5]);
		
		temporary_matrices[0]->copy_from(std::move(surface_flow_coupling_surface->transpose()));
		temporary_matrices[0]->apply(gko_neg_one_dense, velocity_surface, gko_one_dense, iq_rhs_parts[5]);
		
		//Create lhs
		
		/*
			lhs[0][0] = S_solid/dt + dt * G_solid^T * M^-1_solid * G_solid
			lhs[0][3] = dt * G_solid^T * M^-1_solid * H^T_solid_domain
			lhs[0][4] = dt * G_solid^T * M^-1_solid * H^T_solid_surface
			
		*/
		temporary_matrices[0]->copy_from(std::move(gradient_solid->transpose()));
		
		//NOTE: Using iq_lhs_parts[9] and iq_lhs_parts[10] as scratch
		matrix_operations.matrix_matrix_multiplication_a_at_with_diagonal(ginkgo_executor, exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_solid, gradient_solid, iq_lhs_parts[0], cu_dev);
		iq_lhs_parts[9]->copy_from(std::move(scaling_solid));
		iq_lhs_parts[10]->copy_from(gko_identity);
		iq_lhs_parts[10]->apply(gko_one_dense, iq_lhs_parts[9], gko_one_dense, iq_lhs_parts[0]);
		
		matrix_operations.matrix_matrix_multiplication_with_diagonal<true>(ginkgo_executor, exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_solid, coupling_solid_domain, iq_lhs_parts[1], cu_dev);
		
		matrix_operations.matrix_matrix_multiplication_with_diagonal<true>(ginkgo_executor, exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_solid, coupling_solid_surface, iq_lhs_parts[2], cu_dev);
		
		/*
			lhs[1][1] = S_domain/dt + dt * G_domain^T * M^-1_domain * G_domain
			lhs[1][3] = -dt * G_domain^T * M^-1_domain * H^T_domain
			lhs[1][5] = dt * G_domain^T * M^-1_domain * C^T_domain
			
		*/
		temporary_matrices[0]->copy_from(std::move(gradient_domain->transpose()));
		
		//NOTE: Using iq_lhs_parts[9] and iq_lhs_parts[10] as scratch
		//NOTE: iq_lhs_parts[10] already set to identity
		matrix_operations.matrix_matrix_multiplication_a_at_with_diagonal(ginkgo_executor, exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_domain, gradient_domain, iq_lhs_parts[3], cu_dev);
		iq_lhs_parts[9]->copy_from(std::move(scaling_domain));
		iq_lhs_parts[10]->apply(gko_one_dense, iq_lhs_parts[9], gko_one_dense, iq_lhs_parts[3]);
		
		matrix_operations.matrix_matrix_multiplication_with_diagonal<true>(ginkgo_executor, exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_domain, coupling_domain, iq_lhs_parts[4], cu_dev);
		iq_lhs_parts[4]->scale(gko_neg_one_dense);
		
		matrix_operations.matrix_matrix_multiplication_with_diagonal<true>(ginkgo_executor, exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_domain, surface_flow_coupling_domain, iq_lhs_parts[5], cu_dev);
		
		
		/*
			lhs[2][2] = S_surface/dt + dt * G_surface^T * M^-1_surface * G_surface
			lhs[2][4] = -dt * G_surface^T * M^-1_surface * H^T_surface
			lhs[2][5] = -dt * G_surface^T * M^-1_surface * C^T_surface
			
		*/
		temporary_matrices[0]->copy_from(std::move(gradient_surface->transpose()));
		
		//NOTE: Using iq_lhs_parts[9] and iq_lhs_parts[10] as scratch
		//NOTE: iq_lhs_parts[10] already set to identity
		matrix_operations.matrix_matrix_multiplication_a_at_with_diagonal(ginkgo_executor, exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_surface, gradient_surface, iq_lhs_parts[6], cu_dev);
		iq_lhs_parts[9]->copy_from(std::move(scaling_surface));
		iq_lhs_parts[10]->apply(gko_one_dense, iq_lhs_parts[9], gko_one_dense, iq_lhs_parts[6]);
		
		matrix_operations.matrix_matrix_multiplication_with_diagonal<true>(ginkgo_executor, exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_surface, coupling_surface, iq_lhs_parts[7], cu_dev);
		iq_lhs_parts[7]->scale(gko_neg_one_dense);
		
		matrix_operations.matrix_matrix_multiplication_with_diagonal<true>(ginkgo_executor, exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_surface, surface_flow_coupling_surface, iq_lhs_parts[8], cu_dev);
		iq_lhs_parts[8]->scale(gko_neg_one_dense);
		
		/*
			lhs[3][3] += dt * H_solid_domain * M^-1_solid * H^T_solid_domain
			lhs[3][4] = dt * H_solid_domain * M^-1_solid * H^T_solid_surface
			lhs[4][4] += dt * H_solid_surface * M^-1_solid * H^T_solid_surface
			
		*/
		temporary_matrices[0]->copy_from(std::move(coupling_solid_domain->transpose()));
		matrix_operations.matrix_matrix_multiplication_a_at_with_diagonal(ginkgo_executor, exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_solid, coupling_solid_domain, iq_lhs_parts[11], cu_dev);
		
		matrix_operations.matrix_matrix_multiplication_with_diagonal<true>(ginkgo_executor, exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_solid, coupling_solid_surface, iq_lhs_parts[12], cu_dev);
		
		temporary_matrices[0]->copy_from(std::move(coupling_solid_surface->transpose()));
		matrix_operations.matrix_matrix_multiplication_a_at_with_diagonal(ginkgo_executor, exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_solid, coupling_solid_surface, iq_lhs_parts[17], cu_dev);
		
		
		
		/*
			lhs[3][3] += dt * H_domain * M^-1_domain * H^T_domain
			lhs[3][5] = -dt * H_domain * M^-1_domain * C^T_domain
		*/
		//NOTE: Using iq_lhs_parts[9] and iq_lhs_parts[10] as scratch
		//NOTE: iq_lhs_parts[10] already set to identity
		temporary_matrices[0]->copy_from(std::move(coupling_domain->transpose()));
		matrix_operations.matrix_matrix_multiplication_a_at_with_diagonal(ginkgo_executor, exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_domain, coupling_domain, iq_lhs_parts[9], cu_dev);
		iq_lhs_parts[10]->apply(gko_one_dense, iq_lhs_parts[9], gko_one_dense, iq_lhs_parts[11]);
		
		matrix_operations.matrix_matrix_multiplication_with_diagonal<true>(ginkgo_executor, exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_domain, surface_flow_coupling_domain, iq_lhs_parts[13], cu_dev);
		iq_lhs_parts[13]->scale(gko_neg_one_dense);
		
		/*
			lhs[4][4] += dt * H_surface * M^-1_surface * H^T_surface
			lhs[4][5] = dt * H_surface * M^-1_surface * C^T_surface
		*/
		//NOTE: Using iq_lhs_parts[9] and iq_lhs_parts[10] as scratch
		//NOTE: iq_lhs_parts[10] already set to identity
		temporary_matrices[0]->copy_from(std::move(coupling_surface->transpose()));
		matrix_operations.matrix_matrix_multiplication_a_at_with_diagonal(ginkgo_executor, exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_surface, coupling_surface, iq_lhs_parts[9], cu_dev);
		iq_lhs_parts[10]->apply(gko_one_dense, iq_lhs_parts[9], gko_one_dense, iq_lhs_parts[17]);
		
		matrix_operations.matrix_matrix_multiplication_with_diagonal<true>(ginkgo_executor, exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_surface, surface_flow_coupling_surface, iq_lhs_parts[18], cu_dev);
		
		/*
			lhs[5][5] += dt * C_domain * M^-1_domain * C^T_domain
			lhs[5][5] += dt * C_surface * M^-1_surface * C^T_surface
			
		*/
		temporary_matrices[0]->copy_from(std::move(surface_flow_coupling_domain->transpose()));
		matrix_operations.matrix_matrix_multiplication_a_at_with_diagonal(ginkgo_executor, exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_domain, surface_flow_coupling_domain, iq_lhs_parts[23], cu_dev);

		
		//NOTE: Using iq_lhs_parts[9] and iq_lhs_parts[10] as scratch
		//NOTE: iq_lhs_parts[10] already set to identity
		temporary_matrices[0]->copy_from(std::move(surface_flow_coupling_surface->transpose()));
		matrix_operations.matrix_matrix_multiplication_a_at_with_diagonal(ginkgo_executor, exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_surface, surface_flow_coupling_surface, iq_lhs_parts[9], cu_dev);
		iq_lhs_parts[10]->apply(gko_one_dense, iq_lhs_parts[9], gko_one_dense, iq_lhs_parts[23]);
	
		/*
			Current state:
			{
				A11 0   0   A14 A15 0  
				0   A22 0   A24 0   A26
				0   0   A33 0   A35 A36
				-   -   0   A44 A45 A46
				-   0   -   -   A55 A56
				0   -   -   -   -   A66
			}
		*/
		
		//Fill transposed
		/*
			lhs[3][0] = lhs[0][3]^T
			lhs[3][1] = lhs[1][3]^T
			lhs[4][0] = lhs[0][4]^T
			lhs[4][2] = lhs[2][4]^T
			lhs[4][3] = lhs[3][4]^T
			lhs[5][1] = lhs[1][5]^T
			lhs[5][2] = lhs[2][5]^T
			lhs[5][3] = lhs[3][5]^T
			lhs[5][4] = lhs[4][5]^T
		*/
		iq_lhs_parts[9]->copy_from(std::move(iq_lhs_parts[1]->transpose()));
		iq_lhs_parts[10]->copy_from(std::move(iq_lhs_parts[4]->transpose()));
		iq_lhs_parts[14]->copy_from(std::move(iq_lhs_parts[2]->transpose()));
		iq_lhs_parts[15]->copy_from(std::move(iq_lhs_parts[7]->transpose()));
		iq_lhs_parts[16]->copy_from(std::move(iq_lhs_parts[12]->transpose()));
		iq_lhs_parts[19]->copy_from(std::move(iq_lhs_parts[5]->transpose()));
		iq_lhs_parts[20]->copy_from(std::move(iq_lhs_parts[8]->transpose()));
		iq_lhs_parts[21]->copy_from(std::move(iq_lhs_parts[13]->transpose()));
		iq_lhs_parts[22]->copy_from(std::move(iq_lhs_parts[18]->transpose()));
		
		ginkgo_executor->synchronize();
	}
	
	void mass_transfer(
		managed_memory_type& managed_memory,
		std::array<std::vector<particle_buffer_t>, BIN_COUNT>& particle_bins,
		std::vector<Partition<1>>& partitions,
		std::array<std::vector<GridBuffer>, BIN_COUNT>& grid_blocks,
		std::array<std::vector<SurfaceFlowParticleBuffer>, BIN_COUNT>& surface_flow_particle_buffers,
		const int rollid,
		const int partition_block_count,
		const int solid_id,
		const int fluid_id,
		const int surface_flow_id,
		std::shared_ptr<gko::Executor>& ginkgo_executor,
		const int exterior_block_count,
		std::shared_ptr<gko::matrix::Dense<float>>& iq_result,
		gko::array<float>& gko_identity_values,
		std::shared_ptr<gko::matrix::Dense<float>>& gko_neg_one_dense,
		std::vector<std::shared_ptr<gko::matrix::Csr<float, int>>>& iq_lhs_parts,
		Cuda::CudaContext& cu_dev
	) override{
		/*
		//Calculate delta density
		const std::shared_ptr<const gko::matrix::Diagonal<float>> gko_identity = gko::share(
			gko::matrix::Diagonal<float>::create_const(
				  ginkgo_executor
				, exterior_block_count * config::G_BLOCKVOLUME
				, gko_identity_values.as_const_view()
			)
		);
		
		const std::shared_ptr<const gko::matrix::Diagonal<float>> scaling_domain = gko::share(
			gko::matrix::Diagonal<float>::create_const(
				  ginkgo_executor
				, exterior_block_count * config::G_BLOCKVOLUME
				, iq_lhs_scaling_domain_values.as_const_view()
			)
		);
		const std::shared_ptr<const gko::matrix::Diagonal<float>> scaling_surface = gko::share(
			gko::matrix::Diagonal<float>::create_const(
				  ginkgo_executor
				, exterior_block_count * config::G_BLOCKVOLUME
				, iq_lhs_scaling_surface_values.as_const_view()
			)
		);
		
		const std::shared_ptr<const gko::matrix::Diagonal<float>> mass_single_domain = gko::share(
			gko::matrix::Diagonal<float>::create_const(
				  ginkgo_executor
				, exterior_block_count * config::G_BLOCKVOLUME
				, iq_lhs_mass_single_domain_values.as_const_view()
			)
		);
		const std::shared_ptr<const gko::matrix::Diagonal<float>> mass_single_surface = gko::share(
			gko::matrix::Diagonal<float>::create_const(
				  ginkgo_executor
				, exterior_block_count * config::G_BLOCKVOLUME
				, iq_lhs_mass_single_surface_values.as_const_view()
			)
		);
		
		std::shared_ptr<gko::matrix::Dense<float>> pressure_interface = gko::share(
			gko::matrix::Dense<float>::create(
				  ginkgo_executor
				, gko::dim<2>(exterior_block_count * config::G_BLOCKVOLUME, 1)
				, std::move(gko::array<float>::view(ginkgo_executor, exterior_block_count * config::G_BLOCKVOLUME, iq_result->get_values() + 5 * exterior_block_count * config::G_BLOCKVOLUME))
				, 1
			)
		);
		
		std::shared_ptr<gko::matrix::Dense<float>> delta_density_domain = gko::share(
			gko::matrix::Dense<float>::create(
				  ginkgo_executor
				, gko::dim<2>(exterior_block_count * config::G_BLOCKVOLUME, 1)
			)
		);
		
		std::shared_ptr<gko::matrix::Dense<float>> delta_density_surface = gko::share(
			gko::matrix::Dense<float>::create(
				  ginkgo_executor
				, gko::dim<2>(exterior_block_count * config::G_BLOCKVOLUME, 1)
			)
		);
		
		//delta_density_domain = 1/dt * M * D * h^n+1_interface_solid_surface;
		//delta_density_surface = -1/dt * M * D * h^n+1_interface_solid_surface;
		//NOTE: Using iq_lhs_parts[0] as scratch
		scaling_domain->apply(mass_single_domain, iq_lhs_parts[0]);
		iq_lhs_parts[0]->apply(pressure_interface, delta_density_domain);
		
		scaling_surface->apply(mass_single_surface, iq_lhs_parts[0]);
		iq_lhs_parts[0]->apply(pressure_interface, delta_density_surface);
		delta_density_domain->scale(gko_neg_one_dense);
		
		ginkgo_executor->synchronize();
		
		//Clamp delta density to smaller current density
		cu_dev.compute_launch({exterior_block_count, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, simple_surface_flow_get_min_delta_density<iq::NUM_ROWS_PER_BLOCK>, delta_density_domain->get_values(), delta_density_surface->get_const_values(), iq_lhs_mass_single_domain_values.get_const_data(), iq_lhs_mass_single_surface_values.get_const_data());
		
		cu_dev.syncStream<streamIdx::COMPUTE>();
		
		//delta_density_domain now holds the delta density we use
		
		//Calculate pressure difference
		iq_lhs_parts[0]->copy_from(gko_identity);
		scaling_domain->inverse_apply(iq_lhs_parts[0], iq_lhs_parts[0]);
		mass_single_domain->inverse_apply(iq_lhs_parts[0], iq_lhs_parts[0]);
		iq_lhs_parts[0]->apply(delta_density_domain, pressure_interface);
		
		ginkgo_executor->synchronize();
		
		//Apply mass transfer
		match(particle_bins[rollid][solid_id], particle_bins[rollid][fluid_id])([
			this,
			&rollid,
			&partition_block_count,
			&solid_id,
			&fluid_id,
			&surface_flow_id,
			&managed_memory,
			&particle_bins,
			&partitions,
			&grid_blocks,
			&surface_flow_particle_buffers,
			&delta_density_domain,
			&cu_dev
		](auto& particle_buffer_solid, auto& particle_buffer_fluid) {
			auto& next_particle_buffer_solid = get<typename std::decay_t<decltype(particle_buffer_solid)>>(particle_bins[(rollid + 1) % BIN_COUNT][solid_id]);
			auto& next_particle_buffer_fluid = get<typename std::decay_t<decltype(particle_buffer_fluid)>>(particle_bins[(rollid + 1) % BIN_COUNT][fluid_id]);
			
			particle_buffer_solid.bin_offsets = particle_buffer_solid.bin_offsets_virtual;
			particle_buffer_fluid.bin_offsets = particle_buffer_fluid.bin_offsets_virtual;
			next_particle_buffer_solid.particle_bucket_sizes = next_particle_buffer_solid.particle_bucket_sizes_virtual;
			next_particle_buffer_fluid.particle_bucket_sizes = next_particle_buffer_fluid.particle_bucket_sizes_virtual;
			next_particle_buffer_solid.blockbuckets = next_particle_buffer_solid.blockbuckets_virtual;
			next_particle_buffer_fluid.blockbuckets = next_particle_buffer_fluid.blockbuckets_virtual;
			managed_memory.acquire<MemoryType::DEVICE>(
				  particle_buffer_solid.acquire()
				, next_particle_buffer_solid.acquire()
				, particle_buffer_fluid.acquire()
				, next_particle_buffer_fluid.acquire()
				, surface_flow_particle_buffers[rollid][surface_flow_id].acquire()
				, reinterpret_cast<void**>(&particle_buffer_solid.bin_offsets)
				, reinterpret_cast<void**>(&particle_buffer_fluid.bin_offsets)
				, reinterpret_cast<void**>(&next_particle_buffer_solid.particle_bucket_sizes)
				, reinterpret_cast<void**>(&next_particle_buffer_fluid.particle_bucket_sizes)
				, reinterpret_cast<void**>(&next_particle_buffer_solid.blockbuckets)
				, reinterpret_cast<void**>(&next_particle_buffer_fluid.blockbuckets)
			);
			
			cu_dev.compute_launch({partition_block_count, iq::BLOCK_SIZE}, simple_surface_flow_mass_transfer, particle_buffer_solid, get<typename std::decay_t<decltype(particle_buffer_solid)>>(particle_bins[(rollid + 1) % BIN_COUNT][solid_id]), particle_buffer_fluid, get<typename std::decay_t<decltype(particle_buffer_fluid)>>(particle_bins[(rollid + 1) % BIN_COUNT][fluid_id]), partitions[(rollid + 1) % BIN_COUNT], partitions[rollid], grid_blocks[0][fluid_id], surface_flow_particle_buffers[rollid][surface_flow_id], delta_density_domain->get_const_values());
		
			managed_memory.release(
				  particle_buffer_solid.release()
				, next_particle_buffer_solid.release()
				, particle_buffer_fluid.release()
				, next_particle_buffer_fluid.release()
				, surface_flow_particle_buffers[rollid][surface_flow_id].release()
				, particle_buffer_solid.bin_offsets_virtual
				, particle_buffer_fluid.bin_offsets_virtual
				, next_particle_buffer_solid.particle_bucket_sizes_virtual
				, next_particle_buffer_fluid.particle_bucket_sizes_virtual
				, next_particle_buffer_solid.blockbuckets_virtual
				, next_particle_buffer_fluid.blockbuckets_virtual
			);
		});
		
		cu_dev.syncStream<streamIdx::COMPUTE>();
		*/
	}
	
	
	void update_after_solve(
		managed_memory_type& managed_memory,
		std::array<std::vector<particle_buffer_t>, BIN_COUNT>& particle_bins,
		std::vector<Partition<1>>& partitions,
		std::array<std::vector<GridBuffer>, BIN_COUNT>& grid_blocks,
		const int rollid,
		const int exterior_block_count,
		const int partition_block_count,
		const int solid_id,
		const int fluid_id,
		const std::shared_ptr<gko::matrix::Dense<float>>& iq_solve_velocity_result,
		const std::shared_ptr<gko::matrix::Dense<float>>& iq_result,
		Cuda::CudaContext& cu_dev
	) override{
		//Update velocity and strain
		match(particle_bins[rollid][solid_id])([
			this,
			&rollid,
			&exterior_block_count,
			&partition_block_count,
			&solid_id,
			&fluid_id,
			&managed_memory,
			&particle_bins,
			&partitions,
			&grid_blocks,
			&iq_solve_velocity_result,
			&iq_result,
			&cu_dev
		](auto& particle_buffer_solid) {
			auto& next_particle_buffer_solid = get<typename std::decay_t<decltype(particle_buffer_solid)>>(particle_bins[(rollid + 1) % BIN_COUNT][solid_id]);
			
			particle_buffer_solid.bin_offsets = particle_buffer_solid.bin_offsets_virtual;
			next_particle_buffer_solid.particle_bucket_sizes = next_particle_buffer_solid.particle_bucket_sizes_virtual;
			next_particle_buffer_solid.blockbuckets = next_particle_buffer_solid.blockbuckets_virtual;
			managed_memory.acquire<MemoryType::DEVICE>(
				  particle_buffer_solid.acquire()
				, next_particle_buffer_solid.acquire()
				, grid_blocks[0][solid_id].acquire()
				, grid_blocks[0][fluid_id].acquire()
				, reinterpret_cast<void**>(&particle_buffer_solid.bin_offsets)
				, reinterpret_cast<void**>(&next_particle_buffer_solid.particle_bucket_sizes)
				, reinterpret_cast<void**>(&next_particle_buffer_solid.blockbuckets)
			);
			
			cu_dev.compute_launch({partition_block_count, iq::BLOCK_SIZE}, iq::update_velocity_and_strain, particle_buffer_solid, get<typename std::decay_t<decltype(particle_buffer_solid)>>(particle_bins[(rollid + 1) % BIN_COUNT][solid_id]), partitions[(rollid + 1) % BIN_COUNT], partitions[rollid], grid_blocks[0][solid_id], grid_blocks[0][fluid_id], iq_solve_velocity_result->get_const_values(), iq_solve_velocity_result->get_const_values() + 3 * exterior_block_count * config::G_BLOCKVOLUME, iq_result->get_const_values());
		
			managed_memory.release(
				  particle_buffer_solid.release()
				, next_particle_buffer_solid.release()
				, grid_blocks[0][solid_id].release()
				, grid_blocks[0][fluid_id].release()
				, particle_buffer_solid.bin_offsets_virtual
				, next_particle_buffer_solid.particle_bucket_sizes_virtual
				, next_particle_buffer_solid.blockbuckets_virtual
			);
		});
		
		cu_dev.syncStream<streamIdx::COMPUTE>();
	}
};

using surface_flow_model_t = variant<SimpleSurfaceFlowModel>;

}// namespace mn

#endif