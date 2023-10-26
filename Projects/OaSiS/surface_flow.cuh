#ifndef SURFACE_FLOW_CUH
#define SURFACE_FLOW_CUH

#include "particle_buffer.cuh"
#include "surface_flow_kernels.cuh"

namespace mn {
	
//NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables, readability-identifier-naming) Check is buggy and reports variable errors for template arguments
using SurfaceFlowParticleBufferData  = Structural<StructuralType::DENSE, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::SUM_POW2_ALIGN>, ParticleBinDomain, attrib_layout::SOA, f32_, f32_, f32_, f32_>;//mass, velocity
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

virtual class SurfaceFlowModel {
public:

	virtual size_t get_lhs_matrix_size_x() = delete;
	
	virtual size_t get_lhs_matrix_size_y() = delete;
	
	virtual size_t get_lhs_matrix_total_block_count() = delete;
	
	virtual size_t get_solve_velocity_matrix_size_x() = delete;
	
	virtual size_t get_solve_velocity_matrix_size_y() = delete;
	
	virtual size_t get_solve_velocity_matrix_total_block_count() = delete;
	
	virtual size_t* get_lhs_num_blocks_per_row_ptr() = delete;
	
	virtual size_t** get_lhs_block_offsets_per_row_ptr() = delete;
	
	virtual size_t* get_solve_velocity_num_blocks_per_row_ptr() = delete;
	
	virtual size_t** get_solve_velocity_block_offsets_per_row_ptr() = delete;
	
	virtual void fill_matrices() = delete;
	
	virtual void generate_system_matrices() = delete;
	
	virtual void mass_transfer() = delete;
	
	virtual void update_after_solve() = delete;
};

//TODO: Rename; Also at other palces
//FIXME: Boundary condition assumed to be zero, but included in matrix
/*
	Having momentum and mass balance equations and coupling for domain only, for surface flow only and for combined surface flow and domain.
	That way surface and domain are continuous together by Navier-Stokes. Also domain and surface are coupled and move continuous on their own
	Resulting we get a pressure difference (p_surface - p_domain) per node which we can use for mass transfer.
	
	The matrices look like this:
	Non-symmetric version:
	1/dt * M_solid     G_solid             -                  -                   -                  -                   -                  -                   -H^T_solid          -H^T_solid               v^n+1_solid       1/dt * M_solid   * v^n_solid  
	G^T_solid          -1/dt * S_solid     -                  -                   -                  -                   -                  -                   -                   -                        p^n+1_solid       -1/dt * S_solid   * p^n_solid  
	-                  -                   1/dt * M_fluid     G_fluid             -                  -                   -                  -                   H^T_fluid           H^T_fluid                v^n+1_fluid       1/dt * M_fluid   * v^n_fluid  
	-                  -                   G^T_fluid          -1/dt * S_fluid     -                  -                   -                  -                   -                   -                        p^n+1_fluid       -1/dt * S_fluid   * p^n_fluid  
	-                  -                   -                  -                   1/dt * M_domain    G_domain            -                  -                   H^T_domain          -                        v^n+1_domain      1/dt * M_domain  * v^n_domain 
	-                  -                   -                  -                   G^T_domain         -1/dt * S_domain    -                  -                   -                   -                        p^n+1_domain      -1/dt * S_domain  * p^n_domain 
	-                  -                   -                  -                   -                  -                   1/dt * M_surface   G_surface           -                   H^T_surface              v^n+1_surface     1/dt * M_solid   * v^n_solid  
	-                  -                   -                  -                   -                  -                   G^T_surface        -1/dt * S_surface   -                   -                        p^n+1_surface     -1/dt * S_surface * p^n_surface
	-H_solid           -                   H_fluid            -                   -                  -                   -                  -                   -                   -                        h^n+1_domain      0                              
	                                                                                                                                                                                                      *  h^n+1_surface  =  
	Symmetric version:
	(1)* = dt * M^-1_solid   * G^T_solid   * (1) - (2);
	(2)* = dt * M^-1_fluid   * G^T_fluid   * (3) - (4);
	(3)* = dt * M^-1_domain  * G^T_domain  * (5) - (6);
	(4)* = dt * M^-1_surface * G^T_surface * (7) - (8);
	(5)* = -dt * M^-1_solid   * H_solid   * (1) + dt * M^-1_fluid   * H_fluid   * (3) + dt * M^-1_domain  * H_domain  * (5);
	(6)* = -dt * M^-1_solid   * H_solid   * (1) + dt * M^-1_fluid   * H_fluid   * (3) + dt * M^-1_surface * H_surface * (7);
	------------------------------------------------
	A_11    -       -       -       A_15    A_16         p^n+1_solid       S_solid/dt   * p^n_solid   - G^T_solid   * v^n_solid  
	-       A_22    -       -       A_25    A_26         p^n+1_fluid       S_fluid/dt   * p^n_fluid   - G^T_fluid   * v^n_fluid  
	-       -       A_33    -       A_35    -            p^n+1_domain      S_domain/dt  * p^n_domain  - G^T_domain  * v^n_domain 
	-       -       -       A_44    -       A_46         p^n+1_surface     S_surface/dt * p^n_surface - G^T_surface * v^n_surface
	A^T_15  A^T_25  A^T_35  -       A_55    A_56         h^n+1_domain      H_solid   * v^n_solid   - H_fluid   * v^n_fluid   - H_domain  * v^n_domain 
	A^T_16  A^T_26  -       A^T_46  A^T_56  A_66      *  h^n+1_surface  =  H_solid   * v^n_solid   - H_fluid   * v^n_fluid   - H_surface * v^n_surface
	
	A_11 = dt * G^T_solid   * M^-1_solid   * G_solid  
	A_15 = A_16 = -dt * G^T_solid   M^-1_solid   * H^T_solid  
	A_22 = dt * G^T_fluid   * M^-1_fluid   * G_fluid  
	A_25 = A_26 = dt * G^T_fluid   * M^-1_fluid   * H^T_fluid  
	A_33 = dt * G^T_domain  * M^-1_domain  * G_domain 
	A_35 = dt * G^T_domain  * M^-1_domain  * H^T_domain 
	A_44 = dt * G^T_surface * M^-1_surface * G_surface
	A_46 = dt * G^T_surface * M^-1_surface * H^T_surface
	A_55 = dt * H_solid   * M^-1_solid   * H^T_solid   + dt * H_fluid   * M^-1_fluid   * H^T_fluid   + dt * H_domain  * M^-1_domain  * H^T_domain 
	A_56 = dt * H_solid   * M^-1_solid   * H^T_solid   + dt * H_fluid   * M^-1_fluid   * H^T_fluid  
	A_66 = dt * H_solid   * M^-1_solid   * H^T_solid   + dt * H_fluid   * M^-1_fluid   * H^T_fluid   + dt * H_surface * M^-1_surface * H^T_surface
*/
class SimpleSurfaceFlowModel : public SurfaceFlowModel{
	
	constexpr size_t LHS_MATRIX_SIZE_Y = 6;
	constexpr size_t LHS_MATRIX_SIZE_X = 6;

	constexpr size_t LHS_MATRIX_TOTAL_BLOCK_COUNT = 20;
	__device__ const std::array<size_t, LHS_MATRIX_SIZE_Y> lhs_num_blocks_per_row = {
		  3
		, 3
		, 2
		, 2
		, 5
		, 5
	};
	__device__ const std::array<std::array<size_t, LHS_MATRIX_SIZE_X>, LHS_MATRIX_SIZE_Y> lhs_block_offsets_per_row = {{
		  {0, 4, 5, std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()}
		, {1, 4, 5, std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()}
		, {2, 4, std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()}
		, {3, 5, std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()}
		, {0, 1, 2, 4, 5, std::numeric_limits<size_t>::max()}
		, {0, 1, 3, 4, 5, std::numeric_limits<size_t>::max()}
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
	
	size_t get_lhs_matrix_size_x(){
		return LHS_MATRIX_SIZE_X;
	}
	
	size_t get_lhs_matrix_size_y(){
		return LHS_MATRIX_SIZE_Y;
	}
	
	size_t get_lhs_matrix_total_block_count(){
		return LHS_MATRIX_TOTAL_BLOCK_COUNT;
	}
	
	size_t get_solve_velocity_matrix_size_x(){
		return SOLVE_VELOCITY_MATRIX_SIZE_X;
	}
	
	size_t get_solve_velocity_matrix_size_y(){
		return SOLVE_VELOCITY_MATRIX_SIZE_Y;
	}
	
	size_t get_solve_velocity_matrix_total_block_count(){
		return SOLVE_VELOCITY_MATRIX_TOTAL_BLOCK_COUNT;
	}
	
	size_t* get_lhs_num_blocks_per_row_ptr(){
		size_t* lhs_num_blocks_per_row_ptr;
		cudaGetSymbolAddress(reinterpret_cast<void**>(&lhs_num_blocks_per_row_ptr), lhs_num_blocks_per_row);
		
		return lhs_num_blocks_per_row_ptr;
	}
	
	size_t** get_lhs_block_offsets_per_row_ptr(){
		size_t** lhs_block_offsets_per_row_ptr;
		cudaGetSymbolAddress(reinterpret_cast<void**>(&lhs_block_offsets_per_row_ptr), lhs_block_offsets_per_row);
		
		return lhs_block_offsets_per_row_ptr;
	}
	
	size_t* get_solve_velocity_num_blocks_per_row_ptr(){
		size_t* solve_velocity_num_blocks_per_row_ptr;
		cudaGetSymbolAddress(reinterpret_cast<void**>(&solve_velocity_num_blocks_per_row_ptr), solve_velocity_num_blocks_per_row);
		
		return solve_velocity_num_blocks_per_row_ptr;
	}
	
	size_t** get_solve_velocity_block_offsets_per_row_ptr(){
		size_t** solve_velocity_matrix_size_x;
		cudaGetSymbolAddress(reinterpret_cast<void**>(&solve_velocity_matrix_size_x), solve_velocity_block_offsets_per_row);
		
		return solve_velocity_matrix_size_x;
	}
	
	void fill_matrices(){
		
	}
	
	void generate_system_matrices(){
		
	}
	
	void mass_transfer(){
		
	}
	
	void update_after_solve(){
		
	}
};

using surface_flow_model_t = variant<SurfaceFlowModel>;

}// namespace mn

#endif