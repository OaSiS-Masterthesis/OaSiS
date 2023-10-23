#ifndef SETTINGS_H
#define SETTINGS_H
#include <MnBase/Math/Vec.h>
#include <MnBase/Object/Structural.h>

#include <array>

#include "managed_memory.hpp"

//NOLINTNEXTLINE(cppcoreguidelines-macro-usage) Macro usage necessary here for preprocessor if
#define PRINT_CELL_OVERFLOW 0//TODO: Move to another place

//NOLINTNEXTLINE(cppcoreguidelines-macro-usage) Macro usage necessary here for preprocessor if
#define FIXED_COROTATED_GHOST_ENABLE_STRAIN_UPDATE 0//TODO: Move to another place

//NOLINTNEXTLINE(cppcoreguidelines-macro-usage) Macro usage necessary here for preprocessor if
#define VERIFY_IQ_MATRIX 0//TODO: Move to another place

namespace mn {

using ivec3	   = vec<int, 3>;
using vec3	   = vec<float, 3>;
using vec4	   = vec<float, 4>;
using vec9	   = vec<float, 9>;
using vec3x3   = vec<float, 3, 3>;
using vec3x4   = vec<float, 3, 4>;
using vec3x3x3 = vec<float, 3, 3, 3>;

using Duration = std::chrono::duration<float>;

/// sand = Drucker Prager Plasticity, StvkHencky Elasticity
enum class MaterialE {
	J_FLUID = 0,
	FIXED_COROTATED,
	SAND,
	NACC,
	FIXED_COROTATED_GHOST,
	TOTAL
};

/// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html, F.3.16.5
/// benchmark setup
namespace config {
	constexpr int G_DEVICE_COUNT = 1;
	constexpr MaterialE get_material_type(int did) noexcept {
		(void) did;

		return MaterialE::FIXED_COROTATED;
	}
	constexpr int G_TOTAL_FRAME_CNT = 60;
	constexpr int NUM_DIMENSIONS	= 3;

	constexpr int CUDA_WARP_SIZE = 32;
	
	constexpr int DEFAULT_CUDA_BLOCK_SIZE = 256;

	constexpr int GBPCB							   = 16;
	constexpr int G_NUM_GRID_BLOCKS_PER_CUDA_BLOCK = GBPCB;
	constexpr int G_NUM_WARPS_PER_GRID_BLOCK	   = 1;
	constexpr int G_NUM_WARPS_PER_CUDA_BLOCK	   = GBPCB;//>= G_NUM_GRID_BLOCKS_PER_CUDA_BLOCK
	constexpr int G_PARTICLE_BATCH_CAPACITY		   = 128;

	constexpr float MODEL_PPC	= 8.0f;
	constexpr float G_MODEL_PPC = MODEL_PPC;
	constexpr float CFL			= 0.5f;

	// background_grid
	constexpr float GRID_BLOCK_SPACING_INV = 10.0f;

	constexpr int BLOCK_BITS			 = 2;
	constexpr int DOMAIN_BITS			 = 8;
	constexpr float DXINV				 = (GRID_BLOCK_SPACING_INV * (1 << DOMAIN_BITS));
	constexpr int G_DOMAIN_BITS			 = DOMAIN_BITS;
	constexpr int G_DOMAIN_SIZE			 = (1 << DOMAIN_BITS);
	constexpr float G_BOUNDARY_CONDITION = 2.0;
	constexpr float G_DX				 = 1.f / DXINV;
	constexpr float G_DX_INV			 = DXINV;
	constexpr float G_D_INV				 = 4.f * DXINV * DXINV;
	constexpr int G_BLOCKBITS			 = BLOCK_BITS;
	constexpr int G_BLOCKSIZE			 = (1 << BLOCK_BITS);
	constexpr int G_BLOCKMASK			 = ((1 << BLOCK_BITS) - 1);
	constexpr int G_BLOCKVOLUME			 = (1 << (BLOCK_BITS * 3));
	constexpr int G_GRID_BITS			 = (DOMAIN_BITS - BLOCK_BITS);
	constexpr int G_GRID_SIZE			 = (1 << (DOMAIN_BITS - BLOCK_BITS));

	// particle
	constexpr int MAX_PARTICLES_IN_CELL	   = 128;
	constexpr int G_MAX_PARTICLES_IN_CELL  = MAX_PARTICLES_IN_CELL;
	constexpr int G_BIN_CAPACITY		   = 32;
	constexpr int G_PARTICLE_NUM_PER_BLOCK = (MAX_PARTICLES_IN_CELL * (1 << (BLOCK_BITS * 3)));
	
	constexpr int MAX_FACE_IN_CELL	   = 16;
	constexpr int G_FACE_NUM_PER_BLOCK = (MAX_FACE_IN_CELL * (1 << (BLOCK_BITS * 3)));
	constexpr int MAX_GENERATED_PARTICLE_PER_VERTEX = 32;

	// material parameters
	constexpr float DENSITY		   = 1e3;
	constexpr float YOUNGS_MODULUS = 5e3;
	constexpr float POISSON_RATIO  = 0.4f;
	
	//TODO: Maybe move somewhere else
	//TODO: Somehow choose by current used kernel and percentil which shell be considered inside
	constexpr float MARCHING_CUBES_DENSITY_THRESHOLD_FACTOR = 0.75f;

	constexpr float G_GRAVITY = -9.8f;
	
	constexpr float MAX_ALPHA = config::G_DX * config::G_DX * 0.9f;//FIXME:Set to correct value

	/// only used on host
	constexpr int G_MAX_PARTICLE_NUM = 1000000;
	constexpr int G_MAX_TRIANGLE_MESH_VERTICES_NUM = 1000000;
	constexpr int G_MAX_ACTIVE_BLOCK = 10000;/// 62500 bytes for active mask
	constexpr std::size_t calc_particle_bin_count(std::size_t num_active_blocks) noexcept {
		return num_active_blocks * (G_MAX_PARTICLES_IN_CELL * G_BLOCKVOLUME / G_BIN_CAPACITY);
	}
	constexpr std::size_t G_MAX_PARTICLE_BIN = G_MAX_PARTICLE_NUM / G_BIN_CAPACITY;
	constexpr std::size_t G_MAX_HALO_BLOCK	 = 4000;

}// namespace config

using BlockDomain	   = CompactDomain<char, config::G_BLOCKSIZE, config::G_BLOCKSIZE, config::G_BLOCKSIZE>;
using GridDomain	   = CompactDomain<int, config::G_GRID_SIZE, config::G_GRID_SIZE, config::G_GRID_SIZE>;
using GridBufferDomain = CompactDomain<int, config::G_MAX_ACTIVE_BLOCK>;

//FIXME: Move to another place
struct CustomDeviceAllocator {			   // hide the global one
	int gpuid;

	CustomDeviceAllocator(const int gpuid)
	: gpuid(gpuid){}

	void* allocate(std::size_t bytes) {//NOLINT(readability-convert-member-functions-to-static) Method is designed to be a non-static class member
		auto& cu_dev = Cuda::ref_cuda_context(gpuid);
	
		void* ret = nullptr;
		check_cuda_errors(cudaMalloc(&ret, bytes));
		
		//Set memory advice
		//TODO: Enable if using managed memory
		//if(cu_dev.supportsConcurrentManagedAccess()){
		//	check_cuda_errors(cudaMemAdvise(ret, bytes, cudaMemAdviseSetPreferredLocation, cu_dev.get_dev_id()));
		//	check_cuda_errors(cudaMemAdvise(ret, bytes, cudaMemAdviseSetAccessedBy, cu_dev.get_dev_id()));
		//}

		return ret;
	}

	void deallocate(void* p, std::size_t size) {//NOLINT(readability-convert-member-functions-to-static) Method is designed to be a non-static class member
		(void) size;
		check_cuda_errors(cudaFree(p));
	}
};

using managed_memory_type = ManagedMemory<CustomDeviceAllocator, (static_cast<size_t>(9) << 30), (static_cast<size_t>(0) << 30)>;

}// namespace mn

#endif