#ifndef OASIS_SIMULATOR_CUH
#define OASIS_SIMULATOR_CUH
#include <MnBase/Concurrency/Concurrency.h>
#include <MnBase/Meta/ControlFlow.h>
#include <MnBase/Meta/TupleMeta.h>
#include <MnSystem/Cuda/Cuda.h>
#include <MnSystem/IO/IO.h>
#include <fmt/color.h>
#include <fmt/core.h>

#include <MnBase/Profile/CppTimers.hpp>
#include <MnBase/Profile/CudaTimers.cuh>
#include <MnSystem/IO/ParticleIO.hpp>
#include <MnSystem/IO/OBJIO.hpp>
#include <ginkgo/ginkgo.hpp>
#include <array>
#include <vector>
#include <map>
#include <algorithm>
#include <execution>

#include "grid_buffer.cuh"
#include "hash_table.cuh"
#include "kernels.cuh"
#include "particle_buffer.cuh"
#include "settings.h"
#include "triangle_mesh.cuh"
#include "alpha_shapes.cuh"
#include "marching_cubes.cuh"
#include "iq.cuh"
#include "csr_utils.cuh"
#include "managed_memory.hpp"

#define OUTPUT_TRIANGLE_SHELL_OUTER_POS 1
#define UPDATE_SURFACE_BEFORE_OUTPUT 1

namespace mn {

struct OasisSimulator {
	static constexpr Duration DEFAULT_DT = Duration(1e-4);
	static constexpr int DEFAULT_FPS	 = 24;
	static constexpr int DEFAULT_FRAMES	 = 60;

	static constexpr size_t BIN_COUNT = 2;

	using streamIdx		 = Cuda::StreamIndex;
	using eventIdx		 = Cuda::EventIndex;
	using host_allocator = HeapAllocator;

	static_assert(std::is_same_v<GridBufferDomain::index_type, int>, "block index type is not int");
	
	template<typename Allocator>
	struct ManagedMemoryAllocator {
		Allocator* allocator;
	
		ManagedMemoryAllocator(Allocator* allocator)
		: allocator(allocator){}
	
		void* allocate(std::size_t bytes) {//NOLINT(readability-convert-member-functions-to-static) Method is designed to be a non-static class member
			return allocator->allocate(bytes, 256);
		}

		void deallocate(void* p, std::size_t size) {//NOLINT(readability-convert-member-functions-to-static) Method is designed to be a non-static class member
			return allocator->deallocate(p, size);
		}
	};
	
	//Only allocates largest user and shares memory along users
	template<typename Allocator>
	struct ReusableDeviceAllocator {
		int gpuid;
		
		Allocator* allocator;
		
		void* memory;
		size_t current_size;
	
		ReusableDeviceAllocator(Allocator* allocator, const int gpuid)
		: allocator(allocator), gpuid(gpuid), memory(nullptr), current_size(0){}
	
		void* allocate(std::size_t bytes) {//NOLINT(readability-convert-member-functions-to-static) Method is designed to be a non-static class member

			if(bytes > current_size){
				if(memory != nullptr){
					allocator->deallocate(memory, current_size);
					memory = nullptr;
				}
				current_size = bytes;
			}

			if(memory == nullptr){
				memory = allocator->allocate(current_size);
			}

			return memory;
		}

		void deallocate(void* p, std::size_t size) {//NOLINT(readability-convert-member-functions-to-static) Method is designed to be a non-static class member
			(void) size;
			allocator->deallocate(memory, size);
			memory = nullptr;
			current_size = 0;
		}
	};

	template<typename Allocator>
	struct Intermediates {
		
		Allocator* allocator;
		size_t max_block_count;

		int* d_tmp;
		int* active_block_marks;
		int* destinations;
		int* sources;
		int* bin_sizes;
		unsigned int* particle_id_mapping_buffer;
		float* d_max_vel;
		
		Intermediates(Allocator* allocator)
		: allocator(allocator), max_block_count(0) {}
		
		//NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic) Current c++ version does not yet support std::span
		void alloc(size_t max_block_count) {
			this->max_block_count = max_block_count;
			
			//NOLINTBEGIN(readability-magic-numbers) Magic numbers are variable count
			d_tmp			   = static_cast<int*>(allocator->allocate(sizeof(int) * max_block_count));
			active_block_marks = static_cast<int*>(allocator->allocate(sizeof(int) * max_block_count));
			destinations	   = static_cast<int*>(allocator->allocate(sizeof(int) * max_block_count));
			sources			   = static_cast<int*>(allocator->allocate(sizeof(int) * max_block_count));
			bin_sizes		   = static_cast<int*>(allocator->allocate(sizeof(int) * max_block_count));
			particle_id_mapping_buffer = static_cast<unsigned int*>(allocator->allocate(sizeof(unsigned int) * config::G_MAX_PARTICLE_BIN * config::G_BIN_CAPACITY));
			d_max_vel		   = static_cast<float*>(allocator->allocate(sizeof(float)));
			//NOLINTEND(readability-magic-numbers)
		}
		//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		void dealloc() {
			cudaDeviceSynchronize();
			allocator->deallocate(d_tmp, sizeof(int) * max_block_count);
			allocator->deallocate(active_block_marks, sizeof(int) * max_block_count);
			allocator->deallocate(destinations, sizeof(int) * max_block_count);
			allocator->deallocate(sources, sizeof(int) * max_block_count);
			allocator->deallocate(bin_sizes, sizeof(int) * max_block_count);
			allocator->deallocate(particle_id_mapping_buffer, sizeof(unsigned int) * config::G_MAX_PARTICLE_BIN * config::G_BIN_CAPACITY);
			allocator->deallocate(d_max_vel, sizeof(float));
			max_block_count = 0;
		}
		void resize(size_t max_block_count) {
			dealloc();
			alloc(max_block_count);
		}
	};

	///
	managed_memory_type managed_memory;
	ManagedMemoryAllocator<managed_memory_type> managed_memory_allocator;
	int gpuid;
	int nframes;
	int fps;
	/// animation runtime settings
	Duration dt;
	Duration next_dt;
	Duration dt_default;
	Duration cur_time;
	float max_vel;
	uint64_t cur_frame;
	uint64_t cur_step;
	/// data on device, double buffering
	std::array<std::vector<GridBuffer>, BIN_COUNT>  grid_blocks									= {};
	std::array<std::vector<particle_buffer_t>, BIN_COUNT> particle_bins = {};
	std::vector<Partition<1>> partitions								= {};///< with halo info
	std::vector<ParticleArray> particles								= {};
	std::vector<SurfaceParticleBuffer> surface_particle_buffers = {};
	//AlphaShapesGridBuffer alpha_shapes_grid_buffer;
	size_t global_marching_cubes_block_count;
	MarchingCubesGridBuffer marching_cubes_grid_buffer;
	//float* surface_vertex_buffer;
	uint32_t* surface_triangle_buffer;
	uint32_t* surface_vertex_count;
	uint32_t* surface_triangle_count;
	std::vector<std::array<int, 2>> solid_fluid_couplings = {};
	
	size_t max_surface_vertex_count;
	size_t max_surface_triangle_count;

	ReusableDeviceAllocator<ManagedMemoryAllocator<managed_memory_type>> reusable_allocator_surface_transfer;

	Intermediates<ManagedMemoryAllocator<managed_memory_type>> tmps;

	/// data on host
	char rollid;
	std::size_t cur_num_active_blocks;
	std::vector<std::size_t> cur_num_active_bins	  = {};
	std::array<std::size_t, BIN_COUNT> checked_counts = {};
	std::vector<std::size_t> checked_bin_counts		  = {};
	float max_vels;
	int partition_block_count;
	int neighbor_block_count;
	int total_neighbor_block_count;
	int exterior_block_count;///< num blocks
	std::vector<int> bincount									 = {};
	std::vector<uint32_t> particle_counts						 = {};///< num particles
	std::vector<std::array<float, config::NUM_DIMENSIONS>> model = {};
	std::vector<vec3> vel0										 = {};
	void* surface_transfer_device_buffer;
	std::vector<int> surface_point_type_transfer_host_buffer = {};
	std::vector<std::array<float, config::NUM_DIMENSIONS>> surface_normal_transfer_host_buffer = {};
	std::vector<float> surface_mean_curvature_transfer_host_buffer = {};
	std::vector<float> surface_gauss_curvature_transfer_host_buffer = {};
	std::vector<std::array<float, config::NUM_DIMENSIONS>> surface_vertex_transfer_host_buffer = {};
	std::vector<std::array<unsigned int, 3>> surface_triangle_transfer_host_buffer = {};
	
	std::vector<uint32_t> triangle_mesh_vertex_counts = {};
	std::vector<uint32_t> triangle_mesh_face_counts = {};
	std::vector<TriangleMesh> triangle_meshes = {};
	std::array<std::vector<std::vector<TriangleShell>>, BIN_COUNT> triangle_shells = {};
	std::vector<void*> triangle_mesh_transfer_device_buffers = {};
	std::vector<std::vector<std::array<float, config::NUM_DIMENSIONS>>> triangle_mesh_transfer_host_buffers = {};
	std::vector<std::vector<std::array<unsigned int, config::NUM_DIMENSIONS>>> triangle_mesh_face_buffers = {};
	std::array<std::vector<TriangleShellGridBuffer>, BIN_COUNT> triangle_shell_grid_buffer = {};//For mass redistribution and such
	
	//TemporaryGridBuffer temporary_grid_buffer;
	
	std::shared_ptr<gko::Executor> ginkgo_executor;
	gko::array<float> iq_rhs_array;
	gko::array<float> iq_result_array;
	gko::array<int> iq_lhs_rows;
	gko::array<int> iq_lhs_columns;
	gko::array<float> iq_lhs_values;
	gko::array<float> iq_solve_velocity_result_array;
	gko::array<int> iq_solve_velocity_rows;
	gko::array<int> iq_solve_velocity_columns;
	gko::array<float> iq_solve_velocity_values;
	
	gko::array<float> iq_lhs_scaling_solid_values;
	gko::array<float> iq_lhs_scaling_fluid_values;
	
	gko::array<float> iq_lhs_mass_solid_values;
	gko::array<float> iq_lhs_mass_fluid_values;
	
	gko::array<int> iq_lhs_3_1_rows;
	gko::array<int> iq_lhs_3_1_columns;
	
	gko::array<float> iq_lhs_gradient_solid_values;
	gko::array<float> iq_lhs_gradient_fluid_values;
	gko::array<float> iq_lhs_coupling_solid_values;
	gko::array<float> iq_lhs_coupling_fluid_values;
	gko::array<float> iq_lhs_boundary_fluid_values;
	
	gko::array<int> matrix_operations_temporary_rows;
	gko::array<int> matrix_operations_temporary_columns;
	gko::array<float> matrix_operations_temporary_values;
	
	gko::array<float> gko_identity_values;
	
	std::shared_ptr<gko::matrix::Dense<float>> gko_one_dense;
	std::shared_ptr<gko::matrix::Dense<float>> gko_neg_one_dense;
	std::shared_ptr<gko::matrix::Dense<float>> gko_zero_dense;
	
	std::shared_ptr<gko::solver::Cg<float>::Factory> iq_solver_factory;

	explicit OasisSimulator(int gpu = 0, Duration dt = DEFAULT_DT, int fps = DEFAULT_FPS, int frames = DEFAULT_FRAMES)
		: gpuid(gpu)
		, dt_default(dt)
		, cur_time(Duration::zero())
		, rollid(0)
		, cur_frame(0)
		, cur_step(0)
		, fps(fps)
		, nframes(frames)
		, dt()
		, next_dt()
		, max_vel()
		, managed_memory(CustomDeviceAllocator{gpu}, 0)
		, managed_memory_allocator(&managed_memory)
		, tmps(&managed_memory_allocator)
		, cur_num_active_blocks()
		, max_vels()
		, partition_block_count()
		, neighbor_block_count()
		, total_neighbor_block_count()
		, exterior_block_count()
		//, surface_vertex_buffer(nullptr)
		, surface_triangle_buffer(nullptr)
		, max_surface_vertex_count(0)
		, max_surface_triangle_count(0)
		, reusable_allocator_surface_transfer(&managed_memory_allocator, gpu)
		, global_marching_cubes_block_count(0)
		, marching_cubes_grid_buffer(managed_memory_allocator, &managed_memory)
		//, alpha_shapes_grid_buffer(managed_memory_allocator, &managed_memory)
		//, temporary_grid_buffer(managed_memory_allocator)		
		{
		// data
		initialize();
	}
	~OasisSimulator() = default;

	//TODO: Maybe implement?
	OasisSimulator(OasisSimulator& other)				= delete;
	OasisSimulator(OasisSimulator&& other)			= delete;
	OasisSimulator& operator=(OasisSimulator& other)	= delete;
	OasisSimulator& operator=(OasisSimulator&& other) = delete;

	void initialize() {
		auto& cu_dev = Cuda::ref_cuda_context(gpuid);
		cu_dev.set_context();

		//Allocate intermediate data for all blocks
		tmps.alloc(config::G_MAX_ACTIVE_BLOCK);
		
		surface_vertex_count = reinterpret_cast<uint32_t*>(managed_memory_allocator.allocate(sizeof(uint32_t)));
		surface_triangle_count = reinterpret_cast<uint32_t*>(managed_memory_allocator.allocate(sizeof(uint32_t)));

		//Create partitions
		for(int copyid = 0; copyid < BIN_COUNT; copyid++) {
			
			partitions.emplace_back(managed_memory_allocator, &managed_memory, config::G_MAX_ACTIVE_BLOCK);
			checked_counts[copyid] = 0;
		}

		cu_dev.syncStream<streamIdx::COMPUTE>();
		cur_num_active_blocks = config::G_MAX_ACTIVE_BLOCK;
		
		//Create Ginkgo executor and IQ-System stuff
		ginkgo_executor = gko::CudaExecutor::create(cu_dev.get_dev_id(), gko::ReferenceExecutor::create(), false, gko::allocation_mode::device, cu_dev.stream_compute());
		//TODO: Maybe specify csr strategy
		//Initially alloc for 32 blocks
		iq_rhs_array = gko::array<float>(ginkgo_executor, iq::LHS_MATRIX_SIZE_Y * 32 * config::G_BLOCKVOLUME);
		iq_result_array = gko::array<float>(ginkgo_executor, iq::LHS_MATRIX_SIZE_Y * 32 * config::G_BLOCKVOLUME);
		iq_lhs_rows = gko::array<int>(ginkgo_executor, iq::LHS_MATRIX_SIZE_Y * 32 * config::G_BLOCKVOLUME + 1);
		iq_lhs_columns = gko::array<int>(ginkgo_executor, iq::LHS_MATRIX_TOTAL_BLOCK_COUNT * 32 * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
		iq_lhs_values = gko::array<float>(ginkgo_executor, iq::LHS_MATRIX_TOTAL_BLOCK_COUNT * 32 * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
		
		iq_solve_velocity_result_array = gko::array<float>(ginkgo_executor, 3 * iq::SOLVE_VELOCITY_MATRIX_SIZE_Y * 32 * config::G_BLOCKVOLUME);
		iq_solve_velocity_rows = gko::array<int>(ginkgo_executor, 3 * iq::SOLVE_VELOCITY_MATRIX_SIZE_Y * 32 * config::G_BLOCKVOLUME + 1);
		iq_solve_velocity_columns = gko::array<int>(ginkgo_executor, 3 * iq::SOLVE_VELOCITY_MATRIX_TOTAL_BLOCK_COUNT * 32 * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
		iq_solve_velocity_values = gko::array<float>(ginkgo_executor, 3 * iq::SOLVE_VELOCITY_MATRIX_TOTAL_BLOCK_COUNT * 32 * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);

		iq_lhs_scaling_solid_values = gko::array<float>(ginkgo_executor, 32 * config::G_BLOCKVOLUME);
		iq_lhs_scaling_fluid_values = gko::array<float>(ginkgo_executor, 32 * config::G_BLOCKVOLUME);
		iq_lhs_mass_solid_values = gko::array<float>(ginkgo_executor, 3 * 32 * config::G_BLOCKVOLUME);
		iq_lhs_mass_fluid_values = gko::array<float>(ginkgo_executor, 3 * 32 * config::G_BLOCKVOLUME);
		
		iq_lhs_3_1_rows = gko::array<int>(ginkgo_executor, 3 * 32 * config::G_BLOCKVOLUME + 1);
		iq_lhs_3_1_columns = gko::array<int>(ginkgo_executor, 3 * 32 * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
		
		iq_lhs_gradient_solid_values = gko::array<float>(ginkgo_executor,  3 * 32 * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
		iq_lhs_gradient_fluid_values = gko::array<float>(ginkgo_executor, 3 * 32 * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
		iq_lhs_coupling_solid_values = gko::array<float>(ginkgo_executor,  3 * 32 * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
		iq_lhs_coupling_fluid_values = gko::array<float>(ginkgo_executor, 3 * 32 * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
		iq_lhs_boundary_fluid_values = gko::array<float>(ginkgo_executor, 3 * 32 * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
		
		matrix_operations_temporary_rows = gko::array<int>(ginkgo_executor, 32 * config::G_BLOCKVOLUME + 1);
		matrix_operations_temporary_columns = gko::array<int>(ginkgo_executor, 32 * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
		matrix_operations_temporary_values = gko::array<float>(ginkgo_executor,  32 * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
		
		gko_identity_values = gko::array<float>(ginkgo_executor, 32 * config::G_BLOCKVOLUME);
		gko_one_dense = gko::share(gko::initialize<gko::matrix::Dense<float>>({1.0f}, ginkgo_executor));
		gko_neg_one_dense = gko::share(gko::initialize<gko::matrix::Dense<float>>({-1.0f}, ginkgo_executor));
		gko_zero_dense = gko::share(gko::initialize<gko::matrix::Dense<float>>({0.0f}, ginkgo_executor));
		
		//Create IQ-Solver
		{
			const gko::remove_complex<float> tolerance = 1e-8f;
			//Maximum 100 iterations
			std::shared_ptr<gko::stop::Iteration::Factory> iter_stop = gko::share(
				gko::stop::Iteration::build()
				.with_max_iters(10000u)
				.on(ginkgo_executor)
			);
			
			//Tolerance
			std::shared_ptr<gko::stop::ResidualNorm<float>::Factory> tol_stop = gko::share(
				gko::stop::ResidualNorm<float>::build()
				.with_baseline(gko::stop::mode::absolute)//Against residual norm
				.with_reduction_factor(tolerance)
				.on(ginkgo_executor)
			);
			
			//Tolerance for coarse level solver
			std::shared_ptr<gko::stop::ResidualNorm<float>::Factory> exact_tol_stop = gko::share(
				gko::stop::ResidualNorm<float>::build()
				.with_baseline(gko::stop::mode::rhs_norm)//Against residual norm / rhs norm
				.with_reduction_factor(1e-14f)
				.on(ginkgo_executor)
			);

			//Incomplete cholesky for smoothing
			//TODO: Maybe use different smoother
			/*
			std::shared_ptr<gko::preconditioner::Ic<gko::solver::LowerTrs<float>, int>::Factory> ic_gen = gko::share(
				gko::preconditioner::Ic<gko::solver::LowerTrs<float>, int>::build()
				.with_factorization_factory(
					gko::factorization::Ic<float, int>::build()
					.with_skip_sorting(true) //We know that our matrix is sorted
					.on(ginkgo_executor)
				)
				.with_l_solver_factory(
					gko::solver::LowerTrs<float, int>::build()
					.with_algorithm(gko::solver::trisolve_algorithm::syncfree) //Use Ginkgo implementation
					.on(ginkgo_executor)
				)
				.on(ginkgo_executor)
			);
			*/
			
			/*
			std::shared_ptr<gko::preconditioner::Ic<gko::solver::Idr<float>, int>::Factory> ic_gen = gko::share(
				gko::preconditioner::Ic<gko::solver::Idr<float>, int>::build()
				.with_factorization_factory(
					gko::factorization::Ic<float, int>::build()
					.with_skip_sorting(true) //We know that our matrix is sorted
					.on(ginkgo_executor)
				)
				.with_l_solver_factory(
					gko::solver::Idr<float>::build()
					.with_deterministic(false) //Deterministic happens on CPU and is therefore slow
					.with_complex_subspace(true) //Complex subspace causes faster convergation
					.with_kappa(0.7f)
					.with_subspace_dim(2u)
					.with_criteria(gko::stop::Iteration::build().with_max_iters(1u).on(ginkgo_executor))
					.on(ginkgo_executor)
				)
				.on(ginkgo_executor)
			);
			*/
			
			
			std::shared_ptr<gko::preconditioner::Ic<gko::solver::Gmres<float>, int>::Factory> ic_gen = gko::share(
				gko::preconditioner::Ic<gko::solver::Gmres<float>, int>::build()
				.with_factorization_factory(
					gko::factorization::Ic<float, int>::build()
					.with_skip_sorting(true) //We know that our matrix is sorted
					.on(ginkgo_executor)
				)
				.with_l_solver_factory(
					gko::solver::Gmres<float>::build()
					.with_krylov_dim(0u)
					.with_flexible(false)
					.with_criteria(
						  gko::stop::Iteration::build().with_max_iters(10u).on(ginkgo_executor)
						, gko::stop::ResidualNorm<float>::build().with_baseline(gko::stop::mode::rhs_norm).with_reduction_factor(tolerance).on(ginkgo_executor)
					)
					.on(ginkgo_executor)
				)
				.on(ginkgo_executor)
			);
			
			//Smoother
			//TODO: Maybe other smoother or adjust params.
			std::shared_ptr<gko::solver::Ir<float>::Factory> smoother_gen = gko::share(
				gko::solver::build_smoother(ic_gen, 2u, static_cast<float>(0.9f))//Two iterations with relaxation factor 0.9
			);
			
			//Use PGM for generating coarse levels (==amgcl::smoothed_aggregation?)
			//TODO: Maybe somehow use neighbouring information of grid instead
			std::shared_ptr<gko::multigrid::Pgm<float, int>::Factory> mg_level_gen = gko::share(
				gko::multigrid::Pgm<float, int>::build()
				.with_max_iterations(15u)
				.with_max_unassigned_ratio(0.05f)
				.with_deterministic(false) //Deterministic might happen on CPU and is therefore slow
				.with_skip_sorting(true) //We know that our matrix is sorted
				.on(ginkgo_executor)
			);
			
			//Use CG for solving at coarsest level
			//TODO: Maybe other solver
			std::shared_ptr<gko::solver::Cg<float>::Factory> coarsest_gen = gko::share(
				gko::solver::Cg<float>::build()
				.with_preconditioner(ic_gen)//Using same solver for preconditioning as we use in smoother
				.with_criteria(iter_stop, exact_tol_stop)
				.on(ginkgo_executor)
			);

			//Default level_selector, smoother_selector
			std::shared_ptr<gko::LinOpFactory> multigrid_gen = gko::solver::Multigrid::build()
				.with_cycle(gko::solver::multigrid::cycle::w) //Use w-cycle
				.with_max_levels(1u) //Max level count
				.with_min_coarse_rows(static_cast<size_t>(config::G_BLOCKVOLUME)) //Minimum number of rows; Set to Blockvolume, otherwise nothing is solved if we have only one block
				.with_pre_smoother(smoother_gen)
				.with_post_uses_pre(true) //Use same smoother for pre and post smoothing
				.with_mid_case(gko::solver::multigrid::mid_smooth_type::both) //Mid smoothing keeps original behaviour
				.with_mg_level(mg_level_gen)
				.with_coarsest_solver(coarsest_gen)
				.with_default_initial_guess(gko::solver::initial_guess_mode::zero) //Zero as initial guess
				.with_criteria(gko::stop::Iteration::build().with_max_iters(1u).on(ginkgo_executor))//Only one iteration for preconditioning
				.on(ginkgo_executor)
			;

			// Create solver factory
			/*iq_solver_factory = gko::share(
				gko::solver::Bicgstab<float>::build()
				.with_criteria(iter_stop, tol_stop)
				//.with_preconditioner(multigrid_gen)
				.on(ginkgo_executor)
			);*/
			/*iq_solver_factory = gko::share(
				gko::solver::Gmres<float>::build()
				.with_krylov_dim(0u)
				.with_flexible(false)
				.with_criteria(iter_stop, tol_stop)
				.on(ginkgo_executor)
			);*/
			iq_solver_factory = gko::share(
				gko::solver::Cg<float>::build()
				.with_criteria(iter_stop, tol_stop)
				.on(ginkgo_executor)
			);
		}
   
	}
	
	void init_triangle_mesh(const std::vector<std::array<float, config::NUM_DIMENSIONS>>& positions, const std::vector<std::array<unsigned int, config::NUM_DIMENSIONS>>& faces){
		auto& cu_dev = Cuda::ref_cuda_context(gpuid);
		
		//Create buffers for triangle mesh
		triangle_meshes.emplace_back(managed_memory_allocator, &managed_memory, 1);
		
		//Init sizes
		triangle_mesh_vertex_counts.emplace_back(static_cast<uint32_t>(positions.size()));
		triangle_mesh_face_counts.emplace_back(static_cast<uint32_t>(faces.size()));
		
		fmt::print("init {}-th mesh with {} vectices and {} faces\n", triangle_meshes.size() - 1, positions.size(), faces.size());

		//Create transfer buffers
		triangle_mesh_transfer_device_buffers.emplace_back(managed_memory_allocator.allocate(sizeof(float) * config::NUM_DIMENSIONS * positions.size()));
		triangle_mesh_transfer_host_buffers.push_back(positions);
		
		//Keep face data
		triangle_mesh_face_buffers.push_back(faces);
	
		//Create temporary transfer buffer
		void* faces_tmp = managed_memory_allocator.allocate(sizeof(unsigned int) * config::NUM_DIMENSIONS * faces.size());//TODO: Maybe we can directly copy into data structure
		
		cu_dev.syncStream<streamIdx::COMPUTE>();
		
		//Copy positions and face data to device
		void* triangle_mesh_transfer_device_buffer_ptr = triangle_mesh_transfer_device_buffers.back();
		void* faces_tmp_ptr = faces_tmp;
		managed_memory.acquire<MemoryType::DEVICE>(&triangle_mesh_transfer_device_buffer_ptr, &faces_tmp_ptr, triangle_meshes.back().acquire());
		
		cudaMemcpyAsync(triangle_mesh_transfer_device_buffer_ptr, positions.data(), sizeof(float) * config::NUM_DIMENSIONS * positions.size(), cudaMemcpyDefault, cu_dev.stream_compute());
		cudaMemcpyAsync(faces_tmp_ptr, faces.data(), sizeof(unsigned int) * config::NUM_DIMENSIONS * faces.size(), cudaMemcpyDefault, cu_dev.stream_compute());
		
		cu_dev.compute_launch({(std::max(triangle_mesh_vertex_counts.back(), triangle_mesh_face_counts.back())+ config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, copy_triangle_mesh_data_to_device, triangle_meshes.back(), triangle_mesh_vertex_counts.back(), triangle_mesh_face_counts.back(), static_cast<float*>(triangle_mesh_transfer_device_buffer_ptr), static_cast<unsigned int*>(faces_tmp_ptr));
		
		managed_memory.release(triangle_mesh_transfer_device_buffers.back(), faces_tmp, triangle_meshes.back().release());
		
		//Free temporary transfer buffer
		cudaDeviceSynchronize();
		managed_memory_allocator.deallocate(faces_tmp, sizeof(unsigned int) * config::NUM_DIMENSIONS * faces.size());
		
		//Write out initial state to file
		std::string fn = std::string {"mesh"} + "_id[" + std::to_string(triangle_meshes.size() - 1) + "]_frame[0].obj";
		IO::insert_job([fn, positions, faces]() {
			write_triangle_mesh<float, uint32_t, config::NUM_DIMENSIONS>(fn, positions, faces);
		});
		IO::flush();
	}

	template<MaterialE M>
	void init_model(const std::vector<std::array<float, config::NUM_DIMENSIONS>>& model, const mn::vec<float, config::NUM_DIMENSIONS>& v0, const mn::vec<float, config::NUM_DIMENSIONS>& grid_offset) {
		auto& cu_dev = Cuda::ref_cuda_context(gpuid);
		
		cu_dev.syncStream<streamIdx::COMPUTE>();

		//Create particle buffers and grid blocks and reserve buckets
		for(int copyid = 0; copyid < BIN_COUNT; ++copyid) {
			particle_bins[copyid].emplace_back(ParticleBuffer<M>(managed_memory_allocator, &managed_memory, model.size() / config::G_BIN_CAPACITY + config::G_MAX_ACTIVE_BLOCK));
			match(particle_bins[copyid].back())([&](auto& particle_buffer) {
				particle_buffer.reserve_buckets(managed_memory_allocator, config::G_MAX_ACTIVE_BLOCK);
			});
			grid_blocks[copyid].emplace_back(managed_memory_allocator, &managed_memory, grid_offset.data_arr());
		}
		surface_particle_buffers.emplace_back(SurfaceParticleBuffer(managed_memory_allocator, &managed_memory, model.size() / config::G_BIN_CAPACITY + config::G_MAX_ACTIVE_BLOCK));

		//Set initial velocity
		vel0.emplace_back();
		for(int i = 0; i < config::NUM_DIMENSIONS; ++i) {
			vel0.back()[i] = v0[i];
		}

		//Create array for initial particles
		particles.emplace_back(spawn<particle_array_, orphan_signature>(managed_memory_allocator, sizeof(float) * config::NUM_DIMENSIONS * model.size()));

		//Init bin counts
		cur_num_active_bins.emplace_back(config::G_MAX_PARTICLE_BIN);
		bincount.emplace_back(0);
		checked_bin_counts.emplace_back(0);
		
		//Create triangle_shell_grid
		for(int copyid = 0; copyid < BIN_COUNT; copyid++) {
			//FIXME: Outcommented to save memory
			//triangle_shell_grid_buffer[copyid].emplace_back(managed_memory_allocator, &managed_memory);
		}

		//Init particle counts
		particle_counts.emplace_back(static_cast<unsigned int>(model.size()));//NOTE: Explicic narrowing cast

		fmt::print("init {}-th model with {} particles\n", particle_bins[0].size() - 1, particle_counts.back());
		
		/*//Set cache configuration for specific functions
		cudaFuncSetCacheConfig(alpha_shapes<Partition<1>, GridBuffer, M>, cudaFuncCachePreferShared);
		cudaFuncSetAttribute(alpha_shapes<Partition<1>, GridBuffer, M>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
		cudaError_t error = cudaGetLastError();
		if(error != cudaSuccess) {
			printf("Could not set cache config for kernel: %s\n", cudaGetErrorString(error));
		}*/

		//Copy particle positions from host to device
		managed_memory.acquire<MemoryType::DEVICE>(particles.back().acquire());
		
		cudaMemcpyAsync(static_cast<void*>(&particles.back().val_1d(_0, 0)), model.data(), sizeof(std::array<float, config::NUM_DIMENSIONS>) * model.size(), cudaMemcpyDefault, cu_dev.stream_compute());
		cu_dev.syncStream<streamIdx::COMPUTE>();
		
		managed_memory.release(particles.back().release());

		//Write out initial state to file
		std::string fn = std::string {"model"} + "_id[" + std::to_string(particle_bins[0].size() - 1) + "]_frame[0].bgeo";
		IO::insert_job([fn, model]() {
			Partio::ParticlesDataMutable* parts;
			begin_write_partio(&parts, model.size());
			
			write_partio_add(model, std::string("position"), parts);

			end_write_partio(fn, parts);
		});
		IO::flush();
	}
	
	void init_solid_fluid_coupling(const int solid_id, const int fluid_id) {
		if(solid_id < particle_counts.size() && fluid_id < particle_counts.size()){
			bool already_coupled = false;
			for(size_t i = 0; i < solid_fluid_couplings.size(); ++i){
				if(solid_fluid_couplings[i][0] == solid_id || solid_fluid_couplings[i][1] == solid_id || solid_fluid_couplings[i][0] == fluid_id || solid_fluid_couplings[i][1] == fluid_id){
					already_coupled = true;
					fmt::print("Model already coupled: {}<->{} but already exists {}<->{}", solid_id, fluid_id, solid_fluid_couplings[i][0], solid_fluid_couplings[i][1]);
					break;
				}
			}
			
			if(!already_coupled){
				solid_fluid_couplings.push_back({solid_id, fluid_id});
			}
		}else{
			fmt::print("Model ids out of range: {} {} but model count is", solid_id, fluid_id, particle_counts.size());
		}
	}
	
	void update_triangle_mesh_parameters(float mass, const std::function<vec3(Duration, Duration)>& animation_linear_func, const std::function<vec3(Duration, Duration)>& animation_rotational_func) {
		triangle_meshes.back().update_parameters(mass, animation_linear_func, animation_rotational_func);
	}

	void update_fr_parameters(float rho, float vol, float ym, float pr) {
		match(particle_bins[0].back())(
			[&](auto& particle_buffer) {},
			[&](ParticleBuffer<MaterialE::FIXED_COROTATED>& particle_buffer) {
				particle_buffer.update_parameters(rho, vol, ym, pr);
			}
		);
		match(particle_bins[1].back())(
			[&](auto& particle_buffer) {},
			[&](ParticleBuffer<MaterialE::FIXED_COROTATED>& particle_buffer) {
				particle_buffer.update_parameters(rho, vol, ym, pr);
			}
		);
	}
	
	void update_frg_parameters(float rho, float vol, float ym, float pr) {
		match(particle_bins[0].back())(
			[&](auto& particle_buffer) {},
			[&](ParticleBuffer<MaterialE::FIXED_COROTATED_GHOST>& particle_buffer) {
				particle_buffer.update_parameters(rho, vol, ym, pr);
			}
		);
		match(particle_bins[1].back())(
			[&](auto& particle_buffer) {},
			[&](ParticleBuffer<MaterialE::FIXED_COROTATED_GHOST>& particle_buffer) {
				particle_buffer.update_parameters(rho, vol, ym, pr);
			}
		);
	}

	void update_j_fluid_parameters(float rho, float vol, float bulk, float gamma, float viscosity) {
		match(particle_bins[0].back())(
			[&](auto& particle_buffer) {},
			[&](ParticleBuffer<MaterialE::J_FLUID>& particle_buffer) {
				particle_buffer.update_parameters(rho, vol, bulk, gamma, viscosity);
			}
		);
		match(particle_bins[1].back())(
			[&](auto& particle_buffer) {},
			[&](ParticleBuffer<MaterialE::J_FLUID>& particle_buffer) {
				particle_buffer.update_parameters(rho, vol, bulk, gamma, viscosity);
			}
		);
	}

	void update_nacc_parameters(float rho, float vol, float ym, float pr, float beta, float xi) {
		match(particle_bins[0].back())(
			[&](auto& particle_buffer) {},
			[&](ParticleBuffer<MaterialE::NACC>& particle_buffer) {
				particle_buffer.update_parameters(rho, vol, ym, pr, beta, xi);
			}
		);
		match(particle_bins[1].back())(
			[&](auto& particle_buffer) {},
			[&](ParticleBuffer<MaterialE::NACC>& particle_buffer) {
				particle_buffer.update_parameters(rho, vol, ym, pr, beta, xi);
			}
		);
	}

	//Sum up count values from in and store them in out
	template<typename CudaContext>
	void exclusive_scan(int count, int const* const in, int* out, CudaContext& cu_dev) {
		auto policy = thrust::cuda::par.on(static_cast<cudaStream_t>(cu_dev.stream_compute()));
		thrust::exclusive_scan(policy, get_device_ptr(in), get_device_ptr(in) + count, get_device_ptr(out));
		/*
		std::size_t temp_storage_bytes = 0;
		auto plus_op				   = [] __device__(const int& a, const int& b) {
			  return a + b;
		};
		check_cuda_errors(cub::DeviceScan::ExclusiveScan(nullptr, temp_storage_bytes, in, out, plus_op, 0, count, cu_dev.stream_compute()));
		void* d_tmp = tmps[cu_dev.get_dev_id()].d_tmp;
		check_cuda_errors(cub::DeviceScan::ExclusiveScan(d_tmp, temp_storage_bytes, in, out, plus_op, 0, count, cu_dev.stream_compute()));
		*/
	}
	
	//FIXME: Untested
	void matrix_transpose(const size_t num_blocks, const size_t num_columns, std::shared_ptr<gko::matrix::Csr<float, int>>& a, std::shared_ptr<gko::matrix::Csr<float, int>>& b, Cuda::CudaContext& cu_dev){	
		matrix_operations_temporary_rows.resize_and_reset(num_columns + 1);
		
		matrix_operations_temporary_rows.fill(0);
		
		ginkgo_executor->synchronize();
		
		//Copy last rows value
		const int number_of_nonzeros_a = a->get_num_stored_elements();
		cudaMemcpyAsync(a->get_row_ptrs() + (num_blocks * config::G_BLOCKVOLUME), &number_of_nonzeros_a, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute());
		
		//Resize matrix
		matrix_operations_temporary_columns.resize_and_reset(number_of_nonzeros_a);
		matrix_operations_temporary_values.resize_and_reset(number_of_nonzeros_a);
		
		matrix_operations_temporary_columns.fill(0);
		matrix_operations_temporary_values.fill(0.0f);
		
		if(number_of_nonzeros_a > 0){
			//Transpose
			cu_dev.compute_launch({num_blocks, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, csr_transpose<iq::NUM_ROWS_PER_BLOCK, 1>, a->get_const_row_ptrs(), a->get_const_col_idxs(), a->get_const_values(), matrix_operations_temporary_rows.get_data(), matrix_operations_temporary_columns.get_data(), matrix_operations_temporary_values.get_data());
			
			//Scan rows
			exclusive_scan(num_columns + 1, matrix_operations_temporary_rows.get_data(), matrix_operations_temporary_rows.get_data(), cu_dev);
		}
		
		//Copy to output
		b->copy_from(std::move(gko::share(gko::matrix::Csr<float, int>::create_const(
			  ginkgo_executor
			, gko::dim<2>(num_columns, num_blocks * config::G_BLOCKVOLUME)
			, matrix_operations_temporary_values.as_const_view()
			, matrix_operations_temporary_columns.as_const_view()
			, matrix_operations_temporary_rows.as_const_view()
		))));
		
		//Sort columns and values
		b->sort_by_column_index();
		
		ginkgo_executor->synchronize();
	}
	
	//Calculates C = A * B (Gustavson == true) or C = A * B^T
	template<bool Gustavson = false>
	void matrix_matrix_multiplication(const size_t num_blocks, const size_t num_columns, std::shared_ptr<gko::matrix::Csr<float, int>>& a, std::shared_ptr<gko::matrix::Csr<float, int>>& b, std::shared_ptr<gko::matrix::Csr<float, int>>& c, Cuda::CudaContext& cu_dev){					
		matrix_operations_temporary_rows.resize_and_reset(num_blocks * config::G_BLOCKVOLUME + 1);
		
		matrix_operations_temporary_rows.fill(0);
		
		ginkgo_executor->synchronize();
		
		//Copy last rows value
		const int number_of_nonzeros_a = a->get_num_stored_elements();
		const int number_of_nonzeros_b = b->get_num_stored_elements();
		cudaMemcpyAsync(a->get_row_ptrs() + num_blocks * config::G_BLOCKVOLUME, &number_of_nonzeros_a, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute());
		cudaMemcpyAsync(b->get_row_ptrs() + num_blocks * config::G_BLOCKVOLUME, &number_of_nonzeros_b, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute());
		
		if(Gustavson){
			//Resize temporary memory
			matrix_operations_temporary_columns.resize_and_reset(num_blocks * num_columns);
			
			ginkgo_executor->synchronize();
		}
		
		//Calculate amount of memory
		if(Gustavson){
			cu_dev.compute_launch({num_blocks, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, csr_matrix_matrix_multiplication_gustavson_calculate_rows<iq::NUM_ROWS_PER_BLOCK, 1>, num_columns, a->get_const_row_ptrs(), a->get_const_col_idxs(), a->get_const_values(), b->get_const_row_ptrs(), b->get_const_col_idxs(), b->get_const_values(), matrix_operations_temporary_rows.get_data(), matrix_operations_temporary_columns.get_data());
		}else{
			cu_dev.compute_launch({num_blocks, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, csr_matrix_matrix_multiplication<iq::NUM_ROWS_PER_BLOCK, 1, true>, a->get_const_row_ptrs(), a->get_const_col_idxs(), a->get_const_values(), b->get_const_row_ptrs(), b->get_const_col_idxs(), b->get_const_values(), matrix_operations_temporary_rows.get_data(), nullptr, nullptr);
		}
		
		//Scan rows
		exclusive_scan(num_blocks * config::G_BLOCKVOLUME + 1, matrix_operations_temporary_rows.get_data(), matrix_operations_temporary_rows.get_data(), cu_dev);
		
		//Resize matrix
		int number_of_nonzeros_host;
		
		cudaMemcpyAsync(&number_of_nonzeros_host, matrix_operations_temporary_rows.get_data() + (num_blocks * config::G_BLOCKVOLUME), sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute());
		
		cu_dev.syncStream<streamIdx::COMPUTE>();
		
		matrix_operations_temporary_columns.resize_and_reset(number_of_nonzeros_host);
		matrix_operations_temporary_values.resize_and_reset(number_of_nonzeros_host);
		
		matrix_operations_temporary_columns.fill(0);
		matrix_operations_temporary_values.fill(0.0f);
		
		ginkgo_executor->synchronize();
		
		//Perform matrix multiplication
		if(number_of_nonzeros_host > 0){
			if(Gustavson){
				cu_dev.compute_launch({num_blocks, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, csr_matrix_matrix_multiplication_gustavson<iq::NUM_ROWS_PER_BLOCK, 1>, a->get_const_row_ptrs(), a->get_const_col_idxs(), a->get_const_values(), b->get_const_row_ptrs(), b->get_const_col_idxs(), b->get_const_values(), matrix_operations_temporary_rows.get_data(), matrix_operations_temporary_columns.get_data(), matrix_operations_temporary_values.get_data());
			}else{
				cu_dev.compute_launch({num_blocks, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, csr_matrix_matrix_multiplication<iq::NUM_ROWS_PER_BLOCK, 1, false>, a->get_const_row_ptrs(), a->get_const_col_idxs(), a->get_const_values(), b->get_const_row_ptrs(), b->get_const_col_idxs(), b->get_const_values(), matrix_operations_temporary_rows.get_data(), matrix_operations_temporary_columns.get_data(), matrix_operations_temporary_values.get_data());
			}
		}
		
		//Copy to output
		c->copy_from(std::move(gko::share(gko::matrix::Csr<float, int>::create_const(
			  ginkgo_executor
			, gko::dim<2>(num_blocks * config::G_BLOCKVOLUME, num_columns)
			, matrix_operations_temporary_values.as_const_view()
			, matrix_operations_temporary_columns.as_const_view()
			, matrix_operations_temporary_rows.as_const_view()
		))));
		
		ginkgo_executor->synchronize();
		//NOTE: Columns already sorted
	}
	
	//Calculates C = A * D * B (Gustavson == true) or C = A * D * B^T
	template<bool Gustavson = false>
	void matrix_matrix_multiplication_with_diagonal(const size_t num_blocks, const size_t num_columns, std::shared_ptr<gko::matrix::Csr<float, int>>& a, const std::shared_ptr<const gko::matrix::Diagonal<float>>& d, std::shared_ptr<gko::matrix::Csr<float, int>>& b, std::shared_ptr<gko::matrix::Csr<float, int>>& c, Cuda::CudaContext& cu_dev){					
		matrix_operations_temporary_rows.resize_and_reset(num_blocks * config::G_BLOCKVOLUME + 1);
		
		matrix_operations_temporary_rows.fill(0);
		
		ginkgo_executor->synchronize();
		
		//Copy last rows value
		const int number_of_nonzeros_a = a->get_num_stored_elements();
		const int number_of_nonzeros_b = b->get_num_stored_elements();
		cudaMemcpyAsync(a->get_row_ptrs() + num_blocks * config::G_BLOCKVOLUME, &number_of_nonzeros_a, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute());
		cudaMemcpyAsync(b->get_row_ptrs() + num_blocks * config::G_BLOCKVOLUME, &number_of_nonzeros_b, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute());
		
		if(Gustavson){
			//Resize temporary memory
			matrix_operations_temporary_columns.resize_and_reset(num_blocks * num_columns);
			
			ginkgo_executor->synchronize();
		}
		
		//Calculate amount of memory
		if(Gustavson){
			cu_dev.compute_launch({num_blocks, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, csr_matrix_matrix_multiplication_with_diagonal_gustavson_calculate_rows<iq::NUM_ROWS_PER_BLOCK, 1>, num_columns, a->get_const_row_ptrs(), a->get_const_col_idxs(), a->get_const_values(), d->get_const_values(), b->get_const_row_ptrs(), b->get_const_col_idxs(), b->get_const_values(), matrix_operations_temporary_rows.get_data(), matrix_operations_temporary_columns.get_data());
		}else{
			cu_dev.compute_launch({num_blocks, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, csr_matrix_matrix_multiplication_with_diagonal<iq::NUM_ROWS_PER_BLOCK, 1, true>, a->get_const_row_ptrs(), a->get_const_col_idxs(), a->get_const_values(), d->get_const_values(), b->get_const_row_ptrs(), b->get_const_col_idxs(), b->get_const_values(), matrix_operations_temporary_rows.get_data(), nullptr, nullptr);
		}
		
		//Scan rows
		exclusive_scan(num_blocks * config::G_BLOCKVOLUME + 1, matrix_operations_temporary_rows.get_data(), matrix_operations_temporary_rows.get_data(), cu_dev);
		
		//Resize matrix
		int number_of_nonzeros_host;
		
		cudaMemcpyAsync(&number_of_nonzeros_host, matrix_operations_temporary_rows.get_data() + (num_blocks * config::G_BLOCKVOLUME), sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute());
		
		cu_dev.syncStream<streamIdx::COMPUTE>();
		
		matrix_operations_temporary_columns.resize_and_reset(number_of_nonzeros_host);
		matrix_operations_temporary_values.resize_and_reset(number_of_nonzeros_host);
		
		matrix_operations_temporary_columns.fill(0);
		matrix_operations_temporary_values.fill(0.0f);
		
		ginkgo_executor->synchronize();
		
		//Perform matrix multiplication
		if(number_of_nonzeros_host > 0){
			if(Gustavson){
				cu_dev.compute_launch({num_blocks, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, csr_matrix_matrix_multiplication_with_diagonal_gustavson<iq::NUM_ROWS_PER_BLOCK, 1>, a->get_const_row_ptrs(), a->get_const_col_idxs(), a->get_const_values(), d->get_const_values(), b->get_const_row_ptrs(), b->get_const_col_idxs(), b->get_const_values(), matrix_operations_temporary_rows.get_data(), matrix_operations_temporary_columns.get_data(), matrix_operations_temporary_values.get_data());
			}else{
				cu_dev.compute_launch({num_blocks, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, csr_matrix_matrix_multiplication_with_diagonal<iq::NUM_ROWS_PER_BLOCK, 1, false>, a->get_const_row_ptrs(), a->get_const_col_idxs(), a->get_const_values(), d->get_const_values(), b->get_const_row_ptrs(), b->get_const_col_idxs(), b->get_const_values(), matrix_operations_temporary_rows.get_data(), matrix_operations_temporary_columns.get_data(), matrix_operations_temporary_values.get_data());
			}
		}
		
		//Copy to output
		c->copy_from(std::move(gko::share(gko::matrix::Csr<float, int>::create_const(
			  ginkgo_executor
			, gko::dim<2>(num_blocks * config::G_BLOCKVOLUME, num_columns)
			, matrix_operations_temporary_values.as_const_view()
			, matrix_operations_temporary_columns.as_const_view()
			, matrix_operations_temporary_rows.as_const_view()
		))));
		
		ginkgo_executor->synchronize();
		//NOTE: Columns already sorted
	}
	
	//Calculates C = A * A^T
	void matrix_matrix_multiplication_a_at(const size_t num_blocks, const size_t num_columns, std::shared_ptr<gko::matrix::Csr<float, int>>& a, std::shared_ptr<gko::matrix::Csr<float, int>>& a_transposed, std::shared_ptr<gko::matrix::Csr<float, int>>& c, Cuda::CudaContext& cu_dev){					
		matrix_matrix_multiplication<true>(num_blocks, num_columns, a, a_transposed, c, cu_dev);
		
		if(c->get_num_stored_elements() > 0){
			//Mirror
			cu_dev.compute_launch({num_blocks, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, csr_mirror<iq::NUM_ROWS_PER_BLOCK, 1>, c->get_const_row_ptrs(), c->get_const_col_idxs(), c->get_values());
		}
	}
	
	//Calculates C = A * D * A^T
	void matrix_matrix_multiplication_a_at_with_diagonal(const size_t num_blocks, const size_t num_columns, std::shared_ptr<gko::matrix::Csr<float, int>>& a, const std::shared_ptr<const gko::matrix::Diagonal<float>>& d, std::shared_ptr<gko::matrix::Csr<float, int>>& a_transposed, std::shared_ptr<gko::matrix::Csr<float, int>>& c, Cuda::CudaContext& cu_dev){					
		matrix_matrix_multiplication_with_diagonal<true>(num_blocks, num_columns, a, d, a_transposed, c, cu_dev);
		
		if(c->get_num_stored_elements() > 0){
			//Mirror
			cu_dev.compute_launch({num_blocks, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, csr_mirror<iq::NUM_ROWS_PER_BLOCK, 1>, c->get_const_row_ptrs(), c->get_const_col_idxs(), c->get_values());
		}
	}

	float get_mass(int id = 0) {
		return match(particle_bins[rollid][id])([&](const auto& particle_buffer) {
			return particle_buffer.mass;
		});
	}

	[[nodiscard]] int get_model_count() const noexcept {
		return static_cast<int>(particle_bins[0].size());//NOTE: Explicit narrowing cast (But we should not have that much models anyway.)
	}

	//Increase bin and active block count if too low
	void check_capacity() {
		//TODO: Is that right? Maybe create extra parameter for this?
		//NOLINTBEGIN(readability-magic-numbers) Magic numbers are resize thresholds?
		if(exterior_block_count > cur_num_active_blocks * config::NUM_DIMENSIONS / 4 && checked_counts[0] == 0) {
			cur_num_active_blocks = cur_num_active_blocks * config::NUM_DIMENSIONS / 2;
			checked_counts[0]	  = 2;
			fmt::print(fmt::emphasis::bold, "resizing blocks {} -> {}\n", exterior_block_count, cur_num_active_blocks);
		}

		for(int i = 0; i < get_model_count(); ++i) {
			if(bincount[i] > cur_num_active_bins[i] * config::NUM_DIMENSIONS / 4 && checked_bin_counts[i] == 0) {
				cur_num_active_bins[i] = cur_num_active_bins[i] * config::NUM_DIMENSIONS / 2;
				checked_bin_counts[i]  = 2;
				fmt::print(fmt::emphasis::bold, "resizing bins {} -> {}\n", bincount[i], cur_num_active_bins[i]);
			}
		}
		//NOLINTEND(readability-magic-numbers)
	}

	//NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic) Current c++ version does not yet support std::span
	void main_loop() {
		cudaDeviceSynchronize();
		
		partitions[rollid].Instance<block_partition_>::count = partitions[rollid].Instance<block_partition_>::count_virtual;
		partitions[rollid].active_keys = partitions[rollid].active_keys_virtual;
		partitions[rollid].index_table = partitions[rollid].index_table_virtual;
		partitions[(rollid + 1) % BIN_COUNT].Instance<block_partition_>::count = partitions[(rollid + 1) % BIN_COUNT].Instance<block_partition_>::count_virtual;
		partitions[(rollid + 1) % BIN_COUNT].active_keys = partitions[(rollid + 1) % BIN_COUNT].active_keys_virtual;
		partitions[(rollid + 1) % BIN_COUNT].index_table = partitions[(rollid + 1) % BIN_COUNT].index_table_virtual;
		managed_memory.acquire<MemoryType::DEVICE>(
			  reinterpret_cast<void**>(&partitions[rollid].Instance<block_partition_>::count)
			, reinterpret_cast<void**>(&partitions[rollid].active_keys)
			, reinterpret_cast<void**>(&partitions[rollid].index_table)
			, reinterpret_cast<void**>(&partitions[(rollid + 1) % BIN_COUNT].Instance<block_partition_>::count)
			, reinterpret_cast<void**>(&partitions[(rollid + 1) % BIN_COUNT].active_keys)
			, reinterpret_cast<void**>(&partitions[(rollid + 1) % BIN_COUNT].index_table)
		);
		
		//Create triangle shells and init their particle buffer
		{
			auto& cu_dev = Cuda::ref_cuda_context(gpuid);
			
			for(int copyid = 0; copyid < BIN_COUNT; ++copyid) {
				triangle_shells[copyid].resize(get_model_count());
				for(int i = 0; i < get_model_count(); ++i) {
					for(int j = 0; j <triangle_meshes.size(); ++j) {
						triangle_shells[copyid][i].emplace_back(managed_memory_allocator, &managed_memory, 1);
						triangle_shells[copyid][i].back().particle_buffer.reserve_buckets(managed_memory_allocator, config::G_MAX_ACTIVE_BLOCK);
						
						//Write out initial state
						#ifdef OUTPUT_TRIANGLE_SHELL_OUTER_POS
							//Write out initial state to file
							std::string fn = std::string {"shell"} + "_id[" + std::to_string(i) + "_" + std::to_string(j) + "]_frame[" + std::to_string(cur_frame) + "].obj";
							IO::insert_job([this, fn, j, pos = triangle_mesh_transfer_host_buffers[j]]() {
								write_triangle_mesh<float, uint32_t, config::NUM_DIMENSIONS>(fn, pos, triangle_mesh_face_buffers[j]);
							});
						#endif	
					}
				}
			}
		}
		
		/// initial
		const Duration seconds_per_frame(Duration(1) / Duration(fps));
		{
			float max_vel = 0.f;
			for(int i = 0; i < get_model_count(); ++i) {
				const float vel_norm = std::sqrt(vel0[i].l2NormSqr());
				if(vel_norm > max_vel) {
					max_vel = vel_norm;
				}
			}

			dt = compute_dt(max_vel, Duration::zero(), seconds_per_frame, dt_default);
		}

		fmt::print(fmt::emphasis::bold, "{} --{}--> {}, defaultDt: {}\n", cur_time.count(), dt.count(), seconds_per_frame.count(), dt_default.count());
		initial_setup();

		cur_time = Duration::zero();
		
		for(cur_frame = 1; cur_frame <= nframes; ++cur_frame) {
			const Duration next_time = cur_time + seconds_per_frame;
			for(Duration current_step_time = Duration::zero(); current_step_time < seconds_per_frame; current_step_time += dt, cur_time += dt, cur_step++) {
			//for(Duration current_step_time = Duration::zero(); current_step_time < Duration::zero(); current_step_time += dt, cur_time += dt, cur_step++) {
				//Calculate maximum grid velocity and update the grid velocity
				{
					auto& cu_dev = Cuda::ref_cuda_context(gpuid);

					/// check capacity
					check_capacity();

					CudaTimer timer {cu_dev.stream_compute()};
					timer.tick();

					float* d_max_vel = tmps.d_max_vel;
					managed_memory.acquire<MemoryType::DEVICE>(reinterpret_cast<void**>(&d_max_vel));
					
					//Initialize max_vel with 0
					check_cuda_errors(cudaMemsetAsync(d_max_vel, 0, sizeof(float), cu_dev.stream_compute()));
					
					cu_dev.syncStream<streamIdx::COMPUTE>();

					for(int i = 0; i < get_model_count(); ++i) {
						managed_memory.acquire<MemoryType::DEVICE>(grid_blocks[0][i].acquire());
						//Update the grid velocity
						//floor(neighbor_block_count/G_NUM_GRID_BLOCKS_PER_CUDA_BLOCK); G_NUM_WARPS_PER_CUDA_BLOCK (>= G_NUM_GRID_BLOCKS_PER_CUDA_BLOCK)
						cu_dev.compute_launch({(neighbor_block_count + config::G_NUM_GRID_BLOCKS_PER_CUDA_BLOCK - 1) / config::G_NUM_GRID_BLOCKS_PER_CUDA_BLOCK, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, update_grid_velocity_query_max, static_cast<uint32_t>(neighbor_block_count), grid_blocks[0][i], partitions[rollid], dt, d_max_vel);
						
						managed_memory.release(grid_blocks[0][i].release());
						
						cu_dev.syncStream<streamIdx::COMPUTE>();
					}
					//Copy maximum velocity to host site
					check_cuda_errors(cudaMemcpyAsync(&max_vels, d_max_vel, sizeof(float), cudaMemcpyDefault, cu_dev.stream_compute()));
					
					managed_memory.release(tmps.d_max_vel);
					
					cu_dev.syncStream<streamIdx::COMPUTE>();

					timer.tock(fmt::format("GPU[{}] frame {} step {} grid_update_query", gpuid, cur_frame, cur_step));
				}

				/// host: compute maxvel & next dt
				float max_vel = max_vels;
				// if (max_vels > max_vel)
				//  max_vel = max_vels[id];

				//If our maximum velocity is infinity our computation will crash, so we stop here.
				if(std::isinf(max_vel)) {
					std::cout << "Maximum velocity is infinity" << std::endl;
					goto outer_loop_end;
				}

				max_vel = std::sqrt(max_vel);// this is a bug, should insert this line
				next_dt = compute_dt(max_vel, current_step_time, seconds_per_frame, dt_default);
				fmt::print(fmt::emphasis::bold, "{} --{}--> {}, defaultDt: {}, max_vel: {}\n", cur_time.count(), next_dt.count(), next_time.count(), dt_default.count(), max_vel);
				
				//TODO: IQ-Solve!
				{
					auto& cu_dev = Cuda::ref_cuda_context(gpuid);
					CudaTimer timer {cu_dev.stream_compute()};
					
					timer.tick();
					
					/*
					managed_memory.acquire<MemoryType::DEVICE>(alpha_shapes_grid_buffer.acquire());
					
					//Alpha shapes
					for(int i = 0; i < get_model_count(); ++i) {
						//Resize particle buffers if we increased the size of active bins
						if(checked_bin_counts[i] > 0) {
							surface_particle_buffers[i].resize(managed_memory_allocator, cur_num_active_bins[i]);
						}

						match(particle_bins[rollid][i])([this, &cu_dev, &i](auto& particle_buffer) {
							auto& next_particle_buffer = get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]);
							
							particle_buffer.cellbuckets = particle_buffer.cellbuckets_virtual;
							particle_buffer.bin_offsets = particle_buffer.bin_offsets_virtual;
							particle_buffer.cell_particle_counts = particle_buffer.cell_particle_counts_virtual;
							next_particle_buffer.particle_bucket_sizes = next_particle_buffer.particle_bucket_sizes_virtual;
							managed_memory.acquire<MemoryType::DEVICE>(
								  particle_buffer.acquire()
								, next_particle_buffer.acquire()
								, grid_blocks[0][i].acquire()
								, surface_particle_buffers[i].acquire()
								, reinterpret_cast<void**>(&particle_buffer.cellbuckets)
								, reinterpret_cast<void**>(&particle_buffer.bin_offsets)
								, reinterpret_cast<void**>(&particle_buffer.cell_particle_counts)
								, reinterpret_cast<void**>(&next_particle_buffer.particle_bucket_sizes)
							);
							
							//Clear buffer before use
							cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, clear_alpha_shapes_particle_buffer, particle_buffer, next_particle_buffer, partitions[(rollid + 1) % BIN_COUNT], surface_particle_buffers[i]);
							
							//FIXME: Does not yet work, maybe also need to reduce block dimension?
							for(unsigned int start_index = 0; start_index < partition_block_count; start_index += ALPHA_SHAPES_MAX_KERNEL_SIZE){
								LaunchConfig alpha_shapes_launch_config(0, 0);
								alpha_shapes_launch_config.dg = dim3(std::min(ALPHA_SHAPES_MAX_KERNEL_SIZE, partition_block_count - start_index) * config::G_BLOCKVOLUME);
								alpha_shapes_launch_config.db = dim3(ALPHA_SHAPES_BLOCK_SIZE, 1, 1);
								
								//partition_block_count; {config::G_BLOCKSIZE, config::G_BLOCKSIZE, config::G_BLOCKSIZE}
								cu_dev.compute_launch(std::move(alpha_shapes_launch_config), alpha_shapes, particle_buffer, partitions[(rollid + 1) % BIN_COUNT], partitions[rollid], grid_blocks[0][i], surface_particle_buffers[i], alpha_shapes_grid_buffer, static_cast<int*>(nullptr), static_cast<int*>(nullptr), static_cast<unsigned int*>(nullptr), start_index, static_cast<int>(cur_frame));
							}
							
							managed_memory.release(
								  particle_buffer.release()
								, next_particle_buffer.release()
								, grid_blocks[0][i].release()
								, surface_particle_buffers[i].release()
								, particle_buffer.cellbuckets_virtual
								, particle_buffer.bin_offsets_virtual
								, particle_buffer.cell_particle_counts_virtual
								, next_particle_buffer.particle_bucket_sizes_virtual
							);
						});
						
						cu_dev.syncStream<streamIdx::COMPUTE>();
					}
					
					managed_memory.release(alpha_shapes_grid_buffer.release());
					
					cu_dev.syncStream<streamIdx::COMPUTE>();*/
					
					int* d_particle_count = static_cast<int*>(cu_dev.borrow(sizeof(int)));
					
					unsigned int* particle_id_mapping_buffer = tmps.particle_id_mapping_buffer;
			
					void* surface_vertex_count_ptr = surface_vertex_count;
					void* surface_triangle_count_ptr = surface_triangle_count;
					
					managed_memory.acquire<MemoryType::DEVICE>(
						  reinterpret_cast<void**>(&surface_vertex_count_ptr)
						, reinterpret_cast<void**>(&surface_triangle_count_ptr)
						, reinterpret_cast<void**>(&particle_id_mapping_buffer)
					);
						
					//Marching Cubes
					for(int i = 0; i < get_model_count(); ++i) {
						match(particle_bins[(rollid + 1) % BIN_COUNT][i])([this, &cu_dev, &i](auto& particle_buffer) {
							auto& prev_particle_buffer = get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[rollid][i]);
							
							prev_particle_buffer.cellbuckets = prev_particle_buffer.cellbuckets_virtual;
							prev_particle_buffer.bin_offsets = prev_particle_buffer.bin_offsets_virtual;
							prev_particle_buffer.cell_particle_counts = prev_particle_buffer.cell_particle_counts_virtual;
							particle_buffer.particle_bucket_sizes = particle_buffer.particle_bucket_sizes_virtual;
							particle_buffer.blockbuckets = particle_buffer.blockbuckets_virtual;
							managed_memory.acquire<MemoryType::DEVICE>(
								  particle_buffer.acquire()
								, prev_particle_buffer.acquire()
								//, grid_blocks[0][i].acquire()
								, reinterpret_cast<void**>(&prev_particle_buffer.cellbuckets)
								, reinterpret_cast<void**>(&prev_particle_buffer.bin_offsets)
								, reinterpret_cast<void**>(&prev_particle_buffer.cell_particle_counts)
								, reinterpret_cast<void**>(&particle_buffer.particle_bucket_sizes)
								, reinterpret_cast<void**>(&particle_buffer.blockbuckets)
							);
						});
						
						//Calculate bounding box
						std::array<int, 3>* bounding_box_min_device = static_cast<std::array<int, 3>*>(cu_dev.borrow(3 * sizeof(int)));
						std::array<int, 3>* bounding_box_max_device = static_cast<std::array<int, 3>*>(cu_dev.borrow(3 * sizeof(int)));
						
						//Init with max/min
						//NOTE: cannot use std::numeric_limits<int>::max() cause that sets value to -1. Maybe use thrust to fill arrays?
						thrust::fill(thrust::device, reinterpret_cast<int*>(bounding_box_min_device), reinterpret_cast<int*>(bounding_box_min_device) + 3, std::numeric_limits<int>::max());
						thrust::fill(thrust::device, reinterpret_cast<int*>(bounding_box_max_device), reinterpret_cast<int*>(bounding_box_max_device) + 3, std::numeric_limits<int>::min());
						
						cu_dev.syncStream<streamIdx::COMPUTE>();
						
						//match(particle_bins[rollid][i])([this, &cu_dev, &i, &bounding_box_min_device, &bounding_box_max_device](const auto& particle_buffer) {
						//	//partition_block_count; G_PARTICLE_BATCH_CAPACITY
						//	cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, get_bounding_box, partitions[rollid], partitions[(rollid + 1) % BIN_COUNT], particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), grid_blocks[0][i], bounding_box_min_device, bounding_box_max_device);
						//});
						match(particle_bins[rollid][i])([this, &cu_dev, &i, &bounding_box_min_device, &bounding_box_max_device](const auto& particle_buffer) {
							//partition_block_count; G_PARTICLE_BATCH_CAPACITY
							cu_dev.compute_launch({partition_block_count, config::G_BLOCKVOLUME}, get_bounding_box, partitions[rollid], particle_buffer, bounding_box_min_device, bounding_box_max_device);
						});
						//cu_dev.compute_launch({partition_block_count, config::G_BLOCKVOLUME}, get_bounding_box, partition_block_count, partitions[rollid], grid_blocks[0][i], bounding_box_min_device, bounding_box_max_device);
					
						ivec3 bounding_box_min;
						ivec3 bounding_box_max;
						check_cuda_errors(cudaMemcpyAsync(bounding_box_min.data(), bounding_box_min_device, 3 * sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
						check_cuda_errors(cudaMemcpyAsync(bounding_box_max.data(), bounding_box_max_device, 3 * sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
						
						cu_dev.syncStream<streamIdx::COMPUTE>();
						
						//Only perceed if bounds are valid
						if(
							   (bounding_box_min[0] != std::numeric_limits<int>::max())
							&& (bounding_box_min[1] != std::numeric_limits<int>::max())
							&& (bounding_box_min[2] != std::numeric_limits<int>::max())
							&& (bounding_box_max[0] != std::numeric_limits<int>::min())
							&& (bounding_box_max[1] != std::numeric_limits<int>::min())
							&& (bounding_box_max[2] != std::numeric_limits<int>::min())
						){
							//Init particle count with 0
							check_cuda_errors(cudaMemsetAsync(d_particle_count, 0, sizeof(int), cu_dev.stream_compute()));
							
							//Generate particle_id_mapping
							match(particle_bins[rollid][i])([this, &cu_dev, &i, &d_particle_count, &particle_id_mapping_buffer](const auto& particle_buffer) {
								//partition_block_count; G_PARTICLE_BATCH_CAPACITY
								cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, generate_particle_id_mapping, partitions[rollid], partitions[(rollid + 1) % BIN_COUNT], particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), particle_id_mapping_buffer, d_particle_count);
							});
							
							cu_dev.syncStream<streamIdx::COMPUTE>();
							
							//Extend to neighbour cells
							bounding_box_min -= ivec3(static_cast<int>(MARCHING_CUBES_INTERPOLATION_DEGREE) + 1, static_cast<int>(MARCHING_CUBES_INTERPOLATION_DEGREE) + 1, static_cast<int>(MARCHING_CUBES_INTERPOLATION_DEGREE) + 1);
							bounding_box_max += ivec3(static_cast<int>(MARCHING_CUBES_INTERPOLATION_DEGREE) + 1, static_cast<int>(MARCHING_CUBES_INTERPOLATION_DEGREE) + 1, static_cast<int>(MARCHING_CUBES_INTERPOLATION_DEGREE) + 1);
							
							//NOTE: Plus 1 cause both min and max are inclusive
							const ivec3 marching_cubes_grid_size = ((bounding_box_max - bounding_box_min + ivec3(1, 1, 1)) * MARCHING_CUBES_GRID_SCALING).cast<int>();
							const vec3 bounding_box_offset = (vec3(grid_blocks[0][i].get_offset()[0], grid_blocks[0][i].get_offset()[1], grid_blocks[0][i].get_offset()[2]) + bounding_box_min) * config::G_DX;
							const size_t marching_cubes_block_count = marching_cubes_grid_size[0] * marching_cubes_grid_size[1] * marching_cubes_grid_size[2];
							
							LaunchConfig marching_cubes_launch_config(0, 0);
							marching_cubes_launch_config.dg = dim3(((marching_cubes_grid_size[0] + 4 - 1) / 4), ((marching_cubes_grid_size[1] + 4 - 1) / 4), ((marching_cubes_grid_size[2] + 4 - 1) / 4));
							marching_cubes_launch_config.db = dim3(4, 4, 4);
							
							//std::cout << "Min: " << bounding_box_min[0] << " " << bounding_box_min[1] << " " << bounding_box_min[2] << std::endl;
							//std::cout << "Max: " << bounding_box_max[0] << " " << bounding_box_max[1] << " " << bounding_box_max[2] << std::endl;
							//std::cout << "Size: " << marching_cubes_grid_size[0] << " " << marching_cubes_grid_size[1] << " " << marching_cubes_grid_size[2] << std::endl;
							//std::cout << "Offset: " << bounding_box_offset[0] << " " << bounding_box_offset[1] << " " << bounding_box_offset[2] << std::endl;
							
							//Resize and clear grid
							if(global_marching_cubes_block_count < marching_cubes_block_count){
								if(global_marching_cubes_block_count > 0){
									global_marching_cubes_block_count = std::max(marching_cubes_block_count / global_marching_cubes_block_count, static_cast<size_t>(2)) * global_marching_cubes_block_count;
								}else{
									global_marching_cubes_block_count = marching_cubes_block_count;
								}
								
								marching_cubes_grid_buffer.resize(managed_memory_allocator, global_marching_cubes_block_count);
							}
							marching_cubes_grid_buffer.reset(marching_cubes_block_count, cu_dev);
							
							//Resize particle buffers if we increased the size of active bins
							if(checked_bin_counts[i] > 0) {
								surface_particle_buffers[i].resize(managed_memory_allocator, cur_num_active_bins[i]);
							}
							
							cu_dev.syncStream<streamIdx::COMPUTE>();
							
							managed_memory.acquire<MemoryType::DEVICE>(
									marching_cubes_grid_buffer.acquire()
								  , surface_particle_buffers[i].acquire()
							);
							
							//Clear particle buffer
							match(particle_bins[rollid][i])([this, &cu_dev, &i](const auto& particle_buffer) {
								cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, marching_cubes_clear_surface_particle_buffer, particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), partitions[(rollid + 1) % BIN_COUNT], surface_particle_buffers[i]);
							});
							
							//Calculate densities
							match(particle_bins[rollid][i])([this, &cu_dev, &i, &bounding_box_offset, &marching_cubes_grid_size](const auto& particle_buffer) {
								//partition_block_count; G_PARTICLE_BATCH_CAPACITY
								cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, marching_cubes_calculate_density, partitions[rollid], partitions[(rollid + 1) % BIN_COUNT], particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), marching_cubes_grid_buffer, bounding_box_offset.data_arr(), marching_cubes_grid_size.data_arr());
							});
							
							//Init with default values
							match(particle_bins[rollid][i])([this, &cu_dev, &i, &bounding_box_offset, &marching_cubes_grid_size](const auto& particle_buffer) {
								//partition_block_count; G_PARTICLE_BATCH_CAPACITY
								cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, marching_cubes_init_surface_particle_buffer, partitions[rollid], partitions[(rollid + 1) % BIN_COUNT], particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), surface_particle_buffers[i], marching_cubes_grid_buffer, bounding_box_offset.data_arr(), marching_cubes_grid_size.data_arr());
							});
							
							//TODO: Do this before setting grid size to minimize it
							//Sort out invalid cells
							uint32_t* removed_cells = static_cast<uint32_t*>(cu_dev.borrow(sizeof(uint32_t)));
						
							uint32_t removed_cells_host;
							do{
								//Init removed_particles with 0
								check_cuda_errors(cudaMemsetAsync(removed_cells, 0, sizeof(uint32_t), cu_dev.stream_compute()));
								
								cu_dev.syncStream<streamIdx::COMPUTE>();
								
								match(particle_bins[rollid][i])([this, &cu_dev, &marching_cubes_launch_config, &i, &bounding_box_min, &bounding_box_offset, &marching_cubes_grid_size, &removed_cells](const auto& particle_buffer) {
									const float density_threshold = particle_buffer.rho * config::MARCHING_CUBES_DENSITY_THRESHOLD_FACTOR;
									
									//partition_block_count; G_PARTICLE_BATCH_CAPACITY
									cu_dev.compute_launch(std::move(LaunchConfig(marching_cubes_launch_config)), marching_cubes_sort_out_invalid_cells, partitions[rollid], partitions[(rollid + 1) % BIN_COUNT], particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), density_threshold, marching_cubes_grid_buffer, bounding_box_min.data_arr(), bounding_box_offset.data_arr(), marching_cubes_grid_size.data_arr(), removed_cells);
								});
								
								check_cuda_errors(cudaMemcpyAsync(&removed_cells_host, removed_cells, sizeof(uint32_t), cudaMemcpyDefault, cu_dev.stream_compute()));
							
								cu_dev.syncStream<streamIdx::COMPUTE>();
							}while(removed_cells_host > 0);
							
							//Initialize surface_triangle_count and surface_vertex_count with 0
							check_cuda_errors(cudaMemsetAsync(surface_vertex_count_ptr, 0, sizeof(uint32_t), cu_dev.stream_compute()));
							check_cuda_errors(cudaMemsetAsync(surface_triangle_count_ptr, 0, sizeof(uint32_t), cu_dev.stream_compute()));
								
							cu_dev.syncStream<streamIdx::COMPUTE>();
				
							//Get counts
							match(particle_bins[rollid][i])([this, &cu_dev, &i, &marching_cubes_launch_config, &bounding_box_min, &bounding_box_offset, &marching_cubes_grid_size, &surface_vertex_count_ptr, &surface_triangle_count_ptr, &particle_id_mapping_buffer](const auto& particle_buffer) {
								const float density_threshold = particle_buffer.rho * config::MARCHING_CUBES_DENSITY_THRESHOLD_FACTOR;
								
								cu_dev.compute_launch(std::move(LaunchConfig(marching_cubes_launch_config)), marching_cubes_gen_vertices, partitions[(rollid + 1) % BIN_COUNT], particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), bounding_box_min.data_arr(), bounding_box_offset.data_arr(), marching_cubes_grid_size.data_arr(), density_threshold, marching_cubes_grid_buffer, static_cast<uint32_t*>(surface_vertex_count_ptr));
								cu_dev.compute_launch(std::move(LaunchConfig(marching_cubes_launch_config)), marching_cubes_gen_faces, partitions[(rollid + 1) % BIN_COUNT], particle_buffer, particle_id_mapping_buffer, surface_particle_buffers[i], bounding_box_min.data_arr(), bounding_box_offset.data_arr(), marching_cubes_grid_size.data_arr(), density_threshold, marching_cubes_grid_buffer, static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(surface_triangle_count_ptr));
							});
							
							cu_dev.syncStream<streamIdx::COMPUTE>();
								
							managed_memory.release(
								  //grid_blocks[0][i].release()
								  marching_cubes_grid_buffer.release()
							);
						}
						
						match(particle_bins[(rollid + 1) % BIN_COUNT][i])([this, &cu_dev, &i](auto& particle_buffer) {
							auto& prev_particle_buffer = get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[rollid][i]);
							
							managed_memory.release(
								  particle_buffer.release()
								, prev_particle_buffer.release()
								//, grid_blocks[0][i].release()
								, prev_particle_buffer.cellbuckets_virtual
								, prev_particle_buffer.bin_offsets_virtual
								, prev_particle_buffer.cell_particle_counts_virtual
								, particle_buffer.particle_bucket_sizes_virtual
								, particle_buffer.blockbuckets_virtual
							);
						});
					}
					
					managed_memory.release(
						  surface_vertex_count
						, surface_triangle_count
						, tmps.particle_id_mapping_buffer
					);
					
					timer.tock(fmt::format("GPU[{}] frame {} step {} surface_reconstruction", gpuid, cur_frame, cur_step));
					
					timer.tick();
					
					//All active blocks and neighbours (and a bit more). All neighbours of these blocks do also exist (exterior_block_count with each block having 4 cells being suifficient for a kernel up to 4)
					const int coupling_block_count = total_neighbor_block_count;
					
					//std::cout << "Coupling block count: " << coupling_block_count << std::endl;
					
					for(int i = 0; i < solid_fluid_couplings.size(); ++i){
						const int solid_id = solid_fluid_couplings[i][0];
						const int fluid_id = solid_fluid_couplings[i][1];
						
						//Resize and clear matrix and vectors
						iq_rhs_array.resize_and_reset(iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME);
						iq_result_array.resize_and_reset(iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME);
						
						iq_lhs_rows.resize_and_reset(iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME + 1);
						
						iq_solve_velocity_result_array.resize_and_reset(3 * iq::SOLVE_VELOCITY_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME);
						
						iq_solve_velocity_rows.resize_and_reset(3 * iq::SOLVE_VELOCITY_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME + 1);
						
						iq_lhs_scaling_solid_values.resize_and_reset(exterior_block_count * config::G_BLOCKVOLUME);
						iq_lhs_scaling_fluid_values.resize_and_reset(exterior_block_count * config::G_BLOCKVOLUME);
						iq_lhs_mass_solid_values.resize_and_reset(3 * exterior_block_count * config::G_BLOCKVOLUME);
						iq_lhs_mass_fluid_values.resize_and_reset(3 * exterior_block_count * config::G_BLOCKVOLUME);
						
						iq_lhs_3_1_rows.resize_and_reset(3 * exterior_block_count * config::G_BLOCKVOLUME + 1);
						iq_lhs_3_1_columns.resize_and_reset(3 * coupling_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
						
						iq_lhs_gradient_solid_values.resize_and_reset(3 * coupling_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
						iq_lhs_gradient_fluid_values.resize_and_reset(3 * coupling_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
						iq_lhs_coupling_solid_values.resize_and_reset(3 * coupling_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
						iq_lhs_coupling_fluid_values.resize_and_reset(3 * coupling_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
						iq_lhs_boundary_fluid_values.resize_and_reset(3 * coupling_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
						
						gko_identity_values.resize_and_reset(exterior_block_count * config::G_BLOCKVOLUME);
						
						iq_rhs_array.fill(0.0f);
						iq_result_array.fill(0.0f);
						
						iq_solve_velocity_result_array.fill(0.0f);
						
						iq_lhs_scaling_solid_values.fill(0.0f);
						iq_lhs_scaling_fluid_values.fill(0.0f);
						iq_lhs_mass_solid_values.fill(0.0f);
						iq_lhs_mass_fluid_values.fill(0.0f);
						
						iq_lhs_3_1_rows.fill(0);
						iq_lhs_3_1_columns.fill(0);
						
						iq_lhs_gradient_solid_values.fill(0.0f);
						iq_lhs_gradient_fluid_values.fill(0.0f);
						iq_lhs_coupling_solid_values.fill(0.0f);
						iq_lhs_coupling_fluid_values.fill(0.0f);
						iq_lhs_boundary_fluid_values.fill(0.0f);
						
						//Identity is filled with 1's
						gko_identity_values.fill(1.0f);
						
						ginkgo_executor->synchronize();
						
						//Init rows and columns
						size_t* lhs_num_blocks_per_row_ptr;
						size_t* solve_velocity_num_blocks_per_row_ptr;
						size_t* temporary_num_blocks_per_row_ptr;
						std::array<size_t, iq::LHS_MATRIX_SIZE_X>* lhs_block_offsets_per_row_ptr;
						std::array<size_t, iq::SOLVE_VELOCITY_MATRIX_SIZE_X>* solve_velocity_block_offsets_per_row_ptr;
						std::array<size_t, 1>* temporary_block_offsets_per_row_ptr;
						cudaGetSymbolAddress(reinterpret_cast<void**>(&lhs_num_blocks_per_row_ptr), iq::lhs_num_blocks_per_row);
						cudaGetSymbolAddress(reinterpret_cast<void**>(&solve_velocity_num_blocks_per_row_ptr), iq::solve_velocity_num_blocks_per_row);
						cudaGetSymbolAddress(reinterpret_cast<void**>(&lhs_block_offsets_per_row_ptr), iq::lhs_block_offsets_per_row);
						cudaGetSymbolAddress(reinterpret_cast<void**>(&solve_velocity_block_offsets_per_row_ptr), iq::solve_velocity_block_offsets_per_row);
						cudaGetSymbolAddress(reinterpret_cast<void**>(&temporary_num_blocks_per_row_ptr), iq::temporary_num_blocks_per_row);
						cudaGetSymbolAddress(reinterpret_cast<void**>(&temporary_block_offsets_per_row_ptr), iq::temporary_block_offsets_per_row);
						//cu_dev.compute_launch({coupling_block_count, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, iq::clear_iq_system<iq::LHS_MATRIX_SIZE_X, iq::LHS_MATRIX_SIZE_Y, iq::NUM_ROWS_PER_BLOCK, iq::NUM_COLUMNS_PER_BLOCK, 1, Partition<1>>, lhs_num_blocks_per_row_ptr, lhs_block_offsets_per_row_ptr, static_cast<uint32_t>(coupling_block_count), static_cast<uint32_t>(exterior_block_count), partitions[rollid], iq_lhs_rows.get_data(), iq_lhs_columns.get_data(), iq_lhs_values.get_data());
						//cu_dev.compute_launch({coupling_block_count, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, iq::clear_iq_system<iq::SOLVE_VELOCITY_MATRIX_SIZE_X, iq::SOLVE_VELOCITY_MATRIX_SIZE_Y, iq::NUM_ROWS_PER_BLOCK, iq::NUM_COLUMNS_PER_BLOCK, 3, Partition<1>>, solve_velocity_num_blocks_per_row_ptr, solve_velocity_block_offsets_per_row_ptr, static_cast<uint32_t>(coupling_block_count), static_cast<uint32_t>(exterior_block_count), partitions[rollid], iq_solve_velocity_rows.get_data(), iq_solve_velocity_columns.get_data(), iq_solve_velocity_values.get_data());
						
						cu_dev.compute_launch({coupling_block_count, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, iq::clear_iq_system<1, 1, iq::NUM_ROWS_PER_BLOCK, iq::NUM_COLUMNS_PER_BLOCK, 3, Partition<1>>, temporary_num_blocks_per_row_ptr, temporary_block_offsets_per_row_ptr, static_cast<uint32_t>(coupling_block_count), static_cast<uint32_t>(exterior_block_count), partitions[rollid], iq_lhs_3_1_rows.get_data(), iq_lhs_3_1_columns.get_data(), iq_lhs_gradient_solid_values.get_data());
						
						cu_dev.syncStream<streamIdx::COMPUTE>();
						
						//Set last active row + 1 == number nonzero elements
						//const int lhs_number_of_nonzeros = static_cast<int>(iq::LHS_MATRIX_TOTAL_BLOCK_COUNT * coupling_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
						//const int solve_velocity_number_of_nonzeros = static_cast<int>(3 * iq::SOLVE_VELOCITY_MATRIX_TOTAL_BLOCK_COUNT * coupling_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
						const int solve_velocity_temporary_number_of_nonzeros = static_cast<int>(3 * coupling_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
						//cudaMemsetAsync((iq_lhs_rows.get_data() + ((iq::LHS_MATRIX_SIZE_Y - 1) * exterior_block_count + coupling_block_count) * config::G_BLOCKVOLUME), lhs_number_of_nonzeros, sizeof(int), cu_dev.stream_compute());
						//cudaMemsetAsync((iq_solve_velocity_rows.get_data() + 3 * ((iq::SOLVE_VELOCITY_MATRIX_SIZE_Y - 1) * exterior_block_count + coupling_block_count) * config::G_BLOCKVOLUME), solve_velocity_number_of_nonzeros, sizeof(int), cu_dev.stream_compute());
						
						//cudaMemcpyAsync((iq_lhs_rows.get_data() + ((iq::LHS_MATRIX_SIZE_Y - 1) * exterior_block_count + coupling_block_count) * config::G_BLOCKVOLUME), &lhs_number_of_nonzeros, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute());
						//cudaMemcpyAsync((iq_solve_velocity_rows.get_data() + 3 * ((iq::SOLVE_VELOCITY_MATRIX_SIZE_Y - 1) * exterior_block_count + coupling_block_count) * config::G_BLOCKVOLUME), &solve_velocity_number_of_nonzeros, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute());
						
						cudaMemcpyAsync((iq_lhs_3_1_rows.get_data() + (3 * coupling_block_count * config::G_BLOCKVOLUME)), &solve_velocity_temporary_number_of_nonzeros, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute());
						
						cu_dev.syncStream<streamIdx::COMPUTE>();
						
						//Fill empty space in row matrix
						//cu_dev.compute_launch({exterior_block_count, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, iq::fill_empty_rows<iq::LHS_MATRIX_SIZE_Y, iq::NUM_ROWS_PER_BLOCK, 1, Partition<1>>, static_cast<uint32_t>(exterior_block_count), partitions[rollid], iq_lhs_rows.get_data());
						//cu_dev.compute_launch({exterior_block_count, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, iq::fill_empty_rows<iq::SOLVE_VELOCITY_MATRIX_SIZE_Y, iq::NUM_ROWS_PER_BLOCK, 3, Partition<1>>, static_cast<uint32_t>(exterior_block_count), partitions[rollid], iq_solve_velocity_rows.get_data());
						
						cu_dev.compute_launch({exterior_block_count, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, iq::fill_empty_rows<1, iq::NUM_ROWS_PER_BLOCK, 3, Partition<1>>, static_cast<uint32_t>(exterior_block_count), partitions[rollid], iq_lhs_3_1_rows.get_data());
						
						cu_dev.syncStream<streamIdx::COMPUTE>();
						
						//Set last value of rows
						//FIXME: Not sure why, but memcpy does not seem to work correctly
						
						//FIXME: ABC: Memset does not work cause it sets bytes not whole numbers. Use thrust or ginko instead
						
						//cudaMemcpyAsync((iq_lhs_rows.get_data() + iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME), (iq_lhs_rows.get_data() + iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME - 1), sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute());
						//cudaMemcpyAsync((iq_solve_velocity_rows.get_data() + 3 * iq::SOLVE_VELOCITY_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME), (iq_solve_velocity_rows.get_data() + 3 * iq::SOLVE_VELOCITY_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME - 1), sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute());
						//cudaMemsetAsync((iq_lhs_rows.get_data() + iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME), lhs_number_of_nonzeros, sizeof(int), cu_dev.stream_compute());
						//cudaMemsetAsync((iq_solve_velocity_rows.get_data() + 3 * iq::SOLVE_VELOCITY_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME), solve_velocity_number_of_nonzeros, sizeof(int), cu_dev.stream_compute());
						//cudaMemcpyAsync((iq_lhs_rows.get_data() + iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME), &lhs_number_of_nonzeros, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute());
						//cudaMemcpyAsync((iq_solve_velocity_rows.get_data() + 3  * iq::SOLVE_VELOCITY_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME), &solve_velocity_number_of_nonzeros, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute());
						
						
						cu_dev.syncStream<streamIdx::COMPUTE>();
						
						//IQ-System create
						/*
							Creates single arrays for composition of left side, right side and solve velocity matrix (G^s, S^s, M^s, G^f, S^f, M^f, H^s, H^f, B, p^s, p^f, v^s, v^f).
							//NOTE: Storing H^T instead H and dt * M^-1 instead M and S/dt instead S.
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
						match(particle_bins[rollid][solid_id], particle_bins[rollid][fluid_id])([this, &cu_dev, &coupling_block_count, &solid_id, &fluid_id](auto& particle_buffer_solid, auto& particle_buffer_fluid) {
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
								, reinterpret_cast<void**>(&particle_buffer_solid.bin_offsets)
								, reinterpret_cast<void**>(&particle_buffer_fluid.bin_offsets)
								, reinterpret_cast<void**>(&next_particle_buffer_solid.particle_bucket_sizes)
								, reinterpret_cast<void**>(&next_particle_buffer_fluid.particle_bucket_sizes)
								, reinterpret_cast<void**>(&next_particle_buffer_solid.blockbuckets)
								, reinterpret_cast<void**>(&next_particle_buffer_fluid.blockbuckets)
							);
							
							iq::IQCreatePointers pointers;
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
							
							pointers.coupling_solid_rows = iq_lhs_3_1_rows.get_const_data();
							pointers.coupling_solid_columns = iq_lhs_3_1_columns.get_const_data();
							pointers.coupling_solid_values = iq_lhs_coupling_solid_values.get_data();
							pointers.coupling_fluid_rows = iq_lhs_3_1_rows.get_const_data();
							pointers.coupling_fluid_columns = iq_lhs_3_1_columns.get_const_data();
							pointers.coupling_fluid_values = iq_lhs_coupling_fluid_values.get_data();
							
							pointers.boundary_fluid_rows = iq_lhs_3_1_rows.get_const_data();
							pointers.boundary_fluid_columns = iq_lhs_3_1_columns.get_const_data();
							pointers.boundary_fluid_values = iq_lhs_boundary_fluid_values.get_data();
							
							pointers.iq_rhs = iq_rhs_array.get_data();
							pointers.iq_solve_velocity_result = iq_solve_velocity_result_array.get_data();
							
							cu_dev.compute_launch({coupling_block_count, iq::BLOCK_SIZE}, iq::create_iq_system, static_cast<uint32_t>(exterior_block_count), dt, particle_buffer_solid, particle_buffer_fluid, next_particle_buffer_solid, next_particle_buffer_fluid, partitions[(rollid + 1) % BIN_COUNT], partitions[rollid], grid_blocks[0][solid_id], grid_blocks[0][fluid_id], surface_particle_buffers[solid_id], surface_particle_buffers[fluid_id], pointers);
							
							managed_memory.release(
								  particle_buffer_solid.release()
								, particle_buffer_fluid.release()
								, next_particle_buffer_solid.release()
								, next_particle_buffer_fluid.release()
								, grid_blocks[0][solid_id].release()
								, grid_blocks[0][fluid_id].release()
								, surface_particle_buffers[solid_id].release()
								, surface_particle_buffers[fluid_id].release()
								, particle_buffer_solid.bin_offsets_virtual
								, particle_buffer_fluid.bin_offsets_virtual
								, next_particle_buffer_solid.particle_bucket_sizes_virtual
								, next_particle_buffer_fluid.particle_bucket_sizes_virtual
								, next_particle_buffer_solid.blockbuckets_virtual
								, next_particle_buffer_fluid.blockbuckets_virtual
							);
						});
						
						cu_dev.syncStream<streamIdx::COMPUTE>();
						
						constexpr size_t NUM_TEMPORARY_MATRICES = 1;
						std::array<std::shared_ptr<gko::matrix::Csr<float, int>>, iq::LHS_MATRIX_TOTAL_BLOCK_COUNT> iq_lhs_parts;
						std::array<std::shared_ptr<gko::matrix::Csr<float, int>>, iq::SOLVE_VELOCITY_MATRIX_TOTAL_BLOCK_COUNT> iq_solve_velocity_parts;
						std::array<std::shared_ptr<gko::matrix::Csr<float, int>>, NUM_TEMPORARY_MATRICES> temporary_matrices;
						
						for(size_t temporary_block = 0; temporary_block < iq::LHS_MATRIX_TOTAL_BLOCK_COUNT; ++temporary_block){
							iq_lhs_parts[temporary_block] = gko::share(
								gko::matrix::Csr<float, int>::create(
									  ginkgo_executor
									, gko::dim<2>(exterior_block_count * config::G_BLOCKVOLUME, exterior_block_count * config::G_BLOCKVOLUME)
								)
							);
						}
						
						for(size_t temporary_block = 0; temporary_block < iq::SOLVE_VELOCITY_MATRIX_TOTAL_BLOCK_COUNT; ++temporary_block){
							iq_solve_velocity_parts[temporary_block] = gko::share(
								gko::matrix::Csr<float, int>::create(
									  ginkgo_executor
									, gko::dim<2>(3 * exterior_block_count * config::G_BLOCKVOLUME, exterior_block_count * config::G_BLOCKVOLUME)
								)
							);
						}
						
						for(size_t temporary_block = 0; temporary_block < NUM_TEMPORARY_MATRICES; ++temporary_block){
							temporary_matrices[temporary_block] = gko::share(
								gko::matrix::Csr<float, int>::create(
									  ginkgo_executor
									, gko::dim<2>(exterior_block_count * config::G_BLOCKVOLUME, 3 * exterior_block_count * config::G_BLOCKVOLUME)
								)
							);
						}
						
						const std::shared_ptr<const gko::matrix::Diagonal<float>> scaling_solid = gko::share(
							gko::matrix::Diagonal<float>::create_const(
								  ginkgo_executor
								, exterior_block_count * config::G_BLOCKVOLUME
								, iq_lhs_scaling_solid_values.as_const_view()
							)
						);
						const std::shared_ptr<const gko::matrix::Diagonal<float>> scaling_fluid = gko::share(
							gko::matrix::Diagonal<float>::create_const(
								  ginkgo_executor
								, exterior_block_count * config::G_BLOCKVOLUME
								, iq_lhs_scaling_fluid_values.as_const_view()
							)
						);
						
						const std::shared_ptr<const gko::matrix::Diagonal<float>> mass_solid = gko::share(
							gko::matrix::Diagonal<float>::create_const(
								  ginkgo_executor
								, 3 * exterior_block_count * config::G_BLOCKVOLUME
								, iq_lhs_mass_solid_values.as_const_view()
							)
						);
						const std::shared_ptr<const gko::matrix::Diagonal<float>> mass_fluid = gko::share(
							gko::matrix::Diagonal<float>::create_const(
								  ginkgo_executor
								, 3 * exterior_block_count * config::G_BLOCKVOLUME
								, iq_lhs_mass_fluid_values.as_const_view()
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
						std::shared_ptr<gko::matrix::Csr<float, int>> gradient_fluid = gko::share(
							gko::matrix::Csr<float, int>::create(
								  ginkgo_executor
								, gko::dim<2>(3 * exterior_block_count * config::G_BLOCKVOLUME, exterior_block_count * config::G_BLOCKVOLUME)
								, iq_lhs_gradient_fluid_values.as_view()
								, iq_lhs_3_1_columns.as_view()
								, iq_lhs_3_1_rows.as_view()
							)
						);
						
						std::shared_ptr<gko::matrix::Csr<float, int>> coupling_solid = gko::share(
							gko::matrix::Csr<float, int>::create(
								  ginkgo_executor
								, gko::dim<2>(3 * exterior_block_count * config::G_BLOCKVOLUME, exterior_block_count * config::G_BLOCKVOLUME)
								, iq_lhs_coupling_solid_values.as_view()
								, iq_lhs_3_1_columns.as_view()
								, iq_lhs_3_1_rows.as_view()
							)
						);
						std::shared_ptr<gko::matrix::Csr<float, int>> coupling_fluid = gko::share(
							gko::matrix::Csr<float, int>::create(
								  ginkgo_executor
								, gko::dim<2>(3 * exterior_block_count * config::G_BLOCKVOLUME, exterior_block_count * config::G_BLOCKVOLUME)
								, iq_lhs_coupling_fluid_values.as_view()
								, iq_lhs_3_1_columns.as_view()
								, iq_lhs_3_1_rows.as_view()
							)
						);
						
						std::shared_ptr<gko::matrix::Csr<float, int>> boundary_fluid = gko::share(
							gko::matrix::Csr<float, int>::create(
								  ginkgo_executor
								, gko::dim<2>(3 * exterior_block_count * config::G_BLOCKVOLUME, exterior_block_count * config::G_BLOCKVOLUME)
								, iq_lhs_boundary_fluid_values.as_view()
								, iq_lhs_3_1_columns.as_view()
								, iq_lhs_3_1_rows.as_view()
							)
						);
						
						std::shared_ptr<gko::matrix::Dense<float>> pressure_solid = gko::share(
							gko::matrix::Dense<float>::create(
								  ginkgo_executor
								, gko::dim<2>(exterior_block_count * config::G_BLOCKVOLUME, 1)
								, std::move(gko::array<float>::view(ginkgo_executor, exterior_block_count * config::G_BLOCKVOLUME, iq_rhs_array.get_data()))
								, 1
							)
						);
						
						std::shared_ptr<gko::matrix::Dense<float>> pressure_fluid = gko::share(
							gko::matrix::Dense<float>::create(
								  ginkgo_executor
								, gko::dim<2>(exterior_block_count * config::G_BLOCKVOLUME, 1)
								, std::move(gko::array<float>::view(ginkgo_executor, exterior_block_count * config::G_BLOCKVOLUME, iq_rhs_array.get_data() + 1 * exterior_block_count * config::G_BLOCKVOLUME))
								, 1
							)
						);
						
						std::shared_ptr<gko::matrix::Dense<float>> iq_rhs_2 = gko::share(
							gko::matrix::Dense<float>::create(
								  ginkgo_executor
								, gko::dim<2>(exterior_block_count * config::G_BLOCKVOLUME, 1)
								, std::move(gko::array<float>::view(ginkgo_executor, exterior_block_count * config::G_BLOCKVOLUME, iq_rhs_array.get_data() + 2 * exterior_block_count * config::G_BLOCKVOLUME))
								, 1
							)
						);
						
						std::shared_ptr<gko::matrix::Dense<float>> iq_rhs_3 = gko::share(
							gko::matrix::Dense<float>::create(
								  ginkgo_executor
								, gko::dim<2>(exterior_block_count * config::G_BLOCKVOLUME, 1)
								, std::move(gko::array<float>::view(ginkgo_executor, exterior_block_count * config::G_BLOCKVOLUME, iq_rhs_array.get_data() + 3 * exterior_block_count * config::G_BLOCKVOLUME))
								, 1
							)
						);
						
						std::shared_ptr<gko::matrix::Dense<float>> velocity_fluid = gko::share(
							gko::matrix::Dense<float>::create(
								  ginkgo_executor
								, gko::dim<2>(3 * exterior_block_count * config::G_BLOCKVOLUME, 1)
								, std::move(gko::array<float>::view(ginkgo_executor, 3 * exterior_block_count * config::G_BLOCKVOLUME, iq_solve_velocity_result_array.get_data()))
								, 1
							)
						);
						
						std::shared_ptr<gko::matrix::Dense<float>> velocity_solid = gko::share(
							gko::matrix::Dense<float>::create(
								  ginkgo_executor
								, gko::dim<2>(3 * exterior_block_count * config::G_BLOCKVOLUME, 1)
								, std::move(gko::array<float>::view(ginkgo_executor, 3 * exterior_block_count * config::G_BLOCKVOLUME, iq_solve_velocity_result_array.get_data() + 3 * exterior_block_count * config::G_BLOCKVOLUME))
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
						
						//FIXME: Using own matrix_matrix_multiplication cause Ginkgos version (which uses CUDA) fails to produce a symmetric matrix for A * A^T
						//Create solve velocity and rhs
						
						/*
							rhs[0](p^s) = S^s/dt * p^s - G^s^T * v^s
							rhs[1](p^f) = S^f/dt * p^f - G^f^T * v^f
							rhs[2] = B * v^f - b
							rhs[3] = H^s * v^s - H^f * v^f
						*/
						scaling_solid->apply(pressure_solid, pressure_solid);
						temporary_matrices[0]->copy_from(std::move(gradient_solid->transpose()));
						temporary_matrices[0]->apply(gko_neg_one_dense, velocity_solid, gko_one_dense, pressure_solid);
						
						scaling_fluid->apply(pressure_fluid, pressure_fluid);
						temporary_matrices[0]->copy_from(std::move(gradient_fluid->transpose()));
						temporary_matrices[0]->apply(gko_neg_one_dense, velocity_fluid, gko_one_dense, pressure_fluid);
						
						temporary_matrices[0]->copy_from(std::move(boundary_fluid->transpose()));
						temporary_matrices[0]->apply(velocity_fluid, iq_rhs_2);
						//FIXME: Use b! identity->apply(gko_neg_one_dense, b, gko_one_dense, iq_rhs_2);
						
						temporary_matrices[0]->copy_from(std::move(coupling_solid->transpose()));
						temporary_matrices[0]->apply(velocity_solid, iq_rhs_3);
						
						temporary_matrices[0]->copy_from(std::move(coupling_fluid->transpose()));
						temporary_matrices[0]->apply(gko_neg_one_dense, velocity_fluid, gko_one_dense, iq_rhs_3);
						
						/*
							solve_velocity[0][0] = -dt * M^s^-1 * G^s
							solve_velocity[0][3] = dt * M^s^-1 * H^s^T
							solve_velocity[1][1] = -dt * M^f^-1 * G^f
							solve_velocity[1][2] = -dt * M^f^-1 * B
							solve_velocity[1][3] = -dt * M^f^-1 * H^f^T
						*/
						mass_solid->apply(gradient_solid, iq_solve_velocity_parts[0]);
						iq_solve_velocity_parts[0]->scale(gko_neg_one_dense);
						mass_solid->apply(coupling_solid, iq_solve_velocity_parts[1]);
						mass_fluid->apply(gradient_fluid, iq_solve_velocity_parts[2]);
						iq_solve_velocity_parts[2]->scale(gko_neg_one_dense);
						mass_fluid->apply(boundary_fluid, iq_solve_velocity_parts[3]);
						iq_solve_velocity_parts[3]->scale(gko_neg_one_dense);
						mass_fluid->apply(coupling_fluid, iq_solve_velocity_parts[4]);
						iq_solve_velocity_parts[4]->scale(gko_neg_one_dense);
						
						//Create lhs
						
						/*
							lhs[0][0] = S^s/dt + dt * G^s^T * M^s^-1 * G^s
							lhs[0][3] = -dt * G^s^T * M^s^-1 * H^s^T
							
						*/
						temporary_matrices[0]->copy_from(std::move(gradient_solid->transpose()));
						//mass_solid->rapply(temporary_matrices[0], temporary_matrices[0]);
						
						//NOTE: Using iq_lhs_parts[5] and iq_lhs_parts[8] as scratch
						matrix_matrix_multiplication_a_at_with_diagonal(exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_solid, gradient_solid, iq_lhs_parts[0], cu_dev);
						//matrix_matrix_multiplication_with_diagonal<false>(exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_solid, temporary_matrices[0], iq_lhs_parts[0], cu_dev);
						//matrix_matrix_multiplication_with_diagonal<true>(exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_solid, gradient_solid, iq_lhs_parts[0], cu_dev);
						//temporary_matrices[0]->apply(gradient_solid, iq_lhs_parts[0]);
						iq_lhs_parts[5]->copy_from(std::move(scaling_solid));
						iq_lhs_parts[8]->copy_from(gko_identity);
						iq_lhs_parts[8]->apply(gko_one_dense, iq_lhs_parts[5], gko_one_dense, iq_lhs_parts[0]);
						
						//temporary_matrices[1]->copy_from(std::move(coupling_solid->transpose()));
						//matrix_matrix_multiplication_with_diagonal<false>(exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_solid, temporary_matrices[1], iq_lhs_parts[1], cu_dev);
						matrix_matrix_multiplication_with_diagonal<true>(exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_solid, coupling_solid, iq_lhs_parts[1], cu_dev);
						iq_lhs_parts[1]->scale(gko_neg_one_dense);
						//temporary_matrices[0]->apply(gko_neg_one_dense, coupling_solid, gko_zero_dense, iq_lhs_parts[1]);
						
						/*
							lhs[1][1] = S^f/dt + dt * G^f^T * M^f^-1 * G^f
							lhs[1][2] = dt * G^f^T * M^f^-1 * B
							lhs[1][3] = dt * G^f^T * M^f^-1 * H^f^T
							
						*/
						temporary_matrices[0]->copy_from(std::move(gradient_fluid->transpose()));
						//mass_fluid->rapply(temporary_matrices[0], temporary_matrices[0]);
						
						//NOTE: Using iq_lhs_parts[5] and iq_lhs_parts[8] as scratch
						//NOTE: iq_lhs_parts[8] already set to identity
						matrix_matrix_multiplication_a_at_with_diagonal(exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_fluid, gradient_fluid, iq_lhs_parts[2], cu_dev);
						//matrix_matrix_multiplication_with_diagonal<false>(exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_fluid, temporary_matrices[0], iq_lhs_parts[2], cu_dev);
						//matrix_matrix_multiplication_with_diagonal<true>(exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_fluid, gradient_fluid, iq_lhs_parts[2], cu_dev);
						//temporary_matrices[0]->apply(gradient_fluid, iq_lhs_parts[2]);
						iq_lhs_parts[5]->copy_from(std::move(scaling_fluid));
						iq_lhs_parts[8]->apply(gko_one_dense, iq_lhs_parts[5], gko_one_dense, iq_lhs_parts[2]);
						
						//temporary_matrices[1]->copy_from(std::move(boundary_fluid->transpose()));
						//matrix_matrix_multiplication_with_diagonal<false>(exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_fluid, temporary_matrices[1], iq_lhs_parts[3], cu_dev);
						matrix_matrix_multiplication_with_diagonal<true>(exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_fluid, boundary_fluid, iq_lhs_parts[3], cu_dev);
						//temporary_matrices[0]->apply(boundary_fluid, iq_lhs_parts[3]);
						
						//temporary_matrices[1]->copy_from(std::move(coupling_fluid->transpose()));
						//matrix_matrix_multiplication_with_diagonal<false>(exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_fluid, temporary_matrices[1], iq_lhs_parts[4], cu_dev);
						matrix_matrix_multiplication_with_diagonal<true>(exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_fluid, coupling_fluid, iq_lhs_parts[4], cu_dev);
						//temporary_matrices[0]->apply(coupling_fluid, iq_lhs_parts[4]);
						
						/*
							lhs[2][3] = dt * B^T * M^f^-1 * H^f^T
							lhs[2][2] = dt * B^T * M^f^-1 * B
						*/
						temporary_matrices[0]->copy_from(std::move(boundary_fluid->transpose()));
						//mass_fluid->rapply(temporary_matrices[0], temporary_matrices[0]);
						
						//NOTE: temporary_matrices[1] already set to coupling_fluid->transpose()
						//matrix_matrix_multiplication_with_diagonal<false>(exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_fluid, temporary_matrices[1], iq_lhs_parts[7], cu_dev);
						matrix_matrix_multiplication_with_diagonal<true>(exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_fluid, coupling_fluid, iq_lhs_parts[7], cu_dev);
						//temporary_matrices[0]->apply(coupling_fluid, iq_lhs_parts[7]);
						
						matrix_matrix_multiplication_a_at_with_diagonal(exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_fluid, boundary_fluid, iq_lhs_parts[6], cu_dev);
						//matrix_matrix_multiplication_with_diagonal<false>(exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_fluid, temporary_matrices[0], iq_lhs_parts[6], cu_dev);
						//matrix_matrix_multiplication_with_diagonal<true>(exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_fluid, boundary_fluid, iq_lhs_parts[6], cu_dev);
						//temporary_matrices[0]->apply(boundary_fluid, iq_lhs_parts[6]);
						
						
						
						/*
							lhs[3][3] = dt * H^s * M^s^-1 * H^s^T
						*/
						temporary_matrices[0]->copy_from(std::move(coupling_solid->transpose()));
						//mass_solid->rapply(temporary_matrices[0], temporary_matrices[0]);
						
						matrix_matrix_multiplication_a_at_with_diagonal(exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_solid, coupling_solid, iq_lhs_parts[11], cu_dev);
						//matrix_matrix_multiplication_with_diagonal<false>(exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_solid, temporary_matrices[0], iq_lhs_parts[11], cu_dev);
						//matrix_matrix_multiplication_with_diagonal<true>(exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_solid, coupling_solid, iq_lhs_parts[11], cu_dev);
						//temporary_matrices[0]->apply(coupling_solid, iq_lhs_parts[11]);
						
						/*
							lhs[3][3] = dt * H^f * M^f^-1 * H^f^T
						*/
						temporary_matrices[0]->copy_from(std::move(coupling_fluid->transpose()));
						//mass_fluid->rapply(temporary_matrices[0], temporary_matrices[0]);
						
						//NOTE: Using iq_lhs_parts[5] and iq_lhs_parts[8] as scratch
						//NOTE: iq_lhs_parts[8] already set to identity
						matrix_matrix_multiplication_a_at_with_diagonal(exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_fluid, coupling_fluid, iq_lhs_parts[5], cu_dev);
						//matrix_matrix_multiplication_with_diagonal<false>(exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_fluid, temporary_matrices[0], iq_lhs_parts[5], cu_dev);
						//matrix_matrix_multiplication_with_diagonal<true>(exterior_block_count, exterior_block_count * config::G_BLOCKVOLUME, temporary_matrices[0], mass_fluid, coupling_fluid, iq_lhs_parts[5], cu_dev);
						
						iq_lhs_parts[8]->apply(gko_one_dense, iq_lhs_parts[5], gko_one_dense, iq_lhs_parts[11]);
						//temporary_matrices[0]->apply(gko_one_dense, coupling_fluid, gko_one_dense, iq_lhs_parts[11]);
						
						/*
							Current state:
							{
								A11 0   0   A14
								0   A22 A23 A24
								0   -   A33 A34
								-   -   -   A44
							}
						*/
						
						//Fill transposed
						/*
							lhs[2][1] = lhs[1][2]^T
							lhs[3][0] = lhs[0][3]^T
							lhs[3][1] = lhs[1][3]^T
							lhs[3][2] = lhs[2][3]^T
						*/
						iq_lhs_parts[5]->copy_from(std::move(iq_lhs_parts[3]->transpose()));
						iq_lhs_parts[8]->copy_from(std::move(iq_lhs_parts[1]->transpose()));
						iq_lhs_parts[9]->copy_from(std::move(iq_lhs_parts[4]->transpose()));
						iq_lhs_parts[10]->copy_from(std::move(iq_lhs_parts[7]->transpose()));
						
						ginkgo_executor->synchronize();
						
						//Move from temporary to actual arrays
						{
							//Fill arrays
							std::array<const int*, iq::LHS_MATRIX_TOTAL_BLOCK_COUNT> iq_lhs_parts_rows;
							std::array<const int*, iq::LHS_MATRIX_TOTAL_BLOCK_COUNT> iq_lhs_parts_columns;
							std::array<const float*, iq::LHS_MATRIX_TOTAL_BLOCK_COUNT> iq_lhs_parts_values;
							
							for(size_t j = 0; j < iq::LHS_MATRIX_TOTAL_BLOCK_COUNT; ++j){
								iq_lhs_parts_rows[j] = iq_lhs_parts[j]->get_const_row_ptrs();
								iq_lhs_parts_columns[j] = iq_lhs_parts[j]->get_const_col_idxs();
								iq_lhs_parts_values[j] = iq_lhs_parts[j]->get_const_values();
							}
							
							//Verify rows match
							const size_t row_count = exterior_block_count * config::G_BLOCKVOLUME;
							size_t lhs_number_of_nonzeros = 0;
							for(size_t j = 0; j < iq::LHS_MATRIX_TOTAL_BLOCK_COUNT; ++j){
								if(iq_lhs_parts[j]->get_size()[0] != row_count){
									std::cout << "ERROR - Mismatch in row count(" << j << "). Should be " << row_count << " but is " << iq_lhs_parts[j]->get_size()[0] << std::endl;
								}
								
								//Add number of non-zero elements
								const int current_number_of_nonzeros = iq_lhs_parts[j]->get_num_stored_elements();
								lhs_number_of_nonzeros += current_number_of_nonzeros;
								
								//Store last row element
								cudaMemcpyAsync(iq_lhs_parts[j]->get_row_ptrs() + row_count, &current_number_of_nonzeros, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute());
							}
							
							cu_dev.syncStream<streamIdx::COMPUTE>();
							
							//Resize values and columns array; Clear rows array
							iq_lhs_columns.resize_and_reset(lhs_number_of_nonzeros);
							iq_lhs_values.resize_and_reset(lhs_number_of_nonzeros);
							
							iq_lhs_rows.fill(0);
							iq_lhs_columns.fill(0);
							iq_lhs_values.fill(0.0f);
							
							ginkgo_executor->synchronize();
							
							//Add rows
							cu_dev.compute_launch({exterior_block_count, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, iq::add_block_rows<iq::LHS_MATRIX_SIZE_Y, iq::NUM_ROWS_PER_BLOCK, 1, iq::LHS_MATRIX_TOTAL_BLOCK_COUNT, Partition<1>>, lhs_num_blocks_per_row_ptr, static_cast<uint32_t>(exterior_block_count), partitions[rollid], iq_lhs_parts_rows, iq_lhs_rows.get_data());
							
							//Scan rows
							exclusive_scan(iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME + 1, iq_lhs_rows.get_data(), iq_lhs_rows.get_data(), cu_dev);
							
							//Copy columns and values
							cu_dev.compute_launch({exterior_block_count, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, iq::copy_values<iq::LHS_MATRIX_SIZE_X, iq::LHS_MATRIX_SIZE_Y, iq::NUM_ROWS_PER_BLOCK, 1, iq::LHS_MATRIX_TOTAL_BLOCK_COUNT, Partition<1>>, lhs_num_blocks_per_row_ptr, lhs_block_offsets_per_row_ptr, static_cast<uint32_t>(exterior_block_count), partitions[rollid], iq_lhs_parts_rows, iq_lhs_parts_columns, iq_lhs_parts_values, iq_lhs_rows.get_const_data(), iq_lhs_columns.get_data(), iq_lhs_values.get_data());
						}
						
						{
							//Fill arrays
							std::array<const int*, iq::SOLVE_VELOCITY_MATRIX_TOTAL_BLOCK_COUNT> iq_solve_velocity_parts_rows;
							std::array<const int*, iq::SOLVE_VELOCITY_MATRIX_TOTAL_BLOCK_COUNT> iq_solve_velocity_parts_columns;
							std::array<const float*, iq::SOLVE_VELOCITY_MATRIX_TOTAL_BLOCK_COUNT> iq_solve_velocity_parts_values;
							
							for(size_t j = 0; j < iq::SOLVE_VELOCITY_MATRIX_TOTAL_BLOCK_COUNT; ++j){
								iq_solve_velocity_parts_rows[j] = iq_solve_velocity_parts[j]->get_const_row_ptrs();
								iq_solve_velocity_parts_columns[j] = iq_solve_velocity_parts[j]->get_const_col_idxs();
								iq_solve_velocity_parts_values[j] = iq_solve_velocity_parts[j]->get_const_values();
							}
							
							//Verify rows match
							const size_t row_count = 3 * exterior_block_count * config::G_BLOCKVOLUME;
							size_t solve_velocity_number_of_nonzeros = 0;
							for(size_t j = 0; j < iq::SOLVE_VELOCITY_MATRIX_TOTAL_BLOCK_COUNT; ++j){
								if(iq_solve_velocity_parts[j]->get_size()[0] != row_count){
									std::cout << "ERROR - Mismatch in row count(" << j << "). Should be " << row_count << " but is " << iq_solve_velocity_parts[j]->get_size()[0] << std::endl;
								}
								
								//Add number of non-zero elements
								const int current_number_of_nonzeros = iq_solve_velocity_parts[j]->get_num_stored_elements();
								solve_velocity_number_of_nonzeros += current_number_of_nonzeros;
								
								//Store last row element
								cudaMemcpyAsync(iq_solve_velocity_parts[j]->get_row_ptrs() + row_count, &current_number_of_nonzeros, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute());
							}
							
							cu_dev.syncStream<streamIdx::COMPUTE>();
							
							//Resize values and columns array; Clear rows array
							iq_solve_velocity_columns.resize_and_reset(solve_velocity_number_of_nonzeros);
							iq_solve_velocity_values.resize_and_reset(solve_velocity_number_of_nonzeros);
							
							iq_solve_velocity_rows.fill(0);
							iq_solve_velocity_columns.fill(0);
							iq_solve_velocity_values.fill(0.0f);
							
							ginkgo_executor->synchronize();
							
							//Add rows
							cu_dev.compute_launch({exterior_block_count, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, iq::add_block_rows<iq::SOLVE_VELOCITY_MATRIX_SIZE_Y, iq::NUM_ROWS_PER_BLOCK, 3, iq::SOLVE_VELOCITY_MATRIX_TOTAL_BLOCK_COUNT, Partition<1>>, solve_velocity_num_blocks_per_row_ptr, static_cast<uint32_t>(exterior_block_count), partitions[rollid], iq_solve_velocity_parts_rows, iq_solve_velocity_rows.get_data());
							
							//Scan rows
							exclusive_scan(3 * iq::SOLVE_VELOCITY_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME + 1, iq_solve_velocity_rows.get_data(), iq_solve_velocity_rows.get_data(), cu_dev);
							
							//Copy columns and values
							cu_dev.compute_launch({exterior_block_count, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, iq::copy_values<iq::SOLVE_VELOCITY_MATRIX_SIZE_X, iq::SOLVE_VELOCITY_MATRIX_SIZE_Y, iq::NUM_ROWS_PER_BLOCK, 3, iq::SOLVE_VELOCITY_MATRIX_TOTAL_BLOCK_COUNT, Partition<1>>, solve_velocity_num_blocks_per_row_ptr, solve_velocity_block_offsets_per_row_ptr, static_cast<uint32_t>(exterior_block_count), partitions[rollid], iq_solve_velocity_parts_rows, iq_solve_velocity_parts_columns, iq_solve_velocity_parts_values, iq_solve_velocity_rows.get_const_data(), iq_solve_velocity_columns.get_data(), iq_solve_velocity_values.get_data());
						}
						
						cu_dev.syncStream<streamIdx::COMPUTE>();
						
						//Clear iq_solve_velocity_result_array
						iq_solve_velocity_result_array.fill(0.0f);
						
						ginkgo_executor->synchronize();
						
						//TODO: If making this non-const, move array views to prevent copying?
						const std::shared_ptr<const gko::matrix::Csr<float, int>> iq_lhs = gko::share(
							gko::matrix::Csr<float, int>::create_const(
								  ginkgo_executor
								, gko::dim<2>(iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME, iq::LHS_MATRIX_SIZE_X * exterior_block_count * config::G_BLOCKVOLUME)
								, iq_lhs_values.as_const_view()
								, iq_lhs_columns.as_const_view()
								, iq_lhs_rows.as_const_view()
							)
						);
						const std::shared_ptr<const gko::matrix::Dense<float>> iq_rhs = gko::share(
							gko::matrix::Dense<float>::create_const(
								  ginkgo_executor
								, gko::dim<2>(iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME, 1)
								, iq_rhs_array.as_const_view()
								, 1
							)
						);
						std::shared_ptr<gko::matrix::Dense<float>> iq_result = gko::share(
							gko::matrix::Dense<float>::create(
								  ginkgo_executor
								, gko::dim<2>(iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME, 1)
								, std::move(iq_result_array.as_view())
								, 1
							)
						);
						
						const std::shared_ptr<const gko::matrix::Csr<float, int>> iq_solve_velocity = gko::share(
							gko::matrix::Csr<float, int>::create_const(
								  ginkgo_executor
								, gko::dim<2>(3 * iq::SOLVE_VELOCITY_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME, iq::SOLVE_VELOCITY_MATRIX_SIZE_X * exterior_block_count * config::G_BLOCKVOLUME)
								, iq_solve_velocity_values.as_const_view()
								, iq_solve_velocity_columns.as_const_view()
								, iq_solve_velocity_rows.as_const_view()
							)
						);
						std::shared_ptr<gko::matrix::Dense<float>> iq_solve_velocity_result = gko::share(
							gko::matrix::Dense<float>::create(
								  ginkgo_executor
								, gko::dim<2>(3 * iq::SOLVE_VELOCITY_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME, 1)
								, std::move(iq_solve_velocity_result_array.as_view())
								, 1
							)
						);
						
						#if (VERIFY_IQ_MATRIX == 1)
							auto iq_lhs_transposed = iq_lhs->transpose();
							
							std::shared_ptr<gko::matrix::Csr<float, int>> iq_lhs_transposed_csr = gko::share(
								gko::matrix::Csr<float, int>::create(
									ginkgo_executor
								)
							);
							iq_lhs_transposed_csr->copy_from(iq_lhs_transposed);
						
							/*
							const int* max_column_device = thrust::max_element(thrust::device, iq_lhs->get_const_col_idxs(), iq_lhs->get_const_col_idxs() + iq_lhs->get_num_stored_elements());
							
							int max_column;
							cudaMemcpyAsync(&max_column, max_column_device, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute());
							
							cudaDeviceSynchronize();

							std::cout << "Rows: " << iq_lhs->get_size()[0] << " Columns: " << max_column << std::endl;
							std::cout << "Rows: " << iq_lhs->get_size()[0] << " Columns: " << iq_lhs_transposed_csr->get_size()[0] << std::endl;
							std::cout << "Rows: " << iq_lhs->get_num_srow_elements() << " Columns: " << iq_lhs_transposed_csr->get_num_srow_elements() << std::endl;
							*/
							
							//Test if is quadratic
							if(iq_lhs->get_size()[0] != iq_lhs_transposed_csr->get_size()[0]){
								std::cout << std::endl;
								std::cout << "IQ-Matrix is not quadratic. Row count is: " << iq_lhs->get_size()[0] << ", but Column count is: " << iq_lhs_transposed->get_size()[0] << std::endl;
							}
							
							//Test if symmetric
							iq_lhs->apply(gko_one_dense, gko::matrix::Identity<float>::create(ginkgo_executor, iq_lhs->get_size()[0]), gko_neg_one_dense, iq_lhs_transposed_csr);//iq_lhs_transposed_csr = iq_lhs - iq_lhs_transposed_csr;
							thrust::pair<const float*, const float*> min_max_device = thrust::minmax_element(thrust::device, iq_lhs_transposed_csr->get_const_values(), iq_lhs_transposed_csr->get_const_values() + iq_lhs_transposed_csr->get_num_stored_elements());
							
							if(min_max_device.first != (iq_lhs_transposed_csr->get_const_values() + iq_lhs_transposed_csr->get_num_stored_elements())){
								std::pair<float, float> min_max;
								cudaMemcpyAsync(&(min_max.first), min_max_device.first, sizeof(float), cudaMemcpyDefault, cu_dev.stream_compute());
								cudaMemcpyAsync(&(min_max.second), min_max_device.second, sizeof(float), cudaMemcpyDefault, cu_dev.stream_compute());
							
								cudaDeviceSynchronize();
								if(min_max.first != 0.0f || min_max.second != 0.0f){
									const float non_null_value = (min_max.first != 0.0f ? min_max.first : min_max.second);
									const float* non_null_position = (min_max.first != 0.0f ? min_max_device.first : min_max_device.second);
									std::cout << std::endl;
									std::cout << "IQ-Matrix is not symmetric. Non-null-entry at position " << static_cast<size_t>(non_null_position - iq_lhs_transposed_csr->get_const_values()) << " is " << non_null_value << std::endl;
									
									std::cout << "Rows per block: " << (exterior_block_count * config::G_BLOCKVOLUME) << std::endl;
									
									std::vector<int> printout_subtraction_result0(iq_lhs_transposed_csr->get_size()[0]);
									std::vector<int> printout_subtraction_result1(iq_lhs_transposed_csr->get_num_stored_elements());
									
									cudaMemcpyAsync(printout_subtraction_result0.data(), iq_lhs->get_const_row_ptrs(), sizeof(int) * iq_lhs_transposed_csr->get_size()[0] + 1, cudaMemcpyDefault, cu_dev.stream_compute());
									cudaMemcpyAsync(printout_subtraction_result1.data(), iq_lhs->get_const_col_idxs(), sizeof(int) * iq_lhs_transposed_csr->get_num_stored_elements(), cudaMemcpyDefault, cu_dev.stream_compute());
									
									write(std::cout, iq_lhs_transposed_csr);
									
									cudaDeviceSynchronize();
									
									std::cout << std::endl;
									for(size_t j = 0; j < iq_lhs_transposed_csr->get_size()[0] + 1; ++j){
										std::cout << printout_subtraction_result0[j] << " ";
									}
									std::cout << std::endl;
									for(size_t j = 0; j < iq_lhs_transposed_csr->get_num_stored_elements(); ++j){
										std::cout << printout_subtraction_result1[j] << " ";
									}
									std::cout << std::endl;
								}
							}
							
							//TODO: Test positive semi-definite; Maybe use c++ jacobi solver found online (in extra file): https://github.com/jewettaij/jacobi_pd
							
							//Write out matrix for further external checks
							std::string fn = std::string {"matrix"} + "_id[" + std::to_string(i) + "]_step[" + std::to_string(cur_step) + "].txt";
							IO::insert_job([this, coupling_block_count, cu_dev, iq_lhs, fn]() {
								std::vector<int> printout_tmp0(iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME + 1);
								std::vector<int> printout_tmp1(iq_lhs->get_num_stored_elements());
								std::vector<float> printout_tmp2(iq_lhs->get_num_stored_elements());
								
								cudaMemcpyAsync(printout_tmp0.data(), iq_lhs->get_const_row_ptrs(), sizeof(int) * iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME + 1, cudaMemcpyDefault, cu_dev.stream_compute());
								cudaMemcpyAsync(printout_tmp1.data(), iq_lhs->get_const_col_idxs(), sizeof(int) * iq_lhs->get_num_stored_elements(), cudaMemcpyDefault, cu_dev.stream_compute());
								cudaMemcpyAsync(printout_tmp2.data(), iq_lhs->get_const_values(), sizeof(float) * iq_lhs->get_num_stored_elements(), cudaMemcpyDefault, cu_dev.stream_compute());
								
								cudaDeviceSynchronize();
								
								printout_tmp0[iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME] = iq_lhs->get_num_stored_elements();
								
								std::ofstream matrix_file;
								matrix_file.open(fn);
								
								matrix_file << std::setprecision(std::numeric_limits<float>::digits10 + 1);
								
								for(size_t j = 0; j < iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME; ++j){
									matrix_file << printout_tmp0[j] << " ";
								}
								matrix_file << printout_tmp0[iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME] << std::endl;
								for(size_t j = 0; j < iq_lhs->get_num_stored_elements() - 1; ++j){
									matrix_file << printout_tmp1[j] << " ";
								}
								matrix_file << printout_tmp1[iq_lhs->get_num_stored_elements() - 1] << std::endl;
								for(size_t j = 0; j < iq_lhs->get_num_stored_elements() - 1; ++j){
									matrix_file<< printout_tmp2[j] << " ";
								}
								matrix_file << printout_tmp2[iq_lhs->get_num_stored_elements() - 1];
								
								matrix_file.close();
							});
							IO::flush();
							
						#endif
						
						/*
						std::vector<int> printout_tmp0(iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME + 1);
						std::vector<int> printout_tmp1(iq_lhs->get_num_stored_elements());
						std::vector<float> printout_tmp2(iq_lhs->get_num_stored_elements());
						std::vector<float> printout_tmp3(iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME);
						
						cudaMemcpyAsync(printout_tmp0.data(), iq_lhs->get_const_row_ptrs(), sizeof(int) * (iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME + 1), cudaMemcpyDefault, cu_dev.stream_compute());
						cudaMemcpyAsync(printout_tmp1.data(), iq_lhs->get_const_col_idxs(), sizeof(int) * iq_lhs->get_num_stored_elements(), cudaMemcpyDefault, cu_dev.stream_compute());
						cudaMemcpyAsync(printout_tmp2.data(), iq_lhs->get_const_values(), sizeof(float) * iq_lhs->get_num_stored_elements(), cudaMemcpyDefault, cu_dev.stream_compute());
						cudaMemcpyAsync(printout_tmp3.data(), iq_rhs_array.get_const_data(), sizeof(float) * iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME, cudaMemcpyDefault, cu_dev.stream_compute());
						
						cudaDeviceSynchronize();
						
						//std::cout << std::endl;
						//for(size_t j = 0; j < iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME + 1; ++j){
						//	std::cout << printout_tmp0[j] << " ";
						//}
						//std::cout << std::endl;
						//for(size_t j = 0; j < iq_lhs->get_num_stored_elements(); ++j){
						//	std::cout << printout_tmp1[j] << " ";
						//}
						std::cout << std::endl;
						
						printout_tmp0[iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME] = iq_lhs->get_num_stored_elements();
						for(size_t j = 0; j < iq::LHS_MATRIX_SIZE_Y; ++j){
							for(size_t k = 0; k < exterior_block_count * config::G_BLOCKVOLUME; ++k){
								const size_t row_length = (printout_tmp0[j * iq::NUM_ROWS_PER_BLOCK * exterior_block_count + k + 1] - printout_tmp0[j * iq::NUM_ROWS_PER_BLOCK * exterior_block_count + k]);
								if(row_length > 0){
									bool has_non_null_entry = false;
									for(size_t l = 0; l < row_length; ++l){
										if(printout_tmp2[printout_tmp0[j * iq::NUM_ROWS_PER_BLOCK * exterior_block_count + k] + l] != 0.0f){
											has_non_null_entry = true;
											break;
										}
									}
									if(has_non_null_entry){
										std::cout << (j * iq::NUM_ROWS_PER_BLOCK * exterior_block_count + k) << ": ";
										for(size_t l = 0; l < row_length; ++l){
											if(printout_tmp2[printout_tmp0[j * iq::NUM_ROWS_PER_BLOCK * exterior_block_count + k] + l] != 0.0f){
												std::cout << "(" << printout_tmp1[printout_tmp0[j * iq::NUM_ROWS_PER_BLOCK * exterior_block_count + k] + l] << ", " << printout_tmp2[printout_tmp0[j * iq::NUM_ROWS_PER_BLOCK * exterior_block_count + k] + l] << ") ";
											}
										}
										std::cout << std::endl;
									}
								}
							}
							std::cout << "##############" << std::endl;
						}
						std::cout << std::endl;
						
						for(size_t j = 0; j < iq::LHS_MATRIX_SIZE_Y; ++j){
							for(size_t k = 0; k < exterior_block_count * config::G_BLOCKVOLUME; ++k){
								std::cout << printout_tmp3[j * exterior_block_count * config::G_BLOCKVOLUME + k] << " ";
							}
							std::cout << std::endl;
						}
						std::cout << std::endl;
						*/
						
						
						/*
						std::vector<int> printout_tmp0(3 * iq::SOLVE_VELOCITY_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME + 1);
						std::vector<int> printout_tmp1(iq_solve_velocity->get_num_stored_elements());
						std::vector<float> printout_tmp2(iq_solve_velocity->get_num_stored_elements());
						
						cudaMemcpyAsync(printout_tmp0.data(), iq_solve_velocity->get_const_row_ptrs(), sizeof(int) * (3 * iq::SOLVE_VELOCITY_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME + 1), cudaMemcpyDefault, cu_dev.stream_compute());
						cudaMemcpyAsync(printout_tmp1.data(), iq_solve_velocity->get_const_col_idxs(), sizeof(int) * iq_solve_velocity->get_num_stored_elements(), cudaMemcpyDefault, cu_dev.stream_compute());
						cudaMemcpyAsync(printout_tmp2.data(), iq_solve_velocity->get_const_values(), sizeof(float) * iq_solve_velocity->get_num_stored_elements(), cudaMemcpyDefault, cu_dev.stream_compute());
						
						cudaDeviceSynchronize();
						
						//std::cout << std::endl;
						//for(size_t j = 0; j < 3 * iq::SOLVE_VELOCITY_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME + 1; ++j){
						//	std::cout << printout_tmp0[j] << " ";
						//}
						//std::cout << std::endl;
						//for(size_t j = 0; j < iq_solve_velocity->get_num_stored_elements(); ++j){
						//	std::cout << printout_tmp1[j] << " ";
						//}
						std::cout << std::endl;
						
						for(size_t j = 0; j < iq::SOLVE_VELOCITY_MATRIX_SIZE_Y; ++j){
							for(size_t k = 0; k < exterior_block_count * config::G_BLOCKVOLUME; ++k){
								for(size_t m = 0; m < 3; ++m){
									const size_t row_length = (printout_tmp0[j * 3 * iq::NUM_ROWS_PER_BLOCK * exterior_block_count + 3 * k + m + 1] - printout_tmp0[j * 3 * iq::NUM_ROWS_PER_BLOCK * exterior_block_count + 3 * k + m]);
									if(row_length > 0){
										bool has_non_null_entry = false;
										for(size_t l = 0; l < row_length; ++l){
											if(printout_tmp2[printout_tmp0[j * 3 * iq::NUM_ROWS_PER_BLOCK * exterior_block_count + 3 * k + m] + l] != 0.0f){
												has_non_null_entry = true;
												break;
											}
										}
										if(has_non_null_entry){
											std::cout << (j * 3 * iq::NUM_ROWS_PER_BLOCK * exterior_block_count + 3 * k + m) << ": ";
											
											for(size_t l = 0; l < row_length; ++l){
												if(printout_tmp2[printout_tmp0[j * 3 * iq::NUM_ROWS_PER_BLOCK * exterior_block_count + 3 * k + m] + l] != 0.0f){
													std::cout << "(" << printout_tmp1[printout_tmp0[j * 3 * iq::NUM_ROWS_PER_BLOCK * exterior_block_count + 3 * k + m] + l] << ", " << printout_tmp2[printout_tmp0[j * 3 * iq::NUM_ROWS_PER_BLOCK * exterior_block_count + 3 * k + m] + l] << ") ";
												}
											}
											std::cout << std::endl;
										}
									}
								}
							}
							std::cout << "##############" << std::endl;
						}
						std::cout << std::endl;
						*/
						
						//IQ-System solve
						// Create solver
						const std::chrono::steady_clock::time_point tic_generate = std::chrono::steady_clock::now();
						ginkgo_executor->synchronize();
						std::unique_ptr<gko::solver::Cg<float>> iq_solver = iq_solver_factory->generate(iq_lhs);
						
						std::shared_ptr<const gko::log::Convergence<float>> iq_logger = gko::log::Convergence<float>::create();
						iq_solver->add_logger(iq_logger);
						
						ginkgo_executor->synchronize();
						const std::chrono::steady_clock::time_point toc_generate = std::chrono::steady_clock::now();

						// Solve system
						const std::chrono::steady_clock::time_point tic_solve = std::chrono::steady_clock::now();
						iq_solver->apply(iq_rhs, iq_result);
						ginkgo_executor->synchronize();
						const std::chrono::steady_clock::time_point toc_solve = std::chrono::steady_clock::now();
						
						std::cout << "IQ-Solver time: " << std::chrono::duration_cast<std::chrono::milliseconds>(toc_solve - tic_generate).count() << " ms (Generate: " << std::chrono::duration_cast<std::chrono::milliseconds>(toc_generate - tic_generate).count() << " ms, Solve: " << std::chrono::duration_cast<std::chrono::milliseconds>(toc_solve - tic_solve).count() << " ms)" << std::endl;
						
						// Print solver statistics
						if(!iq_logger->has_converged()){
							std::cout << "IQ-Solver has not converged" << std::endl;
							auto res = gko::as<gko::matrix::Dense<float>>(iq_logger->get_residual_norm());
							std::cout << "IQ-Solver residual norm sqrt(r^T r): " << std::endl;
							gko::write(std::cout, res);
						}
						
						//Update velocity and ghost matrix strain
						//v_s,t+1 = v_s,t + (- dt * M_s^-1 * G_s * p_g,t+1 + dt * M_s^-1 * H_s^T * h,t+1)
						//v_f,t+1 = v_f,t + (- dt * M_f^-1 * G_f * p_f,t+1 - dt * M_f^-1 * B * y,t+1 - dt * M_f^-1 * H_f^T * h,t+1)
						
						//Calculate delta_v
						iq_solve_velocity->apply(iq_result, iq_solve_velocity_result);
						
						ginkgo_executor->synchronize();
						
						/*
						std::vector<float> printout_tmp4(3 * iq::SOLVE_VELOCITY_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME);
						std::vector<float> printout_tmp5(iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME);
						
						cudaMemcpyAsync(printout_tmp4.data(), iq_solve_velocity_result->get_const_values(), sizeof(float) * 3 * iq::SOLVE_VELOCITY_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME, cudaMemcpyDefault, cu_dev.stream_compute());
						cudaMemcpyAsync(printout_tmp5.data(), iq_result->get_const_values(), sizeof(float) * iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME, cudaMemcpyDefault, cu_dev.stream_compute());
						
						cudaDeviceSynchronize();
						
						std::cout << std::endl;
						for(size_t k = 0; k < iq::SOLVE_VELOCITY_MATRIX_SIZE_Y; ++k){
							for(size_t j = 0; j < exterior_block_count * config::G_BLOCKVOLUME; ++j){
								std::cout << '{';
								std::cout << printout_tmp4[3 * k * exterior_block_count * config::G_BLOCKVOLUME + 3 * j + 0] << ", ";
								std::cout << printout_tmp4[3 * k * exterior_block_count * config::G_BLOCKVOLUME + 3 * j + 1] << ", ";
								std::cout << printout_tmp4[3 * k * exterior_block_count * config::G_BLOCKVOLUME + 3 * j + 2];
								std::cout << "} ";
							}
							std::cout << std::endl;
						}
						std::cout << std::endl;
						for(size_t k = 0; k < iq::LHS_MATRIX_SIZE_Y; ++k){
							for(size_t j = 0; j < exterior_block_count * config::G_BLOCKVOLUME; ++j){
								std::cout << printout_tmp5[k * exterior_block_count * config::G_BLOCKVOLUME + j] << " ";
							}
							std::cout << std::endl;
						}*/
						
						
						/*
						std::vector<int> printout_tmp0(iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME + 1);
						std::vector<int> printout_tmp1(iq_lhs->get_num_stored_elements());
						std::vector<float> printout_tmp2(iq_lhs->get_num_stored_elements());
						std::vector<float> printout_tmp3(iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME);
						std::vector<float> printout_tmp4(3 * iq::SOLVE_VELOCITY_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME);
						std::vector<float> printout_tmp5(iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME);
						
						cudaMemcpyAsync(printout_tmp0.data(), iq_lhs->get_const_row_ptrs(), sizeof(int) * (iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME + 1), cudaMemcpyDefault, cu_dev.stream_compute());
						cudaMemcpyAsync(printout_tmp1.data(), iq_lhs->get_const_col_idxs(), sizeof(int) iq_lhs->get_num_stored_elements(), cudaMemcpyDefault, cu_dev.stream_compute());
						cudaMemcpyAsync(printout_tmp2.data(), iq_lhs->get_const_values(), sizeof(float) * iq_lhs->get_num_stored_elements(), cudaMemcpyDefault, cu_dev.stream_compute());
						cudaMemcpyAsync(printout_tmp3.data(), iq_rhs_array.get_const_data(), sizeof(float) * iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME, cudaMemcpyDefault, cu_dev.stream_compute());
						cudaMemcpyAsync(printout_tmp4.data(), iq_solve_velocity_result->get_const_values(), sizeof(float) * 3 * iq::SOLVE_VELOCITY_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME, cudaMemcpyDefault, cu_dev.stream_compute());
						cudaMemcpyAsync(printout_tmp5.data(), iq_result->get_const_values(), sizeof(float) * iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME, cudaMemcpyDefault, cu_dev.stream_compute());
						
						cudaDeviceSynchronize();
						
						const std::array<size_t, iq::LHS_MATRIX_SIZE_Y> tmp_num_blocks_per_row = {
							  2
							, 3
							, 3
							, 4
						};
						printout_tmp0[iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME] = iq_lhs->get_num_stored_elements();
						
						std::cout << std::endl;
						for(size_t k = 0; k < iq::LHS_MATRIX_SIZE_Y; ++k){
							for(size_t j = 0; j < exterior_block_count * config::G_BLOCKVOLUME; ++j){
								//if(printout_tmp5[k * exterior_block_count * config::G_BLOCKVOLUME + j] > 1e4){
								if(printout_tmp5[k * exterior_block_count * config::G_BLOCKVOLUME + j] != 0.0f){
									std::cout << "P: " << printout_tmp5[k * exterior_block_count * config::G_BLOCKVOLUME + j] << std::endl;
									
									std::cout << "A: " << std::endl;
									for(size_t k1 = 0; k1 < iq::LHS_MATRIX_SIZE_Y; ++k1){
										for(size_t j1 = 0; j1 < exterior_block_count * config::G_BLOCKVOLUME; ++j1){
											const size_t row_length = (printout_tmp0[k1 * iq::NUM_ROWS_PER_BLOCK * exterior_block_count + j1 + 1] - printout_tmp0[k1 * iq::NUM_ROWS_PER_BLOCK * exterior_block_count + j1]);
											if(row_length > 0){
												bool found_column = false;
												bool not_null = false;
												for(size_t l = 0; l < row_length; ++l){
													if(printout_tmp1[printout_tmp0[k1 * iq::NUM_ROWS_PER_BLOCK * exterior_block_count + j1] + l] == (k * exterior_block_count * config::G_BLOCKVOLUME + j) && printout_tmp2[printout_tmp0[k1 * iq::NUM_ROWS_PER_BLOCK * exterior_block_count + j1] + l] != 0.0f){
														found_column = true;
													}
													if(printout_tmp2[printout_tmp0[k1 * iq::NUM_ROWS_PER_BLOCK * exterior_block_count + j1] + l] != 0.0f){
														not_null = true;
													}
												}
												if(not_null && found_column){
													std::cout << printout_tmp3[k1 * exterior_block_count * config::G_BLOCKVOLUME + j1] << " = ";
													for(size_t l = 0; l < row_length; ++l){
														std::cout << "{" << printout_tmp2[printout_tmp0[k1 * iq::NUM_ROWS_PER_BLOCK * exterior_block_count + j1] + l] << ", ";
														std::cout << printout_tmp5[printout_tmp1[printout_tmp0[k1 * iq::NUM_ROWS_PER_BLOCK * exterior_block_count + j1] + l]] << "} ";
													}
													std::cout << std::endl;
												}
												
											}
										}
									}
								}
							}
						}
						*/
						
						//Update velocity and strain
						match(particle_bins[rollid][solid_id])([this, &cu_dev, &solid_id, &fluid_id, &iq_solve_velocity_result, &iq_result](auto& particle_buffer_solid) {
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
					
					timer.tock(fmt::format("GPU[{}] frame {} step {} IQ solve", gpuid, cur_frame, cur_step));
				}

				/// g2p2g
				{
					auto& cu_dev = Cuda::ref_cuda_context(gpuid);
					CudaTimer timer {cu_dev.stream_compute()};

					//Resize particle buffers if we increased the size of active bins
					//This also removes all particle data of next particle buffer but does not clear it
					for(int i = 0; i < get_model_count(); ++i) {
						if(checked_bin_counts[i] > 0) {
							match(particle_bins[(rollid + 1) % BIN_COUNT][i])([this, &i, &cu_dev](auto& particle_buffer) {
								particle_buffer.resize(managed_memory_allocator, cur_num_active_bins[i]);
							});
							checked_bin_counts[i]--;
						}
					}

					timer.tick();

					//Perform G2P2G step
					for(int i = 0; i < get_model_count(); ++i) {
						//Clear the grid
						grid_blocks[1][i].reset(neighbor_block_count, cu_dev);

						//First clear the count of particles per cell for next buffer
						match(particle_bins[(rollid + 1) % BIN_COUNT][i])([this, &cu_dev, &i](auto& particle_buffer) {
							auto& prev_particle_buffer = get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[rollid][i]);
							
							prev_particle_buffer.bin_offsets = prev_particle_buffer.bin_offsets_virtual;
							particle_buffer.particle_bucket_sizes = particle_buffer.particle_bucket_sizes_virtual;
							particle_buffer.blockbuckets = particle_buffer.blockbuckets_virtual;
							particle_buffer.bin_offsets = particle_buffer.bin_offsets_virtual;
							particle_buffer.cell_particle_counts = particle_buffer.cell_particle_counts_virtual;
							particle_buffer.cellbuckets = particle_buffer.cellbuckets_virtual;
							managed_memory.acquire<MemoryType::DEVICE>(
								  prev_particle_buffer.acquire()
								, particle_buffer.acquire()
								, grid_blocks[0][i].acquire()
								, grid_blocks[1][i].acquire()
								, reinterpret_cast<void**>(&prev_particle_buffer.bin_offsets)
								, reinterpret_cast<void**>(&particle_buffer.particle_bucket_sizes)
								, reinterpret_cast<void**>(&particle_buffer.blockbuckets)
								, reinterpret_cast<void**>(&particle_buffer.bin_offsets)
								, reinterpret_cast<void**>(&particle_buffer.cell_particle_counts)
								, reinterpret_cast<void**>(&particle_buffer.cellbuckets)
							);
							
							check_cuda_errors(cudaMemsetAsync(particle_buffer.cell_particle_counts, 0, sizeof(int) * exterior_block_count * config::G_BLOCKVOLUME, cu_dev.stream_compute()));
						});
						
						cu_dev.syncStream<streamIdx::COMPUTE>();
						
						//Handle collision with shell
						for(int j = 0; j < triangle_shells[rollid][i].size(); ++j) {
							triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes = triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes_virtual;
							triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.blockbuckets = triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.blockbuckets_virtual;
							managed_memory.acquire<MemoryType::DEVICE>(
								  triangle_meshes[j].acquire()
								, triangle_shells[rollid][i][j].acquire()
								, reinterpret_cast<void**>(&triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes)
								, reinterpret_cast<void**>(&triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.blockbuckets)
							);
							
							match(particle_bins[rollid][i])([this, &cu_dev, &i, &j](const auto& particle_buffer) {
								//partition_block_count; G_PARTICLE_BATCH_CAPACITY
								cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, particle_shell_collision, dt, particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), triangle_meshes[j], triangle_shells[rollid][i][j], triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer, partitions[(rollid + 1) % BIN_COUNT], partitions[rollid], grid_blocks[0][i]);
							});
							
							managed_memory.release(
								  triangle_meshes[j].release()
								, triangle_shells[rollid][i][j].release()
								, triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes_virtual
								, triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.blockbuckets_virtual
							);
							
							cu_dev.syncStream<streamIdx::COMPUTE>();
						}

						//Perform g2p2g
						match(particle_bins[rollid][i])([this, &cu_dev, &i](const auto& particle_buffer) {
							//partition_block_count; G_PARTICLE_BATCH_CAPACITY
							cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, g2p2g, dt, next_dt, particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), partitions[(rollid + 1) % BIN_COUNT], partitions[rollid], grid_blocks[0][i], grid_blocks[1][i]);
						});
						
						cu_dev.syncStream<streamIdx::COMPUTE>();
						
						//Clear triangle_shell_grid_buffer
						//FIXME: Outcommented to save memory
						//triangle_shell_grid_buffer[(rollid + 1) % BIN_COUNT][i].reset(neighbor_block_count, cu_dev);
						
						//grid => shell: mom(t-1) => mom(t); 0 => mass(t) shell => grid: mass(t-1) => mass(t)
						for(int j = 0; j < triangle_shells[rollid][i].size(); ++j) {
							triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes = triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes_virtual;
							triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.blockbuckets = triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.blockbuckets_virtual;
							managed_memory.acquire<MemoryType::DEVICE>(
								triangle_shells[rollid][i][j].acquire()
								, reinterpret_cast<void**>(&triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes)
								, reinterpret_cast<void**>(&triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.blockbuckets)
							);
							
							//TODO: Clear triangle shell?

							//Perform g2p
							match(particle_bins[rollid][i])([this, &cu_dev, &i, &j](const auto& particle_buffer) {
								
								//partition_block_count; G_PARTICLE_BATCH_CAPACITY
								cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, grid_to_shell, dt, next_dt, particle_buffer, triangle_shells[rollid][i][j], triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer, partitions[rollid], grid_blocks[0][i]);
							});
							
							managed_memory.release(
								  triangle_shells[rollid][i][j].release()
								, triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes_virtual
								, triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.blockbuckets_virtual
							);
							
							cu_dev.syncStream<streamIdx::COMPUTE>();
						}
						
						match(particle_bins[(rollid + 1) % BIN_COUNT][i])([this, &cu_dev, &i](auto& particle_buffer) {
							auto& prev_particle_buffer = get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[rollid][i]);
							
							managed_memory.release(
								  prev_particle_buffer.release()
								, particle_buffer.release()
								, grid_blocks[0][i].release()
								, grid_blocks[1][i].release()
								, prev_particle_buffer.bin_offsets_virtual
								, particle_buffer.particle_bucket_sizes_virtual
								, particle_buffer.blockbuckets_virtual
								, particle_buffer.bin_offsets_virtual
								, particle_buffer.cell_particle_counts_virtual
								, particle_buffer.cellbuckets_virtual
							);
						});
						
						cu_dev.syncStream<streamIdx::COMPUTE>();
					}
					
					cu_dev.syncStream<streamIdx::COMPUTE>();

					timer.tock(fmt::format("GPU[{}] frame {} step {} g2p2g", gpuid, cur_frame, cur_step));

					//Resize partition if we increased the size of active blocks
					//This also deletes current particle buffer meta data.
					if(checked_counts[0] > 0) {
						managed_memory.release(
							  partitions[rollid].Instance<block_partition_>::count_virtual
							, partitions[rollid].active_keys_virtual
							, partitions[rollid].index_table_virtual
							, partitions[(rollid + 1) % BIN_COUNT].Instance<block_partition_>::count_virtual
							, partitions[(rollid + 1) % BIN_COUNT].active_keys_virtual
							, partitions[(rollid + 1) % BIN_COUNT].index_table_virtual
						);
						
						partitions[(rollid + 1) % BIN_COUNT].resize_partition(managed_memory_allocator, cur_num_active_blocks);
						for(int i = 0; i < get_model_count(); ++i) {
							match(particle_bins[rollid][i])([this, &cu_dev](auto& particle_buffer) {
								particle_buffer.reserve_buckets(managed_memory_allocator, cur_num_active_blocks);
							});
							
							for(int j = 0; j < triangle_shells[rollid][i].size(); ++j) {
								triangle_shells[rollid][i][j].particle_buffer.reserve_buckets(managed_memory_allocator, cur_num_active_blocks);
							}
						}
						checked_counts[0]--;
						
						partitions[rollid].Instance<block_partition_>::count = partitions[rollid].Instance<block_partition_>::count_virtual;
						partitions[rollid].active_keys = partitions[rollid].active_keys_virtual;
						partitions[rollid].index_table = partitions[rollid].index_table_virtual;
						partitions[(rollid + 1) % BIN_COUNT].Instance<block_partition_>::count = partitions[(rollid + 1) % BIN_COUNT].Instance<block_partition_>::count_virtual;
						partitions[(rollid + 1) % BIN_COUNT].active_keys = partitions[(rollid + 1) % BIN_COUNT].active_keys_virtual;
						partitions[(rollid + 1) % BIN_COUNT].index_table = partitions[(rollid + 1) % BIN_COUNT].index_table_virtual;
						managed_memory.acquire<MemoryType::DEVICE>(
							  reinterpret_cast<void**>(&partitions[rollid].Instance<block_partition_>::count)
							, reinterpret_cast<void**>(&partitions[rollid].active_keys)
							, reinterpret_cast<void**>(&partitions[rollid].index_table)
							, reinterpret_cast<void**>(&partitions[(rollid + 1) % BIN_COUNT].Instance<block_partition_>::count)
							, reinterpret_cast<void**>(&partitions[(rollid + 1) % BIN_COUNT].active_keys)
							, reinterpret_cast<void**>(&partitions[(rollid + 1) % BIN_COUNT].index_table)
						);
					}
				}
				
				//TODO: Also divide this into two parts
				
				//TODO: Inner shell part (part of rigid body motion directly transfered; gravity;) (Navier stokes!) (We have to keep track of current state (for outer shell the grid does so))
				//mesh => shell: vel(t-1) => vel(t);
				//TODO: mass(t-1) => mass(t)
				{
					auto& cu_dev = Cuda::ref_cuda_context(gpuid);
					CudaTimer timer {cu_dev.stream_compute()};
					
					timer.tick();
					
					for(int i = 0; i < get_model_count(); ++i) {
						for(int j = 0; j < triangle_meshes.size(); ++j) {
							managed_memory.acquire<MemoryType::DEVICE>(
								  triangle_meshes[j].acquire()
								, triangle_shells[rollid][i][j].acquire()
							);
							
							cu_dev.compute_launch({(triangle_mesh_vertex_counts[j] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, update_triangle_shell_inner, triangle_meshes[j], triangle_shells[rollid][i][j], triangle_mesh_vertex_counts[j], dt);
						
							managed_memory.release(
								  triangle_meshes[j].release()
								, triangle_shells[rollid][i][j].release()
							);
							
							cu_dev.syncStream<streamIdx::COMPUTE>();
						}
					}
					cu_dev.syncStream<streamIdx::COMPUTE>();
					
					timer.tock(fmt::format("GPU[{}] frame {} step {} triangle_shell_inner_update", gpuid, cur_frame, cur_step));
				}
				
				//TODO: Simulate subdomain (=> properties needed for further calculation and for next g2p2g step); Also transfer of physical properties along surface and accross domain.
				{
					auto& cu_dev = Cuda::ref_cuda_context(gpuid);
					CudaTimer timer {cu_dev.stream_compute()};
					
					timer.tick();
					
					for(int i = 0; i < get_model_count(); ++i) {
						for(int j = 0; j < triangle_meshes.size(); ++j) {
							managed_memory.acquire<MemoryType::DEVICE>(
								  triangle_meshes[j].acquire()
								, triangle_shells[(rollid + 1) % BIN_COUNT][i][j].acquire()
							);
							
							//First clear next triangle shell
							cu_dev.compute_launch({(triangle_mesh_vertex_counts[j] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, clear_triangle_shell, triangle_shells[(rollid + 1) % BIN_COUNT][i][j], triangle_mesh_vertex_counts[j]);
						
							//Then update domain
							//cu_dev.compute_launch({(triangle_mesh_face_counts[j] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, update_triangle_shell_subdomain, triangle_meshes[j], triangle_shells[rollid][i][j], triangle_shells[(rollid + 1) % BIN_COUNT][i][j], triangle_mesh_face_counts[j]);
							cu_dev.compute_launch({(triangle_mesh_vertex_counts[j] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, update_triangle_shell_subdomain, triangle_shells[rollid][i][j], triangle_shells[(rollid + 1) % BIN_COUNT][i][j], triangle_mesh_vertex_counts[j]);
						
							managed_memory.release(
								  triangle_meshes[j].release()
								, triangle_shells[(rollid + 1) % BIN_COUNT][i][j].release()
							);
							
							cu_dev.syncStream<streamIdx::COMPUTE>();
						}
					}
					cu_dev.syncStream<streamIdx::COMPUTE>();
					
					timer.tock(fmt::format("GPU[{}] frame {} step {} triangle_shell_step", gpuid, cur_frame, cur_step));
				}
				
				//TODO: Force Feedback to triangle mesh
				
				//TODO: Max velocity and maybe different CFL
				
				//Rigid body motion
				//TODO: Maybe need to split up this for interaction with other things
				{
					auto& cu_dev = Cuda::ref_cuda_context(gpuid);
					CudaTimer timer {cu_dev.stream_compute()};
					
					timer.tick();
					
					for(int i = 0; i < triangle_meshes.size(); ++i) {
						triangle_meshes[i].rigid_body_update(cur_time, dt);
						
						managed_memory.acquire<MemoryType::DEVICE>(
							  triangle_meshes[i].acquire()
						);
						
						cu_dev.compute_launch({(triangle_mesh_vertex_counts[i] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, update_triangle_mesh, triangle_meshes[i], triangle_mesh_vertex_counts[i], triangle_meshes[i].center.data_arr(), triangle_meshes[i].linear_velocity.data_arr(), triangle_meshes[i].rotation.data_arr(), triangle_meshes[i].angular_velocity.data_arr());
					
						managed_memory.release(
							  triangle_meshes[i].release()
						);
						
						cu_dev.syncStream<streamIdx::COMPUTE>();
					}
					cu_dev.syncStream<streamIdx::COMPUTE>();
					
					timer.tock(fmt::format("GPU[{}] frame {} step {} rigid_body_update", gpuid, cur_frame, cur_step));
				
				}
				
				//TODO: Volume redistribute on shell, volume exchange between shell and domain, volume exchange between triangle mesh and domain (maybe already done in simulation step and inner shell update step?!)
				{
					auto& cu_dev = Cuda::ref_cuda_context(gpuid);
					CudaTimer timer {cu_dev.stream_compute()};
					
					timer.tick();
					
					for(int i = 0; i < get_model_count(); ++i) {
						for(int j = 0; j < triangle_shells[(rollid + 1) % BIN_COUNT][i].size(); ++j) {
							
							
							match(particle_bins[rollid][i])([this, &cu_dev, &i, &j](auto& particle_buffer) {
								auto& next_particle_buffer = get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]);
								
								next_particle_buffer.bin_offsets = next_particle_buffer.bin_offsets_virtual;
								next_particle_buffer.particle_bucket_sizes = next_particle_buffer.particle_bucket_sizes_virtual;
								next_particle_buffer.cell_particle_counts = next_particle_buffer.cell_particle_counts_virtual;
								next_particle_buffer.cellbuckets = next_particle_buffer.cellbuckets_virtual;
								triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes = triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes_virtual;
								triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.blockbuckets = triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.blockbuckets_virtual;
								managed_memory.acquire<MemoryType::DEVICE>(
									  particle_buffer.acquire()
									, next_particle_buffer.acquire()
									, grid_blocks[1][i].acquire()
									, triangle_meshes[i].acquire()
									, triangle_shells[rollid][i][j].acquire()
									, triangle_shells[(rollid + 1) % BIN_COUNT][i][j].acquire()
									, reinterpret_cast<void**>(&next_particle_buffer.bin_offsets)
									, reinterpret_cast<void**>(&next_particle_buffer.particle_bucket_sizes)
									, reinterpret_cast<void**>(&next_particle_buffer.cell_particle_counts)
									, reinterpret_cast<void**>(&next_particle_buffer.cellbuckets)
									, reinterpret_cast<void**>(&triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes)
									, reinterpret_cast<void**>(&triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.blockbuckets)
								);
								
								//partition_block_count; G_PARTICLE_BATCH_CAPACITY
								cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, shell_to_grid, dt, next_dt, partition_block_count, particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), triangle_meshes[i], triangle_shells[rollid][i][j], triangle_shells[(rollid + 1) % BIN_COUNT][i][j], triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer, partitions[rollid], grid_blocks[1][i]);
							
								managed_memory.release(
									 particle_buffer.release()
									, next_particle_buffer.release()
									, grid_blocks[1][i].release()
									, triangle_meshes[i].release()
									, triangle_shells[rollid][i][j].release()
									, triangle_shells[(rollid + 1) % BIN_COUNT][i][j].release()
									, next_particle_buffer.bin_offsets_virtual
									, next_particle_buffer.particle_bucket_sizes_virtual
									, next_particle_buffer.cell_particle_counts_virtual
									, next_particle_buffer.cellbuckets_virtual
									, triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes_virtual
									, triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.blockbuckets_virtual
								);
							});
							
							
							
							cu_dev.syncStream<streamIdx::COMPUTE>();
						}
					}
					cu_dev.syncStream<streamIdx::COMPUTE>();
					
					timer.tock(fmt::format("GPU[{}] frame {} step {} p2g", gpuid, cur_frame, cur_step));
				}

				/// update partition
				{
					auto& cu_dev = Cuda::ref_cuda_context(gpuid);
					CudaTimer timer {cu_dev.stream_compute()};

					timer.tick();

					//Copy cell buckets from partition to next particle buffer
					for(int i = 0; i < get_model_count(); ++i) {
						match(particle_bins[(rollid + 1) % BIN_COUNT][i])([this, &cu_dev](auto& particle_buffer) {
							particle_buffer.blockbuckets = particle_buffer.blockbuckets_virtual;
							particle_buffer.particle_bucket_sizes = particle_buffer.particle_bucket_sizes_virtual;
							particle_buffer.cell_particle_counts = particle_buffer.cell_particle_counts_virtual;
							particle_buffer.cellbuckets = particle_buffer.cellbuckets_virtual;
							managed_memory.acquire<MemoryType::DEVICE>(
								  reinterpret_cast<void**>(&particle_buffer.blockbuckets)
								, reinterpret_cast<void**>(&particle_buffer.particle_bucket_sizes)
								, reinterpret_cast<void**>(&particle_buffer.cell_particle_counts)
								, reinterpret_cast<void**>(&particle_buffer.cellbuckets)
							);
							
							//First init sizes with 0
							check_cuda_errors(cudaMemsetAsync(particle_buffer.particle_bucket_sizes, 0, sizeof(int) * (exterior_block_count + 1), cu_dev.stream_compute()));

							//exterior_block_count; G_BLOCKVOLUME
							cu_dev.compute_launch({exterior_block_count, config::G_BLOCKVOLUME}, cell_bucket_to_block, particle_buffer.cell_particle_counts, particle_buffer.cellbuckets, particle_buffer.particle_bucket_sizes, particle_buffer.blockbuckets);
							// partitions[rollid].buildParticleBuckets(cu_dev, exterior_block_count);
						
							managed_memory.release(
								  particle_buffer.blockbuckets_virtual
								, particle_buffer.particle_bucket_sizes_virtual
								, particle_buffer.cell_particle_counts_virtual
								, particle_buffer.cellbuckets_virtual
							);
						});
						
						cu_dev.syncStream<streamIdx::COMPUTE>();
						
						//Store triangle shell outer vertices in buckets
						for(int j = 0; j < triangle_shells[(rollid + 1) % BIN_COUNT][i].size(); ++j) {
							triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes = triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes_virtual;
							triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.face_bucket_sizes = triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.face_bucket_sizes_virtual;
							triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.blockbuckets = triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.blockbuckets_virtual;
							triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.face_blockbuckets = triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.face_blockbuckets_virtual;
							managed_memory.acquire<MemoryType::DEVICE>(
								  triangle_shells[(rollid + 1) % BIN_COUNT][i][j].acquire()
								, reinterpret_cast<void**>(&triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes)
								, reinterpret_cast<void**>(&triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.face_bucket_sizes)
								, reinterpret_cast<void**>(&triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.blockbuckets)
								, reinterpret_cast<void**>(&triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.face_blockbuckets)
							);
							
							//First init sizes with 0
							check_cuda_errors(cudaMemsetAsync(triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes, 0, sizeof(int) * (exterior_block_count + 1), cu_dev.stream_compute()));
							check_cuda_errors(cudaMemsetAsync(triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.face_bucket_sizes, 0, sizeof(int) * (exterior_block_count + 1), cu_dev.stream_compute()));
							
							//Store triangle shell outer vertices in buckets
							cu_dev.compute_launch({(triangle_mesh_vertex_counts[j] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, store_triangle_shell_vertices_in_bucket, triangle_mesh_vertex_counts[j], triangle_shells[(rollid + 1) % BIN_COUNT][i][j], triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer, partitions[rollid], grid_blocks[0][i]);
							
							//Store faces in buckets
							cu_dev.compute_launch({(triangle_mesh_face_counts[j] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, store_triangle_shell_faces_in_bucket, triangle_mesh_face_counts[j], triangle_meshes[j], triangle_shells[(rollid + 1) % BIN_COUNT][i][j], triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer, partitions[rollid], grid_blocks[0][i]);
							
							managed_memory.release(
								  triangle_shells[(rollid + 1) % BIN_COUNT][i][j].release()
								, triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes_virtual
								, triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.face_bucket_sizes_virtual
								, triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.blockbuckets_virtual
								, triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.face_blockbuckets_virtual
							);
							
							cu_dev.syncStream<streamIdx::COMPUTE>();
						}
					}

					int* active_block_marks = tmps.active_block_marks;
					int* destinations		= tmps.destinations;
					int* sources			= tmps.sources;
					managed_memory.acquire<MemoryType::DEVICE>(
						  reinterpret_cast<void**>(&active_block_marks)
						, reinterpret_cast<void**>(&destinations)
						, reinterpret_cast<void**>(&sources)
					);
					
					//Clear marks
					check_cuda_errors(cudaMemsetAsync(active_block_marks, 0, sizeof(int) * neighbor_block_count, cu_dev.stream_compute()));
					
					cu_dev.syncStream<streamIdx::COMPUTE>();

					//Mark cells that have mass bigger 0.0
					for(int i = 0; i < get_model_count(); ++i) {
						managed_memory.acquire<MemoryType::DEVICE>(
							  grid_blocks[1][i].acquire()
						);
						
						//floor(neighbor_block_count * config::G_BLOCKVOLUME/G_PARTICLE_BATCH_CAPACITY); G_PARTICLE_BATCH_CAPACITY
						cu_dev.compute_launch({(neighbor_block_count * config::G_BLOCKVOLUME + config::G_PARTICLE_BATCH_CAPACITY - 1) / config::G_PARTICLE_BATCH_CAPACITY, config::G_PARTICLE_BATCH_CAPACITY}, mark_active_grid_blocks, static_cast<uint32_t>(neighbor_block_count), grid_blocks[1][i], active_block_marks);
					
						managed_memory.release(
							  grid_blocks[1][i].release()
						);
						
						cu_dev.syncStream<streamIdx::COMPUTE>();
					}
					
					//Clear marks
					check_cuda_errors(cudaMemsetAsync(sources, 0, sizeof(int) * (exterior_block_count + 1), cu_dev.stream_compute()));
					
					cu_dev.syncStream<streamIdx::COMPUTE>();

					//Mark particle buckets that have at least one particle
					for(int i = 0; i < get_model_count(); ++i) {
						match(particle_bins[(rollid + 1) % BIN_COUNT][i])([this, &cu_dev, &sources](auto& particle_buffer) {
							particle_buffer.particle_bucket_sizes = particle_buffer.particle_bucket_sizes_virtual;
							managed_memory.acquire<MemoryType::DEVICE>(
								  reinterpret_cast<void**>(&particle_buffer.particle_bucket_sizes)
							);
							
							//floor((exterior_block_count + 1)/G_PARTICLE_BATCH_CAPACITY); G_PARTICLE_BATCH_CAPACITY
							cu_dev.compute_launch({exterior_block_count / config::G_PARTICLE_BATCH_CAPACITY + 1, config::G_PARTICLE_BATCH_CAPACITY}, mark_active_particle_blocks, exterior_block_count + 1, particle_buffer.particle_bucket_sizes, sources);
							
							managed_memory.release(
								  particle_buffer.particle_bucket_sizes_virtual
							);
						});
						
						cu_dev.syncStream<streamIdx::COMPUTE>();
						
						//Mark for triangle shell vertices
						for(int j = 0; j < triangle_shells[(rollid + 1) % BIN_COUNT][i].size(); ++j) {
							triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes = triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes_virtual;
							managed_memory.acquire<MemoryType::DEVICE>(
								  reinterpret_cast<void**>(&triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes)
							);
							
							//floor((exterior_block_count + 1)/G_PARTICLE_BATCH_CAPACITY); G_PARTICLE_BATCH_CAPACITY
							cu_dev.compute_launch({exterior_block_count / config::G_PARTICLE_BATCH_CAPACITY + 1, config::G_PARTICLE_BATCH_CAPACITY}, mark_active_particle_blocks, exterior_block_count + 1, triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes, sources);
						
							managed_memory.release(
							  triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes_virtual
							);
							
							cu_dev.syncStream<streamIdx::COMPUTE>();
						}
					}

					//Sum up number of active buckets and calculate offsets (empty buckets are collapsed
					exclusive_scan(exterior_block_count + 1, sources, destinations, cu_dev);

					/// building new partition

					//Store new bucket count
					check_cuda_errors(cudaMemcpyAsync(partitions[(rollid + 1) % BIN_COUNT].count, destinations + exterior_block_count, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
					check_cuda_errors(cudaMemcpyAsync(&partition_block_count, destinations + exterior_block_count, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
					
					//Calculate indices of block by position
					//floor(exterior_block_count/config::DEFAULT_CUDA_BLOCK_SIZE); config::DEFAULT_CUDA_BLOCK_SIZE
					cu_dev.compute_launch({(exterior_block_count + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, exclusive_scan_inverse, exterior_block_count, static_cast<const int*>(destinations), sources);
					
					managed_memory.release(tmps.destinations);
					
					cu_dev.syncStream<streamIdx::COMPUTE>();

					//Reset partitions
					partitions[(rollid + 1) % BIN_COUNT].reset_table(cu_dev.stream_compute());
					cu_dev.syncStream<streamIdx::COMPUTE>();

					//Check size
					if(partition_block_count > config::G_MAX_ACTIVE_BLOCK) {
						std::cerr << "Too much active blocks: " << partition_block_count << std::endl;
						std::abort();
					}

					//Reinsert buckets
					//floor(partition_block_count/G_PARTICLE_BATCH_CAPACITY); G_PARTICLE_BATCH_CAPACITY
					cu_dev.compute_launch({(partition_block_count + config::G_PARTICLE_BATCH_CAPACITY - 1) / config::G_PARTICLE_BATCH_CAPACITY, config::G_PARTICLE_BATCH_CAPACITY}, update_partition, static_cast<uint32_t>(partition_block_count), static_cast<const int*>(sources), partitions[rollid], partitions[(rollid + 1) % BIN_COUNT]);

					cu_dev.syncStream<streamIdx::COMPUTE>();

					//Copy block buckets and sizes from next particle buffer to current particle buffer
					for(int i = 0; i < get_model_count(); ++i) {
						match(particle_bins[(rollid + 1) % BIN_COUNT][i])([this, &cu_dev, &sources, &i](auto& particle_buffer) {
							auto& next_particle_buffer = get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[rollid][i]);
							
							particle_buffer.particle_bucket_sizes = particle_buffer.particle_bucket_sizes_virtual;
							particle_buffer.blockbuckets = particle_buffer.blockbuckets_virtual;
							next_particle_buffer.particle_bucket_sizes = next_particle_buffer.particle_bucket_sizes_virtual;
							next_particle_buffer.blockbuckets = next_particle_buffer.blockbuckets_virtual;
							managed_memory.acquire<MemoryType::DEVICE>(
								  reinterpret_cast<void**>(&particle_buffer.particle_bucket_sizes)
								, reinterpret_cast<void**>(&particle_buffer.blockbuckets)
								, reinterpret_cast<void**>(&next_particle_buffer.particle_bucket_sizes)
								, reinterpret_cast<void**>(&next_particle_buffer.blockbuckets)
							);
							
							//partition_block_count; G_PARTICLE_BATCH_CAPACITY
							cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, update_buckets, static_cast<uint32_t>(partition_block_count), static_cast<const int*>(sources), particle_buffer, next_particle_buffer);
						
							managed_memory.release(
								  particle_buffer.particle_bucket_sizes_virtual
								, particle_buffer.blockbuckets_virtual
								, next_particle_buffer.particle_bucket_sizes_virtual
								, next_particle_buffer.blockbuckets_virtual
							);
						});
										
						cu_dev.syncStream<streamIdx::COMPUTE>();
						
						for(int j = 0; j < triangle_shells[(rollid + 1) % BIN_COUNT][i].size(); ++j) {
							triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes = triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes_virtual;
							triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.blockbuckets = triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.blockbuckets_virtual;
							triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.face_bucket_sizes = triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.face_bucket_sizes_virtual;
							triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.face_blockbuckets = triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.face_blockbuckets_virtual;
							triangle_shells[rollid][i][j].particle_buffer.particle_bucket_sizes = triangle_shells[rollid][i][j].particle_buffer.particle_bucket_sizes_virtual;
							triangle_shells[rollid][i][j].particle_buffer.blockbuckets = triangle_shells[rollid][i][j].particle_buffer.blockbuckets_virtual;
							triangle_shells[rollid][i][j].particle_buffer.face_bucket_sizes = triangle_shells[rollid][i][j].particle_buffer.face_bucket_sizes_virtual;
							triangle_shells[rollid][i][j].particle_buffer.face_blockbuckets = triangle_shells[rollid][i][j].particle_buffer.face_blockbuckets_virtual;
							managed_memory.acquire<MemoryType::DEVICE>(
								  reinterpret_cast<void**>(&triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes)
								, reinterpret_cast<void**>(&triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.blockbuckets)
								, reinterpret_cast<void**>(&triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.face_bucket_sizes)
								, reinterpret_cast<void**>(&triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.face_blockbuckets)
								, reinterpret_cast<void**>(&triangle_shells[rollid][i][j].particle_buffer.particle_bucket_sizes)
								, reinterpret_cast<void**>(&triangle_shells[rollid][i][j].particle_buffer.blockbuckets)
								, reinterpret_cast<void**>(&triangle_shells[rollid][i][j].particle_buffer.face_bucket_sizes)
								, reinterpret_cast<void**>(&triangle_shells[rollid][i][j].particle_buffer.face_blockbuckets)
							);
							
							//partition_block_count; G_PARTICLE_BATCH_CAPACITY
							cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, update_buckets_triangle_shell, static_cast<uint32_t>(partition_block_count), static_cast<const int*>(sources), triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer, triangle_shells[rollid][i][j].particle_buffer);
						
							managed_memory.release(
								  triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes_virtual
								, triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.blockbuckets_virtual
								, triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.face_bucket_sizes_virtual
								, triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.face_blockbuckets_virtual
								, triangle_shells[rollid][i][j].particle_buffer.particle_bucket_sizes_virtual
								, triangle_shells[rollid][i][j].particle_buffer.blockbuckets_virtual
								, triangle_shells[rollid][i][j].particle_buffer.face_bucket_sizes_virtual
								, triangle_shells[rollid][i][j].particle_buffer.face_blockbuckets_virtual
							);
							
							cu_dev.syncStream<streamIdx::COMPUTE>();
						}
					}
					
					managed_memory.release(tmps.sources);

					//Compute bin capacities, bin offsets and the summed bin size for current particle buffer
					int* bin_sizes = tmps.bin_sizes;
					
					managed_memory.acquire<MemoryType::DEVICE>(
						  reinterpret_cast<void**>(&bin_sizes)
					);
					
					for(int i = 0; i < get_model_count(); ++i) {
						match(particle_bins[rollid][i])([this, &cu_dev, &bin_sizes, &i](auto& particle_buffer) {
							particle_buffer.particle_bucket_sizes = particle_buffer.particle_bucket_sizes_virtual;
							particle_buffer.bin_offsets = particle_buffer.bin_offsets_virtual;
							managed_memory.acquire<MemoryType::DEVICE>(
								  reinterpret_cast<void**>(&particle_buffer.particle_bucket_sizes)
								, reinterpret_cast<void**>(&particle_buffer.bin_offsets)
							);
							
							//floor((exterior_block_count + 1)/G_PARTICLE_BATCH_CAPACITY); G_PARTICLE_BATCH_CAPACITY
							cu_dev.compute_launch({partition_block_count / config::G_PARTICLE_BATCH_CAPACITY + 1, config::G_PARTICLE_BATCH_CAPACITY}, compute_bin_capacity, partition_block_count + 1, static_cast<const int*>(particle_buffer.particle_bucket_sizes), bin_sizes);

							cu_dev.syncStream<streamIdx::COMPUTE>();

							//Ensure we have enough space for new generated particles
							for(int j = 0; j < triangle_shells[(rollid + 1) % BIN_COUNT][i].size(); ++j) {
								triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes = triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes_virtual;
								managed_memory.acquire<MemoryType::DEVICE>(
									  reinterpret_cast<void**>(&triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes)
								);
								
								//floor((exterior_block_count + 1)/G_PARTICLE_BATCH_CAPACITY); G_PARTICLE_BATCH_CAPACITY
								cu_dev.compute_launch({partition_block_count / config::G_PARTICLE_BATCH_CAPACITY + 1, config::G_PARTICLE_BATCH_CAPACITY}, compute_bin_capacity_shell, partition_block_count + 1, static_cast<const int*>(triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes), bin_sizes);
							
								managed_memory.release(
									  triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes_virtual
								);
							}

							//Stores aggregated bin sizes in particle_buffer
							exclusive_scan(partition_block_count + 1, bin_sizes, particle_buffer.bin_offsets, cu_dev);

							//Stores last aggregated size == whole size in bincount
							check_cuda_errors(cudaMemcpyAsync(&bincount[i], particle_buffer.bin_offsets + partition_block_count, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
							
							managed_memory.release(
								  particle_buffer.particle_bucket_sizes_virtual
								, particle_buffer.bin_offsets_virtual
							);
						});
						
						cu_dev.syncStream<streamIdx::COMPUTE>();
					}
					
					managed_memory.release(
						  tmps.bin_sizes
					);
					
					cu_dev.syncStream<streamIdx::COMPUTE>();

					timer.tock(fmt::format("GPU[{}] frame {} step {} update_partition", gpuid, cur_frame, cur_step));

					timer.tick();

					//Activate blocks near active blocks
					//floor(partition_block_count/G_PARTICLE_BATCH_CAPACITY); G_PARTICLE_BATCH_CAPACITY
					cu_dev.compute_launch({(partition_block_count + config::G_PARTICLE_BATCH_CAPACITY - 1) / config::G_PARTICLE_BATCH_CAPACITY, config::G_PARTICLE_BATCH_CAPACITY}, register_neighbor_blocks, static_cast<uint32_t>(partition_block_count), partitions[(rollid + 1) % BIN_COUNT]);

					//Retrieve total count
					auto prev_neighbor_block_count = neighbor_block_count;
					check_cuda_errors(cudaMemcpyAsync(&neighbor_block_count, partitions[(rollid + 1) % BIN_COUNT].count, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
					cu_dev.syncStream<streamIdx::COMPUTE>();

					//Check size
					if(neighbor_block_count > config::G_MAX_ACTIVE_BLOCK) {
						std::cerr << "Too much neighbour blocks: " << neighbor_block_count << std::endl;
						std::abort();
					}
					
					//Activate blocks before active blocks
					//floor(partition_block_count/G_PARTICLE_BATCH_CAPACITY); G_PARTICLE_BATCH_CAPACITY
					cu_dev.compute_launch({(partition_block_count + config::G_PARTICLE_BATCH_CAPACITY - 1) / config::G_PARTICLE_BATCH_CAPACITY, config::G_PARTICLE_BATCH_CAPACITY}, register_prev_neighbor_blocks, static_cast<uint32_t>(partition_block_count), partitions[(rollid + 1) % BIN_COUNT]);

					//Retrieve total count
					check_cuda_errors(cudaMemcpyAsync(&total_neighbor_block_count, partitions[(rollid + 1) % BIN_COUNT].count, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
					cu_dev.syncStream<streamIdx::COMPUTE>();

					//Check size
					if(total_neighbor_block_count > config::G_MAX_ACTIVE_BLOCK) {
						std::cerr << "Too much total neighbour blocks: " << total_neighbor_block_count << std::endl;
						std::abort();
					}

					timer.tock(fmt::format("GPU[{}] frame {} step {} build_partition_for_grid", gpuid, cur_frame, cur_step));

					//Resize grid if necessary
					if(checked_counts[0] > 0) {
						//alpha_shapes_grid_buffer.resize(managed_memory_allocator, cur_num_active_blocks);
						for(int i = 0; i < get_model_count(); ++i) {
							grid_blocks[0][i].resize(managed_memory_allocator, cur_num_active_blocks);
							//FIXME: Outcommented to save memory
							//triangle_shell_grid_buffer[rollid][i].resize(managed_memory_allocator, cur_num_active_blocks);
						}
					}

					timer.tick();

					//Clear the grid
					for(int i = 0; i < get_model_count(); ++i) {
						grid_blocks[0][i].reset(exterior_block_count, cu_dev);
						//FIXME: Outcommented to save memory
						//triangle_shell_grid_buffer[rollid][i].reset(exterior_block_count, cu_dev);
					}

					//Copy values from old grid for active blocks
					for(int i = 0; i < get_model_count(); ++i) {
						managed_memory.acquire<MemoryType::DEVICE>(
							  grid_blocks[1][i].acquire()
							, grid_blocks[0][i].acquire()
							//FIXME: Outcommented to save memory
							//, triangle_shell_grid_buffer[(rollid + 1) % BIN_COUNT][i].acquire()
							//, triangle_shell_grid_buffer[rollid][i].acquire()
						);
						
						//prev_neighbor_block_count; G_BLOCKVOLUME
						cu_dev.compute_launch({prev_neighbor_block_count, config::G_BLOCKVOLUME}, copy_selected_grid_blocks, static_cast<const ivec3*>(partitions[rollid].active_keys), partitions[(rollid + 1) % BIN_COUNT], static_cast<const int*>(active_block_marks), grid_blocks[1][i], grid_blocks[0][i]);

						//FIXME: Outcommented to save memory
						//cu_dev.compute_launch({prev_neighbor_block_count, config::G_BLOCKVOLUME}, copy_selected_grid_blocks_triangle_shell, static_cast<const ivec3*>(partitions[rollid].active_keys), partitions[(rollid + 1) % BIN_COUNT], static_cast<const int*>(active_block_marks), triangle_shell_grid_buffer[(rollid + 1) % BIN_COUNT][i], triangle_shell_grid_buffer[rollid][i]);
					
						managed_memory.release(
							  grid_blocks[1][i].release()
							, grid_blocks[0][i].release()
							//FIXME: Outcommented to save memory
							//, triangle_shell_grid_buffer[(rollid + 1) % BIN_COUNT][i].release()
							//, triangle_shell_grid_buffer[rollid][i].release()
						);
						
						cu_dev.syncStream<streamIdx::COMPUTE>();
					}
					
					managed_memory.release(tmps.active_block_marks);
					
					cu_dev.syncStream<streamIdx::COMPUTE>();

					timer.tock(fmt::format("GPU[{}] frame {} step {} copy_grid_blocks", gpuid, cur_frame, cur_step));

					//Resize grid if necessary
					if(checked_counts[0] > 0) {
						for(int i = 0; i < get_model_count(); ++i) {
							grid_blocks[1][i].resize(managed_memory_allocator, cur_num_active_blocks);
							//FIXME: Outcommented to save memory
							//triangle_shell_grid_buffer[(rollid + 1) % BIN_COUNT][i].resize(managed_memory_allocator, cur_num_active_blocks);
						}
						tmps.resize(cur_num_active_blocks);
						//temporary_grid_buffer.resize(managed_memory_allocator, cur_num_active_blocks);
					}
				}

				{
					auto& cu_dev = Cuda::ref_cuda_context(gpuid);
					CudaTimer timer {cu_dev.stream_compute()};

					timer.tick();

					//Activate blocks near active blocks, including those before that block
					//floor(total_neighbor_block_count/G_PARTICLE_BATCH_CAPACITY); G_PARTICLE_BATCH_CAPACITY
					cu_dev.compute_launch({(total_neighbor_block_count + config::G_PARTICLE_BATCH_CAPACITY - 1) / config::G_PARTICLE_BATCH_CAPACITY, config::G_PARTICLE_BATCH_CAPACITY}, register_exterior_blocks, static_cast<uint32_t>(total_neighbor_block_count), partitions[(rollid + 1) % BIN_COUNT]);

					//Retrieve total count
					check_cuda_errors(cudaMemcpyAsync(&exterior_block_count, partitions[(rollid + 1) % BIN_COUNT].count, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
					cu_dev.syncStream<streamIdx::COMPUTE>();

					//Check size
					if(exterior_block_count > config::G_MAX_ACTIVE_BLOCK) {
						std::cerr << "Too much exterior blocks: " << exterior_block_count << std::endl;
						std::abort();
					}

					fmt::print(fmt::emphasis::bold | fg(fmt::color::yellow), "block count on device {}: {}, {}, {} [{}]\n", gpuid, partition_block_count, neighbor_block_count, exterior_block_count, cur_num_active_blocks);
					for(int i = 0; i < get_model_count(); ++i) {
						fmt::print(fmt::emphasis::bold | fg(fmt::color::yellow), "bin count on device {}: model {}: {} [{}]\n", gpuid, i, bincount[i], cur_num_active_bins[i]);
					}
					timer.tock(fmt::format("GPU[{}] frame {} step {} build_partition_for_particles", gpuid, cur_frame, cur_step));
				}
				
				{
					auto& cu_dev = Cuda::ref_cuda_context(gpuid);
					cu_dev.reset_mem();
				}
				
				rollid = static_cast<char>((rollid + 1) % BIN_COUNT);
				dt	   = next_dt;
			}
			IO::flush();
			output_model();
			
			fmt::print(
				fmt::emphasis::bold | fg(fmt::color::red),
				"-----------------------------------------------------------"
				"-----\n"
			);
		}
		managed_memory.release(
			  partitions[rollid].Instance<block_partition_>::count_virtual
			, partitions[rollid].active_keys_virtual
			, partitions[rollid].index_table_virtual
			, partitions[(rollid + 1) % BIN_COUNT].Instance<block_partition_>::count_virtual
			, partitions[(rollid + 1) % BIN_COUNT].active_keys_virtual
			, partitions[(rollid + 1) % BIN_COUNT].index_table_virtual
		);
	outer_loop_end:
		(void) nullptr;//We need a statement to have a valid jump label
	}
	//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

	void output_model() {
		auto& cu_dev = Cuda::ref_cuda_context(gpuid);
		CudaTimer timer {cu_dev.stream_compute()};

		timer.tick();

		for(int i = 0; i < get_model_count(); ++i) {
			int particle_count	  = 0;
			int* d_particle_count = static_cast<int*>(cu_dev.borrow(sizeof(int)));
			
			//Init particle count with 0
			check_cuda_errors(cudaMemsetAsync(d_particle_count, 0, sizeof(int), cu_dev.stream_compute()));
			
			cu_dev.syncStream<streamIdx::COMPUTE>();

			//Retrieve particle count
			unsigned int* particle_id_mapping_buffer = tmps.particle_id_mapping_buffer;
			
			void* surface_vertex_count_ptr = surface_vertex_count;
			void* surface_triangle_count_ptr = surface_triangle_count;
			
			match(particle_bins[(rollid + 1) % BIN_COUNT][i])([this, &cu_dev, &particle_count, &i, &particle_id_mapping_buffer, &surface_vertex_count_ptr, &surface_triangle_count_ptr](auto& particle_buffer) {
				auto& prev_particle_buffer = get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[rollid][i]);
				
				prev_particle_buffer.cellbuckets = prev_particle_buffer.cellbuckets_virtual;
				prev_particle_buffer.bin_offsets = prev_particle_buffer.bin_offsets_virtual;
				prev_particle_buffer.cell_particle_counts = prev_particle_buffer.cell_particle_counts_virtual;
				particle_buffer.particle_bucket_sizes = particle_buffer.particle_bucket_sizes_virtual;
				particle_buffer.blockbuckets = particle_buffer.blockbuckets_virtual;
				managed_memory.acquire<MemoryType::DEVICE>(
					  particle_buffer.acquire()
					, prev_particle_buffer.acquire()
					//, grid_blocks[0][i].acquire()
					, reinterpret_cast<void**>(&surface_vertex_count_ptr)
					, reinterpret_cast<void**>(&surface_triangle_count_ptr)
					, reinterpret_cast<void**>(&particle_id_mapping_buffer)
					, reinterpret_cast<void**>(&prev_particle_buffer.cellbuckets)
					, reinterpret_cast<void**>(&prev_particle_buffer.bin_offsets)
					, reinterpret_cast<void**>(&prev_particle_buffer.cell_particle_counts)
					, reinterpret_cast<void**>(&particle_buffer.particle_bucket_sizes)
					, reinterpret_cast<void**>(&particle_buffer.blockbuckets)
				);
				
				auto policy = thrust::cuda::par.on(static_cast<cudaStream_t>(cu_dev.stream_compute()));
				thrust::device_ptr<int> host_particle_bucket_sizes = thrust::device_pointer_cast(particle_buffer.particle_bucket_sizes);
				particle_count = thrust::reduce(policy, host_particle_bucket_sizes, host_particle_bucket_sizes + partition_block_count);
			});
			
			cu_dev.syncStream<streamIdx::COMPUTE>();
			
			//Reallocate particle array if necessary
			if(particle_counts[i] < particle_count){
				particle_counts[i] = particle_count;
				particles[i].resize(managed_memory_allocator, sizeof(float) * config::NUM_DIMENSIONS * particle_count);
			}
			//Create/Increase transfer buffers
			//NOTE: as we use a ReusableAllocator on resize the old buffer is automatically freed
			surface_transfer_device_buffer = reusable_allocator_surface_transfer.allocate(sizeof(float) * config::NUM_DIMENSIONS * particle_count);
			
			void* surface_transfer_device_buffer_ptr = surface_transfer_device_buffer;
			
			managed_memory.acquire<MemoryType::DEVICE>(
				  particles[i].acquire()
				, reinterpret_cast<void**>(&surface_transfer_device_buffer_ptr)
			);
			
			//Resize the output model
			model.resize(particle_count);
			surface_point_type_transfer_host_buffer.resize(particle_count);
			surface_normal_transfer_host_buffer.resize(particle_count);
			surface_mean_curvature_transfer_host_buffer.resize(particle_count);
			surface_gauss_curvature_transfer_host_buffer.resize(particle_count);
			
			fmt::print(fg(fmt::color::red), "total number of particles {}\n", particle_count);
			
			//Generate particle_id_mapping
			match(particle_bins[rollid][i])([this, &cu_dev, &i, &d_particle_count, &particle_id_mapping_buffer](const auto& particle_buffer) {
				//partition_block_count; G_PARTICLE_BATCH_CAPACITY
				cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, generate_particle_id_mapping, partitions[rollid], partitions[(rollid + 1) % BIN_COUNT], particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), particle_id_mapping_buffer, d_particle_count);
			});
			
			cu_dev.syncStream<streamIdx::COMPUTE>();
			
			/*
			//std::cout << std::endl << "TEST0" << std::endl;
			#ifdef UPDATE_SURFACE_BEFORE_OUTPUT
			//Recalculate alpha shapes for current buffer state
			{
				//Alpha shapes
				managed_memory.acquire<MemoryType::DEVICE>(alpha_shapes_grid_buffer.acquire());
				
				//Resize particle buffers if we increased the size of active bins
				if(checked_bin_counts[i] > 0) {
					surface_particle_buffers[i].resize(managed_memory_allocator, cur_num_active_bins[i]);
				}
				
				//Initialize surface_triangle_count with 0
				check_cuda_errors(cudaMemsetAsync(surface_triangle_count, 0, sizeof(uint32_t), cu_dev.stream_compute()));

				match(particle_bins[rollid][i])([this, &cu_dev, &i, &particle_id_mapping_buffer](const auto& particle_buffer) {
					managed_memory.acquire<MemoryType::DEVICE>(
						  grid_blocks[0][i].acquire()
						, surface_particle_buffers[i].acquire()
					);
					
					//Clear buffer before use
					cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, alpha_shapes_clear_surface_particle_buffer, particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), partitions[(rollid + 1) % BIN_COUNT], surface_particle_buffers[i]);
					check_cuda_errors(cudaMemset(surface_triangle_buffer, 0, sizeof(int) * MAX_ALPHA_SHAPE_TRIANGLES_PER_MODEL));
					
					//FIXME: Does not yet work, maybe also need to reduce block dimension?
					for(unsigned int start_index = 0; start_index < partition_block_count; start_index += ALPHA_SHAPES_MAX_KERNEL_SIZE){
						LaunchConfig alpha_shapes_launch_config(0, 0);
						alpha_shapes_launch_config.dg = dim3(std::min(ALPHA_SHAPES_MAX_KERNEL_SIZE, partition_block_count - start_index) * config::G_BLOCKVOLUME);
						alpha_shapes_launch_config.db = dim3(ALPHA_SHAPES_BLOCK_SIZE, 1, 1);
						
						//partition_block_count; {config::G_BLOCKSIZE, config::G_BLOCKSIZE, config::G_BLOCKSIZE}
						cu_dev.compute_launch(std::move(alpha_shapes_launch_config), alpha_shapes, particle_buffer, partitions[(rollid + 1) % BIN_COUNT], partitions[rollid], grid_blocks[0][i], surface_particle_buffers[i], alpha_shapes_grid_buffer, surface_triangle_buffer, surface_triangle_count, particle_id_mapping_buffer, start_index, static_cast<int>(cur_frame));
					}
					
					managed_memory.release(
						grid_blocks[0][i].release()
					);
				});
				
				managed_memory.release(alpha_shapes_grid_buffer.release());
			}
			
			cu_dev.syncStream<streamIdx::COMPUTE>();
			
			//std::cout << std::endl << "TEST1" << std::endl;

			uint32_t surface_triangle_count_host;
			check_cuda_errors(cudaMemcpyAsync(&surface_triangle_count_host, surface_triangle_count, sizeof(uint32_t), cudaMemcpyDefault, cu_dev.stream_compute()));
			cu_dev.syncStream<streamIdx::COMPUTE>();
			surface_triangle_transfer_host_buffer.resize(surface_triangle_count_host);
			
			check_cuda_errors(cudaMemcpyAsync(surface_triangle_transfer_host_buffer.data(), surface_triangle_buffer, 3 * sizeof(int) * (surface_triangle_count_host), cudaMemcpyDefault, cu_dev.stream_compute()));
			
			managed_memory.release(
				  surface_triangle_buffer
				, surface_triangle_count
			);
			
			cu_dev.syncStream<streamIdx::COMPUTE>();
			#endif
			
			
			managed_memory.acquire<MemoryType::DEVICE>(
				&surface_transfer_device_buffer_ptr
			);*/
			
			#ifdef UPDATE_SURFACE_BEFORE_OUTPUT
			//Marching Cubes
			{
				//Calculate bounding box
				std::array<int, 3>* bounding_box_min_device = static_cast<std::array<int, 3>*>(cu_dev.borrow(3 * sizeof(int)));
				std::array<int, 3>* bounding_box_max_device = static_cast<std::array<int, 3>*>(cu_dev.borrow(3 * sizeof(int)));
				
				//Init with max/min
				//NOTE: cannot use std::numeric_limits<int>::max() cause that sets value to -1. Maybe use thrust to fill arrays?
				thrust::fill(thrust::device, reinterpret_cast<int*>(bounding_box_min_device), reinterpret_cast<int*>(bounding_box_min_device) + 3, std::numeric_limits<int>::max());
				thrust::fill(thrust::device, reinterpret_cast<int*>(bounding_box_max_device), reinterpret_cast<int*>(bounding_box_max_device) + 3, std::numeric_limits<int>::min());
				
				cu_dev.syncStream<streamIdx::COMPUTE>();
				
				//match(particle_bins[rollid][i])([this, &cu_dev, &i, &bounding_box_min_device, &bounding_box_max_device](const auto& particle_buffer) {
				//	//partition_block_count; G_PARTICLE_BATCH_CAPACITY
				//	cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, get_bounding_box, partitions[rollid], partitions[(rollid + 1) % BIN_COUNT], particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), grid_blocks[0][i], bounding_box_min_device, bounding_box_max_device);
				//});
				match(particle_bins[rollid][i])([this, &cu_dev, &i, &bounding_box_min_device, &bounding_box_max_device](const auto& particle_buffer) {
					//partition_block_count; G_PARTICLE_BATCH_CAPACITY
					cu_dev.compute_launch({partition_block_count, config::G_BLOCKVOLUME}, get_bounding_box, partitions[rollid], particle_buffer, bounding_box_min_device, bounding_box_max_device);
				});
				//cu_dev.compute_launch({partition_block_count, config::G_BLOCKVOLUME}, get_bounding_box, partition_block_count, partitions[rollid], grid_blocks[0][i], bounding_box_min_device, bounding_box_max_device);
			
				ivec3 bounding_box_min;
				ivec3 bounding_box_max;
				check_cuda_errors(cudaMemcpyAsync(bounding_box_min.data(), bounding_box_min_device, 3 * sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
				check_cuda_errors(cudaMemcpyAsync(bounding_box_max.data(), bounding_box_max_device, 3 * sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
				
				cu_dev.syncStream<streamIdx::COMPUTE>();
				
				//Only perceed if bounds are valid
				if(
					   (bounding_box_min[0] != std::numeric_limits<int>::max())
					&& (bounding_box_min[1] != std::numeric_limits<int>::max())
					&& (bounding_box_min[2] != std::numeric_limits<int>::max())
					&& (bounding_box_max[0] != std::numeric_limits<int>::min())
					&& (bounding_box_max[1] != std::numeric_limits<int>::min())
					&& (bounding_box_max[2] != std::numeric_limits<int>::min())
				){
					//Extend to neighbour cells
					bounding_box_min -= ivec3(static_cast<int>(MARCHING_CUBES_INTERPOLATION_DEGREE) + 1, static_cast<int>(MARCHING_CUBES_INTERPOLATION_DEGREE) + 1, static_cast<int>(MARCHING_CUBES_INTERPOLATION_DEGREE) + 1);
					bounding_box_max += ivec3(static_cast<int>(MARCHING_CUBES_INTERPOLATION_DEGREE) + 1, static_cast<int>(MARCHING_CUBES_INTERPOLATION_DEGREE) + 1, static_cast<int>(MARCHING_CUBES_INTERPOLATION_DEGREE) + 1);
					
					//NOTE: Plus 1 cause both min and max are inclusive
					const ivec3 marching_cubes_grid_size = ((bounding_box_max - bounding_box_min + ivec3(1, 1, 1)) * MARCHING_CUBES_GRID_SCALING).cast<int>();
					const vec3 bounding_box_offset = (vec3(grid_blocks[0][i].get_offset()[0], grid_blocks[0][i].get_offset()[1], grid_blocks[0][i].get_offset()[2]) + bounding_box_min) * config::G_DX;
					const size_t marching_cubes_block_count = marching_cubes_grid_size[0] * marching_cubes_grid_size[1] * marching_cubes_grid_size[2];
					
					LaunchConfig marching_cubes_launch_config(0, 0);
					marching_cubes_launch_config.dg = dim3(((marching_cubes_grid_size[0] + 4 - 1) / 4), ((marching_cubes_grid_size[1] + 4 - 1) / 4), ((marching_cubes_grid_size[2] + 4 - 1) / 4));
					marching_cubes_launch_config.db = dim3(4, 4, 4);
					
					//std::cout << "Min: " << bounding_box_min[0] << " " << bounding_box_min[1] << " " << bounding_box_min[2] << std::endl;
					//std::cout << "Max: " << bounding_box_max[0] << " " << bounding_box_max[1] << " " << bounding_box_max[2] << std::endl;
					//std::cout << "Size: " << marching_cubes_grid_size[0] << " " << marching_cubes_grid_size[1] << " " << marching_cubes_grid_size[2] << std::endl;
					//std::cout << "Offset: " << bounding_box_offset[0] << " " << bounding_box_offset[1] << " " << bounding_box_offset[2] << std::endl;
					
					//Resize and clear grid
					if(global_marching_cubes_block_count < marching_cubes_block_count){
						if(global_marching_cubes_block_count > 0){
							global_marching_cubes_block_count = std::max(marching_cubes_block_count / global_marching_cubes_block_count, static_cast<size_t>(2)) * global_marching_cubes_block_count;
						}else{
							global_marching_cubes_block_count = marching_cubes_block_count;
						}
						
						marching_cubes_grid_buffer.resize(managed_memory_allocator, global_marching_cubes_block_count);
					}
					marching_cubes_grid_buffer.reset(marching_cubes_block_count, cu_dev);
					
					//Resize particle buffers if we increased the size of active bins
					if(checked_bin_counts[i] > 0) {
						surface_particle_buffers[i].resize(managed_memory_allocator, cur_num_active_bins[i]);
					}
					
					cu_dev.syncStream<streamIdx::COMPUTE>();
					
					managed_memory.acquire<MemoryType::DEVICE>(
							marching_cubes_grid_buffer.acquire()
						  , surface_particle_buffers[i].acquire()
					);
					
					//Clear particle buffer
					match(particle_bins[rollid][i])([this, &cu_dev, &i](const auto& particle_buffer) {
						cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, marching_cubes_clear_surface_particle_buffer, particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), partitions[(rollid + 1) % BIN_COUNT], surface_particle_buffers[i]);
					});
					
					//Calculate densities
					match(particle_bins[rollid][i])([this, &cu_dev, &i, &bounding_box_offset, &marching_cubes_grid_size](const auto& particle_buffer) {
						//partition_block_count; G_PARTICLE_BATCH_CAPACITY
						cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, marching_cubes_calculate_density, partitions[rollid], partitions[(rollid + 1) % BIN_COUNT], particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), marching_cubes_grid_buffer, bounding_box_offset.data_arr(), marching_cubes_grid_size.data_arr());
					});
					
					//Init with default values
					match(particle_bins[rollid][i])([this, &cu_dev, &i, &bounding_box_offset, &marching_cubes_grid_size](const auto& particle_buffer) {
						//partition_block_count; G_PARTICLE_BATCH_CAPACITY
						cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, marching_cubes_init_surface_particle_buffer, partitions[rollid], partitions[(rollid + 1) % BIN_COUNT], particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), surface_particle_buffers[i], marching_cubes_grid_buffer, bounding_box_offset.data_arr(), marching_cubes_grid_size.data_arr());
					});
					
					//TODO: Do this before setting grid size to minimize it
					//Sort out invalid cells
					uint32_t* removed_cells = static_cast<uint32_t*>(cu_dev.borrow(sizeof(uint32_t)));
				
					uint32_t removed_cells_host;
					do{
						//Init removed_particles with 0
						check_cuda_errors(cudaMemsetAsync(removed_cells, 0, sizeof(uint32_t), cu_dev.stream_compute()));
						
						cu_dev.syncStream<streamIdx::COMPUTE>();
						
						match(particle_bins[rollid][i])([this, &cu_dev, &marching_cubes_launch_config, &i, &bounding_box_min, &bounding_box_offset, &marching_cubes_grid_size, &removed_cells](const auto& particle_buffer) {
							const float density_threshold = particle_buffer.rho * config::MARCHING_CUBES_DENSITY_THRESHOLD_FACTOR;
							
							//partition_block_count; G_PARTICLE_BATCH_CAPACITY
							cu_dev.compute_launch(std::move(LaunchConfig(marching_cubes_launch_config)), marching_cubes_sort_out_invalid_cells, partitions[rollid], partitions[(rollid + 1) % BIN_COUNT], particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), density_threshold, marching_cubes_grid_buffer, bounding_box_min.data_arr(), bounding_box_offset.data_arr(), marching_cubes_grid_size.data_arr(), removed_cells);
						});
						
						check_cuda_errors(cudaMemcpyAsync(&removed_cells_host, removed_cells, sizeof(uint32_t), cudaMemcpyDefault, cu_dev.stream_compute()));
					
						cu_dev.syncStream<streamIdx::COMPUTE>();
					}while(removed_cells_host > 0);
					
					//Initialize surface_triangle_count and surface_vertex_count with 0
					check_cuda_errors(cudaMemsetAsync(surface_vertex_count_ptr, 0, sizeof(uint32_t), cu_dev.stream_compute()));
					check_cuda_errors(cudaMemsetAsync(surface_triangle_count_ptr, 0, sizeof(uint32_t), cu_dev.stream_compute()));
						
					cu_dev.syncStream<streamIdx::COMPUTE>();
		
					//Get counts
					match(particle_bins[rollid][i])([this, &cu_dev, &i, &marching_cubes_launch_config, &bounding_box_min, &bounding_box_offset, &marching_cubes_grid_size, &surface_vertex_count_ptr, &surface_triangle_count_ptr, &particle_id_mapping_buffer](const auto& particle_buffer) {
						const float density_threshold = particle_buffer.rho * config::MARCHING_CUBES_DENSITY_THRESHOLD_FACTOR;
						
						cu_dev.compute_launch(std::move(LaunchConfig(marching_cubes_launch_config)), marching_cubes_gen_vertices, partitions[(rollid + 1) % BIN_COUNT], particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), bounding_box_min.data_arr(), bounding_box_offset.data_arr(), marching_cubes_grid_size.data_arr(), density_threshold, marching_cubes_grid_buffer, static_cast<uint32_t*>(surface_vertex_count_ptr));
						cu_dev.compute_launch(std::move(LaunchConfig(marching_cubes_launch_config)), marching_cubes_gen_faces, partitions[(rollid + 1) % BIN_COUNT], particle_buffer, particle_id_mapping_buffer, surface_particle_buffers[i], bounding_box_min.data_arr(), bounding_box_offset.data_arr(), marching_cubes_grid_size.data_arr(), density_threshold, marching_cubes_grid_buffer, static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(surface_triangle_count_ptr));
					});
					
					cu_dev.syncStream<streamIdx::COMPUTE>();
					
					//Adjust buffers sizes if necessary
					uint32_t surface_vertex_count_host;
					uint32_t surface_triangle_count_host;
					check_cuda_errors(cudaMemcpyAsync(&surface_vertex_count_host, surface_vertex_count_ptr, sizeof(uint32_t), cudaMemcpyDefault, cu_dev.stream_compute()));
					check_cuda_errors(cudaMemcpyAsync(&surface_triangle_count_host, surface_triangle_count_ptr, sizeof(uint32_t), cudaMemcpyDefault, cu_dev.stream_compute()));
					
					cu_dev.syncStream<streamIdx::COMPUTE>();
					
					surface_triangle_count_host /= 3;
					
					//std::cout << surface_vertex_count_host << " " << surface_triangle_count_host << std::endl;
					if(surface_triangle_count_host > 0){
						if(max_surface_vertex_count < surface_vertex_count_host){
							//managed_memory_allocator.deallocate(surface_vertex_buffer, sizeof(float) * config::NUM_DIMENSIONS * max_surface_vertex_count);
							//surface_vertex_buffer = reinterpret_cast<float*>(managed_memory_allocator.allocate(sizeof(float) * config::NUM_DIMENSIONS * surface_vertex_count_host));
							max_surface_vertex_count = surface_vertex_count_host;
						}
						
						if(max_surface_triangle_count < surface_triangle_count_host){
							managed_memory_allocator.deallocate(surface_triangle_buffer, sizeof(uint32_t) * 3 * max_surface_triangle_count);
							surface_triangle_buffer = reinterpret_cast<uint32_t*>(managed_memory_allocator.allocate(sizeof(uint32_t) * 3 * surface_triangle_count_host));
							max_surface_triangle_count = surface_triangle_count_host;
						}
						
						//surface_vertex_transfer_host_buffer.resize(surface_vertex_count_host);
						surface_triangle_transfer_host_buffer.resize(surface_triangle_count_host);
						
						//void* surface_vertex_buffer_ptr = surface_vertex_buffer;
						void* surface_triangle_buffer_ptr = surface_triangle_buffer;
						
						managed_memory.acquire<MemoryType::DEVICE>(
							  //reinterpret_cast<void**>(&surface_vertex_buffer_ptr)
							  reinterpret_cast<void**>(&surface_triangle_buffer_ptr)
						);
						
						//Clear buffers
						//check_cuda_errors(cudaMemsetAsync(surface_vertex_buffer_ptr, 0.0f, sizeof(float) * config::NUM_DIMENSIONS * surface_vertex_count_host, cu_dev.stream_compute()));
						check_cuda_errors(cudaMemsetAsync(surface_triangle_buffer_ptr, 0, sizeof(uint32_t) * 3 * surface_triangle_count_host, cu_dev.stream_compute()));
						
						//Reset counts to 0
						check_cuda_errors(cudaMemsetAsync(surface_triangle_count_ptr, 0, sizeof(uint32_t), cu_dev.stream_compute()));
						check_cuda_errors(cudaMemsetAsync(surface_vertex_count_ptr, 0, sizeof(uint32_t), cu_dev.stream_compute()));
						
						//Calculate triangulation
						match(particle_bins[rollid][i])([this, &cu_dev, &i, &marching_cubes_launch_config, &bounding_box_min, &bounding_box_offset, &marching_cubes_grid_size, &surface_vertex_count_ptr, &surface_triangle_count_ptr, &surface_triangle_buffer_ptr, &particle_id_mapping_buffer](const auto& particle_buffer) {
							const float density_threshold = particle_buffer.rho * config::MARCHING_CUBES_DENSITY_THRESHOLD_FACTOR;
							
							cu_dev.compute_launch(std::move(LaunchConfig(marching_cubes_launch_config)), marching_cubes_gen_vertices, partitions[(rollid + 1) % BIN_COUNT], particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), bounding_box_min.data_arr(), bounding_box_offset.data_arr(), marching_cubes_grid_size.data_arr(), density_threshold, marching_cubes_grid_buffer, static_cast<uint32_t*>(surface_vertex_count_ptr));
							cu_dev.compute_launch(std::move(LaunchConfig(marching_cubes_launch_config)), marching_cubes_gen_faces, partitions[(rollid + 1) % BIN_COUNT], particle_buffer, particle_id_mapping_buffer, surface_particle_buffers[i], bounding_box_min.data_arr(), bounding_box_offset.data_arr(), marching_cubes_grid_size.data_arr(), density_threshold, marching_cubes_grid_buffer, static_cast<uint32_t*>(surface_triangle_buffer_ptr), static_cast<uint32_t*>(surface_triangle_count_ptr));
						});
						
						cu_dev.syncStream<streamIdx::COMPUTE>();
						
						//Create particle properties from triangulation
						
						
						//Copy surface to host
						//check_cuda_errors(cudaMemcpyAsync(surface_vertex_transfer_host_buffer.data(), surface_vertex_buffer_ptr, config::NUM_DIMENSIONS * sizeof(float) * (surface_vertex_count_host), cudaMemcpyDefault, cu_dev.stream_compute()));
						check_cuda_errors(cudaMemcpyAsync(surface_triangle_transfer_host_buffer.data(), surface_triangle_buffer_ptr, 3 * sizeof(uint32_t) * (surface_triangle_count_host), cudaMemcpyDefault, cu_dev.stream_compute()));
						
						cu_dev.syncStream<streamIdx::COMPUTE>();
						
						managed_memory.release(
							  //surface_vertex_buffer
							  surface_triangle_buffer
						);
					}
						
					managed_memory.release(
						  //grid_blocks[0][i].release()
						  marching_cubes_grid_buffer.release()
					);
				}
			}
			#endif

			//Copy particle data to output buffer
			match(particle_bins[rollid][i])([this, &cu_dev, &i, &d_particle_count, &particle_id_mapping_buffer](const auto& particle_buffer) {
				//partition_block_count; G_PARTICLE_BATCH_CAPACITY
				cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, retrieve_particle_buffer, partitions[rollid], partitions[(rollid + 1) % BIN_COUNT], particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), particles[i], particle_id_mapping_buffer);
			});

			cu_dev.syncStream<streamIdx::COMPUTE>();
			
			//Copy the data to the output model
			check_cuda_errors(cudaMemcpyAsync(model.data(), static_cast<void*>(&particles[i].val_1d(_0, 0)), sizeof(std::array<float, config::NUM_DIMENSIONS>) * (particle_count), cudaMemcpyDefault, cu_dev.stream_compute()));
			
			managed_memory.release(particles[i].release());
			
			
			{
				match(particle_bins[rollid][i])([this, &cu_dev, &i, &d_particle_count, &particle_id_mapping_buffer, &surface_transfer_device_buffer_ptr](const auto& particle_buffer) {
					//partition_block_count; G_PARTICLE_BATCH_CAPACITY
					cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, retrieve_particle_buffer_surface, partitions[rollid], partitions[(rollid + 1) % BIN_COUNT], particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), surface_particle_buffers[i], particle_id_mapping_buffer, static_cast<int*>(surface_transfer_device_buffer_ptr), static_cast<float*>(nullptr), static_cast<float*>(nullptr), static_cast<float*>(nullptr));
				});
				
				cu_dev.syncStream<streamIdx::COMPUTE>();
				
				//Copy the data to the output model
				check_cuda_errors(cudaMemcpyAsync(surface_point_type_transfer_host_buffer.data(), surface_transfer_device_buffer_ptr, sizeof(int) * (particle_count), cudaMemcpyDefault, cu_dev.stream_compute()));
			}
			
			{
				match(particle_bins[rollid][i])([this, &cu_dev, &i, &d_particle_count, &particle_id_mapping_buffer, &surface_transfer_device_buffer_ptr](const auto& particle_buffer) {
					//partition_block_count; G_PARTICLE_BATCH_CAPACITY
					cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, retrieve_particle_buffer_surface, partitions[rollid], partitions[(rollid + 1) % BIN_COUNT], particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), surface_particle_buffers[i], particle_id_mapping_buffer, static_cast<int*>(nullptr), static_cast<float*>(surface_transfer_device_buffer_ptr), static_cast<float*>(nullptr), static_cast<float*>(nullptr));
				});
				
				cu_dev.syncStream<streamIdx::COMPUTE>();
				
				//Copy the data to the output model
				check_cuda_errors(cudaMemcpyAsync(surface_normal_transfer_host_buffer.data(), surface_transfer_device_buffer_ptr, sizeof(std::array<float, config::NUM_DIMENSIONS>) * (particle_count), cudaMemcpyDefault, cu_dev.stream_compute()));
			}
			
			{
				match(particle_bins[rollid][i])([this, &cu_dev, &i, &d_particle_count, &particle_id_mapping_buffer, &surface_transfer_device_buffer_ptr](const auto& particle_buffer) {
					//partition_block_count; G_PARTICLE_BATCH_CAPACITY
					cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, retrieve_particle_buffer_surface, partitions[rollid], partitions[(rollid + 1) % BIN_COUNT], particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), surface_particle_buffers[i], particle_id_mapping_buffer, static_cast<int*>(nullptr), static_cast<float*>(nullptr), static_cast<float*>(surface_transfer_device_buffer_ptr), static_cast<float*>(nullptr));
				});
				
				cu_dev.syncStream<streamIdx::COMPUTE>();
				
				//Copy the data to the output model
				check_cuda_errors(cudaMemcpyAsync(surface_mean_curvature_transfer_host_buffer.data(), surface_transfer_device_buffer_ptr, sizeof(float) * (particle_count), cudaMemcpyDefault, cu_dev.stream_compute()));
			}
			
			{
				match(particle_bins[rollid][i])([this, &cu_dev, &i, &d_particle_count, &particle_id_mapping_buffer, &surface_transfer_device_buffer_ptr](const auto& particle_buffer) {
					//partition_block_count; G_PARTICLE_BATCH_CAPACITY
					cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, retrieve_particle_buffer_surface, partitions[rollid], partitions[(rollid + 1) % BIN_COUNT], particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), surface_particle_buffers[i], particle_id_mapping_buffer, static_cast<int*>(nullptr), static_cast<float*>(nullptr), static_cast<float*>(nullptr), static_cast<float*>(surface_transfer_device_buffer_ptr));
				});
				
				cu_dev.syncStream<streamIdx::COMPUTE>();
				
				//Copy the data to the output model
				check_cuda_errors(cudaMemcpyAsync(surface_gauss_curvature_transfer_host_buffer.data(), surface_transfer_device_buffer_ptr, sizeof(float) * (particle_count), cudaMemcpyDefault, cu_dev.stream_compute()));
			}
			
			match(particle_bins[(rollid + 1) % BIN_COUNT][i])([this, &cu_dev, &particle_count, &i, &particle_id_mapping_buffer, &surface_transfer_device_buffer_ptr](auto& particle_buffer) {
				auto& prev_particle_buffer = get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[rollid][i]);
			
				managed_memory.release(
					  (surface_particle_buffers[i].is_locked() ? surface_particle_buffers[i].release() : nullptr)
					, particle_buffer.release()
					, prev_particle_buffer.release()
					, surface_vertex_count
					, surface_triangle_count
					, particle_id_mapping_buffer
					, surface_transfer_device_buffer
					, prev_particle_buffer.cellbuckets_virtual
					, prev_particle_buffer.bin_offsets_virtual
					, prev_particle_buffer.cell_particle_counts_virtual
					, particle_buffer.particle_bucket_sizes_virtual
					, particle_buffer.blockbuckets_virtual
				);
			
			});
			
			cu_dev.syncStream<streamIdx::COMPUTE>();
			
			std::string fn = std::string {"model"} + "_id[" + std::to_string(i) + "]_frame[" + std::to_string(cur_frame) + "].bgeo";

			//Write back file
			IO::insert_job([fn, m = model, surface_point_type = surface_point_type_transfer_host_buffer, surface_normal = surface_normal_transfer_host_buffer, surface_mean_curvature = surface_mean_curvature_transfer_host_buffer, surface_gauss_curvature = surface_gauss_curvature_transfer_host_buffer]() {
				Partio::ParticlesDataMutable* parts;
				begin_write_partio(&parts, m.size());
				
				write_partio_add(m, std::string("position"), parts);
				write_partio_add(surface_point_type, std::string("point_type"), parts);
				write_partio_add(surface_normal, std::string("N"), parts);
				write_partio_add(surface_mean_curvature, std::string("mean_curvature"), parts);
				write_partio_add(surface_gauss_curvature, std::string("gauss_curvature"), parts);

				end_write_partio(fn, parts);
			});
			
			#ifdef UPDATE_SURFACE_BEFORE_OUTPUT
			//Write back alpha shapes mesh
			std::string fn_surface = std::string {"surface"} + "_id[" + std::to_string(i) + "]_frame[" + std::to_string(cur_frame) + "].obj";
			IO::insert_job([this, fn_surface, i, pos = model, faces = surface_triangle_transfer_host_buffer]() {
				write_triangle_mesh<float, uint32_t, config::NUM_DIMENSIONS>(fn_surface, pos, faces);
			});
			#endif
		}
		timer.tock(fmt::format("GPU[{}] frame {} step {} retrieve_particles", gpuid, cur_frame, cur_step));
		
		timer.tick();

		for(int i = 0; i < triangle_meshes.size(); ++i) {
			void* triangle_mesh_transfer_device_buffer_ptr = triangle_mesh_transfer_device_buffers[i];
			managed_memory.acquire<MemoryType::DEVICE>(
				  triangle_meshes[i].acquire()
				, &triangle_mesh_transfer_device_buffer_ptr
			);
			
			//Copy the data to the output model
			cu_dev.compute_launch({(triangle_mesh_vertex_counts[i] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, copy_triangle_mesh_data_to_host, triangle_meshes[i], triangle_mesh_vertex_counts[i], static_cast<float*>(triangle_mesh_transfer_device_buffer_ptr));
			check_cuda_errors(cudaMemcpyAsync(triangle_mesh_transfer_host_buffers[i].data(), triangle_mesh_transfer_device_buffer_ptr, sizeof(std::array<float, config::NUM_DIMENSIONS>) * triangle_mesh_vertex_counts[i], cudaMemcpyDefault, cu_dev.stream_compute()));
			cu_dev.syncStream<streamIdx::COMPUTE>();
			
			managed_memory.release(
				  triangle_meshes[i].release()
				, triangle_mesh_transfer_device_buffers[i]
			);

			//Write out initial state to file
			std::string fn = std::string {"mesh"} + "_id[" + std::to_string(i) + "]_frame[" + std::to_string(cur_frame) + "].obj";
			IO::insert_job([this, fn, i, pos = triangle_mesh_transfer_host_buffers[i]]() {
				write_triangle_mesh<float, uint32_t, config::NUM_DIMENSIONS>(fn, pos, triangle_mesh_face_buffers[i]);
			});
		}
		
		#ifdef OUTPUT_TRIANGLE_SHELL_OUTER_POS
		//Flush IO, so that we can safely reuse our transfer buffer
		IO::flush();
		for(int i = 0; i < get_model_count(); ++i) {
			for(int j = 0; j < triangle_meshes.size(); ++j) {
				void* triangle_mesh_transfer_device_buffer_ptr = triangle_mesh_transfer_device_buffers[i];
				managed_memory.acquire<MemoryType::DEVICE>(
					  triangle_shells[rollid][i][j].acquire()
					, &triangle_mesh_transfer_device_buffer_ptr
				);
					
				//Copy the data to the output model
				cu_dev.compute_launch({(triangle_mesh_vertex_counts[j] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, copy_triangle_shell_data_to_host, triangle_shells[rollid][i][j], triangle_mesh_vertex_counts[j], static_cast<float*>(triangle_mesh_transfer_device_buffer_ptr));
				check_cuda_errors(cudaMemcpyAsync(triangle_mesh_transfer_host_buffers[j].data(), triangle_mesh_transfer_device_buffer_ptr, sizeof(std::array<float, config::NUM_DIMENSIONS>) * triangle_mesh_vertex_counts[j], cudaMemcpyDefault, cu_dev.stream_compute()));
				cu_dev.syncStream<streamIdx::COMPUTE>();
				
				managed_memory.release(
					  triangle_shells[rollid][i][j].release()
					, triangle_mesh_transfer_device_buffers[i]
				);

				//Write out initial state to file
				std::string fn = std::string {"shell"} + "_id[" + std::to_string(i) + "_" + std::to_string(j) + "]_frame[" + std::to_string(cur_frame) + "].obj";
				IO::insert_job([this, fn, j, pos = triangle_mesh_transfer_host_buffers[j]]() {
					write_triangle_mesh<float, uint32_t, config::NUM_DIMENSIONS>(fn, pos, triangle_mesh_face_buffers[j]);
				});
			}
		}
		#endif
		timer.tock(fmt::format("GPU[{}] frame {} step {} retrieve_triangle_mesh", gpuid, cur_frame, cur_step));
	}

	//NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)Current c++ version does not yet support std::span
	void initial_setup() {
		//Initialize triangle meshes
		{
			auto& cu_dev = Cuda::ref_cuda_context(gpuid);
			CudaTimer timer {cu_dev.stream_compute()};
			
			timer.tick();
			
			cu_dev.syncStream<streamIdx::COMPUTE>();
			
			void* init_tmp = managed_memory_allocator.allocate(sizeof(float) * 9);
			void* init_tmp_ptr = init_tmp;
			managed_memory.acquire<MemoryType::DEVICE>(
				  &init_tmp_ptr
			);
			
			for(int i = 0; i < triangle_meshes.size(); ++i) {
				managed_memory.acquire<MemoryType::DEVICE>(
					  triangle_meshes[i].acquire()
				);
				
				//Calculate center of mass
				vec3 center_of_mass;
				check_cuda_errors(cudaMemsetAsync(init_tmp_ptr, 0, sizeof(float) * 9, cu_dev.stream_compute()));
				cu_dev.compute_launch({(triangle_mesh_vertex_counts[i] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, calculate_center_of_mass, triangle_meshes[i], triangle_mesh_vertex_counts[i],  static_cast<float*>(init_tmp_ptr));
				check_cuda_errors(cudaMemcpyAsync(center_of_mass.data(), init_tmp_ptr, sizeof(float) * 3, cudaMemcpyDefault, cu_dev.stream_compute()));
				cu_dev.syncStream<streamIdx::COMPUTE>();
				
				triangle_meshes[i].center = center_of_mass / triangle_meshes[i].mass;
			
				//Calculate inertia
				check_cuda_errors(cudaMemsetAsync(init_tmp_ptr, 0, sizeof(float) * 9, cu_dev.stream_compute()));
				cu_dev.compute_launch({(triangle_mesh_vertex_counts[i] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, calculate_inertia_and_relative_pos, triangle_meshes[i], triangle_mesh_vertex_counts[i], triangle_meshes[i].center.data_arr(),  static_cast<float*>(init_tmp_ptr));
				
				check_cuda_errors(cudaMemcpyAsync(triangle_meshes[i].inertia.data(), init_tmp_ptr, sizeof(float) * 9, cudaMemcpyDefault, cu_dev.stream_compute()));
				
				//Calculate normals per vertex
				cu_dev.compute_launch({(triangle_mesh_face_counts[i] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, calculate_normals_and_base_area, triangle_meshes[i], triangle_mesh_face_counts[i]);
				cu_dev.compute_launch({(triangle_mesh_vertex_counts[i] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, normalize_normals, triangle_meshes[i], triangle_mesh_vertex_counts[i]);
				
				//Apply rigid body pose for timestamp 0
				triangle_meshes[i].rigid_body_update(Duration::zero(), Duration::zero());
				cu_dev.compute_launch({(triangle_mesh_vertex_counts[i] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, update_triangle_mesh, triangle_meshes[i], triangle_mesh_vertex_counts[i], triangle_meshes[i].center.data_arr(), triangle_meshes[i].linear_velocity.data_arr(), triangle_meshes[i].rotation.data_arr(), triangle_meshes[i].angular_velocity.data_arr());
			
				cu_dev.syncStream<streamIdx::COMPUTE>();
			
				for(int j = 0; j < get_model_count(); ++j) {
					managed_memory.acquire<MemoryType::DEVICE>(
						  triangle_shells[rollid][j][i].acquire()
					);
					
					//Clear triangle_shell data
					cu_dev.compute_launch({(triangle_mesh_vertex_counts[i] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, init_triangle_shell, triangle_meshes[i], triangle_shells[rollid][j][i], triangle_mesh_vertex_counts[i]);
					
					//TODO: Remove this, cause initial mass is just != 0 for testing reason?
					/*
					cu_dev.compute_launch({(triangle_mesh_vertex_counts[i] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, activate_blocks_for_shell, triangle_mesh_vertex_counts[i], triangle_shells[rollid][j][i], partitions[(rollid + 1) % BIN_COUNT], grid_blocks[0][i]);
					check_cuda_errors(cudaMemsetAsync(triangle_shells[0][i][j].particle_buffer.particle_bucket_sizes, 0, sizeof(int) * (exterior_block_count + 1), cu_dev.stream_compute()));
					cu_dev.compute_launch({(triangle_mesh_vertex_counts[j] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, store_triangle_shell_vertices_in_bucket, triangle_mesh_vertex_counts[j], triangle_shells[rollid][i][j], triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer, partitions[(rollid + 1) % BIN_COUNT]);
					*/
				
					managed_memory.release(
						  triangle_shells[rollid][j][i].release()
					);
					
					cu_dev.syncStream<streamIdx::COMPUTE>();
				}
				
				managed_memory.release(
					  triangle_meshes[i].release()
				);
				
				cu_dev.syncStream<streamIdx::COMPUTE>();
			}
			cudaDeviceSynchronize();
			managed_memory.release(
				  init_tmp
			);
			
			managed_memory_allocator.deallocate(init_tmp, sizeof(float) * 9);
			
			timer.tock(fmt::format("GPU[{}] step {} init_triangle_mesh", gpuid, cur_step));
		}
		
		{
			//TODO: Verify bounds when model offset is too large

			auto& cu_dev = Cuda::ref_cuda_context(gpuid);
			CudaTimer timer {cu_dev.stream_compute()};

			timer.tick();

			//Activate blocks that contain particles
			for(int i = 0; i < get_model_count(); ++i) {
				managed_memory.acquire<MemoryType::DEVICE>(
					  particles[i].acquire()
					, grid_blocks[0][i].acquire()
				);
				
				//floor(particle_counts[i]/config::DEFAULT_CUDA_BLOCK_SIZE); config::DEFAULT_CUDA_BLOCK_SIZE
				cu_dev.compute_launch({(particle_counts[i] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, activate_blocks, particle_counts[i], particles[i], partitions[(rollid + 1) % BIN_COUNT], grid_blocks[0][i]);
			
				managed_memory.release(
					  particles[i].release()
					, grid_blocks[0][i].release()
				);
				
				cu_dev.syncStream<streamIdx::COMPUTE>();
			}

			//Store count of activated blocks
			check_cuda_errors(cudaMemcpyAsync(&partition_block_count, partitions[(rollid + 1) % BIN_COUNT].count, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
			timer.tock(fmt::format("GPU[{}] step {} init_table", gpuid, cur_step));
			
			cu_dev.syncStream<streamIdx::COMPUTE>();

			timer.tick();
			cu_dev.reset_mem();

			//Store particle ids in block cells
			for(int i = 0; i < get_model_count(); ++i) {
				match(particle_bins[rollid][i])([this, &cu_dev, &i](auto& particle_buffer) {
					particle_buffer.cell_particle_counts = particle_buffer.cell_particle_counts_virtual;
					particle_buffer.cellbuckets = particle_buffer.cellbuckets_virtual;
					managed_memory.acquire<MemoryType::DEVICE>(
						  particles[i].acquire()
						, particle_buffer.acquire()
						, grid_blocks[0][i].acquire()
						, reinterpret_cast<void**>(&particle_buffer.cell_particle_counts)
						, reinterpret_cast<void**>(&particle_buffer.cellbuckets)
					);
					
					//floor(particle_counts[i]/config::DEFAULT_CUDA_BLOCK_SIZE); config::DEFAULT_CUDA_BLOCK_SIZE
					cu_dev.compute_launch({(particle_counts[i] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, build_particle_cell_buckets, particle_counts[i], particles[i], particle_buffer, partitions[(rollid + 1) % BIN_COUNT], grid_blocks[0][i]);
				
					managed_memory.release(
						  particles[i].release()
						, particle_buffer.release()
						, grid_blocks[0][i].release()
						, particle_buffer.cell_particle_counts_virtual
						, particle_buffer.cellbuckets_virtual
					);
				});
				
				cu_dev.syncStream<streamIdx::COMPUTE>();
			}
			cu_dev.syncStream<streamIdx::COMPUTE>();
			
			//Check size
			if(partition_block_count > config::G_MAX_ACTIVE_BLOCK) {
				std::cerr << "Too much active blocks: " << partition_block_count << std::endl;
				std::abort();
			}

			//Copy cell buckets from partition to particle buffer
			for(int i = 0; i < get_model_count(); ++i) {
				match(particle_bins[rollid][i])([this, &cu_dev](auto& particle_buffer) {
					particle_buffer.particle_bucket_sizes = particle_buffer.particle_bucket_sizes_virtual;
					particle_buffer.blockbuckets = particle_buffer.blockbuckets_virtual;
					particle_buffer.cell_particle_counts = particle_buffer.cell_particle_counts_virtual;
					particle_buffer.cellbuckets = particle_buffer.cellbuckets_virtual;
					managed_memory.acquire<MemoryType::DEVICE>(
						  reinterpret_cast<void**>(&particle_buffer.particle_bucket_sizes)
						, reinterpret_cast<void**>(&particle_buffer.blockbuckets)
						, reinterpret_cast<void**>(&particle_buffer.cell_particle_counts)
						, reinterpret_cast<void**>(&particle_buffer.cellbuckets)
					);
					
					//First init sizes with 0
					check_cuda_errors(cudaMemsetAsync(particle_buffer.particle_bucket_sizes, 0, sizeof(int) * (partition_block_count + 1), cu_dev.stream_compute()));

					//partition_block_count; G_BLOCKVOLUME
					cu_dev.compute_launch({partition_block_count, config::G_BLOCKVOLUME}, initial_cell_bucket_to_block, particle_buffer.cell_particle_counts, particle_buffer.cellbuckets, particle_buffer.particle_bucket_sizes, particle_buffer.blockbuckets);
					// partitions[(rollid + 1)%BIN_COUNT].buildParticleBuckets(cu_dev, partition_block_count);
				
					managed_memory.release(
						  particle_buffer.particle_bucket_sizes_virtual
						, particle_buffer.blockbuckets_virtual
						, particle_buffer.cell_particle_counts_virtual
						, particle_buffer.cellbuckets_virtual
					);
				});
				
				cu_dev.syncStream<streamIdx::COMPUTE>();
			}

			//Compute bin capacities, bin offsets and the summed bin size
			//Then initializes the particle buffer
			int* bin_sizes = tmps.bin_sizes;
			managed_memory.acquire<MemoryType::DEVICE>(
				  reinterpret_cast<void**>(&bin_sizes)
			);
			for(int i = 0; i < get_model_count(); ++i) {
				match(particle_bins[rollid][i])([this, &cu_dev, &bin_sizes, &i](auto& particle_buffer) {
					particle_buffer.particle_bucket_sizes = particle_buffer.particle_bucket_sizes_virtual;
					particle_buffer.bin_offsets = particle_buffer.bin_offsets_virtual;
					managed_memory.acquire<MemoryType::DEVICE>(
						  particles[i].acquire()
						, particle_buffer.acquire()
						, reinterpret_cast<void**>(&particle_buffer.particle_bucket_sizes)
						, reinterpret_cast<void**>(&particle_buffer.bin_offsets)
					);
					
					//floor((partition_block_count + 1)/G_PARTICLE_BATCH_CAPACITY); G_PARTICLE_BATCH_CAPACITY
					cu_dev.compute_launch({partition_block_count / config::G_PARTICLE_BATCH_CAPACITY + 1, config::G_PARTICLE_BATCH_CAPACITY}, compute_bin_capacity, partition_block_count + 1, static_cast<const int*>(particle_buffer.particle_bucket_sizes), bin_sizes);

					cu_dev.syncStream<streamIdx::COMPUTE>();

					//Ensure we have enough space for new generated particles
					for(int j = 0; j < triangle_shells[(rollid + 1) % BIN_COUNT][i].size(); ++j) {
						triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes = triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes_virtual;
						managed_memory.acquire<MemoryType::DEVICE>(
							  reinterpret_cast<void**>(&triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes)
						);
						
						//floor((exterior_block_count + 1)/G_PARTICLE_BATCH_CAPACITY); G_PARTICLE_BATCH_CAPACITY
						cu_dev.compute_launch({partition_block_count / config::G_PARTICLE_BATCH_CAPACITY + 1, config::G_PARTICLE_BATCH_CAPACITY}, compute_bin_capacity_shell, partition_block_count + 1, static_cast<const int*>(triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes), bin_sizes);
					
						managed_memory.release(
							  triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes_virtual
						);
						
						cu_dev.syncStream<streamIdx::COMPUTE>();
					}

					//Stores aggregated bin sizes in particle_buffer
					exclusive_scan(partition_block_count + 1, bin_sizes, particle_buffer.bin_offsets, cu_dev);

					//Stores last aggregated size == whole size in bincount
					check_cuda_errors(cudaMemcpyAsync(&bincount[i], particle_buffer.bin_offsets + partition_block_count, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
					cu_dev.syncStream<streamIdx::COMPUTE>();

					//Initialize particle buffer
					//partition_block_count; G_PARTICLE_BATCH_CAPACITY
					cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, array_to_buffer, particles[i], particle_buffer);
					
					managed_memory.release(
						  particles[i].release()
						, particle_buffer.release()
						, reinterpret_cast<void**>(&particle_buffer.particle_bucket_sizes_virtual)
						, reinterpret_cast<void**>(&particle_buffer.bin_offsets_virtual)
					);
				});
				
				cu_dev.syncStream<streamIdx::COMPUTE>();
			}
			
			managed_memory.release(
				  tmps.bin_sizes
			);

			//Activate blocks near active blocks
			//floor(partition_block_count/G_PARTICLE_BATCH_CAPACITY); G_PARTICLE_BATCH_CAPACITY
			cu_dev.compute_launch({(partition_block_count + config::G_PARTICLE_BATCH_CAPACITY - 1) / config::G_PARTICLE_BATCH_CAPACITY, config::G_PARTICLE_BATCH_CAPACITY}, register_neighbor_blocks, static_cast<uint32_t>(partition_block_count), partitions[(rollid + 1) % BIN_COUNT]);

			//Retrieve total count
			check_cuda_errors(cudaMemcpyAsync(&neighbor_block_count, partitions[(rollid + 1) % BIN_COUNT].count, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
			cu_dev.syncStream<streamIdx::COMPUTE>();

			//Check size
			if(neighbor_block_count > config::G_MAX_ACTIVE_BLOCK) {
				std::cerr << "Too much neighbour blocks: " << neighbor_block_count << std::endl;
				std::abort();
			}
			
			//Activate blocks before active blocks
			//floor(partition_block_count/G_PARTICLE_BATCH_CAPACITY); G_PARTICLE_BATCH_CAPACITY
			cu_dev.compute_launch({(partition_block_count + config::G_PARTICLE_BATCH_CAPACITY - 1) / config::G_PARTICLE_BATCH_CAPACITY, config::G_PARTICLE_BATCH_CAPACITY}, register_prev_neighbor_blocks, static_cast<uint32_t>(partition_block_count), partitions[(rollid + 1) % BIN_COUNT]);

			//Retrieve total count
			check_cuda_errors(cudaMemcpyAsync(&total_neighbor_block_count, partitions[(rollid + 1) % BIN_COUNT].count, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
			cu_dev.syncStream<streamIdx::COMPUTE>();

			//Check size
			if(total_neighbor_block_count > config::G_MAX_ACTIVE_BLOCK) {
				std::cerr << "Too much total neighbour blocks: " << total_neighbor_block_count << std::endl;
				std::abort();
			}

			//Activate blocks near active blocks, including those before that block
			//TODO: Only these with offset -1 are not already activated as neighbours
			//floor(total_neighbor_block_count/G_PARTICLE_BATCH_CAPACITY); G_PARTICLE_BATCH_CAPACITY
			cu_dev.compute_launch({(total_neighbor_block_count + config::G_PARTICLE_BATCH_CAPACITY - 1) / config::G_PARTICLE_BATCH_CAPACITY, config::G_PARTICLE_BATCH_CAPACITY}, register_exterior_blocks, static_cast<uint32_t>(total_neighbor_block_count), partitions[(rollid + 1) % BIN_COUNT]);

			//Retrieve total count
			check_cuda_errors(cudaMemcpyAsync(&exterior_block_count, partitions[(rollid + 1) % BIN_COUNT].count, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
			cu_dev.syncStream<streamIdx::COMPUTE>();

			//Check size
			if(exterior_block_count > config::G_MAX_ACTIVE_BLOCK) {
				std::cerr << "Too much exterior blocks: " << exterior_block_count << std::endl;
				std::abort();
			}

			timer.tock(fmt::format("GPU[{}] step {} init_partition", gpuid, cur_step));

			fmt::print(fmt::emphasis::bold | fg(fmt::color::yellow), "block count on device {}: {}, {}, {} [{}]\n", gpuid, partition_block_count, neighbor_block_count, exterior_block_count, cur_num_active_blocks);
			for(int i = 0; i < get_model_count(); ++i) {
				fmt::print(fmt::emphasis::bold | fg(fmt::color::yellow), "bin count on device {}: model {}: {} [{}]\n", gpuid, i, bincount[i], cur_num_active_bins[i]);
			}
		}

		{
			auto& cu_dev = Cuda::ref_cuda_context(gpuid);
			CudaTimer timer {cu_dev.stream_compute()};

			//Copy all blocks to background partition
			partitions[(rollid + 1) % BIN_COUNT].copy_to(partitions[rollid], exterior_block_count, cu_dev.stream_compute());
			check_cuda_errors(cudaMemcpyAsync(partitions[rollid].active_keys, partitions[(rollid + 1) % BIN_COUNT].active_keys, sizeof(ivec3) * exterior_block_count, cudaMemcpyDefault, cu_dev.stream_compute()));

			cu_dev.syncStream<streamIdx::COMPUTE>();

			//Copy all particle data to background particle buffer
			for(int i = 0; i < get_model_count(); ++i) {
				match(particle_bins[rollid][i])([this, &cu_dev, &i](auto& particle_buffer) {
					// bin_offsets, particle_bucket_sizes
					particle_buffer.copy_to(get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), partition_block_count, cu_dev.stream_compute());
				});
				
				cu_dev.syncStream<streamIdx::COMPUTE>();
			}

			timer.tick();

			//Initialize the grid and advection buckets
			for(int i = 0; i < get_model_count(); ++i) {
				//Resize and clear the grid
				//NOTE: Resize not necessary cause grid is initialized with mayimum capacity
				//grid_blocks[0][i].resize(managed_memory_allocator, neighbor_block_count);
				grid_blocks[0][i].reset(neighbor_block_count, cu_dev);
				
				match(particle_bins[(rollid + 1) % BIN_COUNT][i])([this, &cu_dev, &i](auto& particle_buffer) {
					particle_buffer.particle_bucket_sizes = particle_buffer.particle_bucket_sizes_virtual;
					particle_buffer.blockbuckets = particle_buffer.blockbuckets_virtual;
					managed_memory.acquire<MemoryType::DEVICE>(
						  particles[i].acquire()
						, grid_blocks[0][i].acquire()
						, reinterpret_cast<void**>(&particle_buffer.particle_bucket_sizes)
						, reinterpret_cast<void**>(&particle_buffer.blockbuckets)
					);
					
					//Initialize mass and momentum
					//floor(particle_counts[i]/config::DEFAULT_CUDA_BLOCK_SIZE); config::DEFAULT_CUDA_BLOCK_SIZE
					cu_dev.compute_launch({(particle_counts[i] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, rasterize, particle_counts[i], particles[i], grid_blocks[0][i], partitions[rollid], dt, get_mass(i), vel0[i].data_arr());
				
					//Init advection source at offset 0 of destination
					//partition_block_count; G_PARTICLE_BATCH_CAPACITY
					cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, init_adv_bucket, static_cast<const int*>(particle_buffer.particle_bucket_sizes), particle_buffer.blockbuckets);
				
					managed_memory.release(
						  particles[i].release()
						, grid_blocks[0][i].release()
						, particle_buffer.particle_bucket_sizes_virtual
						, particle_buffer.blockbuckets_virtual
					);
				});
				
				cu_dev.syncStream<streamIdx::COMPUTE>();
			}
			cu_dev.syncStream<streamIdx::COMPUTE>();
			timer.tock(fmt::format("GPU[{}] step {} init_grid", gpuid, cur_step));
		}
	}
	//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
};

}// namespace mn

#endif