#ifndef GMPM_SIMULATOR_CUH
#define GMPM_SIMULATOR_CUH
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

#include "grid_buffer.cuh"
#include "hash_table.cuh"
#include "kernels.cuh"
#include "particle_buffer.cuh"
#include "settings.h"
#include "triangle_mesh.cuh"
#include "alpha_shapes.cuh"
#include "iq.cuh"
#include "managed_memory.hpp"

#define OUTPUT_TRIANGLE_SHELL_OUTER_POS 1
#define UPDATE_ALPHA_SHAPES_BEFORE_OUTPUT 1

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
		Allocator allocator;
	
		ManagedMemoryAllocator(Allocator allocator)
		: allocator(allocator){}
	
		void* allocate(std::size_t bytes) {//NOLINT(readability-convert-member-functions-to-static) Method is designed to be a non-static class member
			return allocator.allocate(bytes, 256);
		}

		void deallocate(void* p, std::size_t size) {//NOLINT(readability-convert-member-functions-to-static) Method is designed to be a non-static class member
			return allocator.deallocate(p, size);
		}
	};
	
	//Only allocates largest user and shares memory along users
	template<typename Allocator>
	struct ReusableDeviceAllocator {
		int gpuid;
		
		Allocator allocator;
		
		void* memory;
		size_t current_size;
	
		ReusableDeviceAllocator(Allocator allocator, const int gpuid)
		: allocator(allocator), gpuid(gpuid), memory(nullptr), current_size(0){}
	
		void* allocate(std::size_t bytes) {//NOLINT(readability-convert-member-functions-to-static) Method is designed to be a non-static class member

			if(bytes > current_size){
				if(memory != nullptr){
					allocator.deallocate(memory, current_size);
				}
				current_size = bytes;
			}

			if(memory == nullptr){
				memory = allocator.allocate(current_size);
			}

			return memory;
		}

		void deallocate(void* p, std::size_t size) {//NOLINT(readability-convert-member-functions-to-static) Method is designed to be a non-static class member
			(void) size;
			allocator.deallocate(memory, size);
			memory = nullptr;
			current_size = 0;
		}
	};

	template<typename Allocator>
	struct Intermediates {
		void* base;
		
		Allocator allocator;
		size_t current_size;

		int* d_tmp;
		int* active_block_marks;
		int* destinations;
		int* sources;
		int* bin_sizes;
		unsigned int* particle_id_mapping_buffer;
		float* d_max_vel;
		
		Intermediates(Allocator allocator)
		: allocator(allocator), current_size(0) {}
		
		//NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic) Current c++ version does not yet support std::span
		void alloc(size_t max_block_count) {
			//NOLINTBEGIN(readability-magic-numbers) Magic numbers are variable count
			current_size = sizeof(int) * (max_block_count * 5 + config::G_MAX_PARTICLE_BIN * config::G_BIN_CAPACITY + 1);
			base = allocator.allocate(current_size);
			
			d_tmp			   = static_cast<int*>(base);
			active_block_marks = static_cast<int*>(static_cast<void*>(static_cast<char*>(base) + sizeof(int) * max_block_count));
			destinations	   = static_cast<int*>(static_cast<void*>(static_cast<char*>(base) + sizeof(int) * max_block_count * 2));
			sources			   = static_cast<int*>(static_cast<void*>(static_cast<char*>(base) + sizeof(int) * max_block_count * 3));
			bin_sizes		   = static_cast<int*>(static_cast<void*>(static_cast<char*>(base) + sizeof(int) * max_block_count * 4));
			particle_id_mapping_buffer = static_cast<unsigned int*>(static_cast<void*>(static_cast<char*>(base) + sizeof(int) * max_block_count * 5));
			d_max_vel		   = static_cast<float*>(static_cast<void*>(static_cast<char*>(base) + sizeof(int) * max_block_count * 5 + sizeof(unsigned int) * config::G_MAX_PARTICLE_BIN * config::G_BIN_CAPACITY));
			//NOLINTEND(readability-magic-numbers)
		}
		//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		void dealloc() {
			cudaDeviceSynchronize();
			allocator.deallocate(base, current_size);
			current_size = 0;
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
	std::vector<AlphaShapesParticleBuffer> alpha_shapes_particle_buffers = {};
	AlphaShapesGridBuffer alpha_shapes_grid_buffer;
	int* alpha_shapes_triangle_buffer;
	int* finalized_triangle_count;
	std::vector<std::array<int, 2>> solid_fluid_couplings = {};

	ReusableDeviceAllocator<ManagedMemoryAllocator<managed_memory_type>> reusable_allocator_alpha_shapes_transfer;

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
	int exterior_block_count;///< num blocks
	std::vector<int> bincount									 = {};
	std::vector<uint32_t> particle_counts						 = {};///< num particles
	std::vector<std::array<float, config::NUM_DIMENSIONS>> model = {};
	std::vector<vec3> vel0										 = {};
	void* alpha_shapes_point_type_transfer_device_buffer;
	void* alpha_shapes_normal_transfer_device_buffer;
	void* alpha_shapes_mean_curvature_transfer_device_buffer;
	void* alpha_shapes_gauss_curvature_transfer_device_buffer;
	std::vector<int> alpha_shapes_point_type_transfer_host_buffer = {};
	std::vector<std::array<float, config::NUM_DIMENSIONS>> alpha_shapes_normal_transfer_host_buffer = {};
	std::vector<float> alpha_shapes_mean_curvature_transfer_host_buffer = {};
	std::vector<float> alpha_shapes_gauss_curvature_transfer_host_buffer = {};
	std::vector<std::array<unsigned int, config::NUM_DIMENSIONS>> alpha_shapes_triangle_transfer_host_buffer = {};
	
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
	std::shared_ptr<gko::solver::Bicgstab<float>::Factory> iq_solver_factory;

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
		, managed_memory(CustomDeviceAllocator{gpuid}, 0)
		, managed_memory_allocator(managed_memory)
		, tmps(managed_memory_allocator)
		, cur_num_active_blocks()
		, max_vels()
		, partition_block_count()
		, neighbor_block_count()
		, exterior_block_count()
		, reusable_allocator_alpha_shapes_transfer(managed_memory_allocator, gpuid)
		, alpha_shapes_grid_buffer(managed_memory_allocator, &managed_memory)
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
		
		alpha_shapes_triangle_buffer = reinterpret_cast<int*>(managed_memory_allocator.allocate(sizeof(int) * MAX_ALPHA_SHAPE_TRIANGLES_PER_MODEL));
		finalized_triangle_count = reinterpret_cast<int*>(managed_memory_allocator.allocate(sizeof(int)));
		
		alpha_shapes_point_type_transfer_device_buffer = reusable_allocator_alpha_shapes_transfer.allocate(sizeof(int) * model.size());
		alpha_shapes_normal_transfer_device_buffer = reusable_allocator_alpha_shapes_transfer.allocate(sizeof(float) * config::NUM_DIMENSIONS * model.size());
		alpha_shapes_mean_curvature_transfer_device_buffer = reusable_allocator_alpha_shapes_transfer.allocate(sizeof(float) * model.size());
		alpha_shapes_gauss_curvature_transfer_device_buffer = reusable_allocator_alpha_shapes_transfer.allocate(sizeof(float) * model.size());

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
		iq_lhs_values = gko::array<float>(ginkgo_executor,  iq::LHS_MATRIX_TOTAL_BLOCK_COUNT* 32 * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
		
		iq_solve_velocity_result_array = gko::array<float>(ginkgo_executor, 3 * iq::SOLVE_VELOCITY_MATRIX_SIZE_Y * 32 * config::G_BLOCKVOLUME);
		iq_solve_velocity_rows = gko::array<int>(ginkgo_executor, 3 * iq::SOLVE_VELOCITY_MATRIX_SIZE_Y * 32 * config::G_BLOCKVOLUME + 1);
		iq_solve_velocity_columns = gko::array<int>(ginkgo_executor, 3 * iq::SOLVE_VELOCITY_MATRIX_TOTAL_BLOCK_COUNT * 32 * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
		iq_solve_velocity_values = gko::array<float>(ginkgo_executor,  3 * iq::SOLVE_VELOCITY_MATRIX_TOTAL_BLOCK_COUNT* 32 * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
		
		//Create IQ-Solver
		{
			const gko::remove_complex<float> tolerance = 1e-8f;
			//Maximum 100 iterations
			std::shared_ptr<gko::stop::Iteration::Factory> iter_stop = gko::share(
				gko::stop::Iteration::build()
				.with_max_iters(100u)
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
			std::shared_ptr<gko::preconditioner::Ic<gko::solver::LowerTrs<float>>::Factory> ic_gen = gko::share(
				gko::preconditioner::Ic<gko::solver::LowerTrs<float>>::build()
				.with_factorization_factory(
					gko::factorization::Ic<float, int>::build()
					.with_skip_sorting(true) //We know that our matrix is sorted
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
				.with_deterministic(true) //Ensure same result at different runs
				.with_skip_sorting (true) //We know that our matrix is sorted
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
				.with_max_levels(10u) //Max level count
				.with_min_coarse_rows(static_cast<size_t>(config::G_BLOCKVOLUME)) //Minimum number of rows; Set to Blockvolum, otherwise nothing is solved if we have only one block
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
			iq_solver_factory = gko::share(
				gko::solver::Bicgstab<float>::build()
				.with_criteria(iter_stop, tol_stop)
				.with_preconditioner(multigrid_gen)
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
		
		//Copy positions and face data to device
		cudaMemcpyAsync(triangle_mesh_transfer_device_buffers.back(), positions.data(), sizeof(float) * config::NUM_DIMENSIONS * positions.size(), cudaMemcpyDefault, cu_dev.stream_compute());
		cudaMemcpyAsync(faces_tmp, faces.data(), sizeof(unsigned int) * config::NUM_DIMENSIONS * faces.size(), cudaMemcpyDefault, cu_dev.stream_compute());
		
		cu_dev.compute_launch({(std::max(triangle_mesh_vertex_counts.back(), triangle_mesh_face_counts.back())+ config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, copy_triangle_mesh_data_to_device, triangle_meshes.back(), triangle_mesh_vertex_counts.back(), triangle_mesh_face_counts.back(), static_cast<float*>(triangle_mesh_transfer_device_buffers.back()), static_cast<unsigned int*>(faces_tmp));
		
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

		//Create particle buffers and grid blocks and reserve buckets
		for(int copyid = 0; copyid < BIN_COUNT; ++copyid) {
			particle_bins[copyid].emplace_back(ParticleBuffer<M>(managed_memory_allocator, &managed_memory, model.size() / config::G_BIN_CAPACITY + config::G_MAX_ACTIVE_BLOCK));
			match(particle_bins[copyid].back())([&](auto& particle_buffer) {
				particle_buffer.reserve_buckets(managed_memory_allocator, config::G_MAX_ACTIVE_BLOCK);
			});
			grid_blocks[copyid].emplace_back(managed_memory_allocator, &managed_memory, grid_offset.data_arr());
		}
		alpha_shapes_particle_buffers.emplace_back(AlphaShapesParticleBuffer(managed_memory_allocator, &managed_memory, model.size() / config::G_BIN_CAPACITY + config::G_MAX_ACTIVE_BLOCK));

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
		cudaMemcpyAsync(static_cast<void*>(&particles.back().val_1d(_0, 0)), model.data(), sizeof(std::array<float, config::NUM_DIMENSIONS>) * model.size(), cudaMemcpyDefault, cu_dev.stream_compute());
		cu_dev.syncStream<streamIdx::COMPUTE>();

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
			//FIXME:for(Duration current_step_time = Duration::zero(); current_step_time < seconds_per_frame; current_step_time += dt, cur_time += dt, cur_step++) {
			for(Duration current_step_time = Duration::zero(); current_step_time < Duration::zero(); current_step_time += dt, cur_time += dt, cur_step++) {
				//Calculate maximum grid velocity and update the grid velocity
				{
					auto& cu_dev = Cuda::ref_cuda_context(gpuid);

					/// check capacity
					check_capacity();

					CudaTimer timer {cu_dev.stream_compute()};
					timer.tick();

					float* d_max_vel = tmps.d_max_vel;
					//Initialize max_vel with 0
					check_cuda_errors(cudaMemsetAsync(d_max_vel, 0, sizeof(float), cu_dev.stream_compute()));

					for(int i = 0; i < get_model_count(); ++i) {
						//Update the grid velocity
						//floor(neighbor_block_count/G_NUM_GRID_BLOCKS_PER_CUDA_BLOCK); G_NUM_WARPS_PER_CUDA_BLOCK (>= G_NUM_GRID_BLOCKS_PER_CUDA_BLOCK)
						cu_dev.compute_launch({(neighbor_block_count + config::G_NUM_GRID_BLOCKS_PER_CUDA_BLOCK - 1) / config::G_NUM_GRID_BLOCKS_PER_CUDA_BLOCK, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, update_grid_velocity_query_max, static_cast<uint32_t>(neighbor_block_count), grid_blocks[0][i], partitions[rollid], dt, d_max_vel);
					}
					//Copy maximum velocity to host site
					check_cuda_errors(cudaMemcpyAsync(&max_vels, d_max_vel, sizeof(float), cudaMemcpyDefault, cu_dev.stream_compute()));

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
					
					//Alpha shapes
					for(int i = 0; i < get_model_count(); ++i) {
						//Resize particle buffers if we increased the size of active bins
						if(checked_bin_counts[i] > 0) {
							alpha_shapes_particle_buffers[i].resize(managed_memory_allocator, cur_num_active_bins[i]);
						}

						match(particle_bins[rollid][i])([this, &cu_dev, &i](const auto& particle_buffer) {
							//Clear buffer before use
							cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, clear_alpha_shapes_particle_buffer, particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), partitions[(rollid + 1) % BIN_COUNT], alpha_shapes_particle_buffers[i]);
							
							//FIXME: Does not yet work, maybe also need to reduce block dimension?
							for(unsigned int start_index = 0; start_index < partition_block_count; start_index += ALPHA_SHAPES_MAX_KERNEL_SIZE){
								LaunchConfig alpha_shapes_launch_config(0, 0);
								alpha_shapes_launch_config.dg = dim3(std::min(ALPHA_SHAPES_MAX_KERNEL_SIZE, partition_block_count - start_index) * config::G_BLOCKVOLUME);
								alpha_shapes_launch_config.db = dim3(ALPHA_SHAPES_BLOCK_SIZE, 1, 1);
								
								//partition_block_count; {config::G_BLOCKSIZE, config::G_BLOCKSIZE, config::G_BLOCKSIZE}
								cu_dev.compute_launch(std::move(alpha_shapes_launch_config), alpha_shapes, particle_buffer, partitions[(rollid + 1) % BIN_COUNT], partitions[rollid], grid_blocks[0][i], alpha_shapes_particle_buffers[i], alpha_shapes_grid_buffer, static_cast<int*>(nullptr), static_cast<int*>(nullptr), static_cast<unsigned int*>(nullptr), start_index, static_cast<int>(cur_frame));
							}
						});
					}
					
					cu_dev.syncStream<streamIdx::COMPUTE>();
					
					timer.tock(fmt::format("GPU[{}] frame {} step {} alpha_shapes", gpuid, cur_frame, cur_step));
					
					timer.tick();
					
					for(int i = 0; i < solid_fluid_couplings.size(); ++i){
						const int solid_id = solid_fluid_couplings[i][0];
						const int fluid_id = solid_fluid_couplings[i][1];
						
						//Resize and clear matrix and vectors
						iq_rhs_array.resize_and_reset(iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME);
						iq_result_array.resize_and_reset(iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME);
						
						iq_lhs_rows.resize_and_reset(iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME + 1);
						iq_lhs_columns.resize_and_reset(iq::LHS_MATRIX_TOTAL_BLOCK_COUNT * partition_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
						iq_lhs_values.resize_and_reset(iq::LHS_MATRIX_TOTAL_BLOCK_COUNT * partition_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
						
						iq_solve_velocity_result_array.resize_and_reset(3 * iq::SOLVE_VELOCITY_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME);
						
						iq_solve_velocity_rows.resize_and_reset(3 * iq::SOLVE_VELOCITY_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME + 1);
						iq_solve_velocity_columns.resize_and_reset(3 * iq::SOLVE_VELOCITY_MATRIX_TOTAL_BLOCK_COUNT * partition_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
						iq_solve_velocity_values.resize_and_reset(3 * iq::SOLVE_VELOCITY_MATRIX_TOTAL_BLOCK_COUNT * partition_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
						
						iq_rhs_array.fill(0.0f);
						iq_result_array.fill(0.0f);
						
						iq_lhs_rows.fill(0);
						iq_lhs_columns.fill(0);
						iq_lhs_values.fill(0.0f);
						
						iq_solve_velocity_result_array.fill(0.0f);
						
						iq_solve_velocity_rows.fill(0);
						iq_solve_velocity_columns.fill(0);
						iq_solve_velocity_values.fill(0.0f);
						
						//Init rows and columns
						size_t* lhs_num_blocks_per_row_host;
						size_t* solve_velocity_num_blocks_per_row_host;
						std::array<size_t, iq::LHS_MATRIX_SIZE_X>* lhs_block_offsets_per_row_host;
						std::array<size_t, iq::SOLVE_VELOCITY_MATRIX_SIZE_X>* solve_velocity_block_offsets_per_row_host;
						cudaGetSymbolAddress(reinterpret_cast<void**>(&lhs_num_blocks_per_row_host), iq::lhs_num_blocks_per_row);
						cudaGetSymbolAddress(reinterpret_cast<void**>(&solve_velocity_num_blocks_per_row_host), iq::solve_velocity_num_blocks_per_row);
						cudaGetSymbolAddress(reinterpret_cast<void**>(&lhs_block_offsets_per_row_host), iq::lhs_block_offsets_per_row);
						cudaGetSymbolAddress(reinterpret_cast<void**>(&solve_velocity_block_offsets_per_row_host), iq::solve_velocity_block_offsets_per_row);
						cu_dev.compute_launch({partition_block_count, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, iq::clear_iq_system<iq::LHS_MATRIX_SIZE_X, iq::LHS_MATRIX_SIZE_Y, iq::NUM_ROWS_PER_BLOCK, iq::NUM_COLUMNS_PER_BLOCK, 1, Partition<1>>, lhs_num_blocks_per_row_host, lhs_block_offsets_per_row_host, static_cast<uint32_t>(partition_block_count), static_cast<uint32_t>(exterior_block_count), partitions[rollid], iq_lhs_rows.get_data(), iq_lhs_columns.get_data(), iq_lhs_values.get_data());
						cu_dev.compute_launch({partition_block_count, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, iq::clear_iq_system<iq::SOLVE_VELOCITY_MATRIX_SIZE_X, iq::SOLVE_VELOCITY_MATRIX_SIZE_Y, iq::NUM_ROWS_PER_BLOCK, iq::NUM_COLUMNS_PER_BLOCK, 3, Partition<1>>, solve_velocity_num_blocks_per_row_host, solve_velocity_block_offsets_per_row_host, static_cast<uint32_t>(partition_block_count), static_cast<uint32_t>(exterior_block_count), partitions[rollid], iq_solve_velocity_rows.get_data(), iq_solve_velocity_columns.get_data(), iq_solve_velocity_values.get_data());
						
						cu_dev.syncStream<streamIdx::COMPUTE>();
						
						//Set last active row + 1 == number nonzero elements
						const int lhs_number_of_nonzeros = static_cast<int>(iq::LHS_MATRIX_TOTAL_BLOCK_COUNT * partition_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
						const int solve_velocity_number_of_nonzeros = static_cast<int>(3 * iq::SOLVE_VELOCITY_MATRIX_TOTAL_BLOCK_COUNT * partition_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
						//cudaMemsetAsync((iq_lhs_rows.get_data() + ((iq::LHS_MATRIX_SIZE_Y - 1) * exterior_block_count + partition_block_count) * config::G_BLOCKVOLUME), lhs_number_of_nonzeros, sizeof(int), cu_dev.stream_compute());
						//cudaMemsetAsync((iq_solve_velocity_rows.get_data() + 3 * ((iq::SOLVE_VELOCITY_MATRIX_SIZE_Y - 1) * exterior_block_count + partition_block_count) * config::G_BLOCKVOLUME), solve_velocity_number_of_nonzeros, sizeof(int), cu_dev.stream_compute());
						cudaMemcpyAsync((iq_lhs_rows.get_data() + ((iq::LHS_MATRIX_SIZE_Y - 1) * exterior_block_count + partition_block_count) * config::G_BLOCKVOLUME), &lhs_number_of_nonzeros, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute());
						cudaMemcpyAsync((iq_solve_velocity_rows.get_data() + 3 * ((iq::SOLVE_VELOCITY_MATRIX_SIZE_Y - 1) * exterior_block_count + partition_block_count) * config::G_BLOCKVOLUME), &solve_velocity_number_of_nonzeros, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute());
						
						cu_dev.syncStream<streamIdx::COMPUTE>();
						
						//Fill empty space in row matrix
						cu_dev.compute_launch({exterior_block_count, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, iq::fill_empty_rows<iq::LHS_MATRIX_SIZE_Y, iq::NUM_ROWS_PER_BLOCK, 1, Partition<1>>, static_cast<uint32_t>(exterior_block_count), partitions[rollid], iq_lhs_rows.get_data());
						cu_dev.compute_launch({exterior_block_count, config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE * config::G_NUM_WARPS_PER_GRID_BLOCK}, iq::fill_empty_rows<iq::SOLVE_VELOCITY_MATRIX_SIZE_Y, iq::NUM_ROWS_PER_BLOCK, 3, Partition<1>>, static_cast<uint32_t>(exterior_block_count), partitions[rollid], iq_solve_velocity_rows.get_data());
						
						cu_dev.syncStream<streamIdx::COMPUTE>();
						
						//Set last value of rows
						//FIXME: Not sure why, but memcpy does not seem to work correctly
						//cudaMemcpyAsync((iq_lhs_rows.get_data() + iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME), (iq_lhs_rows.get_data() + iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME - 1), sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute());
						//cudaMemcpyAsync((iq_solve_velocity_rows.get_data() + 3 * iq::SOLVE_VELOCITY_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME), (iq_solve_velocity_rows.get_data() + 3 * iq::SOLVE_VELOCITY_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME - 1), sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute());
						//cudaMemsetAsync((iq_lhs_rows.get_data() + iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME), lhs_number_of_nonzeros, sizeof(int), cu_dev.stream_compute());
						//cudaMemsetAsync((iq_solve_velocity_rows.get_data() + 3 * iq::SOLVE_VELOCITY_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME), solve_velocity_number_of_nonzeros, sizeof(int), cu_dev.stream_compute());
						//cudaMemcpyAsync((iq_lhs_rows.get_data() + iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME), &lhs_number_of_nonzeros, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute());
						//cudaMemcpyAsync((iq_solve_velocity_rows.get_data() + 3  * iq::SOLVE_VELOCITY_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME), &solve_velocity_number_of_nonzeros, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute());
						
						
						cu_dev.syncStream<streamIdx::COMPUTE>();
						
						//IQ-System create
						match(particle_bins[rollid][solid_id], particle_bins[rollid][fluid_id])([this, &cu_dev, &solid_id, &fluid_id](const auto& particle_buffer_solid, const auto& particle_buffer_fluid) {
							cu_dev.compute_launch({partition_block_count, iq::BLOCK_SIZE}, iq::create_iq_system, static_cast<uint32_t>(exterior_block_count), dt, particle_buffer_solid, particle_buffer_fluid, get<typename std::decay_t<decltype(particle_buffer_solid)>>(particle_bins[(rollid + 1) % BIN_COUNT][solid_id]), get<typename std::decay_t<decltype(particle_buffer_fluid)>>(particle_bins[(rollid + 1) % BIN_COUNT][fluid_id]), partitions[(rollid + 1) % BIN_COUNT], partitions[rollid], grid_blocks[0][solid_id], grid_blocks[0][fluid_id], iq_lhs_rows.get_const_data(), iq_lhs_columns.get_const_data(), iq_lhs_values.get_data(), iq_rhs_array.get_data(), iq_solve_velocity_rows.get_const_data(), iq_solve_velocity_columns.get_const_data(), iq_solve_velocity_values.get_data());
						});
						
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
						
						/*
						std::vector<int> printout_tmp0(iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME + 1);
						std::vector<int> printout_tmp1(iq::LHS_MATRIX_TOTAL_BLOCK_COUNT * partition_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
						std::vector<float> printout_tmp2(iq::LHS_MATRIX_TOTAL_BLOCK_COUNT * partition_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
						std::vector<float> printout_tmp3(iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME);
						
						cudaMemcpyAsync(printout_tmp0.data(), iq_lhs->get_const_row_ptrs(), sizeof(int) * iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME + 1, cudaMemcpyDefault, cu_dev.stream_compute());
						cudaMemcpyAsync(printout_tmp1.data(), iq_lhs->get_const_col_idxs(), sizeof(int) * iq::LHS_MATRIX_TOTAL_BLOCK_COUNT * partition_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK, cudaMemcpyDefault, cu_dev.stream_compute());
						cudaMemcpyAsync(printout_tmp2.data(), iq_lhs->get_const_values(), sizeof(float) * iq::LHS_MATRIX_TOTAL_BLOCK_COUNT * partition_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK, cudaMemcpyDefault, cu_dev.stream_compute());
						cudaMemcpyAsync(printout_tmp3.data(), iq_rhs_array.get_const_data(), sizeof(float) * iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME, cudaMemcpyDefault, cu_dev.stream_compute());
						
						cudaDeviceSynchronize();
						
						std::cout << std::endl;
						for(size_t j = 0; j < iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME + 1; ++j){
							std::cout << printout_tmp0[j] << " ";
						}
						std::cout << std::endl;
						for(size_t j = 0; j < iq::LHS_MATRIX_TOTAL_BLOCK_COUNT * partition_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK; ++j){
							std::cout << printout_tmp1[j] << " ";
						}
						std::cout << std::endl;
						
						const std::array<size_t, iq::LHS_MATRIX_SIZE_Y> tmp_num_blocks_per_row = {
							  2
							, 3
							, 3
							, 4
						};
						printout_tmp0[iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME] = lhs_number_of_nonzeros;
						for(size_t j = 0; j < iq::LHS_MATRIX_SIZE_Y; ++j){
							for(size_t block = 0; block < tmp_num_blocks_per_row[j]; ++block){
								
								for(size_t k = 0; k < exterior_block_count * config::G_BLOCKVOLUME; ++k){
									if((printout_tmp0[j * iq::NUM_ROWS_PER_BLOCK * exterior_block_count + k + 1] - printout_tmp0[j * iq::NUM_ROWS_PER_BLOCK * exterior_block_count + k]) > 0){
										for(size_t l = 0; l < iq::NUM_COLUMNS_PER_BLOCK; ++l){
											std::cout << printout_tmp2[printout_tmp0[j * iq::NUM_ROWS_PER_BLOCK * exterior_block_count + k] + block * iq::NUM_COLUMNS_PER_BLOCK + l] << " ";
										}
										std::cout << std::endl;
									}
								}
								std::cout << "##############" << std::endl;
							}
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
						std::vector<int> printout_tmp1(3 * iq::SOLVE_VELOCITY_MATRIX_TOTAL_BLOCK_COUNT * partition_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
						std::vector<float> printout_tmp2(3 * iq::SOLVE_VELOCITY_MATRIX_TOTAL_BLOCK_COUNT * partition_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
						
						cudaMemcpyAsync(printout_tmp0.data(), iq_solve_velocity->get_const_row_ptrs(), sizeof(int) * 3 * iq::SOLVE_VELOCITY_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME + 1, cudaMemcpyDefault, cu_dev.stream_compute());
						cudaMemcpyAsync(printout_tmp1.data(), iq_solve_velocity->get_const_col_idxs(), sizeof(int) * 3 * iq::SOLVE_VELOCITY_MATRIX_TOTAL_BLOCK_COUNT * partition_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK, cudaMemcpyDefault, cu_dev.stream_compute());
						cudaMemcpyAsync(printout_tmp2.data(), iq_solve_velocity->get_const_values(), sizeof(float) * 3 * iq::SOLVE_VELOCITY_MATRIX_TOTAL_BLOCK_COUNT * partition_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK, cudaMemcpyDefault, cu_dev.stream_compute());
						
						cudaDeviceSynchronize();
						
						std::cout << std::endl;
						for(size_t j = 0; j < 3 * iq::SOLVE_VELOCITY_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME + 1; ++j){
							std::cout << printout_tmp0[j] << " ";
						}
						std::cout << std::endl;
						for(size_t j = 0; j < 3 * iq::SOLVE_VELOCITY_MATRIX_TOTAL_BLOCK_COUNT * partition_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK; ++j){
							std::cout << printout_tmp1[j] << " ";
						}
						std::cout << std::endl;
						
						const std::array<size_t, iq::SOLVE_VELOCITY_MATRIX_SIZE_Y> tmp_num_blocks_per_row = {
							  2
							, 3
						};
						for(size_t j = 0; j < iq::SOLVE_VELOCITY_MATRIX_SIZE_Y; ++j){
							for(size_t block = 0; block < tmp_num_blocks_per_row[j]; ++block){
								
								for(size_t k = 0; k < exterior_block_count * config::G_BLOCKVOLUME; ++k){
									for(size_t m = 0; m < 3; ++m){
										if((printout_tmp0[j * 3 * iq::NUM_ROWS_PER_BLOCK * exterior_block_count + 3 * k + m + 1] - printout_tmp0[j * 3 * iq::NUM_ROWS_PER_BLOCK * exterior_block_count + 3 * k + m]) > 0){
											for(size_t l = 0; l < iq::NUM_COLUMNS_PER_BLOCK; ++l){
												std::cout << printout_tmp2[printout_tmp0[j * 3 * iq::NUM_ROWS_PER_BLOCK * exterior_block_count + 3 * k + m] + block * iq::NUM_COLUMNS_PER_BLOCK + l] << " ";
											}
											std::cout << std::endl;
										}
									}
								}
								std::cout << "##############" << std::endl;
							}
						}
						std::cout << std::endl;
						*/
						
						
						//IQ-System solve
						// Create solver
						ginkgo_executor->synchronize();
						std::unique_ptr<gko::solver::Bicgstab<float>> iq_solver = iq_solver_factory->generate(iq_lhs);
						ginkgo_executor->synchronize();

						// Solve system
						iq_solver->apply(iq_rhs, iq_result);
						ginkgo_executor->synchronize();
						
						//Update velocity and ghost matrix strain
						//v_s,t+1 = v_s,t + (- dt * M_s^-1 * G_s * p_g,t+1 + dt * M_s^-1 * H_s^T * h,t+1)
						//v_f,t+1 = v_f,t + (- dt * M_f^-1 * G_f * p_f,t+1 - dt * M_f^-1 * B * y,t+1 - dt * M_f^-1 * H_f^T * h,t+1)
						
						//Calculate delta_v
						iq_solve_velocity->apply(iq_result, iq_solve_velocity_result);
						
						ginkgo_executor->synchronize();
						
						//Update velocity and strain
						match(particle_bins[rollid][solid_id])([this, &cu_dev, &solid_id, &fluid_id, &iq_solve_velocity_result, &iq_result](const auto& particle_buffer_solid) {
							cu_dev.compute_launch({partition_block_count, iq::BLOCK_SIZE}, iq::update_velocity_and_strain, particle_buffer_solid, get<typename std::decay_t<decltype(particle_buffer_solid)>>(particle_bins[(rollid + 1) % BIN_COUNT][solid_id]), partitions[(rollid + 1) % BIN_COUNT], partitions[rollid], grid_blocks[0][solid_id], grid_blocks[0][fluid_id], iq_solve_velocity_result->get_const_values(), iq_solve_velocity_result->get_const_values() + 3 * exterior_block_count * config::G_BLOCKVOLUME, iq_result->get_const_values());
						});
						
						cu_dev.syncStream<streamIdx::COMPUTE>();
						
						/*
						std::vector<float> printout_tmp4(3 * iq::SOLVE_VELOCITY_MATRIX_TOTAL_BLOCK_COUNT * partition_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
						std::vector<float> printout_tmp5(iq::LHS_MATRIX_SIZE_Y * iq::SOLVE_VELOCITY_MATRIX_TOTAL_BLOCK_COUNT * partition_block_count * config::G_BLOCKVOLUME * iq::NUM_COLUMNS_PER_BLOCK);
						
						cudaMemcpyAsync(printout_tmp4.data(), iq_solve_velocity_result->get_const_values(), sizeof(float) * 3 * exterior_block_count * config::G_BLOCKVOLUME, cudaMemcpyDefault, cu_dev.stream_compute());
						cudaMemcpyAsync(printout_tmp5.data(), iq_result->get_const_values(), sizeof(float) * iq::LHS_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME, cudaMemcpyDefault, cu_dev.stream_compute());
						
						cudaDeviceSynchronize();
						
						std::cout << std::endl;
						for(size_t j = 0; j < 3 * iq::SOLVE_VELOCITY_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME; ++j){
							std::cout << printout_tmp4[j] << " ";
						}
						std::cout << std::endl;
						for(size_t k = 0; k < iq::LHS_MATRIX_SIZE_Y; ++k){
							for(size_t j = 0; j < iq::SOLVE_VELOCITY_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME; ++j){
								std::cout << printout_tmp5[k * iq::SOLVE_VELOCITY_MATRIX_SIZE_Y * exterior_block_count * config::G_BLOCKVOLUME + j] << " ";
							}
							std::cout << std::endl;
						}
						*/
						
						
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
						match(particle_bins[(rollid + 1) % BIN_COUNT][i])([this, &cu_dev](auto& particle_buffer) {
							check_cuda_errors(cudaMemsetAsync(particle_buffer.cell_particle_counts, 0, sizeof(int) * exterior_block_count * config::G_BLOCKVOLUME, cu_dev.stream_compute()));
						});
						
						//Handle collision with shell
						for(int j = 0; j < triangle_shells[rollid][i].size(); ++j) {
							match(particle_bins[rollid][i])([this, &cu_dev, &i, &j](const auto& particle_buffer) {
								//partition_block_count; G_PARTICLE_BATCH_CAPACITY
								cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, particle_shell_collision, dt, particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), triangle_meshes[j], triangle_shells[rollid][i][j], triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer, partitions[(rollid + 1) % BIN_COUNT], partitions[rollid], grid_blocks[0][i]);
							});
						}

						//Perform g2p2g
						match(particle_bins[rollid][i])([this, &cu_dev, &i](const auto& particle_buffer) {
							//partition_block_count; G_PARTICLE_BATCH_CAPACITY
							cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, g2p2g, dt, next_dt, particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), partitions[(rollid + 1) % BIN_COUNT], partitions[rollid], grid_blocks[0][i], grid_blocks[1][i]);
						});
						
						//Clear triangle_shell_grid_buffer
						//FIXME: Outcommented to save memory
						//triangle_shell_grid_buffer[(rollid + 1) % BIN_COUNT][i].reset(neighbor_block_count, cu_dev);
						
						//grid => shell: mom(t-1) => mom(t); 0 => mass(t) shell => grid: mass(t-1) => mass(t)
						for(int j = 0; j < triangle_shells[rollid][i].size(); ++j) {
							//TODO: Clear triangle shell?

							//Perform g2p
							match(particle_bins[rollid][i])([this, &cu_dev, &i, &j](const auto& particle_buffer) {
								
								//partition_block_count; G_PARTICLE_BATCH_CAPACITY
								cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, grid_to_shell, dt, next_dt, particle_buffer, triangle_shells[rollid][i][j], triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer, partitions[rollid], grid_blocks[0][i]);
							});
						}
					}
					
					cu_dev.syncStream<streamIdx::COMPUTE>();

					timer.tock(fmt::format("GPU[{}] frame {} step {} g2p2g", gpuid, cur_frame, cur_step));

					//Resize partition if we increased the size of active blocks
					//This also deletes current particle buffer meta data.
					if(checked_counts[0] > 0) {
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
							cu_dev.compute_launch({(triangle_mesh_vertex_counts[j] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, update_triangle_shell_inner, triangle_meshes[j], triangle_shells[rollid][i][j], triangle_mesh_vertex_counts[j], dt);
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
							//First clear next triangle shell
							cu_dev.compute_launch({(triangle_mesh_vertex_counts[j] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, clear_triangle_shell, triangle_shells[(rollid + 1) % BIN_COUNT][i][j], triangle_mesh_vertex_counts[j]);
						
							//Then update domain
							//cu_dev.compute_launch({(triangle_mesh_face_counts[j] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, update_triangle_shell_subdomain, triangle_meshes[j], triangle_shells[rollid][i][j], triangle_shells[(rollid + 1) % BIN_COUNT][i][j], triangle_mesh_face_counts[j]);
							cu_dev.compute_launch({(triangle_mesh_vertex_counts[j] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, update_triangle_shell_subdomain, triangle_shells[rollid][i][j], triangle_shells[(rollid + 1) % BIN_COUNT][i][j], triangle_mesh_vertex_counts[j]);
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
						cu_dev.compute_launch({(triangle_mesh_vertex_counts[i] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, update_triangle_mesh, triangle_meshes[i], triangle_mesh_vertex_counts[i], triangle_meshes[i].center.data_arr(), triangle_meshes[i].linear_velocity.data_arr(), triangle_meshes[i].rotation.data_arr(), triangle_meshes[i].angular_velocity.data_arr());
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
							match(particle_bins[rollid][i])([this, &cu_dev, &i, &j](const auto& particle_buffer) {
								//partition_block_count; G_PARTICLE_BATCH_CAPACITY
								cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, shell_to_grid, dt, next_dt, partition_block_count, particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), triangle_meshes[i], triangle_shells[rollid][i][j], triangle_shells[(rollid + 1) % BIN_COUNT][i][j], triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer, partitions[rollid], grid_blocks[1][i]);
							});
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
							//First init sizes with 0
							check_cuda_errors(cudaMemsetAsync(particle_buffer.particle_bucket_sizes, 0, sizeof(int) * (exterior_block_count + 1), cu_dev.stream_compute()));

							//exterior_block_count; G_BLOCKVOLUME
							cu_dev.compute_launch({exterior_block_count, config::G_BLOCKVOLUME}, cell_bucket_to_block, particle_buffer.cell_particle_counts, particle_buffer.cellbuckets, particle_buffer.particle_bucket_sizes, particle_buffer.blockbuckets);
							// partitions[rollid].buildParticleBuckets(cu_dev, exterior_block_count);
						});
						
						//Store triangle shell outer vertices in buckets
						for(int j = 0; j < triangle_shells[(rollid + 1) % BIN_COUNT][i].size(); ++j) {
							//First init sizes with 0
							check_cuda_errors(cudaMemsetAsync(triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes, 0, sizeof(int) * (exterior_block_count + 1), cu_dev.stream_compute()));
							check_cuda_errors(cudaMemsetAsync(triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.face_bucket_sizes, 0, sizeof(int) * (exterior_block_count + 1), cu_dev.stream_compute()));
							
							//Store triangle shell outer vertices in buckets
							cu_dev.compute_launch({(triangle_mesh_vertex_counts[j] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, store_triangle_shell_vertices_in_bucket, triangle_mesh_vertex_counts[j], triangle_shells[(rollid + 1) % BIN_COUNT][i][j], triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer, partitions[rollid], grid_blocks[0][i]);
							
							//Store faces in buckets
							cu_dev.compute_launch({(triangle_mesh_face_counts[j] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, store_triangle_shell_faces_in_bucket, triangle_mesh_face_counts[j], triangle_meshes[j], triangle_shells[(rollid + 1) % BIN_COUNT][i][j], triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer, partitions[rollid], grid_blocks[0][i]);
						}
					}

					int* active_block_marks = tmps.active_block_marks;
					int* destinations		= tmps.destinations;
					int* sources			= tmps.sources;
					//Clear marks
					check_cuda_errors(cudaMemsetAsync(active_block_marks, 0, sizeof(int) * neighbor_block_count, cu_dev.stream_compute()));

					//Mark cells that have mass bigger 0.0
					for(int i = 0; i < get_model_count(); ++i) {
						//floor(neighbor_block_count * config::G_BLOCKVOLUME/G_PARTICLE_BATCH_CAPACITY); G_PARTICLE_BATCH_CAPACITY
						cu_dev.compute_launch({(neighbor_block_count * config::G_BLOCKVOLUME + config::G_PARTICLE_BATCH_CAPACITY - 1) / config::G_PARTICLE_BATCH_CAPACITY, config::G_PARTICLE_BATCH_CAPACITY}, mark_active_grid_blocks, static_cast<uint32_t>(neighbor_block_count), grid_blocks[1][i], active_block_marks);
					}
					
					//Clear marks
					check_cuda_errors(cudaMemsetAsync(sources, 0, sizeof(int) * (exterior_block_count + 1), cu_dev.stream_compute()));

					//Mark particle buckets that have at least one particle
					for(int i = 0; i < get_model_count(); ++i) {
						match(particle_bins[(rollid + 1) % BIN_COUNT][i])([this, &cu_dev, &sources](auto& particle_buffer) {
							//floor((exterior_block_count + 1)/G_PARTICLE_BATCH_CAPACITY); G_PARTICLE_BATCH_CAPACITY
							cu_dev.compute_launch({exterior_block_count / config::G_PARTICLE_BATCH_CAPACITY + 1, config::G_PARTICLE_BATCH_CAPACITY}, mark_active_particle_blocks, exterior_block_count + 1, particle_buffer.particle_bucket_sizes, sources);
						});
						
						//Mark for triangle shell vertices
						for(int j = 0; j < triangle_shells[(rollid + 1) % BIN_COUNT][i].size(); ++j) {
							//floor((exterior_block_count + 1)/G_PARTICLE_BATCH_CAPACITY); G_PARTICLE_BATCH_CAPACITY
							cu_dev.compute_launch({exterior_block_count / config::G_PARTICLE_BATCH_CAPACITY + 1, config::G_PARTICLE_BATCH_CAPACITY}, mark_active_particle_blocks, exterior_block_count + 1, triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes, sources);
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

					//Copy block buckets and sizes from next particle buffer to current particle buffer
					for(int i = 0; i < get_model_count(); ++i) {
						match(particle_bins[(rollid + 1) % BIN_COUNT][i])([this, &cu_dev, &sources, &i](auto& particle_buffer) {
							auto& next_particle_buffer = get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[rollid][i]);
							//partition_block_count; G_PARTICLE_BATCH_CAPACITY
							cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, update_buckets, static_cast<uint32_t>(partition_block_count), static_cast<const int*>(sources), particle_buffer, next_particle_buffer);
						});
						
						for(int j = 0; j < triangle_shells[(rollid + 1) % BIN_COUNT][i].size(); ++j) {
							//partition_block_count; G_PARTICLE_BATCH_CAPACITY
							cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, update_buckets_triangle_shell, static_cast<uint32_t>(partition_block_count), static_cast<const int*>(sources), triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer, triangle_shells[rollid][i][j].particle_buffer);
						}
					}

					//Compute bin capacities, bin offsets and the summed bin size for current particle buffer
					int* bin_sizes = tmps.bin_sizes;
					for(int i = 0; i < get_model_count(); ++i) {
						match(particle_bins[rollid][i])([this, &cu_dev, &bin_sizes, &i](auto& particle_buffer) {
							//floor((exterior_block_count + 1)/G_PARTICLE_BATCH_CAPACITY); G_PARTICLE_BATCH_CAPACITY
							cu_dev.compute_launch({partition_block_count / config::G_PARTICLE_BATCH_CAPACITY + 1, config::G_PARTICLE_BATCH_CAPACITY}, compute_bin_capacity, partition_block_count + 1, static_cast<const int*>(particle_buffer.particle_bucket_sizes), bin_sizes);

							//Ensure we have enough space for new generated particles
							for(int j = 0; j < triangle_shells[(rollid + 1) % BIN_COUNT][i].size(); ++j) {
								//floor((exterior_block_count + 1)/G_PARTICLE_BATCH_CAPACITY); G_PARTICLE_BATCH_CAPACITY
								cu_dev.compute_launch({partition_block_count / config::G_PARTICLE_BATCH_CAPACITY + 1, config::G_PARTICLE_BATCH_CAPACITY}, compute_bin_capacity_shell, partition_block_count + 1, static_cast<const int*>(triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes), bin_sizes);
							}

							//Stores aggregated bin sizes in particle_buffer
							exclusive_scan(partition_block_count + 1, bin_sizes, particle_buffer.bin_offsets, cu_dev);

							//Stores last aggregated size == whole size in bincount
							check_cuda_errors(cudaMemcpyAsync(&bincount[i], particle_buffer.bin_offsets + partition_block_count, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
							cu_dev.syncStream<streamIdx::COMPUTE>();
						});
					}

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

					timer.tock(fmt::format("GPU[{}] frame {} step {} build_partition_for_grid", gpuid, cur_frame, cur_step));

					//Resize grid if necessary
					if(checked_counts[0] > 0) {
						alpha_shapes_grid_buffer.resize(managed_memory_allocator, cur_num_active_blocks);
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
						//prev_neighbor_block_count; G_BLOCKVOLUME
						cu_dev.compute_launch({prev_neighbor_block_count, config::G_BLOCKVOLUME}, copy_selected_grid_blocks, static_cast<const ivec3*>(partitions[rollid].active_keys), partitions[(rollid + 1) % BIN_COUNT], static_cast<const int*>(active_block_marks), grid_blocks[1][i], grid_blocks[0][i]);

						cu_dev.compute_launch({prev_neighbor_block_count, config::G_BLOCKVOLUME}, copy_selected_grid_blocks_triangle_shell, static_cast<const ivec3*>(partitions[rollid].active_keys), partitions[(rollid + 1) % BIN_COUNT], static_cast<const int*>(active_block_marks), triangle_shell_grid_buffer[(rollid + 1) % BIN_COUNT][i], triangle_shell_grid_buffer[rollid][i]);
					}
					
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
					//floor(partition_block_count/G_PARTICLE_BATCH_CAPACITY); G_PARTICLE_BATCH_CAPACITY
					cu_dev.compute_launch({(partition_block_count + config::G_PARTICLE_BATCH_CAPACITY - 1) / config::G_PARTICLE_BATCH_CAPACITY, config::G_PARTICLE_BATCH_CAPACITY}, register_exterior_blocks, static_cast<uint32_t>(partition_block_count), partitions[(rollid + 1) % BIN_COUNT]);

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

			//Retrieve particle count
			match(particle_bins[(rollid + 1) % BIN_COUNT][i])([this, &cu_dev, &particle_count](const auto& particle_buffer) {
				auto policy = thrust::cuda::par.on(static_cast<cudaStream_t>(cu_dev.stream_compute()));
				thrust::device_ptr<int> host_particle_bucket_sizes = thrust::device_pointer_cast(particle_buffer.particle_bucket_sizes);
				particle_count = thrust::reduce(policy, host_particle_bucket_sizes, host_particle_bucket_sizes + partition_block_count);
			});
			
			//Reallocate particle array if necessary
			if(particle_counts[i] < particle_count){
				particle_counts[i] = particle_count;
				particles[i].resize(managed_memory_allocator, sizeof(float) * config::NUM_DIMENSIONS * particle_count);
				//TODO: Resize device alpha shapes buffers
			}
			
			//Resize the output model
			model.resize(particle_count);
			alpha_shapes_point_type_transfer_host_buffer.resize(particle_count);
			alpha_shapes_normal_transfer_host_buffer.resize(particle_count);
			alpha_shapes_mean_curvature_transfer_host_buffer.resize(particle_count);
			alpha_shapes_gauss_curvature_transfer_host_buffer.resize(particle_count);
			
			fmt::print(fg(fmt::color::red), "total number of particles {}\n", particle_count);
			
			//Generate particle_id_mapping
			unsigned int* particle_id_mapping_buffer = tmps.particle_id_mapping_buffer;
			match(particle_bins[rollid][i])([this, &cu_dev, &i, &d_particle_count, &particle_id_mapping_buffer](const auto& particle_buffer) {
				//partition_block_count; G_PARTICLE_BATCH_CAPACITY
				cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, generate_particle_id_mapping, partitions[rollid], partitions[(rollid + 1) % BIN_COUNT], particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), particle_id_mapping_buffer, d_particle_count);
			});
			
			cu_dev.syncStream<streamIdx::COMPUTE>();
			
			std::cout << std::endl << "TEST0" << std::endl;
			#ifdef UPDATE_ALPHA_SHAPES_BEFORE_OUTPUT
			//Recalculate alpha shapes for current buffer state
			{
				//Alpha shapes
				//Resize particle buffers if we increased the size of active bins
				if(checked_bin_counts[i] > 0) {
					alpha_shapes_particle_buffers[i].resize(managed_memory_allocator, cur_num_active_bins[i]);
				}
				
				//Initialize finalized_triangle_count with 0
				check_cuda_errors(cudaMemsetAsync(finalized_triangle_count, 0, sizeof(int), cu_dev.stream_compute()));

				match(particle_bins[rollid][i])([this, &cu_dev, &i, &particle_id_mapping_buffer](const auto& particle_buffer) {
					//Clear buffer before use
					cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, clear_alpha_shapes_particle_buffer, particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), partitions[(rollid + 1) % BIN_COUNT], alpha_shapes_particle_buffers[i]);
					check_cuda_errors(cudaMemset(alpha_shapes_triangle_buffer, 0, sizeof(int) * MAX_ALPHA_SHAPE_TRIANGLES_PER_MODEL));
					
					//FIXME: Does not yet work, maybe also need to reduce block dimension?
					for(unsigned int start_index = 0; start_index < partition_block_count; start_index += ALPHA_SHAPES_MAX_KERNEL_SIZE){
						LaunchConfig alpha_shapes_launch_config(0, 0);
						alpha_shapes_launch_config.dg = dim3(std::min(ALPHA_SHAPES_MAX_KERNEL_SIZE, partition_block_count - start_index) * config::G_BLOCKVOLUME);
						alpha_shapes_launch_config.db = dim3(ALPHA_SHAPES_BLOCK_SIZE, 1, 1);
						
						//partition_block_count; {config::G_BLOCKSIZE, config::G_BLOCKSIZE, config::G_BLOCKSIZE}
						cu_dev.compute_launch(std::move(alpha_shapes_launch_config), alpha_shapes, particle_buffer, partitions[(rollid + 1) % BIN_COUNT], partitions[rollid], grid_blocks[0][i], alpha_shapes_particle_buffers[i], alpha_shapes_grid_buffer, alpha_shapes_triangle_buffer, finalized_triangle_count, particle_id_mapping_buffer, start_index, static_cast<int>(cur_frame));
					}
				});
			}
			
			cu_dev.syncStream<streamIdx::COMPUTE>();
			
			std::cout << std::endl << "TEST1" << std::endl;

			int finalized_triangle_count_host;
			check_cuda_errors(cudaMemcpyAsync(&finalized_triangle_count_host, finalized_triangle_count, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
			cu_dev.syncStream<streamIdx::COMPUTE>();
			alpha_shapes_triangle_transfer_host_buffer.resize(finalized_triangle_count_host);
			
			check_cuda_errors(cudaMemcpyAsync(alpha_shapes_triangle_transfer_host_buffer.data(), alpha_shapes_triangle_buffer, 3 * sizeof(int) * (finalized_triangle_count_host), cudaMemcpyDefault, cu_dev.stream_compute()));
			#endif

			//Copy particle data to output buffer
			match(particle_bins[rollid][i])([this, &cu_dev, &i, &d_particle_count, &particle_id_mapping_buffer](const auto& particle_buffer) {
				//partition_block_count; G_PARTICLE_BATCH_CAPACITY
				cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, retrieve_particle_buffer, partitions[rollid], partitions[(rollid + 1) % BIN_COUNT], particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), alpha_shapes_particle_buffers[i], particles[i], particle_id_mapping_buffer);
			});

			cu_dev.syncStream<streamIdx::COMPUTE>();
			
			//Copy the data to the output model
			check_cuda_errors(cudaMemcpyAsync(model.data(), static_cast<void*>(&particles[i].val_1d(_0, 0)), sizeof(std::array<float, config::NUM_DIMENSIONS>) * (particle_count), cudaMemcpyDefault, cu_dev.stream_compute()));
			
			{
				match(particle_bins[rollid][i])([this, &cu_dev, &i, &d_particle_count, &particle_id_mapping_buffer](const auto& particle_buffer) {
					//partition_block_count; G_PARTICLE_BATCH_CAPACITY
					cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, retrieve_particle_buffer_alpha_shapes, partitions[rollid], partitions[(rollid + 1) % BIN_COUNT], particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), alpha_shapes_particle_buffers[i], particle_id_mapping_buffer, static_cast<int*>(alpha_shapes_point_type_transfer_device_buffer), static_cast<float*>(nullptr), static_cast<float*>(nullptr), static_cast<float*>(nullptr));
				});
				
				cu_dev.syncStream<streamIdx::COMPUTE>();
				
				//Copy the data to the output model
				check_cuda_errors(cudaMemcpyAsync(alpha_shapes_point_type_transfer_host_buffer.data(), alpha_shapes_point_type_transfer_device_buffer, sizeof(int) * (particle_count), cudaMemcpyDefault, cu_dev.stream_compute()));
			}
			
			{
				match(particle_bins[rollid][i])([this, &cu_dev, &i, &d_particle_count, &particle_id_mapping_buffer](const auto& particle_buffer) {
					//partition_block_count; G_PARTICLE_BATCH_CAPACITY
					cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, retrieve_particle_buffer_alpha_shapes, partitions[rollid], partitions[(rollid + 1) % BIN_COUNT], particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), alpha_shapes_particle_buffers[i], particle_id_mapping_buffer, static_cast<int*>(nullptr), static_cast<float*>(alpha_shapes_normal_transfer_device_buffer), static_cast<float*>(nullptr), static_cast<float*>(nullptr));
				});
				
				cu_dev.syncStream<streamIdx::COMPUTE>();
				
				//Copy the data to the output model
				check_cuda_errors(cudaMemcpyAsync(alpha_shapes_normal_transfer_host_buffer.data(), alpha_shapes_normal_transfer_device_buffer, sizeof(std::array<float, config::NUM_DIMENSIONS>) * (particle_count), cudaMemcpyDefault, cu_dev.stream_compute()));
			}
			
			{
				match(particle_bins[rollid][i])([this, &cu_dev, &i, &d_particle_count, &particle_id_mapping_buffer](const auto& particle_buffer) {
					//partition_block_count; G_PARTICLE_BATCH_CAPACITY
					cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, retrieve_particle_buffer_alpha_shapes, partitions[rollid], partitions[(rollid + 1) % BIN_COUNT], particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), alpha_shapes_particle_buffers[i], particle_id_mapping_buffer, static_cast<int*>(nullptr), static_cast<float*>(nullptr), static_cast<float*>(alpha_shapes_mean_curvature_transfer_device_buffer), static_cast<float*>(nullptr));
				});
				
				cu_dev.syncStream<streamIdx::COMPUTE>();
				
				//Copy the data to the output model
				check_cuda_errors(cudaMemcpyAsync(alpha_shapes_mean_curvature_transfer_host_buffer.data(), alpha_shapes_mean_curvature_transfer_device_buffer, sizeof(float) * (particle_count), cudaMemcpyDefault, cu_dev.stream_compute()));
			}
			
			{
				match(particle_bins[rollid][i])([this, &cu_dev, &i, &d_particle_count, &particle_id_mapping_buffer](const auto& particle_buffer) {
					//partition_block_count; G_PARTICLE_BATCH_CAPACITY
					cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, retrieve_particle_buffer_alpha_shapes, partitions[rollid], partitions[(rollid + 1) % BIN_COUNT], particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), alpha_shapes_particle_buffers[i], particle_id_mapping_buffer, static_cast<int*>(nullptr), static_cast<float*>(nullptr), static_cast<float*>(nullptr), static_cast<float*>(alpha_shapes_gauss_curvature_transfer_device_buffer));
				});
				
				cu_dev.syncStream<streamIdx::COMPUTE>();
				
				//Copy the data to the output model
				check_cuda_errors(cudaMemcpyAsync(alpha_shapes_gauss_curvature_transfer_host_buffer.data(), alpha_shapes_gauss_curvature_transfer_device_buffer, sizeof(float) * (particle_count), cudaMemcpyDefault, cu_dev.stream_compute()));
			}
			
			cu_dev.syncStream<streamIdx::COMPUTE>();
			
			std::string fn = std::string {"model"} + "_id[" + std::to_string(i) + "]_frame[" + std::to_string(cur_frame) + "].bgeo";

			//Write back file
			IO::insert_job([fn, m = model, alpha_shapes_point_type = alpha_shapes_point_type_transfer_host_buffer, alpha_shapes_normal = alpha_shapes_normal_transfer_host_buffer, alpha_shapes_mean_curvature = alpha_shapes_mean_curvature_transfer_host_buffer, alpha_shapes_gauss_curvature = alpha_shapes_gauss_curvature_transfer_host_buffer]() {
				Partio::ParticlesDataMutable* parts;
				begin_write_partio(&parts, m.size());
				
				write_partio_add(m, std::string("position"), parts);
				write_partio_add(alpha_shapes_point_type, std::string("point_type"), parts);
				write_partio_add(alpha_shapes_normal, std::string("N"), parts);
				write_partio_add(alpha_shapes_mean_curvature, std::string("mean_curvature"), parts);
				write_partio_add(alpha_shapes_gauss_curvature, std::string("gauss_curvature"), parts);

				end_write_partio(fn, parts);
			});
			#ifdef UPDATE_ALPHA_SHAPES_BEFORE_OUTPUT
			//Write back alpha shapes mesh
			std::string fn_alpha_shape = std::string {"alpha_shape"} + "_id[" + std::to_string(i) + "]_frame[" + std::to_string(cur_frame) + "].obj";
			IO::insert_job([this, fn_alpha_shape, i, pos = model, faces = alpha_shapes_triangle_transfer_host_buffer]() {
				write_triangle_mesh<float, uint32_t, config::NUM_DIMENSIONS>(fn_alpha_shape, pos, faces);
			});
			#endif
		}
		timer.tock(fmt::format("GPU[{}] frame {} step {} retrieve_particles", gpuid, cur_frame, cur_step));
		
		timer.tick();

		for(int i = 0; i < triangle_meshes.size(); ++i) {
			//Copy the data to the output model
			cu_dev.compute_launch({(triangle_mesh_vertex_counts[i] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, copy_triangle_mesh_data_to_host, triangle_meshes[i], triangle_mesh_vertex_counts[i], static_cast<float*>(triangle_mesh_transfer_device_buffers[i]));
			check_cuda_errors(cudaMemcpyAsync(triangle_mesh_transfer_host_buffers[i].data(), triangle_mesh_transfer_device_buffers[i], sizeof(std::array<float, config::NUM_DIMENSIONS>) * triangle_mesh_vertex_counts[i], cudaMemcpyDefault, cu_dev.stream_compute()));
			cu_dev.syncStream<streamIdx::COMPUTE>();

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
				//Copy the data to the output model
				cu_dev.compute_launch({(triangle_mesh_vertex_counts[j] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, copy_triangle_shell_data_to_host, triangle_shells[rollid][i][j], triangle_mesh_vertex_counts[j], static_cast<float*>(triangle_mesh_transfer_device_buffers[j]));
				check_cuda_errors(cudaMemcpyAsync(triangle_mesh_transfer_host_buffers[j].data(), triangle_mesh_transfer_device_buffers[j], sizeof(std::array<float, config::NUM_DIMENSIONS>) * triangle_mesh_vertex_counts[j], cudaMemcpyDefault, cu_dev.stream_compute()));
				cu_dev.syncStream<streamIdx::COMPUTE>();

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
			
			void* init_tmp = managed_memory_allocator.allocate(sizeof(float) * 9);
			for(int i = 0; i < triangle_meshes.size(); ++i) {
				//Calculate center of mass
				vec3 center_of_mass;
				check_cuda_errors(cudaMemsetAsync(init_tmp, 0, sizeof(float) * 9, cu_dev.stream_compute()));
				cu_dev.compute_launch({(triangle_mesh_vertex_counts[i] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, calculate_center_of_mass, triangle_meshes[i], triangle_mesh_vertex_counts[i],  static_cast<float*>(init_tmp));
				check_cuda_errors(cudaMemcpyAsync(center_of_mass.data(), init_tmp, sizeof(float) * 3, cudaMemcpyDefault, cu_dev.stream_compute()));
				cu_dev.syncStream<streamIdx::COMPUTE>();
				
				triangle_meshes[i].center = center_of_mass / triangle_meshes[i].mass;
			
				//Calculate inertia
				check_cuda_errors(cudaMemsetAsync(init_tmp, 0, sizeof(float) * 9, cu_dev.stream_compute()));
				cu_dev.compute_launch({(triangle_mesh_vertex_counts[i] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, calculate_inertia_and_relative_pos, triangle_meshes[i], triangle_mesh_vertex_counts[i], triangle_meshes[i].center.data_arr(),  static_cast<float*>(init_tmp));
				
				check_cuda_errors(cudaMemcpyAsync(triangle_meshes[i].inertia.data(), init_tmp, sizeof(float) * 9, cudaMemcpyDefault, cu_dev.stream_compute()));
				
				//Calculate normals per vertex
				cu_dev.compute_launch({(triangle_mesh_face_counts[i] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, calculate_normals_and_base_area, triangle_meshes[i], triangle_mesh_face_counts[i]);
				cu_dev.compute_launch({(triangle_mesh_vertex_counts[i] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, normalize_normals, triangle_meshes[i], triangle_mesh_vertex_counts[i]);
				
				//Apply rigid body pose for timestamp 0
				triangle_meshes[i].rigid_body_update(Duration::zero(), Duration::zero());
				cu_dev.compute_launch({(triangle_mesh_vertex_counts[i] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, update_triangle_mesh, triangle_meshes[i], triangle_mesh_vertex_counts[i], triangle_meshes[i].center.data_arr(), triangle_meshes[i].linear_velocity.data_arr(), triangle_meshes[i].rotation.data_arr(), triangle_meshes[i].angular_velocity.data_arr());
			
				for(int j = 0; j < get_model_count(); ++j) {
					//Clear triangle_shell data
					cu_dev.compute_launch({(triangle_mesh_vertex_counts[i] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, init_triangle_shell, triangle_meshes[i], triangle_shells[rollid][j][i], triangle_mesh_vertex_counts[i]);
					
					//TODO: Remove this, cause initial mass is just != 0 for testing reason?
					/*
					cu_dev.compute_launch({(triangle_mesh_vertex_counts[i] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, activate_blocks_for_shell, triangle_mesh_vertex_counts[i], triangle_shells[rollid][j][i], partitions[(rollid + 1) % BIN_COUNT], grid_blocks[0][i]);
					check_cuda_errors(cudaMemsetAsync(triangle_shells[0][i][j].particle_buffer.particle_bucket_sizes, 0, sizeof(int) * (exterior_block_count + 1), cu_dev.stream_compute()));
					cu_dev.compute_launch({(triangle_mesh_vertex_counts[j] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, store_triangle_shell_vertices_in_bucket, triangle_mesh_vertex_counts[j], triangle_shells[rollid][i][j], triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer, partitions[(rollid + 1) % BIN_COUNT]);
					*/
				}
			}
			cudaDeviceSynchronize();
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
				//floor(particle_counts[i]/config::DEFAULT_CUDA_BLOCK_SIZE); config::DEFAULT_CUDA_BLOCK_SIZE
				cu_dev.compute_launch({(particle_counts[i] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, activate_blocks, particle_counts[i], particles[i], partitions[(rollid + 1) % BIN_COUNT], grid_blocks[0][i]);
			}

			//Store count of activated blocks
			check_cuda_errors(cudaMemcpyAsync(&partition_block_count, partitions[(rollid + 1) % BIN_COUNT].count, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
			timer.tock(fmt::format("GPU[{}] step {} init_table", gpuid, cur_step));

			timer.tick();
			cu_dev.reset_mem();

			//Store particle ids in block cells
			for(int i = 0; i < get_model_count(); ++i) {
				match(particle_bins[rollid][i])([this, &cu_dev, &i](auto& particle_buffer) {
					//floor(particle_counts[i]/config::DEFAULT_CUDA_BLOCK_SIZE); config::DEFAULT_CUDA_BLOCK_SIZE
					cu_dev.compute_launch({(particle_counts[i] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, build_particle_cell_buckets, particle_counts[i], particles[i], particle_buffer, partitions[(rollid + 1) % BIN_COUNT], grid_blocks[0][i]);
				});
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
					//First init sizes with 0
					check_cuda_errors(cudaMemsetAsync(particle_buffer.particle_bucket_sizes, 0, sizeof(int) * (partition_block_count + 1), cu_dev.stream_compute()));

					//partition_block_count; G_BLOCKVOLUME
					cu_dev.compute_launch({partition_block_count, config::G_BLOCKVOLUME}, initial_cell_bucket_to_block, particle_buffer.cell_particle_counts, particle_buffer.cellbuckets, particle_buffer.particle_bucket_sizes, particle_buffer.blockbuckets);
					// partitions[(rollid + 1)%BIN_COUNT].buildParticleBuckets(cu_dev, partition_block_count);
				});
			}

			//Compute bin capacities, bin offsets and the summed bin size
			//Then initializes the particle buffer
			int* bin_sizes = tmps.bin_sizes;
			for(int i = 0; i < get_model_count(); ++i) {
				match(particle_bins[rollid][i])([this, &cu_dev, &bin_sizes, &i](auto& particle_buffer) {
					//floor((partition_block_count + 1)/G_PARTICLE_BATCH_CAPACITY); G_PARTICLE_BATCH_CAPACITY
					cu_dev.compute_launch({partition_block_count / config::G_PARTICLE_BATCH_CAPACITY + 1, config::G_PARTICLE_BATCH_CAPACITY}, compute_bin_capacity, partition_block_count + 1, static_cast<const int*>(particle_buffer.particle_bucket_sizes), bin_sizes);

					//Ensure we have enough space for new generated particles
					for(int j = 0; j < triangle_shells[(rollid + 1) % BIN_COUNT][i].size(); ++j) {
						//floor((exterior_block_count + 1)/G_PARTICLE_BATCH_CAPACITY); G_PARTICLE_BATCH_CAPACITY
						cu_dev.compute_launch({partition_block_count / config::G_PARTICLE_BATCH_CAPACITY + 1, config::G_PARTICLE_BATCH_CAPACITY}, compute_bin_capacity_shell, partition_block_count + 1, static_cast<const int*>(triangle_shells[(rollid + 1) % BIN_COUNT][i][j].particle_buffer.particle_bucket_sizes), bin_sizes);
					}

					//Stores aggregated bin sizes in particle_buffer
					exclusive_scan(partition_block_count + 1, bin_sizes, particle_buffer.bin_offsets, cu_dev);

					//Stores last aggregated size == whole size in bincount
					check_cuda_errors(cudaMemcpyAsync(&bincount[i], particle_buffer.bin_offsets + partition_block_count, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
					cu_dev.syncStream<streamIdx::COMPUTE>();

					//Initialize particle buffer
					//partition_block_count; G_PARTICLE_BATCH_CAPACITY
					cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, array_to_buffer, particles[i], particle_buffer);
				});
			}

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

			//Activate blocks near active blocks, including those before that block
			//TODO: Only these with offset -1 are not already activated as neighbours
			//floor(partition_block_count/G_PARTICLE_BATCH_CAPACITY); G_PARTICLE_BATCH_CAPACITY
			cu_dev.compute_launch({(partition_block_count + config::G_PARTICLE_BATCH_CAPACITY - 1) / config::G_PARTICLE_BATCH_CAPACITY, config::G_PARTICLE_BATCH_CAPACITY}, register_exterior_blocks, static_cast<uint32_t>(partition_block_count), partitions[(rollid + 1) % BIN_COUNT]);

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

			//Copy all particle data to background particle buffer
			for(int i = 0; i < get_model_count(); ++i) {
				match(particle_bins[rollid][i])([this, &cu_dev, &i](const auto& particle_buffer) {
					// bin_offsets, particle_bucket_sizes
					particle_buffer.copy_to(get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), partition_block_count, cu_dev.stream_compute());
				});
			}
			cu_dev.syncStream<streamIdx::COMPUTE>();

			timer.tick();

			//Initialize the grid and advection buckets
			for(int i = 0; i < get_model_count(); ++i) {
				//Clear the grid
				grid_blocks[0][i].reset(neighbor_block_count, cu_dev);
				
				//Initialize mass and momentum
				//floor(particle_counts[i]/config::DEFAULT_CUDA_BLOCK_SIZE); config::DEFAULT_CUDA_BLOCK_SIZE
				cu_dev.compute_launch({(particle_counts[i] + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, rasterize, particle_counts[i], particles[i], grid_blocks[0][i], partitions[rollid], dt, get_mass(i), vel0[i].data_arr());

				//Init advection source at offset 0 of destination
				match(particle_bins[(rollid + 1) % BIN_COUNT][i])([this, &cu_dev](auto& particle_buffer) {
					//partition_block_count; G_PARTICLE_BATCH_CAPACITY
					cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, init_adv_bucket, static_cast<const int*>(particle_buffer.particle_bucket_sizes), particle_buffer.blockbuckets);
				});
			}
			cu_dev.syncStream<streamIdx::COMPUTE>();
			timer.tock(fmt::format("GPU[{}] step {} init_grid", gpuid, cur_step));
		}
	}
	//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
};

}// namespace mn

#endif