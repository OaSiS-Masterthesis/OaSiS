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
#include <array>
#include <vector>

#include "grid_buffer.cuh"
#include "hash_table.cuh"
#include "kernels.cuh"
#include "particle_buffer.cuh"
#include "settings.h"
#include "triangle_mesh.cuh"
#include "alpha_shapes.cuh"

#define OUTPUT_TRIANGLE_SHELL_OUTER_POS 1

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

	struct DeviceAllocator {			   // hide the global one
		void* allocate(std::size_t bytes) {//NOLINT(readability-convert-member-functions-to-static) Method is designed to be a non-static class member
			void* ret = nullptr;
			check_cuda_errors(cudaMalloc(&ret, bytes));
			return ret;
		}

		void deallocate(void* p, std::size_t size) {//NOLINT(readability-convert-member-functions-to-static) Method is designed to be a non-static class member
			(void) size;
			check_cuda_errors(cudaFree(p));
		}
	};

	struct Intermediates {
		void* base;

		int* d_tmp;
		int* active_block_marks;
		int* destinations;
		int* sources;
		int* bin_sizes;
		float* d_max_vel;
		//NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic) Current c++ version does not yet support std::span
		void alloc(size_t max_block_count) {
			//NOLINTBEGIN(readability-magic-numbers) Magic numbers are variable count
			check_cuda_errors(cudaMalloc(&base, sizeof(int) * (max_block_count * 5 + 1)));

			d_tmp			   = static_cast<int*>(base);
			active_block_marks = static_cast<int*>(static_cast<void*>(static_cast<char*>(base) + sizeof(int) * max_block_count));
			destinations	   = static_cast<int*>(static_cast<void*>(static_cast<char*>(base) + sizeof(int) * max_block_count * 2));
			sources			   = static_cast<int*>(static_cast<void*>(static_cast<char*>(base) + sizeof(int) * max_block_count * 3));
			bin_sizes		   = static_cast<int*>(static_cast<void*>(static_cast<char*>(base) + sizeof(int) * max_block_count * 4));
			d_max_vel		   = static_cast<float*>(static_cast<void*>(static_cast<char*>(base) + sizeof(int) * max_block_count * 5));
			//NOLINTEND(readability-magic-numbers)
		}
		//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		void dealloc() const {
			cudaDeviceSynchronize();
			check_cuda_errors(cudaFree(base));
		}
		void resize(size_t max_block_count) {
			dealloc();
			alloc(max_block_count);
		}
	};

	///
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

	Intermediates tmps;

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
	std::vector<void*> alpha_shapes_point_type_transfer_device_buffers = {};
	std::vector<void*> alpha_shapes_normal_transfer_device_buffers = {};
	std::vector<void*> alpha_shapes_mean_curvature_transfer_device_buffers = {};
	std::vector<void*> alpha_shapes_gauss_curvature_transfer_device_buffers = {};
	std::vector<int> alpha_shapes_point_type_transfer_host_buffer = {};
	std::vector<std::array<float, config::NUM_DIMENSIONS>> alpha_shapes_normal_transfer_host_buffer = {};
	std::vector<float> alpha_shapes_mean_curvature_transfer_host_buffer = {};
	std::vector<float> alpha_shapes_gauss_curvature_transfer_host_buffer = {};
	
	std::vector<uint32_t> triangle_mesh_vertex_counts = {};
	std::vector<uint32_t> triangle_mesh_face_counts = {};
	std::vector<TriangleMesh> triangle_meshes = {};
	std::array<std::vector<std::vector<TriangleShell>>, BIN_COUNT> triangle_shells = {};
	std::vector<void*> triangle_mesh_transfer_device_buffers = {};
	std::vector<std::vector<std::array<float, config::NUM_DIMENSIONS>>> triangle_mesh_transfer_host_buffers = {};
	std::vector<std::vector<std::array<unsigned int, config::NUM_DIMENSIONS>>> triangle_mesh_face_buffers = {};
	std::array<std::vector<TriangleShellGridBuffer>, BIN_COUNT> triangle_shell_grid_buffer = {};//For mass redistribution and such

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
		, tmps()
		, cur_num_active_blocks()
		, max_vels()
		, partition_block_count()
		, neighbor_block_count()
		, exterior_block_count()
		, alpha_shapes_grid_buffer(DeviceAllocator {})		{
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

		//Create partitions
		for(int copyid = 0; copyid < BIN_COUNT; copyid++) {
			
			partitions.emplace_back(DeviceAllocator {}, config::G_MAX_ACTIVE_BLOCK);
			checked_counts[copyid] = 0;
		}

		cu_dev.syncStream<streamIdx::COMPUTE>();
		cur_num_active_blocks = config::G_MAX_ACTIVE_BLOCK;
	}
	
	void init_triangle_mesh(const std::vector<std::array<float, config::NUM_DIMENSIONS>>& positions, const std::vector<std::array<unsigned int, config::NUM_DIMENSIONS>>& faces){
		auto& cu_dev = Cuda::ref_cuda_context(gpuid);
		
		//Create buffers for triangle mesh
		triangle_meshes.emplace_back(spawn<TriangleMesh, orphan_signature>(DeviceAllocator {}, 1));
		
		//Init sizes
		triangle_mesh_vertex_counts.emplace_back(static_cast<uint32_t>(positions.size()));
		triangle_mesh_face_counts.emplace_back(static_cast<uint32_t>(faces.size()));
		
		fmt::print("init {}-th mesh with {} vectices and {} faces\n", triangle_meshes.size() - 1, positions.size(), faces.size());

		//Create transfer buffers
		triangle_mesh_transfer_device_buffers.emplace_back();
		check_cuda_errors(cudaMalloc(&triangle_mesh_transfer_device_buffers.back(), sizeof(float) * config::NUM_DIMENSIONS * positions.size()));
		triangle_mesh_transfer_host_buffers.push_back(positions);
		
		
		//Keep face data
		triangle_mesh_face_buffers.push_back(faces);
	
		//Create temporary transfer buffer
		void* faces_tmp;
		check_cuda_errors(cudaMalloc(&faces_tmp, sizeof(unsigned int) * config::NUM_DIMENSIONS * faces.size()));//TODO: MAybe we can dorectly copy into data structure
		
		//Copy positions and face data to device
		cudaMemcpyAsync(triangle_mesh_transfer_device_buffers.back(), positions.data(), sizeof(float) * config::NUM_DIMENSIONS * positions.size(), cudaMemcpyDefault, cu_dev.stream_compute());
		cudaMemcpyAsync(faces_tmp, faces.data(), sizeof(unsigned int) * config::NUM_DIMENSIONS * faces.size(), cudaMemcpyDefault, cu_dev.stream_compute());
		
		cu_dev.compute_launch({(std::max(triangle_mesh_vertex_counts.back(), triangle_mesh_face_counts.back())+ config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE}, copy_triangle_mesh_data_to_device, triangle_meshes.back(), triangle_mesh_vertex_counts.back(), triangle_mesh_face_counts.back(), static_cast<float*>(triangle_mesh_transfer_device_buffers.back()), static_cast<unsigned int*>(faces_tmp));
		
		//Free temporary transfer buffer
		cudaDeviceSynchronize();
		check_cuda_errors(cudaFree(faces_tmp));
		
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
			particle_bins[copyid].emplace_back(ParticleBuffer<M>(DeviceAllocator {}, model.size() / config::G_BIN_CAPACITY + config::G_MAX_ACTIVE_BLOCK));
			match(particle_bins[copyid].back())([&](auto& particle_buffer) {
				particle_buffer.reserve_buckets(DeviceAllocator {}, config::G_MAX_ACTIVE_BLOCK);
			});
			grid_blocks[copyid].emplace_back(DeviceAllocator {}, grid_offset.data_arr());
		}
		alpha_shapes_particle_buffers.emplace_back(AlphaShapesParticleBuffer(DeviceAllocator {}, model.size() / config::G_BIN_CAPACITY + config::G_MAX_ACTIVE_BLOCK));

		//Set initial velocity
		vel0.emplace_back();
		for(int i = 0; i < config::NUM_DIMENSIONS; ++i) {
			vel0.back()[i] = v0[i];
		}
		
		alpha_shapes_point_type_transfer_device_buffers.emplace_back();
		alpha_shapes_normal_transfer_device_buffers.emplace_back();
		alpha_shapes_mean_curvature_transfer_device_buffers.emplace_back();
		alpha_shapes_gauss_curvature_transfer_device_buffers.emplace_back();
		check_cuda_errors(cudaMalloc(&alpha_shapes_point_type_transfer_device_buffers.back(), sizeof(int) * model.size()));
		check_cuda_errors(cudaMalloc(&alpha_shapes_normal_transfer_device_buffers.back(), sizeof(float) * config::NUM_DIMENSIONS * model.size()));
		check_cuda_errors(cudaMalloc(&alpha_shapes_mean_curvature_transfer_device_buffers.back(), sizeof(float) * model.size()));
		check_cuda_errors(cudaMalloc(&alpha_shapes_gauss_curvature_transfer_device_buffers.back(), sizeof(float) * model.size()));

		//Create array for initial particles
		particles.emplace_back(spawn<particle_array_, orphan_signature>(DeviceAllocator {}, sizeof(float) * config::NUM_DIMENSIONS * model.size()));

		//Init bin counts
		cur_num_active_bins.emplace_back(config::G_MAX_PARTICLE_BIN);
		bincount.emplace_back(0);
		checked_bin_counts.emplace_back(0);
		
		//Create triangle_shell_grid
		for(int copyid = 0; copyid < BIN_COUNT; copyid++) {
			triangle_shell_grid_buffer[copyid].emplace_back(DeviceAllocator {});
		}

		//Init particle counts
		particle_counts.emplace_back(static_cast<unsigned int>(model.size()));//NOTE: Explicic narrowing cast

		fmt::print("init {}-th model with {} particles\n", particle_bins[0].size() - 1, particle_counts.back());

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
		for(int copyid = 0; copyid < BIN_COUNT; ++copyid) {
			triangle_shells[copyid].resize(get_model_count());
			for(int i = 0; i < get_model_count(); ++i) {
				for(int j = 0; j <triangle_meshes.size(); ++j) {
					triangle_shells[copyid][i].emplace_back(spawn<TriangleShell, orphan_signature>(DeviceAllocator {}, 1));
					triangle_shells[copyid][i].back().particle_buffer.reserve_buckets(DeviceAllocator {}, config::G_MAX_ACTIVE_BLOCK);
					
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
							alpha_shapes_particle_buffers[i].resize(DeviceAllocator {}, cur_num_active_bins[i]);
						}
						
						
						match(particle_bins[rollid][i])([this, &cu_dev, &i](const auto& particle_buffer) {
							//Clear buffer before use
							cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, clear_alpha_shapes_particle_buffer, particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), partitions[(rollid + 1) % BIN_COUNT], alpha_shapes_particle_buffers[i]);
							
							//FIXME: Does not yet work, maybe also need to reduce block dimension?
							for(unsigned int start_index = 0; start_index < partition_block_count; start_index += ALPHA_SHAPES_MAX_KERNEL_SIZE){
								LaunchConfig alpha_shapes_launch_config(0, 0);
								alpha_shapes_launch_config.dg = dim3(std::min(ALPHA_SHAPES_MAX_KERNEL_SIZE, partition_block_count - start_index));
								alpha_shapes_launch_config.db = dim3(config::G_BLOCKSIZE, config::G_BLOCKSIZE, config::G_BLOCKSIZE);
								
								//partition_block_count; {config::G_BLOCKSIZE, config::G_BLOCKSIZE, config::G_BLOCKSIZE}
								cu_dev.compute_launch(std::move(alpha_shapes_launch_config), alpha_shapes, particle_buffer, partitions[(rollid + 1) % BIN_COUNT], partitions[rollid], grid_blocks[0][i], alpha_shapes_particle_buffers[i], alpha_shapes_grid_buffer, start_index);
							}
						});
					}
					
					timer.tock(fmt::format("GPU[{}] frame {} step {} alpha_shapes", gpuid, cur_frame, cur_step));
				}

				/// g2p2g
				{
					auto& cu_dev = Cuda::ref_cuda_context(gpuid);
					CudaTimer timer {cu_dev.stream_compute()};

					//Resize particle buffers if we increased the size of active bins
					//This also removes all particle data of next particle buffer but does not clear it
					for(int i = 0; i < get_model_count(); ++i) {
						if(checked_bin_counts[i] > 0) {
							match(particle_bins[(rollid + 1) % BIN_COUNT][i])([this, &i](auto& particle_buffer) {
								particle_buffer.resize(DeviceAllocator {}, cur_num_active_bins[i]);
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
						triangle_shell_grid_buffer[(rollid + 1) % BIN_COUNT][i].reset(neighbor_block_count, cu_dev);
						
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
						partitions[(rollid + 1) % BIN_COUNT].resize_partition(DeviceAllocator {}, cur_num_active_blocks);
						for(int i = 0; i < get_model_count(); ++i) {
							match(particle_bins[rollid][i])([this, &cu_dev](auto& particle_buffer) {
								particle_buffer.reserve_buckets(DeviceAllocator {}, cur_num_active_blocks);
							});
							
							for(int j = 0; j < triangle_shells[rollid][i].size(); ++j) {
								triangle_shells[rollid][i][j].particle_buffer.reserve_buckets(DeviceAllocator {}, cur_num_active_blocks);
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
						for(int i = 0; i < get_model_count(); ++i) {
							grid_blocks[0][i].resize(DeviceAllocator {}, cur_num_active_blocks);
							alpha_shapes_grid_buffer.resize(DeviceAllocator {}, cur_num_active_blocks);
							triangle_shell_grid_buffer[rollid][i].resize(DeviceAllocator {}, cur_num_active_blocks);
						}
					}

					timer.tick();

					//Clear the grid
					for(int i = 0; i < get_model_count(); ++i) {
						grid_blocks[0][i].reset(exterior_block_count, cu_dev);
						triangle_shell_grid_buffer[rollid][i].reset(exterior_block_count, cu_dev);
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
							grid_blocks[1][i].resize(DeviceAllocator {}, cur_num_active_blocks);
							triangle_shell_grid_buffer[(rollid + 1) % BIN_COUNT][i].resize(DeviceAllocator {}, cur_num_active_blocks);
						}
						tmps.resize(cur_num_active_blocks);
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
				particles[i].resize(DeviceAllocator{}, sizeof(float) * config::NUM_DIMENSIONS * particle_count);
				//TODO: Resize device alpha shapes buffers
			}
			
			//Resize the output model
			model.resize(particle_count);
			alpha_shapes_point_type_transfer_host_buffer.resize(particle_count);
			alpha_shapes_normal_transfer_host_buffer.resize(particle_count);
			alpha_shapes_mean_curvature_transfer_host_buffer.resize(particle_count);
			alpha_shapes_gauss_curvature_transfer_host_buffer.resize(particle_count);
			
			fmt::print(fg(fmt::color::red), "total number of particles {}\n", particle_count);

			//Copy particle data to output buffer
			match(particle_bins[rollid][i])([this, &cu_dev, &i, &d_particle_count](const auto& particle_buffer) {
				//partition_block_count; G_PARTICLE_BATCH_CAPACITY
				cu_dev.compute_launch({partition_block_count, config::G_PARTICLE_BATCH_CAPACITY}, retrieve_particle_buffer, partitions[rollid], partitions[(rollid + 1) % BIN_COUNT], particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[(rollid + 1) % BIN_COUNT][i]), alpha_shapes_particle_buffers[i], particles[i], static_cast<int*>(alpha_shapes_point_type_transfer_device_buffers[i]), static_cast<float*>(alpha_shapes_normal_transfer_device_buffers[i]), static_cast<float*>(alpha_shapes_mean_curvature_transfer_device_buffers[i]), static_cast<float*>(alpha_shapes_gauss_curvature_transfer_device_buffers[i]), d_particle_count);
			});

			cu_dev.syncStream<streamIdx::COMPUTE>();

			//Copy the data to the output model
			check_cuda_errors(cudaMemcpyAsync(model.data(), static_cast<void*>(&particles[i].val_1d(_0, 0)), sizeof(std::array<float, config::NUM_DIMENSIONS>) * (particle_count), cudaMemcpyDefault, cu_dev.stream_compute()));
			check_cuda_errors(cudaMemcpyAsync(alpha_shapes_point_type_transfer_host_buffer.data(), alpha_shapes_point_type_transfer_device_buffers[i], sizeof(int) * (particle_count), cudaMemcpyDefault, cu_dev.stream_compute()));
			check_cuda_errors(cudaMemcpyAsync(alpha_shapes_normal_transfer_host_buffer.data(), alpha_shapes_normal_transfer_device_buffers[i], sizeof(std::array<float, config::NUM_DIMENSIONS>) * (particle_count), cudaMemcpyDefault, cu_dev.stream_compute()));
			check_cuda_errors(cudaMemcpyAsync(alpha_shapes_mean_curvature_transfer_host_buffer.data(), alpha_shapes_mean_curvature_transfer_device_buffers[i], sizeof(float) * (particle_count), cudaMemcpyDefault, cu_dev.stream_compute()));
			check_cuda_errors(cudaMemcpyAsync(alpha_shapes_gauss_curvature_transfer_host_buffer.data(), alpha_shapes_gauss_curvature_transfer_device_buffers[i], sizeof(float) * (particle_count), cudaMemcpyDefault, cu_dev.stream_compute()));
			cu_dev.syncStream<streamIdx::COMPUTE>();
			
			std::string fn = std::string {"model"} + "_id[" + std::to_string(i) + "]_frame[" + std::to_string(cur_frame) + "].bgeo";

			//Write back file
			IO::insert_job([fn, m = model, alpha_shapes_point_type = alpha_shapes_point_type_transfer_host_buffer, alpha_shapes_normal = alpha_shapes_normal_transfer_host_buffer, alpha_shapes_mean_curvature = alpha_shapes_mean_curvature_transfer_host_buffer, alpha_shapes_gauss_curvature = alpha_shapes_gauss_curvature_transfer_host_buffer]() {
				Partio::ParticlesDataMutable* parts;
				begin_write_partio(&parts, m.size());
				
				write_partio_add(m, std::string("position"), parts);
				write_partio_add(alpha_shapes_point_type, std::string("point_type"), parts);
				write_partio_add(alpha_shapes_normal, std::string("normal"), parts);
				write_partio_add(alpha_shapes_mean_curvature, std::string("mean_curvature"), parts);
				write_partio_add(alpha_shapes_gauss_curvature, std::string("gauss_curvature"), parts);

				end_write_partio(fn, parts);
			});
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
			
			void* init_tmp;
			check_cuda_errors(cudaMalloc(&init_tmp, sizeof(float) * 9));
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
			check_cuda_errors(cudaFree(init_tmp));
			
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