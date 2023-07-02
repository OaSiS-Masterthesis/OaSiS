#ifndef ALPHA_SHAPES_CUH
#define ALPHA_SHAPES_CUH

#include <MnBase/Math/Matrix/MatrixUtils.h>

#include <thrust/sort.h>

#include "particle_buffer.cuh"
#include "settings.h"
#include "utility_funcs.hpp"
#include "triangle_mesh.cuh"
#include "kernels.cuh"

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

//Alpha shapes

//TODO: Acceleration struct for fast neighbour search; Weighted alpha (by density); Different triangulation (e.g. regular);
//TODO: Large alpha spanning across more than two cells (using threads to handle several cells x^3 as a block; if spanning way too far we need more sophistocated approaches (e.g. somehow tracking the end points or using uniform grid search as proposed in some papers)
//TODO: Handle near isolated points as curves or thin surfaces instead all as drops (e.g. based on alpha radius; must happen accross boundary; after all other steps or beforhand?)
//TODO: Use threads for acceleration?
//FIXME: Maybe ensure global ordering so that different cells return the same triangulation for a subset of particles; Maybe not possible only in this functiosn (depends on initial triangles)
//FIXME: Seepline currently not working
//FIXME: Ensure curvature is correct
//FIXME: Ensure we are finalkizing the correct particles as "on triangle" particles and all of them and that the result is correct.

//NOTE: Not using CUB cause it heavily impacted compile time
namespace mn {
	
//Big enough to cover all cells near the current cell that can contain particles near enough to make a face an alpha face
constexpr size_t ALPHA_SHAPES_KERNEL_SIZE = static_cast<size_t>(const_sqrt(config::MAX_ALPHA) / config::G_DX) + 1;//NOTE:Static cast required for expression being const
constexpr size_t ALPHA_SHAPES_KERNEL_LENGTH = 2 * ALPHA_SHAPES_KERNEL_SIZE + 1;//Sidelength of the kernel cube
constexpr size_t ALPHA_SHAPES_NUMBER_OF_CELLS = ALPHA_SHAPES_KERNEL_LENGTH * ALPHA_SHAPES_KERNEL_LENGTH * ALPHA_SHAPES_KERNEL_LENGTH;

constexpr size_t ALPHA_SHAPES_MAX_PARTICLE_COUNT = ALPHA_SHAPES_NUMBER_OF_CELLS * config::G_MAX_PARTICLES_IN_CELL;
constexpr size_t ALPHA_SHAPES_MAX_TRIANGLE_COUNT = 2 * ALPHA_SHAPES_MAX_PARTICLE_COUNT - 4;//Max value by euler characteristic

constexpr unsigned int ALPHA_SHAPES_MAX_KERNEL_SIZE = config::G_MAX_ACTIVE_BLOCK;

constexpr float ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD = 1e-9;//TODO: Maybe adjust threshold
constexpr float ALPHA_SHAPES_LINE_DISTANCE_TEST_THRESHOLD = 1e-9;//TODO: Maybe adjust threshold

//NOTE: Actually this should equal the max value of particles as theoretically all particles can lie in the same circumsphere; But we are limited due to may memory bounds of local/shared memory
constexpr __device__ size_t ALPHA_SHAPES_MAX_CIRCUMSPHERE_POINTS = 100;//TODO:Maybe set to correct value
constexpr __device__ size_t ALPHA_SHAPES_MAX_CIRCUMSPHERE_TRIANGLES = 2 * (ALPHA_SHAPES_MAX_CIRCUMSPHERE_POINTS) - 4;//TODO: Correct triangle count (can actually be max_triangle_count_for_used_points - num_used_points + 1, so better find trick to avoid storing it)

constexpr __device__ size_t ALPHA_SHAPES_BLOCK_SIZE = config::CUDA_WARP_SIZE;//FIXME:config::G_NUM_WARPS_PER_CUDA_BLOCK * config::CUDA_WARP_SIZE;
constexpr __device__ size_t ALPHA_SHAPES_MAX_PARTICLE_COUNT_PER_THREAD = (ALPHA_SHAPES_MAX_PARTICLE_COUNT + ALPHA_SHAPES_BLOCK_SIZE - 1) / ALPHA_SHAPES_BLOCK_SIZE;
constexpr __device__ size_t ALPHA_SHAPES_MAX_OWN_PARTICLE_COUNT_PER_THREAD = (config::G_MAX_PARTICLES_IN_CELL + ALPHA_SHAPES_BLOCK_SIZE - 1) / ALPHA_SHAPES_BLOCK_SIZE;
constexpr __device__ size_t ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD = (ALPHA_SHAPES_MAX_TRIANGLE_COUNT + ALPHA_SHAPES_BLOCK_SIZE - 1) / ALPHA_SHAPES_BLOCK_SIZE;

using AlphaShapesBlockDomain	   = CompactDomain<char, config::G_BLOCKSIZE, config::G_BLOCKSIZE, config::G_BLOCKSIZE>;
using AlphaShapesGridBufferDomain  = CompactDomain<int, config::G_MAX_ACTIVE_BLOCK>;
using AlphaShapesParticleDomain	   = CompactDomain<int, ALPHA_SHAPES_MAX_PARTICLE_COUNT>;
using AlphaShapesTriangleDomain	   = CompactDomain<int, ALPHA_SHAPES_MAX_TRIANGLE_COUNT>;

//NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables, readability-identifier-naming) Check is buggy and reports variable errors for template arguments
using AlphaShapesParticleBufferData  = Structural<StructuralType::DENSE, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::SUM_POW2_ALIGN>, ParticleBinDomain, attrib_layout::SOA, f32_, f32_, f32_, f32_, f32_, f32_, f32_, f32_>;//Point type (integer bytes as float, needs to be casted accordingly), normal, mean_curvature, gauss_curvature ; temporary: summed_area, normal, summed_angles, summed_laplacians
//NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables, readability-identifier-naming)

struct AlphaShapesParticleBuffer : Instance<particle_buffer_<AlphaShapesParticleBufferData>> {
	using base_t							 = Instance<particle_buffer_<AlphaShapesParticleBufferData>>;

	AlphaShapesParticleBuffer() = default;

	template<typename Allocator>
	AlphaShapesParticleBuffer(Allocator allocator, std::size_t count)
		: base_t {spawn<particle_buffer_<AlphaShapesParticleBufferData>, orphan_signature>(allocator, count)}
		{}
};

//TODO: Make magic numbers to constants where suitable
//TODO: Ensure call dimensions and such are small enough to allow narrowing conversations. Or directly use unsigned where possible
//TODO: Maybe use names instead of formula signs for better understanding
//NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers, readability-identifier-naming, misc-definitions-in-headers) CUDA does not yet support std::span; Common names for physical formulas; Cannot declare __global__ functions inline

enum class AlphaShapesPointType {
	OUTER_POINT = 0,
	INNER_POINT,
	ISOLATED_POINT,
	CURVE_1D,
	CURVE_2D,
	TOTAL
};

// Shared memory
struct SharedMemoryType
{
	std::array<int, ALPHA_SHAPES_MAX_CIRCUMSPHERE_POINTS> circumsphere_points;
	std::array<int, ALPHA_SHAPES_MAX_CIRCUMSPHERE_TRIANGLES> circumsphere_triangles;
};

//Blocked
/*template<size_t BLOCK_SIZE, size_t ITEMS_PER_THREAD>
constexpr size_t alpha_shapes_get_thread_index(const size_t index){
	return index / ITEMS_PER_THREAD;
}

template<size_t BLOCK_SIZE, size_t ITEMS_PER_THREAD>
constexpr size_t alpha_shapes_get_thread_offset(const size_t index){
	return index % ITEMS_PER_THREAD;
}

template<size_t BLOCK_SIZE, size_t ITEMS_PER_THREAD>
constexpr size_t alpha_shapes_get_thread_count(const size_t thread_id, const size_t global_count){
	return std::min(ITEMS_PER_THREAD, global_count - std::min(global_count, thread_id * ITEMS_PER_THREAD));
}

template<size_t BLOCK_SIZE, size_t ITEMS_PER_THREAD>
constexpr size_t alpha_shapes_get_global_index(const size_t thread_id, const size_t offset){
	return thread_id * ITEMS_PER_THREAD + offset;
}*/

//Sliced
template<size_t BLOCK_SIZE, size_t ITEMS_PER_THREAD>
constexpr size_t alpha_shapes_get_thread_index(const size_t index){
	return index % BLOCK_SIZE;
}

template<size_t BLOCK_SIZE, size_t ITEMS_PER_THREAD>
constexpr size_t alpha_shapes_get_thread_offset(const size_t index){
	if(index / BLOCK_SIZE >= ITEMS_PER_THREAD){
		printf("FAILURE0 %d %d %d\n", static_cast<int>(index), static_cast<int>(BLOCK_SIZE), static_cast<int>(ITEMS_PER_THREAD));
	}
	return index / BLOCK_SIZE;
}

template<size_t BLOCK_SIZE, size_t ITEMS_PER_THREAD>
constexpr size_t alpha_shapes_get_thread_count(const size_t thread_id, const size_t global_count){
	if(((global_count / BLOCK_SIZE) + (global_count % BLOCK_SIZE > thread_id ? 1 : 0)) > ITEMS_PER_THREAD){
		printf("FAILURE1 %d %d %d %d %d %d %d\n", static_cast<int>(thread_id), static_cast<int>(global_count), static_cast<int>(BLOCK_SIZE), static_cast<int>(ITEMS_PER_THREAD), static_cast<int>(global_count / BLOCK_SIZE), static_cast<int>(global_count % BLOCK_SIZE), static_cast<int>((global_count / BLOCK_SIZE) + (global_count % BLOCK_SIZE > thread_id ? 1 : 0)));
	}
	return (global_count / BLOCK_SIZE) + (global_count % BLOCK_SIZE > thread_id ? 1 : 0);
}

template<size_t BLOCK_SIZE, size_t ITEMS_PER_THREAD>
constexpr size_t alpha_shapes_get_global_index(const size_t thread_id, const size_t offset){
	if(offset * BLOCK_SIZE + thread_id > BLOCK_SIZE * ITEMS_PER_THREAD){
		printf("FAILURE2 %d %d %d %d\n", static_cast<int>(thread_id), static_cast<int>(offset), static_cast<int>(BLOCK_SIZE), static_cast<int>(ITEMS_PER_THREAD));
	}
	
	return offset * BLOCK_SIZE + thread_id;
}

template<typename T, size_t BLOCK_SIZE, size_t ITEMS_PER_THREAD>
__forceinline__ __device__ void alpha_shapes_spread_data(T (&data)[ITEMS_PER_THREAD]){
	constexpr size_t DATA_SIZE = BLOCK_SIZE * ITEMS_PER_THREAD;
	__shared__ T shmem[DATA_SIZE];
	
	for(size_t i = 0; i < ITEMS_PER_THREAD; ++i){
		shmem[alpha_shapes_get_global_index<ALPHA_SHAPES_BLOCK_SIZE, ITEMS_PER_THREAD>(threadIdx.x, i)] = data[i];
	}
	__syncthreads();
	for(size_t i = 0; i < ITEMS_PER_THREAD; ++i){
		data[i] = shmem[threadIdx.x * ITEMS_PER_THREAD + i];
	}
}

template<typename T, size_t ITEMS_PER_THREAD>
__forceinline__ __device__ void alpha_shapes_block_swap(T (&data)[ITEMS_PER_THREAD], const size_t a, const size_t b){
	__shared__ T tmp[2];
	if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ITEMS_PER_THREAD>(a) == threadIdx.x){
		tmp[0] = data[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ITEMS_PER_THREAD>(a)];
	}
	if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ITEMS_PER_THREAD>(b) == threadIdx.x){
		tmp[1] = data[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ITEMS_PER_THREAD>(b)];
	}
	
	__syncthreads();
	
	if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ITEMS_PER_THREAD>(a) == threadIdx.x){
		data[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ITEMS_PER_THREAD>(a)] = tmp[1];
	}
	if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ITEMS_PER_THREAD>(b) == threadIdx.x){
		data[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ITEMS_PER_THREAD>(b)] = tmp[0];
	}
}

template<typename T, size_t BLOCK_SIZE, size_t ITEMS_PER_THREAD, typename ReductionOp>
__forceinline__ __device__ T alpha_shapes_reduce(const T (&data)[ITEMS_PER_THREAD], ReductionOp f){
	constexpr size_t NUM_WARPS = BLOCK_SIZE / config::CUDA_WARP_SIZE;
	
	__shared__ T shmem[NUM_WARPS];
	
	T last = data[0];
	for(int i = 1; i < ITEMS_PER_THREAD; ++i){
		last = f(last, data[i]);
	}
	
	for(int iter = 1; iter % config::CUDA_WARP_SIZE; iter <<= 1) {
		const T tmp = __shfl_down_sync(0xffffffff, last, iter, config::CUDA_WARP_SIZE);
		if((threadIdx.x % config::CUDA_WARP_SIZE) + iter < config::CUDA_WARP_SIZE) {
			last = f(last, tmp);
		}
	}
	
	shmem[threadIdx.x / config::CUDA_WARP_SIZE] = last;
	
	for(int interval = NUM_WARPS >> 1; interval > 0; interval >>= 1) {
		if(threadIdx.x < interval) {
			shmem[threadIdx.x] = f(shmem[threadIdx.x], shmem[static_cast<int>(threadIdx.x) + interval]);
		}
		__syncthreads();
	}
	
	return shmem[0];
}

template<typename T, size_t BLOCK_SIZE, size_t ITEMS_PER_THREAD, typename SortOp>
__forceinline__ __device__ void alpha_shapes_merge_sort(T (&data)[ITEMS_PER_THREAD], SortOp f){
	constexpr size_t DATA_SIZE = BLOCK_SIZE * ITEMS_PER_THREAD;
	
	__shared__ T shmem[DATA_SIZE];
	
	//Sort locally
	thrust::sort(thrust::seq, &(data[0]), &(data[0]) + ITEMS_PER_THREAD, f);
	
	//Copy to shared mem
	thrust::copy(thrust::seq, &(data[0]), &(data[0]) + ITEMS_PER_THREAD, &(shmem[threadIdx.x * ITEMS_PER_THREAD]));
	
	//TODO: Optimize
	//Merge
	for(int merge_size = 2; merge_size < BLOCK_SIZE; merge_size <<= 1){
		if(threadIdx.x % merge_size == 0){
			const int merged_items_count = std::min(merge_size * ITEMS_PER_THREAD, DATA_SIZE - threadIdx.x * ITEMS_PER_THREAD);
			
			//Copied from https://www.geeksforgeeks.org/in-place-merge-sort/
			for(int gap = (merged_items_count + 1) / 2; gap > 0; gap = (gap == 1 ? 0 : (gap + 1) / 2)){
				for(int i = 0; (i + gap) < merged_items_count; i++){
					const int j = i + gap;
					if(f(shmem[threadIdx.x * ITEMS_PER_THREAD + j], shmem[threadIdx.x * ITEMS_PER_THREAD + i])){
						thrust::swap(shmem[threadIdx.x * ITEMS_PER_THREAD + i], shmem[threadIdx.x * ITEMS_PER_THREAD + j]);
					}
				}
			}
		}
		__syncthreads();
	}
	
	//Copy back results
	thrust::copy(thrust::seq, &(shmem[threadIdx.x * ITEMS_PER_THREAD]), &(shmem[threadIdx.x * ITEMS_PER_THREAD]) + ITEMS_PER_THREAD, &(data[0]));
}

template<size_t ITEMS_PER_THREAD, MaterialE MaterialType>
__forceinline__ __device__ void alpha_shapes_fetch_particles(const ParticleBuffer<MaterialType> particle_buffer, int (&particle_indices)[ITEMS_PER_THREAD], const int particles_in_cell, const int src_blockno, const ivec3 blockid_offset, const int cellno, const int start_index){
	for(int particle_id = 0; particle_id < particles_in_cell; particle_id++) {
		if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ITEMS_PER_THREAD>(start_index + particle_id) == threadIdx.x){
			//Fetch index of the advection source
			ivec3 offset;
			int source_pidib;
			{
				//Fetch advection (direction at high bits, particle in in cell at low bits)
				const int advect = particle_buffer.cellbuckets[src_blockno * config::G_PARTICLE_NUM_PER_BLOCK + cellno * config::G_MAX_PARTICLES_IN_CELL + particle_id];
				
				//Retrieve the direction (first stripping the particle id by division)
				dir_components(advect / config::G_PARTICLE_NUM_PER_BLOCK, offset.data_arr());

				//Retrieve the particle id by AND for lower bits
				source_pidib = advect & (config::G_PARTICLE_NUM_PER_BLOCK - 1);
			}
			
			//Calculate offset from current blockid
			const int dirtag = dir_offset((blockid_offset + offset).data_arr());
			
			particle_indices[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ITEMS_PER_THREAD>(start_index + particle_id)] = (dirtag * config::G_PARTICLE_NUM_PER_BLOCK) | source_pidib;
		}
	}
}

template<typename Partition>
__forceinline__ __device__ void alpha_shapes_fetch_id(const Partition prev_partition, const int particle_id, const ivec3 blockid, int& advection_source_blockno, int& source_pidib){
	//Fetch index of the advection source
	{
		//Fetch advection (direction at high bits, particle in in cell at low bits)
		const int advect = particle_id;

		//Retrieve the direction (first stripping the particle id by division)
		ivec3 offset;
		dir_components(advect / config::G_PARTICLE_NUM_PER_BLOCK, offset.data_arr());

		//Retrieve the particle id by AND for lower bits
		source_pidib = advect & (config::G_PARTICLE_NUM_PER_BLOCK - 1);

		//Get global index by adding blockid and offset
		const ivec3 global_advection_index = blockid + offset;

		//Get block_no from partition
		advection_source_blockno = prev_partition.query(global_advection_index);
	}
}

template<typename Partition, MaterialE MaterialType>
__forceinline__ __device__ std::array<float, 3> alpha_shapes_get_particle_position(const ParticleBuffer<MaterialType> particle_buffer, const Partition prev_partition, const int particle_id, const ivec3 blockid){
	//Fetch index of the advection source
	int advection_source_blockno;
	int source_pidib;
	alpha_shapes_fetch_id(prev_partition, particle_id, blockid, advection_source_blockno, source_pidib);
	
	FetchParticleBufferDataIntermediate fetch_particle_buffer_tmp = {};
	fetch_particle_buffer_data<MaterialType>(particle_buffer, advection_source_blockno, source_pidib, fetch_particle_buffer_tmp);
	return fetch_particle_buffer_tmp.pos;
}

__forceinline__ __host__ __device__ std::array<float, 3> alpha_shapes_calculate_triangle_normal(const std::array<std::array<float, 3>, 3>& triangle_positions){
	const std::array<vec3, 3> positions {
		  vec3(triangle_positions[0][0], triangle_positions[0][1], triangle_positions[0][2])
		, vec3(triangle_positions[1][0], triangle_positions[1][1], triangle_positions[1][2])
		, vec3(triangle_positions[2][0], triangle_positions[2][1], triangle_positions[2][2])
	};
	
	vec3 face_normal;
	vec_cross_vec_3d(face_normal.data_arr(), (positions[1] - positions[0]).data_arr(), (positions[2] - positions[0]).data_arr());
	
	const float face_normal_length = sqrt(face_normal[0] * face_normal[0] + face_normal[1] * face_normal[1] + face_normal[2] * face_normal[2]);
	
	//Normalize
	face_normal = face_normal / face_normal_length;
	
	return face_normal.data_arr();
}

__forceinline__ __device__ void alpha_shapes_get_circumcircle(const std::array<float, 3>& a, const std::array<float, 3>& b, const std::array<float, 3>& c, std::array<float, 3>& center, float& radius){
	constexpr size_t CG_STEPS = 32;
	constexpr float CG_EPSILON = 1.0f;
	 
	const mn::vec<float, 3> a_vec{a[0], a[1], a[2]};
	const mn::vec<float, 3> b_vec{b[0], b[1], b[2]};
	const mn::vec<float, 3> c_vec{c[0], c[1], c[2]};
	
	//Iteratively recalculating radius and center descending towards smaller radii
	//Actually Feasible direction approach from https://www.jstor.org/stable/pdf/170443.pdf?refreqid=excelsior%3A91bcaa67290275ae8fabb8cb0c805d21&ab_segments=&origin=&initiator=&acceptTC=1 / An algorithm for the minimax Weber problem
	float r;
	float r_a;
	float r_b;
	float r_c;
	mn::vec<float, 3> center_vec = (a_vec + b_vec + c_vec) * (1.0f / 3.0f);
	for(size_t i = 0; i < CG_STEPS; ++i){
		//Recalculate radius
		r_a = std::sqrt((center_vec - a_vec).dot(center_vec - a_vec));
		r_b = std::sqrt((center_vec - b_vec).dot(center_vec - b_vec));
		r_c = std::sqrt((center_vec - c_vec).dot(center_vec - c_vec));
		
		const float r_min = std::min(std::min(r_a, r_b), r_c);
		const float r_max = std::max(std::max(r_a, r_b), r_c);
		
		float step_size = std::numeric_limits<float>::max();
		mn::vec<float, 3> direction {0.0f, 0.0f, 0.0f};
		if(r_max - r_a <= CG_EPSILON){
			if(r_max - r_a > 0.0f){
				step_size = std::min(step_size, r_max - r_a);
			}
			direction += (a_vec - center_vec) / std::sqrt(r_a);
		}
		if(r_max - r_b <= CG_EPSILON){
			if(r_max - r_b > 0.0f){
				step_size = std::min(step_size, r_max - r_b);
			}
			direction += (b_vec - center_vec) / std::sqrt(r_b);
		}
		if(r_max - r_c <= CG_EPSILON){
			if(r_max - r_c > 0.0f){
				step_size = std::min(step_size, r_max - r_c);
			}
			direction += (c_vec - center_vec) / std::sqrt(r_c);
		}
		
		//Convergated (only r_max == r_a)
		if(step_size == std::numeric_limits<float>::max()){
			break;
		}
		
		//Recalculate center
		center_vec += direction * step_size;
	}
	
	//Recalculate radius
	r_a = std::sqrt((center_vec - a_vec).dot(center_vec - a_vec));
	r_b = std::sqrt((center_vec - b_vec).dot(center_vec - b_vec));
	r_c = std::sqrt((center_vec - c_vec).dot(center_vec - c_vec));
	
	r = (r_a + r_b + r_c) * (1.0f / 3.0f);
	//r = r_max;
	
	center = center_vec.data_arr();
	radius = r;
}

__forceinline__ __device__ void alpha_shapes_get_circumsphere(const std::array<float, 3>& a, const std::array<float, 3>& b, const std::array<float, 3>& c, const std::array<float, 3>& d, std::array<float, 3>& center, float& radius){
	constexpr size_t CG_STEPS = 32;
	constexpr float CG_EPSILON = 1.0f;
	 
	const mn::vec<float, 3> a_vec{a[0], a[1], a[2]};
	const mn::vec<float, 3> b_vec{b[0], b[1], b[2]};
	const mn::vec<float, 3> c_vec{c[0], c[1], c[2]};
	const mn::vec<float, 3> d_vec{d[0], d[1], d[2]};
	
	//Iteratively recalculating radius and center descending towards smaller radii
	//Actually Feasible direction approach from https://www.jstor.org/stable/pdf/170443.pdf?refreqid=excelsior%3A91bcaa67290275ae8fabb8cb0c805d21&ab_segments=&origin=&initiator=&acceptTC=1 / An algorithm for the minimax Weber problem
	float r;
	float r_a;
	float r_b;
	float r_c;
	float r_d;
	mn::vec<float, 3> center_vec = (a_vec + b_vec + c_vec + d_vec) * 0.25f;
	for(size_t i = 0; i < CG_STEPS; ++i){
		//Recalculate radius
		r_a = std::sqrt((center_vec - a_vec).dot(center_vec - a_vec));
		r_b = std::sqrt((center_vec - b_vec).dot(center_vec - b_vec));
		r_c = std::sqrt((center_vec - c_vec).dot(center_vec - c_vec));
		r_d = std::sqrt((center_vec - d_vec).dot(center_vec - d_vec));
		
		const float r_min = std::min(std::min(r_a, r_b), std::min(r_c, r_d));
		const float r_max = std::max(std::max(r_a, r_b), std::max(r_c, r_d));
		
		float step_size = std::numeric_limits<float>::max();
		mn::vec<float, 3> direction {0.0f, 0.0f, 0.0f};
		if(r_max - r_a <= CG_EPSILON){
			if(r_max - r_a > 0.0f){
				step_size = std::min(step_size, r_max - r_a);
			}
			direction += (a_vec - center_vec) / std::sqrt(r_a);
		}
		if(r_max - r_b <= CG_EPSILON){
			if(r_max - r_b > 0.0f){
				step_size = std::min(step_size, r_max - r_b);
			}
			direction += (b_vec - center_vec) / std::sqrt(r_b);
		}
		if(r_max - r_c <= CG_EPSILON){
			if(r_max - r_c > 0.0f){
				step_size = std::min(step_size, r_max - r_c);
			}
			direction += (c_vec - center_vec) / std::sqrt(r_c);
		}
		if(r_max - r_d <= CG_EPSILON){
			if(r_max - r_d > 0.0f){
				step_size = std::min(step_size, r_max - r_d);
			}
			direction += (d_vec - center_vec) / std::sqrt(r_d);
		}
		
		//Convergated (only r_max == r_a)
		if(step_size == std::numeric_limits<float>::max()){
			break;
		}
		
		//Recalculate center
		center_vec += direction * step_size;
		
		//TODO: Use inverse squareroot
		//center_vec = (r * ((1.0f / r_a) + (1.0f / r_b) + (1.0f / r_c) + (1.0f / r_d)) - 3.0f) * center_vec + summed - r * ((1.0f / r_a) * a_vec + (1.0f / r_b) * b_vec  + (1.0f / r_c) * c_vec  + (1.0f / r_d) * d_vec);
		//center_vec += (1.0f - (r / r_a0)) * ca + (1.0f - (r / r_b0)) * cb + (1.0f - (r / r_c0)) * cc + (1.0f - (r / r_d0)) * cd;
	}
	
	//Recalculate radius
	r_a = std::sqrt((center_vec - a_vec).dot(center_vec - a_vec));
	r_b = std::sqrt((center_vec - b_vec).dot(center_vec - b_vec));
	r_c = std::sqrt((center_vec - c_vec).dot(center_vec - c_vec));
	r_d = std::sqrt((center_vec - d_vec).dot(center_vec - d_vec));
	
	r = (r_a + r_b + r_c + r_d) * 0.25f;
	//r = r_max;
	
	center = center_vec.data_arr();
	radius = r;
}

template<typename Partition, MaterialE MaterialType>
__forceinline__ __device__ bool alpha_shapes_get_first_triangle(const ParticleBuffer<MaterialType> particle_buffer, const Partition prev_partition, SharedMemoryType* __restrict__ shared_memory_storage, int (&particle_indices)[ALPHA_SHAPES_MAX_PARTICLE_COUNT_PER_THREAD], const ivec3 blockid, std::array<int, 3>& triangle){
	//Pick first point
	
	//Transfer data to all threads
	__shared__ volatile int shared_p0;
	if(threadIdx.x == 0){
		const int p0 = particle_indices[0];
		shared_p0 = p0;
	}
	
	__threadfence_block();
	
	const std::array<float, 3> p0_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, shared_p0, blockid);
	const vec3 p0_position {p0_position_arr[0], p0_position_arr[1], p0_position_arr[2]};
	
	//Find nearest point
	//Returns -1 if no point was found
	const int p1 = alpha_shapes_reduce<int, ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_PARTICLE_COUNT_PER_THREAD>(particle_indices, [&particle_buffer, &prev_partition, &blockid, &p0_position](const int& a, const int& b){
		if(a == -1){
			return b;
		}else if(b == -1){
			return a;
		}
		
		const std::array<float, 3> particle_position_arr_a = alpha_shapes_get_particle_position(particle_buffer, prev_partition, a, blockid);
		const std::array<float, 3> particle_position_arr_b = alpha_shapes_get_particle_position(particle_buffer, prev_partition, b, blockid);
		const vec3 particle_position_a {particle_position_arr_a[0], particle_position_arr_a[1], particle_position_arr_a[2]};
		const vec3 particle_position_b {particle_position_arr_b[0], particle_position_arr_b[1], particle_position_arr_b[2]};
		
		//Skip particles with same position
		if(particle_position_arr_a[0] == p0_position[0] && particle_position_arr_a[1] == p0_position[1] && particle_position_arr_a[2] == p0_position[2] && particle_position_arr_b[0] == p0_position[0] && particle_position_arr_b[1] == p0_position[1] && particle_position_arr_b[2] == p0_position[2]){
			return -1;
		}else if(particle_position_arr_a[0] == p0_position[0] && particle_position_arr_a[1] == p0_position[1] && particle_position_arr_a[2] == p0_position[2]){
			return b;
		}else if(particle_position_arr_b[0] == p0_position[0] && particle_position_arr_b[1] == p0_position[1] && particle_position_arr_b[2] == p0_position[2]){
			return a;
		}else{
			//Calculate distance
			const vec3 diff_a = p0_position - particle_position_a;
			const vec3 diff_b = p0_position - particle_position_b;
			const float squared_distance_a = diff_a[0] * diff_a[0] + diff_a[1] * diff_a[1] + diff_a[2] * diff_a[2];
			const float squared_distance_b = diff_b[0] * diff_b[0] + diff_b[1] * diff_b[1] + diff_b[2] * diff_b[2];
			
			if(squared_distance_a < squared_distance_b){
				return a;
			}else{
				return b;
			}
		}
	});
	
	//Transfer data to all threads
	__shared__ volatile int shared_p1;
	if(threadIdx.x == 0){
		shared_p1 = p1;
	}
	
	__syncthreads();
	
	if(shared_p1 != -1){
		const std::array<float, 3> p1_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, shared_p1, blockid);
		const vec3 p1_position {p1_position_arr[0], p1_position_arr[1], p1_position_arr[2]};
		
		const vec3 normal = (p1_position - p0_position) / std::sqrt((p1_position - p0_position).dot(p1_position - p0_position));
		
		//Find smallest meridian sphere
		//Returns -1 if no point was found
		const int p2 = alpha_shapes_reduce<int, ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_PARTICLE_COUNT_PER_THREAD>(particle_indices, [&particle_buffer, &prev_partition, &blockid, &p0_position, &p1_position, &normal](const int& a, const int& b){
			if(a == -1){
				return b;
			}else if(b == -1){
				return a;
			}
			
			const std::array<float, 3> particle_position_arr_a = alpha_shapes_get_particle_position(particle_buffer, prev_partition, a, blockid);
			const std::array<float, 3> particle_position_arr_b = alpha_shapes_get_particle_position(particle_buffer, prev_partition, b, blockid);
			const vec3 particle_position_a {particle_position_arr_a[0], particle_position_arr_a[1], particle_position_arr_a[2]};
			const vec3 particle_position_b {particle_position_arr_b[0], particle_position_arr_b[1], particle_position_arr_b[2]};
			
			const vec3 diff_to_line_a = (particle_position_a - p0_position) - (normal).dot(particle_position_a - p0_position) * normal;
			const vec3 diff_to_line_b = (particle_position_b - p0_position) - (normal).dot(particle_position_b - p0_position) * normal;
			const float squared_distance_to_line_a = diff_to_line_a[0] * diff_to_line_a[0] + diff_to_line_a[1] * diff_to_line_a[1] + diff_to_line_a[2] * diff_to_line_a[2];
			const float squared_distance_to_line_b = diff_to_line_b[0] * diff_to_line_b[0] + diff_to_line_b[1] * diff_to_line_b[1] + diff_to_line_b[2] * diff_to_line_b[2];
			
			//Skip particles on same line
			if(squared_distance_to_line_a <= ALPHA_SHAPES_LINE_DISTANCE_TEST_THRESHOLD && squared_distance_to_line_b <= ALPHA_SHAPES_LINE_DISTANCE_TEST_THRESHOLD){
				return -1;
			}else if(squared_distance_to_line_a <= ALPHA_SHAPES_LINE_DISTANCE_TEST_THRESHOLD){
				return b;
			}else if(squared_distance_to_line_b <= ALPHA_SHAPES_LINE_DISTANCE_TEST_THRESHOLD){
				return a;
			}else{
				vec3 circle_center_a;
				vec3 circle_center_b;
				float circle_radius_a;
				float circle_radius_b;
				alpha_shapes_get_circumcircle(p0_position.data_arr(), p1_position.data_arr(), particle_position_a.data_arr(), circle_center_a.data_arr(), circle_radius_a);
				alpha_shapes_get_circumcircle(p0_position.data_arr(), p1_position.data_arr(), particle_position_b.data_arr(), circle_center_b.data_arr(), circle_radius_b);
				
				if(circle_radius_a < circle_radius_b){
					return a;
				}else{
					return b;
				}
			}
		});
		
		//Transfer data to all threads
		__shared__ volatile int shared_p2;
		if(threadIdx.x == 0){
			shared_p2 = p2;
		}
		
		__syncthreads();
		
		//Return indices
		if(threadIdx.x == 0){
			triangle[0] = shared_p0;
			triangle[1] = shared_p1;
			triangle[2] = shared_p2;
		}
		
		return (shared_p2 != -1);
	}else{
		return false;
	}
}

template<typename Partition, MaterialE MaterialType>
__forceinline__ __device__ void alpha_shapes_check_contact_condition(const ParticleBuffer<MaterialType> particle_buffer, const Partition prev_partition, AlphaShapesParticleBuffer alpha_shapes_particle_buffer, SharedMemoryType* __restrict__ shared_memory_storage, int (&own_particle_indices)[ALPHA_SHAPES_MAX_OWN_PARTICLE_COUNT_PER_THREAD], std::array<int, 3> (&triangles)[ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD], bool (&triangles_is_alpha)[ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD], volatile int& current_triangle_count, volatile int& next_triangle_count, volatile int& temporary_hull_triangles_count, volatile int& next_temporary_hull_triangles_count, const ivec3 blockid, const int triangle_index, const int p3_id, const bool is_alpha, const int finalize_particles_start, const int finalize_particles_end){
	const std::array<int, 3> current_triangle = triangles[triangle_index];
	
	__shared__ std::array<int, 4> contact_indices;
	__shared__ int face_contacts;
	
	if(threadIdx.x == 0){
		face_contacts = 0;
	}
	__syncthreads();
	
	//Find triangles in touch
	//NOTE: current_triangle always is in contact
	for(int contact_triangle_index = alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(0); contact_triangle_index < alpha_shapes_get_thread_count<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(threadIdx.x, current_triangle_count + next_triangle_count); contact_triangle_index++) {
		const std::array<int, 3>& contact_triangle = triangles[contact_triangle_index];
		
		//Test for face contact
		if(
			   (contact_triangle[0] == current_triangle[0] || contact_triangle[0] == current_triangle[1] || contact_triangle[0] == current_triangle[2] || contact_triangle[0] == p3_id)
			&& (contact_triangle[1] == current_triangle[0] || contact_triangle[1] == current_triangle[1] || contact_triangle[1] == current_triangle[2] || contact_triangle[1] == p3_id)
			&& (contact_triangle[2] == current_triangle[0] || contact_triangle[2] == current_triangle[1] || contact_triangle[2] == current_triangle[2] || contact_triangle[2] == p3_id)
		){
			const int index = atomicAdd(&face_contacts, 1);
			contact_indices[index] = alpha_shapes_get_global_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(threadIdx.x, contact_triangle_index);
		}
	}
	__syncthreads();
	
	int next_triangle_start = current_triangle_count;
	
	int swap_mapping_count = 0;
	std::array<std::pair<int, int>, 12> swap_mapping = {};
	
	__shared__ std::array<int, 3> triangle_counts_per_vertex;
	if(threadIdx.x == 0){
		thrust::fill(thrust::seq, triangle_counts_per_vertex.begin(), triangle_counts_per_vertex.end(), 0);
	}
	for(int contact_triangle_index = face_contacts - 1; contact_triangle_index >= 0; contact_triangle_index--) {//NOTE: Loop goes backwards to handle big indices first which allows easier swapping
		__shared__ std::array<volatile int, 3> current_contact_triangle;
		__shared__ volatile bool current_contact_triangle_is_alpha;
		
		if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(contact_indices[contact_triangle_index]) == threadIdx.x){
			thrust::copy(thrust::seq, triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(contact_indices[contact_triangle_index])].begin(), triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(contact_indices[contact_triangle_index])].end(), current_contact_triangle.begin());
			current_contact_triangle_is_alpha = triangles_is_alpha[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(contact_indices[contact_triangle_index])];
		}
		__threadfence_block();
		
		//If the new tetrahedron is in alpha and the current face is not, then the current face is a boundary face that has to be kept; Same the other way round
		if(current_contact_triangle_is_alpha != is_alpha){
			alpha_shapes_finalize_triangle(particle_buffer, prev_partition, alpha_shapes_particle_buffer, shared_memory_storage, own_particle_indices, triangles, current_triangle_count, next_triangle_count, current_contact_triangle, blockid, finalize_particles_start, finalize_particles_end);
		}
		
		//Check which vertices are contacting
		if(threadIdx.x == 0){
			for(int vertex_index = 0; vertex_index < 3; vertex_index++) {
				const int particle_index = current_contact_triangle[vertex_index];
				if(particle_index == current_triangle[0]){
					triangle_counts_per_vertex[0]++;
				}else if(particle_index == current_triangle[1]){
					triangle_counts_per_vertex[1]++;
				}else if(particle_index == current_triangle[2]){
					triangle_counts_per_vertex[2]++;
				}//particle_index == p3_id
			}
		}
		
		//Swap contact triangles to end of list to remove them
		int swap_index;
		if(contact_indices[contact_triangle_index] < current_triangle_count){
			swap_index = current_triangle_count - 1;//Swap with last active triangle
			
			if(threadIdx.x == 0){
				//Decrease triangle count
				current_triangle_count--;
			}
		}else{
			swap_index = next_triangle_start + next_triangle_count - 1;//Swap with first next triangle
			
			if(threadIdx.x == 0){
				//Decrease next triangle count
				next_triangle_count--;
			}
		}
		
		//Swap contacting triangle to the end
		alpha_shapes_block_swap<std::array<int, 3>, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(triangles, contact_indices[contact_triangle_index], swap_index);
		alpha_shapes_block_swap<bool, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(triangles_is_alpha, contact_indices[contact_triangle_index], swap_index);

		//Update mappings
		if(threadIdx.x == 0){
			int first_swap_mapping = -1;
			int second_swap_mapping = -1;
			for(int j = 0; j < swap_mapping_count; ++j){
				if(swap_mapping[j].first == contact_indices[contact_triangle_index]){
					first_swap_mapping = j;
				}
				if(swap_mapping[j].first == swap_index){
					second_swap_mapping = j;
				}
			}
			if(first_swap_mapping >= 0 && second_swap_mapping >= 0){
				thrust::swap(swap_mapping[first_swap_mapping].second, swap_mapping[second_swap_mapping].second);
			}else if(first_swap_mapping >= 0){
				swap_mapping[swap_mapping_count].first = swap_index;
				swap_mapping[swap_mapping_count].second = swap_mapping[first_swap_mapping].second;
				swap_mapping_count++;
				swap_mapping[first_swap_mapping].second = swap_index;
			}else if(second_swap_mapping >= 0){
				swap_mapping[swap_mapping_count].first = contact_indices[contact_triangle_index];
				swap_mapping[swap_mapping_count].second = swap_mapping[second_swap_mapping].second;
				swap_mapping_count++;
				swap_mapping[second_swap_mapping].second = contact_indices[contact_triangle_index];
			}else{
				swap_mapping[swap_mapping_count].first = contact_indices[contact_triangle_index];
				swap_mapping[swap_mapping_count].second = swap_index;
				swap_mapping_count++;
				if(contact_indices[contact_triangle_index] != swap_index){
					swap_mapping[swap_mapping_count].first = swap_index;
					swap_mapping[swap_mapping_count].second = contact_indices[contact_triangle_index];
					swap_mapping_count++;
				}
			}
		}
		__threadfence_block();
	}
	
	//Fill gap between current list and next list
	for(int i = 0; i < next_triangle_start - current_triangle_count; ++i) {
		alpha_shapes_block_swap<std::array<int, 3>, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(triangles, next_triangle_start - 1 - i, next_triangle_start + next_triangle_count - 1 - i);
		alpha_shapes_block_swap<bool, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(triangles_is_alpha, next_triangle_start - 1 - i, next_triangle_start + next_triangle_count - 1 - i);
		
		if(threadIdx.x == 0){
			int first_swap_mapping = -1;
			int second_swap_mapping = -1;
			for(int j = 0; j < swap_mapping_count; ++j){
				if(swap_mapping[j].first == next_triangle_start - 1 - i){
					first_swap_mapping = j;
				}
				if(swap_mapping[j].first == next_triangle_start + next_triangle_count - 1 - i){
					second_swap_mapping = j;
				}
			}
			if(first_swap_mapping >= 0 && second_swap_mapping >= 0){
				thrust::swap(swap_mapping[first_swap_mapping].second, swap_mapping[second_swap_mapping].second);
			}else if(first_swap_mapping >= 0){
				swap_mapping[swap_mapping_count].first = next_triangle_start + next_triangle_count - 1 - i;
				swap_mapping[swap_mapping_count].second = swap_mapping[first_swap_mapping].second;
				swap_mapping_count++;
				swap_mapping[first_swap_mapping].second = next_triangle_start + next_triangle_count - 1 - i;
			}
			//Other cases cannot appear, cause we are only swapping indices previously already swapped
		}
	}
	
	//Remove all contacting triangles from temporary hull
	if(threadIdx.x == 0){
		int next_temporary_hull_triangles_start = temporary_hull_triangles_count;
		for(int i = temporary_hull_triangles_count + next_temporary_hull_triangles_count - 1; i >= 0; i--) {//NOTE: Loop goes backwards to handle big indices first which allows easier swapping
			//Override swapped indices
			for(int j = 0; j < swap_mapping_count; ++j){
				//Search entry thatcontains our current value and retrieve its new index
				if(shared_memory_storage->circumsphere_triangles[i] == swap_mapping[j].second){
					shared_memory_storage->circumsphere_triangles[i] = swap_mapping[j].first;
					break;//Don't apply further swaps, otherwise we override previous swaps
				}
			}
			//If index is out of bounds it was swapped out and can be removed
			if(shared_memory_storage->circumsphere_triangles[i] >= current_triangle_count + next_triangle_count){
				int swap_index;
				if(i < temporary_hull_triangles_count){
					swap_index = temporary_hull_triangles_count - 1;//Swap with last active triangle
					
					//Decrease triangle count
					temporary_hull_triangles_count--;
				}else{
					swap_index = next_temporary_hull_triangles_start + next_temporary_hull_triangles_count - 1;//Swap with first next triangle
					
					//Decrease next triangle count
					next_temporary_hull_triangles_count--;
				}
				
				//Swap contacting triangle to the end
				thrust::swap(shared_memory_storage->circumsphere_triangles[i], shared_memory_storage->circumsphere_triangles[swap_index]);
			}
		}
	
		//Fill gap between current list and next list
		for(int i = 0; i < next_temporary_hull_triangles_start - temporary_hull_triangles_count; ++i) {
			thrust::swap(shared_memory_storage->circumsphere_triangles[next_temporary_hull_triangles_start - 1 - i], shared_memory_storage->circumsphere_triangles[next_temporary_hull_triangles_start + next_temporary_hull_triangles_count - 1 - i]);
		}
	}
	
	int next_triangle_end = current_triangle_count + next_triangle_count;
	if(threadIdx.x == 0){
		next_triangle_count += (4 - face_contacts);
		next_temporary_hull_triangles_count += (4 - face_contacts);
		
		if(current_triangle_count + next_triangle_count > static_cast<int>(ALPHA_SHAPES_MAX_TRIANGLE_COUNT)){
			printf("Too much triangles: May not be more than %d, but is %d\n", static_cast<int>(ALPHA_SHAPES_MAX_TRIANGLE_COUNT), current_triangle_count + next_triangle_count);
		}
		
		//All new added triangles are in next temporary convex hull
		for(int i = 0; i < (4 - face_contacts); ++i) {
			shared_memory_storage->circumsphere_triangles[temporary_hull_triangles_count + next_temporary_hull_triangles_count - 1 - i] = next_triangle_end + i;
		}
	}
	
	__syncthreads();
	
	//Add new triangles
	//Ensure correct order (current_triangle cw normal points outwards)
	if(face_contacts == 1){
		if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end) == threadIdx.x){
			triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end)][0] = current_triangle[0];
			triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end)][1] = current_triangle[1];
			triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end)][2] = p3_id;
			
			triangles_is_alpha[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end)] = is_alpha;
		}
		if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end + 1) == threadIdx.x){
			triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end + 1)][0] = current_triangle[1];
			triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end + 1)][1] = current_triangle[2];
			triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end + 1)][2] = p3_id;
			
			triangles_is_alpha[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end + 1)] = is_alpha;
		}
		if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end + 2) == threadIdx.x){
			triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end + 2)][0] = current_triangle[2];
			triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end + 2)][1] = current_triangle[0];
			triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end + 2)][2] = p3_id;
			
			triangles_is_alpha[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end + 2)] = is_alpha;
		}
	}else if(face_contacts == 2){
		if(triangle_counts_per_vertex[0] == 1){
			if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end) == threadIdx.x){
				triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end)][0] = current_triangle[0];
				triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end)][1] = current_triangle[1];
				triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end)][2] = p3_id;
				
				triangles_is_alpha[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end)] = is_alpha;
			}
			if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end + 1) == threadIdx.x){
				triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end + 1)][0] = current_triangle[2];
				triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end + 1)][1] = current_triangle[0];
				triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end + 1)][2] = p3_id;
				
				triangles_is_alpha[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end + 1)] = is_alpha;
			}
		}else if(triangle_counts_per_vertex[1] == 1){
			if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end) == threadIdx.x){
				triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end)][0] = current_triangle[0];
				triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end)][1] = current_triangle[1];
				triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end)][2] = p3_id;
				
				triangles_is_alpha[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end)] = is_alpha;
			}
			if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end + 1) == threadIdx.x){
				triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end + 1)][0] = current_triangle[1];
				triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end + 1)][1] = current_triangle[2];
				triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end + 1)][2] = p3_id;
				
				triangles_is_alpha[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end + 1)] = is_alpha;
			}
		}else {//triangle_counts_per_vertex[2] == 1
			if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end) == threadIdx.x){
				triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end)][0] = current_triangle[1];
				triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end)][1] = current_triangle[2];
				triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end)][2] = p3_id;
				
				triangles_is_alpha[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end)] = is_alpha;
			}
			if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end + 1) == threadIdx.x){
				triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end + 1)][0] = current_triangle[2];
				triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end + 1)][1] = current_triangle[0];
				triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end + 1)][2] = p3_id;
				
				triangles_is_alpha[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end + 1)] = is_alpha;
			}
		}
	}else if(face_contacts == 3){
		if(triangle_counts_per_vertex[0] == 3){
			if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end) == threadIdx.x){
				triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end)][0] = current_triangle[1];
				triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end)][1] = current_triangle[2];
				triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end)][2] = p3_id;
				
				triangles_is_alpha[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end)] = is_alpha;
			}
		}else if(triangle_counts_per_vertex[1] == 3){
			if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end) == threadIdx.x){
				triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end)][0] = current_triangle[2];
				triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end)][1] = current_triangle[0];
				triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end)][2] = p3_id;
				
				triangles_is_alpha[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end)] = is_alpha;
			}
		}else {//triangle_counts_per_vertex[2] == 3
			if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end) == threadIdx.x){
				triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end)][0] = current_triangle[0];
				triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end)][1] = current_triangle[1];
				triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end)][2] = p3_id;
				
				triangles_is_alpha[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(next_triangle_end)] = is_alpha;
			}
		}
	}//Otherwise nothing to do, just faces removed
}

//TODO: In this and the following functions (<- plural!) actually we do not need atomic add (just normal add). Also we need synchronization if we would use several threads and atomic
template<typename Partition, MaterialE MaterialType>
__forceinline__ __device__ void alpha_shapes_accumulate_triangle_at_vertex(const ParticleBuffer<MaterialType> particle_buffer, const Partition prev_partition, AlphaShapesParticleBuffer alpha_shapes_particle_buffer, const std::array<volatile int, 3>& triangle, const ivec3 blockid, const int contact_index){
	const float cotan_clamp_min_rad = 1.0f / std::tan(3.0f * (180.0f / static_cast<float>(M_PI)));
	const float cotan_clamp_max_rad = 1.0f / std::tan(177.0f * (180.0f / static_cast<float>(M_PI)));
	
	const std::array<float, 3> p0_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangle[0], blockid);
	const std::array<float, 3> p1_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangle[1], blockid);
	const std::array<float, 3> p2_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangle[2], blockid);
	
	const vec3 p0_position {p0_position_arr[0], p0_position_arr[1], p0_position_arr[2]};
	const vec3 p1_position {p1_position_arr[0], p1_position_arr[1], p1_position_arr[2]};
	const vec3 p2_position {p2_position_arr[0], p2_position_arr[1], p2_position_arr[2]};
	
	const std::array<vec3, 3> triangle_positions {
		  p0_position
		, p1_position
		, p2_position
	};
	
	vec3 face_normal;
	vec_cross_vec_3d(face_normal.data_arr(), (triangle_positions[1] - triangle_positions[0]).data_arr(), (triangle_positions[2] - triangle_positions[0]).data_arr());
	
	const float face_normal_length = sqrt(face_normal[0] * face_normal[0] + face_normal[1] * face_normal[1] + face_normal[2] * face_normal[2]);
	
	//Normalize
	face_normal = face_normal / face_normal_length;
	
	const float face_area = 0.5f * face_normal_length;
	
	const vec3 particle_position = triangle_positions[contact_index];
		
	int advection_source_blockno;
	int source_pidib;
	alpha_shapes_fetch_id(prev_partition, triangle[contact_index], blockid, advection_source_blockno, source_pidib);
	
	auto particle_bin													= alpha_shapes_particle_buffer.ch(_0, particle_buffer.bin_offsets[advection_source_blockno] + source_pidib / config::G_BIN_CAPACITY);
	const int particle_id_in_bin = source_pidib  % config::G_BIN_CAPACITY;
	
	float cosine = (triangle_positions[(contact_index + 1) % 3] - triangle_positions[contact_index]).dot(triangle_positions[(contact_index + 2) % 3] - triangle_positions[contact_index]) / sqrt((triangle_positions[(contact_index + 1) % 3] - triangle_positions[contact_index]).dot(triangle_positions[(contact_index + 1) % 3] - triangle_positions[contact_index]) * (triangle_positions[(contact_index + 2) % 3] - triangle_positions[contact_index]).dot(triangle_positions[(contact_index + 2) % 3] - triangle_positions[contact_index]));
	cosine = std::min(std::max(cosine, -1.0f), 1.0f);
	const float angle = std::acos(cosine);
	
	//Normal
	atomicAdd(&particle_bin.val(_1, particle_id_in_bin), angle * face_normal[0]);
	atomicAdd(&particle_bin.val(_2, particle_id_in_bin), angle * face_normal[1]);
	atomicAdd(&particle_bin.val(_3, particle_id_in_bin), angle * face_normal[2]);
	
	//Gauss curvature
	atomicAdd(&particle_bin.val(_0, particle_id_in_bin), face_area * (1.0f / 3.0f));
	atomicAdd(&particle_bin.val(_4, particle_id_in_bin), angle);
	
	//Mean curvature
	float current_cotan;
	float next_cotan;
	{
		const vec3 a = particle_position - triangle_positions[(contact_index + 2) % 3];
		const vec3 b = triangle_positions[(contact_index + 1) % 3] - triangle_positions[(contact_index + 2) % 3];
		
		vec3 cross;
		vec_cross_vec_3d(cross.data_arr(), a.data_arr(), b.data_arr());
		const float cross_norm = sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);

		current_cotan = a.dot(b) / cross_norm;
		current_cotan = std::min(std::max(current_cotan, cotan_clamp_min_rad), cotan_clamp_max_rad);
		
		next_cotan = a.dot(b) / cross_norm;
		next_cotan = std::min(std::max(next_cotan, cotan_clamp_min_rad), cotan_clamp_max_rad);
	}
	
	const vec3 laplacian = current_cotan * (triangle_positions[contact_index] - particle_position) + next_cotan * (triangle_positions[(contact_index + 1) % 3] - particle_position);
	atomicAdd(&particle_bin.val(_5, particle_id_in_bin), laplacian[0]);
	atomicAdd(&particle_bin.val(_6, particle_id_in_bin), laplacian[1]);
	atomicAdd(&particle_bin.val(_7, particle_id_in_bin), laplacian[2]);
}

template<typename Partition, MaterialE MaterialType>
__forceinline__ __device__ void alpha_shapes_finalize_triangle(const ParticleBuffer<MaterialType> particle_buffer, const Partition prev_partition, AlphaShapesParticleBuffer alpha_shapes_particle_buffer, SharedMemoryType* __restrict__ shared_memory_storage, int (&own_particle_indices)[ALPHA_SHAPES_MAX_OWN_PARTICLE_COUNT_PER_THREAD], const std::array<int, 3> (&triangles)[ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD], const volatile int& current_triangle_count, const volatile int& next_triangle_count, const std::array<volatile int, 3> triangle, const ivec3 blockid, const int range_start, const int range_end){
	const std::array<float, 3> p0_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangle[0], blockid);
	const std::array<float, 3> p1_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangle[1], blockid);
	const std::array<float, 3> p2_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangle[2], blockid);
	
	const vec3 p0_position {p0_position_arr[0], p0_position_arr[1], p0_position_arr[2]};
	const vec3 p1_position {p1_position_arr[0], p1_position_arr[1], p1_position_arr[2]};
	const vec3 p2_position {p2_position_arr[0], p2_position_arr[1], p2_position_arr[2]};
	
	const std::array<vec3, 3> triangle_positions {
		  p0_position
		, p1_position
		, p2_position
	};
	
	vec3 face_normal;
	vec_cross_vec_3d(face_normal.data_arr(), (triangle_positions[1] - triangle_positions[0]).data_arr(), (triangle_positions[2] - triangle_positions[0]).data_arr());
	
	const float face_normal_length = sqrt(face_normal[0] * face_normal[0] + face_normal[1] * face_normal[1] + face_normal[2] * face_normal[2]);
	
	//Normalize
	face_normal = face_normal / face_normal_length;
	
	const float face_area = 0.5f * face_normal_length;
	
	for(size_t contact_index = 0; contact_index < 3; ++contact_index){
		//Search for particle id
		const int found = alpha_shapes_reduce<int, ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_OWN_PARTICLE_COUNT_PER_THREAD>(own_particle_indices, [&triangle, &contact_index](const int& a, const int& b){
			if(triangle[contact_index] == a){
				return a;
			}else if(triangle[contact_index] == b){
				return b;
			}else{
				return -1;
			}
		});
		
		__syncthreads();
		
		//Only add to own active particles
		if(threadIdx.x == 0){
			if(found > 0){
				alpha_shapes_accumulate_triangle_at_vertex(particle_buffer, prev_partition, alpha_shapes_particle_buffer, triangle, blockid, contact_index);
			}
		}
	}
	
	//For all particles in current convex hull that lie near the current triangle finalize them if they only lie in convex hull due to threshold of the current triangle
	for(int particle_id = range_start; particle_id < range_end; particle_id++) {
		__shared__ volatile int current_particle_index;
		
		if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_OWN_PARTICLE_COUNT_PER_THREAD>(particle_id) == threadIdx.x){
			current_particle_index = own_particle_indices[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_OWN_PARTICLE_COUNT_PER_THREAD>(particle_id)];
		}
		__threadfence_block();
		
		int advection_source_blockno;
		int source_pidib;
		alpha_shapes_fetch_id(prev_partition, current_particle_index, blockid, advection_source_blockno, source_pidib);
		
		auto particle_bin													= alpha_shapes_particle_buffer.ch(_0, particle_buffer.bin_offsets[advection_source_blockno] + source_pidib / config::G_BIN_CAPACITY);
		const int particle_id_in_bin = source_pidib  % config::G_BIN_CAPACITY;
		
		const std::array<float, 3> particle_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, current_particle_index, blockid);
		const vec3 particle_position {particle_position_arr[0], particle_position_arr[1], particle_position_arr[2]};
		
		bool in_convex_hull = true;
		bool triangle_vertex = false;
		for(int triangle_id = alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(0); triangle_id < alpha_shapes_get_thread_count<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(threadIdx.x, current_triangle_count + next_triangle_count); ++triangle_id){
			//Skip particles at vertices of a triangle, cause they are already handled
			if(current_particle_index == triangles[triangle_id][0] || current_particle_index == triangles[triangle_id][1] || current_particle_index == triangles[triangle_id][2]){
				triangle_vertex = true;
				break;
			}
			
			const std::array<float, 3> current_p0_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[triangle_id][0], blockid);
			const std::array<float, 3> current_p1_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[triangle_id][1], blockid);
			const std::array<float, 3> current_p2_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[triangle_id][2], blockid);
			
			const vec3 current_p0_position {current_p0_position_arr[0], current_p0_position_arr[1], current_p0_position_arr[2]};
			const vec3 current_p1_position {current_p1_position_arr[0], current_p1_position_arr[1], current_p1_position_arr[2]};
			const vec3 current_p2_position {current_p2_position_arr[0], current_p2_position_arr[1], current_p2_position_arr[2]};
			
			const std::array<float, 3> current_triangle_normal = alpha_shapes_calculate_triangle_normal({
				current_p0_position_arr,
				current_p1_position_arr,
				current_p2_position_arr
			});
			const vec3 current_triangle_normal_vec {current_triangle_normal[0], current_triangle_normal[1], current_triangle_normal[2]};
							
			//Perform halfspace test
			const bool current_in_halfspace = current_triangle_normal_vec.dot(particle_position - current_p0_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
			
			if(current_in_halfspace){
				in_convex_hull = false;
				break;
			}
		}
		
		const int is_invalid = __syncthreads_or((triangle_vertex || !in_convex_hull) ? 1 : 0);
		
		const bool in_halfspace_without_threshold = face_normal.dot(particle_position - triangle_positions[0]) > 0.0f;
		
		if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_OWN_PARTICLE_COUNT_PER_THREAD>(particle_id) == threadIdx.x && (is_invalid == 0) && in_halfspace_without_threshold){
			//If point is not part of a triangle check if it lies on another triangle; This may be the case for degenerated points
					
			const vec9 barycentric_projection_matrix {
					  triangle_positions[0][0] - triangle_positions[2][0], triangle_positions[0][1] - triangle_positions[2][1], triangle_positions[0][2] - triangle_positions[2][2]
					, triangle_positions[1][0] - triangle_positions[2][0], triangle_positions[1][1] - triangle_positions[2][1], triangle_positions[1][2] - triangle_positions[2][2]
					, -face_normal[0], -face_normal[1], -face_normal[2]
			};
			
			vec3 contact_barycentric;
			solve_linear_system(barycentric_projection_matrix.data_arr(), contact_barycentric.data_arr(),  (particle_position - triangle_positions[2]).data_arr());
		
			//Calculate last barycentric coordinate
			contact_barycentric[2] = 1.0f - contact_barycentric[0] - contact_barycentric[1];
			
			//Accumulate values
			if(contact_barycentric[0] > 0.0f){//Point somewhere in triangle
				//Use face normal
				atomicAdd(&particle_bin.val(_1, particle_id_in_bin), face_normal[0]);
				atomicAdd(&particle_bin.val(_2, particle_id_in_bin), face_normal[1]);
				atomicAdd(&particle_bin.val(_3, particle_id_in_bin), face_normal[2]);
				
				//Gauss curvature
				atomicAdd(&particle_bin.val(_0, particle_id_in_bin), 1.0f);//Just ensure this is not zero
				atomicAdd(&particle_bin.val(_4, particle_id_in_bin), 2.0f * static_cast<float>(M_PI));
			}else if(contact_barycentric[0] == 1.0f || contact_barycentric[1] == 1.0f || contact_barycentric[0] == 1.0f){//Point on vertex
				int contact_index;
				if(contact_barycentric[0] == 1.0f){
					contact_index = 0;
				}else if(contact_barycentric[1] == 1.0f){
					contact_index = 1;
				}else {//contact_barycentric[2] == 1.0f
					contact_index = 2;
				}
			
				alpha_shapes_accumulate_triangle_at_vertex(particle_buffer, prev_partition, alpha_shapes_particle_buffer, triangle, blockid, contact_index);
			}else{//Point on edge
				//Use half normal
				atomicAdd(&particle_bin.val(_1, particle_id_in_bin), 0.5f * face_normal[0]);
				atomicAdd(&particle_bin.val(_2, particle_id_in_bin), 0.5f * face_normal[1]);
				atomicAdd(&particle_bin.val(_3, particle_id_in_bin), 0.5f * face_normal[2]);
				
				//Gauss curvature
				atomicAdd(&particle_bin.val(_0, particle_id_in_bin), face_area * (1.0f / 3.0f));//Just ensure this is not zero
				atomicAdd(&particle_bin.val(_4, particle_id_in_bin), static_cast<float>(M_PI));
				
				int opposite_index;
				if(contact_barycentric[0] == 0.0f){
					opposite_index = 0;
				}else if(contact_barycentric[1] == 0.0f){
					opposite_index = 1;
				}else {//contact_barycentric[2] == 0.0f
					opposite_index = 2;
				}
				
				//Mean curvature
				//FIXME: Is this correct?
				const vec3 laplacian = (triangle_positions[(opposite_index + 1) % 3] - particle_position) + (triangle_positions[(opposite_index + 2) % 3] - particle_position) + 2.0f * (triangle_positions[opposite_index] - particle_position);
				atomicAdd(&particle_bin.val(_5, particle_id_in_bin), laplacian[0]);
				atomicAdd(&particle_bin.val(_6, particle_id_in_bin), laplacian[1]);
				atomicAdd(&particle_bin.val(_7, particle_id_in_bin), laplacian[2]);
			}
		}
	}
}

template<typename Partition, MaterialE MaterialType>
__forceinline__ __device__ void alpha_shapes_finalize_particles(const ParticleBuffer<MaterialType> particle_buffer, const Partition prev_partition, AlphaShapesParticleBuffer alpha_shapes_particle_buffer, const int (&own_particle_indices)[ALPHA_SHAPES_MAX_OWN_PARTICLE_COUNT_PER_THREAD], const ivec3 blockid, const int range_start, const int range_end){
	printf("DEF %d %d\n", range_start, range_end);
	for(int particle_id = alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_OWN_PARTICLE_COUNT_PER_THREAD>(range_start); (alpha_shapes_get_global_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_OWN_PARTICLE_COUNT_PER_THREAD>(threadIdx.x, particle_id) >= range_start) && particle_id < alpha_shapes_get_thread_count<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_OWN_PARTICLE_COUNT_PER_THREAD>(threadIdx.x, range_end); particle_id++) {
		const int current_particle_index = own_particle_indices[particle_id];
		
		int advection_source_blockno;
		int source_pidib;
		alpha_shapes_fetch_id(prev_partition, current_particle_index, blockid, advection_source_blockno, source_pidib);
		
		auto particle_bin													= alpha_shapes_particle_buffer.ch(_0, particle_buffer.bin_offsets[advection_source_blockno] + source_pidib / config::G_BIN_CAPACITY);
		const int particle_id_in_bin = source_pidib  % config::G_BIN_CAPACITY;
		
		float summed_angles = particle_bin.val(_4, particle_id_in_bin);
		float summed_area = particle_bin.val(_0, particle_id_in_bin);
		vec3 summed_laplacians;
		summed_laplacians[0] = particle_bin.val(_5, particle_id_in_bin);
		summed_laplacians[1] = particle_bin.val(_6, particle_id_in_bin);
		summed_laplacians[2] = particle_bin.val(_7, particle_id_in_bin);
		vec3 normal;
		normal[0] = particle_bin.val(_1, particle_id_in_bin);
		normal[1] = particle_bin.val(_2, particle_id_in_bin);
		normal[2] = particle_bin.val(_3, particle_id_in_bin);
		
		AlphaShapesPointType point_type;
		if(summed_area > 0.0f){
			point_type = AlphaShapesPointType::OUTER_POINT;
		}else{
			//Isolated point or point in shell
			point_type = AlphaShapesPointType::ISOLATED_POINT;//FIXME: All are currently treated as isolated points
			
			summed_angles += 2.0f * static_cast<float>(M_PI);
			summed_area += 1.0f;//Just ensure this is not zero
			
			//TODO: Or might be part of curve or thin surface
			//TODO: Decide whether interior or exterior point
		}
		
		summed_laplacians /= 2.0f * summed_area;
		const float laplacian_norm = sqrt(summed_laplacians[0] * summed_laplacians[0] + summed_laplacians[1] * summed_laplacians[1] + summed_laplacians[2] * summed_laplacians[2]);

		const float gauss_curvature = (2.0f * static_cast<float>(M_PI) - summed_angles) / summed_area;
		const float mean_curvature = 0.5f * laplacian_norm;
		
		const float normal_length = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
		
		particle_bin.val(_0, particle_id_in_bin) = *reinterpret_cast<float*>(&point_type);
		particle_bin.val(_1, particle_id_in_bin) = normal[0] / normal_length;
		particle_bin.val(_2, particle_id_in_bin) = normal[1] / normal_length;
		particle_bin.val(_3, particle_id_in_bin) = normal[2] / normal_length;
		particle_bin.val(_4, particle_id_in_bin) = mean_curvature;
		particle_bin.val(_5, particle_id_in_bin) = gauss_curvature;
	}
}

template<typename Partition, MaterialE MaterialType>
__forceinline__ __device__ void tmp_write_particle_data(const ParticleBuffer<MaterialType> particle_buffer, const Partition prev_partition, AlphaShapesParticleBuffer alpha_shapes_particle_buffer, const int id, const ivec3 blockid, const int point_type, const std::array<float, 3>& normal, float mean_curvature, float gauss_curvature){
	int advection_source_blockno;
	int source_pidib;
	alpha_shapes_fetch_id(prev_partition, id, blockid, advection_source_blockno, source_pidib);
	
	auto particle_bin													= alpha_shapes_particle_buffer.ch(_0, particle_buffer.bin_offsets[advection_source_blockno] + source_pidib / config::G_BIN_CAPACITY);
	const int particle_id_in_bin = source_pidib  % config::G_BIN_CAPACITY;
	
	particle_bin.val(_0, particle_id_in_bin) = point_type;
	particle_bin.val(_1, particle_id_in_bin) = normal[0];
	particle_bin.val(_2, particle_id_in_bin) = normal[1];
	particle_bin.val(_3, particle_id_in_bin) = normal[2];
	particle_bin.val(_4, particle_id_in_bin) = mean_curvature;
	particle_bin.val(_5, particle_id_in_bin) = gauss_curvature;
}

//Creates triangulation for contact_triangles and additional_contact_particles
template<typename Partition, MaterialE MaterialType>
__forceinline__ __device__ void alpha_shapes_build_tetrahedra(const ParticleBuffer<MaterialType> particle_buffer, const Partition prev_partition, AlphaShapesParticleBuffer alpha_shapes_particle_buffer, SharedMemoryType* __restrict__ shared_memory_storage, const int (&particle_indices)[ALPHA_SHAPES_MAX_PARTICLE_COUNT_PER_THREAD], const int particle_indices_start, const int particle_indices_count, int (&own_particle_indices)[ALPHA_SHAPES_MAX_OWN_PARTICLE_COUNT_PER_THREAD], std::array<int, 3> (&triangles)[ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD], bool (&triangles_is_alpha)[ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD], volatile int& current_triangle_count, volatile int& next_triangle_count, const ivec3 blockid, const float alpha, const int finalize_particles_start, const int finalize_particles_end, const int contact_triangles_count, const int additional_contact_particles_count, volatile float& minimum_x, volatile float& maximum_x){
	__shared__ volatile int temporary_convex_hull_triangles_count;
	__shared__ volatile int next_temporary_convex_hull_triangles_count;
	
	__shared__ volatile int new_particle_count;
	
	//Init counts
	if(threadIdx.x == 0){
		temporary_convex_hull_triangles_count = contact_triangles_count;
		next_temporary_convex_hull_triangles_count = 0;
		new_particle_count = additional_contact_particles_count;
	}
	
	__syncthreads();
	
	//Process additional particles
	while(new_particle_count > 0){
		//Get particle with smallest convex hull
		int current_smallest_in_hull_count = std::numeric_limits<int>::max();
		__shared__ volatile int current_smallest_index;
		if(threadIdx.x == 0){
			current_smallest_index = 0;//Default is 0
		}
		__threadfence_block();
		
		for(int m = 0; m < new_particle_count - 1; ++m){
			const std::array<float, 3> current_outest_point_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, shared_memory_storage->circumsphere_points[m], blockid);
			const vec3 current_outest_point_position {current_outest_point_position_arr[0], current_outest_point_position_arr[1], current_outest_point_position_arr[2]};
		
			int current_in_hull_count = 0;
			for(int j = 0; j < new_particle_count - 1; ++j){
				const std::array<float, 3> current_particle_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, shared_memory_storage->circumsphere_points[j], blockid);
				const vec3 current_particle_position {current_particle_position_arr[0], current_particle_position_arr[1], current_particle_position_arr[2]};
				
				if(m != j){
					__shared__  volatile bool in_new_convex_hull;
					if(threadIdx.x == 0){
						in_new_convex_hull = true;
					}
					__threadfence_block();
					for(int i = 0; i < temporary_convex_hull_triangles_count; ++i){
						__shared__ std::array<volatile int, 3> current_triangle;
						if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(shared_memory_storage->circumsphere_triangles[i]) == threadIdx.x){
							thrust::copy(thrust::seq, triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(shared_memory_storage->circumsphere_triangles[i])].begin(), triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(shared_memory_storage->circumsphere_triangles[i])].end(), current_triangle.begin());
						}
						__threadfence_block();
						
						const std::array<float, 3> current_p0_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, current_triangle[0], blockid);
						const std::array<float, 3> current_p1_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, current_triangle[1], blockid);
						const std::array<float, 3> current_p2_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, current_triangle[2], blockid);
						
						bool current_in_halfspace;
						{
							const std::array<float, 3> current_triangle_normal = alpha_shapes_calculate_triangle_normal({
								current_p0_position_arr,
								current_p1_position_arr,
								current_p2_position_arr
							});
							
							const vec3 current_p0_position {current_p0_position_arr[0], current_p0_position_arr[1], current_p0_position_arr[2]};
							const vec3 current_triangle_normal_vec {current_triangle_normal[0], current_triangle_normal[1], current_triangle_normal[2]};
							
							//Perform halfspace test
							current_in_halfspace = current_triangle_normal_vec.dot(current_outest_point_position - current_p0_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
						}
						
						if(current_in_halfspace){
							//Find outer edge (the one not being shared by another triangle in this set)
							__shared__ std::array<int, 3> edge_count;//Count for opposite edge
							if(threadIdx.x == 0){
								thrust::fill(thrust::seq, edge_count.begin(), edge_count.end(), 0);
							}
							__syncthreads();
							for(int k = 0; k < temporary_convex_hull_triangles_count; ++k){
								if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(shared_memory_storage->circumsphere_triangles[k]) == threadIdx.x){
									const std::array<int, 3> current_compare_circumsphere_triangle = triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(shared_memory_storage->circumsphere_triangles[k])];
									std::array<bool, 3> vertex_contact = {};
									if(
										(current_triangle[0] == current_compare_circumsphere_triangle[0] || current_triangle[0] == current_compare_circumsphere_triangle[1] || current_triangle[0] == current_compare_circumsphere_triangle[2])
									){
										vertex_contact[0] = true;
									}
									if(
										(current_triangle[1] == current_compare_circumsphere_triangle[0] || current_triangle[1] == current_compare_circumsphere_triangle[1] || current_triangle[1] == current_compare_circumsphere_triangle[2])
									){
										vertex_contact[1] = true;
									}
									if(
										(current_triangle[2] == current_compare_circumsphere_triangle[0] || current_triangle[2] == current_compare_circumsphere_triangle[1] || current_triangle[2] == current_compare_circumsphere_triangle[2])
									){
										vertex_contact[2] = true;
									}
									
									//All edges that have a neighbour face have count 1
									if(vertex_contact[1] && vertex_contact[2]){
										atomicAdd(&(edge_count[0]), 1);
									}
									if(vertex_contact[2] && vertex_contact[0]){
										atomicAdd(&(edge_count[1]), 1);
									}
									if(vertex_contact[0] && vertex_contact[1]){
										atomicAdd(&(edge_count[2]), 1);
									}//Otherwise no edge connection
								}
							}
							__syncthreads();
							
							//Test all outer faces (only one face at the edge)
							if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(shared_memory_storage->circumsphere_triangles[i]) == threadIdx.x){
								const std::array<float, 3> current_p0_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, current_triangle[0], blockid);
								const std::array<float, 3> current_p1_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, current_triangle[1], blockid);
								const std::array<float, 3> current_p2_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, current_triangle[2], blockid);
								
								//Test all outer faces (only one face at the edge)
								if(edge_count[0] == 1){
									const std::array<float, 3> current_triangle_normal = alpha_shapes_calculate_triangle_normal({
										current_p1_position_arr,
										current_p2_position_arr,
										current_outest_point_position_arr
									});
									
									const vec3 current_triangle_normal_vec {current_triangle_normal[0], current_triangle_normal[1], current_triangle_normal[2]};
									
									//Perform halfspace test
									const bool current_in_halfspace = current_triangle_normal_vec.dot(current_particle_position - current_outest_point_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
									
									//If point is in an outer halfspace it is not in convex hull
									if(current_in_halfspace){
										in_new_convex_hull = false;
									}
								}
								if(edge_count[1] == 1){
									const std::array<float, 3> current_triangle_normal = alpha_shapes_calculate_triangle_normal({
										current_p2_position_arr,
										current_p0_position_arr,
										current_outest_point_position_arr
									});
									
									const vec3 current_triangle_normal_vec {current_triangle_normal[0], current_triangle_normal[1], current_triangle_normal[2]};
									
									//Perform halfspace test
									const bool current_in_halfspace = current_triangle_normal_vec.dot(current_particle_position - current_outest_point_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
									
									//If point is in an outer halfspace it is not in convex hull
									if(current_in_halfspace){
										in_new_convex_hull = false;
									}
								}
								if(edge_count[2] == 1){
									const std::array<float, 3> current_triangle_normal = alpha_shapes_calculate_triangle_normal({
										current_p0_position_arr,
										current_p1_position_arr,
										current_outest_point_position_arr
									});
									
									const vec3 current_triangle_normal_vec {current_triangle_normal[0], current_triangle_normal[1], current_triangle_normal[2]};
									
									//Perform halfspace test
									const bool current_in_halfspace = current_triangle_normal_vec.dot(current_particle_position - current_outest_point_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
									
									//If point is in an outer halfspace it is not in convex hull
									if(current_in_halfspace){
										in_new_convex_hull = false;
									}
								}//Otherwise triangle is no border triangle
							}
							__threadfence_block();
							
							if(!in_new_convex_hull){
								break;
							}
						}
					}
					
					
					if(in_new_convex_hull){
						current_in_hull_count++;
					}
				}
			}
			
			if(threadIdx.x == 0){
				if(current_in_hull_count < current_smallest_in_hull_count){
					current_smallest_in_hull_count = current_in_hull_count;
					current_smallest_index = m;
				}
			}
		}
		__threadfence_block();
		
		const std::array<float, 3> particle_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, shared_memory_storage->circumsphere_points[current_smallest_index], blockid);
		const vec3 particle_position {particle_position_arr[0], particle_position_arr[1], particle_position_arr[2]};
		
		//For each triangle of the convex hull facing towards the point (point is in triangle halfspace) try to add a tetrahedron formed by this triangle and this point
		//If the point lies in an existing triangle, ignore it.
		for(int j = 0; j < temporary_convex_hull_triangles_count; ++j){
			__shared__ std::array<volatile int, 3> current_triangle;
			if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(shared_memory_storage->circumsphere_triangles[j]) == threadIdx.x){
				thrust::copy(thrust::seq, triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(shared_memory_storage->circumsphere_triangles[j])].begin(), triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(shared_memory_storage->circumsphere_triangles[j])].end(), current_triangle.begin());
			}
			__threadfence_block();
			
			const std::array<float, 3> p0_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, current_triangle[0], blockid);
			const std::array<float, 3> p1_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, current_triangle[1], blockid);
			const std::array<float, 3> p2_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, current_triangle[2], blockid);
			
			const vec3 p0_position {p0_position_arr[0], p0_position_arr[1], p0_position_arr[2]};
			const vec3 p1_position {p1_position_arr[0], p1_position_arr[1], p1_position_arr[2]};
			const vec3 p2_position {p2_position_arr[0], p2_position_arr[1], p2_position_arr[2]};
			
			const std::array<std::array<float, 3>, 3> triangle_positions {
				  p0_position.data_arr()
				, p1_position.data_arr()
				, p2_position.data_arr()
			};
			
			const std::array<float, 3>& triangle_normal0 = alpha_shapes_calculate_triangle_normal(triangle_positions);
			const vec3 triangle_normal_vec0 {triangle_normal0[0], triangle_normal0[1], triangle_normal0[2]};
			
			//Halfspace test
			const bool in_halfspace = triangle_normal_vec0.dot(particle_position - p0_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
			
			//Connect to all faces in halfspace except for last point. Last point is connected to all faces to form a convex hull
			if((new_particle_count == 1) || in_halfspace){
				vec3 sphere_center;
				float radius;
				alpha_shapes_get_circumsphere(triangle_positions[0], triangle_positions[1], triangle_positions[2], particle_position_arr, sphere_center.data_arr(), radius);
				
				const float lambda = triangle_normal_vec0.dot(sphere_center - p0_position);
				
				//Convex test
				bool convex = true;
				/*for(int iter = 0; iter < particle_indices_count; ++iter){
					const std::array<float, 3> particle_position_arr_a = alpha_shapes_get_particle_position(particle_buffer, prev_partition, particle_indices[iter], blockid);
					const vec3 particle_position_a {particle_position_arr_a[0], particle_position_arr_a[1], particle_position_arr_a[2]};
					
					//Don't test tetrahedron corner points
					if(
						   (particle_position_arr_a[0] != triangle_positions[0][0] || particle_position_arr_a[1] != triangle_positions[0][1] || particle_position_arr_a[2] != triangle_positions[0][2])
						&& (particle_position_arr_a[0] != triangle_positions[1][0] || particle_position_arr_a[1] != triangle_positions[1][1] || particle_position_arr_a[2] != triangle_positions[1][2])
						&& (particle_position_arr_a[0] != triangle_positions[2][0] || particle_position_arr_a[1] != triangle_positions[2][1] || particle_position_arr_a[2] != triangle_positions[2][2])
						&& (particle_position_arr_a[0] != particle_position[0] || particle_position_arr_a[1] != particle_position[1] || particle_position_arr_a[2] != particle_position[2])
					){
					
						const bool in_halfspace_a0 = triangle_normal_vec0.dot(particle_position_a - p0_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
						
						//Only test points in halfspace
						if(in_halfspace_a0){
							//NOTE: Testing all 4 spheres of resulting tetrahedron. If one is more near to one side and on the inner side of the tetrahedron we cannot creater a new tetrahedron with the current point cause at least one side needs to create a tetrahedron with another vertex
							vec3 sphere_center_a0;
							vec3 sphere_center_a1;
							vec3 sphere_center_a2;
							vec3 sphere_center_a3;
							float radius_a0;
							float radius_a1;
							float radius_a2;
							float radius_a3;
							alpha_shapes_get_circumsphere(triangle_positions[0], triangle_positions[1], triangle_positions[2], particle_position_arr_a, sphere_center_a0.data_arr(), radius_a0);
							alpha_shapes_get_circumsphere(triangle_positions[1], triangle_positions[0], particle_position_arr, particle_position_arr_a, sphere_center_a1.data_arr(), radius_a1);
							alpha_shapes_get_circumsphere(triangle_positions[2], triangle_positions[1], particle_position_arr, particle_position_arr_a, sphere_center_a2.data_arr(), radius_a2);
							alpha_shapes_get_circumsphere(triangle_positions[0], triangle_positions[2], particle_position_arr, particle_position_arr_a, sphere_center_a3.data_arr(), radius_a3);
							
							const std::array<float, 3>& triangle_normal1 = alpha_shapes_calculate_triangle_normal({
								triangle_positions[1],
								triangle_positions[0],
								particle_position_arr
							});
							const std::array<float, 3>& triangle_normal2 = alpha_shapes_calculate_triangle_normal({
								triangle_positions[2],
								triangle_positions[1],
								particle_position_arr
							});
							const std::array<float, 3>& triangle_normal3 = alpha_shapes_calculate_triangle_normal({
								triangle_positions[0],
								triangle_positions[2],
								particle_position_arr
							});
							const vec3 triangle_normal_vec1 {triangle_normal1[0], triangle_normal1[1], triangle_normal1[2]};
							const vec3 triangle_normal_vec2 {triangle_normal2[0], triangle_normal2[1], triangle_normal2[2]};
							const vec3 triangle_normal_vec3 {triangle_normal3[0], triangle_normal3[1], triangle_normal3[2]};
							
							const bool in_halfspace_a1 = triangle_normal_vec1.dot(particle_position_a - particle_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
							const bool in_halfspace_a2 = triangle_normal_vec2.dot(particle_position_a - particle_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
							const bool in_halfspace_a3 = triangle_normal_vec3.dot(particle_position_a - particle_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
							
							
							//const float lambda_a = triangle_normal_vec.dot(sphere_center_a - p0_position);
							
							//Smaller delaunay sphere is the on that does not contain the other point
							//const vec3 diff = sphere_center_a - particle_position;
							//const float squared_distance = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
							//const vec3 diff_a = sphere_center - particle_position_a;
							//const float squared_distance_a = diff_a[0] * diff_a[0] + diff_a[1] * diff_a[1] + diff_a[2] * diff_a[2];
							
							//a lies in sphere_b, but b does not lie in sphere_a
							//TODO: Maybe use threshold here?
							//bool in_sphere = (squared_distance > (radius_a * radius_a) && squared_distance_a <= (radius * radius));
							
							//If both points lie in each others sphere or none lies in each others sphere use lambda to determine smaller
							//if(!in_sphere){
							//	in_sphere = (lambda_a - lambda) <= ALPHA_SHAPES_MIN_SPHERE_THRESHOLD;
							//}
							
							bool in_sphere = ((radius_a0 - radius) <= ALPHA_SHAPES_MIN_SPHERE_RADIUS_THRESHOLD);
							in_sphere = in_sphere || (in_halfspace_a1 && ((radius_a1 - radius) <= ALPHA_SHAPES_MIN_SPHERE_RADIUS_THRESHOLD));
							in_sphere = in_sphere || (in_halfspace_a2 && ((radius_a2 - radius) <= ALPHA_SHAPES_MIN_SPHERE_RADIUS_THRESHOLD));
							in_sphere = in_sphere || (in_halfspace_a3 && ((radius_a3 - radius) <= ALPHA_SHAPES_MIN_SPHERE_RADIUS_THRESHOLD));
							
							//If a is in circumsphere the tetrahedron is invalid
							if(in_sphere){
								convex = false;
								printf("INTERSECT %d # %d %d %d # %d\n", tmpabc, triangles[shared_memory_storage->circumsphere_triangles[j]][0], triangles[shared_memory_storage->circumsphere_triangles[j]][1], triangles[shared_memory_storage->circumsphere_triangles[j]][2], particle_indices[iter]);
								if(tmpabc == test_index){
									//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, triangles[shared_memory_storage->circumsphere_triangles[j]][0], blockid, 1, triangle_normal0, 0.0f, 0.0f);
									//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, triangles[shared_memory_storage->circumsphere_triangles[j]][1], blockid, 1, triangle_normal0, 0.0f, 0.0f);
									//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, triangles[shared_memory_storage->circumsphere_triangles[j]][2], blockid, 1, triangle_normal0, 0.0f, 0.0f);
									//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, additional_contact_particles[i], blockid, 2, {0.0f, 0.0f, 0.0f}, 0.0f, 0.0f);
									//tmp_write_particle_data(particle_buffer, prev_partition, alpha_shapes_particle_buffer, particle_indices[iter], blockid, 3, {0.0f, 0.0f, 0.0f}, 0.0f, 0.0f);
								}
								//FIXME:break;
							}
						}
					}
				}*/
				
				//If the tetrahedron is valid, evaluate contact conditions; Otherwise mark the conecting triangle as non-convex and don't generate the tetrahedron
				//A non-convex triangle is finalized if it is alpha
				if(convex){
					//Check alpha shape condition and mark faces accordingly
					const bool is_alpha = (radius * radius <= alpha);
					
					//Check contact conditions and update triangle list
					//NOTE: This changes the triangle counts
					alpha_shapes_check_contact_condition(particle_buffer, prev_partition, alpha_shapes_particle_buffer, shared_memory_storage, own_particle_indices, triangles, triangles_is_alpha, current_triangle_count, next_triangle_count, temporary_convex_hull_triangles_count, next_temporary_convex_hull_triangles_count, blockid, shared_memory_storage->circumsphere_triangles[j], shared_memory_storage->circumsphere_points[current_smallest_index], is_alpha,finalize_particles_start, finalize_particles_end);
				
					//Update bounds
					//FIXME: Correct distance (e.g. intersection of triangle halfspace with xy and xz boundaries
					if(threadIdx.x == 0){
						minimum_x = std::numeric_limits<float>::min();//std::min(minimum_x, sphere_center[0] - radius);
						maximum_x = std::numeric_limits<float>::max();//std::max(maximum_x, sphere_center[0] + radius);
					}
				}else{
					__shared__ volatile bool current_triangle_is_alpha;
					
					//Remove the face and move it to alpha if it is alpha
					if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(shared_memory_storage->circumsphere_triangles[j]) == threadIdx.x){
						current_triangle_is_alpha = triangles_is_alpha[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(shared_memory_storage->circumsphere_triangles[j])];
					}
					__threadfence_block();
					
					//Remove the face and move it to alpha if it is alpha
						
					//NOTE: Always false for first triangle
					if(current_triangle_is_alpha){
						alpha_shapes_finalize_triangle(particle_buffer, prev_partition, alpha_shapes_particle_buffer, shared_memory_storage, own_particle_indices, triangles, current_triangle_count, next_triangle_count, current_triangle, blockid, finalize_particles_start, finalize_particles_end);
					}
		
					//Swap contact triangles to end of list to remove them
					int swap_index;
					bool from_current_triangles;
					if(shared_memory_storage->circumsphere_triangles[j] < current_triangle_count){
						from_current_triangles = true;
						swap_index = current_triangle_count - 1;//Swap with last active triangle
						
						//Decrease triangle count
						current_triangle_count--;
					}else{
						from_current_triangles = false;
						swap_index = current_triangle_count + next_triangle_count - 1;//Swap with first next triangle
						
						//Decrease next triangle count
						next_triangle_count--;
					}
					
					//Swap contacting triangle to the end
					alpha_shapes_block_swap<std::array<int, 3>, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(triangles, shared_memory_storage->circumsphere_triangles[j], swap_index);
					alpha_shapes_block_swap<bool, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(triangles_is_alpha, shared_memory_storage->circumsphere_triangles[j], swap_index);
					
					//Fill gap between current list and next list
					if(from_current_triangles){
						alpha_shapes_block_swap<std::array<int, 3>, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(triangles, current_triangle_count, current_triangle_count + next_triangle_count - 1);
						alpha_shapes_block_swap<bool, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(triangles_is_alpha, current_triangle_count, current_triangle_count + next_triangle_count - 1);
					}
					
					if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(shared_memory_storage->circumsphere_triangles[j]) == threadIdx.x){
						thrust::swap(shared_memory_storage->circumsphere_triangles[j], shared_memory_storage->circumsphere_triangles[temporary_convex_hull_triangles_count - 1]);
						//Fill gap between current list and next list
						thrust::swap(shared_memory_storage->circumsphere_triangles[temporary_convex_hull_triangles_count - 1], shared_memory_storage->circumsphere_triangles[temporary_convex_hull_triangles_count + next_temporary_convex_hull_triangles_count - 1]);
						
						//Decrease triangle count
						temporary_convex_hull_triangles_count--;
					}
					__syncthreads();
				}
				
				//NOTE: No index before j can be swapped cause they all were out of halfspace for this point and the point is not part of them (we only add additional points not already handled); Actually only j can be swapped, but the contact handle algorithm is designed to support swaps at any position
				//Reduce j to revisit current, new swapped index
				//Only swap if j is not last, cause if so we swapped j with itself
				if(j != temporary_convex_hull_triangles_count){
					j--;
				}
			}
		}
		
		//Swap particle to end while stipp keeping our final point the last
		if(threadIdx.x == 0){
			thrust::swap(shared_memory_storage->circumsphere_points[current_smallest_index], shared_memory_storage->circumsphere_points[new_particle_count - 1]);
			new_particle_count--;
			if(new_particle_count > 0){
				thrust::swap(shared_memory_storage->circumsphere_points[current_smallest_index], shared_memory_storage->circumsphere_points[new_particle_count - 1]);
			}
			
			//Swap lists
			temporary_convex_hull_triangles_count += next_temporary_convex_hull_triangles_count;
			next_temporary_convex_hull_triangles_count = 0;
		}
		__syncthreads();
	}
}

template<typename Partition, MaterialE MaterialType>
__forceinline__ __device__ bool alpha_shapes_handle_triangle_compare_func(const ParticleBuffer<MaterialType> particle_buffer, const Partition prev_partition, const ivec3 blockid, const std::array<std::array<float, 3>, 3> triangle_positions, const std::array<float, 3> triangle_normal_arr, const int& a, const int& b){
	const vec3 p0_position {triangle_positions[0][0], triangle_positions[0][1], triangle_positions[0][2]};
	const vec3 triangle_normal_vec {triangle_normal_arr[0], triangle_normal_arr[1], triangle_normal_arr[2]};
		
	const std::array<float, 3> particle_position_arr_a = alpha_shapes_get_particle_position(particle_buffer, prev_partition, a, blockid);
	const std::array<float, 3> particle_position_arr_b = alpha_shapes_get_particle_position(particle_buffer, prev_partition, b, blockid);
	
	const vec3 particle_position_a {particle_position_arr_a[0], particle_position_arr_a[1], particle_position_arr_a[2]};
	const vec3 particle_position_b {particle_position_arr_b[0], particle_position_arr_b[1], particle_position_arr_b[2]};
	
	//Test if in half_space; Also sorts out particles that lie in a plane with the triangle
	const bool in_halfspace_a = triangle_normal_vec.dot(particle_position_a - p0_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
	const bool in_halfspace_b = triangle_normal_vec.dot(particle_position_b - p0_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
	
	bool ret = false;
	//Triangle positions are always bigger
	if(
		   (particle_position_arr_a[0] == triangle_positions[0][0] && particle_position_arr_a[1] == triangle_positions[0][1] && particle_position_arr_a[2] == triangle_positions[0][2])
		|| (particle_position_arr_a[0] == triangle_positions[1][0] && particle_position_arr_a[1] == triangle_positions[1][1] && particle_position_arr_a[2] == triangle_positions[1][2])
		|| (particle_position_arr_a[0] == triangle_positions[2][0] && particle_position_arr_a[1] == triangle_positions[2][1] && particle_position_arr_a[2] == triangle_positions[2][2])
	){
		ret = false;
	}else if(
		   (particle_position_arr_b[0] == triangle_positions[0][0] && particle_position_arr_b[1] == triangle_positions[0][1] && particle_position_arr_b[2] == triangle_positions[0][2])
		|| (particle_position_arr_b[0] == triangle_positions[1][0] && particle_position_arr_b[1] == triangle_positions[1][1] && particle_position_arr_b[2] == triangle_positions[1][2])
		|| (particle_position_arr_b[0] == triangle_positions[2][0] && particle_position_arr_b[1] == triangle_positions[2][1] && particle_position_arr_b[2] == triangle_positions[2][2])
	){
		ret = in_halfspace_a;
	} else if(in_halfspace_a && in_halfspace_b){
		//Calculate delaunay spheres
		vec3 sphere_center_a;
		vec3 sphere_center_b;
		float radius_a;
		float radius_b;
		alpha_shapes_get_circumsphere(triangle_positions[0], triangle_positions[1], triangle_positions[2], particle_position_arr_a, sphere_center_a.data_arr(), radius_a);
		alpha_shapes_get_circumsphere(triangle_positions[0], triangle_positions[1], triangle_positions[2], particle_position_arr_b, sphere_center_b.data_arr(), radius_b);
		
		/*
		const float lambda_a = triangle_normal_vec.dot(sphere_center_a - p0_position);
		const float lambda_b = triangle_normal_vec.dot(sphere_center_b - p0_position);
		
		//Smaller delaunay sphere is the on that does not contain the other point
		const vec3 diff_a = sphere_center_b - particle_position_a;
		const float squared_distance_a = diff_a[0] * diff_a[0] + diff_a[1] * diff_a[1] + diff_a[2] * diff_a[2];
		const vec3 diff_b = sphere_center_a - particle_position_b;
		const float squared_distance_b = diff_b[0] * diff_b[0] + diff_b[1] * diff_b[1] + diff_b[2] * diff_b[2];
		
		//a lies in sphere_b, but b does not lie in sphere_a
		ret = (squared_distance_b > (radius_a * radius_a) && squared_distance_a <= (radius_b * radius_b));
		
		//If both points lie in each others sphere or none lies in each others sphere use lambda to determine smaller
		if(!ret){
			ret = (lambda_a < lambda_b);
		}
		*/
		ret = (radius_a < radius_b);
	}else if(in_halfspace_a){
		ret = true;
	}
	return ret;
};

template<typename Partition, MaterialE MaterialType>
__forceinline__ __device__ void alpha_shapes_handle_triangle(const ParticleBuffer<MaterialType> particle_buffer, const Partition prev_partition, AlphaShapesParticleBuffer alpha_shapes_particle_buffer, SharedMemoryType* __restrict__ shared_memory_storage, int (&particle_indices)[ALPHA_SHAPES_MAX_PARTICLE_COUNT_PER_THREAD], const int particle_indices_start, const int particle_indices_count, int (&own_particle_indices)[ALPHA_SHAPES_MAX_OWN_PARTICLE_COUNT_PER_THREAD], std::array<int, 3> (&triangles)[ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD], bool (&triangles_is_alpha)[ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD], volatile  int& current_triangle_count, volatile int& next_triangle_count, const ivec3 blockid, const float alpha, const int finalize_particles_start, const int finalize_particles_end, bool is_first_triangle, const int triangle_index, volatile float& minimum_x, volatile float& maximum_x){
	__shared__ std::array<volatile int, 3> current_triangle;
	if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(triangle_index) == threadIdx.x){
		thrust::copy(thrust::seq, triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(triangle_index)].begin(), triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(triangle_index)].end(), current_triangle.begin());			
	}
	__threadfence_block();
	
	const std::array<float, 3> p0_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, current_triangle[0], blockid);
	const std::array<float, 3> p1_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, current_triangle[1], blockid);
	const std::array<float, 3> p2_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, current_triangle[2], blockid);
	
	const vec3 p0_position {p0_position_arr[0], p0_position_arr[1], p0_position_arr[2]};
	const vec3 p1_position {p1_position_arr[0], p1_position_arr[1], p1_position_arr[2]};
	const vec3 p2_position {p2_position_arr[0], p2_position_arr[1], p2_position_arr[2]};
	
	const std::array<std::array<float, 3>, 3> triangle_positions {
		  p0_position_arr
		, p1_position_arr
		, p2_position_arr
	};
	
	std::array<float, 3> triangle_normal = alpha_shapes_calculate_triangle_normal(triangle_positions);
	vec3 triangle_normal_vec {triangle_normal[0], triangle_normal[1], triangle_normal[2]};
	
	//If we handle the first triangle we first have to determine normal direction
	__shared__ volatile bool flipped_normal;
	if(is_first_triangle){
		const int p3 = alpha_shapes_reduce<int, ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_PARTICLE_COUNT_PER_THREAD>(particle_indices, [&particle_buffer, &prev_partition, &blockid, &triangle_positions, &triangle_normal_vec, &p0_position](const int& a, const int& b){
			if(a == -1){
				return b;
			}else if(b == -1){
				return a;
			}
			const std::array<float, 3> particle_position_arr_a = alpha_shapes_get_particle_position(particle_buffer, prev_partition, a, blockid);
			const std::array<float, 3> particle_position_arr_b = alpha_shapes_get_particle_position(particle_buffer, prev_partition, b, blockid);
			
			const vec3 particle_position_a {particle_position_arr_a[0], particle_position_arr_a[1], particle_position_arr_a[2]};
			const vec3 particle_position_b {particle_position_arr_b[0], particle_position_arr_b[1], particle_position_arr_b[2]};
			
			int ret = -1;
			//Triangle positions are always bigger
			if(
				   (particle_position_arr_a[0] == triangle_positions[0][0] && particle_position_arr_a[1] == triangle_positions[0][1] && particle_position_arr_a[2] == triangle_positions[0][2])
				|| (particle_position_arr_a[0] == triangle_positions[1][0] && particle_position_arr_a[1] == triangle_positions[1][1] && particle_position_arr_a[2] == triangle_positions[1][2])
				|| (particle_position_arr_a[0] == triangle_positions[2][0] && particle_position_arr_a[1] == triangle_positions[2][1] && particle_position_arr_a[2] == triangle_positions[2][2])
			){
				ret = b;
			}else if(
				   (particle_position_arr_b[0] == triangle_positions[0][0] && particle_position_arr_b[1] == triangle_positions[0][1] && particle_position_arr_b[2] == triangle_positions[0][2])
				|| (particle_position_arr_b[0] == triangle_positions[1][0] && particle_position_arr_b[1] == triangle_positions[1][1] && particle_position_arr_b[2] == triangle_positions[1][2])
				|| (particle_position_arr_b[0] == triangle_positions[2][0] && particle_position_arr_b[1] == triangle_positions[2][1] && particle_position_arr_b[2] == triangle_positions[2][2])
			){
				ret = a;
			} else {
				//Calculate delaunay spheres
				vec3 sphere_center_a;
				vec3 sphere_center_b;
				float radius_a;
				float radius_b;
				alpha_shapes_get_circumsphere(triangle_positions[0], triangle_positions[1], triangle_positions[2], particle_position_arr_a, sphere_center_a.data_arr(), radius_a);
				alpha_shapes_get_circumsphere(triangle_positions[0], triangle_positions[1], triangle_positions[2], particle_position_arr_b, sphere_center_b.data_arr(), radius_b);
				
				/*
				const float lambda_a = triangle_normal_vec.dot(sphere_center_a - p0_position);
				const float lambda_b = triangle_normal_vec.dot(sphere_center_b - p0_position);
				
				//Smaller delaunay sphere is the on that does not contain the other point
				const vec3 diff_a = sphere_center_b - particle_position_a;
				const float squared_distance_a = diff_a[0] * diff_a[0] + diff_a[1] * diff_a[1] + diff_a[2] * diff_a[2];
				const vec3 diff_b = sphere_center_a - particle_position_b;
				const float squared_distance_b = diff_b[0] * diff_b[0] + diff_b[1] * diff_b[1] + diff_b[2] * diff_b[2];
				
				//a lies in sphere_b, but b does not lie in sphere_a
				ret = (squared_distance_b > (radius_a * radius_a) && squared_distance_a <= (radius_b * radius_b));
				
				//If both points lie in each others sphere or none lies in each others sphere use lambda to determine smaller
				if(!ret){
					ret = (std::abs(lambda_a) < std::abs(lambda_b));
				}
				*/
				if(radius_a < radius_b){
					ret = a;
				}else{
					ret = b;
				}
			}
			return ret;
		});
		
		if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(triangle_index) == threadIdx.x){
			const std::array<float, 3> particle_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, p3, blockid);
			const vec3 particle_position {particle_position_arr[0], particle_position_arr[1], particle_position_arr[2]};
			
			//If nearest point is not in halfspace of the normal, flip the normal
			const bool in_halfspace = triangle_normal_vec.dot(particle_position - p0_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
			if(!in_halfspace){
				flipped_normal = true;
			}
		}
	
		__threadfence_block();
		
		if(flipped_normal){
			triangle_normal[0] = -triangle_normal[0];
			triangle_normal[1] = -triangle_normal[1];
			triangle_normal[2] = -triangle_normal[2];
			triangle_normal_vec = -triangle_normal_vec;
			if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(triangle_index) == threadIdx.x){
				thrust::swap(triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(triangle_index)][1], triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(triangle_index)][2]);
			}
		}
	}
	
	__shared__ volatile int circumsphere_points_count;
	__shared__ int circumsphere_triangles_count;
	
	//Find smallest point for triangle
	const int p3 = alpha_shapes_reduce<int, ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_PARTICLE_COUNT_PER_THREAD>(particle_indices, [&particle_buffer, &prev_partition, &blockid, &triangle_positions, &triangle_normal](const int& a, const int& b){
		if(a == -1){
			return b;
		}else if(b == -1){
			return a;
		}
		
		if(alpha_shapes_handle_triangle_compare_func(particle_buffer, prev_partition, blockid, triangle_positions, triangle_normal, a, b)){
			return a;
		}else{
			return b;
		}
	});
	
	if(threadIdx.x == 0){
		shared_memory_storage->circumsphere_points[0] = p3;
		circumsphere_points_count = 1;
		circumsphere_triangles_count = 0;
	}
	
	__syncthreads();
	
	//Add all triangles in halfspace of point
	bool local_in_halfspace_of_current_triangle = false;
	{
		const std::array<float, 3> current_particle_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, shared_memory_storage->circumsphere_points[0], blockid);
		const vec3 current_particle_position {current_particle_position_arr[0], current_particle_position_arr[1], current_particle_position_arr[2]};
		
		for(int j = alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(0); j < alpha_shapes_get_thread_count<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(threadIdx.x, current_triangle_count + next_triangle_count); ++j){
			const std::array<float, 3> current_p0_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[j][0], blockid);
			const std::array<float, 3> current_p1_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[j][1], blockid);
			const std::array<float, 3> current_p2_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[j][2], blockid);
			
			const vec3 current_p0_position {current_p0_position_arr[0], current_p0_position_arr[1], current_p0_position_arr[2]};
			
			const std::array<float, 3> current_triangle_normal = alpha_shapes_calculate_triangle_normal({
				current_p0_position_arr,
				current_p1_position_arr,
				current_p2_position_arr
			});
			
			const vec3 current_triangle_normal_vec {current_triangle_normal[0], current_triangle_normal[1], current_triangle_normal[2]};
				
			//Perform halfspace test
			const bool current_in_halfspace = current_triangle_normal_vec.dot(current_particle_position - current_p0_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
			
			//If point is in an outer halfspace of the triangle, add the triangle to active set
			if(current_in_halfspace){
				//If we are out of space print a warning
				if(circumsphere_triangles_count >= ALPHA_SHAPES_MAX_CIRCUMSPHERE_TRIANGLES){
					printf("More triangles in circumsphere than we have memory for.\r\n");
				}else{
					const int index = atomicAdd(&circumsphere_triangles_count, 1);
					shared_memory_storage->circumsphere_triangles[index] = alpha_shapes_get_global_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(threadIdx.x, j);
				}
				if(alpha_shapes_get_global_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(threadIdx.x, j) == triangle_index){
					local_in_halfspace_of_current_triangle = true;
				}
			}
		}
	}
	
	__syncthreads();
	
	const bool in_halfspace_of_current_triangle = (__syncthreads_or(local_in_halfspace_of_current_triangle ? 1 : 0) == 1);
	
	bool first_is_alpha = false;
	//Only proceed if we have at least one point
	if(in_halfspace_of_current_triangle){
		//Add all points between current convex hull and new convex hull formed with new point. Including points within a threshold
		const std::array<float, 3> current_outest_point_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, shared_memory_storage->circumsphere_points[0], blockid);
		const vec3 current_outest_point_position {current_outest_point_position_arr[0], current_outest_point_position_arr[1], current_outest_point_position_arr[2]};
		
		//For the first triangle set alpha to the value of the newly generated tetrahedron to avoid finalization of the face.
		if(is_first_triangle){
			vec3 first_sphere_center;
			float first_radius;
			alpha_shapes_get_circumsphere(triangle_positions[0], triangle_positions[1], triangle_positions[2], current_outest_point_position_arr, first_sphere_center.data_arr(), first_radius);
			first_is_alpha = (first_radius * first_radius <= alpha);
		}
		
		for(int i = particle_indices_start; i < particle_indices_start + particle_indices_count; ++i){
			__shared__ volatile int current_particle_index;
			if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_PARTICLE_COUNT_PER_THREAD>(i) == threadIdx.x){
				current_particle_index = particle_indices[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_PARTICLE_COUNT_PER_THREAD>(i)];
			}
			__threadfence_block();
			
			const std::array<float, 3> current_particle_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, current_particle_index, blockid);
			const vec3 current_particle_position {current_particle_position_arr[0], current_particle_position_arr[1], current_particle_position_arr[2]};
			
			//Skip current point
			if(current_particle_index == shared_memory_storage->circumsphere_points[0]){
				continue;
			}

			//Test if point is on correct side; This also means that it is not in current convex hull
			__shared__ volatile bool on_correct_side;
			if(threadIdx.x == 0){
				on_correct_side = false;
			}
			__threadfence_block();
			for(int j = 0; j < circumsphere_triangles_count; ++j){
				if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(shared_memory_storage->circumsphere_triangles[j]) == threadIdx.x){
					const std::array<float, 3> current_p0_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(shared_memory_storage->circumsphere_triangles[j])][0], blockid);
					const std::array<float, 3> current_p1_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(shared_memory_storage->circumsphere_triangles[j])][1], blockid);
					const std::array<float, 3> current_p2_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(shared_memory_storage->circumsphere_triangles[j])][2], blockid);
					
					const vec3 current_p0_position {current_p0_position_arr[0], current_p0_position_arr[1], current_p0_position_arr[2]};
					
					const std::array<float, 3> current_triangle_normal = alpha_shapes_calculate_triangle_normal({
						current_p0_position_arr,
						current_p1_position_arr,
						current_p2_position_arr
					});
					
					const vec3 current_triangle_normal_vec {current_triangle_normal[0], current_triangle_normal[1], current_triangle_normal[2]};
						
					//Perform halfspace test
					const bool current_in_halfspace = current_triangle_normal_vec.dot(current_particle_position - current_p0_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
					
					//If point is in an outer halfspace it is not in convex hull and it is on correct side
					if(current_in_halfspace){
						on_correct_side = true;
					}
				}
				__threadfence_block();
				
				if(on_correct_side){
					break;
				}
			}
			
			//We are only searching for points on the correct side of our active triangle set
			if(on_correct_side){
				__shared__ volatile bool in_new_convex_hull;
				if(threadIdx.x == 0){
					in_new_convex_hull = true;
				}
				__threadfence_block();
				for(int j = 0; j < circumsphere_triangles_count; ++j){
					__shared__ std::array<volatile int, 3> current_circumsphere_triangle;
					if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(shared_memory_storage->circumsphere_triangles[j]) == threadIdx.x){
						thrust::copy(thrust::seq, triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(shared_memory_storage->circumsphere_triangles[j])].begin(), triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(shared_memory_storage->circumsphere_triangles[j])].end(), current_circumsphere_triangle.begin());
					}
					__threadfence_block();
						
					//Find outer edge (the one not being shared by another triangle in this set)
					__shared__ std::array<int, 3> edge_count;//Count for opposite edge
					if(threadIdx.x == 0){
						thrust::fill(thrust::seq, edge_count.begin(), edge_count.end(), 0);
					}
					__syncthreads();
					for(int k = 0; k < circumsphere_triangles_count; ++k){
						if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(shared_memory_storage->circumsphere_triangles[k]) == threadIdx.x){
							const std::array<int, 3> current_compare_circumsphere_triangle = triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(shared_memory_storage->circumsphere_triangles[k])];
							std::array<bool, 3> vertex_contact = {};
							if(
								(current_circumsphere_triangle[0] == current_compare_circumsphere_triangle[0] || current_circumsphere_triangle[0] == current_compare_circumsphere_triangle[1] || current_circumsphere_triangle[0] == current_compare_circumsphere_triangle[2])
							){
								vertex_contact[0] = true;
							}
							if(
								(current_circumsphere_triangle[1] == current_compare_circumsphere_triangle[0] || current_circumsphere_triangle[1] == current_compare_circumsphere_triangle[1] || current_circumsphere_triangle[1] == current_compare_circumsphere_triangle[2])
							){
								vertex_contact[1] = true;
							}
							if(
								(current_circumsphere_triangle[2] == current_compare_circumsphere_triangle[0] || current_circumsphere_triangle[2] == current_compare_circumsphere_triangle[1] || current_circumsphere_triangle[2] == current_compare_circumsphere_triangle[2])
							){
								vertex_contact[2] = true;
							}
							
							//All edges that have a neighbour face have count 1
							if(vertex_contact[1] && vertex_contact[2]){
								atomicAdd(&(edge_count[0]), 1);
							}
							if(vertex_contact[2] && vertex_contact[0]){
								atomicAdd(&(edge_count[1]), 1);
							}
							if(vertex_contact[0] && vertex_contact[1]){
								atomicAdd(&(edge_count[2]), 1);
							}//Otherwise no edge connection
						}
					}
					__syncthreads();
					
						
					if(threadIdx.x == 0){
						const std::array<float, 3> current_p0_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, current_circumsphere_triangle[0], blockid);
						const std::array<float, 3> current_p1_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, current_circumsphere_triangle[1], blockid);
						const std::array<float, 3> current_p2_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, current_circumsphere_triangle[2], blockid);
						
						//Test all outer faces (only one face at the edge)
						if(edge_count[0] == 1){
							const std::array<float, 3> current_triangle_normal = alpha_shapes_calculate_triangle_normal({
								current_p1_position_arr,
								current_p2_position_arr,
								current_outest_point_position_arr
							});
							
							const vec3 current_triangle_normal_vec {current_triangle_normal[0], current_triangle_normal[1], current_triangle_normal[2]};
							
							//Perform halfspace test
							const bool current_in_halfspace = current_triangle_normal_vec.dot(current_particle_position - current_outest_point_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
							
							//If point is in an outer halfspace it is not in convex hull
							if(current_in_halfspace){
								in_new_convex_hull = false;
							}
						}
						if(edge_count[1] == 1){
							const std::array<float, 3> current_triangle_normal = alpha_shapes_calculate_triangle_normal({
								current_p2_position_arr,
								current_p0_position_arr,
								current_outest_point_position_arr
							});
							
							const vec3 current_triangle_normal_vec {current_triangle_normal[0], current_triangle_normal[1], current_triangle_normal[2]};
							
							//Perform halfspace test
							const bool current_in_halfspace = current_triangle_normal_vec.dot(current_particle_position - current_outest_point_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
							
							//If point is in an outer halfspace it is not in convex hull
							if(current_in_halfspace){
								in_new_convex_hull = false;
							}
						}
						if(edge_count[2] == 1){
							const std::array<float, 3> current_triangle_normal = alpha_shapes_calculate_triangle_normal({
								current_p0_position_arr,
								current_p1_position_arr,
								current_outest_point_position_arr
							});
							
							const vec3 current_triangle_normal_vec {current_triangle_normal[0], current_triangle_normal[1], current_triangle_normal[2]};
							
							//Perform halfspace test
							const bool current_in_halfspace = current_triangle_normal_vec.dot(current_particle_position - current_outest_point_position) > ALPHA_SHAPES_HALFSPACE_TEST_THRESHOLD;
							
							//If point is in an outer halfspace it is not in convex hull
							if(current_in_halfspace){
								in_new_convex_hull = false;
							}
						}//Otherwise triangle is no border triangle
					}
					__threadfence_block();
					
					if(!in_new_convex_hull){
						break;
					}
				}
				
				//If point is in new convex hull, add it to our set
				if(threadIdx.x == 0){
					if(in_new_convex_hull){
						//If we are out of space print a warning
						if(circumsphere_points_count >= ALPHA_SHAPES_MAX_CIRCUMSPHERE_POINTS){
							printf("More points in circumsphere than we have memory for.\r\n");
						}else{
							shared_memory_storage->circumsphere_points[circumsphere_points_count++] = current_particle_index;
						}
					}
				}
			}
		}
	}

	//Swap max point to the end of the list
	if(threadIdx.x == 0){
		thrust::swap(shared_memory_storage->circumsphere_points[0], shared_memory_storage->circumsphere_points[circumsphere_points_count - 1]);
	}
	
	__syncthreads();
	
	//Only handle tetrahedra if we found at least one point. If not all triangles are convex we finalize the current triangle
	if(in_halfspace_of_current_triangle){
		//Build tetrahedra
		alpha_shapes_build_tetrahedra(particle_buffer, prev_partition, alpha_shapes_particle_buffer, shared_memory_storage, particle_indices, particle_indices_start, particle_indices_count, own_particle_indices, triangles, triangles_is_alpha, current_triangle_count, next_triangle_count, blockid, alpha, finalize_particles_start, finalize_particles_end, circumsphere_triangles_count, circumsphere_points_count, minimum_x, maximum_x);
		
		//If we are handling the first triangle, re-add it with flipped normal as outer triangle
		if(is_first_triangle){
			if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(current_triangle_count + next_triangle_count) == threadIdx.x){
				if(flipped_normal){
					triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(current_triangle_count + next_triangle_count)][0] = current_triangle[0];
					triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(current_triangle_count + next_triangle_count)][1] = current_triangle[1];
					triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(current_triangle_count + next_triangle_count)][2] = current_triangle[2];
				}else{
					triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(current_triangle_count + next_triangle_count)][0] = current_triangle[0];
					triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(current_triangle_count + next_triangle_count)][1] = current_triangle[2];
					triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(current_triangle_count + next_triangle_count)][2] = current_triangle[1];
				}
				triangles_is_alpha[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(current_triangle_count + next_triangle_count)] = first_is_alpha;
				next_triangle_count++;
			}
			__threadfence_block();
		}
	}else{
		__shared__ std::array<volatile int, 3> current_triangle;
		__shared__ volatile bool current_triangle_is_alpha;
		
		//Remove the face and move it to alpha if it is alpha
		if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(triangle_index) == threadIdx.x){
			thrust::copy(thrust::seq, triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(triangle_index)].begin(), triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(triangle_index)].end(), current_triangle.begin());
			current_triangle_is_alpha = triangles_is_alpha[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(triangle_index)];
		}
		__threadfence_block();
		
		//Remove the face and move it to alpha if it is alpha
			
		//NOTE: Always false for first triangle
		if(current_triangle_is_alpha){
			alpha_shapes_finalize_triangle(particle_buffer, prev_partition, alpha_shapes_particle_buffer, shared_memory_storage, own_particle_indices, triangles, current_triangle_count, next_triangle_count, current_triangle, blockid, finalize_particles_start, finalize_particles_end);
		}
		
		//Swap contacting triangle to the end
		alpha_shapes_block_swap<std::array<int, 3>, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(triangles, triangle_index, current_triangle_count - 1);
		alpha_shapes_block_swap<bool, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(triangles_is_alpha, triangle_index, current_triangle_count - 1);
		
		//Fill gap between current list and next list
		alpha_shapes_block_swap<std::array<int, 3>, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(triangles, current_triangle_count - 1, current_triangle_count + next_triangle_count - 1);
		alpha_shapes_block_swap<bool, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(triangles_is_alpha, current_triangle_count - 1, current_triangle_count + next_triangle_count - 1);
		
		if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(triangle_index) == threadIdx.x){
			//Decrease triangle count
			current_triangle_count--;
		}
		__threadfence_block();
	}
}

template<typename Partition, typename Grid, MaterialE MaterialType>
__global__ void alpha_shapes(const ParticleBuffer<MaterialType> particle_buffer, const Partition prev_partition, const Partition partition, const Grid grid, AlphaShapesParticleBuffer alpha_shapes_particle_buffer, const unsigned int start_index) {
	//const int src_blockno		   = static_cast<int>(blockIdx.x / config::G_BLOCKVOLUME);
	const ivec3 blockid			   = partition.active_keys[blockIdx.x / config::G_BLOCKVOLUME];
	const ivec3 cellid = blockid * static_cast<int>(config::G_BLOCKSIZE) + ivec3((static_cast<int>(blockIdx.x) / (config::G_BLOCKSIZE * config::G_BLOCKSIZE)) % config::G_BLOCKSIZE, (static_cast<int>(blockIdx.x) / config::G_BLOCKSIZE) % config::G_BLOCKSIZE, static_cast<int>(blockIdx.x) % config::G_BLOCKSIZE);
	
    // Shared memory
	//static_assert(sizeof(SharedMemoryType) <= 48 * (2 << 10));//TODO: Actually should be even a little bit smaller cause we have some other shared variables too
	__shared__ char shmem[sizeof(SharedMemoryType)];
	SharedMemoryType* shared_memory_storage = reinterpret_cast<SharedMemoryType*>(shmem);
	
	const int prev_blockno = prev_partition.query(blockid);
	const int cellno = ((cellid[0] & config::G_BLOCKMASK) << (config::G_BLOCKBITS << 1)) | ((cellid[1] & config::G_BLOCKMASK) << config::G_BLOCKBITS) | (cellid[2] & config::G_BLOCKMASK);
	const int particles_in_cell = particle_buffer.cell_particle_counts[prev_blockno * config::G_BLOCKVOLUME + cellno];
	
	//TODO: If still not enogh memory we can iterate through all particles. Also maybe we can reduce triangle count and maybe merge the arrays or only save alpha triangles; Maybe also we can somehow utilize shared memory?; Also we can split up into several iterations maybe, reusing the same memory; Or first sort globally; Or save bool as bits
	int particle_indices[ALPHA_SHAPES_MAX_PARTICLE_COUNT_PER_THREAD] = {};
	int own_particle_indices[ALPHA_SHAPES_MAX_OWN_PARTICLE_COUNT_PER_THREAD] = {};
	
	std::array<int, 3> triangles[ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD] = {};
	bool triangles_is_alpha[ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD] = {};
	
	/*
	constexpr size_t TOTAL_ARRAY_SIZE = sizeof(particle_indices)
			   + sizeof(own_particle_indices)
			   + sizeof(triangles)
			   + sizeof(triangles_is_alpha);
	
	constexpr unsigned long LOCAL_MEMORY_SIZE = (static_cast<size_t>(1) << 17);
			   
	printf("%lu %lu - ", static_cast<unsigned long>(TOTAL_ARRAY_SIZE), static_cast<unsigned long>(LOCAL_MEMORY_SIZE));
	
	static_assert(TOTAL_ARRAY_SIZE < LOCAL_MEMORY_SIZE && "Not enough local memory");
	*/
	
	const int own_particle_bucket_size = particles_in_cell;
	
	//If we have no particles in the bucket return
	if(own_particle_bucket_size == 0) {
		return;
	}
	
	const float alpha = 0.1f * config::MAX_ALPHA;
	
	//If alpha is too big print a warning
	if(alpha > config::MAX_ALPHA){
		printf("Alpha too big. Is %.28f, but should not be bigger than %.28f.", alpha, config::MAX_ALPHA);
	}
	
	//TODO: Actually we only need to handle blocks in radius of sqrt(alpha) around box
	//Fetch particles
	__shared__ volatile int tmp_particle_bucket_size;
	if(threadIdx.x == 0){
		tmp_particle_bucket_size = 0;
	}
	__threadfence_block();
	for(int i = -static_cast<int>(ALPHA_SHAPES_KERNEL_SIZE); i <= static_cast<int>(ALPHA_SHAPES_KERNEL_SIZE); ++i){
		for(int j = -static_cast<int>(ALPHA_SHAPES_KERNEL_SIZE); j <= static_cast<int>(ALPHA_SHAPES_KERNEL_SIZE); ++j){
			for(int k = -static_cast<int>(ALPHA_SHAPES_KERNEL_SIZE); k <= static_cast<int>(ALPHA_SHAPES_KERNEL_SIZE); ++k){
				const ivec3 cellid_offset {i, j, k};
				const ivec3 current_cellid = cellid + cellid_offset;
				const int current_cellno = ((current_cellid[0] & config::G_BLOCKMASK) << (config::G_BLOCKBITS << 1)) | ((current_cellid[1] & config::G_BLOCKMASK) << config::G_BLOCKBITS) | (current_cellid[2] & config::G_BLOCKMASK);
				
				const ivec3 current_blockid = current_cellid / static_cast<int>(config::G_BLOCKSIZE);
				const int current_blockno = prev_partition.query(current_blockid);
				const ivec3 blockid_offset = current_blockid - blockid;
				
				//FIXME:Hide conditional in std::max; Add in every loop iteration; Otherwise cuda crashes when using particle_bucket_size for array access (maybe though this was just an out-of-memory bug not recognized by cuda); Also this might increase performance
				//For empty blocks (blockno = -1) current_bucket_size will be zero
				const int current_bucket_size = particle_buffer.cell_particle_counts[(std::max(current_blockno, 0) * config::G_BLOCKVOLUME + current_cellno)] * std::min(current_blockno + 1, 1);
				
				alpha_shapes_fetch_particles<ALPHA_SHAPES_MAX_PARTICLE_COUNT_PER_THREAD>(particle_buffer, particle_indices, current_bucket_size, current_blockno, blockid_offset, current_cellno, tmp_particle_bucket_size);
				
				if(i == 0 && j == 0 && k == 0){
					alpha_shapes_fetch_particles<ALPHA_SHAPES_MAX_OWN_PARTICLE_COUNT_PER_THREAD>(particle_buffer, own_particle_indices, current_bucket_size, current_blockno, blockid_offset, current_cellno, 0);
				}
				
				if(threadIdx.x == 0){
					tmp_particle_bucket_size += current_bucket_size;
				}
				__threadfence_block();
			}
		}
	}
	
	//TODO: Actually we only need to handle blocks in radius of sqrt(alpha) around box
	//Filter by max distance; This cannot affect particles of current cell
	const vec3 cell_center = (cellid + 2.0f + vec3(grid.get_offset()[0], grid.get_offset()[1], grid.get_offset()[2]) * config::G_BLOCKSIZE) * config::G_DX;//0->2.0 1->3.0 ...; 1.5 is lower bound of block 0, 5.5 is lower bound of block 1, ...; 1.5 is lower bound of cell 0, 2.5 is lower bound of cell 1, ...
	int thread_particle_count = alpha_shapes_get_thread_count<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_PARTICLE_COUNT_PER_THREAD>(threadIdx.x, tmp_particle_bucket_size);
	for(int particle_id = 0; particle_id < thread_particle_count; particle_id++) {
		const std::array<float, 3> particle_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, particle_indices[particle_id], blockid);
		const vec3 particle_position {particle_position_arr[0], particle_position_arr[1], particle_position_arr[2]};

		//Calculate distance to center
		const vec3 diff = cell_center - particle_position;
		const float squared_distance = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
		
		//Cell boundary + 2.0f * alpha as max radius
		if(squared_distance > (0.75f * config::G_DX * config::G_DX + 2.0f * alpha)){
			//Remove by exchanging with last element
			thrust::swap(particle_indices[particle_id], particle_indices[thread_particle_count - 1]);
			
			//Decrease count
			thread_particle_count--;
			
			//Revisit swapped particle
			particle_id--;
		}
	}
	
	//Set all empty elements to -1;
	for(int i = thread_particle_count; i < ALPHA_SHAPES_MAX_PARTICLE_COUNT_PER_THREAD; ++i){
		particle_indices[i] = -1;
	}
	for(int i = alpha_shapes_get_thread_count<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_OWN_PARTICLE_COUNT_PER_THREAD>(threadIdx.x, own_particle_bucket_size); i < ALPHA_SHAPES_MAX_OWN_PARTICLE_COUNT_PER_THREAD; ++i){
		own_particle_indices[i] = -1;
	}
	
	__shared__ int particle_bucket_size;
	if(threadIdx.x == 0){
		particle_bucket_size = 0;
	}
	__syncthreads();
	atomicAdd(&particle_bucket_size, thread_particle_count);
	
	/*//Sort such that all negative indices (== removed indices) are at the end
	alpha_shapes_merge_sort<int, ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_PARTICLE_COUNT_PER_THREAD>(particle_indices, [](const int& a, const int& b){
		return (a != -1) && (b == -1);
	});*/
	
	//NOTE: Data is not sliced anymore
	
	//Sort by ascending x
	//Also sort such that all negative indices (== removed indices) are at the end
	alpha_shapes_merge_sort<int, ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_PARTICLE_COUNT_PER_THREAD>(particle_indices, [&particle_buffer, &prev_partition, &blockid](const int& a, const int& b){
		if(a == -1){
			return false;
		}else if(b == -1){
			return true;
		}else{
			const std::array<float, 3> particle_position_arr_a = alpha_shapes_get_particle_position(particle_buffer, prev_partition, a, blockid);
			const std::array<float, 3> particle_position_arr_b = alpha_shapes_get_particle_position(particle_buffer, prev_partition, b, blockid);
			return particle_position_arr_a[0] < particle_position_arr_b[0];
		}
	});
	
	alpha_shapes_merge_sort<int, ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_OWN_PARTICLE_COUNT_PER_THREAD>(own_particle_indices, [&particle_buffer, &prev_partition, &blockid](const int& a, const int& b){
		if(a == -1){
			return false;
		}else if(b == -1){
			return true;
		}else{
			const std::array<float, 3> particle_position_arr_a = alpha_shapes_get_particle_position(particle_buffer, prev_partition, a, blockid);
			const std::array<float, 3> particle_position_arr_b = alpha_shapes_get_particle_position(particle_buffer, prev_partition, b, blockid);
			return particle_position_arr_a[0] < particle_position_arr_b[0];
		}
	});
	__syncthreads();
	
	if(cellid[0] == 126 && cellid[1] == 106 && cellid[2] == 126){
		if(threadIdx.x == 0){
			printf("B %d - ", particle_bucket_size);
		}
	}
	
	//NOTE: Data is blocked now
	
	//Distribute accross threads based on indexing methods
	//NOTE: It must be ensured that previouse code does not rely on this indexing
	alpha_shapes_spread_data<int, ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_PARTICLE_COUNT_PER_THREAD>(particle_indices);
	alpha_shapes_spread_data<int, ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_OWN_PARTICLE_COUNT_PER_THREAD>(own_particle_indices);
	
	//Build delaunay triangulation with this points and keep all these intersecting the node
	
	//Create first triangle
	const bool found_initial_triangle = alpha_shapes_get_first_triangle(particle_buffer, prev_partition, shared_memory_storage, particle_indices, blockid, triangles[0]);
	
	__shared__ volatile int current_triangle_count;
	__shared__ volatile int next_triangle_count;
	if(threadIdx.x == 0){
		current_triangle_count = 0;
		next_triangle_count = 0;
	}
	__threadfence_block();
	
	bool found_initial_tetrahedron = false;
	__shared__ volatile float minimum_x;
	__shared__ volatile float maximum_x;
	if(found_initial_triangle){
		if(threadIdx.x == 0){
			current_triangle_count = 1;
		}
		__threadfence_block();
		
		//Create first tetrahedron
		alpha_shapes_handle_triangle(particle_buffer, prev_partition, alpha_shapes_particle_buffer, shared_memory_storage, particle_indices, 0, particle_bucket_size, own_particle_indices, triangles, triangles_is_alpha, current_triangle_count, next_triangle_count, blockid, alpha, 0, own_particle_bucket_size, true, 0, minimum_x, maximum_x);
		
		if(threadIdx.x == 0){
			current_triangle_count = next_triangle_count;
			next_triangle_count = 0;
		}
		__threadfence_block();
		
		found_initial_tetrahedron = (current_triangle_count > 1);
	}
	
	__shared__ int active_particles_start;
	__shared__ int active_particles_count;
	
	__shared__ int own_active_particles_start;
	__shared__ int own_active_particles_count;
	__shared__ int own_last_active_particles_start;
	if(threadIdx.x == 0){
		active_particles_start = 0;
		active_particles_count = 0;
		own_active_particles_start = 0;
		own_active_particles_count = 0;
		own_last_active_particles_start = 0;
	}
	__syncthreads();
	
	if(found_initial_tetrahedron){
		//Init sweep line
		
		
		//Move upper bound; Activate additional particles based on range to new triangles
		int local_new_active_particles = 0;
		for(int particle_id = alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_PARTICLE_COUNT_PER_THREAD>(active_particles_start + active_particles_count); (alpha_shapes_get_global_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_PARTICLE_COUNT_PER_THREAD>(threadIdx.x, particle_id) >= active_particles_start + active_particles_count) && particle_id < alpha_shapes_get_thread_count<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_PARTICLE_COUNT_PER_THREAD>(threadIdx.x, particle_bucket_size); particle_id++) {
			const std::array<float, 3> particle_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, particle_indices[particle_id], blockid);
			if(particle_position_arr[0] > maximum_x){
				break;
			}
			local_new_active_particles++;
		}
		atomicAdd(&active_particles_count, local_new_active_particles);
		__syncthreads();
		
		if(cellid[0] == 126 && cellid[1] == 106 && cellid[2] == 126){
			if(threadIdx.x == 0){
				printf("A %d %d %d\n", own_active_particles_start, own_active_particles_count, own_particle_bucket_size);
			}
		}
		
		int local_new_own_active_particles = 0;
		for(int particle_id = alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_OWN_PARTICLE_COUNT_PER_THREAD>(own_active_particles_start + own_active_particles_count); (alpha_shapes_get_global_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_OWN_PARTICLE_COUNT_PER_THREAD>(threadIdx.x, particle_id) >= own_active_particles_start + own_active_particles_count) && particle_id < alpha_shapes_get_thread_count<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_OWN_PARTICLE_COUNT_PER_THREAD>(threadIdx.x, own_particle_bucket_size); particle_id++) {
			const std::array<float, 3> particle_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, own_particle_indices[particle_id], blockid);
			if(particle_position_arr[0] > maximum_x){
				break;
			}
			local_new_own_active_particles++;
		}
		atomicAdd(&own_active_particles_count, local_new_own_active_particles);
		__syncthreads();
		
		if(cellid[0] == 126 && cellid[1] == 106 && cellid[2] == 126){
			if(threadIdx.x == 0){
				printf("B %d %d %d\n", own_active_particles_start, own_active_particles_count, own_particle_bucket_size);
			}
		}
		
		while(own_active_particles_count > 0){
			if(threadIdx.x == 0){
				minimum_x = std::numeric_limits<float>::max();
				maximum_x = std::numeric_limits<float>::min();
			}
			
			while(current_triangle_count > 0){
				alpha_shapes_handle_triangle(particle_buffer, prev_partition, alpha_shapes_particle_buffer, shared_memory_storage, particle_indices, active_particles_start, active_particles_count, own_particle_indices, triangles, triangles_is_alpha, current_triangle_count, next_triangle_count, blockid, alpha, own_last_active_particles_start, own_active_particles_start + own_active_particles_count, false, 0, minimum_x, maximum_x);
			}
			
			//All triangles have been handled, either checked for tetrahedron or removed due to contacting
			
			//Swap triangle lists
			if(threadIdx.x == 0){
				current_triangle_count = next_triangle_count;
				next_triangle_count = 0;
			
				//If we have more triangles than our convex hull can have this is an error
				if(current_triangle_count > 2 * particle_bucket_size - 4){
					printf("More triangles than convex hull can have\n");
					current_triangle_count = 0;
				}
			}
			__threadfence_block();
			
			//Move sweep line
			
			//Move lower bound; Remove particles that are out of range
			local_new_active_particles = 0;
			for(int particle_id = alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_PARTICLE_COUNT_PER_THREAD>(active_particles_start); (alpha_shapes_get_global_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_PARTICLE_COUNT_PER_THREAD>(threadIdx.x, particle_id) >= active_particles_start) && particle_id < alpha_shapes_get_thread_count<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_PARTICLE_COUNT_PER_THREAD>(threadIdx.x, active_particles_start + active_particles_count); particle_id++) {
				const std::array<float, 3> particle_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, particle_indices[particle_id], blockid);
				if(particle_position_arr[0] >= minimum_x){
					break;
				}
				local_new_active_particles++;
			}
			atomicAdd(&active_particles_start, local_new_active_particles);
			atomicSub(&active_particles_count, local_new_active_particles);
			__syncthreads();
			
			//Move upper bound; Activate additional particles based on range to new triangles
			local_new_active_particles = 0;
			for(int particle_id = alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_PARTICLE_COUNT_PER_THREAD>(active_particles_start + active_particles_count); (alpha_shapes_get_global_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_PARTICLE_COUNT_PER_THREAD>(threadIdx.x, particle_id) >= active_particles_start + active_particles_count) && particle_id < alpha_shapes_get_thread_count<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_PARTICLE_COUNT_PER_THREAD>(threadIdx.x, particle_bucket_size); particle_id++) {
				const std::array<float, 3> particle_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, particle_indices[particle_id], blockid);
				if(particle_position_arr[0] > maximum_x){
					break;
				}
				local_new_active_particles++;
			}
			atomicAdd(&active_particles_count, local_new_active_particles);
			__syncthreads();
			
			if(cellid[0] == 126 && cellid[1] == 106 && cellid[2] == 126){
				if(threadIdx.x == 0){
					printf("C %d %d %d\n", own_active_particles_start, own_active_particles_count, own_particle_bucket_size);
				}
			}
			
			//Move lower bound; Remove particles that are out of range
			local_new_own_active_particles = 0;
			for(int particle_id = alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_OWN_PARTICLE_COUNT_PER_THREAD>(own_active_particles_start); (alpha_shapes_get_global_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_OWN_PARTICLE_COUNT_PER_THREAD>(threadIdx.x, particle_id) >= own_active_particles_start) && particle_id < alpha_shapes_get_thread_count<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_OWN_PARTICLE_COUNT_PER_THREAD>(threadIdx.x, own_active_particles_start + own_active_particles_count); particle_id++) {
				const std::array<float, 3> particle_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, own_particle_indices[particle_id], blockid);
				if(particle_position_arr[0] >= minimum_x){
					break;
				}
				local_new_own_active_particles++;
			}
			atomicAdd(&own_active_particles_start, local_new_own_active_particles);
			atomicSub(&own_active_particles_count, local_new_own_active_particles);
			__syncthreads();
			
			if(cellid[0] == 126 && cellid[1] == 106 && cellid[2] == 126){
				printf("D %d # %d %d %d # %d %d\n", static_cast<int>(threadIdx.x), own_active_particles_start, own_active_particles_count, own_particle_bucket_size, static_cast<int>(alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_OWN_PARTICLE_COUNT_PER_THREAD>(own_active_particles_start + own_active_particles_count)), static_cast<int>(alpha_shapes_get_thread_count<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_OWN_PARTICLE_COUNT_PER_THREAD>(threadIdx.x, own_particle_bucket_size)));
			}
			
			//Move upper bound; Activate additional particles based on range to new triangles
			local_new_own_active_particles = 0;
			for(int particle_id = alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_OWN_PARTICLE_COUNT_PER_THREAD>(own_active_particles_start + own_active_particles_count); (alpha_shapes_get_global_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_OWN_PARTICLE_COUNT_PER_THREAD>(threadIdx.x, particle_id) >= own_active_particles_start + own_active_particles_count) && particle_id < alpha_shapes_get_thread_count<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_OWN_PARTICLE_COUNT_PER_THREAD>(threadIdx.x, own_particle_bucket_size); particle_id++) {
				const std::array<float, 3> particle_position_arr = alpha_shapes_get_particle_position(particle_buffer, prev_partition, own_particle_indices[particle_id], blockid);
				if(particle_position_arr[0] > maximum_x){
					break;
				}
				local_new_own_active_particles++;
			}
			atomicAdd(&own_active_particles_count, local_new_own_active_particles);
			__syncthreads();
			
			if(cellid[0] == 126 && cellid[1] == 106 && cellid[2] == 126){
				if(threadIdx.x == 0){
					printf("E %d %d %d\n", own_active_particles_start, own_active_particles_count, own_particle_bucket_size);
				}
			}
			
			//Finalize particles and faces if possible (based on sweep line)
			alpha_shapes_finalize_particles(particle_buffer, prev_partition, alpha_shapes_particle_buffer, own_particle_indices, blockid, own_last_active_particles_start, own_active_particles_start);
			
			if(threadIdx.x == 0){
				//Save last bounds
				own_last_active_particles_start = own_active_particles_start;
			}
			__syncthreads();
		}
	}
	
	//Add all faces left over at the end to alpha_triangles when no more points are found if they are marked as alpha.
	//NOTE: We are counting backwards to avoid overriding triangles
	for(int current_triangle_index = current_triangle_count - 1; current_triangle_index >= 0; current_triangle_index--) {
		__shared__ std::array<volatile int, 3> current_triangle;
		__shared__ volatile bool current_triangle_is_alpha;
		
		//Remove the face and move it to alpha if it is alpha
		if(alpha_shapes_get_thread_index<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(current_triangle_index) == threadIdx.x){
			thrust::copy(thrust::seq, triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(current_triangle_index)].begin(), triangles[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(current_triangle_index)].end(), current_triangle.begin());
			current_triangle_is_alpha = triangles_is_alpha[alpha_shapes_get_thread_offset<ALPHA_SHAPES_BLOCK_SIZE, ALPHA_SHAPES_MAX_TRIANGLE_COUNT_PER_THREAD>(current_triangle_index)];
		}
		__threadfence_block();
		
		if(current_triangle_is_alpha){
			alpha_shapes_finalize_triangle(particle_buffer, prev_partition, alpha_shapes_particle_buffer, shared_memory_storage, own_particle_indices, triangles, current_triangle_count, next_triangle_count, current_triangle, blockid, own_last_active_particles_start, own_particle_bucket_size);
		}
	}
	
	//Finalize all particles left
	alpha_shapes_finalize_particles(particle_buffer, prev_partition, alpha_shapes_particle_buffer, own_particle_indices, blockid, own_last_active_particles_start, own_particle_bucket_size);
}

template<typename Partition, MaterialE MaterialType>
__global__ void clear_alpha_shapes_particle_buffer(const ParticleBuffer<MaterialType> particle_buffer, const ParticleBuffer<MaterialType> next_particle_buffer, const Partition prev_partition, AlphaShapesParticleBuffer alpha_shapes_particle_buffer){
	const int src_blockno		   = static_cast<int>(blockIdx.x);
	const int particle_bucket_size = next_particle_buffer.particle_bucket_sizes[src_blockno];
	
	//If we have no particles in the bucket return
	if(particle_bucket_size == 0) {
		return;
	}
	
	for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < particle_bucket_size; particle_id_in_block += static_cast<int>(blockDim.x)) {
		auto particle_bin													= alpha_shapes_particle_buffer.ch(_0, next_particle_buffer.bin_offsets[src_blockno] + particle_id_in_block / config::G_BIN_CAPACITY);
		//point_type/summed_area
		particle_bin.val(_0, particle_id_in_block % config::G_BIN_CAPACITY) = 0;
		//normal
		particle_bin.val(_1, particle_id_in_block % config::G_BIN_CAPACITY) = 0.0f;
		particle_bin.val(_2, particle_id_in_block % config::G_BIN_CAPACITY) = 0.0f;
		particle_bin.val(_3, particle_id_in_block % config::G_BIN_CAPACITY) = 0.0f;
		//mean_curvature/summed_angle
		particle_bin.val(_4, particle_id_in_block % config::G_BIN_CAPACITY) = 0.0f;
		//gauss_curvature/summed_laplacian
		particle_bin.val(_5, particle_id_in_block % config::G_BIN_CAPACITY) = 0.0f;
		particle_bin.val(_6, particle_id_in_block % config::G_BIN_CAPACITY) = 0.0f;
		particle_bin.val(_7, particle_id_in_block % config::G_BIN_CAPACITY) = 0.0f;
	}
}

//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers, readability-identifier-naming, misc-definitions-in-headers)
}// namespace mn

#endif