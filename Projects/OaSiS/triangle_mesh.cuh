#ifndef TRIANGLE_MESH_CUH
#define TRIANGLE_MESH_CUH
#include <MnBase/Meta/Polymorphism.h>

#include "settings.h"
#include "particle_buffer.cuh"

namespace mn {

using TriangleBinDomain = AlignedDomain<int, config::G_MAX_TRIANGLE_MESH_VERTICES_NUM>;

using TriangleMeshDataDomain = CompactDomain<int, 1>;
using TriangleShellDataDomain = CompactDomain<int, 1>;
using TriangleMeshDomain = CompactDomain<int, 1>;
using TriangleShellDomain  = CompactDomain<int, 1>;


//NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables, readability-identifier-naming) Check is buggy and reports variable errors for template arguments
using TriangleMeshVertexData  = Structural<StructuralType::DENSE, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::SUM_POW2_ALIGN>, TriangleBinDomain, attrib_layout::SOA, f32_, f32_, f32_, f32_, f32_, f32_, f32_, f32_, f32_, f32_, f32_, f32_>;//relative_pos, global_pos, velocity, normal
using TriangleMeshFaceData  = Structural<StructuralType::DENSE, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::SUM_POW2_ALIGN>, TriangleBinDomain, attrib_layout::SOA, u32_, u32_, u32_>;//index
using TriangleMeshData  = Structural<StructuralType::DENSE, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::SUM_POW2_ALIGN>, TriangleMeshDataDomain, attrib_layout::AOS, TriangleMeshVertexData, TriangleMeshFaceData>;

using TriangleShellInnerData  = Structural<StructuralType::DENSE, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::SUM_POW2_ALIGN>, TriangleBinDomain, attrib_layout::SOA, f32_, f32_, f32_, f32_>;//mass, momentum
using TriangleShellOuterData  = Structural<StructuralType::DENSE, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::SUM_POW2_ALIGN>, TriangleBinDomain, attrib_layout::SOA, f32_, f32_, f32_, f32_, f32_, f32_, f32_>;//mass, pos (actually no need to store it, but can be useful for debugging), momentum	
using TriangleShellData  = Structural<StructuralType::DENSE, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::SUM_POW2_ALIGN>, TriangleShellDataDomain, attrib_layout::AOS, TriangleShellInnerData, TriangleShellOuterData>;

template<typename TriangleMeshStruct>
using TriangleMeshBuffer = Structural<StructuralType::DYNAMIC, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::COMPACT>, TriangleMeshDomain, attrib_layout::AOS, TriangleMeshStruct>;

template<typename TriangleShellStruct>
using TriangleShellBuffer  = Structural<StructuralType::DYNAMIC, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::COMPACT>, TriangleShellDomain, attrib_layout::AOS, TriangleShellStruct>;

using TriangleShellGridBlockData  = Structural<StructuralType::DENSE, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::SUM_POW2_ALIGN>, BlockDomain, attrib_layout::SOA, f32_, f32_, f32_, f32_>;//mass, momentum
using TriangleShellGridBufferData = Structural<StructuralType::DYNAMIC, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::COMPACT>, GridBufferDomain, attrib_layout::AOS, TriangleShellGridBlockData>;
//NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables, readability-identifier-naming)


struct TriangleMesh : Instance<TriangleMeshBuffer<TriangleMeshData>> {
	using base_t							 = Instance<TriangleMeshBuffer<TriangleMeshData>>;
	
	
	float mass;
	
	vec3 linear_velocity;
	vec3 angular_momentum;
	vec3 center;
	vec4 rotation;
	vec3 angular_velocity;
	vec9 inertia;
	
	std::function<vec3(Duration, Duration)> animation_linear_func;
	std::function<vec3(Duration, Duration)> animation_rotational_func;

	TriangleMesh()
	: linear_velocity(0.0f)
	, angular_momentum(0.0f)
	, center(0.0f)
	, rotation {0.0f, 0.0f, 0.0f, 1.0f}
	{};

	template<typename Allocator>
	TriangleMesh(Allocator allocator, std::size_t count)
		: base_t {spawn<TriangleMeshBuffer<TriangleMeshData>, orphan_signature>(allocator, count)}
		, linear_velocity(0.0f)
		, angular_momentum(0.0f)
		, center(0.0f)
		, rotation {0.0f, 0.0f, 0.0f, 1.0f}
		{}
	
	void update_parameters(float mass, const std::function<vec3(Duration, Duration)>& animation_linear_func, const std::function<vec3(Duration, Duration)>& animation_rotational_func) {
		this->mass = mass;
		this->animation_linear_func = animation_linear_func;
		this->animation_rotational_func = animation_rotational_func;
	}
	
	void rigid_body_update(Duration curr_time, Duration dt){
		const vec3 linear_force = animation_linear_func(curr_time, dt);
		const vec3 torque = animation_rotational_func(curr_time, dt);
		
		//Update linear velocity
		this->linear_velocity += linear_force/this->mass * dt.count();
		
		//Update center
		this->center += linear_velocity * dt.count();
		
		//Update rotation
		vec4 rotation_tmp;
		//Calculate delta quaternion (see https://gamedev.stackexchange.com/questions/108920/applying-angular-velocity-to-quaternion)
		vec_quat_multiplication(this->angular_velocity.data_arr(), this->rotation.data_arr(), rotation_tmp.data_arr());
		rotation_tmp *= 0.5f;
		rotation_tmp = this->rotation + rotation_tmp * dt.count();//Adding an infinitesimal rotation
		//Normalize rotation
		this->rotation = rotation_tmp / std::sqrt(rotation_tmp[0] * rotation_tmp[0] + rotation_tmp[1] * rotation_tmp[1] + rotation_tmp[2] * rotation_tmp[2] + rotation_tmp[3] * rotation_tmp[3]);
		
		//Update inertia
		vec9 inertia_tmp = this->inertia;
		rotate_by_quat(inertia_tmp.data_arr(), this->rotation.data_arr(), this->inertia.data_arr());
		
		//Update angular momentum
		this->angular_momentum += torque * dt.count();
		
		//Update angular velocity
		solve_linear_system(this->inertia.data_arr(), this->angular_velocity.data_arr(),  this->angular_momentum.data_arr());
	}
};

struct TriangleShellParticleBuffer{
	std::size_t num_active_blocks;
	int* particle_bucket_sizes;
	int* face_bucket_sizes;
	int* blockbuckets;
	int* face_blockbuckets;

	TriangleShellParticleBuffer() = default;
		
	template<typename Allocator>
	void reserve_buckets(Allocator allocator, std::size_t num_block_count) {
		if(blockbuckets) {
			allocator.deallocate(particle_bucket_sizes, sizeof(int) * num_active_blocks);
			allocator.deallocate(face_bucket_sizes, sizeof(int) * num_active_blocks);
			allocator.deallocate(blockbuckets, sizeof(int) * num_active_blocks * config::G_PARTICLE_NUM_PER_BLOCK);
			allocator.deallocate(face_blockbuckets, sizeof(int) * num_active_blocks * config::G_FACE_NUM_PER_BLOCK);
		}
		num_active_blocks	  = num_block_count;
		particle_bucket_sizes = static_cast<int*>(allocator.allocate(sizeof(int) * num_active_blocks));
		face_bucket_sizes = static_cast<int*>(allocator.allocate(sizeof(int) * num_active_blocks));
		blockbuckets		  = static_cast<int*>(allocator.allocate(sizeof(int) * num_active_blocks * config::G_PARTICLE_NUM_PER_BLOCK));
		face_blockbuckets		  = static_cast<int*>(allocator.allocate(sizeof(int) * num_active_blocks * config::G_FACE_NUM_PER_BLOCK));
	}
};

struct TriangleShellGridBuffer : Instance<TriangleShellGridBufferData> {
	using base_t = Instance<TriangleShellGridBufferData>;

	template<typename Allocator>
	explicit TriangleShellGridBuffer(Allocator allocator)
		: base_t {spawn<TriangleShellGridBufferData, orphan_signature>(allocator)} {}

	template<typename Allocator>
	void check_capacity(Allocator allocator, std::size_t capacity) {
		if(capacity > this->capacity) {
			this->resize(allocator, capacity);
		}
	}

	template<typename CudaContext>
	void reset(int block_count, CudaContext& cu_dev) {
		cu_dev.compute_launch({block_count, config::G_BLOCKVOLUME}, clear_grid_triangle_shell, *this);
	}
};

struct TriangleShell : Instance<TriangleShellBuffer<TriangleShellData>> {
	using base_t							 = Instance<TriangleShellBuffer<TriangleShellData>>;
	
	TriangleShellParticleBuffer particle_buffer;

	TriangleShell() = default;

	template<typename Allocator>
	TriangleShell(Allocator allocator, std::size_t count)
		: base_t {spawn<TriangleShellBuffer<TriangleShellData>, orphan_signature>(allocator, count)}
		{}
};

}
#endif