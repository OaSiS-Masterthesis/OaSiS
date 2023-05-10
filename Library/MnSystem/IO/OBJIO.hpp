#ifndef OBJ_IO_HPP_
#define OBJ_IO_HPP_
#include <thinks/obj_io/obj_io.h>

#include <array>
#include <string>
#include <vector>

//TODO: Different library, that supports other obj files
namespace mn {

template<typename PositionType, typename IndexType, std::size_t Dim>
void write_triangle_mesh(const std::string& filename, const std::vector<std::array<PositionType, Dim>>& positions, const std::vector<std::array<IndexType, 3>>& faces) {
	// Positions.
	const auto vtx_iend = std::end(positions);
	auto pos_vtx_iter = std::begin(positions);
	auto pos_mapper = [&pos_vtx_iter, vtx_iend]() {
		using ObjPositionType = thinks::ObjPosition<float, Dim>;

		if (pos_vtx_iter == vtx_iend) {
			// End indicates that no further calls should be made to this mapper,
			// in this case because the captured iterator has reached the end
			// of the vector.
			return thinks::ObjEnd<ObjPositionType>();
		}

		// Map indicates that additional positions may be available after this one.
		ObjPositionType ret;
		ret.values = *(pos_vtx_iter++);
		return thinks::ObjMap(ret);
	};

	// Faces.
	const auto idx_iend = std::end(faces);
	auto idx_iter = std::begin(faces);
	auto face_mapper = [&idx_iter, idx_iend]() {
		using ObjIndexType = thinks::ObjIndex<IndexType>;
		using ObjFaceType = thinks::ObjTriangleFace<ObjIndexType>;
		
		if (idx_iter == idx_iend) {
			// End indicates that no further calls should be made to this mapper,
			// in this case because the captured iterator has reached the end
			// of the vector.
			return thinks::ObjEnd<ObjFaceType>();
		}

		// Map indicates that additional positions may be available after this one.
		const auto face = *(idx_iter++);
		return thinks::ObjMap(ObjFaceType(ObjIndexType(face[0]), ObjIndexType(face[1]), ObjIndexType(face[2])));
	};
	
	// Open the OBJ file and pass in the mappers, which will be called
	// internally to write the contents of the mesh to the file.
	auto ofs = std::ofstream(filename);
	assert(ofs);
	const auto result = thinks::WriteObj(ofs, pos_mapper, face_mapper);
	ofs.close();
}

template<typename PositionType, typename IndexType, std::size_t Dim>
void read_triangle_mesh(const std::string& fn, const std::array<PositionType, Dim>& offset, const std::array<PositionType, Dim>& scale, std::vector<std::array<PositionType, Dim>>& positions, std::vector<std::array<IndexType, 3>>& faces) {
	std::string filename = std::string(AssetDirPath) + "TriMesh/" + fn;
	
	auto add_position = thinks::MakeObjAddFunc<thinks::ObjPosition<PositionType, Dim>>(
		[&positions, &offset, &scale](const auto& pos) {
			positions.push_back({
				  pos.values[0] * scale[0] + offset[0]
				, pos.values[1] * scale[1] + offset[1]
				, pos.values[2] * scale[2] + offset[2]
			});
		}
	);

	auto add_face = thinks::MakeObjAddFunc<thinks::ObjTriangleFace<thinks::ObjIndex<IndexType>>>(
		[&faces](const auto& face) {
			faces.push_back({face.values[0].value, face.values[1].value, face.values[2].value});
		}
	);
	
	// Open the OBJ file and populate the mesh while parsing it.
	auto ifs = std::ifstream(filename);
	assert(ifs);
	const auto result = thinks::ReadObj(ifs, add_position, add_face);
	ifs.close();
}

}// namespace mn

#endif