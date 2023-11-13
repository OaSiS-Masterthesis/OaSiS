#ifndef __VOLUME_IO_HPP_
#define __VOLUME_IO_HPP_
#include <openvdb/openvdb.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/points/PointRasterizeSDF.h>

#include <array>
#include <string>
#include <vector>
#include <execution>

#include "MnBase/Math/Vec.h"

namespace mn {

template<typename T>
void write_vdb(const std::string& filename, const std::vector<std::array<T, 3>>& data, const std::string& tag = std::string {"position"}, const T density = 1.0) {
	throw runtime_exception("Not implemented!");
}

template<typename T>
void begin_write_vdb(openvdb::points::PointDataGrid::Ptr* point_grid, openvdb::tools::PointIndexGrid::Ptr* point_index_grid, const size_t particle_count, const std::vector<std::array<T, 3>>& positions) {
	constexpr unsigned int PARTICLES_PER_VOXEL = 1;
	
	//Init openvdb. It is save to call this function more than once
	openvdb::initialize();
	
	//Create VDB vector
	std::vector<openvdb::Vec3f> positions_vdb(particle_count);
	std::transform(std::execution::par_unseq, positions.begin(), positions.end(), positions_vdb.begin(), [](const std::array<T, 3>& pos){
		return openvdb::Vec3f(pos[0], pos[1], pos[2]);
	});
	
	//Wrapper for points
    openvdb::points::PointAttributeVector<openvdb::Vec3f> positions_wrapper(positions_vdb);
	
    // This method computes a voxel-size to match the number of
    // points / voxel requested. Although it won't be exact, it typically offers
    // a good balance of memory against performance.
    const float voxelSize = openvdb::points::computeVoxelSize(positions_wrapper, PARTICLES_PER_VOXEL);

    // Create a transform using this voxel-size.
    const openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform(voxelSize);
	
	// Create a PointIndexGrid. This can be done automatically on creation of
    // the grid, however as this index grid is required for the position and
    // radius attributes, we create one we can use for both attribute creation.
    openvdb::tools::PointIndexGrid::Ptr index_grid = openvdb::tools::createPointIndexGrid<openvdb::tools::PointIndexGrid>(positions_wrapper, *transform);
	
    // Create a PointDataGrid containing the points and using the
    // transform given. This function has two template parameters, (1) the codec
    // to use for storing the position, (2) the grid we want to create
    // (ie a PointDataGrid).
    // We use no compression here for the positions.
    openvdb::points::PointDataGrid::Ptr grid = openvdb::points::createPointDataGrid<openvdb::points::NullCodec, openvdb::points::PointDataGrid>(*point_index_grid, positions_wrapper, *transform);
	
    // Set the name of the grid
    grid->setName("Points");
	
	*point_grid = grid;
	*point_index_grid = index_grid;
}

void end_write_vdb(const std::string& filename, openvdb::points::PointDataGrid::Ptr point_grid, openvdb::tools::PointIndexGrid::Ptr point_index_grid, const std::string& radius_name = std::string {"radius"}) {
	//Retrieve attribute names
	const openvdb::points::AttributeSet attributes = point_grid->tree().beginLeaf()->attributeSet();

	std::vector<std::string> attribute_names(attributes.descriptor().map().size());
	std::transform(std::execution::par_unseq, attributes.descriptor().map().begin(), attributes.descriptor().map().end(), attribute_names.begin(), [](const auto& entry){
		return std::get<0>(entry);
	});
	
	
	//Convert to volume
	openvdb::GridPtrVec grids = openvdb::points::rasterizeSmoothSpheres(*point_grid, radius_name, 1.0f, 0.4f, attribute_names);
	
	// Create a VDB file object.
    openvdb::io::File file(filename);
	
    // Write out the contents of the container.
    file.write(grids);
    file.close();
}

template<typename T, typename std::enable_if<std::is_floating_point<T>::value, bool>::type = true>
void write_vdb_add(const std::vector<T>& data, const std::string& tag, openvdb::points::PointDataGrid::Ptr point_grid, openvdb::tools::PointIndexGrid::Ptr point_index_grid){
	//Wrapper for points
    openvdb::points::PointAttributeVector<float> data_wrapper(data);
	
	openvdb::points::appendAttribute<float, openvdb::points::NullCodec, openvdb::points::PointDataGrid>(point_grid->tree(), tag, openvdb::points::TypedAttributeArray<float>::attributeType());
	openvdb::points::populateAttribute<openvdb::points::PointDataTree, openvdb::tools::PointIndexTree, openvdb::points::PointAttributeVector<float>>(point_grid->tree(), point_index_grid->tree(), tag, data_wrapper);
}

template<typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
void write_vdb_add(const std::vector<T>& data, const std::string& tag, openvdb::points::PointDataGrid::Ptr point_grid, openvdb::tools::PointIndexGrid::Ptr point_index_grid){
	//Wrapper for points
    openvdb::points::PointAttributeVector<int> data_wrapper(data);
	
	openvdb::points::appendAttribute<int, openvdb::points::NullCodec, openvdb::points::PointDataGrid>(point_grid->tree(), tag, openvdb::points::TypedAttributeArray<int>::attributeType());
	openvdb::points::populateAttribute<openvdb::points::PointDataTree, openvdb::tools::PointIndexTree, openvdb::points::PointAttributeVector<int>>(point_grid->tree(), point_index_grid->tree(), tag, data_wrapper);
}

template<typename T, typename std::enable_if<std::is_floating_point<T>::value, bool>::type = true>
void write_vdb_add(const std::vector<std::array<T, 3>>& data, const std::string& tag, openvdb::points::PointDataGrid::Ptr point_grid, openvdb::tools::PointIndexGrid::Ptr point_index_grid){
	//Create VDB vector
	std::vector<openvdb::Vec3f> data_vdb(data.size());
	std::transform(std::execution::par_unseq, data.begin(), data.end(), data_vdb.begin(), [](const std::array<T, 3>& val){
		return openvdb::Vec3f(val[0], val[1], val[2]);
	});
	
	//Wrapper for points
    openvdb::points::PointAttributeVector<openvdb::Vec3f> data_wrapper(data_vdb);
	
	openvdb::points::appendAttribute<openvdb::Vec3f, openvdb::points::NullCodec, openvdb::points::PointDataGrid>(point_grid->tree(), tag, openvdb::points::TypedAttributeArray<openvdb::Vec3f>::attributeType());
	openvdb::points::populateAttribute<openvdb::points::PointDataTree, openvdb::tools::PointIndexTree, openvdb::points::PointAttributeVector<openvdb::Vec3f>>(point_grid->tree(), point_index_grid->tree(), tag, data_wrapper);
}

}// namespace mn

#endif