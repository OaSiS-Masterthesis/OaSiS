#include "VolumeIO.hpp"

namespace mn {
	
void begin_write_vdb(openvdb::points::PointDataGrid::Ptr* point_grid, openvdb::tools::PointIndexGrid::Ptr* point_index_grid, const size_t particle_count, const std::vector<std::array<float, 3>>& positions) {
	constexpr unsigned int PARTICLES_PER_VOXEL = 1;
	
	//Init openvdb. It is save to call this function more than once
	openvdb::initialize();
	
	//Create VDB vector
	std::vector<openvdb::Vec3f> positions_vdb(particle_count);
	std::transform(std::execution::par_unseq, positions.begin(), positions.end(), positions_vdb.begin(), [](const std::array<float, 3>& pos){
		return openvdb::Vec3f(pos[0], pos[1], pos[2]);
	});
	
	//Wrapper for points
    openvdb::points::PointAttributeVector<openvdb::Vec3f> positions_wrapper(positions_vdb);
	
    // This method computes a voxel-size to match the number of
    // points / voxel requested. Although it won't be exact, it typically offers
    // a good balance of memory against performance.
    //const float voxelSize = openvdb::points::computeVoxelSize(positions_wrapper, PARTICLES_PER_VOXEL);
	const float voxelSize = 1.0f;

    // Create a transform using this voxel-size.
    const openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform(voxelSize);
	
	// Create a PointIndexGrid. This can be done automatically on creation of
    // the grid, however as this index grid is required for the position and
    // radius attributes, we create one we can use for both attribute creation.
    openvdb::tools::PointIndexGrid::Ptr index_grid = openvdb::tools::createPointIndexGrid<openvdb::tools::PointIndexGrid>(positions_wrapper, *transform);
	//openvdb::tools::PointIndexGrid::Ptr index_grid = openvdb::points::PointIndexGrid::create();
	
    // Create a PointDataGrid containing the points and using the
    // transform given. This function has two template parameters, (1) the codec
    // to use for storing the position, (2) the grid we want to create
    // (ie a PointDataGrid).
    // We use no compression here for the positions.
    openvdb::points::PointDataGrid::Ptr grid = openvdb::points::createPointDataGrid<openvdb::points::NullCodec, openvdb::points::PointDataGrid>(*index_grid, positions_wrapper, *transform);
	//openvdb::points::PointDataGrid::Ptr grid = openvdb::points::PointDataGrid::create();
	
    // Set the name of the grid
    grid->setName("Points");
	
	*point_grid = grid;
	*point_index_grid = index_grid;
}

void end_write_vdb(const std::string& filename, openvdb::points::PointDataGrid::Ptr point_grid, openvdb::tools::PointIndexGrid::Ptr point_index_grid, const std::string& radius_name) {
	//Retrieve attribute names
	const openvdb::points::AttributeSet attributes = point_grid->tree().beginLeaf()->attributeSet();

	std::vector<std::string> attribute_names(attributes.descriptor().map().size());
	std::transform(std::execution::par_unseq, attributes.descriptor().map().begin(), attributes.descriptor().map().end(), attribute_names.begin(), [](const auto& entry){
		return std::get<0>(entry);
	});
	
	//Convert to volume
	openvdb::GridPtrVec grids = openvdb::points::rasterizeSmoothSpheres<openvdb::points::PointDataGrid, openvdb::TypeList<float, int, openvdb::Vec3f>, float, openvdb::FloatGrid>(*point_grid, radius_name, 1.0, 0.4, attribute_names);
	//openvdb::GridPtrVec grids = openvdb::points::rasterizeSpheres<openvdb::points::PointDataGrid, openvdb::TypeList<float, int, openvdb::Vec3f>, float, openvdb::FloatGrid>(*point_grid, radius_name, attribute_names, 1.0);
	
	// Create a VDB file object.
    openvdb::io::File file(filename);
	
    // Write out the contents of the container.
    file.write(grids);
    file.close();
}

void write_vdb_add(const std::vector<float>& data, const std::string& tag, openvdb::points::PointDataGrid::Ptr point_grid, openvdb::tools::PointIndexGrid::Ptr point_index_grid){
	//Wrapper for points
    openvdb::points::PointAttributeVector<float> data_wrapper(data);
	
	//Append attribute
	//openvdb::points::TypedAttributeArray<float, openvdb::points::NullCodec>::registerType();
	openvdb::points::appendAttribute(point_grid->tree(), tag, openvdb::points::TypedAttributeArray<float>::attributeType());
	openvdb::points::populateAttribute<openvdb::points::PointDataTree, openvdb::tools::PointIndexTree, openvdb::points::PointAttributeVector<float>>(point_grid->tree(), point_index_grid->tree(), tag, data_wrapper);
}

void write_vdb_add(const std::vector<int>& data, const std::string& tag, openvdb::points::PointDataGrid::Ptr point_grid, openvdb::tools::PointIndexGrid::Ptr point_index_grid){
	//Wrapper for points
    openvdb::points::PointAttributeVector<int> data_wrapper(data);
	
	//Append attribute
	//openvdb::points::TypedAttributeArray<int, openvdb::points::NullCodec>::registerType();
	openvdb::points::appendAttribute(point_grid->tree(), tag, openvdb::points::TypedAttributeArray<int>::attributeType());
	openvdb::points::populateAttribute<openvdb::points::PointDataTree, openvdb::tools::PointIndexTree, openvdb::points::PointAttributeVector<int>>(point_grid->tree(), point_index_grid->tree(), tag, data_wrapper);
}

void write_vdb_add(const std::vector<std::array<float, 3>>& data, const std::string& tag, openvdb::points::PointDataGrid::Ptr point_grid, openvdb::tools::PointIndexGrid::Ptr point_index_grid){
	//Create VDB vector
	std::vector<openvdb::Vec3f> data_vdb(data.size());
	std::transform(std::execution::par_unseq, data.begin(), data.end(), data_vdb.begin(), [](const std::array<float, 3>& val){
		return openvdb::Vec3f(val[0], val[1], val[2]);
	});
	
	//Wrapper for points
    openvdb::points::PointAttributeVector<openvdb::Vec3f> data_wrapper(data_vdb);
	
	//Append attribute
	//openvdb::points::TypedAttributeArray<openvdb::Vec3f, openvdb::points::NullCodec>::registerType();
	openvdb::points::appendAttribute(point_grid->tree(), tag, openvdb::points::TypedAttributeArray<openvdb::Vec3f>::attributeType());
	openvdb::points::populateAttribute<openvdb::points::PointDataTree, openvdb::tools::PointIndexTree, openvdb::points::PointAttributeVector<openvdb::Vec3f>>(point_grid->tree(), point_index_grid->tree(), tag, data_wrapper);
}

}// namespace mn