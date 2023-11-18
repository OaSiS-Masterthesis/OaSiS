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

namespace mn {

template<typename T>
void write_vdb(const std::string& filename, const std::vector<std::array<T, 3>>& data, const std::string& tag = std::string {"position"}) {
	throw runtime_exception("Not implemented!");
}

void begin_write_vdb(openvdb::points::PointDataGrid::Ptr* point_grid, openvdb::tools::PointIndexGrid::Ptr* point_index_grid, const size_t particle_count, const std::vector<std::array<float, 3>>& positions);

void end_write_vdb(const std::string& filename, openvdb::points::PointDataGrid::Ptr point_grid, openvdb::tools::PointIndexGrid::Ptr point_index_grid, const std::string& radius_name = std::string {"radius"});

void write_vdb_add(const std::vector<float>& data, const std::string& tag, openvdb::points::PointDataGrid::Ptr point_grid, openvdb::tools::PointIndexGrid::Ptr point_index_grid);

void write_vdb_add(const std::vector<int>& data, const std::string& tag, openvdb::points::PointDataGrid::Ptr point_grid, openvdb::tools::PointIndexGrid::Ptr point_index_grid);

void write_vdb_add(const std::vector<std::array<float, 3>>& data, const std::string& tag, openvdb::points::PointDataGrid::Ptr point_grid, openvdb::tools::PointIndexGrid::Ptr point_index_grid);

}// namespace mn

#endif