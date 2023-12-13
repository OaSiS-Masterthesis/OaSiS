#ifndef __PARTICLE_IO_HPP_
#define __PARTICLE_IO_HPP_
#include <Partio.h>

#include <array>
#include <string>
#include <vector>

#include "MnBase/Math/Vec.h"
#include "PoissonDisk/SampleGenerator.h"

namespace mn {

template<typename T, std::size_t dim>
void write_partio(const std::string& filename, const std::vector<std::array<T, dim>>& data, const std::string& tag = std::string {"position"}) {
	Partio::ParticlesDataMutable* parts = Partio::create();

	Partio::ParticleAttribute attrib = parts->addAttribute(tag.c_str(), Partio::VECTOR, dim);

	parts->addParticles(data.size());
	for(int idx = 0; idx < (int) data.size(); ++idx) {
		float* val = parts->dataWrite<float>(attrib, idx);
		for(int k = 0; k < dim; k++) {
			val[k] = data[idx][k];
		}
	}
	Partio::write(filename.c_str(), *parts);
	parts->release();
}

void begin_write_partio(Partio::ParticlesDataMutable** parts, const size_t particle_count) {
	*parts = Partio::create();
	(*parts)->addParticles(particle_count);
}

void end_write_partio(const std::string& filename, Partio::ParticlesDataMutable* parts) {
	Partio::write(filename.c_str(), *parts);
	parts->release();
}

template<typename T, typename std::enable_if<std::is_floating_point<T>::value, bool>::type = true>
void write_partio_add(const std::vector<T>& data, const std::string& tag, Partio::ParticlesDataMutable* parts){
	Partio::ParticleAttribute attrib = parts->addAttribute(tag.c_str(), Partio::FLOAT, 1);

	for(int idx = 0; idx < (int) data.size(); ++idx) {
		float* val = parts->dataWrite<float>(attrib, idx);
		*val = data[idx];
	}
}

template<typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
void write_partio_add(const std::vector<T>& data, const std::string& tag, Partio::ParticlesDataMutable* parts){
	Partio::ParticleAttribute attrib = parts->addAttribute(tag.c_str(), Partio::INT, 1);

	for(int idx = 0; idx < (int) data.size(); ++idx) {
		int* val = parts->dataWrite<int>(attrib, idx);
		*val = data[idx];
	}
}

template<typename T, std::size_t dim, typename std::enable_if<std::is_floating_point<T>::value, bool>::type = true>
void write_partio_add(const std::vector<std::array<T, dim>>& data, const std::string& tag, Partio::ParticlesDataMutable* parts){
	Partio::ParticleAttribute attrib = parts->addAttribute(tag.c_str(), Partio::VECTOR, dim);

	for(int idx = 0; idx < (int) data.size(); ++idx) {
		float* val = parts->dataWrite<float>(attrib, idx);
		for(int k = 0; k < dim; k++) {
			val[k] = data[idx][k];
		}
	}
}

template<typename T, std::size_t dim>
void write_points(const std::string& filename, const std::vector<std::array<T, dim>>& data, int frame, const std::string& tag = std::string {"position"}) {
	std::ofstream file_stream;
	file_stream.open(filename);
	
	file_stream << frame << std::endl;
	
	file_stream << "VECTOR " << tag << ' ' << dim << " VALUE";

	for(int idx = 0; idx < (int) data.size(); ++idx) {
		for(int k = 0; k < dim; k++) {
			file_stream << ' ' << data[idx][k];
		}
	}
	
	file_stream << std::endl;
	
	file_stream.close();
}

void begin_write_points(std::ofstream** file_stream, const std::string& filename, int frame, const size_t particle_count) {
	*file_stream = new std::ofstream();
	(*file_stream)->open(filename);
	
	(**file_stream) << frame << std::endl;
}

void end_write_points(std::ofstream* file_stream) {
	file_stream->close();
	
	delete file_stream;
}

template<typename T, typename std::enable_if<std::is_floating_point<T>::value, bool>::type = true>
void write_points_add(const std::vector<T>& data, const std::string& tag, std::ofstream* file_stream){
	(*file_stream) << "VALUE " << tag;
	
	for(int idx = 0; idx < (int) data.size(); ++idx) {
		(*file_stream) << ' ' << data[idx];
	}
	
	(*file_stream) << std::endl;
}

template<typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
void write_points_add(const std::vector<T>& data, const std::string& tag, std::ofstream* file_stream){
	(*file_stream) << "INT " << tag;

	for(int idx = 0; idx < (int) data.size(); ++idx) {
		(*file_stream) << ' ' << data[idx];
	}
	
	(*file_stream) << std::endl;
}

template<typename T, std::size_t dim, typename std::enable_if<std::is_floating_point<T>::value, bool>::type = true>
void write_points_add(const std::vector<std::array<T, dim>>& data, const std::string& tag, std::ofstream* file_stream){
	(*file_stream) << "VECTOR " << tag << ' ' << dim << " VALUE";

	for(int idx = 0; idx < (int) data.size(); ++idx) {
		for(int k = 0; k < dim; k++) {
			(*file_stream) << ' ' << data[idx][k];
		}
	}
	
	(*file_stream) << std::endl;
}

/// have issues
auto read_sdf(const std::string& fn, float ppc, float dx, vec<float, 3> offset, vec<float, 3> lengths, bool uniform_scale = true) {
	std::vector<std::array<float, 3>> data;
	std::string filename = std::string(AssetDirPath) + "MpmParticles/" + fn;

	float levelsetDx;
	SampleGenerator pd;
	std::vector<float> samples;
	vec<float, 3> mins;
	vec<float, 3> maxs;
	vec<float, 3> scales;
	vec<int, 3> maxns;
	pd.LoadSDF(filename, levelsetDx, mins[0], mins[1], mins[2], maxns[0], maxns[1], maxns[2]);
	maxs		= maxns.cast<float>() * levelsetDx;
	scales		= lengths / (maxs - mins);
	float scale = scales[0] < scales[1] ? scales[0] : scales[1];
	scale		= scales[2] < scale ? scales[2] : scale;

	float samplePerLevelsetCell = ppc * levelsetDx / dx * scale;

	pd.GenerateUniformSamples(samplePerLevelsetCell, samples);

	for(int i = 0, size = samples.size() / 3; i < size; i++) {
		vec<float, 3> p {samples[i * 3 + 0], samples[i * 3 + 1], samples[i * 3 + 2]};
		if(uniform_scale){
			p = (p - mins) * scale + offset;
		}else{
			p = (p - mins) * scales + offset;
		}
		// particle[0] = ((samples[i * 3 + 0]) + offset[0]);
		// particle[1] = ((samples[i * 3 + 1]) + offset[1]);
		// particle[2] = ((samples[i * 3 + 2]) + offset[2]);
		data.push_back(std::array<float, 3> {p[0], p[1], p[2]});
	}
	printf("[%f, %f, %f] - [%f, %f, %f], scale %f, parcount %d, lsdx %f, dx %f\n", mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2], scale, (int) data.size(), levelsetDx, dx);
	return data;
}

auto read_sdf(const std::string& fn, float ppc, float dx, int domainsize, vec<float, 3> offset, vec<float, 3> lengths, bool uniform_scale = true) {
	std::vector<std::array<float, 3>> data;
	std::string filename = std::string(AssetDirPath) + "MpmParticles/" + fn;

	float levelsetDx;
	SampleGenerator pd;
	std::vector<float> samples;
	vec<float, 3> mins;
	vec<float, 3> maxs;
	vec<float, 3> scales;
	vec<int, 3> maxns;
	pd.LoadSDF(filename, levelsetDx, mins[0], mins[1], mins[2], maxns[0], maxns[1], maxns[2]);
	maxs = maxns.cast<float>() * levelsetDx;

	scales						= maxns.cast<float>() / domainsize;
	float scale					= scales[0] < scales[1] ? scales[0] : scales[1];
	scale						= scales[2] < scale ? scales[2] : scale;
	float samplePerLevelsetCell = ppc * scale;

	pd.GenerateUniformSamples(samplePerLevelsetCell, samples);

	scales = lengths / (maxs - mins) / maxns.cast<float>();
	scale  = scales[0] < scales[1] ? scales[0] : scales[1];
	scale  = scales[2] < scale ? scales[2] : scale;

	for(int i = 0, size = samples.size() / 3; i < size; i++) {
		vec<float, 3> p {samples[i * 3 + 0], samples[i * 3 + 1], samples[i * 3 + 2]};
		if(uniform_scale){
			p = (p - mins) * scale + offset;
		}else{
			p = (p - mins) * scales + offset;
		}
		data.push_back(std::array<float, 3> {p[0], p[1], p[2]});
	}
	printf("[%f, %f, %f] - [%f, %f, %f], scale %f, parcount %d, lsdx %f, dx %f\n", mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2], scale, (int) data.size(), levelsetDx, dx);
	return data;
}

}// namespace mn

#endif