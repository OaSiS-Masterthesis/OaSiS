#include <MnBase/Geometry/GeometrySampler.h>
#include <MnBase/Math/Vec.h>
#include <MnSystem/Cuda/Cuda.h>
#include <MnSystem/IO/IO.h>
#include <fmt/color.h>
#include <fmt/core.h>

#include <MnSystem/IO/ParticleIO.hpp>
#include <cxxopts.hpp>
#include <filesystem>
#include <fstream>

#include "oasis_simulator.cuh"
#include "animation_curves.cuh"
namespace fs = std::filesystem;

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
namespace rj = rapidjson;
namespace {
const std::array K_TYPE_NAMES {"Null", "False", "True", "Object", "Array", "String", "Number"};
}// namespace

// dragon_particles.bin, 775196
// cube256_2.bin, 1048576
// two_dragons.bin, 388950

decltype(auto) load_model(std::size_t particle_counts, const std::string& filename) {
	std::vector<std::array<float, mn::config::NUM_DIMENSIONS>> rawpos(particle_counts);
	auto addr_str = std::string(AssetDirPath) + "MpmParticles/";
	FILE* f;
	fopen_s(&f, (addr_str + filename).c_str(), "rb");
	std::fread(rawpos.data(), sizeof(float), rawpos.size() * mn::config::NUM_DIMENSIONS, f);
	std::fclose(f);
	return rawpos;
}

struct SimulatorConfigs {
	int dim					  = 0;
	float dx				  = NAN;
	float dx_inv			  = NAN;
	int resolution			  = 0;
	float gravity			  = NAN;
	std::vector<float> offset = {};
} const SIM_CONFIGS;

namespace {
template<typename T>
inline bool check_member(const T& model, const char* member) {
	if(!model.HasMember(member)) {
		fmt::print("Member not found: {}\n", member);
		return false;
	} else {
		return true;
	}
}
}// namespace

template<typename T>
inline std::function<mn::vec3(mn::Duration, mn::Duration)> parse_animation_curve(const T& params){
	const std::string type {params["type"].GetString()};
	if(type == "UpAndDown"){
		if(!check_member(params, "range_start") || !check_member(params, "range_end") || !check_member(params, "init") || !check_member(params, "speed")) {
			return std::function<mn::vec3(mn::Duration, mn::Duration)>();
		}
		return UpAndDown(
			  params["range_start"].GetFloat()
			, params["range_end"].GetFloat()
			, params["init"].GetFloat()
			, params["speed"].GetFloat()
		);
	}else if(type == std::string("RotateAroundY")){
		if(!check_member(params, "magnitude")) {
			return std::function<mn::vec3(mn::Duration, mn::Duration)>();
		}
		return RotateAroundY(
			  params["magnitude"].GetFloat()
		);
	}else{
		fmt::print("Unknown animation curve: {}", type);
		return std::function<mn::vec3(mn::Duration, mn::Duration)>();
	}
}

//NOLINTBEGIN(clang-analyzer-cplusplus.PlacementNew) check_member prevents the error case
void parse_scene(const std::string& fn, std::unique_ptr<mn::OasisSimulator>& benchmark) {
	fs::path p {fn};
	if(!p.is_absolute()) {
		p = fs::relative(p);
	}
	if(!fs::exists(p)) {
		fmt::print("file not exist {}\n", fn);
	} else {
		std::size_t size = fs::file_size(p);
		std::string configs;
		configs.resize(size);

		std::ifstream istrm(fn);
		if(!istrm.is_open()) {
			fmt::print("cannot open file {}\n", fn);
		} else {
			istrm.read(configs.data(), static_cast<std::streamsize>(configs.size()));
		}
		istrm.close();
		fmt::print("load the scene file of size {}\n", size);

		rj::Document doc;
		doc.Parse(configs.data());
		for(rj::Value::ConstMemberIterator itr = doc.MemberBegin(); itr != doc.MemberEnd(); ++itr) {
			fmt::print("Scene member {} is {}\n", itr->name.GetString(), K_TYPE_NAMES[itr->value.GetType()]);
		}
		{
			auto it = doc.FindMember("simulation");
			if(it != doc.MemberEnd()) {
				auto& sim = it->value;
				if(sim.IsObject()) {
					fmt::print(fg(fmt::color::cyan), "simulation: gpuid[{}], defaultDt[{}], fps[{}], frames[{}]\n", sim["gpuid"].GetInt(), sim["default_dt"].GetFloat(), sim["fps"].GetInt(), sim["frames"].GetInt());
					benchmark = std::make_unique<mn::OasisSimulator>(sim["gpuid"].GetInt(), mn::Duration(sim["default_dt"].GetFloat()), sim["fps"].GetInt(), sim["frames"].GetInt());
				}
			}
		}///< end simulation parsing
		{
			auto it = doc.FindMember("models");
			if(it != doc.MemberEnd()) {
				if(it->value.IsArray()) {
					fmt::print("has {} models\n", it->value.Size());
					for(auto& model: it->value.GetArray()) {
						if(!check_member(model, "file")) {
							return;
						}

						fs::path p {model["file"].GetString()};
						if(p.extension() == ".sdf") {
							if(!check_member(model, "constitutive")) {
								return;
							}

							std::string constitutive {model["constitutive"].GetString()};

							fmt::print(fg(fmt::color::green), "model constitutive[{}], file[{}]\n", constitutive, model["file"].GetString());

							auto init_model = [&](auto& positions, auto& velocity) {
								if(constitutive == "fixed_corotated") {
									if(!check_member(model, "rho") || !check_member(model, "volume") || !check_member(model, "youngs_modulus") || !check_member(model, "poisson_ratio")) {
										return;
									}

									benchmark->init_model<mn::MaterialE::FIXED_COROTATED>(positions, velocity);
									benchmark->update_fr_parameters(model["rho"].GetFloat(), model["volume"].GetFloat(), model["youngs_modulus"].GetFloat(), model["poisson_ratio"].GetFloat());
								} else if(constitutive == "jfluid") {
									if(!check_member(model, "rho") || !check_member(model, "volume") || !check_member(model, "bulk_modulus") || !check_member(model, "gamma") || !check_member(model, "viscosity")) {
										return;
									}

									benchmark->init_model<mn::MaterialE::J_FLUID>(positions, velocity);
									benchmark->update_j_fluid_parameters(model["rho"].GetFloat(), model["volume"].GetFloat(), model["bulk_modulus"].GetFloat(), model["gamma"].GetFloat(), model["viscosity"].GetFloat());
								} else if(constitutive == "nacc") {
									if(!check_member(model, "rho") || !check_member(model, "volume") || !check_member(model, "youngs_modulus") || !check_member(model, "poisson_ratio") || !check_member(model, "beta") || !check_member(model, "xi")) {
										return;
									}

									benchmark->init_model<mn::MaterialE::NACC>(positions, velocity);
									benchmark->update_nacc_parameters(model["rho"].GetFloat(), model["volume"].GetFloat(), model["youngs_modulus"].GetFloat(), model["poisson_ratio"].GetFloat(), model["beta"].GetFloat(), model["xi"].GetFloat());
								} else if(constitutive == "sand") {
									benchmark->init_model<mn::MaterialE::SAND>(positions, velocity);
								} else {
									fmt::print("Unknown constitutive: {}", constitutive);
								}
							};
							mn::vec<float, mn::config::NUM_DIMENSIONS> offset;
							mn::vec<float, mn::config::NUM_DIMENSIONS> span;
							mn::vec<float, mn::config::NUM_DIMENSIONS> velocity;
							if(!check_member(model, "offset") || !check_member(model, "span") || !check_member(model, "velocity")) {
								return;
							}

							for(int d = 0; d < mn::config::NUM_DIMENSIONS; ++d) {
								offset[d]	= model["offset"].GetArray()[d].GetFloat();
								span[d]		= model["span"].GetArray()[d].GetFloat();
								velocity[d] = model["velocity"].GetArray()[d].GetFloat();
							}
						
							auto positions = mn::read_sdf(model["file"].GetString(), mn::config::MODEL_PPC, mn::config::G_DX, mn::config::G_DOMAIN_SIZE, offset, span);
							mn::IO::insert_job([&]() {
								mn::write_partio<float, mn::config::NUM_DIMENSIONS>(p.stem().string() + ".bgeo", positions);
							});
							mn::IO::flush();
							init_model(positions, velocity);
						}else if(p.extension() == ".obj"){
							if(!check_member(model, "mass") || !check_member(model, "animation_linear") || !check_member(model, "animation_rotational")) {
								return;
							}
							
							auto animation_linear_it = model.FindMember("animation_linear");
							auto& animation_linear_params = animation_linear_it->value;
							if(!animation_linear_params.IsObject()) {
								fmt::print("animation_linear is no object but of type {}", K_TYPE_NAMES[animation_linear_params.GetType()]);
							}
							auto animation_rotational_it = model.FindMember("animation_rotational");
							auto& animation_rotational_params = animation_rotational_it->value;
							if(!animation_rotational_params.IsObject()) {
								fmt::print("animation_rotational is no object but of type {}", K_TYPE_NAMES[animation_linear_params.GetType()]);
							}
							
							if(!check_member(animation_linear_params, "type") || !check_member(animation_rotational_params, "type")) {
								return;
							}
							
							std::function<mn::vec3(mn::Duration, mn::Duration)> animation_linear = parse_animation_curve(animation_linear_params);
							std::function<mn::vec3(mn::Duration, mn::Duration)> animation_rotational = parse_animation_curve(animation_rotational_params);
							
							if(!animation_linear || !animation_rotational){
								return;
							}
							
							mn::vec<float, mn::config::NUM_DIMENSIONS> offset;
							mn::vec<float, mn::config::NUM_DIMENSIONS> scale;
							if(!check_member(model, "offset") || !check_member(model, "scale")) {
								return;
							}

							for(int d = 0; d < mn::config::NUM_DIMENSIONS; ++d) {
								offset[d]	= model["offset"].GetArray()[d].GetFloat();
								scale[d]		= model["scale"].GetArray()[d].GetFloat();
							}
							
							std::vector<std::array<float, mn::config::NUM_DIMENSIONS>> positions;
							std::vector<std::array<uint32_t, 3>> faces;
							mn::read_triangle_mesh(model["file"].GetString(), offset.data_arr(), scale.data_arr(), positions, faces);

							benchmark->init_triangle_mesh(positions, faces);//CUBE
							benchmark->update_triangle_mesh_parameters(model["mass"].GetFloat(), animation_linear, animation_rotational);
						}else{
							fmt::print("Unknown file type: {}", model["file"].GetString());
						}
					}
					
					//TODO: Create and use loader for triangle meshes
					/*benchmark->init_triangle_mesh(
						{
							 {0.4f, 0.4f, 0.4f}
							,{0.4f, 0.4f, 0.6f}
							,{0.4f, 0.6f, 0.4f}
							,{0.4f, 0.6f, 0.6f}
							,{0.6f, 0.4f, 0.4f}
							,{0.6f, 0.4f, 0.6f}
							,{0.6f, 0.6f, 0.4f}
							,{0.6f, 0.6f, 0.6f}
							
						},
						{
							 {0, 1, 2}
							,{1, 3, 2}
							,{1, 5, 3}
							,{5, 7, 3}
							,{5, 4, 7}
							,{4, 6, 7}
							,{4, 0, 6}
							,{0, 2, 6}
							,{2, 3, 6}
							,{3, 7, 6}
							,{0, 1, 4}
							,{1, 5, 4}
						}
					);//CUBE
					benchmark->update_triangle_mesh_parameters(1.0f, UpAndDown(-1500.0f, 1500.0f, 1500.0f, 100000.0f), RotateAroundY(10.0f));*/
				}
			}
		}///< end models parsing
	}
}
//NOLINTEND(clang-analyzer-cplusplus.PlacementNew)

int main(int argc, char* argv[]) {
	mn::Cuda::startup();

	cxxopts::Options options("Scene_Loader", "Read simulation scene");
	options.add_options()("f,file", "Scene Configuration File", cxxopts::value<std::string>()->default_value("scenes/scene.json"));
	auto results = options.parse(argc, argv);
	auto fn		 = results["file"].as<std::string>();
	fmt::print("loading scene [{}]\n", fn);

	std::unique_ptr<mn::OasisSimulator> benchmark;
	parse_scene(fn, benchmark);
	/*
	benchmark = std::make_unique<mn::OasisSimulator>(1, 1e-4, 24, 60);

	constexpr auto LEN		 = 46;
	constexpr auto STRIDE	 = 56;
	constexpr auto MODEL_COUNT = 3;
	for(int did = 0; did < 2; ++did) {
		std::vector<std::array<float, 3>> model;
		for(int i = 0; i < MODEL_COUNT; ++i) {
			auto idx = (did * MODEL_COUNT + i);
			model	 = sample_uniform_box(
				   gdx,
				   ivec3 {18 + (idx & 1 ? STRIDE : 0), 18, 18},
				   ivec3 {18 + (idx & 1 ? STRIDE : 0) + LEN, 18 + LEN, 18 + LEN}
			   );
		}
		benchmark->init_model<mn::MaterialE::FixedCorotated>(
			model,
			vec<float, 3> {0.f, 0.f, 0.f}
		);
	}
	*/
	getchar();

	benchmark->main_loop();
	///
	mn::IO::flush();
	benchmark.reset();
	///
	mn::Cuda::shutdown();
	return 0;
}