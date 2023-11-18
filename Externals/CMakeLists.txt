find_package(Git)
include(FetchContent)
include(ExternalProject)
include(DownloadProject)

# partio
message("## building partio")
add_subdirectory(partio)

# volume_io
message("## building volume_io")

#Set directories
set(volume_io_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/VolumeIO)
set(volume_io_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/VolumeIO)

file(RELATIVE_PATH RELATIVE_INSTALL_PREFIX ${CMAKE_BINARY_DIR} ${CMAKE_INSTALL_PREFIX})
set(volume_io_INSTALL_PREFIX ${RELATIVE_INSTALL_PREFIX})

#FIXME: Force compiling as debug because VDB has memory corruption in other configuration on our system
set(volume_io_CONFIG_TYPE "Debug" CACHE INTERNAL "Config/build type for volume_io")

execute_process(
	COMMAND ${CMAKE_COMMAND}
	-S ${volume_io_SOURCE_DIR}
	-B ${volume_io_BINARY_DIR}
	-G ${CMAKE_GENERATOR}
	-DCMAKE_EXPORT_COMPILE_COMMANDS=ON
	-DCMAKE_AR:FILEPATH=${CMAKE_AR}
	-DCMAKE_LINKER:FILEPATH=${CMAKE_LINKER}
	-DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
	-DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
	-DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
	-DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
	-DCMAKE_INSTALL_PREFIX:PATH=${volume_io_INSTALL_PREFIX}
	#WORKING_DIRECTORY ${volume_io_BINARY_DIR}
	COMMAND_ECHO STDOUT
	# OUTPUT_FILE       ${volume_io_BINARY_DIR}/configure_output.log
	# ERROR_FILE        ${volume_io_BINARY_DIR}/configure_output.log
	RESULT_VARIABLE   result_config
)

if(result_config)
	message(FATAL_ERROR "Failed volume_io configuration")
endif()


message(STATUS "Building volume_io... with CONFIG: ${volume_io_CONFIG_TYPE}")

execute_process(
COMMAND ${CMAKE_COMMAND}
	--build ${volume_io_BINARY_DIR}
	--target install
	--config ${volume_io_CONFIG_TYPE}
	WORKING_DIRECTORY ${volume_io_BINARY_DIR}
	COMMAND_ECHO STDOUT
	# OUTPUT_FILE       ${volume_io_BINARY_DIR}/build_output.log
	# ERROR_FILE        ${volume_io_BINARY_DIR}/build_output.log
	RESULT_VARIABLE   result_build
)

if(result_build)
	message(FATAL_ERROR "Failed to build volume_io")
endif()

set (VOLUME_IO_INSTALL_PATH ${volume_io_BINARY_DIR}/${volume_io_INSTALL_PREFIX} PARENT_SCOPE)


# optional
#FetchContent_Declare(
#    optional
#    GIT_REPOSITORY https://github.com/TartanLlama/optional.git
#    GIT_TAG 83a0b49e5e9571438c3ebae638e9a37502911b8c
#)
#FetchContent_MakeAvailable(optional)

# function_ref
#FetchContent_Declare(
#    function_ref
#    GIT_REPOSITORY https://github.com/TartanLlama/function_ref.git
#    GIT_TAG 556c2c3fbfc144971c01ba6d177aaab48cab6870
#)
##FetchContent_MakeAvailable(function_ref)
#if (NOT function_ref_POPULATED)
#    FetchContent_Populate(function_ref)
#endif()

# variant
#FetchContent_Declare(
#    variant
#    GIT_REPOSITORY https://github.com/mpark/variant.git
#    GIT_TAG 4988879a9f5a95d72308eca2b1779db6ed9b135d
#)
#FetchContent_MakeAvailable(variant)
