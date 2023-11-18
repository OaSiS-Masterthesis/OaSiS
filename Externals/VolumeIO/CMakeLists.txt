CMAKE_MINIMUM_REQUIRED( VERSION 3.8.0 )
SET( CMAKE_SKIP_RPATH TRUE )

## project name & version
PROJECT(volume_io LANGUAGES CXX VERSION 1.5.2)

#FIXME: Enable optimizations as we only compile as debug as work-around
#set(CMAKE_CXX_FLAGS_DEBUG "/O2 /Ob2 /Zi")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake)

# Enable C++11
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED YES)
SET(CMAKE_CXX_EXTENSIONS NO)

## Setup platform specific helper defines build variants
IF(WIN32)
  ADD_DEFINITIONS (-D_USE_MATH_DEFINES)
ELSE()
  ADD_COMPILE_OPTIONS (-Wextra -Wno-unused-parameter)
ENDIF()
IF(APPLE)
  set (CMAKE_SHARED_LINKER_FLAGS "-undefined dynamic_lookup")
ENDIF()

## Set install location
IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    IF(WIN32)
        set(VARIANT_DIRECTORY "Windows-x86_64")
    ELSE()
        execute_process(COMMAND sh -c "echo `uname`-`uname -r | cut -d'-' -f1`-`uname -m`" OUTPUT_VARIABLE VARIANT_DIRECTORY OUTPUT_STRIP_TRAILING_WHITESPACE)
    ENDIF()
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/${VARIANT_DIRECTORY}")
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)


file(GLOB sources "*.cpp")

add_library(volume_io SHARED ${sources})
set_target_properties(volume_io PROPERTIES OUTPUT_NAME volume_io POSITION_INDEPENDENT_CODE ON)
if (WIN32)
if (MSVC)
  #target_compile_options(volume_io PUBLIC /MT)
  target_compile_options(volume_io PRIVATE /bigobj)
  target_link_options(volume_io PRIVATE /INCREMENTAL:NO)
endif()
endif()

target_include_directories(volume_io
	SYSTEM
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)


#OpenVDB
set(OPENVDB_USE_STATIC_LIBS ON)
find_package(OpenVDB REQUIRED)

target_link_libraries(volume_io 
    PRIVATE      OpenVDB::openvdb
)

install(TARGETS volume_io
	#EXPORT volume_ioTARGETS
    FILE_SET HEADERS
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

#install(EXPORT volume_ioTARGETS
#	FILE volume_ioTARGETS.cmake
#	DESTINATION lib/cmake/volume_io
#)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
	${CMAKE_CURRENT_BINARY_DIR}/volume_ioConfigVersion.cmake
	VERSION 1.0
	COMPATIBILITY SameMajorVersion
)
install(FILES
	volume_ioConfig.cmake
	${CMAKE_CURRENT_BINARY_DIR}/volume_ioConfigVersion.cmake
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/volume_io
)

install( 
	DIRECTORY .
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} 
	FILES_MATCHING 
	PATTERN "*.h"
	PATTERN "*.hpp"
	PATTERN "*.cuh"
) 