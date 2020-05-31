cmake_minimum_required(VERSION 3.16)
project(zlearn CXX)
set(CMAKE_CXX_STANDARD 17)

option(BUILD_TEST "build tests; requires catch2 to be already installed locally" OFF)

include(ExternalProject)
find_package(spdlog REQUIRED)
find_package(Eigen3 REQUIRED)
add_library(external INTERFACE)
target_link_libraries(external INTERFACE spdlog::spdlog_header_only Eigen3::Eigen)

target_include_directories(external INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(external INTERFACE
		-Wall -Wextra -march=native
		$<$<CONFIG:Debug>:-g2 -O0>
		$<$<CONFIG:Release>:-O2>
		$<$<CONFIG:RelWithDbgInfo>:--gstabs -save-temps -O2>
		)


# workaround for link time error with bundled version of libfmt in spdlog
target_link_libraries(external INTERFACE fmt::fmt-header-only)

add_subdirectory(src)

if (BUILD_TEST)
	find_package(Catch2 REQUIRED)
	include(Catch)
	add_subdirectory(test)
endif ()


#include(ExternalProject)
#find_package(PkgConfig REQUIRED)
#add_library(external INTERFACE)
#
##find_package(SDL2 REQUIRED)
##target_link_libraries(external INTERFACE SDL2::SDL2)
#ExternalProject_Add(project_SDL2
#		SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ext/SDL2-2.0.12
#		PREFIX ${CMAKE_CURRENT_BINARY_DIR}/ext/SDL
#		CMAKE_ARGS
#		-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
#		-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
#		)
#ExternalProject_Get_Property(project_SDL2 INSTALL_DIR)
#add_library(my_SDL2 SHARED IMPORTED)
#add_dependencies(my_SDL2 project_SDL2)
#target_include_directories(my_SDL2 INTERFACE ${INSTALL_DIR}/include)
#if (CMAKE_BUILD_TYPE STREQUAL Debug)
#	set_target_properties(my_SDL2 PROPERTIES IMPORTED_LOCATION ${INSTALL_DIR}/lib/libSDL2d.so)
#else ()
#	set_target_properties(my_SDL2 PROPERTIES IMPORTED_LOCATION ${INSTALL_DIR}/lib/libSDL2.so)
#endif ()
#target_link_libraries(external INTERFACE my_SDL2 ${CMAKE_DL_LIBS})
