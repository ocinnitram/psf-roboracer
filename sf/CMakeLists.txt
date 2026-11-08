cmake_minimum_required(VERSION 3.15)
project(ppo_sf_roboracer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -march=native -g")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
# Evaluations must reproduce on every machine. -ffp-contract=off keeps the
# compiler from choosing its own FMA fusions; -fno-builtin-sin/cos keep it from
# fusing sin(x) and cos(x) into libm's sincos, which glibc 2.35 rounds
# differently from its own sin and cos (sin and cos agree across releases).
add_compile_options(-ffp-contract=off -fno-builtin-sin -fno-builtin-cos)

# Detector debug stream: sends detector packets to UDP port 9876 for an external
# plotter that is not part of this repository. Off in every normal build.
option(SF_DEBUG_UDP "Compile the detector's UDP debug sender" OFF)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

add_subdirectory(externals/pybind11)

# vehicle/params.h is the one definition. vehicle/params.yaml is generated
# from it for the sim's Python side and the ROS node; never edited by hand.
get_filename_component(REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
add_custom_command(
  OUTPUT ${REPO_ROOT}/vehicle/params.yaml
  COMMAND ${Python_EXECUTABLE} ${REPO_ROOT}/vehicle/gen_params.py
  DEPENDS ${REPO_ROOT}/vehicle/params.h ${REPO_ROOT}/vehicle/gen_params.py
  COMMENT "vehicle/params.h -> vehicle/params.yaml")
add_custom_target(vehicle_params ALL DEPENDS ${REPO_ROOT}/vehicle/params.yaml)

file(GLOB SF_CORE_SOURCES CONFIGURE_DEPENDS src/*.cpp)
list(REMOVE_ITEM SF_CORE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp)
add_library(sf_core OBJECT ${SF_CORE_SOURCES} externals/core-math/atan.c)
target_include_directories(sf_core PUBLIC
  ${REPO_ROOT} include externals/nanoflann externals/core-math)
set_property(TARGET sf_core PROPERTY POSITION_INDEPENDENT_CODE ON)
if(SF_DEBUG_UDP)
  target_compile_definitions(sf_core PUBLIC SF_DEBUG_UDP)
endif()

pybind11_add_module(ppo_sf_roboracer src/bindings.cpp)
target_link_libraries(ppo_sf_roboracer PRIVATE sf_core)

option(SF_TESTS "Build the ctest executables" ON)
if(SF_TESTS)
  enable_testing()
  file(GLOB SF_TESTS CONFIGURE_DEPENDS tests/test_*.cpp)
  foreach(test_src ${SF_TESTS})
    get_filename_component(name ${test_src} NAME_WE)
    add_executable(${name} ${test_src})
    target_link_libraries(${name} PRIVATE sf_core)
    add_test(NAME ${name} COMMAND ${name} WORKING_DIRECTORY
      ${CMAKE_CURRENT_SOURCE_DIR})
  endforeach()
endif()
