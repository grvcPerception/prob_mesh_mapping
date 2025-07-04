cmake_minimum_required(VERSION 3.10)

project(probabilistic_mesh_mapping)

find_package (Eigen3 3.3 REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS -std=c++17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# plane_estimation
add_executable(plane_estimation plane_estimation.cpp)
add_executable(plane_measurement_contribution plane_measurement_contribution.cpp)
add_executable(similarity_computation similarity_computation.cpp)
add_executable(plane_to_plane_merging plane_to_plane_merging.cpp)
add_executable(point_to_plane_merging point_to_plane_merging.cpp)
add_executable(plane_quadric plane_quadric.cpp)