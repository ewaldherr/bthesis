cmake_minimum_required(VERSION 3.16)
project(LubyGPU CXX)
find_package(Kokkos REQUIRED)
add_executable(LubyGPU luby.cpp)
target_link_libraries(LubyGPU Kokkos::kokkos)