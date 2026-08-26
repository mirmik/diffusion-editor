cmake_minimum_required(VERSION 3.20)

project(diffusion_editor_cgal_alpha_wrap LANGUAGES CXX)

find_package(CGAL REQUIRED)

add_executable(cgal-alpha-wrap main.cpp)
target_compile_features(cgal-alpha-wrap PRIVATE cxx_std_17)
target_link_libraries(cgal-alpha-wrap PRIVATE CGAL::CGAL)

