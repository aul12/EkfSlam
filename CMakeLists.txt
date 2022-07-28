cmake_minimum_required(VERSION 3.11)

project(EKFSlam)
set(CMAKE_CXX_STANDARD 20)

add_subdirectory(src)

option(BUILD_EXAMPLES "Build examples" OFF)
option(USE_INSTALLED_EIGEN "Do not download Eigen, use already installed version" OFF)

if (${BUILD_EXAMPLES})
    add_subdirectory(examples)
endif ()
