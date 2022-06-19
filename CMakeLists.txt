cmake_minimum_required(VERSION 3.11)

project(EKFSlam)
set(CMAKE_CXX_STANDARD 20)

add_subdirectory(src)

option(BUILD_EXAMPLES "Build examples" ON)
if (${BUILD_EXAMPLES})
    add_subdirectory(examples)
endif ()
