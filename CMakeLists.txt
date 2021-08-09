cmake_minimum_required(VERSION 3.10)
project(EKFSlam)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(CMAKE_CXX_STANDARD 11)

add_library(${PROJECT_NAME} STATIC EkfSlamManager.cpp Util.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

option(ekf_examples "Build example for EKF Slam" OFF)

if (${ekf_examples})
    add_executable(main.cpp main.cpp)
    target_link_libraries(main.cpp ${PROJECT_NAME})
endif ()
