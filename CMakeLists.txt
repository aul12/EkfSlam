cmake_minimum_required(VERSION 3.17)
project(EKFSlam)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -W -Werror -Wpedantic -fno-omit-frame-pointer -fsanitize=address")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -Wall -W -Werror -Wpedantic -fno-omit-frame-pointer -fsanitize=address")

add_executable(${PROJECT_NAME} main.cpp EkfSlamManager.cpp Util.cpp EkfSlam.hpp)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
