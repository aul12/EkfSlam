find_package(Eigen3 3.3 REQUIRED NO_MODULE)
add_library(${PROJECT_NAME} EkfSlamManager.cpp Util.cpp)
target_include_directories(${PROJECT_NAME} INTERFACE .)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)
