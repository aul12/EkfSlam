if (USE_INSTALLED_EIGEN)
    find_package(Eigen3 3.3 REQUIRED NO_MODULE)
else ()
    include(FetchContent)

    FetchContent_Declare(
            eigen
            GIT_REPOSITORY "https://gitlab.com/libeigen/eigen"
            GIT_TAG "3.3.7" # Do not change or upgrade both Eigen 3.4 and master are broken
    )
    FetchContent_Populate(eigen)
endif ()


add_library(${PROJECT_NAME} Util.cpp)
target_include_directories(${PROJECT_NAME} INTERFACE .)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror)

if (USE_INSTALLED_EIGEN)
    target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)
else ()
    target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${eigen_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR})
endif ()
