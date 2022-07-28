project(EKFSlamExample)

include(FetchContent)

FetchContent_Declare(
        json
        GIT_REPOSITORY "https://github.com/nlohmann/json.git"
)
FetchContent_MakeAvailable(json)
set(JSON_BuildTests OFF CACHE INTERNAL "")

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE EKFSlam nlohmann_json::nlohmann_json)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror)

