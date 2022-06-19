# EKF-Slam

## Including the library using FetchContent
The library can be automatically downloaded using CMakes FetchContent Module:
```cmake
include(FetchContent)

FetchContent_Declare(ekf_slam
        GIT_REPOSITORY "https://github.com/aul12/EkfSlam.git")
FetchContent_MakeAvailable(ekf_slam)

target_link_libraries(foo PRIVATE EKFSlam)
```

## Example
