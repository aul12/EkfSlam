# EKF-SLAM

## Architecture

The EKF Slam library is structured into multiple parts to allow for maximum flexibility:

* The `slam` module is the pure mathematical implementation of an EKF-SLAM
* The `models` module provides the definition of system descriptions for both the vehicle and tracked objects
* The `manager` module allows for the combination of the `EKfSlam` module with use selected models
* The `association` module allows for the definition of different assosciation strategies for the slam module

## Including the library using FetchContent

The library can be automatically downloaded using CMakes FetchContent Module:

```cmake
include(FetchContent)

FetchContent_Declare(ekf_slam
        GIT_REPOSITORY "https://github.com/aul12/EkfSlam.git")
FetchContent_MakeAvailable(ekf_slam)

target_link_libraries(foo PRIVATE EKFSlam)
```
