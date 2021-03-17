#include <cfenv>
#include <chrono>
#include <iostream>
#include <thread>

#include "EkfSlamManager.hpp"

int main() {
    feenableexcept(FE_INVALID | FE_OVERFLOW | FE_DIVBYZERO); // Floating point exceptions

    ekf_slam::VehicleParams vehicleParams{0.1, 0.1, 1000 , 1000};
    ekf_slam::ObjectParams objectParams{1, 1};
    ekf_slam::Manager manager{vehicleParams, objectParams};

    for (std::size_t c=0; c<1000; ++c) {
        using namespace std::chrono_literals;
        ekf_slam::Manager::VehicleMeas vehicleMeas{1, 0};
        auto [vehicle, cones] = manager.update(vehicleMeas, {}, 0.1);
        std::cout
                    << c * 0.1
                    << "\tMeas:\t " << static_cast<ekf_slam::Manager::VehicleMeas::Vec>(vehicleMeas).transpose()
                    << "\tEst\t" << static_cast<ekf_slam::Manager::VehicleState::Vec>(vehicle).transpose()
                    << "\tNumber of Objects\t"  << cones.size() << std::endl;
        std::this_thread::sleep_for(100ms);

    }
}
