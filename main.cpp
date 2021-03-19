#include <cfenv>
#include <chrono>
#include <iostream>
#include <thread>

#include "EkfSlamManager.hpp"

int main() {
    feenableexcept(FE_INVALID | FE_OVERFLOW | FE_DIVBYZERO); // Floating point exceptions
    auto dt = 0.1;

    ekf_slam::VehicleParams vehicleParams{0.1, 0.1, 1000, 1000};
    ekf_slam::ObjectParams objectParams{1, 1};
    ekf_slam::Manager manager{vehicleParams, objectParams};

    std::vector<ekf_slam::Manager::ObjectState> cones;
    cones.emplace_back(1, 1);
    cones.emplace_back(2, 1);
    ekf_slam::Manager::VehicleState vehicleState{0, 0, 0, 0, 0};
    auto f = ekf_slam::single_track_model::make<double>(dt, 0, 0, 0, 0).f;
    auto coneH = ekf_slam::constant_position_model::make<double>(0, 0).h;

    auto ddPsi = [](auto t) -> double {
        if (t < 1) {
            return 0;
        } else if (t < 2) {
            return 0.1;
        } else {
            return -0 - 1;
        }
    };

    auto a = [](auto t) -> double {
        if (t < 1) {
            return 1;
        } else {
            return 0;
        }
    };

    for (std::size_t c = 0; c < 1000; ++c) {
        vehicleState.dPsi += ddPsi(c * dt);
        vehicleState.v += a(c * dt);
        vehicleState =
                ekf_slam::Manager::VehicleState{f(static_cast<ekf_slam::Manager::VehicleState::Vec>(vehicleState))};
        ekf_slam::Manager::VehicleMeas vehicleMeas{vehicleState.v, vehicleState.dPsi};

        std::vector<ekf_slam::Manager::ObjectMeas> conesMeasured;
        for (auto cone : cones) {
            auto coneLocal = ekf_slam::Manager::ObjectMeas{
                    coneH(static_cast<ekf_slam::Manager::ObjectState::Vec>(cone),
                          static_cast<ekf_slam::Manager::VehicleState::Vec>(vehicleState))};
            if (coneLocal.xPos > 0) {
                conesMeasured.emplace_back(coneLocal);
            }
        }

        auto [vehicle, cones] = manager.update(vehicleMeas, conesMeasured, dt);
        std::cout << "State:" << static_cast<ekf_slam::Manager::VehicleState::Vec>(vehicleState).transpose()
                  << "\tMeas:" << static_cast<ekf_slam::Manager::VehicleMeas::Vec>(vehicleMeas).transpose()
                  << "\tEst:" << static_cast<ekf_slam::Manager::VehicleState::Vec>(vehicle).transpose()
                  << "\tNumber of Objects\t"
                  << cones.size() << std::endl;

        using namespace std::chrono_literals;
        std::this_thread::sleep_for(100ms);
    }
}
