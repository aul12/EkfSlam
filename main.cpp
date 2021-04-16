#include <cfenv>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>

#include "EkfSlamManager.hpp"

int main() {
    feenableexcept(FE_INVALID | FE_OVERFLOW | FE_DIVBYZERO); // Floating point exceptions
    auto dt = 0.1;

    ekf_slam::VehicleParams vehicleParams{1000, 1000, 100, 1};
    ekf_slam::ObjectParams objectParams{0, 1};
    ekf_slam::Manager manager{vehicleParams, objectParams};

    std::vector<ekf_slam::Manager::ObjectState> cones;
    for (auto c = 0; c < 100; c += 20) {
        cones.emplace_back(c, 2);
        cones.emplace_back(c, -2);
    }

    ekf_slam::Manager::VehicleState vehicleState{0, 0, 0, 0, 0};
    auto f = ekf_slam::single_track_model::make<double>(dt, 0, 0, 0, 0).f;
    auto coneH = ekf_slam::constant_position_model::make<double>(0, 0).h;

    auto ddPsi = [](double t) -> double {
        return 0;
        if (t < 1) {
            return 0;
        } else if (t < 2) {
            return 0.1;
        } else {
            return -0 - 1;
        }
    };

    auto a = [](double t) -> double {
        if (t < 1) {
            return 1;
        } else {
            return 0;
        }
    };

    for (std::size_t c = 0; c < 1000; ++c) {
        vehicleState.dPsi += ddPsi(c * dt);
        vehicleState.v += a(c * dt);
        vehicleState = ekf_slam::Manager::VehicleState(f(vehicleState.getVec()));
        ekf_slam::Manager::VehicleMeas vehicleMeas{vehicleState.v, vehicleState.dPsi};

        std::vector<ekf_slam::Manager::ObjectMeas> conesMeasured;
        for (auto cone : cones) {
            auto coneLocal = ekf_slam::Manager::ObjectMeas{coneH(cone.getVec(), vehicleState.getVec())};
            if (coneLocal.xPos > 0) {
                conesMeasured.emplace_back(coneLocal);
            }
        }

        auto result = manager.update(vehicleMeas, conesMeasured, dt);
        ekf_slam::Manager::VehicleState vehicle = result.first;
        std::vector<ekf_slam::Manager::ObjectState> estimatedCones = result.second;
        std::cout << std::setw(5) << std::setprecision(1) << std::fixed;
        std::cout << "State: " << vehicleState.getVec().transpose()
                  << "\tMeas:" << vehicleMeas.getVec().transpose()
                  << "\tEst:" << vehicle.getVec().transpose()
                  << "\tNumber of Objects\t"
                  << estimatedCones.size() << std::endl;

        /*for (const auto &estimatedCone : estimatedCones) {
            std::cout << "\t[" << estimatedCone.xPos << ", " << estimatedCone.yPos << "]" << std::endl;
        }*/


        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}
