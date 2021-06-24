#include <cfenv>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>

#include "EkfSlamManager.hpp"

auto main() -> int {
    feenableexcept(FE_INVALID | FE_OVERFLOW | FE_DIVBYZERO); // Floating point exceptions
    auto dt = 0.1;

    ekf_slam::VehicleParams vehicle_params{1000, 1000, 100, 1};
    ekf_slam::ObjectParams object_params{0, 1};
    ekf_slam::Manager manager{vehicle_params, object_params};

    std::vector<ekf_slam::Manager::ObjectState> cones;
    for (auto c = 0; c < 100; c += 20) {
        cones.emplace_back(c, 2);
        cones.emplace_back(c, -2);
    }

    ekf_slam::Manager::VehicleState vehicle_state{0, 0, 0, 0, 0};
    auto f = ekf_slam::single_track_model::make<double>(dt, 0, 0, 0, 0).f;
    auto cone_h = ekf_slam::constant_position_model::make<double>(0, 0).h;

    auto dd_psi = [](double t) -> double {
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
        }
        return 0;
    };

    for (std::size_t c = 0; c < 1000; ++c) {
        vehicle_state.set_d_psi(vehicle_state.get_d_psi() + dd_psi(c * dt));
        vehicle_state.set_v(vehicle_state.get_v() + a(c * dt));
        vehicle_state = ekf_slam::Manager::VehicleState(f(vehicle_state.get_vec()));
        ekf_slam::Manager::VehicleMeas vehicle_meas{vehicle_state.get_v(), vehicle_state.get_d_psi()};

        std::vector<ekf_slam::Manager::ObjectMeas> cones_measured;
        for (auto cone : cones) {
            auto cone_local = ekf_slam::Manager::ObjectMeas{cone_h(cone.get_vec(), vehicle_state.get_vec())};
            if (cone_local.get_x_pos() > 0) {
                cones_measured.emplace_back(cone_local);
            }
        }
        std::cout << cones_measured.size() << std::endl;
        auto result = manager.update(vehicle_meas, cones_measured, dt);
        ekf_slam::Manager::VehicleState vehicle = result.first;
        std::vector<ekf_slam::Manager::ObjectState> estimated_cones = result.second;
        std::cout << std::setw(5) << std::setprecision(1) << std::fixed;
        std::cout << "State: " << vehicle_state.get_vec().transpose() << "\tMeas:" << vehicle_meas.get_vec().transpose()
                  << "\tEst:" << vehicle.get_vec().transpose() << "\tNumber of Objects\t" << estimated_cones.size()
                  << std::endl;

        /*for (const auto &estimatedCone : estimatedCones) {
            std::cout << "\t[" << estimatedCone.xPos << ", " << estimatedCone.yPos << "]" << std::endl;
        }*/


        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}
