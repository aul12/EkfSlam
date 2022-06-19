#include <EkfSlamManager.hpp>
#include <cfenv>
#include <fstream>
#include <nlohmann/json.hpp>

namespace nlohmann {
    template<typename Scalar, int Rows, int Cols>
    struct adl_serializer<Eigen::Matrix<Scalar, Rows, Cols>> {
        using Mat = Eigen::Matrix<Scalar, Rows, Cols>;
        static void to_json(json &js, const Mat &mat) {
            for (auto j = 0; j < mat.cols(); ++j) {
                json col;
                for (auto i = 0; i < mat.rows(); ++i) {
                    col.emplace_back(mat(i, j));
                }
                js.emplace_back(col);
            }
        }

        // static void from_json(const json &j, Mat &mat) {}
    };
} // namespace nlohmann

int main() {
    feenableexcept(FE_INVALID | FE_OVERFLOW | FE_DIVBYZERO); // Floating point exceptions
    auto dt = 0.01;

    ekf_slam::VehicleParams vehicleParams{1000000, 1000000, 0.000001, 0.000001};
    ekf_slam::ObjectParams objectParams{0.1, 0.000001};
    ekf_slam::Manager manager{vehicleParams, objectParams};

    std::vector<ekf_slam::Manager::ObjectState> cones;
    for (auto c = 0; c < 100; c += 5) {
        cones.emplace_back(c, 2);
        cones.emplace_back(c, -2);
    }

    ekf_slam::Manager::VehicleState vehicleState{0, 0, 0, 0, 0};
    auto f = ekf_slam::single_track_model::make<double>(dt, 0, 0, 0, 0).f;
    auto coneH = ekf_slam::constant_position_model::make<double>(0, 0).h;

    auto ddPsi = [](auto t) -> double {
        if (t < 1) {
            return 0;
        } else if (t < 2) {
            return 0.1;
        } else {
            return 0;
        }
    };

    auto a = [](auto t) -> double {
        if (t < 1) {
            return 1;
        } else {
            return 0;
        }
    };

    nlohmann::json json;
    for (std::size_t c = 0; c < 15; ++c) {
        vehicleState.dPsi += ddPsi(c * dt);
        vehicleState.v += a(c * dt);
        vehicleState = ekf_slam::Manager::VehicleState(f(vehicleState.getVec()));
        ekf_slam::Manager::VehicleMeas vehicleMeas{vehicleState.v, vehicleState.dPsi};

        std::vector<ekf_slam::Manager::ObjectMeas> conesMeasured;
        for (auto cone : cones) {
            auto coneLocal = ekf_slam::Manager::ObjectMeas{coneH(cone.getVec(), vehicleState.getVec())};
            if (coneLocal.xPos > 0 and coneLocal.xPos < 30) { // In front of the vehicle, max 10m
                conesMeasured.emplace_back(coneLocal);
            }
        }

        auto [vehicle, estimatedCones] = manager.update(vehicleMeas, conesMeasured, dt);

        nlohmann::json snapshot;
        snapshot["vehicle"]["state"] = vehicleState.getVec();
        snapshot["vehicle"]["meas"] = vehicleMeas.getVec();
        snapshot["vehicle"]["est"] = vehicle.getVec();
        for (const auto &cone : estimatedCones) {
            snapshot["estimatedCones"].emplace_back(cone.getVec());
        }
        for (const auto &cone : conesMeasured) {
            snapshot["measuredCones"].emplace_back(cone.getVec());
        }
        for (const auto &cone : cones) {
            snapshot["cones"].emplace_back(cone.getVec());
        }
        json.emplace_back(snapshot);

        std::ofstream ofstream{"result.json", std::ios_base::trunc};
        ofstream << json.dump(4) << std::endl;
        ofstream.close();
    }
}
