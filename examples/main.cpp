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

using AdditionalData = ekf_slam::models::Color;

int main() {
    std::map<AdditionalData, std::string> colorMap{{AdditionalData::YELLOW, "yellow"},
                                                   {AdditionalData::BLUE, "blue"},
                                                   {AdditionalData::ORANGE, "orange"}};

    feenableexcept(FE_INVALID | FE_OVERFLOW | FE_DIVBYZERO); // Floating point exceptions
    auto dt = 0.01;

    ekf_slam::Manager<>::Vehicle::Params vehicleParams{1e9, 1e6, 1e-8, 1e-8};
    ekf_slam::Manager<>::Object::Params objectParams{1e0, 1e-8};
    ekf_slam::Manager manager{vehicleParams, objectParams};

    std::vector<std::pair<ekf_slam::Manager<>::Object::State, AdditionalData>> cones;
    for (auto c = 3; c < 100; c += 10) {
        cones.emplace_back(ekf_slam::Manager<>::Object::State{static_cast<double>(c), 2}, AdditionalData::BLUE);
        cones.emplace_back(ekf_slam::Manager<>::Object::State{static_cast<double>(c), -2}, AdditionalData::YELLOW);
    }

    ekf_slam::Manager<>::Vehicle::State vehicleState{0, 0, 0, 0, 0};
    auto f = ekf_slam::models::single_track<double>::make(dt, 0, 0, 0, 0).f;
    auto coneH = ekf_slam::models::constant_position<double>::make(0, 0).h;

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
    for (std::size_t c = 0; c < 200; ++c) {
        std::cout << c << std::endl;
        vehicleState.dPsi += ddPsi(c * dt);
        vehicleState.v += a(c * dt);
        vehicleState = ekf_slam::Manager<>::Vehicle::State(f(vehicleState.getVec()));
        ekf_slam::Manager<>::Vehicle::Meas vehicleMeas{vehicleState.v, vehicleState.dPsi};

        std::vector<std::pair<ekf_slam::Manager<>::Object::Meas, AdditionalData>> conesMeasured;
        for (auto cone : cones) {
            auto coneLocal = ekf_slam::Manager<>::Object::Meas{coneH(cone.first.getVec(), vehicleState.getVec())};
            if (coneLocal.xPos > 0 and coneLocal.xPos < 30) { // In front of the vehicle, max 30m
                conesMeasured.emplace_back(coneLocal, cone.second);
            }
        }

        auto [vehicle, estimatedCones] = manager.update(vehicleMeas, conesMeasured, dt);

        nlohmann::json snapshot;
        snapshot["vehicle"]["state"] = vehicleState.getVec();
        snapshot["vehicle"]["meas"] = vehicleMeas.getVec();
        snapshot["vehicle"]["est"] = vehicle.getVec();
        for (const auto &cone : estimatedCones) {
            nlohmann::json jsonCone;
            jsonCone["state"] = cone.first.getVec();
            jsonCone["color"] = colorMap[cone.second];
            snapshot["estimatedCones"].emplace_back(jsonCone);
        }
        for (const auto &cone : conesMeasured) {
            nlohmann::json jsonCone;
            jsonCone["state"] = cone.first.getVec();
            jsonCone["color"] = colorMap[cone.second];
            snapshot["measuredCones"].emplace_back(jsonCone);
        }
        for (const auto &cone : cones) {
            nlohmann::json jsonCone;
            jsonCone["state"] = cone.first.getVec();
            jsonCone["color"] = colorMap[cone.second];
            snapshot["cones"].emplace_back(jsonCone);
        }
        json.emplace_back(snapshot);

        std::ofstream ofstream{"result.json", std::ios_base::trunc};
        ofstream << json.dump(4) << std::endl;
        ofstream.close();
    }
}
