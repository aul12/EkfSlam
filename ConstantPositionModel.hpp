/**
 * @file ConstantPositionModel.hpp.h
 * @author paul
 * @date 17.03.21
 * Description here TODO
 */
#ifndef EKFSLAM_CONSTANTPOSITIONMODEL_HPP
#define EKFSLAM_CONSTANTPOSITIONMODEL_HPP

#include <Eigen/Eigen>
#include <cmath>

#include "SingleTrackModel.hpp"

namespace ekf_slam::constant_position_model {
    template<typename T>
    struct State {
        T xPos, yPos;
        static constexpr std::size_t DIM = 2;
        using Vec = Eigen::Matrix<T, DIM, 1>;
        using Mat = Eigen::Matrix<T, DIM, DIM>;

        State(T xPos, T yPos) : xPos{xPos}, yPos{yPos} {};

        explicit State(Vec x) : xPos{x(0)}, yPos{x(1)} {};

        explicit operator Vec() {
            Vec ret{};
            ret(0) = xPos;
            ret(1) = yPos;
            return ret;
        }
    };

    template<typename T>
    using Meas = State<T>;

    template<typename T>
    auto make(T q, T r) {
        auto f = [](auto x) { return x; };                                   // State remains the same
        auto J_F = [](auto x) { return State<T>::Mat::Zero(); };             // Jacobian is zero
        auto Q_func = [q](auto x) { return State<T>::Mat::Identity() * q; }; // Equal noise on both coordinates

        // Conversion to local coordinates
        auto h = [](auto x_obj, auto x_vehicle) {
            State obj{x_obj};
            single_track_model::State vehicle{x_vehicle};
            auto dx = obj.xPos - vehicle.xPos;
            auto dy = obj.yPos - vehicle.yPos;
            Meas<T> meas{std::cos(-vehicle.psi) * dx - std::sin(-vehicle.psi) * dy,
                         std::sin(-vehicle.psi) * dx + std::cos(-vehicle.psi) * dy};

            return static_cast<typename Meas<T>::Vec>(meas);
        };

        auto J_H = [](auto x_obj, auto x_vehicle) {
            State obj{x_obj};
            single_track_model::State vehicle{x_vehicle};
            typename Meas<T>::Mat j_h{};
            // clang-format off
            j_h << std::cos(-vehicle.psi), -std::sin(-vehicle.psi),
                   std::sin(-vehicle.psi), std::cos(vehicle.psi);
            // clang-format on
            return j_h;
        };

        auto R_func = [r]() { return Meas<T>::Mat::Identity() * r; }; // Equal measurement noise on both coordinates

        return Dynamic<State<T>::DIM, Meas<T>::DIM, typename single_track_model::State<T>::Vec, T>{f, J_F, Q_func,
                                                                                                   h, J_H, R_func};
    }

    template<typename T>
    auto getInitialPosition(typename Meas<T>::Vec z, typename single_track_model::State<T>::Vec xVehicle) -> typename State<T>::Vec {
        Meas<T> object{z};
        single_track_model::State vehicle{xVehicle};
        State newState{object.xPos * std::cos(vehicle.psi) - object.yPos * std::sin(vehicle.psi) + vehicle.xPos,
                       object.xPos * std::sin(vehicle.psi) + object.yPos * std::cos(vehicle.psi) + vehicle.yPos};

        return static_cast<typename State<T>::Vec>(newState);
    }

    template<typename T>
    auto getInitialCovariance(typename Meas<T>::Vec z, typename single_track_model::State<T>::Vec xVehicle,
                              typename single_track_model::State<T>::Mat pVehicle) -> typename State<T>::Mat {
        single_track_model::State vehicle{xVehicle};
        return typename State<T>::Mat{Eigen::Rotation2D<T>(vehicle.psi) * pVehicle.block(0, 0, 2, 2)};
    }

} // namespace ekf_slam::constant_position_model

#endif // EKFSLAM_CONSTANTPOSITIONMODEL_HPP
