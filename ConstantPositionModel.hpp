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

namespace ekf_slam { namespace constant_position_model {
    template<typename T>
    struct State {
        T xPos, yPos;
        static constexpr std::size_t DIM = 2;
        using Vec = Eigen::Matrix<T, DIM, 1>;
        using Mat = Eigen::Matrix<T, DIM, DIM>;

        State(T xPos, T yPos) : xPos{xPos}, yPos{yPos} {};

        explicit State(Vec x) : xPos{x(0)}, yPos{x(1)} {};

        Vec get_vec() const {
            Vec ret{};
            ret(0) = xPos;
            ret(1) = yPos;
            return ret;
        }
    };

    template<typename T>
    using Meas = State<T>;

    template<typename T>
    auto make(T q, T r) -> ObjectDynamicContainer<State<T>::DIM, Meas<T>::DIM, single_track_model::State<T>::DIM, T> {
        ObjectDynamicContainer<State<T>::DIM, Meas<T>::DIM, single_track_model::State<T>::DIM, T>
                object_dynamic_container;
        object_dynamic_container.f = [](typename State<T>::Vec x) -> typename State<T>::Vec { return x; }; // State remains the same
        object_dynamic_container.j_f = [](typename State<T>::Vec x) ->
                typename State<T>::Mat { return State<T>::Mat::Zero(); }; // Jacobian is zero
        object_dynamic_container.q_func = [q](typename State<T>::Vec x) ->
                typename State<T>::Mat { return State<T>::Mat::Identity() * q; }; // Equal noise on both coordinates

        // Conversion to local coordinates
        object_dynamic_container.h = [](typename State<T>::Vec x_obj, typename single_track_model::State<T>::Vec x_vehicle) -> typename Meas<T>::Vec {
            State<T> obj{x_obj};
            single_track_model::State<T> vehicle{x_vehicle};
            auto dx = obj.xPos - vehicle.xPos;
            auto dy = obj.yPos - vehicle.yPos;
            Meas<T> meas{std::cos(-vehicle.psi) * dx - std::sin(-vehicle.psi) * dy,
                         std::sin(-vehicle.psi) * dx + std::cos(-vehicle.psi) * dy};

            return meas.get_vec();
        };

        object_dynamic_container.j_h_object = [](typename State<T>::Vec x_obj, typename single_track_model::State<T>::Vec x_vehicle) -> typename Meas<T>::Mat {
            single_track_model::State<T> vehicle{x_vehicle};
            typename Meas<T>::Mat j_h{};
            // clang-format off
            j_h << std::cos(-vehicle.psi), -std::sin(-vehicle.psi),
                   std::sin(-vehicle.psi), std::cos(vehicle.psi);
            // clang-format on
            return j_h;
        };

        object_dynamic_container.j_h_vehicle = [](typename State<T>::Vec x_obj, typename single_track_model::State<T>::Vec x_vehicle) -> Eigen::Matrix<T, Meas<T>::DIM, single_track_model::State<T>::DIM> {
            State<T> obj{x_obj};
            single_track_model::State<T> vehicle{x_vehicle};
            auto dx = obj.xPos - vehicle.xPos;
            auto dy = obj.yPos - vehicle.yPos;
            Eigen::Matrix<T, Meas<T>::DIM, single_track_model::State<T>::DIM> j_h;
            // clang-format off
            j_h << std::cos(-vehicle.psi) * (-1), -std::sin(-vehicle.psi) * (-1), 0, -std::sin(-vehicle.psi) * (-1) * dx - std::cos(-vehicle.psi) * dy * (-1), 0,
                   std::sin(-vehicle.psi) * (-1), std::cos(-vehicle.psi) * (-1),  0, std::cos(-vehicle.psi) * (-1) * dx + (-std::sin(-vehicle.psi)) * dy * (-1), 0;
            // clang-format on
            return j_h;
        };

        object_dynamic_container.r_func = [r]() -> typename Meas<T>::Mat {
            return Meas<T>::Mat::Identity() * r;
        }; // Equal measurement noise on both coordinates

        return object_dynamic_container;
    }

    template<typename T>
    auto get_initial_position(typename Meas<T>::Vec z, typename single_track_model::State<T>::Vec xVehicle) ->
            typename State<T>::Vec {
        Meas<T> object{z};
        single_track_model::State<T> vehicle{xVehicle};
        State<T> new_state{object.xPos * std::cos(vehicle.psi) - object.yPos * std::sin(vehicle.psi) + vehicle.xPos,
                       object.xPos * std::sin(vehicle.psi) + object.yPos * std::cos(vehicle.psi) + vehicle.yPos};

        return new_state.get_vec();
    }

    template<typename T>
    auto get_initial_covariance(typename Meas<T>::Vec z, typename single_track_model::State<T>::Vec xVehicle,
                              typename single_track_model::State<T>::Mat pVehicle) -> typename State<T>::Mat {
        return pVehicle.block(0, 0, 2, 2);
    }

} } // namespace ekf_slam::constant_position_model

#endif // EKFSLAM_CONSTANTPOSITIONMODEL_HPP
