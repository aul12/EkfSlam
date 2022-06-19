/**
 * @headerfile ConstantPositionModel.hpp
 * @author paul
 * @date 17.03.21
 * Description here TODO
 */
#ifndef EKFSLAM_CONSTANTPOSITIONMODEL_HPP
#define EKFSLAM_CONSTANTPOSITIONMODEL_HPP

#include <Eigen/Eigen>
#include <cmath>

#include "../DynamicContainer.hpp"
#include "SingleTrackModel.hpp"

namespace ekf_slam::models {
    template<typename T>
    struct constant_position {
        struct State {
            T xPos, yPos;
            static constexpr std::size_t DIM = 2;
            using Vec = Eigen::Matrix<T, DIM, 1>;
            using Mat = Eigen::Matrix<T, DIM, DIM>;

            State(T xPos, T yPos) : xPos{xPos}, yPos{yPos} {};

            explicit State(Vec x) : xPos{x(0)}, yPos{x(1)} {};

            Vec getVec() const {
                Vec ret{};
                ret(0) = xPos;
                ret(1) = yPos;
                return ret;
            }
        };

        using Meas = State;

        struct Params {
            T sigmaPos2, sigmaMeas;
        };

        static auto make(T q, T r) {
            ObjectDynamicContainer<State::DIM, Meas::DIM, single_track<T>::State::DIM, T> objectDynamicContainer;
            objectDynamicContainer.f = [](auto x) -> typename State::Vec { return x; }; // State remains the same
            objectDynamicContainer.j_f = [](auto x) ->
                    typename State::Mat { return State::Mat::Identity(); }; // Jacobian is identity
            objectDynamicContainer.q_func = [q](auto x) ->
                    typename State::Mat { return State::Mat::Identity() * q; }; // Equal noise on both coordinates

            // Conversion to local coordinates
            objectDynamicContainer.h = [](auto x_obj, auto x_vehicle) -> typename Meas::Vec {
                State obj{x_obj};
                typename single_track<T>::State vehicle{x_vehicle};
                auto dx = obj.xPos - vehicle.xPos;
                auto dy = obj.yPos - vehicle.yPos;
                Meas meas{std::cos(-vehicle.psi) * dx - std::sin(-vehicle.psi) * dy,
                          std::sin(-vehicle.psi) * dx + std::cos(-vehicle.psi) * dy};

                return meas.getVec();
            };

            objectDynamicContainer.j_h_object = [](auto x_obj, auto x_vehicle) -> typename Meas::Mat {
                typename single_track<T>::State vehicle{x_vehicle};
                typename Meas::Mat j_h{};
                // clang-format off
                j_h << std::cos(-vehicle.psi), -std::sin(-vehicle.psi),
                       std::sin(-vehicle.psi), std::cos(vehicle.psi);
                // clang-format on
                return j_h;
            };

            objectDynamicContainer.j_h_vehicle = [](auto x_obj, auto x_vehicle) {
                State obj{x_obj};
                typename single_track<T>::State vehicle{x_vehicle};
                auto dx = obj.xPos - vehicle.xPos;
                auto dy = obj.yPos - vehicle.yPos;
                Eigen::Matrix<T, Meas::DIM, single_track<T>::State::DIM> j_h;
                // clang-format off
                j_h << std::cos(-vehicle.psi) * (-1), -std::sin(-vehicle.psi) * (-1), 0, -std::sin(-vehicle.psi) * (-1) * dx - std::cos(-vehicle.psi) * dy * (-1), 0,
                       std::sin(-vehicle.psi) * (-1), std::cos(-vehicle.psi) * (-1),  0, std::cos(-vehicle.psi) * (-1) * dx + (-std::sin(-vehicle.psi)) * dy * (-1), 0;
                // clang-format on
                return j_h;
            };

            objectDynamicContainer.r_func = [r]() -> typename Meas::Mat {
                return Meas::Mat::Identity() * r;
            }; // Equal measurement noise on both coordinates

            return objectDynamicContainer;
        }

        static auto getInitialPosition(typename Meas::Vec z, typename single_track<T>::State::Vec xVehicle) ->
                typename State::Vec {
            Meas object{z};
            typename single_track<T>::State vehicle{xVehicle};
            State newState{object.xPos * std::cos(vehicle.psi) - object.yPos * std::sin(vehicle.psi) + vehicle.xPos,
                           object.xPos * std::sin(vehicle.psi) + object.yPos * std::cos(vehicle.psi) + vehicle.yPos};

            return newState.getVec();
        }

        static auto getInitialCovariance(typename Meas::Vec z, typename single_track<T>::State::Vec xVehicle,
                                         typename single_track<T>::State::Mat pVehicle) -> typename State::Mat {
            return pVehicle.block(0, 0, 2, 2);
        }
    };

} // namespace ekf_slam::models

#endif // EKFSLAM_CONSTANTPOSITIONMODEL_HPP
