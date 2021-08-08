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

namespace ekf_slam {
    namespace constant_position_model {
        template<typename T>
        struct State {
          private:
            T xPos, yPos;

          public:
            static constexpr std::size_t DIM = 2;
            using Vec = Eigen::Matrix<T, DIM, 1>;
            using Mat = Eigen::Matrix<T, DIM, DIM>;

            State(T xPos, T yPos) : xPos{xPos}, yPos{yPos} {};

            explicit State(Vec x) : xPos{x(0)}, yPos{x(1)} {};

            auto get_vec() const -> Vec {
                Vec ret{};
                ret(0) = get_x_pos();
                ret(1) = get_y_pos();
                return ret;
            }

            auto get_x_pos() const -> T {
                return this->xPos;
            }

            auto get_y_pos() const -> T {
                return this->yPos;
            }
        };

        template<typename T>
        using Meas = State<T>;

        template<typename T>
        auto make(T q, T r)
                -> ObjectDynamicContainer<State<T>::DIM, Meas<T>::DIM, single_track_model::State<T>::DIM, T> {
            ObjectDynamicContainer<State<T>::DIM, Meas<T>::DIM, single_track_model::State<T>::DIM, T>
                    object_dynamic_container;
            object_dynamic_container.f = [](typename State<T>::Vec x) ->
                    typename State<T>::Vec { return x; }; // State remains the same
            object_dynamic_container.j_f = [](typename State<T>::Vec /*x*/) ->
                    typename State<T>::Mat { return State<T>::Mat::Identity(); }; // Jacobian is Identity
            object_dynamic_container.q_func = [q](typename State<T>::Vec /*x*/) ->
                    typename State<T>::Mat { return State<T>::Mat::Identity() * q; }; // Equal noise on both coordinates

            // Conversion to local coordinates
            object_dynamic_container.h = [](typename State<T>::Vec x_obj,
                                            typename single_track_model::State<T>::Vec x_vehicle) ->
                    typename Meas<T>::Vec {
                        State<T> obj{x_obj};
                        single_track_model::State<T> vehicle{x_vehicle};
                        auto dx = obj.get_x_pos() - vehicle.get_x_pos();
                        auto dy = obj.get_y_pos() - vehicle.get_y_pos();
                        Meas<T> meas{std::cos(-vehicle.get_psi()) * dx - std::sin(-vehicle.get_psi()) * dy,
                                     std::sin(-vehicle.get_psi()) * dx + std::cos(-vehicle.get_psi()) * dy};

                        return meas.get_vec();
                    };

            object_dynamic_container.j_h_object =
                    [](typename State<T>::Vec /*x_obj*/,
                       typename single_track_model::State<T>::Vec x_vehicle) -> typename Meas<T>::Mat {
                single_track_model::State<T> vehicle{x_vehicle};
                typename Meas<T>::Mat j_h{};
                // clang-format off
                j_h << std::cos(-vehicle.get_psi()), -std::sin(-vehicle.get_psi()),
                       std::sin(-vehicle.get_psi()), std::cos(vehicle.get_psi());
                // clang-format on
                return j_h;
            };

            object_dynamic_container.j_h_vehicle = [](typename State<T>::Vec x_obj,
                                                      typename single_track_model::State<T>::Vec x_vehicle)
                    -> Eigen::Matrix<T, Meas<T>::DIM, single_track_model::State<T>::DIM> {
                State<T> obj{x_obj};
                single_track_model::State<T> vehicle{x_vehicle};
                auto dx = obj.get_x_pos() - vehicle.get_x_pos();
                auto dy = obj.get_y_pos() - vehicle.get_y_pos();
                Eigen::Matrix<T, Meas<T>::DIM, single_track_model::State<T>::DIM> j_h;
                // clang-format off
                j_h <<  std::cos(-vehicle.get_psi()) * (-1), -std::sin(-vehicle.get_psi()) * (-1), 0, -std::sin(-vehicle.get_psi()) * (-1) * dx - std::cos(-vehicle.get_psi()) * dy * (-1), 0,
                        std::sin(-vehicle.get_psi()) * (-1), std::cos(-vehicle.get_psi()) * (-1),  0, std::cos(-vehicle.get_psi()) * (-1) * dx + (-std::sin(-vehicle.get_psi())) * dy * (-1), 0;
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
            State<T> new_state{object.get_x_pos() * std::cos(vehicle.get_psi()) - object.get_y_pos() * std::sin(vehicle.get_psi()) + vehicle.get_x_pos(),
                               object.get_x_pos() * std::sin(vehicle.get_psi()) + object.get_y_pos() * std::cos(vehicle.get_psi()) +
                                       vehicle.get_y_pos()};

            return new_state.get_vec();
        }

        template<typename T>
        auto get_initial_covariance(typename Meas<T>::Vec /*z*/,
                                    typename single_track_model::State<T>::Vec /*xVehicle*/,
                                    typename single_track_model::State<T>::Mat pVehicle) -> typename State<T>::Mat {
            return pVehicle.block(0, 0, 2, 2);
        }

    } // namespace constant_position_model
} // namespace ekf_slam

#endif // EKFSLAM_CONSTANTPOSITIONMODEL_HPP
