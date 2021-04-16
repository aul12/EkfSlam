/**
 * @file SingleTrackModel.hpp.h
 * @author paul
 * @date 17.03.21
 * Description here TODO
 */
#ifndef EKFSLAM_SINGLETRACKMODEL_HPP
#define EKFSLAM_SINGLETRACKMODEL_HPP

#include "DynamicContainer.hpp"
#include "Util.hpp"

namespace ekf_slam {
    namespace single_track_model {
        template<typename T>
        struct State {
          private:
            T xPos, yPos;
            T v;
            T psi, dPsi;

          public:
            static constexpr std::size_t DIM = 5;
            using Vec = Eigen::Matrix<T, DIM, 1>;
            using Mat = Eigen::Matrix<T, DIM, DIM>;

            State(T xPos, T yPos, T v, T psi, T dPsi) : xPos{xPos}, yPos{yPos}, v{v}, psi{psi}, dPsi{dPsi} {};

            explicit State(Vec x) : xPos{x(0)}, yPos{x(1)}, v{x(2)}, psi{x(3)}, dPsi{x(4)} {};

            auto get_vec() const -> Vec {
                Vec ret{};
                ret(0) = xPos;
                ret(1) = yPos;
                ret(2) = v;
                ret(3) = psi;
                ret(4) = dPsi;
                return ret;
            }
            
            auto get_psi() const -> T {
                return this->psi;
            }
            auto get_d_psi() const -> T {
                return this->dPsi;
            }
            auto get_x_pos() const -> T {
                return this->xPos;
            }
            auto get_y_pos() const -> T {
                return this->yPos;
            }
            auto get_v() const -> T {
                return this->v;
            }

            void set_d_psi(T val) {
                this->dPsi = val;
            }

            void set_v(T val) {
                this->v = val;
            }
        };

        template<typename T>
        struct Meas {
          private:
            T v, dPsi;

          public:
            static constexpr std::size_t DIM = 2;
            using Vec = Eigen::Matrix<T, DIM, 1>;
            using Mat = Eigen::Matrix<T, DIM, DIM>;

            Meas(T v, T dPsi) : v{v}, dPsi{dPsi} {};

            explicit Meas(Vec z) : v(z(0)), dPsi(z(1)){};

            auto get_vec() const -> Vec {
                Vec ret{};
                ret(0) = v;
                ret(1) = dPsi;
                return ret;
            }

            auto get_v() const -> T {
                return this->v;
            }

            auto get_d_psi() const -> T {
                return this->dPsi;
            }
        };

        template<typename T>
        auto make(const T &dt, T sigmaA2, T sigmaDDPsi2, T sigmaV2, T sigmaDPsi2)
                -> VehicleDynamicContainer<State<T>::DIM, Meas<T>::DIM, T> {
            VehicleDynamicContainer<State<T>::DIM, Meas<T>::DIM, T> vehicle_dynamic_container;
            vehicle_dynamic_container.f = [&dt](typename State<T>::Vec x) -> typename State<T>::Vec {
                State<T> state{x};
                // clang-format off
                State<T> new_state{state.get_x_pos() + std::cos(state.get_psi()) * state.get_v() * dt,
                                  state.get_y_pos() + std::sin(state.get_psi()) * state.get_v() * dt,
                                  state.get_v(),
                                  state.get_psi() + state.get_d_psi() * dt,
                                  state.get_d_psi()};
                // clang-format on
                return new_state.get_vec();
            };

            vehicle_dynamic_container.j_f = [&dt](typename State<T>::Vec x) -> typename State<T>::Mat {
                State<T> state{x};
                typename State<T>::Mat j_f;
                // clang-format off
                j_f <<
                    1, 0, std::cos(state.get_psi()) * dt, -std::sin(state.get_psi()) * state.get_v() *dt, 0,
                    0, 1, std::sin(state.get_psi()) * dt, std::cos(state.get_psi()) * state.get_v() * dt , 0,
                    0, 0, 1, 0, 0,
                    0, 0, 0, 1, dt,
                    0, 0, 0, 0, 1;
                // clang-format on
                return j_f;
            };

            vehicle_dynamic_container.q_func = [&dt, sigmaA2, sigmaDDPsi2](typename State<T>::Vec x) ->
                    typename State<T>::Mat {
                        State<T> state{x};
                        Eigen::Matrix<T, 3, 1> gamma_a;
                        // clang-format off
                        gamma_a <<
                                0.5 * dt * dt * std::cos(state.get_psi()),
                                0.5 * dt * dt * std::sin(state.get_psi()),
                                dt;
                        // clang-format on
                        Eigen::Matrix<T, 2, 1> gamma_dd_psi;
                        // clang-format off
                        gamma_dd_psi <<
                                0.5 * dt * dt,
                                dt;
                        // clang-format on
                        Eigen::Matrix<T, 3, 3> q_a = gamma_a * gamma_a.transpose() * sigmaA2;
                        Eigen::Matrix<T, 2, 2> q_dd_psi = gamma_dd_psi * gamma_dd_psi.transpose() * sigmaDDPsi2;
                        ASSERT_COV(q_a);
                        ASSERT_COV(q_dd_psi);
                        Eigen::Matrix<T, 5, 5> q = Eigen::Matrix<T, 5, 5>::Zero();
                        q.block(0, 0, 3, 3) = q_a;
                        q.block(3, 3, 2, 2) = q_dd_psi;

                        return q;
                    };

            vehicle_dynamic_container.h = [](typename State<T>::Vec x) -> typename Meas<T>::Vec {
                State<T> state{x};
                Meas<T> meas{state.get_v(), state.get_psi()};
                return meas.get_vec();
            };

            vehicle_dynamic_container.j_h =
                    [](typename State<T>::Vec /*x*/) -> Eigen::Matrix<T, Meas<T>::DIM, State<T>::DIM> {
                Eigen::Matrix<T, Meas<T>::DIM, State<T>::DIM> c = Eigen::Matrix<T, Meas<T>::DIM, State<T>::DIM>::Zero();
                // clang-format off
                c <<
                    0, 0, 1, 0, 0,
                    0, 0, 0, 0, 1;
                // clang-format on
                return c;
            };

            vehicle_dynamic_container.r_func = [sigmaV2, sigmaDPsi2]() -> typename Meas<T>::Mat {
                Eigen::Matrix<T, Meas<T>::DIM, Meas<T>::DIM> r = Eigen::Matrix<T, Meas<T>::DIM, Meas<T>::DIM>::Zero();
                // clang-format off
                r <<    sigmaV2, 0,
                        0, sigmaDPsi2;
                // clang-format on
                return r;
            };

            return vehicle_dynamic_container;
        }
    } // namespace single_track_model
} // namespace ekf_slam

#endif // EKFSLAM_SINGLETRACKMODEL_HPP
