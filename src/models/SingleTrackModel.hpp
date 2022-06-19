/**
 * @headerfile SingleTrackModel.hpp
 * @author paul
 * @date 17.03.21
 * Description here TODO
 */
#ifndef EKFSLAM_SINGLETRACKMODEL_HPP
#define EKFSLAM_SINGLETRACKMODEL_HPP

#include <Eigen/Eigen>
#include <cmath>

#include "../DynamicContainer.hpp"
#include "../Util.hpp"

namespace ekf_slam::models {
    template<typename T>
    struct single_track {
        struct State {
            T xPos, yPos;
            T v;
            T psi, dPsi;

            static constexpr std::size_t DIM = 5;
            using Vec = Eigen::Matrix<T, DIM, 1>;
            using Mat = Eigen::Matrix<T, DIM, DIM>;

            State(T xPos, T yPos, T v, T psi, T dPsi) : xPos{xPos}, yPos{yPos}, v{v}, psi{psi}, dPsi{dPsi} {};

            explicit State(Vec x) : xPos{x(0)}, yPos{x(1)}, v{x(2)}, psi{x(3)}, dPsi{x(4)} {};

            Vec getVec() const {
                Vec ret{};
                ret(0) = xPos;
                ret(1) = yPos;
                ret(2) = v;
                ret(3) = psi;
                ret(4) = dPsi;
                return ret;
            }
        };

        struct Meas {
            T v, dPsi;

            static constexpr std::size_t DIM = 2;
            using Vec = Eigen::Matrix<T, DIM, 1>;
            using Mat = Eigen::Matrix<T, DIM, DIM>;

            Meas(T v, T dPsi) : v{v}, dPsi{dPsi} {};

            explicit Meas(Vec z) : v(z(0)), dPsi(z(1)){};

            Vec getVec() const {
                Vec ret{};
                ret(0) = v;
                ret(1) = dPsi;
                return ret;
            }
        };

        struct Params {
            T sigmaA2, sigmaDDPsi2, sigmaV2, sigmaDPsi2;
        };

        static auto make(const T &dt, T sigmaA2, T sigmaDDPsi2, T sigmaV2, T sigmaDPsi2) {
            VehicleDynamicContainer<State::DIM, Meas::DIM, T> vehicleDynamicContainer;
            vehicleDynamicContainer.f = [&dt](auto x) -> typename State::Vec {
                State state{x};
                // clang-format off
                State newState{state.xPos + std::cos(state.psi) * state.v * dt,
                              state.yPos + std::sin(state.psi) * state.v * dt,
                              state.v,
                              state.psi + state.dPsi * dt,
                              state.dPsi};
                // clang-format on
                return newState.getVec();
            };

            vehicleDynamicContainer.j_f = [&dt](auto x) -> typename State::Mat {
                State state{x};
                typename State::Mat j_f;
                // clang-format off
                j_f <<
                    1, 0, std::cos(state.psi) * dt, -std::sin(state.psi) * state.v * dt, 0,
                    0, 1, std::sin(state.psi) * dt, std::cos(state.psi) * state.v * dt , 0,
                    0, 0, 1, 0, 0,
                    0, 0, 0, 1, dt,
                    0, 0, 0, 0, 1;
                // clang-format on
                return j_f;
            };

            vehicleDynamicContainer.q_func = [&dt, sigmaA2, sigmaDDPsi2](auto x) -> typename State::Mat {
                State state{x};
                Eigen::Matrix<T, 3, 1> GammaA;
                // clang-format off
                GammaA <<
                        0.5 * dt * dt * std::cos(state.psi),
                        0.5 * dt * dt * std::sin(state.psi),
                        dt;
                // clang-format on
                Eigen::Matrix<T, 2, 1> GammaDDPsi;
                // clang-format off
                GammaDDPsi <<
                        0.5 * dt * dt,
                        dt;
                // clang-format on
                Eigen::Matrix<T, 3, 3> Q_a = GammaA * GammaA.transpose() * sigmaA2;
                Eigen::Matrix<T, 2, 2> Q_DDPsi = GammaDDPsi * GammaDDPsi.transpose() * sigmaDDPsi2;
                ASSERT_COV(Q_a)
                ASSERT_COV(Q_DDPsi)
                Eigen::Matrix<T, 5, 5> Q = Eigen::Matrix<T, 5, 5>::Zero();
                Q.block(0, 0, 3, 3) = Q_a;
                Q.block(3, 3, 2, 2) = Q_DDPsi;
                ASSERT_COV(Q)
                return Q;
            };

            vehicleDynamicContainer.h = [](auto x) -> typename Meas::Vec {
                State state{x};
                Meas meas{state.v, state.dPsi};
                return meas.getVec();
            };

            vehicleDynamicContainer.j_h = [](auto x) -> Eigen::Matrix<T, Meas::DIM, State::DIM> {
                Eigen::Matrix<T, Meas::DIM, State::DIM> c = Eigen::Matrix<T, Meas::DIM, State::DIM>::Zero();
                // clang-format off
                c <<
                    0, 0, 1, 0, 0,
                    0, 0, 0, 0, 1;
                // clang-format on
                return c;
            };

            vehicleDynamicContainer.r_func = [sigmaV2, sigmaDPsi2]() -> typename Meas::Mat {
                Eigen::Matrix<T, Meas::DIM, Meas::DIM> R = Eigen::Matrix<T, Meas::DIM, Meas::DIM>::Zero();
                // clang-format off
                R <<
                    sigmaV2, 0,
                    0, sigmaDPsi2;
                // clang-format on
                ASSERT_COV(R)
                return R;
            };

            return vehicleDynamicContainer;
        }
    };
} // namespace ekf_slam::models

#endif // EKFSLAM_SINGLETRACKMODEL_HPP
