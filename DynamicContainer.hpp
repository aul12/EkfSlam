/**
 * @file DynamicContainer.hpp.h
 * @author paul
 * @date 17.03.21
 * Description here TODO
 */
#ifndef EKFSLAM_DYNAMICCONTAINER_HPP
#define EKFSLAM_DYNAMICCONTAINER_HPP

#include <cstdint>

namespace ekf_slam {
    template<std::size_t STATE_DIM, std::size_t MEAS_DIM, typename T>
    struct DynamicContainer {
        using X = Eigen::Matrix<T, STATE_DIM, 1>;
        using P = Eigen::Matrix<T, STATE_DIM, STATE_DIM>;
        using Q = Eigen::Matrix<T, STATE_DIM, STATE_DIM>;

        using Z = Eigen::Matrix<T, MEAS_DIM, 1>;
        using R = Eigen::Matrix<T, MEAS_DIM, MEAS_DIM>;

        using C = Eigen::Matrix<T, MEAS_DIM, STATE_DIM>;
    };

    template<std::size_t STATE_DIM, std::size_t MEAS_DIM, typename T>
    struct VehicleDynamicContainer : public DynamicContainer<STATE_DIM, MEAS_DIM, T> {
        using Dynamic = DynamicContainer<STATE_DIM, MEAS_DIM, T>;
        using typename Dynamic::C;
        using typename Dynamic::P;
        using typename Dynamic::Q;
        using typename Dynamic::R;
        using typename Dynamic::X;
        using typename Dynamic::Z;

        using F_func = std::function<auto(X)->X>;
        using dF_func = std::function<auto(X)->P>;

        using Q_func = std::function<auto(X)->Q>;

        using H_func = std::function<auto(X)->Z>;
        using dH_func = std::function<auto(X)->C>;

        using R_func = std::function<auto()->R>;

        F_func f;
        dF_func j_f;
        Q_func q_func;

        H_func h;
        dH_func j_h;
        R_func r_func;
    };

    template<std::size_t STATE_DIM, std::size_t MEAS_DIM, typename VehicleState, typename T>
    struct ObjectDynamicContainer : public DynamicContainer<STATE_DIM, MEAS_DIM, T> {
        using Dynamic = DynamicContainer<STATE_DIM, MEAS_DIM, T>;
        using typename Dynamic::C;
        using typename Dynamic::P;
        using typename Dynamic::Q;
        using typename Dynamic::R;
        using typename Dynamic::X;
        using typename Dynamic::Z;

        using F_func = std::function<auto(X)->X>;
        using dF_func = std::function<auto(X)->P>;

        using Q_func = std::function<auto(X)->Q>;

        using H_func = std::function<auto(X, VehicleState)->Z>;
        using dH_func = std::function<auto(X, VehicleState)->C>;

        using R_func = std::function<auto()->R>;

        F_func f;
        dF_func j_f;
        Q_func q_func;

        H_func h;
        dH_func j_h; // TODO block vehicle column, object row
        R_func r_func;
    };


} // namespace ekf_slam

#endif // EKFSLAM_DYNAMICCONTAINER_HPP
