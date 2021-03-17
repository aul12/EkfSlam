/**
 * @file Dynamic.hpp.h
 * @author paul
 * @date 17.03.21
 * Description here TODO
 */
#ifndef EKFSLAM_DYNAMIC_HPP
#define EKFSLAM_DYNAMIC_HPP

#include <cstdint>

namespace ekf_slam {
    struct EmptyType {};

    template<std::size_t STATE_DIM, std::size_t MEAS_DIM, typename FuncParam, typename T>
    class Dynamic {
      public:
        using X = Eigen::Matrix<T, STATE_DIM, 1>;
        using P = Eigen::Matrix<T, STATE_DIM, STATE_DIM>;
        using Q = P;

        using Z = Eigen::Matrix<T, MEAS_DIM, 1>;
        using R = Eigen::Matrix<T, MEAS_DIM, MEAS_DIM>;

        using C = Eigen::Matrix<T, MEAS_DIM, STATE_DIM>;

        using F = std::function<auto(X)->X>;
        using J_F = std::function<auto(X)->P>;

        using Q_func = std::function<auto(X)->Q>;

        using H = std::function<auto(X, FuncParam)->Z>;
        using J_H = std::function<auto(X, FuncParam)->C>;

        using R_func = std::function<auto()->R>;

        F f;
        J_F j_f;
        Q_func q_func;

        H h;
        J_H j_h; // TODO block vehicle column, object row
        R_func r_func;
    };
} // namespace ekf_slam

#endif // EKFSLAM_DYNAMIC_HPP
