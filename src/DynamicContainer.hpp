/**
 * @headerfile DynamicContainer.hpp
 * @author paul
 * @date 17.03.21
 * Description here TODO
 */
#ifndef EKFSLAM_DYNAMICCONTAINER_HPP
#define EKFSLAM_DYNAMICCONTAINER_HPP

#include <Eigen/Eigen>
#include <cstdint>
#include <functional>

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

    template<std::size_t STATE_DIM, std::size_t MEAS_DIM, std::size_t VEHICLE_STATE_DIM, typename T>
    struct ObjectDynamicContainer : public DynamicContainer<STATE_DIM, MEAS_DIM, T> {
        using Dynamic = DynamicContainer<STATE_DIM, MEAS_DIM, T>;
        using typename Dynamic::C;
        using typename Dynamic::P;
        using typename Dynamic::Q;
        using typename Dynamic::R;
        using typename Dynamic::X;
        using typename Dynamic::Z;
        using CVehicle = Eigen::Matrix<T, MEAS_DIM, VEHICLE_STATE_DIM>;
        using XVehicle = Eigen::Matrix<T, VEHICLE_STATE_DIM, 1>;

        using F_func = std::function<auto(X)->X>;
        using dF_func = std::function<auto(X)->P>;

        using Q_func = std::function<auto(X)->Q>;

        using H_func = std::function<auto(X, XVehicle)->Z>;
        using dH_Object_func = std::function<auto(X, XVehicle)->C>;
        using dH_Vehicle_func = std::function<auto(X, XVehicle)->CVehicle>;

        using R_func = std::function<auto()->R>;

        F_func f;
        dF_func j_f;
        Q_func q_func;

        H_func h;
        dH_Object_func j_h_object;
        dH_Vehicle_func j_h_vehicle;
        R_func r_func;
    };


} // namespace ekf_slam

#endif // EKFSLAM_DYNAMICCONTAINER_HPP
