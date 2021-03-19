/**
 * @file EkSlam.hpp.h
 * @author paul
 * @date 17.03.21
 * Description here TODO
 */
#ifndef EKFSLAM_EKFSLAM_HPP
#define EKFSLAM_EKFSLAM_HPP

#include <Eigen/Eigen>
#include <cstdint>
#include <iostream>
#include <numeric>

#include "Dynamic.hpp"

namespace ekf_slam {
    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    class EKFSlam {
      public:
        using VehicleDynamic = Dynamic<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, EmptyType, T>;
        using ObjectDynamic = Dynamic<OBJECT_STATE_DIM, OBJECT_MEAS_DIM, typename VehicleDynamic::X, T>;

        using X = Eigen::VectorXd;
        using P = Eigen::MatrixXd;
        using Z = Eigen::VectorXd;
        using S = Eigen::MatrixXd;
        using Measurements = std::vector<typename ObjectDynamic::Z>;

        using InitialEstFunc =
                std::function<auto(typename ObjectDynamic::Z, typename VehicleDynamic::X)->typename ObjectDynamic::X>;
        using InitialCovFunc =
                std::function<auto(typename ObjectDynamic::Z, typename VehicleDynamic::X, typename VehicleDynamic::P)
                                      ->typename ObjectDynamic::P>; // R is added separately

        EKFSlam(VehicleDynamic vehicleDynamic, ObjectDynamic objectDynamic, InitialEstFunc initialEstFunc,
                InitialCovFunc initialCovFunc, typename VehicleDynamic::X initialEst = VehicleDynamic::X::Zero(),
                typename VehicleDynamic::P initialCov = VehicleDynamic::P::Identity());

        void update(typename VehicleDynamic::Z z_v, Measurements measurements);

        auto getVehicle() const -> typename VehicleDynamic::X;

        auto getObject(std::size_t id) const -> typename ObjectDynamic::X;

        auto getNumberOfObjects() const -> std::size_t;

      private:
        auto predict(X x, P p) const -> std::pair<X, P>;

        auto measure(X x, P p) const -> std::pair<Z, S>;

        auto slam(X x, P p, typename VehicleDynamic::Z vehicleMeas, Measurements measurements) const -> std::pair<X, P>;

        auto dataAssociation(Z z, S s, Measurements measurements) const -> std::map<std::size_t, std::size_t>;

        // Helper functions single objects <-> vector conversion
        auto getJf(X x) const;
        auto getQ(X x) const;
        auto getJH(X x) const;
        auto getR(X x) const;
        auto x_v(X x) const;
        auto numObjects(X x) const;
        auto x_o(X x, std::size_t i) const;

        // Constant member
        VehicleDynamic vehicleDynamic;
        ObjectDynamic objectDynamic;
        InitialEstFunc initialEstFunc;
        InitialCovFunc initialCovFunc;
        T gate = 2;

        // Actual prediction
        X lastX;
        P lastP;
    };

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::EKFSlam(
            EKFSlam::VehicleDynamic vehicleDynamic, EKFSlam::ObjectDynamic objectDynamic, InitialEstFunc initialEstFunc,
            InitialCovFunc initialCovFunc, typename VehicleDynamic::X initialEst,
            typename VehicleDynamic::P initialCov) :
        vehicleDynamic{vehicleDynamic},
        objectDynamic{objectDynamic},
        initialEstFunc{initialEstFunc},
        initialCovFunc{initialCovFunc},
        lastX{initialEst},
        lastP{initialCov} {
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    void EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::update(
            typename VehicleDynamic::Z z_v, Measurements measurements) {
        std::tie(lastX, lastP) = predict(lastX, lastP);
        std::tie(lastX, lastP) = slam(lastX, lastP, z_v, measurements);
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::predict(X x, P p) const
            -> std::pair<X, P> {
        // x_hat = f(x)
        x_v(x) = vehicleDynamic.f(x_v(x));
        for (auto c = 0U; c<numObjects(x); ++c) {
            x_o(x, c) = objectDynamic.f(x_o(x, c));
        }

        // P_hat = J_F * P * J_F^T + Q
        auto J_F = getJf(x);
        auto Q = getQ(x);
        p = J_F * p * J_F.transpose() + Q;

        return std::make_pair(x, p);
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::measure(X x, P p) const
            -> std::pair<Z, S> {
        // z = h(x)
        Eigen::VectorXd z = Eigen::VectorXd::Zero(VEHICLE_MEAS_DIM + numObjects(x) * OBJECT_MEAS_DIM);
        z.block(0, 0, VEHICLE_MEAS_DIM, 1) = vehicleDynamic.h(x_v(x), EmptyType{});
        for (auto c = 0U; c < numObjects(x); ++c) {
            z.block(VEHICLE_MEAS_DIM + c * OBJECT_MEAS_DIM, 0, OBJECT_MEAS_DIM, 1) = objectDynamic.h(x_o(x, c), x_v(x));
        }

        // S = J_H * P * J_H^T + R
        auto J_H = getJH(x);
        auto R = getR(x);
        auto S = J_H * p * J_H.transpose() + R;

        return std::make_pair(z, S);
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::slam(
            X x, P p, typename VehicleDynamic::Z vehicleMeas, Measurements measurements) const -> std::pair<X, P> {
        auto [z_hat, S] = measure(x, p);

        // New Tracks
        for (const auto &z : measurements) {
            auto minMhd = std::numeric_limits<T>::max();
            for (auto c = 0U; c < numObjects(x); ++c) {
                auto offset = VEHICLE_MEAS_DIM + c * OBJECT_MEAS_DIM;
                auto z_track = z_hat.block(offset, 0, OBJECT_MEAS_DIM, 1);
                auto cov = S.block(offset, offset, OBJECT_MEAS_DIM, OBJECT_MEAS_DIM); // @TODO check
                auto z_tilde = z_track - z;

                auto mhd2 = z_tilde.transpose() * cov.inverse() * z_tilde;
                assert(mhd2.rows() == 1 and mhd2.cols() == 1);
                auto mhd = std::sqrt(mhd2(0, 0));

                if (mhd < minMhd) {
                    minMhd = mhd;
                }
            }

            if (minMhd > gate) {
                auto initialEstimate = initialEstFunc(z, x_v(x));
                auto initialCov = initialCovFunc(z, x_v(x), p.block<VEHICLE_STATE_DIM, VEHICLE_STATE_DIM>(0, 0)) +
                                  vehicleDynamic.r_func();

                X newX = X::Zero(x.size() + OBJECT_STATE_DIM);
                P newP = P::Zero(p.rows() + OBJECT_STATE_DIM, p.cols() + OBJECT_STATE_DIM);
                newX.block(0, 0, x.size(), 1) = x;
                newX.block(x.size(), 0, OBJECT_STATE_DIM, 1) = initialEstimate;
                newP.block(0, 0, x.size(), x.size()) = p;
                newP.block<OBJECT_STATE_DIM, OBJECT_STATE_DIM>(x.size(), x.size()) = initialCov;

                x = newX;
                p = newP;
                std::tie(z_hat, S) = measure(x, p);
            }
        }

        // Temporary remove tracks not visible
        std::vector<std::pair<typename ObjectDynamic::X, typename ObjectDynamic::P>> invisibleObjects;
        for (auto c = 0U; c < numObjects(x); ++c) {
            auto minMhd = std::numeric_limits<T>::max();

            for (const auto &z_o : measurements) {
                auto offset = VEHICLE_MEAS_DIM + c * OBJECT_MEAS_DIM;
                auto z_track = z_hat.block(offset, 0, OBJECT_MEAS_DIM, 1);
                auto cov = S.block(offset, offset, OBJECT_MEAS_DIM, OBJECT_MEAS_DIM); // @TODO check
                auto z_tilde = z_track - z_o;

                auto mhd2 = z_tilde.transpose() * cov.inverse() * z_tilde;
                assert(mhd2.rows() == 1 and mhd2.cols() == 1);
                auto mhd = std::sqrt(mhd2(0, 0));

                if (mhd < minMhd) {
                    minMhd = mhd;
                }
            }

            if (minMhd > gate) {
                auto offset = VEHICLE_STATE_DIM + c * OBJECT_STATE_DIM;
                auto x_track = x.block(offset, 0, OBJECT_STATE_DIM, 1);
                auto p_track = p.block(offset, offset, OBJECT_STATE_DIM, OBJECT_STATE_DIM);

                invisibleObjects.emplace_back(x_track, p_track);

                // Swap with last track
                auto lastIndex = numObjects(x) - 1;
                if (c != lastIndex) {
                    auto lastOffset = VEHICLE_STATE_DIM + lastIndex * OBJECT_STATE_DIM;
                    x.block(offset, 0, OBJECT_STATE_DIM, 1) = x.block(lastOffset, 0, OBJECT_STATE_DIM, 1);
                    p.block(offset, 0, OBJECT_STATE_DIM, p.cols()) = p.block(lastOffset, 0, OBJECT_STATE_DIM, p.cols());
                    p.block(0, offset, p.rows(), OBJECT_STATE_DIM) = p.block(0, lastOffset, p.rows(), OBJECT_STATE_DIM);
                }

                // Remove last track
                x = x.block(0, 0, x.size() - OBJECT_STATE_DIM, 1);
                p = p.block(0, 0, p.rows() - OBJECT_STATE_DIM, p.cols() - OBJECT_STATE_DIM);
                std::tie(z_hat, S) = measure(x, p);
            }
        }

        assert(numObjects(x) == measurements.size());

        // Data association
        auto associationMap = dataAssociation(z_hat, S, measurements);

        // Reorder z to match x
        Measurements reorderedMeasurements(measurements.size());
        for (auto c = 0U; c < measurements.size(); ++c) {
            reorderedMeasurements[c] = measurements[associationMap[c]];
        }

        // Build Z Vector
        Eigen::VectorXd z(VEHICLE_MEAS_DIM + reorderedMeasurements.size() * OBJECT_MEAS_DIM);
        z.block(0, 0, VEHICLE_MEAS_DIM, 1) = vehicleMeas;
        for (auto c = 0U; c < reorderedMeasurements.size(); ++c) {
            z.block(VEHICLE_MEAS_DIM + c * OBJECT_MEAS_DIM, 0, OBJECT_MEAS_DIM, 1) = reorderedMeasurements[c];
        }

        // Calculate Kalman Gain
        Eigen::MatrixXd K = p * getJH(x).transpose() * S.inverse();

        // Innovation
        Eigen::VectorXd tildeZ = z - z_hat;
        x = x + K * tildeZ;
        p = p - K * S * K.transpose();

        // Readd tracks
        X completeX = X::Zero(x.size() + invisibleObjects.size() * OBJECT_STATE_DIM);
        P completeP = P::Zero(p.rows() + invisibleObjects.size() * OBJECT_STATE_DIM,
                              p.cols() + invisibleObjects.size() * OBJECT_STATE_DIM);
        completeX.block(0, 0, x.size(), 1) = x;
        completeP.block(0, 0, x.size(), x.size()) = p;
        for (auto c = 0U; c < invisibleObjects.size(); ++c) {
            auto offset = x.size() + c * OBJECT_STATE_DIM;
            completeX.block(offset, 0, OBJECT_STATE_DIM, 1) = invisibleObjects[c].first;
            completeP.block(offset, offset, OBJECT_STATE_DIM, OBJECT_STATE_DIM) = invisibleObjects[c].second;
        }

        return std::make_pair(completeX, completeP);
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::dataAssociation(
            Z z_hat, S s, Measurements measurements) const -> std::map<std::size_t, std::size_t> {
        // i is track, j is measurement
        Eigen::MatrixXd associationMatrix{measurements.size(), measurements.size()};
        std::vector<T> prices(measurements.size());
        std::vector<bool> observationMapped(measurements.size());

        // track index -> measurement index
        std::map<std::size_t, std::size_t> map;

        const auto epsilon = 1.0 / (z_hat.size() + 1U);

        for (auto i = 0U; i < measurements.size(); ++i) {
            for (auto j = 0U; j < measurements.size(); ++j) {
                auto offset = VEHICLE_MEAS_DIM + i * OBJECT_MEAS_DIM;
                auto z_track = z_hat.block(offset, 0, OBJECT_MEAS_DIM, 1);
                auto cov = s.block(offset, offset, OBJECT_MEAS_DIM, OBJECT_MEAS_DIM);
                auto z_tilde = z_track - measurements[j];

                auto mhd2 = z_tilde.transpose() * cov.inverse() * z_tilde;

                assert(mhd2.rows() == 1 and mhd2.cols() == 1);

                associationMatrix(i, j) = gate - mhd2(0, 0);
            }
        }

        while (true) {
            auto allMapped = std::reduce(observationMapped.cbegin(), observationMapped.cend(), true,
                                         [](auto a, auto b) { return a and b; });
            if (allMapped) {
                break;
            }

            std::size_t j = 0;
            for (; j < measurements.size(); ++j) {
                if (not observationMapped[j]) {
                    break;
                }
            }

            auto iMax = 0;
            auto maxVal = std::numeric_limits<T>::min();

            for (auto i = 0U; i < measurements.size(); ++i) {
                auto val = associationMatrix(i, j) - prices[i];
                if (val > maxVal) {
                    iMax = i;
                    maxVal = val;
                }
            }

            map[iMax] = j;

            auto secondIMax = 0;
            auto secondVal = std::numeric_limits<T>::min();

            for (auto i = 0U; i < measurements.size(); ++i) {
                auto val = associationMatrix(i, j) - prices[i];
                if (val > secondVal and secondIMax != iMax) {
                    secondIMax = i;
                    secondVal = val;
                }
            }
            auto y_i = maxVal - secondVal;
            prices[iMax] += y_i + epsilon;
            observationMapped[j] = true;
        }

        return map;
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::getVehicle() const ->
            typename VehicleDynamic::X {
        return x_v(lastX);
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto
    EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::getObject(std::size_t id) const
            -> typename ObjectDynamic::X {
        return x_o(lastX, id);
    }

    /*
     * Helper functions
     */
    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::getJf(X x) const {
        Eigen::MatrixXd J_F = Eigen::MatrixXd::Zero(x.size(), x.size());

        J_F.block(0, 0, VEHICLE_STATE_DIM, VEHICLE_STATE_DIM) = vehicleDynamic.j_f(x_v(x));

        for (auto c = 0U; c < numObjects(x); ++c) {
            auto offset = VEHICLE_STATE_DIM + c * OBJECT_STATE_DIM;
            J_F.block(offset, offset, OBJECT_STATE_DIM, OBJECT_STATE_DIM) = objectDynamic.j_f(x_o(x, c));
        }

        return J_F;
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::getQ(X x) const {
        Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(x.size(), x.size());

        Q.block<VEHICLE_STATE_DIM, VEHICLE_STATE_DIM>(0, 0, VEHICLE_STATE_DIM, VEHICLE_STATE_DIM) =
                vehicleDynamic.q_func(x_v(x));

        for (auto c = 0U; c < numObjects(x); ++c) {
            auto offset = VEHICLE_STATE_DIM + c * OBJECT_STATE_DIM;
            Q.block(offset, offset, OBJECT_STATE_DIM, OBJECT_STATE_DIM) = objectDynamic.q_func(x_o(x, c));
        }

        return Q;
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::getJH(X x) const {
        Eigen::MatrixXd J_H = Eigen::MatrixXd::Zero(VEHICLE_MEAS_DIM + numObjects(x) * OBJECT_MEAS_DIM, x.size());
        J_H.block(0, 0, VEHICLE_MEAS_DIM, VEHICLE_STATE_DIM) = vehicleDynamic.j_h(x_v(x), EmptyType{});

        for (auto c = 0U; c < numObjects(x); ++c) {
            J_H.block(VEHICLE_MEAS_DIM + c * OBJECT_MEAS_DIM, VEHICLE_STATE_DIM + c * OBJECT_STATE_DIM, OBJECT_MEAS_DIM,
                      OBJECT_STATE_DIM) = objectDynamic.j_h(x_o(x, c), x_v(x));
        }
        return J_H;
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::getR(X x) const {
        Eigen::MatrixXd R = Eigen::MatrixXd::Zero(VEHICLE_MEAS_DIM + numObjects(x) * OBJECT_MEAS_DIM,
                                                  VEHICLE_MEAS_DIM + numObjects(x) * OBJECT_MEAS_DIM);
        R.block(0, 0, VEHICLE_MEAS_DIM, VEHICLE_MEAS_DIM) = vehicleDynamic.r_func();

        for (auto c = 0U; c < numObjects(x); ++c) {
            R.block(VEHICLE_MEAS_DIM + c * OBJECT_MEAS_DIM, VEHICLE_MEAS_DIM + c * OBJECT_MEAS_DIM, OBJECT_MEAS_DIM,
                    OBJECT_MEAS_DIM) = objectDynamic.r_func();
        }
        return R;
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::x_v(X x) const {
        return x.block<VEHICLE_STATE_DIM, 1>(0, 0);
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::numObjects(X x) const {
        assert((x.size() - VEHICLE_STATE_DIM) % OBJECT_STATE_DIM == 0);
        return (x.size() - VEHICLE_STATE_DIM) / OBJECT_STATE_DIM;
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::x_o(X x,
                                                                                                 std::size_t i) const {
        assert(i < numObjects(x));
        return x.block<OBJECT_STATE_DIM, 1>(VEHICLE_STATE_DIM + i * OBJECT_STATE_DIM, 0, OBJECT_STATE_DIM, 1);
    }
    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::getNumberOfObjects() const
            -> std::size_t {
        return numObjects(lastX);
    }
} // namespace ekf_slam

#endif // EKFSLAM_EKFSLAM_HPP
