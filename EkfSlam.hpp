/**
 * @headerfile EkSlam.hpp
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

#include "DynamicContainer.hpp"
#include "Util.hpp"

namespace ekf_slam {
    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    class EKFSlam {
      public:
        // Type declarations
        using VehicleDynamic = VehicleDynamicContainer<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, T>;
        using ObjectDynamic = ObjectDynamicContainer<OBJECT_STATE_DIM, OBJECT_MEAS_DIM, VEHICLE_STATE_DIM, T>;

        using X = Eigen::Matrix<T, Eigen::Dynamic, 1>;
        using P = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
        using Z = Eigen::Matrix<T, Eigen::Dynamic, 1>;
        using S = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

        using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
        using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

        using ObjectMeasurements = std::vector<typename ObjectDynamic::Z>;
        using TrackList = std::vector<std::pair<typename ObjectDynamic::X, typename ObjectDynamic::P>>;

        using InitialEstFunc =
                std::function<auto(typename ObjectDynamic::Z, typename VehicleDynamic::X)->typename ObjectDynamic::X>;
        using InitialCovFunc =
                std::function<auto(typename ObjectDynamic::Z, typename VehicleDynamic::X, typename VehicleDynamic::P)
                                      ->typename ObjectDynamic::P>; // R is added separately

        // Functions to be used by user
        EKFSlam(VehicleDynamic vehicleDynamic, ObjectDynamic objectDynamic, InitialEstFunc initialEstFunc,
                InitialCovFunc initialCovFunc, typename VehicleDynamic::X initialEst = VehicleDynamic::X::Zero(),
                typename VehicleDynamic::P initialCov = VehicleDynamic::P::Identity());

        void update(typename VehicleDynamic::Z vehicleMeasurement, ObjectMeasurements objectMeasurements);

        auto getVehicle() const -> typename VehicleDynamic::X;

        auto getObject(std::size_t id) const -> typename ObjectDynamic::X;

        [[nodiscard]] auto getNumberOfObjects() const -> std::size_t;

      private:
        [[nodiscard]] auto predict(X x, P p) const -> std::pair<X, P>;

        [[nodiscard]] auto measure(X x, P p) const -> std::pair<Z, S>;

        auto slam(X x, P p, typename VehicleDynamic::Z vehicleMeas, ObjectMeasurements measurements) const
                -> std::pair<X, P>;

        auto addTracks(X x, P p, typename VehicleDynamic::Z vehicleMeas, ObjectMeasurements measurements) const
                -> std::pair<X, P>;

        auto separateTracksNotMeasured(X x, P p, ObjectMeasurements objectMeasurements) const
                -> std::tuple<X, P, TrackList>;

        auto dataAssociation(Z z, S s, ObjectMeasurements measurements) const -> std::map<std::size_t, std::size_t>;

        // Helper functions single objects <-> vector conversion
        auto getdf(const X &x) const;
        auto getQ(const X &x) const;
        auto getdh(const X &x) const;
        auto getR(const X &x) const;
        static auto x_v(X &x);
        static auto x_v(const X &x);
        static auto numObjects(const X &x);
        static auto x_o(X &x, std::size_t i);
        static auto x_o(const X &x, std::size_t i);

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
            typename VehicleDynamic::Z vehicleMeasurement, ObjectMeasurements objectMeasurements) {
        // Predict all objects, independent of their visibility, this leads to an increase in covariance over time
        std::tie(lastX, lastP) = predict(lastX, lastP);

        // The innovation step is combined with the track management
        std::tie(lastX, lastP) = slam(lastX, lastP, vehicleMeasurement, objectMeasurements);

        // Just some plausibility checks
        ASSERT_COV(lastP);
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::predict(X x, P p) const
            -> std::pair<X, P> {
        // x_hat = f(x)
        x_v(x) = vehicleDynamic.f(x_v(x));
        for (auto c = 0U; c < numObjects(x); ++c) {
            x_o(x, c) = objectDynamic.f(x_o(x, c));
        }

        // P_hat = dF * P * dF^T + q
        auto df = getdf(x);
        auto q = getQ(x);
        p = df * p * df.transpose() + q;
        ASSERT_COV(p);

        return std::make_pair(x, p);
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::measure(X x, P p) const
            -> std::pair<Z, S> {
        // z = h(x)
        Z z = Z::Zero(VEHICLE_MEAS_DIM + numObjects(x) * OBJECT_MEAS_DIM);
        z.block(0, 0, VEHICLE_MEAS_DIM, 1) = vehicleDynamic.h(x_v(x));
        for (auto c = 0U; c < numObjects(x); ++c) {
            z.block(VEHICLE_MEAS_DIM + c * OBJECT_MEAS_DIM, 0, OBJECT_MEAS_DIM, 1) = objectDynamic.h(x_o(x, c), x_v(x));
        }

        // s = dh * P * dh^T + r
        auto dh = getdh(x);
        auto r = getR(x);
        S s = dh * p * dh.transpose() + r;
        ASSERT_COV(s);

        return std::make_pair(z, s);
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::slam(
            X x, P p, typename VehicleDynamic::Z vehicleMeas, ObjectMeasurements measurements) const
            -> std::pair<X, P> {

        // New Tracks
        std::tie(x, p) = addTracks(x, p, vehicleMeas, measurements);

        // Temporary remove tracks not visible
        TrackList invisibleObjects;
        std::tie(x, p, invisibleObjects) = separateTracksNotMeasured(x, p, measurements);

        assert(numObjects(x) == measurements.size());

        auto [z_hat, s] = measure(x, p);

        // Data association
        auto associationMap = dataAssociation(z_hat, s, measurements);

        // Reorder z to match x
        ObjectMeasurements reorderedMeasurements(measurements.size());
        for (auto c = 0U; c < measurements.size(); ++c) {
            reorderedMeasurements[c] = measurements[associationMap[c]];
        }

        // Build Z Vector
        Z z(VEHICLE_MEAS_DIM + reorderedMeasurements.size() * OBJECT_MEAS_DIM);
        z.block(0, 0, VEHICLE_MEAS_DIM, 1) = vehicleMeas;
        for (auto c = 0U; c < reorderedMeasurements.size(); ++c) {
            z.block(VEHICLE_MEAS_DIM + c * OBJECT_MEAS_DIM, 0, OBJECT_MEAS_DIM, 1) = reorderedMeasurements[c];
        }

        // Calculate Kalman Gain
        Mat K = p * getdh(x).transpose() * s.inverse();

        // Innovation
        Vec tildeZ = z - z_hat;
        x = x + K * tildeZ;
        p = p - K * s * K.transpose();
        ASSERT_COV(p);

        // Readd tracks
        X completeX = X::Zero(x.size() + invisibleObjects.size() * OBJECT_STATE_DIM);
        P completeP = P::Zero(p.rows() + invisibleObjects.size() * OBJECT_STATE_DIM,
                              p.cols() + invisibleObjects.size() * OBJECT_STATE_DIM);
        completeX.block(0, 0, x.size(), 1) = x;
        completeP.block(0, 0, p.rows(), p.cols()) = p;
        for (auto c = 0U; c < invisibleObjects.size(); ++c) {
            auto offset = x.size() + c * OBJECT_STATE_DIM;
            completeX.block(offset, 0, OBJECT_STATE_DIM, 1) = invisibleObjects[c].first;
            completeP.block(offset, offset, OBJECT_STATE_DIM, OBJECT_STATE_DIM) = invisibleObjects[c].second;
        }

        ASSERT_COV(completeP);

        return std::make_pair(completeX, completeP);
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::addTracks(
            X x, P p, typename VehicleDynamic::Z vehicleMeas, ObjectMeasurements measurements) const
            -> std::pair<X, P> {
        auto [z_hat, s] = measure(x, p);

        // We do not update in place as this can result in missing cones if a new cone can be matched to two
        // measurements
        X updatedX = x;
        P updatedP = p;

        for (const auto &z : measurements) {
            auto minMhd = std::numeric_limits<T>::max();
            for (auto c = 0U; c < numObjects(x); ++c) {
                auto offset = VEHICLE_MEAS_DIM + c * OBJECT_MEAS_DIM;
                auto z_track = z_hat.block(offset, 0, OBJECT_MEAS_DIM, 1);
                S cov = s.block(offset, offset, OBJECT_MEAS_DIM, OBJECT_MEAS_DIM);
                auto z_tilde = z_track - z;

                ASSERT_COV(cov);

                auto mhd2 = z_tilde.transpose() * cov.inverse() * z_tilde;
                assert(mhd2.rows() == 1 and mhd2.cols() == 1);
                auto mhd = std::sqrt(mhd2(0, 0));

                if (mhd < minMhd) {
                    minMhd = mhd;
                }
            }

            if (minMhd > gate) {
                auto initialEstimate = initialEstFunc(z, x_v(x));
                Mat initialCov = initialCovFunc(z, x_v(x), p.block(0, 0, VEHICLE_STATE_DIM, VEHICLE_STATE_DIM)) +
                                 vehicleDynamic.r_func();

                ASSERT_COV(initialCov);

                X extendedX = X::Zero(updatedX.size() + OBJECT_STATE_DIM);
                P extendedP = P::Zero(updatedP.rows() + OBJECT_STATE_DIM, updatedP.cols() + OBJECT_STATE_DIM);
                extendedX.block(0, 0, updatedX.size(), 1) = updatedX;
                extendedX.block(updatedX.size(), 0, OBJECT_STATE_DIM, 1) = initialEstimate;
                extendedP.block(0, 0, updatedX.size(), updatedX.size()) = updatedP;
                extendedP.block(updatedX.size(), updatedX.size(), OBJECT_STATE_DIM, OBJECT_STATE_DIM) = initialCov;

                updatedX = extendedX;
                updatedP = extendedP;
                ASSERT_COV(updatedP);
                // std::tie(z_hat, s) = measure(x, p);
            }
        }

        return std::make_pair(updatedX, updatedP);
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::separateTracksNotMeasured(
            EKFSlam::X x, EKFSlam::P p, ObjectMeasurements objectMeasurements) const -> std::tuple<X, P, TrackList> {

        auto [z_hat, s] = measure(x, p);

        std::vector<std::pair<typename ObjectDynamic::X, typename ObjectDynamic::P>> invisibleObjects;
        for (auto c = 0U; c < numObjects(x); ++c) {
            auto minMhd = std::numeric_limits<T>::max();

            for (const auto &z_o : objectMeasurements) {
                auto offset = VEHICLE_MEAS_DIM + c * OBJECT_MEAS_DIM;
                auto z_track = z_hat.block(offset, 0, OBJECT_MEAS_DIM, 1);
                auto cov = s.block(offset, offset, OBJECT_MEAS_DIM, OBJECT_MEAS_DIM);
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

                ASSERT_COV(p_track);

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
                x.conservativeResize(x.size() - OBJECT_STATE_DIM);
                p.conservativeResize(p.rows() - OBJECT_STATE_DIM, p.cols() - OBJECT_STATE_DIM);
                ASSERT_COV(p);
                std::tie(z_hat, s) = measure(x, p);

                // We eliminated this track, thus we now need to check the track at this index (the following track)
                // again
                c -= 1;
            }
        }

        return std::make_tuple(x, p, invisibleObjects);
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::dataAssociation(
            Z z_hat, S s, ObjectMeasurements measurements) const -> std::map<std::size_t, std::size_t> {
        // i is track, j is measurement
        Mat associationMatrix{measurements.size(), measurements.size()};

        const auto epsilon = 1 / (z_hat.size() + 1.);

        for (auto i = 0U; i < measurements.size(); ++i) {
            for (auto j = 0U; j < measurements.size(); ++j) {
                auto offset = VEHICLE_MEAS_DIM + i * OBJECT_MEAS_DIM;
                auto z_track = z_hat.block(offset, 0, OBJECT_MEAS_DIM, 1);
                auto cov = s.block(offset, offset, OBJECT_MEAS_DIM, OBJECT_MEAS_DIM);
                auto z_tilde = z_track - measurements[j];

                ASSERT_COV(cov);

                auto mhd2 = z_tilde.transpose() * cov.inverse() * z_tilde;

                assert(mhd2.rows() == 1 and mhd2.cols() == 1);

                associationMatrix(i, j) = gate - mhd2(0, 0);
            }
        }

        // Auction algorithm
        std::vector<T> trackPrices(measurements.size());
        std::vector<bool> observationMapped(measurements.size());

        // track index -> measurement index
        std::map<std::size_t, std::size_t> map;

        auto count = 0U;
        while (true) {
            ++count;
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
            auto maxVal = std::numeric_limits<T>::lowest();

            for (auto i = 0U; i < measurements.size(); ++i) {
                auto val = associationMatrix(i, j) - trackPrices[i];
                if (val > maxVal) {
                    iMax = i;
                    maxVal = val;
                }
            }

            if (map.find(iMax) != map.cend()) {
                observationMapped[map[iMax]] = false;
            }
            map[iMax] = j;

            auto secondIMax = 0;
            auto secondVal = std::numeric_limits<T>::lowest();

            for (auto i = 0U; i < measurements.size(); ++i) {
                auto val = associationMatrix(i, j) - trackPrices[i];
                if (val > secondVal and secondIMax != iMax) {
                    secondIMax = i;
                    secondVal = val;
                }
            }
            auto y_i = maxVal - secondVal;
            trackPrices[iMax] += y_i + epsilon;
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
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::getdf(const X &x) const {
        Mat J_F = Mat::Zero(x.size(), x.size());

        J_F.block(0, 0, VEHICLE_STATE_DIM, VEHICLE_STATE_DIM) = vehicleDynamic.j_f(x_v(x));

        for (auto c = 0U; c < numObjects(x); ++c) {
            auto offset = VEHICLE_STATE_DIM + c * OBJECT_STATE_DIM;
            J_F.block(offset, offset, OBJECT_STATE_DIM, OBJECT_STATE_DIM) = objectDynamic.j_f(x_o(x, c));
        }

        return J_F;
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::getQ(const X &x) const {
        Mat Q = Mat::Zero(x.size(), x.size());

        Q.block(0, 0, VEHICLE_STATE_DIM, VEHICLE_STATE_DIM) = vehicleDynamic.q_func(x_v(x));
        ASSERT_COV(Q);

        for (auto c = 0U; c < numObjects(x); ++c) {
            auto offset = VEHICLE_STATE_DIM + c * OBJECT_STATE_DIM;
            Q.block(offset, offset, OBJECT_STATE_DIM, OBJECT_STATE_DIM) = objectDynamic.q_func(x_o(x, c));
            ASSERT_COV(Q);
        }

        return Q;
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::getdh(const X &x) const {
        Mat J_H = Mat::Zero(VEHICLE_MEAS_DIM + numObjects(x) * OBJECT_MEAS_DIM, x.size());
        J_H.block(0, 0, VEHICLE_MEAS_DIM, VEHICLE_STATE_DIM) = vehicleDynamic.j_h(x_v(x));

        for (auto c = 0U; c < numObjects(x); ++c) {
            J_H.block(VEHICLE_MEAS_DIM + c * OBJECT_MEAS_DIM, VEHICLE_STATE_DIM + c * OBJECT_STATE_DIM, OBJECT_MEAS_DIM,
                      OBJECT_STATE_DIM) = objectDynamic.j_h_object(x_o(x, c), x_v(x));
            J_H.block(VEHICLE_MEAS_DIM + c * OBJECT_MEAS_DIM, 0, OBJECT_MEAS_DIM, VEHICLE_STATE_DIM) =
                    objectDynamic.j_h_vehicle(x_o(x, c), x_v(x));
        }

        return J_H;
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::getR(const X &x) const {
        Mat R = Mat::Zero(VEHICLE_MEAS_DIM + numObjects(x) * OBJECT_MEAS_DIM,
                          VEHICLE_MEAS_DIM + numObjects(x) * OBJECT_MEAS_DIM);
        R.block(0, 0, VEHICLE_MEAS_DIM, VEHICLE_MEAS_DIM) = vehicleDynamic.r_func();

        for (auto c = 0U; c < numObjects(x); ++c) {
            R.block(VEHICLE_MEAS_DIM + c * OBJECT_MEAS_DIM, VEHICLE_MEAS_DIM + c * OBJECT_MEAS_DIM, OBJECT_MEAS_DIM,
                    OBJECT_MEAS_DIM) = objectDynamic.r_func();
        }

        ASSERT_COV(R);

        return R;
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::x_v(X &x) {
        return x.block(0, 0, VEHICLE_STATE_DIM, 1);
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::x_v(const X &x) {
        return x.block(0, 0, VEHICLE_STATE_DIM, 1);
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::numObjects(const X &x) {
        assert((x.size() - VEHICLE_STATE_DIM) % OBJECT_STATE_DIM == 0);
        return (x.size() - VEHICLE_STATE_DIM) / OBJECT_STATE_DIM;
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::x_o(X &x, std::size_t i) {
        assert(i < numObjects(x));
        return x.block(VEHICLE_STATE_DIM + i * OBJECT_STATE_DIM, 0, OBJECT_STATE_DIM, 1);
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::x_o(const X &x,
                                                                                                 std::size_t i) {
        assert(i < numObjects(x));
        return x.block(VEHICLE_STATE_DIM + i * OBJECT_STATE_DIM, 0, OBJECT_STATE_DIM, 1);
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::getNumberOfObjects() const
            -> std::size_t {
        return numObjects(lastX);
    }
} // namespace ekf_slam

#endif // EKFSLAM_EKFSLAM_HPP
