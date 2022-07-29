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
#include <set>

#include "DynamicContainer.hpp"
#include "Util.hpp"

namespace ekf_slam {
    struct AssociationResult {
        using TrackId = std::size_t;
        using MeasId = std::size_t;

        std::map<TrackId, MeasId> track2Measure;
        std::vector<MeasId> newTracks;
        std::vector<TrackId> tracksToDelete; // @TODO set
    };

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T, typename AdditionalData>
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

        using ObjectMeasurements = std::vector<std::pair<typename ObjectDynamic::Z, AdditionalData>>;
        using TrackList = std::vector<std::pair<typename ObjectDynamic::X, typename ObjectDynamic::P>>;

        using InitialEstFunc =
                std::function<auto(typename ObjectDynamic::Z, typename VehicleDynamic::X)->typename ObjectDynamic::X>;
        using InitialCovFunc =
                std::function<auto(typename ObjectDynamic::Z, typename VehicleDynamic::X, typename VehicleDynamic::P)
                                      ->typename ObjectDynamic::P>; // R is added separately

        /*
         * Tracks + Covariance and Measurements -> Association
         */
        using AssociationFunc =
                std::function<auto(const std::vector<std::tuple<typename ObjectDynamic::Z, typename ObjectDynamic::R,
                                                                AdditionalData>> &,
                                   const std::vector<std::pair<typename ObjectDynamic::Z, AdditionalData>> &)
                                      ->AssociationResult>;

        // Functions to be used by user
        EKFSlam(VehicleDynamic vehicleDynamic, ObjectDynamic objectDynamic, InitialEstFunc initialEstFunc,
                InitialCovFunc initialCovFunc, typename VehicleDynamic::X initialEst = VehicleDynamic::X::Zero(),
                typename VehicleDynamic::P initialCov = VehicleDynamic::P::Identity());

        void update(typename VehicleDynamic::Z vehicleMeasurement, ObjectMeasurements objectMeasurements,
                    AssociationFunc associationFunc);

        auto getVehicle() const -> typename VehicleDynamic::X;

        auto getObject(std::size_t id) const -> std::pair<typename ObjectDynamic::X, AdditionalData>;

        [[nodiscard]] auto getNumberOfObjects() const -> std::size_t;

      private:
        [[nodiscard]] auto predict(X x, P p) const -> std::pair<X, P>;

        [[nodiscard]] auto measure(X x, P p) const -> std::pair<Z, S>;

        auto slam(X x, P p, std::vector<AdditionalData> oldAdditionalData, typename VehicleDynamic::Z vehicleMeas,
                  ObjectMeasurements measurements, AssociationFunc associationFunc) const
                -> std::tuple<X, P, std::vector<AdditionalData>>;

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

        // Actual prediction
        X lastX;
        P lastP;
        std::vector<AdditionalData> additionalData;
    };

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T, typename AdditionalData>
    EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T, AdditionalData>::EKFSlam(
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
             std::size_t OBJECT_MEAS_DIM, typename T, typename AdditionalData>
    void EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T, AdditionalData>::update(
            typename VehicleDynamic::Z vehicleMeasurement, ObjectMeasurements objectMeasurements,
            AssociationFunc associationFunc) {
        // Predict all objects, independent of their visibility, this leads to an increase in covariance over time
        std::tie(lastX, lastP) = predict(lastX, lastP);

        // The innovation step is combined with the track management
        std::tie(lastX, lastP, additionalData) =
                slam(lastX, lastP, additionalData, vehicleMeasurement, objectMeasurements, associationFunc);

        // Just some plausibility checks
        ASSERT_COV(lastP);
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T, typename AdditionalData>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T, AdditionalData>::predict(
            X x, P p) const -> std::pair<X, P> {
        // x_hat = f(x)
        x_v(x) = vehicleDynamic.f(x_v(x));
        for (auto c = 0U; c < numObjects(x); ++c) {
            x_o(x, c) = objectDynamic.f(x_o(x, c));
        }

        // P_hat = dF * P * dF^T + q
        auto df = getdf(x);
        auto q = getQ(x);
        ASSERT_COV(q);
        ASSERT_COV(p);
        p = df * p * df.transpose() + q;
        ASSERT_COV(p);

        return std::make_pair(x, p);
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T, typename AdditionalData>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T, AdditionalData>::measure(
            X x, P p) const -> std::pair<Z, S> {
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
             std::size_t OBJECT_MEAS_DIM, typename T, typename AdditionalData>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T, AdditionalData>::slam(
            X x, P p, std::vector<AdditionalData> oldAdditionalData, typename VehicleDynamic::Z vehicleMeas,
            ObjectMeasurements measurements, AssociationFunc associationFunc) const
            -> std::tuple<X, P, std::vector<AdditionalData>> {

        auto [z_hat, s] = measure(x, p);

        std::vector<std::tuple<typename ObjectDynamic::Z, typename ObjectDynamic::R, AdditionalData>> trackMeasurements;
        for (auto c = 0U; c < numObjects(x); ++c) {
            auto offset = VEHICLE_MEAS_DIM + c * OBJECT_MEAS_DIM;
            auto meas = z_hat.template block<OBJECT_MEAS_DIM, 1>(offset, 0);
            auto cov = s.template block<OBJECT_MEAS_DIM, OBJECT_MEAS_DIM>(offset, offset);
            trackMeasurements.emplace_back(meas, cov, oldAdditionalData[c]);
        }

        auto associationResult = associationFunc(trackMeasurements, measurements);

        // Find tracks not associated
        std::set<std::size_t> associatedTracks;
        for (const auto &[track, _] : associationResult.track2Measure) {
            associatedTracks.emplace(track);
        }


        // Split objects into associated objects and objects that are not associated or should be deleted

        // Warning: Only stored the covariance of the invisible object not the relation to other objects and especially
        // the vehicle. Not optimal, but probably works
        std::vector<std::tuple<typename ObjectDynamic::X, typename ObjectDynamic::P, AdditionalData>> invisibleObjects;

        auto reducedSize = VEHICLE_STATE_DIM + associatedTracks.size() * OBJECT_STATE_DIM;
        Eigen::Matrix<T, Eigen::Dynamic, 1> reducedX = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(reducedSize);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> reducedP =
                Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(reducedSize, reducedSize);
        std::vector<AdditionalData> reducedAdditionalData;
        std::map<std::size_t, std::size_t> oldId2TrackId;
        auto reducedIndex = 0;

        reducedX.template block<VEHICLE_STATE_DIM, 1>(0, 0) = x_v(x);
        reducedP.template block<VEHICLE_STATE_DIM, VEHICLE_STATE_DIM>(0, 0) =
                p.template block<VEHICLE_STATE_DIM, VEHICLE_STATE_DIM>(0, 0);

        for (auto i = 0U; i < numObjects(x); ++i) {
            bool trackToDelete = false;
            for (auto toDelete : associationResult.tracksToDelete) {
                if (toDelete == i) {
                    trackToDelete = true;
                    break;
                }
            }

            auto offset = VEHICLE_STATE_DIM + i * OBJECT_STATE_DIM;
            auto state = x_o(x, i);
            auto cov = p.template block<OBJECT_STATE_DIM, OBJECT_STATE_DIM>(offset, offset);

            if (not associatedTracks.contains(i) and not trackToDelete) {
                invisibleObjects.emplace_back(state, cov, oldAdditionalData[i]);
            } else if (associatedTracks.contains(i)) {
                auto reducedOffset = VEHICLE_STATE_DIM + reducedIndex * OBJECT_STATE_DIM;
                reducedX.template block<OBJECT_STATE_DIM, 1>(reducedOffset, 0) = state;
                reducedP.template block<OBJECT_STATE_DIM, OBJECT_STATE_DIM>(reducedOffset, reducedOffset) = cov;
                reducedAdditionalData.emplace_back(additionalData[i]);
                oldId2TrackId[i] = reducedIndex;
                reducedIndex += 1;
            }
        }

        // Update track-measurement vector based on reduced state vector
        std::tie(z_hat, s) = measure(reducedX, reducedP);

        // Reorder measurements to match tracks and only include associated measurements
        ObjectMeasurements reorderedMeasurements(associatedTracks.size());
        for (auto oldTrackId : associatedTracks) {
            auto newTrackId = oldId2TrackId.at(oldTrackId);
            auto meas = associationResult.track2Measure[oldTrackId];
            reorderedMeasurements.at(newTrackId) = measurements.at(meas);
        }

        // Build Z Vector
        Z z(VEHICLE_MEAS_DIM + reorderedMeasurements.size() * OBJECT_MEAS_DIM);
        z.block(0, 0, VEHICLE_MEAS_DIM, 1) = vehicleMeas;
        for (auto c = 0U; c < reorderedMeasurements.size(); ++c) {
            z.template block<OBJECT_MEAS_DIM, 1>(VEHICLE_MEAS_DIM + c * OBJECT_MEAS_DIM, 0) =
                    reorderedMeasurements[c].first;
        }

        // Calculate Kalman Gain
        Mat K = reducedP * getdh(reducedX).transpose() * s.inverse();

        // Innovation
        Vec tildeZ = z - z_hat;
        reducedX = reducedX + K * tildeZ;
        reducedP = reducedP - K * s * K.transpose();
        ASSERT_COV(reducedP);

        // Readd tracks and add new track
        auto newSize =
                reducedX.size() + (invisibleObjects.size() + associationResult.newTracks.size()) * OBJECT_STATE_DIM;
        X completeX = X::Zero(newSize);
        P completeP = P::Zero(newSize, newSize);
        completeX.block(0, 0, reducedX.size(), 1) = reducedX;
        completeP.block(0, 0, reducedP.rows(), reducedP.cols()) = reducedP;
        // Readd tracks
        for (auto c = 0U; c < invisibleObjects.size(); ++c) {
            auto offset = reducedX.size() + c * OBJECT_STATE_DIM;
            completeX.block(offset, 0, OBJECT_STATE_DIM, 1) = std::get<0>(invisibleObjects[c]);
            completeP.block(offset, offset, OBJECT_STATE_DIM, OBJECT_STATE_DIM) = std::get<1>(invisibleObjects[c]);
            oldAdditionalData.emplace_back(std::get<2>(invisibleObjects[c]));
        }

        // New tracks
        for (auto c = 0U; c < associationResult.newTracks.size(); ++c) {
            auto offset = reducedX.size() + invisibleObjects.size() * OBJECT_STATE_DIM + c * OBJECT_STATE_DIM;

            auto measId = associationResult.newTracks[c];
            auto initialEstimate = initialEstFunc(measurements[measId].first, x_v(x));
            Mat initialCov = initialCovFunc(measurements[measId].first, x_v(x),
                                            p.block(0, 0, VEHICLE_STATE_DIM, VEHICLE_STATE_DIM)) +
                             vehicleDynamic.r_func();


            completeX.block(offset, 0, OBJECT_STATE_DIM, 1) = initialEstimate;
            completeP.block(offset, offset, OBJECT_STATE_DIM, OBJECT_STATE_DIM) = initialCov;
            oldAdditionalData.emplace_back(measurements[measId].second);
        }

        ASSERT_COV(completeP);

        return std::make_tuple(completeX, completeP, oldAdditionalData);
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T, typename AdditionalData>
    auto
    EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T, AdditionalData>::getVehicle()
            const -> typename VehicleDynamic::X {
        return x_v(lastX);
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T, typename AdditionalData>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T, AdditionalData>::getObject(
            std::size_t id) const -> std::pair<typename ObjectDynamic::X, AdditionalData> {
        return std::pair{x_o(lastX, id), additionalData[id]};
    }

    /*
     * Helper functions
     */
    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T, typename AdditionalData>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T, AdditionalData>::getdf(
            const X &x) const {
        Mat J_F = Mat::Zero(x.size(), x.size());

        J_F.block(0, 0, VEHICLE_STATE_DIM, VEHICLE_STATE_DIM) = vehicleDynamic.j_f(x_v(x));

        for (auto c = 0U; c < numObjects(x); ++c) {
            auto offset = VEHICLE_STATE_DIM + c * OBJECT_STATE_DIM;
            J_F.block(offset, offset, OBJECT_STATE_DIM, OBJECT_STATE_DIM) = objectDynamic.j_f(x_o(x, c));
        }

        return J_F;
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T, typename AdditionalData>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T, AdditionalData>::getQ(
            const X &x) const {
        Mat Q = Mat::Zero(x.size(), x.size());

        Q.template block<VEHICLE_STATE_DIM, VEHICLE_STATE_DIM>(0, 0) = vehicleDynamic.q_func(x_v(x));

        for (auto c = 0U; c < numObjects(x); ++c) {
            auto offset = VEHICLE_STATE_DIM + c * OBJECT_STATE_DIM;
            Q.template block<OBJECT_STATE_DIM, OBJECT_STATE_DIM>(offset, offset) = objectDynamic.q_func(x_o(x, c));
        }

        return Q;
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T, typename AdditionalData>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T, AdditionalData>::getdh(
            const X &x) const {
        Mat J_H = Mat::Zero(VEHICLE_MEAS_DIM + numObjects(x) * OBJECT_MEAS_DIM, x.size());
        J_H.template block<VEHICLE_MEAS_DIM, VEHICLE_STATE_DIM>(0, 0) = vehicleDynamic.j_h(x_v(x));

        for (auto c = 0U; c < numObjects(x); ++c) {
            J_H.template block<OBJECT_MEAS_DIM, OBJECT_STATE_DIM>(VEHICLE_MEAS_DIM + c * OBJECT_MEAS_DIM,
                                                                  VEHICLE_STATE_DIM + c * OBJECT_STATE_DIM) =
                    objectDynamic.j_h_object(x_o(x, c), x_v(x));
            J_H.template block<OBJECT_MEAS_DIM, VEHICLE_STATE_DIM>(VEHICLE_MEAS_DIM + c * OBJECT_MEAS_DIM, 0) =
                    objectDynamic.j_h_vehicle(x_o(x, c), x_v(x));
        }

        return J_H;
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T, typename AdditionalData>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T, AdditionalData>::getR(
            const X &x) const {
        Mat R = Mat::Zero(VEHICLE_MEAS_DIM + numObjects(x) * OBJECT_MEAS_DIM,
                          VEHICLE_MEAS_DIM + numObjects(x) * OBJECT_MEAS_DIM);
        R.template block<VEHICLE_MEAS_DIM, VEHICLE_MEAS_DIM>(0, 0) = vehicleDynamic.r_func();

        for (auto c = 0U; c < numObjects(x); ++c) {
            R.template block<OBJECT_MEAS_DIM, OBJECT_MEAS_DIM>(VEHICLE_MEAS_DIM + c * OBJECT_MEAS_DIM,
                                                               VEHICLE_MEAS_DIM + c * OBJECT_MEAS_DIM) =
                    objectDynamic.r_func();
        }

        ASSERT_COV(R);

        return R;
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T, typename AdditionalData>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T, AdditionalData>::x_v(X &x) {
        return x.template block<VEHICLE_STATE_DIM, 1>(0, 0);
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T, typename AdditionalData>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T, AdditionalData>::x_v(
            const X &x) {
        return x.template block<VEHICLE_STATE_DIM, 1>(0, 0);
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T, typename AdditionalData>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T, AdditionalData>::numObjects(
            const X &x) {
        assert((x.size() - VEHICLE_STATE_DIM) % OBJECT_STATE_DIM == 0);
        return (x.size() - VEHICLE_STATE_DIM) / OBJECT_STATE_DIM;
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T, typename AdditionalData>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T, AdditionalData>::x_o(
            X &x, std::size_t i) {
        assert(i < numObjects(x));
        return x.template block<OBJECT_STATE_DIM, 1>(VEHICLE_STATE_DIM + i * OBJECT_STATE_DIM, 0);
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T, typename AdditionalData>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T, AdditionalData>::x_o(
            const X &x, std::size_t i) {
        assert(i < numObjects(x));
        return x.template block<OBJECT_STATE_DIM, 1>(VEHICLE_STATE_DIM + i * OBJECT_STATE_DIM, 0);
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T, typename AdditionalData>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T,
                 AdditionalData>::getNumberOfObjects() const -> std::size_t {
        return numObjects(lastX);
    }
} // namespace ekf_slam

#endif // EKFSLAM_EKFSLAM_HPP
