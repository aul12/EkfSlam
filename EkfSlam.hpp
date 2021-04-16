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

        auto get_vehicle() const -> typename VehicleDynamic::X;

        auto get_object(std::size_t id) const -> typename ObjectDynamic::X;

        [[nodiscard]] auto get_number_of_objects() const -> std::size_t;

      private:
        [[nodiscard]] auto predict(X x, P p) const -> std::pair<X, P>;

        [[nodiscard]] auto measure(X x, P p) const -> std::pair<Z, S>;

        auto slam(X x, P p, typename VehicleDynamic::Z vehicleMeas, ObjectMeasurements measurements) const
                -> std::pair<X, P>;

        auto add_tracks(X x, P p, typename VehicleDynamic::Z vehicleMeas, ObjectMeasurements measurements) const
                -> std::pair<X, P>;

        auto separate_tracks_not_measured(X x, P p, ObjectMeasurements objectMeasurements) const
                -> std::tuple<X, P, TrackList>;

        auto data_association(Z z, S s, ObjectMeasurements measurements) const -> std::map<std::size_t, std::size_t>;

        // Helper functions single objects <-> vector conversion
        auto getdf(const X &x) const -> Mat;
        auto get_q(const X &x) const -> Mat;
        auto getdh(const X &x) const -> Mat;
        auto get_r(const X &x) const -> Mat;
        auto x_v(X x) const -> X;
        auto num_objects(const X &x) const -> std::size_t;
        auto x_o(X x, std::size_t i) const -> X;

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
        for (auto c = 0U; c < num_objects(x); ++c) {
            x_o(x, c) = objectDynamic.f(x_o(x, c));
        }

        // P_hat = dF * P * dF^T + q
        auto df = getdf(x);
        auto q = get_q(x);
        p = df * p * df.transpose() + q;
        ASSERT_COV(p);

        return std::make_pair(x, p);
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::measure(X x, P p) const
            -> std::pair<Z, S> {
        // z = h(x)
        Z z = Z::Zero(VEHICLE_MEAS_DIM + num_objects(x) * OBJECT_MEAS_DIM);
        z.block(0, 0, VEHICLE_MEAS_DIM, 1) = vehicleDynamic.h(x_v(x));
        for (auto c = 0U; c < num_objects(x); ++c) {
            z.block(VEHICLE_MEAS_DIM + c * OBJECT_MEAS_DIM, 0, OBJECT_MEAS_DIM, 1) = objectDynamic.h(x_o(x, c), x_v(x));
        }

        // s = dh * P * dh^T + r
        auto dh = getdh(x);
        auto r = get_r(x);
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
        std::tie(x, p) = add_tracks(x, p, vehicleMeas, measurements);

        // Temporary remove tracks not visible
        TrackList invisible_objects;
        std::tie(x, p, invisible_objects) = separate_tracks_not_measured(x, p, measurements);

        assert(num_objects(x) == measurements.size());

        auto result = measure(x, p);
        Z z_hat = result.first;
        S s = result.second;

        // Data association
        auto association_map = data_association(z_hat, s, measurements);

        // Reorder z to match x
        ObjectMeasurements reordered_measurements(measurements.size());
        for (auto c = 0U; c < measurements.size(); ++c) {
            reordered_measurements[c] = measurements[association_map[c]];
        }

        // Build Z Vector
        Z z(VEHICLE_MEAS_DIM + reordered_measurements.size() * OBJECT_MEAS_DIM);
        z.block(0, 0, VEHICLE_MEAS_DIM, 1) = vehicleMeas;
        for (auto c = 0U; c < reordered_measurements.size(); ++c) {
            z.block(VEHICLE_MEAS_DIM + c * OBJECT_MEAS_DIM, 0, OBJECT_MEAS_DIM, 1) = reordered_measurements[c];
        }

        // Calculate Kalman Gain
        Mat k = p * getdh(x).transpose() * s.inverse();

        // Innovation
        Vec tilde_z = z - z_hat;
        x = x + k * tilde_z;
        p = p - k * s * k.transpose();
        ASSERT_COV(p);

        // Readd tracks
        X complete_x = X::Zero(x.size() + invisible_objects.size() * OBJECT_STATE_DIM);
        P complete_p = P::Zero(p.rows() + invisible_objects.size() * OBJECT_STATE_DIM,
                              p.cols() + invisible_objects.size() * OBJECT_STATE_DIM);
        complete_x.block(0, 0, x.size(), 1) = x;
        complete_p.block(0, 0, p.rows(), p.cols()) = p;
        for (auto c = 0U; c < invisible_objects.size(); ++c) {
            auto offset = x.size() + c * OBJECT_STATE_DIM;
            complete_x.block(offset, 0, OBJECT_STATE_DIM, 1) = invisible_objects[c].first;
            complete_p.block(offset, offset, OBJECT_STATE_DIM, OBJECT_STATE_DIM) = invisible_objects[c].second;
        }

        ASSERT_COV(complete_p);

        return std::make_pair(complete_x, complete_p);
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::add_tracks(
            X x, P p, typename VehicleDynamic::Z vehicleMeas, ObjectMeasurements measurements) const
            -> std::pair<X, P> {
        auto result = measure(x, p);
        Z z_hat = result.first;
        S s = result.second;

        // We do not update in place as this can result in missing cones if a new cone can be matched to two
        // measurements
        X updated_x = x;
        P updated_p = p;

        for (const auto &z : measurements) {
            auto min_mhd = std::numeric_limits<T>::max();
            for (auto c = 0U; c < num_objects(x); ++c) {
                auto offset = VEHICLE_MEAS_DIM + c * OBJECT_MEAS_DIM;
                auto z_track = z_hat.block(offset, 0, OBJECT_MEAS_DIM, 1);
                S cov = s.block(offset, offset, OBJECT_MEAS_DIM, OBJECT_MEAS_DIM);
                auto z_tilde = z_track - z;

                ASSERT_COV(cov);

                auto mhd2 = z_tilde.transpose() * cov.inverse() * z_tilde;
                assert(mhd2.rows() == 1 and mhd2.cols() == 1);
                auto mhd = std::sqrt(mhd2(0, 0));

                if (mhd < min_mhd) {
                    min_mhd = mhd;
                }
            }

            if (min_mhd > gate) {
                auto initial_estimate = initialEstFunc(z, x_v(x));
                Mat initial_cov = initialCovFunc(z, x_v(x), p.block(0, 0, VEHICLE_STATE_DIM, VEHICLE_STATE_DIM)) +
                                 vehicleDynamic.r_func();

                ASSERT_COV(initial_cov);

                X extended_x = X::Zero(updated_x.size() + OBJECT_STATE_DIM);
                P extended_p = P::Zero(updated_p.rows() + OBJECT_STATE_DIM, updated_p.cols() + OBJECT_STATE_DIM);
                extended_x.block(0, 0, updated_x.size(), 1) = updated_x;
                extended_x.block(updated_x.size(), 0, OBJECT_STATE_DIM, 1) = initial_estimate;
                extended_p.block(0, 0, updated_x.size(), updated_x.size()) = updated_p;
                extended_p.block(updated_x.size(), updated_x.size(), OBJECT_STATE_DIM, OBJECT_STATE_DIM) = initial_cov;

                updated_x = extended_x;
                updated_p = extended_p;
                ASSERT_COV(updated_p);
                // std::tie(z_hat, s) = measure(x, p);
            }
        }

        return std::make_pair(updated_x, updated_p);
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::separate_tracks_not_measured(
            EKFSlam::X x, EKFSlam::P p, ObjectMeasurements objectMeasurements) const -> std::tuple<X, P, TrackList> {

        auto result = measure(x, p);
        Z z_hat = result.first;
        S s = result.second;

        std::vector<std::pair<typename ObjectDynamic::X, typename ObjectDynamic::P>> invisible_objects;
        for (auto c = 0U; c < num_objects(x); ++c) {
            auto min_mhd = std::numeric_limits<T>::max();

            for (const auto &z_o : objectMeasurements) {
                auto offset = VEHICLE_MEAS_DIM + c * OBJECT_MEAS_DIM;
                auto z_track = z_hat.block(offset, 0, OBJECT_MEAS_DIM, 1);
                auto cov = s.block(offset, offset, OBJECT_MEAS_DIM, OBJECT_MEAS_DIM);
                auto z_tilde = z_track - z_o;

                auto mhd2 = z_tilde.transpose() * cov.inverse() * z_tilde;
                assert(mhd2.rows() == 1 and mhd2.cols() == 1);
                auto mhd = std::sqrt(mhd2(0, 0));

                if (mhd < min_mhd) {
                    min_mhd = mhd;
                }
            }

            if (min_mhd > gate) {
                auto offset = VEHICLE_STATE_DIM + c * OBJECT_STATE_DIM;
                auto x_track = x.block(offset, 0, OBJECT_STATE_DIM, 1);
                auto p_track = p.block(offset, offset, OBJECT_STATE_DIM, OBJECT_STATE_DIM);

                ASSERT_COV(p_track);

                invisible_objects.emplace_back(x_track, p_track);

                // Swap with last track
                auto last_index = num_objects(x) - 1;
                if (c != last_index) {
                    auto lastOffset = VEHICLE_STATE_DIM + last_index * OBJECT_STATE_DIM;
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

        return std::make_tuple(x, p, invisible_objects);
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::data_association(
            Z z_hat, S s, ObjectMeasurements measurements) const -> std::map<std::size_t, std::size_t> {
        // i is track, j is measurement
        Mat association_matrix{measurements.size(), measurements.size()};

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

                association_matrix(i, j) = gate - mhd2(0, 0);
            }
        }

        // Auction algorithm
        std::vector<T> track_prices(measurements.size());
        std::vector<bool> observation_mapped(measurements.size());

        // track index -> measurement index
        std::map<std::size_t, std::size_t> map;

        auto count = 0U;
        while (true) {
            ++count;
            bool all_mapped = true;
            for(auto const &observation : observation_mapped) {
                if(!observation) {
                    all_mapped = false;
                    break;
                }
            }
            if (all_mapped) {
                break;
            }

            std::size_t j = 0;
            for (; j < measurements.size(); ++j) {
                if (not observation_mapped[j]) {
                    break;
                }
            }

            auto i_max = 0;
            auto max_val = std::numeric_limits<T>::lowest();

            for (auto i = 0U; i < measurements.size(); ++i) {
                auto val = association_matrix(i, j) - track_prices[i];
                if (val > max_val) {
                    i_max = i;
                    max_val = val;
                }
            }

            if (map.find(i_max) != map.cend()) {
                observation_mapped[map[i_max]] = false;
            }
            map[i_max] = j;

            auto second_i_max = 0;
            auto second_val = std::numeric_limits<T>::lowest();

            for (auto i = 0U; i < measurements.size(); ++i) {
                auto val = association_matrix(i, j) - track_prices[i];
                if (val > second_val and second_i_max != i_max) {
                    second_i_max = i;
                    second_val = val;
                }
            }
            auto y_i = max_val - second_val;
            track_prices[i_max] += y_i + epsilon;
            observation_mapped[j] = true;
        }

        return map;
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::get_vehicle() const ->
            typename VehicleDynamic::X {
        return x_v(lastX);
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto
    EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::get_object(std::size_t id) const
            -> typename ObjectDynamic::X {
        return x_o(lastX, id);
    }

    /*
     * Helper functions
     */
    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::getdf(const X &x) const -> Mat {
        Mat j_f = Mat::Zero(x.size(), x.size());

        j_f.block(0, 0, VEHICLE_STATE_DIM, VEHICLE_STATE_DIM) = vehicleDynamic.j_f(x_v(x));

        for (auto c = 0U; c < num_objects(x); ++c) {
            auto offset = VEHICLE_STATE_DIM + c * OBJECT_STATE_DIM;
            j_f.block(offset, offset, OBJECT_STATE_DIM, OBJECT_STATE_DIM) = objectDynamic.j_f(x_o(x, c));
        }

        return j_f;
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::get_q(const X &x) const -> Mat {
        Mat q = Mat::Zero(x.size(), x.size());

        q.block(0, 0, VEHICLE_STATE_DIM, VEHICLE_STATE_DIM) = vehicleDynamic.q_func(x_v(x));
        ASSERT_COV(q);

        for (auto c = 0U; c < num_objects(x); ++c) {
            auto offset = VEHICLE_STATE_DIM + c * OBJECT_STATE_DIM;
            q.block(offset, offset, OBJECT_STATE_DIM, OBJECT_STATE_DIM) = objectDynamic.q_func(x_o(x, c));
            ASSERT_COV(q);
        }

        return q;
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::getdh(const X &x) const -> Mat {
        Mat j_h = Mat::Zero(VEHICLE_MEAS_DIM + num_objects(x) * OBJECT_MEAS_DIM, x.size());
        j_h.block(0, 0, VEHICLE_MEAS_DIM, VEHICLE_STATE_DIM) = vehicleDynamic.j_h(x_v(x));

        for (auto c = 0U; c < num_objects(x); ++c) {
            j_h.block(VEHICLE_MEAS_DIM + c * OBJECT_MEAS_DIM, VEHICLE_STATE_DIM + c * OBJECT_STATE_DIM, OBJECT_MEAS_DIM,
                      OBJECT_STATE_DIM) = objectDynamic.j_h_object(x_o(x, c), x_v(x));
            j_h.block(VEHICLE_MEAS_DIM + c * OBJECT_MEAS_DIM, 0, OBJECT_MEAS_DIM, VEHICLE_STATE_DIM) =
                    objectDynamic.j_h_vehicle(x_o(x, c), x_v(x));
        }

        return j_h;
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::get_r(const X &x) const -> Mat {
        Mat r = Mat::Zero(VEHICLE_MEAS_DIM + num_objects(x) * OBJECT_MEAS_DIM,
                          VEHICLE_MEAS_DIM + num_objects(x) * OBJECT_MEAS_DIM);
        r.block(0, 0, VEHICLE_MEAS_DIM, VEHICLE_MEAS_DIM) = vehicleDynamic.r_func();

        for (auto c = 0U; c < num_objects(x); ++c) {
            r.block(VEHICLE_MEAS_DIM + c * OBJECT_MEAS_DIM, VEHICLE_MEAS_DIM + c * OBJECT_MEAS_DIM, OBJECT_MEAS_DIM,
                    OBJECT_MEAS_DIM) = objectDynamic.r_func();
        }

        ASSERT_COV(r);

        return r;
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::x_v(X x) const -> X {
        return x.block(0, 0, VEHICLE_STATE_DIM, 1);
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto
    EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::num_objects(const X &x) const -> std::size_t {
        assert((x.size() - VEHICLE_STATE_DIM) % OBJECT_STATE_DIM == 0);
        return (x.size() - VEHICLE_STATE_DIM) / OBJECT_STATE_DIM;
    }

    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::x_o(X x, std::size_t i) const -> X{
        assert(i < num_objects(x));
        return x.block(VEHICLE_STATE_DIM + i * OBJECT_STATE_DIM, 0, OBJECT_STATE_DIM, 1);
    }
    template<std::size_t VEHICLE_STATE_DIM, std::size_t VEHICLE_MEAS_DIM, std::size_t OBJECT_STATE_DIM,
             std::size_t OBJECT_MEAS_DIM, typename T>
    auto EKFSlam<VEHICLE_STATE_DIM, VEHICLE_MEAS_DIM, OBJECT_STATE_DIM, OBJECT_MEAS_DIM, T>::get_number_of_objects() const
            -> std::size_t {
        return num_objects(lastX);
    }
} // namespace ekf_slam

#endif // EKFSLAM_EKFSLAM_HPP
