/**
 * @file BasicAssociation.hpp
 * @author paul
 * @date 21.06.22
 * Description here TODO
 */
#ifndef EKFSLAM_BASICASSOCIATION_HPP
#define EKFSLAM_BASICASSOCIATION_HPP

#include "../EkfSlam.hpp"

namespace ekf_slam::association {
    template<std::size_t DIM, typename T, typename AdditionalData>
    auto basic_association(
            const std::vector<std::tuple<Eigen::Matrix<T, DIM, 1>, Eigen::Matrix<T, DIM, DIM>, AdditionalData>> &tracks,
            const std::vector<std::pair<Eigen::Matrix<T, DIM, 1>, AdditionalData>> &measurements) -> AssociationResult {
        // i is track, j is measurement
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> associationMatrix{tracks.size(), measurements.size()};

        const auto epsilon = 1 / (tracks.size() + 1.);

        for (auto i = 0U; i < tracks.size(); ++i) {
            for (auto j = 0U; j < measurements.size(); ++j) {
                auto z_track = std::get<0>(tracks[i]);
                auto cov = std::get<1>(tracks[i]);
                auto z_tilde = z_track - measurements[j].first;

                ASSERT_COV(cov);

                auto mhd2 = z_tilde.transpose() * cov.inverse() * z_tilde;

                assert(mhd2.rows() == 1 and mhd2.cols() == 1);

                associationMatrix(i, j) = -mhd2(0, 0);
            }
        }

        // Auction algorithm
        std::vector<T> trackPrices(tracks.size());
        std::vector<bool> observationMapped(measurements.size());

        // track index -> measurement index
        std::map<std::size_t, std::size_t> track2Meas;

        while (track2Meas.size() < std::min(measurements.size(), tracks.size())) {
            std::size_t j = 0;
            for (; j < measurements.size(); ++j) {
                if (not observationMapped[j]) {
                    break;
                }
            }

            auto iMax = 0U;
            auto maxVal = std::numeric_limits<T>::lowest();

            for (auto i = 0U; i < tracks.size(); ++i) {
                auto val = associationMatrix(i, j) - trackPrices[i];
                if (val > maxVal) {
                    iMax = i;
                    maxVal = val;
                }
            }

            if (track2Meas.contains(iMax)) {
                observationMapped[track2Meas[iMax]] = false;
            }
            track2Meas[iMax] = j;

            auto secondIMax = 0;
            auto secondVal = std::numeric_limits<T>::lowest();

            for (auto i = 0U; i < tracks.size(); ++i) {
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

        // Find measurements without tracks
        std::set<std::size_t> associatedMeasurements;
        for (const auto &[_, meas] : track2Meas) {
            associatedMeasurements.emplace(meas);
        }
        std::vector<std::size_t> newTracks;
        for (auto c = 0U; c < measurements.size(); ++c) {
            if (not associatedMeasurements.contains(c)) {
                newTracks.emplace_back(c);
            }
        }

        AssociationResult result{.track2Measure = track2Meas,
                                 .newTracks = newTracks,
                                 .tracksToDelete = std::vector<std::size_t>{}};

        return result;
    }
} // namespace ekf_slam::association

#endif // EKFSLAM_BASICASSOCIATION_HPP
