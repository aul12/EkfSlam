/**
 * @file BasicAssociation.hpp
 * @author bjoern
 * @date 17.07.22
 * Description here TODO
 */

#ifndef EKFSLAM_NEARESTNEIGHBORASSOCIATION_HPP
#define EKFSLAM_NEARESTNEIGHBORASSOCIATION_HPP

#include "../EkfSlam.hpp"

namespace ekf_slam::association {
    template<std::size_t DIM, typename T, typename AdditionalData>
    auto nearest_neighbor_association(
            const std::vector<std::tuple<Eigen::Matrix<T, DIM, 1>, Eigen::Matrix<T, DIM, DIM>, AdditionalData>> &tracks,
            const std::vector<std::pair<Eigen::Matrix<T, DIM, 1>, AdditionalData>> &measurements) -> AssociationResult {
        static constexpr double GATE = 2;

        // track index -> measurement index
        std::map<std::size_t, std::size_t> track2Meas;
        for (std::size_t i = 0; i < tracks.size(); ++i) {
            double lowest = std::numeric_limits<double>::max();
            std::size_t index = std::numeric_limits<std::size_t>::max();
            for (std::size_t j = 0; j < measurements.size(); ++j) {
                auto zTilde = std::get<0>(tracks[i]) - std::get<0>(measurements[j]);
                auto mhd = zTilde.transpose() * std::get<1>(tracks[i]).inverse() * zTilde;
                if (mhd < GATE && lowest > mhd) {
                    lowest = mhd;
                    index = j;
                }
            }
            if (index < std::numeric_limits<std::size_t>::max()) {
                track2Meas[i] = index;
            }
        }


        // Invert map to find measurements without tracks and measurements associated to multiple tracks
        std::map<std::size_t, std::size_t> meas2Track;
        // Store tracks to erase for track2Meas, they are not erased directly to simplify iteration over map
        std::set<std::size_t> trackIdsToErase;
        for (const auto &[newTrackId, measId] : track2Meas) {
            if (meas2Track.contains(measId)) { // Measurement is already associated to another track
                auto meas = std::get<0>(measurements[measId]);

                auto oldTrackId = meas2Track[measId];
                auto oldTrack = tracks[oldTrackId];
                auto oldMhd = (std::get<0>(oldTrack) - meas).transpose() * std::get<1>(oldTrack).inverse() *
                              (std::get<0>(oldTrack) - meas);

                auto newTrack = tracks[newTrackId];
                auto newMhd = (std::get<0>(newTrack) - meas).transpose() * std::get<1>(newTrack).inverse() *
                              (std::get<0>(newTrack) - meas);

                if (oldMhd < newMhd) {                   // Old association is better
                    trackIdsToErase.emplace(newTrackId); // Erase new association
                } else {
                    trackIdsToErase.emplace(oldTrackId); // Erase old association
                    meas2Track[measId] = newTrackId;     // Update association of measurement to new track
                }
            } else {
                meas2Track.emplace(measId, newTrackId);
            }
        }

        // Actually delete tracks
        for (auto trackId : trackIdsToErase) {
            track2Meas.erase(trackId);
        }


        std::vector<std::size_t> newTracks;
        for (auto c = 0U; c < measurements.size(); ++c) {
            if (not meas2Track.contains(c)) {
                newTracks.emplace_back(c);
            }
        }

        AssociationResult result{.track2Measure = track2Meas,
                                 .newTracks = newTracks,
                                 .tracksToDelete = std::vector<std::size_t>{}};

        return result;
    }
} // namespace ekf_slam::association

#endif // EKFSLAM_NEARESTNEIGHBORASSOCIATION_HPP
