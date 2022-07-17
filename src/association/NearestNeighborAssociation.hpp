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
    template<std::size_t DIM, typename T>
    auto nearest_neighbor_association(const std::vector<std::pair<Eigen::Matrix<T, DIM, 1>, Eigen::Matrix<T, DIM, DIM>>> &tracks,
                           const std::vector<Eigen::Matrix<T, DIM, 1>> &measurements) -> AssociationResult {
        static constexpr double GATE = 0.5;

        // track index -> measurement index
        std::map<std::size_t, std::size_t> track2_meas;
        for (std::size_t i = 0; i < tracks.size(); ++i) {
            double lowest = std::numeric_limits<double>::max();
            std::size_t index = std::numeric_limits<std::size_t>::max();
            for (std::size_t j = 0; j < measurements.size(); ++j) {
                double distance = std::hypot(tracks[i].first.x() - measurements[j].x(),
                                             tracks[i].first.y() - measurements[j].y());
                if (distance < GATE && lowest > distance) {
                    lowest = distance;
                    index = j;
                }
            }
            if (index < std::numeric_limits<std::size_t>::max()) {
                track2_meas[i] = index;
            }
        }


        // Find measurements without tracks
        std::set<std::size_t> associated_measurements;
        for (const auto &[_, meas] : track2_meas) {
            associated_measurements.emplace(meas);
        }
        std::vector<std::size_t> new_tracks;
        for (auto c = 0U; c < measurements.size(); ++c) {
            if (not associated_measurements.contains(c)) {
                new_tracks.emplace_back(c);
            }
        }

        AssociationResult result{.track2Measure = track2_meas,
                                 .newTracks = new_tracks,
                                 .tracksToDelete = std::vector<std::size_t>{}};

        return result;
    }
} // namespace ekf_slam::association

#endif // EKFSLAM_NEARESTNEIGHBORASSOCIATION_HPP
