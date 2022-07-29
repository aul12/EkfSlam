/**
 * @file Types.hpp
 * @author paul
 * @date 29.07.22
 * Description here TODO
 */
#ifndef EKFSLAM_TYPES_HPP
#define EKFSLAM_TYPES_HPP

#include <map>
#include <set>

namespace ekf_slam::types {
    struct AssociationResult {
        using TrackId = std::size_t;
        using MeasId = std::size_t;

        std::map<TrackId, MeasId> track2Measure;
        std::set<MeasId> newTracks;
        std::set<TrackId> tracksToDelete;
    };

    template<std::size_t DIM, typename T, typename AdditionalData>
    struct ObjectMeasurement {
        Eigen::Matrix<T, DIM, 1> meas;
        AdditionalData additionalData;
    };

    template<std::size_t DIM, typename T, typename AdditionalData>
    struct Track {
        Eigen::Matrix<T, DIM, 1> state;
        Eigen::Matrix<T, DIM, DIM> cov;
        AdditionalData additionalData;
    };
} // namespace ekf_slam::types

#endif // EKFSLAM_TYPES_HPP
