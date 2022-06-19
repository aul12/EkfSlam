/**
 * @headerfile EkfSlamManager.hpp
 * @author paul
 * @date 17.03.21
 * Description here TODO
 */
#ifndef EKFSLAM_EKFSLAMMANAGER_HPP
#define EKFSLAM_EKFSLAMMANAGER_HPP

#include "ConstantPositionModel.hpp"
#include "EkfSlam.hpp"
#include "SingleTrackModel.hpp"

namespace ekf_slam {
    struct VehicleParams {
        double sigmaA2, sigmaDDPsi2, sigmaV2, sigmaDPsi2;
    };

    struct ObjectParams {
        double sigmaPos2, sigmaMeas;
    };

    class Manager {
      public:
        using T = double;
        using VehicleState = single_track_model::State<T>;
        using VehicleMeas = single_track_model::Meas<T>;
        using ObjectState = constant_position_model::State<T>;
        using ObjectMeas = constant_position_model::Meas<T>;

        using EKF = EKFSlam<VehicleState::DIM, VehicleMeas::DIM, ObjectState::DIM, ObjectMeas::DIM, T>;

        Manager(VehicleParams vehicleParams, ObjectParams objectParams);

        auto update(VehicleMeas vehicleMeas, const std::vector<ObjectMeas> &objectMeasurements, double dt) ->
                std::pair<VehicleState, std::vector<ObjectState>>;

      private:
        double dt;
        EKF ekf;
    };
}


#endif // EKFSLAM_EKFSLAMMANAGER_HPP
