/**
 * @headerfile EkfSlamManager.hpp
 * @author paul
 * @date 17.03.21
 * Description here TODO
 */
#ifndef EKFSLAM_EKFSLAMMANAGER_HPP
#define EKFSLAM_EKFSLAMMANAGER_HPP

#include "EkfSlam.hpp"
#include "models/ConstantPositionModel.hpp"
#include "models/SingleTrackModel.hpp"

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
        using Vehicle = models::single_track<T>;
        using Object = models::constant_position<T>;

        using EKF = EKFSlam<Vehicle::State::DIM, Vehicle::Meas::DIM, Object::State::DIM, Object::Meas::DIM, T>;

        Manager(VehicleParams vehicleParams, ObjectParams objectParams);

        auto update(Vehicle::Meas vehicleMeas, const std::vector<Object::Meas> &objectMeasurements, double dt)
                -> std::pair<Vehicle::State, std::vector<Object::State>>;

      private:
        double dt;
        EKF ekf;
    };
} // namespace ekf_slam


#endif // EKFSLAM_EKFSLAMMANAGER_HPP
