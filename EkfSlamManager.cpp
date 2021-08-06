/**
 * @file EkfSlamManager.cpp.cc
 * @author paul
 * @date 17.03.21
 * Description here TODO
 */
#include "EkfSlamManager.hpp"

namespace ekf_slam {
    Manager::Manager(VehicleParams vehicleParams, ObjectParams objectParams) :
        ekf{single_track_model::make(dt, vehicleParams.sigmaA2, vehicleParams.sigmaDDPsi2, vehicleParams.sigmaV2,
                                     vehicleParams.sigmaDPsi2),
            constant_position_model::make(objectParams.sigmaPos2, objectParams.sigmaMeas),
            constant_position_model::getInitialPosition<T>, constant_position_model::getInitialCovariance<T>} {
    }

    auto Manager::update(VehicleMeas vehicleMeas, const std::vector<ObjectMeas> &objectMeasurements, double dt)
            -> std::pair<VehicleState, std::vector<ObjectState>> {
        this->dt = dt;
        auto z_vehicle = vehicleMeas.getVec();
        std::vector<ObjectMeas::Vec> z_object;
        std::transform(objectMeasurements.cbegin(), objectMeasurements.cend(), std::back_inserter(z_object),
                       [](const ObjectMeas &meas) { return meas.getVec(); });
        ekf.update(z_vehicle, z_object);

        VehicleState vehicleState{ekf.getVehicle()};
        std::vector<ObjectState> objects;
        for (auto c = 0U; c < ekf.getNumberOfObjects(); ++c) {
            objects.emplace_back(ekf.getObject(c));
        }

        return std::make_pair(vehicleState, objects);
    }
} // namespace ekf_slam
