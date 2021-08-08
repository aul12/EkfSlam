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
            constant_position_model::get_initial_position<T>, constant_position_model::get_initial_covariance<T>} {
    }

    auto Manager::update(VehicleMeas vehicleMeas, const std::vector<ObjectMeas> &objectMeasurements, double delta_time)
            -> std::pair<VehicleState, std::vector<ObjectState>> {
        this->dt = delta_time;
        auto z_vehicle = vehicleMeas.get_vec();
        std::vector<ObjectMeas::Vec> z_object;
        std::transform(objectMeasurements.cbegin(), objectMeasurements.cend(), std::back_inserter(z_object),
                       [](const ObjectMeas &meas) { return meas.get_vec(); });
        ekf.update(z_vehicle, z_object);

        VehicleState vehicle_state{ekf.get_vehicle()};
        std::vector<ObjectState> objects;
        for (auto c = 0U; c < ekf.get_number_of_objects(); ++c) {
            objects.emplace_back(ekf.get_object(c));
        }

        return std::make_pair(vehicle_state, objects);
    }
} // namespace ekf_slam
