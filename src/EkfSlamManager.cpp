/**
 * @file EkfSlamManager.cpp
 * @author paul
 * @date 17.03.21
 * Description here TODO
 */
#include "EkfSlamManager.hpp"

namespace ekf_slam {
    Manager::Manager(VehicleParams vehicleParams, ObjectParams objectParams) :
        ekf{Vehicle::make(dt, vehicleParams.sigmaA2, vehicleParams.sigmaDDPsi2, vehicleParams.sigmaV2,
                          vehicleParams.sigmaDPsi2),
            Object::make(objectParams.sigmaPos2, objectParams.sigmaMeas), Object::getInitialPosition,
            Object::getInitialCovariance},
        dt{NAN} {
    }

    auto Manager::update(Vehicle::Meas vehicleMeas, const std::vector<Object::Meas> &objectMeasurements, double dt)
            -> std::pair<Vehicle::State, std::vector<Object::State>> {
        this->dt = dt;
        auto z_vehicle = vehicleMeas.getVec();
        std::vector<Object::Meas::Vec> z_object;
        std::transform(objectMeasurements.cbegin(), objectMeasurements.cend(), std::back_inserter(z_object),
                       [](const Object::Meas &meas) { return meas.getVec(); });
        ekf.update(z_vehicle, z_object);

        Vehicle::State vehicleState{ekf.getVehicle()};
        std::vector<Object::State> objects;
        for (auto c = 0U; c < ekf.getNumberOfObjects(); ++c) {
            objects.emplace_back(ekf.getObject(c));
        }

        return std::make_pair(vehicleState, objects);
    }
} // namespace ekf_slam
