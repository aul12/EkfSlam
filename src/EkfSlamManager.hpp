/**
 * @headerfile EkfSlamManager.hpp
 * @author paul
 * @date 17.03.21
 * Description here TODO
 */
#ifndef EKFSLAM_EKFSLAMMANAGER_HPP
#define EKFSLAM_EKFSLAMMANAGER_HPP

#include "EkfSlam.hpp"
#include "association/BasicAssociation.hpp"
#include "association/NearestNeighborAssociation.hpp"
#include "models/Color.hpp"
#include "models/ConstantPositionModel.hpp"
#include "models/SingleTrackModel.hpp"

namespace ekf_slam {
    template<typename T_ = double, typename Vehicle_ = models::single_track<T_>,
             typename Object_ = models::constant_position<T_>, typename AdditionalData_ = models::Color>
    class Manager {
      public:
        using T = T_;
        using Vehicle = Vehicle_;
        using Object = Object_;
        using AdditionalData = AdditionalData_;

        using EKF = EKFSlam<Vehicle::State::DIM, Vehicle::Meas::DIM, Object::State::DIM, Object::Meas::DIM, T,
                            AdditionalData>;

        Manager(typename Vehicle::Params vehicleParams, typename Object::Params objectParams) :
            dt{NAN},
            ekf{Vehicle::make(dt, vehicleParams.sigmaA2, vehicleParams.sigmaDDPsi2, vehicleParams.sigmaV2,
                              vehicleParams.sigmaDPsi2),
                Object::make(objectParams.sigmaPos2, objectParams.sigmaMeas), Object::getInitialPosition,
                Object::getInitialCovariance} {
        }

        auto update(typename Vehicle::Meas vehicleMeas,
                    const std::vector<std::pair<typename Object::Meas, AdditionalData>> &objectMeasurements, double dt,
                    typename EKF::AssociationFunc associationFunc =
                            association::nearest_neighbor_association<Object::Meas::DIM, T, AdditionalData>)
                -> std::pair<typename Vehicle::State, std::vector<std::pair<typename Object::State, AdditionalData>>> {
            this->dt = dt;
            auto z_vehicle = vehicleMeas.getVec();
            typename EKF::ObjectMeasurements z_object;
            std::transform(objectMeasurements.cbegin(), objectMeasurements.cend(), std::back_inserter(z_object),
                           [](const std::pair<typename Object::Meas, AdditionalData> &meas) {
                               return typename EKF::ObjectMeasurement{meas.first.getVec(), meas.second};
                           });

            ekf.update(z_vehicle, z_object, associationFunc);

            typename Vehicle::State vehicleState{ekf.getVehicle()};
            std::vector<std::pair<typename Object::State, AdditionalData>> objects;
            for (auto c = 0U; c < ekf.getNumberOfObjects(); ++c) {
                objects.emplace_back(ekf.getObject(c));
            }

            return std::make_pair(vehicleState, objects);
        }

      private:
        double dt;
        EKF ekf;
    };
} // namespace ekf_slam


#endif // EKFSLAM_EKFSLAMMANAGER_HPP
