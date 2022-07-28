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
#include "models/ConstantPositionModel.hpp"
#include "models/SingleTrackModel.hpp"

namespace ekf_slam {
    template<typename T_ = double, typename Vehicle_ = models::single_track<T_>,
             typename Object_ = models::constant_position<T_>>
    class Manager {
      public:
        using T = T_;
        using Vehicle = Vehicle_;
        using Object = Object_;

        using EKF = EKFSlam<Vehicle::State::DIM, Vehicle::Meas::DIM, Object::State::DIM, Object::Meas::DIM, T>;

        Manager(typename Vehicle::Params vehicleParams, typename Object::Params objectParams) :
            dt{NAN},
            ekf{Vehicle::make(dt, vehicleParams.sigmaA2, vehicleParams.sigmaDDPsi2, vehicleParams.sigmaV2,
                              vehicleParams.sigmaDPsi2),
                Object::make(objectParams.sigmaPos2, objectParams.sigmaMeas), Object::getInitialPosition,
                Object::getInitialCovariance} {
        }

        auto update(typename Vehicle::Meas vehicleMeas, const std::vector<typename Object::Meas> &objectMeasurements,
                    double dt,
                    typename EKF::AssociationFunc associationFunc =
                            association::nearest_neighbor_association<Object::Meas::DIM, T>)
                -> std::pair<typename Vehicle::State, std::vector<typename Object::State>> {
            this->dt = dt;
            auto z_vehicle = vehicleMeas.getVec();
            std::vector<typename Object::Meas::Vec> z_object;
            std::transform(objectMeasurements.cbegin(), objectMeasurements.cend(), std::back_inserter(z_object),
                           [](const typename Object::Meas &meas) { return meas.getVec(); });
            ekf.update(z_vehicle, z_object, associationFunc);

            typename Vehicle::State vehicleState{ekf.getVehicle()};
            std::vector<typename Object::State> objects;
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
