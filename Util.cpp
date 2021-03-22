/**
 * @file Util.cpp.c
 * @author paul
 * @date 22.03.21
 * Description here TODO
 */
#include "Util.hpp"

namespace ekf_slam::util {
    auto isCovarianceMatrix(const Eigen::MatrixXd &mat) -> bool {
        return (mat.rows() == mat.cols()) and (mat.determinant() >= 0) and
               ((mat - mat.transpose()).lpNorm<Eigen::Infinity>() < 0.1);
    }
} // namespace ekf_slam::util
