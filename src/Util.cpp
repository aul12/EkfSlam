/**
 * @file Util.cpp
 * @author paul
 * @date 22.03.21
 * Description here TODO
 */
#include "Util.hpp"

namespace ekf_slam::util {
    auto isCovarianceMatrix(const Eigen::MatrixXd &mat) -> bool {
        return (mat.rows() == mat.cols()) /*and isPsd(mat)*/ and
               ((mat - mat.transpose()).lpNorm<Eigen::Infinity>() < 0.1);
    }

    void printMatImpl(const char *file, int line, const char *name, const Eigen::MatrixXd &mat) {
        std::cout << file << ":" << line << "\t" << name;
        if (mat.rows() == 1 or mat.cols() == 1) {
            if (mat.cols() == 1) {
                std::cout << "^T=" << mat.transpose() << std::endl;
            } else {
                std::cout << "=" << mat << std::endl;
            }
        } else {
            std::cout << "=\n" << mat << std::endl;
        }
    }
} // namespace ekf_slam::util
