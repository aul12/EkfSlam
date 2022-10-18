/**
 * @headerfile Util.hpp
 * @author paul
 * @date 22.03.21
 * Description here TODO
 */
#ifndef EKFSLAM_UTIL_HPP
#define EKFSLAM_UTIL_HPP

#include <Eigen/Eigen>
#include <iostream>

namespace ekf_slam::util {
    auto isCovarianceMatrix(const Eigen::MatrixXd &mat) -> bool;

    void printMatImpl(const char *file, int line, const char *name, const Eigen::MatrixXd &mat);
} // namespace ekf_slam::util

#define ASSERT_COV(x)                                                                                                  \
    /*if (not ekf_slam::util::isCovarianceMatrix(x)) {                                                                   \
        std::cout << __FILE__ << ":" << __LINE__ << ":\n\t" << #x << " is not a covariance Matrix:\n"                  \
                  << x << "\n"                                                                                         \
                  << "with det(" << #x << ")=" << x.determinant() << std::endl;                                        \
        std::abort();                                                                                                  \
    }*/


#define PRINT_MAT(x) ekf_slam::util::printMatImpl(__FILE__, __LINE__, #x, x);

#endif // EKFSLAM_UTIL_HPP
