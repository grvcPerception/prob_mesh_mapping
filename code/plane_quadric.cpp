#include <iostream>
#include <random>

#include "math.h"

int main(int argc, char** argv) {
  srand(static_cast<unsigned>(time(0)));

  // Generate a random plane by:
  // 1. Starting from a plane that intersects the origin and is parallel to the z-axis
  // 2. Adding a random perturbation both in the orientation and the distance to the origin
  const Eigen::Vector3d z_unit = (Eigen::Vector3d() << 0.0, 0.0, 1.0).finished();
  const Eigen::Vector3d perturbation1 =
      Eigen::Vector3d::Random().cwiseProduct(Eigen::Vector3d(0.1, 0.1, 3));
  const Eigen::Vector4d plane1_mean =
      s2_BP(Eigen::Vector4d(0.0, 0.0, 1.0, 0.0), perturbation1);

  // Generate a random covariance for the plane
  const Eigen::Matrix<double, 3, 3> plane1_cov =
      generateCovarianceMatrix<double, 3>() * 0.1;

  // Estimate the quadric of the plane
  const auto estimated_quadric = esti_quadric(plane1_mean, plane1_cov);

  std::cout << "Estimated quadric: " << estimated_quadric << std::endl;
}