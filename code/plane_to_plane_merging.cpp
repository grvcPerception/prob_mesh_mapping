#include <iostream>
#include <random>

#include "math.h"

int main(int argc, char** argv) {
  srand(static_cast<unsigned>(time(0)));

  // Generate two random planes by:
  // 1. Starting from a plane that intersects the origin and is parallel to the z-axis
  // 2. Adding a random perturbation both in the orientation and the distance to the origin
  const Eigen::Vector3d z_unit = (Eigen::Vector3d() << 0.0, 0.0, 1.0).finished();
  const Eigen::Vector3d perturbation1 =
      Eigen::Vector3d::Random().cwiseProduct(Eigen::Vector3d(0.1, 0.1, 3));
  const Eigen::Vector4d plane1_mean =
      s2_BP(Eigen::Vector4d(0.0, 0.0, 1.0, 0.0), perturbation1);
  const Eigen::Vector3d plane1_normal = plane1_mean.head(3);
  const Eigen::Vector3d perturbation2 =
      Eigen::Vector3d::Random().cwiseProduct(Eigen::Vector3d(0.1, 0.1, 3));
  const Eigen::Vector4d plane2_mean =
      s2_BP(Eigen::Vector4d(0.0, 0.0, 1.0, 0.0), perturbation2);
  const Eigen::Vector3d plane2_normal = plane2_mean.head(3);

  // Generate two random covariances for the two planes
  const Eigen::Matrix<double, 3, 3> plane1_cov =
      generateCovarianceMatrix<double, 3>() * 0.1;
  const Eigen::Matrix<double, 3, 3> plane2_cov =
      generateCovarianceMatrix<double, 3>() * 0.1;

  // Merge the two planes into one
  const auto [estimated_plane, estimated_cov] =
      aggregateProbabilisticPlanes(plane1_mean, plane2_mean, plane1_cov, plane2_cov);

  std::cout << "Estimated plane: " << estimated_plane << std::endl;
  std::cout << "Estimated covariance: " << estimated_cov << std::endl;
}