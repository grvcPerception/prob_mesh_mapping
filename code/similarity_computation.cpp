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
      Eigen::Vector3d::Random().cwiseProduct(Eigen::Vector3d(0.01, 0.01, 3));
  const Eigen::Vector4d plane1_mean =
      s2_BP(Eigen::Vector4d(0.0, 0.0, 1.0, 0.0), perturbation1);
  const Eigen::Vector3d perturbation2 =
      Eigen::Vector3d::Random().cwiseProduct(Eigen::Vector3d(0.01, 0.01, 3));
  const Eigen::Vector4d plane2_mean =
      s2_BP(Eigen::Vector4d(0.0, 0.0, 1.0, 0.0), perturbation2);

  // Generate two random covariances for the two planes
  const Eigen::Matrix<double, 3, 3> plane1_cov =
      generateCovarianceMatrix<double, 3>() * 0.01;
  const Eigen::Matrix<double, 3, 3> plane2_cov =
      generateCovarianceMatrix<double, 3>() * 0.01;

  // Get the similarity between the two planes (closer to 1 is better)
  // Note: This metric is only used when the two planes are considered adjacent,
  // since it is based on the cosine of the expected angle between the two plane normals.
  // Refer to the paper for a more in-depth discussion of this metric.
  const double similarity =
      estimateAngleBetweenPlanes(plane1_mean, plane2_mean, plane1_cov, plane2_cov);

  // Alternative implementation using MATLAB-generated code.
  // We provide this version for specific systems where it may be faster than the above function,
  // but please refer to the original implementation for understanding the underlying math.
  //const double similarity = estimateAngleBetweenPlanes_optimized(plane1_mean, plane2_mean,
  //                                            plane1_cov, plane2_cov);

  std::cout << "Estimated similarity between planes: " << similarity << std::endl;
}