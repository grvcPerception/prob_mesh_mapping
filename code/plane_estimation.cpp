#include <iostream>
#include <random>

#include "math.h"

#define N_sample_points 5

int main(int argc, char** argv) {
  srand(static_cast<unsigned>(time(0)));

  // Generate a random plane by:
  // 1. Starting from a plane that intersects the origin and is parallel to the z-axis
  // 2. Adding a random perturbation both in the orientation and the distance to the origin
  // Note: This plane is just for creating dummy data, we are not trying to estimate this exact plane
  // Refer to the paper for a validation of plane and plane covariance estimation
  const Eigen::Vector3d z_unit = (Eigen::Vector3d() << 0.0, 0.0, 1.0).finished();
  const Eigen::Vector3d perturbation =
      Eigen::Vector3d::Random().cwiseProduct(Eigen::Vector3d(0.1, 0.1, 3));
  const Eigen::Vector4d plane = s2_BP(Eigen::Vector4d(0.0, 0.0, 1.0, 0.0), perturbation);

  // Generate N random points (on the plane) and their covariances:
  PointVector<double> point_means;
  CovVector<double> point_covs;
  for (int i = 0; i < N_sample_points; ++i) {
    const auto [point, covariance] = generateRandomPointOnPlane(plane);
    point_means.push_back(point);
    point_covs.push_back(covariance);
  }

  const auto [estimated_plane, estimated_cov] =
      estimateProbabilisticPlane(point_means, point_covs);
  std::cout << "Estimated plane: " << estimated_plane << std::endl;
  std::cout << "Estimated covariance: " << estimated_cov << std::endl;
}