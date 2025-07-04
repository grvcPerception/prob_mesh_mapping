#include <iostream>
#include <random>

#include "math.h"

#define N_sample_points 5

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

  // Generate a random covariance for the first plane
  const Eigen::Matrix<double, 3, 3> plane1_cov =
      generateCovarianceMatrix<double, 3>() * 0.1;

  // Generate N random points on the second plane and their covariances:
  PointVector<double> points_plane2_means;
  CovVector<double> points_plane2_covs;
  for (int i = 0; i < N_sample_points; ++i) {
    const auto [point, covariance] = generateRandomPointOnPlane(plane2_mean);
    points_plane2_means.push_back(point);
    points_plane2_covs.push_back(covariance);
  }

  // Merge the plane and the points into one plane
  const auto [estimated_plane, estimated_cov] = aggregateProbabilisticPlaneAndPoints(
      plane1_mean, plane1_cov, points_plane2_means, points_plane2_covs);

  std::cout << "Estimated plane: " << estimated_plane << std::endl;
  std::cout << "Estimated covariance: " << estimated_cov << std::endl;
}