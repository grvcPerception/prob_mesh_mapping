#include <iostream>
#include <random>

#include "math.h"

#define N_sample_neighbor_points 5

int main(int argc, char** argv) {
  srand(static_cast<unsigned>(time(0)));

  // Generate a random plane by:
  // 1. Starting from a plane that intersects the origin and is parallel to the z-axis
  // 2. Adding a random perturbation both in the orientation and the distance to the origin
  // Note: This plane is just for creating dummy data, we are not trying to estimate the distance to this exact plane
  // Refer to the paper for a validation of plane and plane covariance estimation
  const Eigen::Vector3d z_unit = (Eigen::Vector3d() << 0.0, 0.0, 1.0).finished();
  const Eigen::Vector3d perturbation =
      Eigen::Vector3d::Random().cwiseProduct(Eigen::Vector3d(0.1, 0.1, 3));
  const Eigen::Vector4d plane = s2_BP(Eigen::Vector4d(0.0, 0.0, 1.0, 0.0), perturbation);

  // Generate a random measured point on the plane and its covariance
  const auto [measurement_point, measurement_cov] = generateRandomPointOnPlane(plane);

  // Generate N random neighbor points on the plane and their covariances
  // (these will be used to calculate a representation of the locally planar map)
  PointVector<double> neighbor_points_means;
  CovVector<double> neighbor_points_cov;
  for (int i = 0; i < N_sample_neighbor_points; ++i) {
    const auto [point, covariance] = generateRandomPointOnPlane(plane);
    neighbor_points_means.push_back(point);
    neighbor_points_cov.push_back(covariance);
  }

  // Estimate the map plane and its covariance from the neighbor points
  const auto [estimated_plane, estimated_cov] =
      estimateProbabilisticPlane(neighbor_points_means, neighbor_points_cov);

  // Obtain the Jacobian D_q as the derivative of the measurement function (point-plane-distance) with
  // respect to the plane noise
  const Eigen::Matrix<double, 3, 1> estimated_plane_normal = estimated_plane.head(3);
  const Eigen::Matrix<double, 1, 3> D_q =
      (Eigen::Matrix<double, 1, 3>() << -measurement_point.transpose() *
                                            skew(estimated_plane_normal) *
                                            s2_B(estimated_plane_normal),
       1)
          .finished();

  // Obtain the Jacobian D_p as the derivative of the measurement function (point-plane-distance) with
  // respect to the point noise.
  // Note: In this case the rotation R_W_L is the identity matrix since we assume no change in the
  // state between scans
  const Eigen::Matrix<double, 1, 3> D_p = estimated_plane_normal.transpose();

  // Obtain the distance and distance covariance for this pair of measurement and neighbor points
  // that would be used in a state estimation scheme
  const double distance =
      measurement_point.dot(estimated_plane_normal) + estimated_plane(3);
  const double distance_cov = (D_q * estimated_cov * D_q.transpose()).value() +
                              (D_p * measurement_cov * D_p.transpose()).value();

  std::cout << "Estimated distance: " << distance << std::endl;
  std::cout << "Estimated distance covariance: " << distance_cov << std::endl;
}