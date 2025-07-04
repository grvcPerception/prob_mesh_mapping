#pragma once

#include <eigen3/Eigen/Dense>

/*
  Helper types for Eigen
*/
template <typename T>
using PointVector =
    std::vector<Eigen::Matrix<T, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, 3, 1>>>;

template <typename T>
using PointVectorToMatrixMap = Eigen::Map<const Eigen::Matrix<T, 3, Eigen::Dynamic>>;

template <typename T>
using CovVector =
    std::vector<Eigen::Matrix<T, 3, 3>, Eigen::aligned_allocator<Eigen::Matrix<T, 3, 3>>>;

/*
  Helper functions for data generation
*/
// Generate a dense n x n symmetric, positive definite matrix
template <typename T, int n>
const Eigen::Matrix<T, n, n> generateCovarianceMatrix() {
  const Eigen::Matrix<T, n, n> Temp = Eigen::Matrix<T, n, n>::Random();
  return Temp * Temp.transpose();
}

// Generate multivariate Gaussian noise from a given covariance matrix
template <typename T, int n>
const Eigen::Matrix<T, n, 1> generateMultivariateNoiseFromCovariance(
    const Eigen::Matrix<T, n, n>& covariance) {
  // Create random number generator
  static std::random_device rd{};
  static std::mt19937 gen{rd()};
  static std::normal_distribution<T> dist{0.0, 1.0};

  // Generate standard normal random numbers
  Eigen::Matrix<T, n, 1> standard_normal;
  for (int i = 0; i < n; i++) {
    standard_normal(i) = dist(gen);
  }

  // Perform Cholesky decomposition of covariance matrix
  Eigen::LLT<Eigen::Matrix<T, n, n>> llt(covariance);
  const Eigen::Matrix<T, n, n> L = llt.matrixL();

  // Transform standard normal to desired distribution
  return L * standard_normal;
}

// Generate a random point on a plane and its covariance by:
// 1. Generating the random point in 3D
// 2. Projecting it onto the plane
// 3. Generating a covariance matrix for the point
// 4. Adding a random noise to the point based on the covariance matrix
template <typename T>
std::pair<Eigen::Matrix<T, 3, 1>, Eigen::Matrix<T, 3, 3>> generateRandomPointOnPlane(
    const Eigen::Matrix<T, 4, 1>& plane) {
  const Eigen::Matrix<T, 3, 1> random_point = Eigen::Matrix<T, 3, 1>::Random() * 2;
  const T dist = random_point.dot(plane.template head<3>()) + plane(3);

  const Eigen::Matrix<T, 3, 3> covariance = generateCovarianceMatrix<T, 3>() * 0.01;
  const Eigen::Matrix<T, 3, 1> point =
      random_point - dist * plane.template head<3>() +
      generateMultivariateNoiseFromCovariance(covariance);

  return std::make_pair(point, covariance);
}

/*
  Helper functions for on-manifold operations
*/
// Generator of the basis matrix for the S2 manifold
// This version is taken from Gao et. al. RAL 2022 Anderson Acceleration for on-Manifold IESKF
// at https://github.com/gaoxiang12/faster-lio
template <typename T>
const inline Eigen::Matrix<T, 3, 2> s2_B(const Eigen::Matrix<T, 3, 1>& vec) {
  const T a1Sq = vec(0) * vec(0) + vec(1) * vec(1) + vec(2) * vec(2);
  const T a1 = sqrt(a1Sq);
  const T a = vec(0);
  const T b = vec(1);
  const T c = vec(2);

  return (Eigen::Matrix<T, 3, 2>() << -b, -c, a1 - b * b / (a1 + a), -b * c / (a1 + a),
          -b * c / (a1 + a), a1 - c * c / (a1 + a))
      .finished();  // Optimized version of the rotation times the span matrix
}

// Skew-symmetric matrix operator
template <typename T>
const inline Eigen::Matrix<T, 3, 3> skew(const Eigen::Matrix<T, 3, 1>& vec) {
  return (Eigen::Matrix<T, 3, 3>() << 0, -vec(2), vec(1), vec(2), 0, -vec(0), -vec(1),
          vec(0), 0)
      .finished();
}

// Exponential map for the SO(3) manifold (rotation matrix)
template <typename T>
const inline Eigen::Matrix<T, 3, 3> so3_Exp(const Eigen::Matrix<T, 3, 1>& vec) {
  T mod = vec.norm();
  if (mod < 0.0001) {
    return Eigen::Matrix<T, 3, 3>::Identity();
  } else {
    const Eigen::Matrix<T, 3, 3> vec_skew = skew(vec);
    const Eigen::Matrix<T, 3, 3> vec_skew_sq = vec_skew * vec_skew;
    return Eigen::Matrix<T, 3, 3>::Identity() + (sin(mod) / mod) * vec_skew +
           ((T(1.0) - cos(mod)) / (mod * mod)) * vec_skew_sq;
  }
}

// Boxplus operator for the S2 manifold (unit sphere) (without precomputed basis matrix)
template <typename T>
const inline Eigen::Matrix<T, 3, 1> s2_BP(const Eigen::Matrix<T, 3, 1>& x,
                                          const Eigen::Matrix<T, 2, 1>& u) {
  const Eigen::Matrix<T, 3, 1> aux = s2_B(x) * u;
  return so3_Exp(aux) * x;
}

// Boxplus operator for the S2 manifold (unit sphere) (with precomputed basis matrix)
template <typename T>
const inline Eigen::Matrix<T, 3, 1> s2_BP(const Eigen::Matrix<T, 3, 1>& x,
                                          const Eigen::Matrix<T, 2, 1>& u,
                                          const Eigen::Matrix<T, 3, 2>& Bx) {
  const Eigen::Matrix<T, 3, 1> aux = Bx * u;
  return so3_Exp(aux) * x;
}

// Boxplus operator for the S2 manifold (unit sphere) (without precomputed basis matrix)
// Abuse of notation in the function name, this is technically s2xN_BP
template <typename T>
const inline Eigen::Matrix<T, 4, 1> s2_BP(const Eigen::Matrix<T, 4, 1>& x,
                                          const Eigen::Matrix<T, 3, 1>& u) {
  const Eigen::Matrix<T, 2, 1> uh = u.head(2);
  const Eigen::Matrix<T, 3, 1> xh = x.head(3);
  return (Eigen::Matrix<T, 4, 1>() << s2_BP(xh, uh), x(3) + u(2)).finished();
}

// Boxplus operator for the S2 manifold (unit sphere) (with precomputed basis matrix)
// Abuse of notation in the function name, this is technically s2xN_BP
template <typename T>
const inline Eigen::Matrix<T, 4, 1> s2_BP(const Eigen::Matrix<T, 4, 1>& x,
                                          const Eigen::Matrix<T, 3, 1>& u,
                                          const Eigen::Matrix<T, 3, 2>& Bx) {
  const Eigen::Matrix<T, 2, 1> uh = u.head(2);
  const Eigen::Matrix<T, 3, 1> xh = x.head(3);
  return (Eigen::Matrix<T, 4, 1>() << s2_BP(xh, uh, Bx), x(3) + u(2)).finished();
}

// Helper map (similar to the exponential map for the SO(3) manifold) provided in https://github.com/hku-mars/IKFoM
template <typename T>
const inline Eigen::Matrix<T, 3, 3> s2_A(const Eigen::Matrix<T, 3, 1>& vec) {
  T mod = vec.norm();
  if (mod < 0.0001) {
    return Eigen::Matrix<T, 3, 3>::Identity();
  } else {
    const Eigen::Matrix<T, 3, 3> vec_skew = skew(vec);
    const Eigen::Matrix<T, 3, 3> vec_skew_sq = vec_skew * vec_skew;
    return Eigen::Matrix<T, 3, 3>::Identity() +
           (T(1.0) - sin(mod) / mod) / (mod * mod) * vec_skew_sq +
           ((T(1.0) - cos(mod)) / (mod * mod)) * vec_skew;
  }
}

// Boxminus operator for the S2 manifold (unit sphere) (without precomputed basis matrix)
template <typename T>
const inline Eigen::Matrix<T, 2, 1> s2_BM(const Eigen::Matrix<T, 3, 1>& y,
                                          const Eigen::Matrix<T, 3, 1>& x) {
  const T theta = atan2((skew(x) * y).norm(), x.transpose() * y);
  if (theta < 0.0001)
    return Eigen::Matrix<T, 2, 1>::Zero();

  const Eigen::Matrix<T, 3, 1> svec = skew(x) * y;
  return (Eigen::Matrix<T, 2, 1>() << s2_B(x).transpose() * theta * svec / svec.norm())
      .finished();
}

// Boxminus operator for the S2 manifold (unit sphere) (with precomputed basis matrix)
template <typename T>
const inline Eigen::Matrix<T, 2, 1> s2_BM(const Eigen::Matrix<T, 3, 1>& y,
                                          const Eigen::Matrix<T, 3, 1>& x,
                                          const Eigen::Matrix<T, 3, 2>& Bx) {
  const T theta = atan2((skew(x) * y).norm(), x.transpose() * y);
  if (theta < 0.0001)
    return Eigen::Matrix<T, 2, 1>::Zero();

  const Eigen::Matrix<T, 3, 1> svec = skew(x) * y;
  return (Eigen::Matrix<T, 2, 1>() << Bx.transpose() * theta * svec / svec.norm())
      .finished();
}

// Boxminus operator for the S2xN manifold (plane) (without precomputed basis matrix)
// Abuse of notation in the function name, this is technically s2xN_BM
template <typename T>
const inline Eigen::Matrix<T, 3, 1> s2_BM(const Eigen::Matrix<T, 4, 1>& y,
                                          const Eigen::Matrix<T, 4, 1>& x) {
  const Eigen::Matrix<T, 3, 1> yh = y.head(3);
  const Eigen::Matrix<T, 3, 1> xh = x.head(3);
  return (Eigen::Matrix<T, 3, 1>() << s2_BM(yh, xh), y(3) - x(3)).finished();
}

// Boxminus operator for the S2xN manifold (plane) (with precomputed basis matrix)
// Abuse of notation in the function name, this is technically s2xN_BM
template <typename T>
const inline Eigen::Matrix<T, 3, 1> s2_BM(const Eigen::Matrix<T, 4, 1>& y,
                                          const Eigen::Matrix<T, 4, 1>& x,
                                          const Eigen::Matrix<T, 3, 2>& Bx) {
  const Eigen::Matrix<T, 3, 1> yh = y.head(3);
  const Eigen::Matrix<T, 3, 1> xh = x.head(3);
  return (Eigen::Matrix<T, 3, 1>() << s2_BM(yh, xh, Bx), y(3) - x(3)).finished();
}

// Rotation matrix used to generate the orthonormal basis of the tangent space
// at a given orientation. This function can be used with any orthonormal basis
// vector, but in the paper we always use the first basis vector e1 = [1; 0; 0].
template <typename T>
const inline Eigen::Matrix<T, 3, 3> s2_R1(const Eigen::Matrix<T, 3, 1>& x) {
  const static Eigen::Matrix<T, 3, 1> e1(T(1), T(0), T(0));
  const static Eigen::Matrix<T, 3, 3> e1_skew = skew(e1);
  const Eigen::Matrix<T, 3, 1> aux = e1_skew * x;
  const T uN = aux.norm();
  if (uN > T(0.000000001)) {
    const Eigen::Matrix<T, 3, 1> aux2 = aux * std::atan2(uN, x(0)) / uN;
    return so3_Exp(aux2);
  } else
    return Eigen::Matrix<T, 3, 3>::Identity();
}

// Jacobian to perform covariance reset in an iterated Kalman filter
// Refer to https://github.com/hku-mars/IKFoM for more details
template <typename T>
const inline Eigen::Matrix<T, 3, 3> s2_L(const Eigen::Matrix<T, 3, 1>& nj,
                                         const Eigen::Matrix<T, 3, 1>& n0,
                                         const Eigen::Matrix<T, 2, 1>& dno,
                                         const Eigen::Matrix<T, 3, 2>& Bj,
                                         const Eigen::Matrix<T, 3, 2>& B0) {
  const Eigen::Matrix<T, 3, 3> n0_skew = skew(n0);
  const Eigen::Matrix<T, 3, 1> prodAux = B0 * dno;
  return (Eigen::Matrix<T, 3, 3>() << -Bj.transpose() * so3_Exp(prodAux) * n0_skew *
                                          n0_skew * s2_A(prodAux).transpose() * B0,
          Eigen::Vector2d::Zero(), 0, 0, 1)
      .finished();
}

/*
  Core functions of the methods in the paper
*/
// Function for estimating a plane from a set of points
// Note: This function uses Singular Value decomposition of a  Nx3 mean-centered points matrix
// for better numerical stability in these examples. Other approaches, such as Z-axis distance minimization,
// or using an incremental summation matrix (see for example the implementation of Voxel-map) are also valid
// and generally faster (but less accurate). In the paper, we use the incremmental summation matrix approach
// following the comments from the reviewers, to give fairer performance comparisons with the state-of-the-art.
template <typename T>
const Eigen::Matrix<T, 4, 1> estimatePlane(const PointVector<T>& points) {
  Eigen::Matrix<T, 3, Eigen::Dynamic> A =
      PointVectorToMatrixMap<T>(points[0].data(), 3, points.size());
  const Eigen::Matrix<T, 3, 1> Am = A.rowwise().mean();
  A.row(0) = A.row(0).array() - Am(0);
  A.row(1) = A.row(1).array() - Am(1);
  A.row(2) = A.row(2).array() - Am(2);

  auto svd = A.jacobiSvd(Eigen::ComputeFullU);
  const Eigen::Matrix<T, 3, 1> normvec = svd.matrixU().template rightCols<1>();

  T n = normvec.norm();

  return (Eigen::Matrix<T, 4, 1>() << -normvec(0) / n, -normvec(1) / n, -normvec(2) / n,
          (normvec(0) * Am(0) + normvec(1) * Am(1) + normvec(2) * Am(2)) / n)
      .finished();
}

// Function for estimating a probabilistic plane (mean and covariance) from a set of points
// Note: As explained in the paper, the approach followed in this function is independent on the estimation
// of the mean of the plane, as long as it is accurate enough. Using SVD, Z-axis distance minimization,
// or the incremental summation matrix approach are all valid approaches.
template <typename T>
const std::pair<const Eigen::Matrix<T, 4, 1>, const Eigen::Matrix<T, 3, 3>>
estimateProbabilisticPlane(const PointVector<T>& points, const CovVector<T>& covs) {
  const Eigen::Matrix<T, 4, 1> mean = estimatePlane(points);

  const Eigen::Matrix<T, 3, 1> normal = mean.head(3);
  const T distance = mean(3);
  const Eigen::Matrix<T, 3, 2> Bnormal = s2_B(normal);
  const Eigen::Matrix<T, 3, 3> normalSkew = skew(normal);

  Eigen::Matrix<T, Eigen::Dynamic, 3> J =
      Eigen::Matrix<T, Eigen::Dynamic, 3>::Zero(points.size(), 3);

  for (int j = 0; j < points.size(); ++j) {
    const Eigen::Matrix<T, 3, 3> Sigma_j = covs.at(j);
    const Eigen::Matrix<T, 3, 1> p_j = points.at(j);
    const T Rinv_j = T(1.0) / (normal.transpose() * Sigma_j * normal);
    const T sqrtRinv_j = sqrt(Rinv_j);

    J.template block<1, 2>(j, 0) = (p_j.dot(normal) + distance) * normal.transpose() *
                                   Sigma_j * normalSkew * Bnormal * sqrtRinv_j * Rinv_j;
    J.template block<1, 2>(j, 0) -= p_j.transpose() * normalSkew * Bnormal * sqrtRinv_j;
    J(j, 2) = sqrtRinv_j;
  }

  // Calculation of (J^T * J)^-1 in an Eigen-efficient way
  const Eigen::Matrix<T, 3, 3> covariance =
      (J.transpose() * J).llt().solve(Eigen::Matrix<T, 3, 3>::Identity(3, 3));

  return std::make_pair(mean, covariance);
}

// Function for estimating the similarity between two planes, based on the second-order
// error propagation of the cosine of the expected angle between two planes with on-manifold uncertainty.
template <typename T>
const T estimateAngleBetweenPlanes(const Eigen::Matrix<T, 4, 1>& plane_1_mean,
                                   const Eigen::Matrix<T, 4, 1>& plane_2_mean,
                                   const Eigen::Matrix<T, 3, 3>& plane_1_cov,
                                   const Eigen::Matrix<T, 3, 3>& plane_2_cov) {
  const Eigen::Matrix<T, 3, 1> n1 = plane_1_mean.head(3);
  const Eigen::Matrix<T, 3, 1> n2 = plane_2_mean.head(3);
  const Eigen::Matrix<T, 2, 2> n1_cov = plane_1_cov.template block<2, 2>(0, 0);
  const Eigen::Matrix<T, 2, 2> n2_cov = plane_2_cov.template block<2, 2>(0, 0);

  const Eigen::Matrix<T, 4, 4> combined_normal_cov =
      (Eigen::Matrix<T, 4, 4>() << n1_cov, Eigen::Matrix<T, 2, 2>::Zero(),
       Eigen::Matrix<T, 2, 2>::Zero(), n2_cov)
          .finished();

  const Eigen::Matrix<T, 3, 2> Bn1 = s2_B(n1);
  const Eigen::Matrix<T, 3, 1> Bn1_1 = Bn1.col(0);
  const Eigen::Matrix<T, 3, 1> Bn1_2 = Bn1.col(1);
  const Eigen::Matrix<T, 3, 2> Bn2 = s2_B(n2);
  const Eigen::Matrix<T, 3, 1> Bn2_1 = Bn2.col(0);
  const Eigen::Matrix<T, 3, 1> Bn2_2 = Bn2.col(1);

  const Eigen::Matrix<T, 3, 3> b11_skew = skew(Bn1_1);
  const Eigen::Matrix<T, 3, 3> b12_skew = skew(Bn1_2);
  const Eigen::Matrix<T, 3, 3> b21_skew = skew(Bn2_1);
  const Eigen::Matrix<T, 3, 3> b22_skew = skew(Bn2_2);

  const Eigen::Matrix<T, 3, 3> n1_skew = skew(n1);
  const Eigen::Matrix<T, 3, 3> n2_skew = skew(n2);

  const Eigen::Matrix<T, 1, 4> J =
      (Eigen::Matrix<T, 1, 4>() << -n2.transpose() * n1_skew * Bn1,
       -n1.transpose() * n2_skew * Bn2)
          .finished();

  const Eigen::Matrix<T, 3, 3> g_n1_b11 =
      T(0.5) * (n1_skew * b11_skew) - b11_skew * n1_skew;
  const Eigen::Matrix<T, 3, 3> g_n1_b12 =
      T(0.5) * (n1_skew * b12_skew) - b12_skew * n1_skew;
  const Eigen::Matrix<T, 3, 3> g_n2_b21 =
      T(0.5) * (n2_skew * b21_skew) - b21_skew * n2_skew;
  const Eigen::Matrix<T, 3, 3> g_n2_b22 =
      T(0.5) * (n2_skew * b22_skew) - b22_skew * n2_skew;

  const Eigen::Matrix<T, 2, 2> H_11 =
      (Eigen::Matrix<T, 2, 3>() << n2.transpose() * g_n1_b11, n2.transpose() * g_n1_b12)
          .finished() *
      Bn1;
  const Eigen::Matrix<T, 2, 2> H_22 =
      (Eigen::Matrix<T, 2, 3>() << n1.transpose() * g_n2_b21, n1.transpose() * g_n2_b22)
          .finished() *
      Bn2;
  const Eigen::Matrix<T, 2, 2> H_12 = (n1_skew * Bn1).transpose() * n2_skew * Bn2;
  const Eigen::Matrix<T, 2, 2> H_21 = (n2_skew * Bn2).transpose() * n1_skew * Bn1;

  const Eigen::Matrix<T, 4, 4> H =
      (Eigen::Matrix<T, 4, 4>() << H_11, H_12, H_21, H_22).finished();

  const T gamma = n1.transpose() * n2 + T(0.5) * (H * combined_normal_cov).trace();
  const T gamma_cov_sq =
      J * combined_normal_cov * J.transpose() +
      T(0.5) * (H * combined_normal_cov * H * combined_normal_cov).trace();

  return gamma - T(3.0) * sqrt(gamma_cov_sq);
}

// Simple symbolic optimization of the above function based on MATLAB's symbolic toolbox
// We provide this version for specific systems where it may be faster than the above function,
// but please refer to the original implementation for understanding the underlying math.
template <typename T>
const T estimateAngleBetweenPlanes_optimized(const Eigen::Matrix<T, 4, 1>& plane_1_mean,
                                             const Eigen::Matrix<T, 4, 1>& plane_2_mean,
                                             const Eigen::Matrix<T, 3, 3>& plane_1_cov,
                                             const Eigen::Matrix<T, 3, 3>& plane_2_cov) {
  const T n1_1 = plane_1_mean(0);
  const T n1_2 = plane_1_mean(1);
  const T n1_3 = plane_1_mean(2);
  const T n2_1 = plane_2_mean(0);
  const T n2_2 = plane_2_mean(1);
  const T n2_3 = plane_2_mean(2);
  const T S1_1_1 = plane_1_cov(0, 0);
  const T S1_1_2 = plane_1_cov(0, 1);
  const T S1_2_2 = plane_1_cov(1, 1);
  const T S2_1_1 = plane_2_cov(0, 0);
  const T S2_1_2 = plane_2_cov(0, 1);
  const T S2_2_2 = plane_2_cov(1, 1);

  T t10, t100, t101, t104, t105, t107, t108, t110, t111, t112, t113, t12, t13, t133, t134,
      t135, t136, t138, t139, t140, t141, t150, t153, t156, t159, t165, t17, t178, t18,
      t181, t19, t20, t21, t22, t23, t26, t262, t264, t265, t27, t273, t306, t307, t308,
      t309, t32, t33, t354, t355, t36, t37, t375, t44, t45, t48, t49, t62, t63, t64, t65,
      t76, t77, t80, t81, t82_tmp, t84, t85, t88, t88_tmp, t89, t9, t90, t91, t94, t95,
      t96, t97, t98, t99;

  t9 = n1_2 * n1_2;
  t10 = n1_3 * n1_3;
  t12 = n2_2 * n2_2;
  t13 = n2_3 * n2_3;
  t17 = t9 + t10;
  t18 = t12 + t13;
  t19 = n1_1 * n2_2 + -(n1_2 * n2_1);
  t20 = n1_1 * n2_3 + -(n1_3 * n2_1);
  t21 = n1_2 * n2_3 + -(n1_3 * n2_2);
  t22 = T(1.0) / t17;
  t23 = T(1.0) / t18;
  t26 = std::sqrt(t17);
  t27 = std::sqrt(t18);
  t32 = T(1.0) / t26;
  t33 = T(1.0) / t27;
  t36 = std::sqrt(n1_1 * n1_1 + t17);
  t37 = std::sqrt(n2_1 * n2_1 + t18);
  t44 = std::atan2(t26, n1_1);
  t45 = std::atan2(t27, n2_1);
  t26 = T(1.0) / (n1_1 + t36);
  t27 = T(1.0) / (n2_1 + t37);
  t48 = t44 * t44;
  t49 = t45 * t45;
  t62 = t9 * t22 * t48;
  t63 = t10 * t22 * t48;
  t64 = t12 * t23 * t49;
  t65 = t13 * t23 * t49;
  t17 = t62 + t63;
  t18 = t64 + t65;
  t76 = T(1.0) / t17;
  t77 = T(1.0) / t18;
  t17 = std::sqrt(t17);
  t18 = std::sqrt(t18);
  t80 = T(1.0) / t17;
  t81 = T(1.0) / t18;
  t82_tmp = std::cos(t17);
  t354 = std::cos(t18);
  t84 = std::sin(t17);
  t85 = std::sin(t18);
  t88_tmp = n1_2 * n1_3;
  t88 = (n1_2 * t21 + t88_tmp * t19 * t26) + t20 * (t36 + -(t9 * t26));
  t355 = n2_2 * n2_3;
  t89 = (n2_2 * t21 + t355 * t19 * t27) + t20 * (t37 + -(t12 * t27));
  t90 = (-(n1_3 * t21) + t88_tmp * t20 * t26) + t19 * (t36 + -(t10 * t26));
  t91 = (-(n2_3 * t21) + t355 * t20 * t27) + t19 * (t37 + -(t13 * t27));
  t36 = t88_tmp * t32 * t44 * t80 * t84;
  t37 = t355 * t33 * t45 * t81 * t85;
  t94 = t9 * t32 * t44 * t80 * t84;
  t95 = t10 * t32 * t44 * t80 * t84;
  t96 = t12 * t33 * t45 * t81 * t85;
  t97 = t13 * t33 * t45 * t81 * t85;
  t17 = n1_1 * n1_2;
  t104 = t17 * t32 * t44 * t80 * t84 * T(0.5);
  t105 = n1_1 * n1_3 * t32 * t44 * t80 * t84 * T(0.5);
  t18 = n2_1 * n2_2;
  t107 = t18 * t33 * t45 * t81 * t85 * T(0.5);
  t108 = n2_1 * n2_3 * t33 * t45 * t81 * t85 * T(0.5);
  t98 = t62 * t76 * (t82_tmp - T(1.0));
  t99 = t63 * t76 * (t82_tmp - T(1.0));
  t100 = t64 * t77 * (t354 - T(1.0));
  t101 = t65 * t77 * (t354 - T(1.0));
  t17 = t17 * n1_3 * t22 * t48 * t76 * (t82_tmp - T(1.0));
  t18 = t18 * n2_3 * t23 * t49 * t77 * (t354 - T(1.0));
  t110 = n1_2 * t99;
  t111 = n1_3 * t98;
  t112 = n2_2 * t101;
  t113 = n2_3 * t100;
  t26 = t36 + t17;
  t27 = t37 + t18;
  t21 = n1_1 * (t98 + T(1.0));
  t133 = n1_1 * (t99 + T(1.0));
  t134 = n1_2 * (t99 + T(1.0));
  t135 = n1_3 * (t98 + T(1.0));
  t136 = n2_1 * (t100 + T(1.0));
  t9 = n2_1 * (t101 + T(1.0));
  t138 = n2_2 * (t101 + T(1.0));
  t139 = n2_3 * (t100 + T(1.0));
  t140 = t110 * T(0.5);
  t141 = t111 * T(0.5);
  t10 = t112 * T(0.5);
  t12 = t113 * T(0.5);
  t150 = n1_2 * (t98 + T(1.0)) * T(0.5);
  t153 = n1_3 * (t99 + T(1.0)) * T(0.5);
  t156 = n2_2 * (t100 + T(1.0)) * T(0.5);
  t159 = n2_3 * (t101 + T(1.0)) * T(0.5);
  t65 = n2_1 * t26;
  t165 = n1_1 * t27;
  t19 = t36 + t17 * T(0.5);
  t20 = t37 + t18 * T(0.5);
  t178 = n2_1 * (t17 + t36 * T(0.5));
  t181 = n1_1 * (t18 + t37 * T(0.5));
  t17 = t94 + t21;
  t18 = t95 + t133;
  t36 = t96 + t136;
  t37 = t97 + t9;
  t262 = n2_3 * t26 + n2_2 * t17;
  t63 = n2_2 * t26 + n2_3 * t18;
  t264 = n1_3 * t27 + n1_2 * t36;
  t265 = n1_2 * t27 + n1_3 * t37;
  t26 = t111 - t135;
  t27 = n2_1 * t17 + -n2_3 * t26;
  t13 = t110 - t134;
  t18 = n2_1 * t18 + -n2_2 * t13;
  t62 = t113 - t139;
  t64 = n1_1 * t36 + -n1_3 * t62;
  t375 = t112 - t138;
  t273 = n1_1 * t37 + -n1_2 * t375;
  t306 = (n2_3 * t19 + n2_2 * (t94 + t21 * T(0.5))) + n2_1 * (t140 + t150);
  t307 = (n2_2 * t19 + n2_3 * (t95 + t133 * T(0.5))) + n2_1 * (t141 + t153);
  t308 = (n1_3 * t20 + n1_2 * (t96 + t136 * T(0.5))) + n1_1 * (t10 + t156);
  t309 = (n1_2 * t20 + n1_3 * (t97 + t9 * T(0.5))) + n1_1 * (t12 + t159);
  t94 = (n2_1 * (t94 * T(0.5) + t21) + n2_2 * (t104 + -t140)) + n2_3 * (t135 + -t141);
  t20 = (n2_1 * (t95 * T(0.5) + t133) + n2_3 * (t105 + -t141)) + n2_2 * (t134 + -t140);
  t133 = (n1_1 * (t96 * T(0.5) + t136) + n1_2 * (t107 + -t10)) + n1_3 * (t139 + -t12);
  t21 = (n1_1 * (t97 * T(0.5) + t9) + n1_3 * (t108 + -t12)) + n1_2 * (t138 + -t10);
  t19 = t355 * t23 * t49 * t77 * (t354 - T(1.0));
  t17 = t65 + n2_2 * t26;
  t37 = n2_2 * t33 * t45 * t81 * t85;
  t23 = (t37 * t262 + t19 * t17) + (t100 + T(1.0)) * t27;
  t36 = n2_3 * t33 * t45 * t81 * t85;
  t85 = (t36 * t262 + t19 * t27) + (t101 + T(1.0)) * t17;
  t17 = t65 + n2_3 * t13;
  t77 = (t37 * t63 + t19 * t18) + (t100 + T(1.0)) * t17;
  t49 = (t36 * t63 + t19 * t17) + (t101 + T(1.0)) * t18;
  t27 = t88_tmp * t22 * t48 * t76 * (t82_tmp - T(1.0));
  t17 = t165 + n1_2 * t62;
  t26 = n1_2 * t32 * t44 * t80 * t84;
  t354 = (t26 * t264 + t27 * t17) + (t98 + T(1.0)) * t64;
  t18 = n1_3 * t32 * t44 * t80 * t84;
  t355 = (t18 * t264 + t27 * t64) + (t99 + T(1.0)) * t17;
  t17 = t165 + n1_3 * t375;
  t165 = (t26 * t265 + t27 * t273) + (t98 + T(1.0)) * t17;
  t264 = (t18 * t265 + t27 * t17) + (t99 + T(1.0)) * t273;
  t17 = (t178 + n2_3 * (t104 - t150)) + n2_2 * (t111 - t135 * T(0.5));
  t273 = (t26 * t306 + t27 * t17) + (t98 + T(1.0)) * t94;
  t375 = (t18 * t306 + t27 * t94) + (t99 + T(1.0)) * t17;
  t17 = (t178 + n2_2 * (t105 - t153)) + n2_3 * (t110 - t134 * T(0.5));
  t97 = (t26 * t307 + t27 * t20) + (t98 + T(1.0)) * t17;
  t96 = (t18 * t307 + t27 * t17) + (t99 + T(1.0)) * t20;
  t17 = (t181 + n1_3 * (t107 - t156)) + n1_2 * (t113 - t139 * T(0.5));
  t95 = (t37 * t308 + t19 * t17) + (t100 + T(1.0)) * t133;
  t94 = (t36 * t308 + t19 * t133) + (t101 + T(1.0)) * t17;
  t17 = (t181 + n1_2 * (t108 - t159)) + n1_3 * (t112 - t138 * T(0.5));
  t262 = (t37 * t309 + t19 * t21) + (t100 + T(1.0)) * t17;
  t13 = (t36 * t309 + t19 * t17) + (t101 + T(1.0)) * t21;
  t17 = S1_1_1 * t273;
  t18 = S1_1_2 * t375;
  t26 = S1_1_2 * t97;
  t27 = S1_2_2 * t96;
  t36 = S2_1_1 * t95;
  t37 = S2_1_2 * t94;
  t10 = S2_1_2 * t262;
  t12 = S2_2_2 * t13;
  const T gamma = (((((((((t17 * -T(0.5) - t18 * T(0.5)) - t26 * T(0.5)) - t27 * T(0.5)) -
                        t36 * T(0.5)) -
                       t37 * T(0.5)) -
                      t10 * T(0.5)) -
                     t12 * T(0.5)) +
                    n1_1 * n2_1) +
                   n1_2 * n2_2) +
                  n1_3 * n2_3;
  t62 = S2_1_1 * t23 + S2_1_2 * t85;
  t63 = S2_1_1 * t77 + S2_1_2 * t49;
  t64 = S2_1_2 * t23 + S2_2_2 * t85;
  t65 = S2_1_2 * t77 + S2_2_2 * t49;
  t133 = S1_1_1 * t354 + S1_1_2 * t355;
  t140 = S1_1_1 * t165 + S1_1_2 * t264;
  t141 = S1_1_2 * t354 + S1_2_2 * t355;
  t136 = S1_1_2 * t165 + S1_2_2 * t264;
  t20 = t17 + t18;
  t21 = S1_1_2 * t273 + S1_2_2 * t375;
  t9 = S1_1_1 * t97 + S1_1_2 * t96;
  t19 = t26 + t27;
  t18 = t36 + t37;
  t26 = S2_1_2 * t95 + S2_2_2 * t94;
  t27 = S2_1_1 * t262 + S2_1_2 * t13;
  t17 = t10 + t12;
  const T gamma_std = std::sqrt(
      ((((((((((S1_1_1 * (((t354 * t62 + t165 * t64) + t273 * t20) + t97 * t21) * T(0.5) +
                S1_1_2 * (((t355 * t62 + t264 * t64) + t375 * t20) + t96 * t21) *
                    T(0.5)) +
               S1_1_2 * (((t354 * t63 + t165 * t65) + t273 * t9) + t97 * t19) * T(0.5)) +
              S1_2_2 * (((t355 * t63 + t264 * t65) + t375 * t9) + t96 * t19) * T(0.5)) +
             S2_1_1 * (((t23 * t133 + t77 * t141) + t95 * t18) + t262 * t26) * T(0.5)) +
            S2_1_2 * (((t85 * t133 + t49 * t141) + t94 * t18) + t13 * t26) * T(0.5)) +
           S2_1_2 * (((t23 * t140 + t77 * t136) + t95 * t27) + t262 * t17) * T(0.5)) +
          S2_2_2 * (((t85 * t140 + t49 * t136) + t94 * t27) + t13 * t17) * T(0.5)) +
         t88 * (S1_1_1 * t88 - S1_1_2 * t90)) -
        t90 * (S1_1_2 * t88 - S1_2_2 * t90)) +
       t89 * (S2_1_1 * t89 - S2_1_2 * t91)) -
      t91 * (S2_1_2 * t89 - S2_2_2 * t91));

  return gamma - T(3.0) * gamma_std;
}

// Function for aggregating two probabilistic planes into a joint probabilistic plane.
template <typename T>
const std::pair<const Eigen::Matrix<T, 4, 1>, const Eigen::Matrix<T, 3, 3>>
aggregateProbabilisticPlanes(const Eigen::Matrix<T, 4, 1>& plane_1_mean,
                             const Eigen::Matrix<T, 4, 1>& plane_2_mean,
                             const Eigen::Matrix<T, 3, 3>& plane_1_cov,
                             const Eigen::Matrix<T, 3, 3>& plane_2_cov) {

  const Eigen::Matrix<T, 3, 1> n1 = plane_1_mean.template head<3>();
  const Eigen::Matrix<T, 3, 3> n1_skew = skew(n1);
  const Eigen::Matrix<T, 3, 1> n2 = plane_2_mean.template head<3>();
  const Eigen::Matrix<T, 3, 2> B1 = s2_B(n1);
  const Eigen::Matrix<T, 3, 2> B2 = s2_B(n2);

  const Eigen::Matrix<T, 3, 1> from_1_to_2 =
      s2_BM(plane_2_mean, plane_1_mean, B1) * T(0.5);
  const Eigen::Matrix<T, 4, 1> plane_half(s2_BP(plane_1_mean, from_1_to_2, B1));
  const Eigen::Matrix<T, 3, 1> n0 = plane_half.template head<3>();
  const Eigen::Matrix<T, 3, 2> B0 = s2_B(n0);

  // Note: This part of the code is an iterated on-manifold kalman filter, that
  // could be iterated until a desired convergence is reached. However, in the paper
  // we demonstrated accurate results with a single iteration, and thus we remove the iteration from
  // the code to provide a simpler implementation.
  // Refer to https://github.com/hku-mars/IKFoM for more details on the filter implementation
  const Eigen::Matrix<T, 3, 1> prodAux = B1 * s2_BM(n0, n1, B1);
  const Eigen::Matrix<T, 3, 3> J =
      (Eigen::Matrix<T, 3, 3>() << -B0.transpose() * so3_Exp(prodAux) * n1_skew *
                                       n1_skew * s2_A(prodAux).transpose() * B1,
       Eigen::Matrix<T, 2, 1>::Zero(), 0, 0, 1)
          .finished();
  const Eigen::Matrix<T, 3, 3> S = J * plane_1_cov * J.transpose() + plane_2_cov;
  const Eigen::Matrix<T, 3, 3> K =
      S.transpose()
          .llt()
          .solve((J * plane_1_cov * J.transpose()).transpose())
          .transpose();
  const Eigen::Matrix<T, 3, 1> delta_q =
      -J * s2_BM(plane_half, plane_1_mean, B1) +
      K * (s2_BM(plane_2_mean, plane_half, B0) + J * s2_BM(plane_half, plane_1_mean, B1));
  // End of the iterated on-manifold kalman filter

  // Final calculations after the iterated on-manifold kalman filter has finished
  const Eigen::Matrix<T, 4, 1> plane_join = s2_BP(plane_half, delta_q, B0);
  const Eigen::Matrix<T, 3, 1> n_join = plane_join.template head<3>();
  const Eigen::Matrix<T, 2, 1> delta_n = delta_q.head(2);
  const Eigen::Matrix<T, 3, 2> B_join = s2_B(n_join);
  const Eigen::Matrix<T, 3, 3> L = s2_L(n_join, n0, delta_n, B_join, B0);
  const Eigen::Matrix<T, 3, 3> P =
      (Eigen::Matrix<T, 3, 3>::Identity() - K) * J * plane_1_cov * J.transpose();
  const Eigen::Matrix<T, 3, 3> cov_join = L * P * L.transpose();

  return std::make_pair(plane_join, cov_join);
}

// Function for aggregating a probabilistic plane and a set of probabilistic points
// into a joint probabilistic plane.
template <typename T>
const std::pair<const Eigen::Matrix<T, 4, 1>, const Eigen::Matrix<T, 3, 3>>
aggregateProbabilisticPlaneAndPoints(const Eigen::Matrix<T, 4, 1>& plane_mean,
                                     const Eigen::Matrix<T, 3, 3>& plane_cov,
                                     const PointVector<T>& points_means,
                                     const CovVector<T>& points_covs) {

  const Eigen::Matrix<T, 3, 1> n1 = plane_mean.template head<3>();
  const T d1 = plane_mean(3);
  const Eigen::Matrix<T, 3, 3> n1_skew = skew(n1);
  const Eigen::Matrix<T, 3, 2> B1 = s2_B(n1);

  // Note: This part of the code is an iterated on-manifold kalman filter, that
  // could be iterated until a desired convergence is reached. However, in the paper
  // we demonstrated accurate results with a single iteration, and thus we remove the iteration from
  // the code to provide a simpler implementation.
  // Refer to https://github.com/hku-mars/IKFoM for more details on the filter implementation

  const Eigen::Matrix<T, 3, 3> J =
      (Eigen::Matrix<T, 3, 3>() << -B1.transpose() * n1_skew * n1_skew * B1,
       Eigen::Matrix<T, 2, 1>::Zero(), 0, 0, 1)
          .finished();
  // Note: The calculation of  J is simplified since the initial value of prodAux is [0;0;0] because
  // the filter is initialized at plane_mean. For reference, the lines without the simplification are:
  //const Eigen::Matrix<T, 3, 1> prodAux = B1 * s2_BM(n1, n1, B1);  // [0;0;0]
  //const Eigen::Matrix<T, 3, 3> J =
  //    (Eigen::Matrix<T, 3, 3>() << -B1.transpose() * so3_Exp(prodAux) * n1_skew * n1_skew *
  //                                     s2_A(prodAux).transpose() * B1,
  //     Eigen::Matrix<T, 2, 1>::Zero(), 0, 0, 1)
  //        .finished();

  const size_t num_points_added = points_means.size();
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> R(num_points_added, num_points_added);
  R.setZero();
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> R_inv(num_points_added,
                                                         num_points_added);
  R_inv.setZero();
  Eigen::Matrix<T, Eigen::Dynamic, 1> r(num_points_added, 1);
  Eigen::Matrix<T, Eigen::Dynamic, 3> H(num_points_added, 3);
  for (int i = 0; i < num_points_added; i++) {
    r(i) = -n1.dot(points_means.at(i)) - d1;
    R(i, i) = n1.transpose() * points_covs.at(i) * n1;
    R_inv(i, i) = 1.0 / R(i, i);
    H.template block<1, 3>(i, 0) =
        (Eigen::Matrix<T, 1, 3>() << -points_means.at(i).transpose() * n1_skew * B1, T(1))
            .finished();
  }
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> S =
      H * J * plane_cov * J.transpose() * H.transpose() + R;
  const Eigen::Matrix<T, 3, Eigen::Dynamic> K =
      S.transpose()
          .llt()
          .solve((J * plane_cov * J.transpose() * H.transpose()).transpose())
          .transpose();

  const Eigen::Matrix<T, 3, 1> delta_q = K * r;
  // Note: Again, the calculation of delta_q is simplified since the filter is initialized at plane_mean.
  // For reference, the calculation without the simplification is:
  //const Eigen::Matrix<T, 3, 1> delta_q =
  //    -J * s2_BM(plane_mean, plane_mean, B1) +
  //    K * (r + J * s2_BM(plane_mean, plane_mean, B1));

  // End of the iterated on-manifold kalman filter

  // Final calculations after the iterated on-manifold kalman filter has finished
  const Eigen::Matrix<T, 4, 1> plane_join = s2_BP(plane_mean, delta_q, B1);
  const Eigen::Matrix<T, 3, 3> cov_join =
      (Eigen::Matrix<T, 3, 3>::Identity() - K * H) * J * plane_cov * J.transpose();

  return std::make_pair(plane_join, cov_join);
}

// Function for estimating the quadric of a probabilistic plane in a virtual reference frame
// where the plane is parallel to the z-axis.
template <typename T>
const Eigen::Matrix<T, 4, 4> estimatePlaneQuadricVirtual(const Eigen::Matrix<T, 3, 3> cov,
                                                         const T& d) {
  const T x = cov(0, 0);
  const T y = cov(1, 1);

  // Coefficients coming from a polynomial approximation of the MonteCarlo estimation
  // of the virtual quadric. The original data of the MonteCarlo estimation is given
  // at the folder "code/montecarlo_estimation".
  T p00, p10, p01, p20, p11, p02, p30, p21, p12, p03;
  T z_p00, z_p10, z_p01, z_p20, z_p11, z_p02, z_p30, z_p21, z_p12, z_p03;
  if (x < 0.3 && y < 0.3) {
    p00 = T(7.915e-05);
    p10 = T(-0.0009964);
    p01 = T(0.9969);
    p20 = T(0.00555);
    p11 = T(-0.3193);
    p02 = T(-0.9617);
    p30 = T(-0.01024);
    p21 = T(0.0971);
    p12 = T(0.1926);
    p03 = T(0.4793);
    z_p00 = T(0.9998);
    z_p10 = T(-0.9952);
    z_p01 = T(-0.996);
    z_p20 = T(0.9549);
    z_p11 = T(0.6336);
    z_p02 = T(0.9581);
    z_p30 = T(-0.4748);
    z_p21 = T(-0.2699);
    z_p12 = T(-0.2866);
    z_p03 = T(-0.4736);
  } else if (x < 1.0 && y < 1.0) {
    p00 = T(0.001512);
    p10 = T(-0.006493);
    p01 = T(0.9708);
    p20 = T(0.0127);
    p11 = T(-0.2753);
    p02 = T(-0.8402);
    p30 = T(-0.008678);
    p21 = T(0.06104);
    p12 = T(0.1205);
    p03 = T(0.314);
    z_p00 = T(0.9963);
    z_p10 = T(-0.96);
    z_p01 = T(-0.9593);
    z_p20 = T(0.8159);
    z_p11 = T(0.545);
    z_p02 = T(0.8142);
    z_p30 = T(-0.2954);
    z_p21 = T(-0.179);
    z_p12 = T(-0.1769);
    z_p03 = T(-0.2945);
  } else {
    p00 = T(0.2482);
    p10 = T(-0.1506);
    p01 = T(0.262);
    p20 = T(0.04473);
    p11 = T(-0.009113);
    p02 = T(-0.09304);
    p30 = T(-0.004832);
    p21 = T(-0.0002978);
    p12 = T(0.002155);
    p03 = T(0.01088);
    z_p00 = T(0.581);
    z_p10 = T(-0.1757);
    z_p01 = T(-0.1772);
    z_p20 = T(0.07033);
    z_p11 = T(0.03563);
    z_p02 = T(0.07116);
    z_p30 = T(-0.008574);
    z_p21 = T(-0.003728);
    z_p12 = T(-0.003708);
    z_p03 = T(-0.008717);
  }
  const T Q11 = p00 + p10 * y + p01 * x + p20 * y * y + p11 * y * x + p02 * x * x +
                p30 * y * y * y + p21 * y * y * x + p12 * y * x * x + p03 * x * x * x;
  const T Q22 = p00 + p10 * x + p01 * y + p20 * x * x + p11 * x * y + p02 * y * y +
                p30 * x * x * x + p21 * x * x * y + p12 * x * y * y + p03 * y * y * y;
  const T Q33 = z_p00 + z_p10 * x + z_p01 * y + z_p20 * x * x + z_p11 * x * y +
                z_p02 * y * y + z_p30 * x * x * x + z_p21 * x * x * y +
                z_p12 * x * y * y + z_p03 * y * y * y;

  // Coefficients coming from the 4-th order uncertainty propagation
  const T Q41 = cov(0, 2) * (T(1) - T(0.5) * (cov(0, 0) + cov(1, 1) / T(3)));
  const T Q42 = cov(1, 2) * (T(1) - T(0.5) * (cov(1, 1) + cov(0, 0) / T(3)));
  const T Q43 = d * (T(1) - T(0.5) * (cov(0, 0) + cov(1, 1)) +
                     (cov(0, 0) * cov(0, 0) + cov(1, 1) * cov(1, 1)) / T(8));

  // Coefficient coming from an exact solution of uncertainty propagation
  const T Q44 = d * d + cov(2, 2);

  return (Eigen::Matrix<T, 4, 4>() << Q11, 0, 0, Q41, 0, Q22, 0, Q42, 0, 0, Q33, Q43, Q41,
          Q42, Q43, Q44)
      .finished();
}

// Function for estimating the quadric of a probabilistic plane
template <typename T>
const Eigen::Matrix<T, 4, 4> esti_quadric(const Eigen::Matrix<T, 4, 1>& plane_mean,
                                          const Eigen::Matrix<T, 3, 3>& plane_cov) {
  const T alpha =
      T(0.5) * std::atan2(2.0 * plane_cov(0, 1), plane_cov(0, 0) - plane_cov(1, 1));
  const T cos_alpha = std::cos(alpha);
  const T sin_alpha = std::sin(alpha);
  const Eigen::Matrix<T, 3, 3> R_alpha =
      (Eigen::Matrix<T, 3, 3>() << cos_alpha, -sin_alpha, 0, sin_alpha, cos_alpha, 0, 0,
       0, T(1))
          .finished();
  const Eigen::Matrix<T, 3, 3> Sigma_virtual = R_alpha.transpose() * plane_cov * R_alpha;

  const Eigen::Matrix<T, 4, 4> Q_virtual =
      estimatePlaneQuadricVirtual(Sigma_virtual, plane_mean(3));

  const Eigen::Matrix<T, 4, 4> R_alpha_4 =
      (Eigen::Matrix<T, 4, 4>() << cos_alpha, -sin_alpha, 0, 0, sin_alpha, cos_alpha, 0,
       0, 0, 0, T(1), 0, 0, 0, 0, T(1))
          .finished();

  const Eigen::Matrix<T, 3, 1> plane_mean_normal = plane_mean.template head<3>();
  const static Eigen::Matrix<T, 3, 3> s2_R1z_transposed =
      (Eigen::Matrix<T, 3, 3>() << 0, 0, T(1), 0, T(1), 0, -T(1), 0, 0).finished();
  const Eigen::Matrix<T, 4, 4> R_virtual =
      (Eigen::Matrix<T, 4, 4>() << s2_R1(plane_mean_normal) * s2_R1z_transposed,
       Eigen::Matrix<T, 3, 1>::Zero(), 0, 0, 0, 1)
          .finished();

  return R_virtual * R_alpha_4 * Q_virtual * R_alpha_4.transpose() *
         R_virtual.transpose();
}