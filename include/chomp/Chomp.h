/*
CHOMP: Covariant Hamiltonian Optimization for Motion Planning
Copyright (C) 2020 Xuning Yang

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#pragma once

#include <iostream>

#include <Eigen/Eigen>
#include <ros/ros.h>
#include <parameter_utils/ParameterUtils.h>
#include <visualization_msgs/MarkerArray.h>

#include <cost_map/CostMap.h>


namespace planner {

/* This package implements CHOMP: Covariant Hamiltonian Optimization for Motion
* Planning
* The package depends on an ESDF map. I am using the HKUST implementation in
* package hkust-kinodynamic/plan_env.
*/

class CHOMP
{
public:
  CHOMP() {}
  ~CHOMP() {}

  typedef Eigen::Matrix<double, Eigen::Dynamic, 3> EigenMatrixX3d;

  std::shared_ptr<CostMap> cost_map_;

  void initialize(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private);
  inline double costFunction(const EigenMatrixX3d& xi, const EigenMatrixX3d& xi_d);

  inline EigenMatrixX3d gradientFunction(const EigenMatrixX3d& xi, const EigenMatrixX3d& xi_d);

  bool covariantGradientDescent(const EigenMatrixX3d& initial_path, EigenMatrixX3d& final_path);

  inline void getIntermediatePaths(std::vector<CHOMP::EigenMatrixX3d>& all_paths);

  visualization_msgs::Marker visualizePath(const CHOMP::EigenMatrixX3d& path, const std_msgs::ColorRGBA& color, int i = 0);

  bool checkPathSafe(const EigenMatrixX3d& path);
  bool checkPointSafe(const Eigen::Vector3d& point);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:

  int N_;         // Number of waypoints (including initial and final positions)
  Eigen::MatrixXd A_;
  Eigen::MatrixXd A_inv_;     // inverse of A
  Eigen::MatrixXd b_;
  Eigen::MatrixXd c_;
  Eigen::MatrixXd K_;        // Finite diferencing matrix
  Eigen::MatrixXd e_;

  EigenMatrixX3d guide_;    // guide path.

  double w_smooth_;
  double w_obs_;
  double w_close_;
  double w_close_end_;
  bool w_close_linear_taper_;
  Eigen::VectorXd w_close_vec_;

  double epsilon_;  // distance away from the obstacle that should incur zero cost in the cost map.
  double eta_;
  double cost_improvement_cutoff_;
  bool decrease_wt_;
  double progress_; // decrease stepsize weight.
  int max_iter_;
  double max_time_;

  double vehicle_radius_;

  bool map_initialized_;

  std::vector<CHOMP::EigenMatrixX3d> all_paths_;

  bool verbose_;

  void setupAK(int N);
  void setupBCE(const Eigen::Vector3d& start, const Eigen::Vector3d& goal);

  // Cost functions
  double costSmoothness(const EigenMatrixX3d& xi);
  EigenMatrixX3d gradCostSmoothness(const EigenMatrixX3d& xi);

  double costObstacle(const EigenMatrixX3d& xi, const EigenMatrixX3d& xi_d);
  EigenMatrixX3d gradCostObstacle(const EigenMatrixX3d& xi, const EigenMatrixX3d& xi_d);

  double costGuidePath(const EigenMatrixX3d& xi, const EigenMatrixX3d& guide);
  EigenMatrixX3d gradCostGuidePath(const EigenMatrixX3d& xi, const EigenMatrixX3d& guide);

  EigenMatrixX3d approximateDifferentiation(const EigenMatrixX3d& path);

};

// get intermediate paths (including initial and final)
inline void CHOMP::getIntermediatePaths(std::vector<CHOMP::EigenMatrixX3d>& all_paths)
{
  all_paths = all_paths_;
}

inline double CHOMP::costFunction(const EigenMatrixX3d& xi, const EigenMatrixX3d& xi_d)
{
  // Base cost functions are just smoothness and obstacles.
  double cost = w_obs_ * costObstacle(xi, xi_d)
               + w_smooth_ * costSmoothness(xi);

  // Additional cost functions
  if (w_close_ > 0.0 || w_close_end_ > 0.0) cost += costGuidePath(xi, guide_);

  return cost;
}

inline CHOMP::EigenMatrixX3d CHOMP::gradientFunction(const EigenMatrixX3d& xi, const EigenMatrixX3d& xi_d)
{
  EigenMatrixX3d grad = w_obs_ * gradCostObstacle(xi, xi_d)
                      + w_smooth_ * gradCostSmoothness(xi);

  // Additional cost functions
  if (w_close_ > 0.0 || w_close_end_ > 0.0) grad += gradCostGuidePath(xi, guide_);

  return grad;
}

}  // namespace planner
