/*
cost_map: obstacle cost map for gradient based planners (chomp)
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
#include <geometry_msgs/PoseStamped.h>

#include <plan_env/sdf_map.h>

namespace planner {

/* This package implements a cost map based on the cost function given in CHOMP:
*
*  c(x) = { -D(x) + 0.5e           , if D(x) < 0
*            1/(2e)*(D(x) - e)^2   , if 0 < D(x) <= e
*            0                     , otherwise          }
*
* See Section 4.2 of CHOMP: Covariant Hamiltonian Optimization for Motion
* Planning for details.
* The package depends on an ESDF map. I am using the HKUST implementation in
* package hkust-kinodynamic/plan_env.
*/

class CostMap
{
public:
  CostMap() {}
  ~CostMap() {}

  SDFMap::Ptr sdf_map_;

  void initialize(ros::NodeHandle& nh, SDFMap::Ptr map, double epsilon);

  bool getPathCost(const Eigen::Matrix<double, Eigen::Dynamic, 3>& path, Eigen::VectorXd& cost);

  bool getPathGradient(const Eigen::Matrix<double, Eigen::Dynamic, 3>& path, Eigen::Matrix<double, Eigen::Dynamic, 3>& grad_along_path);

  bool getPathDist(const Eigen::Matrix<double, Eigen::Dynamic, 3>& path, Eigen::VectorXd& dist);

  void setCostMapSliceHeight(double height);
  
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  // Parameters
  double resolution_;
  double resolution_inv_;
  double epsilon_; // Distance away from the obstacle that has zero cost
  double epsilon_times_half_;
  double half_divided_by_epsilon_;
  double cost_map_slice_height_;

  // cost computation functions
  double cost(double dist);

  double getCostOfPos(const Eigen::Vector3d& pos);
  double getCostOfIdx(const Eigen::Vector3i& id);

  bool gradient(const Eigen::Vector3d& x, Eigen::Vector3d& grad);

  // Map visualization functions
  void publishCostMapCallback(const ros::TimerEvent&);
  ros::Timer cost_map_timer_;
  ros::Publisher cost_map_pub_;

};

}  // namespace planner
