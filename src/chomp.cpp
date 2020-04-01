/*
CHOMP: Covariant Hamiltonian Optimization for Motion Planning
Copyright (C) 2020 Xuning Yang

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <limits>
#include <chomp/Chomp.h>

namespace pu = parameter_utils;

namespace planner {

void CHOMP::initialize(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private)
{
  pu::get("chomp/w_smooth", w_smooth_, 1.0);
  pu::get("chomp/w_obs", w_obs_, 100.0);
  pu::get("chomp/w_close", w_close_, 0.0);
  pu::get("chomp/w_close_end", w_close_end_, 0.0);
  pu::get("chomp/w_close_linear_taper", w_close_linear_taper_, false);

  pu::get("chomp/epsilon", epsilon_, 0.3);
  pu::get("chomp/eta", eta_, 400.0);
  pu::get("chomp/cost_improvement_cutoff", cost_improvement_cutoff_, 1e-5);
  pu::get("chomp/decrease_wt", decrease_wt_, true);
  pu::get("chomp/progress", progress_, 0.0);

  pu::get("chomp/stopping_conditions/max_iter", max_iter_, 100);
  pu::get("chomp/stopping_conditions/max_time", max_time_, 10.0);

  pu::get("chomp/verbose", verbose_, false);

  // print parameters

  std::cout << "=================================================" << std::endl;
  std::cout << "       CHOMP Parameters " << std::endl;
  std::cout << "=================================================" << std::endl;
  std::cout << "Weights:" << std::endl;
  std::cout << "\tSmoothness\t\t" << w_smooth_ << std::endl;
  std::cout << "\tObstacles\t\t" << w_obs_ << std::endl;
  std::cout << "\tCloseness to orig. path\t" << w_close_ << std::endl;
  if (w_close_linear_taper_)
  {
    std::cout << "\t   Linear taper enabled; wt scaling along path to " << w_close_end_ << std::endl;
  }

  // Step size parameters
  std::cout << "---" << std::endl;
  std::cout << "epsilon (distance away from the obstacle that would incur zero cost): " << epsilon_ << std::endl;
  std::cout << "eta: \t\t\t\t" << eta_ << std::endl;
  std::cout << "step_size (1/eta): \t\t" << 1/eta_ << std::endl;
  if (decrease_wt_)
  {
    std::cout << "decrease weight every iteration: ON\n\tProgress:\t\t" << progress_ << std::endl;
  } else
  {
    std::cout << "decrease weight every iteration: OFF" << std::endl;
  }

  // Cutoff parameters
  std::cout << "---" << std::endl;
  std::cout << "cost improvement cutoff: \t" << cost_improvement_cutoff_ << std::endl;
  std::cout << "max iter: \t\t\t" << max_iter_ << std::endl;
  std::cout << "=================================================" << std::endl;


  cost_map_ = std::make_shared<CostMap>(nh, nh_private, epsilon_);

  map_initialized_ = true;
}

void CHOMP::setupAK(int N)
{
  // K needs to be constructed with this N.
  N = N-2;

  K_.setZero(N+1, N);

  // K = [ 1, 0, 0, ...
  //      -1, 1, 0, ...
  //      0, -1, 1, ...
  for (int i = 0; i < N-1; i++)
  {
    K_(i+1, i) = -1;
    K_(i+1, i+1) = 1;
  }
  K_(0, 0) = 1;
  K_(N, N-1) = -1;

  A_ = K_.transpose() * K_;

  A_inv_ = A_.inverse();

  return;
}

void CHOMP::setupBCE(const Eigen::Vector3d& start, const Eigen::Vector3d& goal)
{
  e_.setZero(N_-1, 3);
  e_.row(0) = -start;
  e_.row(N_-2) = goal;

  b_ = K_.transpose()*e_;
  c_ = 0.5 * (e_.transpose() * e_);

  return;
}

bool CHOMP::covariantGradientDescent(const CHOMP::EigenMatrixX3d& initial_path, CHOMP::EigenMatrixX3d& final_path)
{
  N_ = initial_path.rows();

  if (!map_initialized_)
  {
    ROS_ERROR("[CHOMP::covariantGradientDescent] cost_map not available! please initialize cost_map with an esdf.");
    return false;
  }

  // setup smoothness matrices
  setupAK(N_);
  setupBCE(initial_path.row(0), initial_path.row(N_-1));

  // Vector \xi does not contain the initial and end points
  EigenMatrixX3d xi(N_-2, 3);
  xi = initial_path.block(1, 0, N_-2, 3);

  // remove last point in path
  EigenMatrixX3d path_notail(N_-1, 3);
  path_notail = initial_path.topRows(N_-1);

  // Compute time derivative of xi
  EigenMatrixX3d xi_d(N_-2, 3);
  xi_d = approximateDifferentiation(path_notail); // x'

  // set the guide path
  guide_ = xi;

  // setup linear taper function
  w_close_vec_.resize(N_-2);
  if (w_close_linear_taper_) {
    w_close_vec_.setLinSpaced(N_-2, w_close_, w_close_end_);
  } else {
    w_close_vec_.setConstant(w_close_);
  }

  // compute the initial cost. If safe, return.
  double cost = costFunction(xi, xi_d);

  if (cost < std::numeric_limits<double>::min())
  {
    ROS_INFO("[CHOMP] initial trajectory has ZERO cost, safe! Exiting.");
    final_path = initial_path;
    return true;
  }

  // Else, continue.
  std::vector<double> cost_history;
  cost_history.push_back(cost);

  EigenMatrixX3d grad = gradientFunction(xi, xi_d);

  // store intermediate paths
  all_paths_.clear();
  auto path_i = initial_path;
  all_paths_.push_back(path_i);

  ROS_INFO("[CHOMP::covariantGradientDescent] starting gradient descent with initial cost %.3f...", cost);
  double step_size = 1/eta_;

  for (int i = 0; i < max_iter_; i++)
  {
    if (decrease_wt_ == true)
    {
      step_size = 1/eta_ * (1/std::sqrt(i+1+progress_));
    }

    // Gradient descent step
    EigenMatrixX3d step = step_size * A_inv_ * grad;
    xi = xi - step;

    // Compute cost and gradient on new trajectory.
    cost = costFunction(xi, xi_d);
    grad = gradientFunction(xi, xi_d);

    // Update cost
    cost_history.push_back(cost);

    // store intermediate path.
    path_i.block(1, 0, N_-2, 3) = xi; // middle block is the updated xi.
    all_paths_.push_back(path_i);

    // Check if we can terminate
    double cost_improvement = std::abs(cost_history[i+1] - cost_history[i])/cost_history[i];

    if (verbose_) {
      Eigen::Vector3d step_avg = step.colwise().norm();
      Eigen::Vector3d grad_avg = grad.colwise().norm();

      std::cout << "[Iter " << i << "] step_size: " << step_size << "   average step: [" << step_avg(0) << ", " << step_avg(1) << ", " << step_avg(2) << "]  average grad: [" << grad_avg(0) << ", " << grad_avg(1) << ", " << grad_avg(2) << "]   cost: " << cost << "   cost improvement: " << cost_improvement << std::endl;
    }

    if (cost_improvement < cost_improvement_cutoff_) {
      ROS_INFO("[CHOMP::covariantGradientDescent] Success. Path found after %d iterations with cost %.3f", i, cost);

      // construct final path
      final_path = initial_path;
      final_path.block(1, 0, N_-2, 3) = xi; // middle block is the updated xi.

      return true;
    }

  }

  // construct final path
  final_path = initial_path;
  final_path.block(1, 0, N_-2, 3) = xi; // middle block is the updated xi.

  ROS_INFO("[CHOMP::covariantGradientDescent] Exceeded max iteration, terminating with path with cost %.3f", cost);

  return true;


}

double CHOMP::costSmoothness(const EigenMatrixX3d& xi)
{
  int N = xi.rows();

  Eigen::Matrix3d cost_matrix = 0.5*xi.transpose() * A_ * xi + xi.transpose() * b_ + c_;

  double cost = cost_matrix.trace() * (N+1);

  return cost;
}

double CHOMP::costObstacle(const EigenMatrixX3d& xi, const EigenMatrixX3d& xi_d)
{
  int N = xi.rows();

  Eigen::VectorXd cost_wpts;
  cost_map_->getPathCost(xi, cost_wpts);

  // compute approximate time diffs
  EigenMatrixX3d dxdt_N = xi_d.array() * (N+1);
  Eigen::VectorXd dxdt_norm = dxdt_N.rowwise().norm();

  double cost = (dxdt_norm.array()*cost_wpts.array()*1/(N+1)).matrix().sum();

  return cost;
}

double CHOMP::costGuidePath(const EigenMatrixX3d& xi, const EigenMatrixX3d& guide)
{
  auto diff = (xi - guide).rowwise().squaredNorm().array()* w_close_vec_.array();
  double cost =  diff.sum();

  return cost;
}


CHOMP::EigenMatrixX3d CHOMP::gradCostSmoothness(const CHOMP::EigenMatrixX3d& xi)
{
  int N = xi.rows();

  EigenMatrixX3d grad = (N+1) * (A_* xi + b_).array();

  return grad;
}

CHOMP::EigenMatrixX3d CHOMP::gradCostObstacle(const CHOMP::EigenMatrixX3d& xi, const EigenMatrixX3d& xi_d)
{
  int N = xi.rows();

  // First time derivatives and their norms
  EigenMatrixX3d dxdt_N = xi_d.array() * (N+1);
  EigenMatrixX3d hat_dxdt = dxdt_N.rowwise().normalized(); //hat x' = x'/||x'||
  Eigen::VectorXd dxdt_squarednorm = dxdt_N.rowwise().squaredNorm(); //||x'||^2
  Eigen::VectorXd dxdt_norm = dxdt_N.rowwise().norm(); //||x'||

  // Second time derivatives
  EigenMatrixX3d ddxddt(N, 3);
  ddxddt.row(0) = Eigen::Vector3d(0, 0, 0);
  ddxddt.bottomRows(N-1) = approximateDifferentiation(xi_d);
  EigenMatrixX3d ddxddt_N = ddxddt.array() * (N+1);

  Eigen::VectorXd cost;
  cost_map_->getPathCost(xi, cost);

  // Compute cost_gradient term
  EigenMatrixX3d grad_cost;
  cost_map_->getPathGradient(xi, grad_cost);

  // Compute grad F_obs piece by piece.

  // kappa_unnormalized = (I − x̂'x̂')x''
  auto temp = (hat_dxdt.array()*ddxddt_N.array()).rowwise().sum();
  Eigen::MatrixX3d dxdt_ddxddt(N, 3);
  dxdt_ddxddt.col(0) = temp;
  dxdt_ddxddt.col(1) = temp;
  dxdt_ddxddt.col(2) = temp;
  Eigen::MatrixX3d k_unnorm_second_part = hat_dxdt.array()*dxdt_ddxddt.array();

  EigenMatrixX3d kappa_unnormalized(N, 3);
  kappa_unnormalized = (ddxddt_N - k_unnorm_second_part);

  // Compute curvature kappa = 1/||x'||^2 * kappa_unnormalized
  EigenMatrixX3d kappa = kappa_unnormalized.cwiseQuotient(dxdt_squarednorm.replicate<1, 3>());

  // Compute firstterm = (I − x̂'x̂')grad_c
  auto temp2 = (hat_dxdt.array()*grad_cost.array()).rowwise().sum();
  Eigen::MatrixX3d dxdt_gradcost(N, 3);
  dxdt_gradcost.col(0) = temp2;
  dxdt_gradcost.col(1) = temp2;
  dxdt_gradcost.col(2) = temp2;
  Eigen::MatrixX3d firstterm_secondpart = hat_dxdt.array()*dxdt_gradcost.array();

  auto firstterm = grad_cost - firstterm_secondpart;

  // Compute grad = ||x'|| * (firstterm + cost*kappa)
  EigenMatrixX3d grad = (firstterm - kappa.cwiseProduct(cost.replicate<1, 3>())).cwiseProduct(dxdt_norm.replicate<1, 3>()) * 1/(N+1);

  return grad;
}


CHOMP::EigenMatrixX3d CHOMP::gradCostGuidePath(const EigenMatrixX3d& xi, const EigenMatrixX3d& guide)
{

  EigenMatrixX3d grad = 2 * (xi - guide);

  EigenMatrixX3d grad_weighted = grad.cwiseProduct(w_close_vec_.replicate<1, 3>());

  return grad_weighted;
}

CHOMP::EigenMatrixX3d CHOMP::approximateDifferentiation(const CHOMP::EigenMatrixX3d& path)
{
  // Takes in a vector of N, and outputs the time difference in a vector of N-1
  // If X is a vector of length m, then Y = diff(X) returns a vector of length m-1. The elements of Y are the differences between adjacent elements of X.
  // Y = [X(2)-X(1) X(3)-X(2) ... X(m)-X(m-1)]

  int N = path.rows();

  EigenMatrixX3d M1(N-1, 3); // the first N cols of this matrix
  EigenMatrixX3d M2(N-1, 3); // shifted

  M1 = path.topRows(N-1);

  M2 = path.bottomRows(N-1);

  return M2 - M1;
}

visualization_msgs::Marker CHOMP::visualizePath(const CHOMP::EigenMatrixX3d& path, const std_msgs::ColorRGBA& color, int i)
{
  visualization_msgs::Marker marker;

  marker.header.stamp = ros::Time::now();
  marker.id = i;
  marker.type = visualization_msgs::Marker::SPHERE_LIST;
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.04;
  marker.scale.y = 0.04;
  marker.scale.z = 0.04;
  marker.color = color;

  marker.header.frame_id = "world";

  for (int i = 0; i < path.rows(); i++)
  {
    geometry_msgs::Point msg;
    msg.x = path(i,0);
    msg.y = path(i,1);
    msg.z = path(i,2);
    marker.points.push_back(msg);
  }

 return marker;
}



}  // namespace planner
