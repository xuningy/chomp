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
namespace print = print_utils;

namespace planner {

void CHOMP::initialize(ros::NodeHandle& nh)
{
  pu::get("chomp/w_smooth", w_smooth_, 1.0);
  pu::get("chomp/w_obs", w_obs_, 100.0);

  pu::get("chomp/epsilon", epsilon_, 0.3);
  pu::get("chomp/eta", eta_, 400.0);
  pu::get("chomp/min_cost_improv_frac", min_cost_improv_frac_, 1e-5);
  pu::get("chomp/decrease_wt", decrease_wt_, true);
  pu::get("chomp/progress", progress_, 0.0);

  pu::get("chomp/stopping_conditions/max_iter", max_iter_, 100);
  pu::get("chomp/stopping_conditions/min_iter", min_iter_, 10);
  pu::get("chomp/stopping_conditions/max_time", max_time_, 10.0);

  map_initialized_ = false;
}

void CHOMP::initializeCostMap(ros::NodeHandle& nh, std::shared_ptr<SDFMap> sdf_map)
{
  cost_map_ = std::make_shared<CostMap>();
  cost_map_->initialize(nh, sdf_map, epsilon_);
  // cost_map_->initialize(nh, epsilon_); // for testing 2DFixedCostMap

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

  // set costmap height
  cost_map_->setCostMapSliceHeight(initial_path(0, 2));

  // setup smoothness matrices
  setupAK(N_);
  setupBCE(initial_path.row(0), initial_path.row(N_-1));

  print::print(initial_path, "initial_path");

  // Vector \xi does not contain the initial and end points
  EigenMatrixX3d xi(N_-2, 3);
  xi = initial_path.block(1, 0, N_-2, 3);

  // remove last point in path
  EigenMatrixX3d path_notail(N_-1, 3);
  path_notail = initial_path.topRows(N_-1);

  // Compute time derivative of xi
  EigenMatrixX3d xi_d(N_-2, 3);
  xi_d = approximateDifferentiation(path_notail); // x'

  // compute the initial cost

  Eigen::VectorXd cost_wpts;
  cost_map_->getPathCost(xi, cost_wpts);
  print::print(cost_wpts, "cost_wpts");

  double cost = w_obs_ * costObstacle(xi, xi_d) + w_smooth_ * costSmoothness(xi);

  std::cout << "initial cost: " << cost << std::endl;

  if (cost < std::numeric_limits<double>::min())
  {
    ROS_INFO("[CHOMP] initial trajectory has ZERO cost, safe! Exiting.");
    final_path = initial_path;
    return true;
  }
  std::vector<double> cost_history;
  cost_history.push_back(cost);

  EigenMatrixX3d grad = w_obs_ * gradCostObstacle(xi, xi_d) + w_smooth_ * gradCostSmoothness(xi);

  // Compute cost_gradient term
  EigenMatrixX3d grad_cost;
  cost_map_->getPathGradient(xi, grad_cost);
  print::print(grad_cost, "initial path gradient");

  EigenMatrixX3d grad_obs = gradCostObstacle(xi, xi_d);
  print::print(grad_obs, "initial cost obstacle");

  // store intermediate paths
  all_paths_.clear();
  auto path_i = initial_path;
  all_paths_.push_back(path_i);

  ROS_INFO("[CHOMP::covariantGradientDescent] starting gradient descent...");

  for (int i = 0; i < max_iter_; i++)
  {
    std::cout << "[Iter " << i << std::flush;
    double step_size = 1/eta_;
    if (decrease_wt_ == true)
    {
      step_size = 1/eta_ * (1/std::sqrt(i+1+progress_));
    }

    // Gradient descent step
    xi = xi - step_size * A_inv_ * grad;

    EigenMatrixX3d step = step_size * A_inv_ * grad;
    Eigen::Vector3d step_avg = step.colwise().norm();
    std::cout << "] step_size: " << step_size << " ... average step: [" << step_avg(0) << ", " << step_avg(1) << ", " << step_avg(2) << std::flush;

    Eigen::Vector3d grad_avg = grad.colwise().norm();
    std::cout << "] ...average grad: [" << grad_avg(0) << ", " << grad_avg(1) << ", " << grad_avg(2) << std::flush;

    // Compute cost and gradient.
    cost = w_obs_ * costObstacle(xi, xi_d) + w_smooth_ * costSmoothness(xi);
    grad = w_obs_ * gradCostObstacle(xi, xi_d) + w_smooth_ * gradCostSmoothness(xi);

    // Update cost
    std::cout <<"] ... cost: " << cost << std::flush;
    cost_history.push_back(cost);

    // store intermediate path.
    path_i.block(1, 0, N_-2, 3) = xi; // middle block is the updated xi.
    all_paths_.push_back(path_i);

    // Check if we can terminate
    if (i >= min_iter_)
    {
      std::cout << " ... Cost improvement: " << std::abs(cost_history[i+1] - cost_history[i])/cost_history[i] << std::flush;
      if (std::abs(cost_history[i+1] - cost_history[i])/cost_history[i] < min_cost_improv_frac_) break;
    }
    std::cout << std::endl;

  }

  // construct final path
  final_path = initial_path;
  final_path.block(1, 0, N_-2, 3) = xi; // middle block is the updated xi.

  ROS_INFO("[CHOMP::covariantGradientDescent] gd success. Path found with cost %.3f", cost);

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
  // print::print(cost_wpts, "cost_wpts");

  // compute approximate time diffs
  EigenMatrixX3d dxdt_N = xi_d.array() * (N+1);
  Eigen::VectorXd dxdt_norm = dxdt_N.rowwise().norm();

  double cost = (dxdt_norm.array()*cost_wpts.array()*1/(N+1)).matrix().sum();

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
  // print::print(hat_dxdt, "hat_dxdt");
  // print::print(dxdt_squarednorm, "dxdt_squarednorm");

  // Second time derivatives

  EigenMatrixX3d ddxddt(N, 3);
  ddxddt.row(0) = Eigen::Vector3d(0, 0, 0);
  ddxddt.bottomRows(N-1) = approximateDifferentiation(xi_d);
  EigenMatrixX3d ddxddt_N = ddxddt.array() * (N+1);
  // print::print(ddxddt, "ddxddt");

  Eigen::VectorXd cost;
  cost_map_->getPathCost(xi, cost);

  // Compute cost_gradient term
  EigenMatrixX3d grad_cost;
  cost_map_->getPathGradient(xi, grad_cost);

  // Compute grad F_obs piece by piece. The for-loops is because when i tried
  // to do this with matrix operations it wouldn't compile even though stack
  // overflow said otherwise. Leaving it for now til I find a fix.

  // kappa_unnormalized = (I − x̂'x̂')x''
  auto multiplied0 = hat_dxdt.array()*ddxddt_N.array();
  // print::print(multiplied0, "multiplied0");
  auto summed0 = multiplied0.rowwise().sum();
  // print::print(summed0, "summed0");
  Eigen::MatrixX3d repmatted0(N, 3);
  repmatted0.col(0) = summed0;
  repmatted0.col(1) = summed0;
  repmatted0.col(2) = summed0;
  // print::print(repmatted0, "repmatted0");
  Eigen::MatrixX3d k_unnorm_sceond_part = hat_dxdt.array()*repmatted0.array();
  // print::print(k_unnorm_sceond_part, "k_unnorm_sceond_part");

  EigenMatrixX3d kappa_unnormalized(N, 3);
  // kappa_unnormalized = (ddxddt_N - hat_dxdt * hat_dxdt.transpose() * ddxddt_N);
  kappa_unnormalized = (ddxddt_N - k_unnorm_sceond_part);
  // print::print(kappa_unnormalized, "kappa_unnormalized");

  // Compute curvature kappa = 1/||x'||^2 * kappa_unnormalized
  EigenMatrixX3d kappa(N, 3);
  for (int i = 0; i < N; i++)
  {
    kappa.row(i) = kappa_unnormalized.row(i).array() / dxdt_squarednorm(i);
  }
  // kappa = kappa.array().rowwise() / dxdt_squarednorm.transpose().array();
  // print::print(kappa, "kappa");

  // Compute firstterm = (I − x̂'x̂')grad_c
  auto multiplied = hat_dxdt.array()*grad_cost.array();
  // print::print(multiplied, "multiplied");
  auto summed = multiplied.rowwise().sum();
  // print::print(summed, "summed");
  Eigen::MatrixX3d repmatted(N, 3);
  repmatted.col(0) = summed;
  repmatted.col(1) = summed;
  repmatted.col(2) = summed;
  // print::print(repmatted, "repmatted");
  Eigen::MatrixX3d firstterm_secondpart = hat_dxdt.array()*repmatted.array();
  // print::print(firstterm_secondpart, "firstterm_secondpart");

  // auto firstterm = grad_cost - hat_dxdt * hat_dxdt.transpose() * grad_cost;
  auto firstterm = grad_cost - firstterm_secondpart;
  // print::print(firstterm, "firstterm");

  // Compute ckappa = cost * kappa
  EigenMatrixX3d ckappa(N, 3);
  for (int i = 0; i < N; i++)
  {
    ckappa.row(i) = cost(i) * kappa_unnormalized.row(i);
  }

  // Compute grad = ||x'|| * (firstterm + cost*kappa)
  EigenMatrixX3d grad(N, 3);
  for (int i = 0; i < N; i++)
  {
    grad.row(i) = (firstterm.row(i) + ckappa.row(i)) * dxdt_norm(i) * 1/(N+1);
  }
  // auto grad = (firstterm + cost_wpts.array() * kappa.array().rowwise()).array().rowwise() * dxdt_norm.array();

  // print::print(grad, "full gradient");
  return grad;
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
