/*
cost_map: obstacle cost map for gradient based planners (chomp)
Copyright (C) 2020 Xuning Yang

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include <cost_map/CostMap.h>

namespace planner {

CostMap::CostMap(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private, double epsilon)
: nh_(nh),
  nh_private_(nh_private),
  voxblox_server_(nh_, nh_private_)
{
  resolution_ = (double)voxblox_server_.getEsdfMapPtr()->voxel_size();
  resolution_inv_ = 1.0/resolution_;
  epsilon_ = epsilon;
  epsilon_times_half_ = 0.5*epsilon;
  half_divided_by_epsilon_ = 0.5/epsilon;

  // setup map
  double robot_radius = 1.0; //TODO
  voxblox_server_.setTraversabilityRadius(robot_radius);
  voxblox_server_.publishTraversable();

  cost_map_slice_height_ = voxblox_server_.getSliceLevel();

  // visualization parameters
  visualize_all_ = false;
  visualize_slice_ = true;

  if (visualize_all_ || visualize_slice_)
  {
    ros::NodeHandle n(nh);
    cost_map_pub_ = n.advertise<sensor_msgs::PointCloud2>("/cost_map", 10);
    cost_map_timer_ = n.createTimer(ros::Duration(0.05), &CostMap::publishCostMapCallback, this);
  }

  ROS_INFO("[VoxbloxCostMap] Voxblox Cost Map initialized with resolution %.1f. cost map slice height at %.1f", resolution_, cost_map_slice_height_);

}

double CostMap::getMapDistance(const Eigen::Vector3d& position) const {
  if (!voxblox_server_.getEsdfMapPtr()) {
    ROS_WARN("[CostMap::getMapDistance] voxblox esdf map pointer is NULL!");
    return 0.0;
  }

  if (!voxblox_server_.getEsdfMapPtr()->isObserved(position))
  {
    // ROS_WARN("[CostMap::getMapDistance] point (%.2f, %.2f, %.2f) is NOT OBSERVED!", position(0), position(1), position(2));
    return 0.0;
  }

  double distance = 0.0;
  if (!voxblox_server_.getEsdfMapPtr()->getDistanceAtPosition(position,
                                                              &distance)) {
    // ROS_WARN("[CostMap::getMapDistance] point (%.2f, %.2f, %.2f) not in map, distance set to zero!", position(0), position(1), position(2));

    return 0.0;
  }

  return distance;
}


void CostMap::setCostMapSliceHeight(double height)
{
  cost_map_slice_height_ = height;
  std::cout << "[CostMap] setting cost map height to " << height << std::endl;
}

// c(x)
double CostMap::cost(double dist)
{
  // c(x) = { -D(x) + 0.5e           , if D(x) < 0
  //           1/(2e)*(D(x) - e)^2   , if 0 < D(x) <= e
  //           0                     , otherwise          }
  if (dist < 0.0)
    return -dist + epsilon_times_half_;
  else if (dist <= epsilon_)
    return half_divided_by_epsilon_ * (dist - epsilon_) * (dist - epsilon_);
  else
    return 0.0;
}

double CostMap::getCostOfPos(const Eigen::Vector3d& pos) {
  double dist = getMapDistance(pos);
  return cost(dist);
}

bool CostMap::gradient(const Eigen::Vector3d& pos, Eigen::Vector3d& grad)
{

  /* use trilinear interpolation */

  // Assuming uniform resolution
  double delta = resolution_;
  double xd = delta;
  double yd = delta;
  double zd = delta;
  Eigen::Vector3d diff(xd, yd, zd);
  Eigen::Vector3d step = diff*2;

  Eigen::Vector3d c000 = pos - diff;

  // Find the corners of the cube that surrounds the point pos.
  double values[2][2][2];
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        Eigen::Vector3d cijk = c000 + (Eigen::Array3d(i,j,k)*step.array()).matrix();

        values[i][j][k] = getCostOfPos(cijk);
      }
    }
  }

  // Linearly interpolate to find values v1, v0, v00, v01, v10, v11
  double v00 = (1 - xd) * values[0][0][0] + xd * values[1][0][0];
  double v01 = (1 - xd) * values[0][0][1] + xd * values[1][0][1];
  double v10 = (1 - xd) * values[0][1][0] + xd * values[1][1][0];
  double v11 = (1 - xd) * values[0][1][1] + xd * values[1][1][1];
  double v0 = (1 - yd) * v00 + yd * v10;
  double v1 = (1 - yd) * v01 + yd * v11;
  double dist = (1 - zd) * v0 + zd * v1;

  grad[2] = (v1 - v0) * resolution_inv_;
  grad[1] = ((1 - zd) * (v10 - v00) + zd * (v11 - v01)) * resolution_inv_;
  grad[0] = (1 - zd) * (1 - yd) * (values[1][0][0] - values[0][0][0]);
  grad[0] += (1 - zd) * yd * (values[1][1][0] - values[0][1][0]);
  grad[0] += zd * (1 - yd) * (values[1][0][1] - values[0][0][1]);
  grad[0] += zd * yd * (values[1][1][1] - values[0][1][1]);
  grad[0] *= resolution_inv_;

  return true;
}

bool CostMap::getPathDist(const Eigen::Matrix<double, Eigen::Dynamic, 3>& path, Eigen::VectorXd& dist)
{
  dist.resize(path.rows());
  //
  // if (!voxblox_server_.getEsdfMapPtr()) {
  //   return false;
  // }
  //
  // Eigen::VectorXi observed;
  // voxblox_server_.getEsdfMapPtr()->batchGetDistanceAtPosition(path.transpose(),dist, observed);
  for (int i = 0; i < path.rows(); i++)
  {
    Eigen::Vector3d point = path.row(i);
    dist(i) = getMapDistance(point);
  }
  return true;
}

bool CostMap::getPathCost(const Eigen::Matrix<double, Eigen::Dynamic, 3>& path, Eigen::VectorXd& cost)
{
  cost.resize(path.rows());
  for (int i = 0; i < path.rows(); i++)
  {
    cost(i) = getCostOfPos(path.row(i));
  }
  return true;
}

bool CostMap::getPathGradient(const Eigen::Matrix<double, Eigen::Dynamic, 3>& path, Eigen::Matrix<double, Eigen::Dynamic, 3>& grad_along_path)
{
  grad_along_path.resize(path.rows(), 3);
  for (int i = 0; i < path.rows(); i++)
  {
    Eigen::Vector3d grad(0, 0, 0);
    if (!gradient(path.row(i), grad))
    {
      ROS_WARN("[CostMap::gradient] Cannot compute gradient for point (%.2f, %.2f %.2f). Gradient is set to zero.", path(i, 0), path(i, 1), path(i, 2));
    }
    grad_along_path.row(i) = grad;
  }
  return true;
}

// Visualizes the cost map
void CostMap::publishCostMapCallback(const ros::TimerEvent&)
{
  pcl::PointCloud<pcl::PointXYZI> cloud;

  constexpr double min_cost = 0.0;
  constexpr double max_cost = 15.0;

  if (visualize_all_)
  {
    voxblox::createDistancePointcloudFromEsdfLayer(voxblox_server_.getEsdfMapPtr()->getEsdfLayer(), &cloud);

    for (size_t i = 0; i < cloud.points.size(); i++)
    {
      auto point = cloud.points[i];
      double cost = getCostOfPos(Eigen::Vector3d(point.x, point.y, point.z));
      cost = std::min(cost, max_cost);
      cost = std::max(cost, min_cost);

      cloud.points[i].intensity = (cost - min_cost) / (max_cost - min_cost);
    }
  } else if (visualize_slice_)
  {
    constexpr int kZAxisIndex = 2;

    voxblox::createDistancePointcloudFromEsdfLayerSlice(voxblox_server_.getEsdfMapPtr()->getEsdfLayer(), kZAxisIndex, cost_map_slice_height_, &cloud);

    for (size_t i = 0; i < cloud.points.size(); i++)
    {
      auto point = cloud.points[i];
      double cost = getCostOfPos(Eigen::Vector3d(point.x, point.y, point.z));
      cost = std::min(cost, max_cost);
      cost = std::max(cost, min_cost);

      cloud.points[i].z = cost; // cost map slice height is the cost value
      cloud.points[i].intensity = (cost - min_cost) / (max_cost - min_cost);
    }
  }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = "world";
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);

  cost_map_pub_.publish(cloud_msg);

}


}  // namespace planner
