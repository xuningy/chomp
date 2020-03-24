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

void CostMap::initialize(ros::NodeHandle& nh, std::shared_ptr<SDFMap> sdf_map, double epsilon)
{
  sdf_map_ = sdf_map;
  resolution_ = sdf_map_->getResolution();
  resolution_inv_ = 1.0/resolution_;
  epsilon_ = epsilon;
  epsilon_times_half_ = 0.5*epsilon;
  half_divided_by_epsilon_ = 0.5/epsilon;

  ros::NodeHandle n(nh);
  cost_map_pub_ = n.advertise<sensor_msgs::PointCloud2>("/cost_map", 10);
  cost_map_timer_ = n.createTimer(ros::Duration(0.05), &CostMap::publishCostMapCallback, this);

  // feel free to replace this.
  cost_map_slice_height_ = sdf_map_->mp_.esdf_slice_height_;

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
  double dist = sdf_map_->getDistance(pos);
  return cost(dist);
}

double CostMap::getCostOfIdx(const Eigen::Vector3i& id) {
  double dist = sdf_map_->getDistance(id);
  return cost(dist);
}


bool CostMap::gradient(const Eigen::Vector3d& x, Eigen::Vector3d& grad)
{
  if(!sdf_map_->isInMap(x))
  {
    grad.setZero();
    return false;
  }

  /* use trilinear interpolation */
  Eigen::Vector3d pos_m = x - 0.5 * resolution_ * Eigen::Vector3d::Ones();

  Eigen::Vector3i idx;
  sdf_map_->posToIndex(pos_m, idx);

  Eigen::Vector3d idx_pos, diff;
  sdf_map_->indexToPos(idx, idx_pos);

  diff = (x - idx_pos) * resolution_inv_;

  double values[2][2][2];
  for (int x = 0; x < 2; x++) {
    for (int y = 0; y < 2; y++) {
      for (int z = 0; z < 2; z++) {
        Eigen::Vector3i current_idx = idx + Eigen::Vector3i(x, y, z);
        values[x][y][z] = getCostOfIdx(current_idx);
      }
    }
  }

  double v00 = (1 - diff[0]) * values[0][0][0] + diff[0] * values[1][0][0];
  double v01 = (1 - diff[0]) * values[0][0][1] + diff[0] * values[1][0][1];
  double v10 = (1 - diff[0]) * values[0][1][0] + diff[0] * values[1][1][0];
  double v11 = (1 - diff[0]) * values[0][1][1] + diff[0] * values[1][1][1];
  double v0 = (1 - diff[1]) * v00 + diff[1] * v10;
  double v1 = (1 - diff[1]) * v01 + diff[1] * v11;
  double dist = (1 - diff[2]) * v0 + diff[2] * v1;

  grad[2] = (v1 - v0) * resolution_inv_;
  grad[1] = ((1 - diff[2]) * (v10 - v00) + diff[2] * (v11 - v01)) * resolution_inv_;
  grad[0] = (1 - diff[2]) * (1 - diff[1]) * (values[1][0][0] - values[0][0][0]);
  grad[0] += (1 - diff[2]) * diff[1] * (values[1][1][0] - values[0][1][0]);
  grad[0] += diff[2] * (1 - diff[1]) * (values[1][0][1] - values[0][0][1]);
  grad[0] += diff[2] * diff[1] * (values[1][1][1] - values[0][1][1]);

  grad[0] *= resolution_inv_;

}

bool CostMap::getPathDist(const Eigen::Matrix<double, Eigen::Dynamic, 3>& path, Eigen::VectorXd& dist)
{
  dist.resize(path.rows());
  for (size_t i = 0; i < path.rows(); i++)
  {
    Eigen::Vector3d point = path.row(i);
    dist(i) = sdf_map_->getDistance(point);
  }
  return true;
}

bool CostMap::getPathCost(const Eigen::Matrix<double, Eigen::Dynamic, 3>& path, Eigen::VectorXd& cost)
{
  cost.resize(path.rows());
  for (size_t i = 0; i < path.rows(); i++)
  {
    cost(i) = getCostOfPos(path.row(i));
  }
  return true;
}

bool CostMap::getPathGradient(const Eigen::Matrix<double, Eigen::Dynamic, 3>& path, Eigen::Matrix<double, Eigen::Dynamic, 3>& grad_along_path)
{
  grad_along_path.resize(path.rows(), 3);
  for (size_t i = 0; i < path.rows(); i++)
  {
    Eigen::Vector3d grad(0, 0, 0);
    if (!gradient(path.row(i), grad))
    {
      ROS_WARN("[CostMap::gradient] Point (%.2f, %.2f %.2f) is outside of the map. Gradient is set to zero.");
    }
    grad_along_path.row(i) = grad;
  }
  return true;
}

// Visualizes the cost map with Z as the cost, taken at a 2D slice with Z given
// by sdf_map_->mp_.esdf_slice_height_.
void CostMap::publishCostMapCallback(const ros::TimerEvent&)
{
  double cost;
  pcl::PointCloud<pcl::PointXYZI> cloud;
  pcl::PointXYZI pt;

  const double min_cost = 0.0;
  const double max_cost = 3.0;

  Eigen::Vector3i min_cut = sdf_map_->md_.local_bound_min_ -
      Eigen::Vector3i(sdf_map_->mp_.local_map_margin_, sdf_map_->mp_.local_map_margin_, sdf_map_->mp_.local_map_margin_);
  Eigen::Vector3i max_cut = sdf_map_->md_.local_bound_max_ +
      Eigen::Vector3i(sdf_map_->mp_.local_map_margin_, sdf_map_->mp_.local_map_margin_, sdf_map_->mp_.local_map_margin_);
  sdf_map_->boundIndex(min_cut);
  sdf_map_->boundIndex(max_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y) {

      Eigen::Vector3d pos;
      sdf_map_->indexToPos(Eigen::Vector3i(x, y, 1), pos);
      pos(2) = cost_map_slice_height_;

      cost = getCostOfPos(pos);
      cost = min(cost, max_cost);
      cost = max(cost, min_cost);

      pt.x = pos(0);
      pt.y = pos(1);
      pt.z = getCostOfPos(pos); // Xuning: use the z axis to display the cost map
      pt.intensity = (cost - min_cost) / (max_cost - min_cost);
      cloud.push_back(pt);
    }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = sdf_map_->mp_.frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);

  cost_map_pub_.publish(cloud_msg);

}


}  // namespace planner
