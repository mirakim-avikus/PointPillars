#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <iostream>
#include <array>
#include <memory>
#include <tuple>
#include <vector>
#include <torch/torch.h>
#include "pillar_layer.h"

class PointPillarsPre {
    public:
    PointPillarsPre(const std::vector<float>& voxel_size,
                    const std::vector<float>& point_cloud_range,
                    int max_num_points = 32,
                    std::pair<int, int> max_voxels = {16000, 40000},
                    bool training = false)
        : pillar_layer_(voxel_size, point_cloud_range, max_num_points, max_voxels, training) {}

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(const std::vector<torch::Tensor>& batched_pts) {
        return pillar_layer_.forward(batched_pts);
    }
    private:
    PillarLayer pillar_layer_;
};

int main(int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);

    if (pcl::io::loadPCDFile<pcl::PointXYZI>("../data/avikus/motorboat/007/lidar/flippedData/1736093516111.avikus.pcd", *cloud) == -1) 
    {
        PCL_ERROR("Couldn't read the file.\n");
        return -1;
    }

    std::cout << "Loaded " << cloud->width * cloud->height << " data points from PCD file." << std::endl;

    for (size_t i = 0; i < std::min((size_t)5, cloud->points.size()); ++i)
    {
        std::cout << "Point[" << i << "] : " << cloud->points[i].x << ", " 
                                            << cloud->points[i].y << ", " 
                                            << cloud->points[i].z << ", "
                                            << cloud->points[i].intensity << std::endl;
    }

    size_t num_points = cloud->points.size();
    torch::Tensor pt_tensor = torch::empty({(long)num_points, 4}, torch::kFloat32);
    for (size_t i = 0; i < num_points; i++)
    {
        pt_tensor[i][0] = cloud->points[i].x;
        pt_tensor[i][1] = cloud->points[i].y;
        pt_tensor[i][2] = cloud->points[i].z;
        pt_tensor[i][3] = cloud->points[i].intensity;
    }

    std::vector<torch::Tensor> pt_tensor_vec;
    pt_tensor_vec.push_back(pt_tensor);

    // point cloud range
    std::vector<float> point_cloud_range = {0.0f, -50.0f, -10.0f, 250.0f, 50.0f, 10.0f};
    std::vector<float> voxel_size = {0.16f, 0.16, 4.0f};

    PointPillarsPre pointpillarpre(voxel_size, point_cloud_range);
    std::tuple<at::Tensor, at::Tensor, at::Tensor> results = pointpillarpre.forward(pt_tensor_vec);
    
    torch::Tensor pillar_list = std::get<0>(results);
    torch::Tensor coors_list = std::get<1>(results);
    torch::Tensor npoints_list = std::get<2>(results);

    std::cout << "pillar list : "<< std::endl;
    std::cout << pillar_list.index({torch::indexing::Slice(0, 10)});

    std::cout << "coors list : " << std::endl;
    std::cout << coors_list.index({torch::indexing::Slice(0, 10)});

    std::cout << "npoints list : " << std::endl;
    std::cout << npoints_list.index({torch::indexing::Slice(0, 10)});

    return 0;
}