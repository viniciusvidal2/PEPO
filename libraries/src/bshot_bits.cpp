#include <iostream>
#include <bitset>
#include <vector>
#include <ctime>

#include <dirent.h> // for looping over the files in the directory

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/range_image/range_image.h>
#include <pcl/range_image/range_image_planar.h>
//#include <pcl/visualization/range_image_visualizer.h>
//#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/visualization/point_cloud_color_handlers.h>
//#include <pcl/visualization/point_cloud_geometry_handlers.h>
//#include <pcl/console/parse.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
//#include <pcl/visualization/cloud_viewer.h>
//#include <pcl/visualization/cloud_viewer.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
//#include <pcl/visualization/histogram_visualizer.h>
#include <pcl/filters/voxel_grid.h>
//#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/common/common_headers.h>
//#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
//#include <pcl/io/vtk_lib_io.h>
//#include <pcl/io/vtk_io.h>
//#include <pcl/console/print.h>
//#include <pcl/console/parse.h>
//#include <pcl/console/time.h>
#include <string>
#include <fstream>
#include <string>


using namespace std;
using namespace Eigen;
using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;

#define PI 3.14159265


template< typename T >
T minVect(const T *v, int n, int *ind=NULL)
{
    assert(n > 0);

    T min = v[0];
    if (ind != NULL) *ind = 0;
    for (int i=1; i<n; i++)
        if (v[i] < min) {
            min = v[i];
            if (ind != NULL) *ind=i;
        }

    return min;
}


class bshot_descriptor
{
public:
    std::bitset< 352 > bits;
};


class bshot
{

public :

    pcl::PointCloud<pcl::PointXYZ> cloud1, cloud2;
    pcl::PointCloud<pcl::Normal> cloud1_normals, cloud2_normals;
    pcl::PointCloud<pcl::PointXYZ> cloud1_keypoints, cloud2_keypoints;

    pcl::PointCloud<pcl::SHOT352> cloud1_shot, cloud2_shot;

    std::vector<bshot_descriptor> cloud1_bshot, cloud2_bshot;


    void calculate_normals ( int n )
    {
        // Estimate the normals.
        pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normalEstimation;
        normalEstimation.setKSearch(n);
        normalEstimation.setNumberOfThreads(12);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
        normalEstimation.setSearchMethod(kdtree);

        normalEstimation.setInputCloud(cloud1.makeShared());
        normalEstimation.compute(cloud1_normals);

#pragma omp parallel for
        for(size_t i=0; i < cloud1_normals.size(); i++){
            Eigen::Vector3f normal, cp;
            normal << cloud1_normals.points[i].normal_x, cloud1_normals.points[i].normal_y, cloud1_normals.points[i].normal_z;
            cp     << -cloud1.points[i].x, -cloud1.points[i].y, -cloud1.points[i].z;
            float cos_theta = (normal.dot(cp))/(normal.norm()*cp.norm());
            if(cos_theta <= 0){ // Esta apontando errado, deve inverter
                cloud1_normals.points[i].normal_x = -cloud1_normals.points[i].normal_x;
                cloud1_normals.points[i].normal_y = -cloud1_normals.points[i].normal_y;
                cloud1_normals.points[i].normal_z = -cloud1_normals.points[i].normal_z;
            }
        }

        normalEstimation.setInputCloud(cloud2.makeShared());
        normalEstimation.compute(cloud2_normals);

#pragma omp parallel for
        for(size_t i=0; i < cloud2_normals.size(); i++){
            Eigen::Vector3f normal, cp;
            normal << cloud2_normals.points[i].normal_x, cloud2_normals.points[i].normal_y, cloud2_normals.points[i].normal_z;
            cp     << -cloud2.points[i].x, -cloud2.points[i].y, -cloud2.points[i].z;
            float cos_theta = (normal.dot(cp))/(normal.norm()*cp.norm());
            if(cos_theta <= 0){ // Esta apontando errado, deve inverter
                cloud2_normals.points[i].normal_x = -cloud2_normals.points[i].normal_x;
                cloud2_normals.points[i].normal_y = -cloud2_normals.points[i].normal_y;
                cloud2_normals.points[i].normal_z = -cloud2_normals.points[i].normal_z;
            }
        }

    }


    void  calculate_voxel_grid_keypoints ( float leaf_size )
    {
        // Find Keypoints on the input cloud
        pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
        voxel_grid.setLeafSize(leaf_size, leaf_size, leaf_size);

        voxel_grid.setInputCloud(cloud1.makeShared());
        voxel_grid.filter(cloud1_keypoints);

        voxel_grid.setInputCloud(cloud2.makeShared());
        voxel_grid.filter(cloud2_keypoints);


    }


    void calculate_SHOT ( float radius )
    {

        // SHOT estimation object.
        pcl::SHOTEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot;
        shot.setRadiusSearch(radius);

        shot.setNumberOfThreads(12);

        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
        shot.setSearchMethod(kdtree);

        shot.setInputCloud(cloud1_keypoints.makeShared());
        shot.setSearchSurface(cloud1.makeShared());
        shot.setInputNormals(cloud1_normals.makeShared());
        shot.compute(cloud1_shot);

        shot.setInputCloud(cloud2_keypoints.makeShared());
        shot.setSearchSurface(cloud2.makeShared());
        shot.setInputNormals(cloud2_normals.makeShared());
        shot.compute(cloud2_shot);

    }


    void compute_bshot()
    {
        compute_bshot_from_SHOT( cloud1_shot, cloud1_bshot);
        compute_bshot_from_SHOT( cloud2_shot, cloud2_bshot);
    }

    void compute_bshot_from_SHOT(pcl::PointCloud<pcl::SHOT352>& shot_descriptors_here, std::vector<bshot_descriptor>& bshot_descriptors)
    {
        bshot_descriptors.resize(shot_descriptors_here.size());
        for (int i = 0; i < (int)shot_descriptors_here.size(); i++)
        {
            std::bitset < 352 > temp;
            temp.reset();

            for (int j = 0 ; j < 88 ; j++)
            {
                float vec[4] = { 0 };
                for (int k = 0 ; k < 4 ; k++)
                {
                    vec[k] = shot_descriptors_here[i].descriptor[ j*4 + k ];

                }

                std::bitset< 4 > bit;
                bit.reset();

                float sum = vec[0]+vec[1]+vec[2]+vec[3];

                if (vec[0] == 0 and vec [1] == 0 and vec[2] == 0 and vec[3] == 0)
                {
                    //bin[0] = bin[1] = bin[2] = bin[3] = 0;
                    // by default , they are all ZEROS
                }
                else if ( vec[0] > (0.9 * (sum) ) )
                {
                    bit.set(0);
                }
                else if ( vec[1] > (0.9 * (sum) ) )
                {
                    bit.set(1);
                }
                else if ( vec[2] > (0.9 * (sum) ) )
                {
                    bit.set(2);
                }
                else if ( vec[3] > (0.9 * (sum) ) )
                {

                    bit.set(3);
                }
                else if ( (vec[0]+vec[1]) > (0.9 * (sum))  )
                {

                    bit.set(0);
                    bit.set(1);
                }
                else if ( (vec[1]+vec[2]) > (0.9 * (sum)) )
                {

                    bit.set(1);
                    bit.set(2);
                }

                else if ( (vec[2]+vec[3]) > (0.9 * (sum)) )
                {
                    ;
                    bit.set(2);
                    bit.set(3);
                }
                else if ( (vec[0]+vec[3]) > (0.9 * (sum)) )
                {

                    bit.set(0);
                    bit.set(3);
                }
                else if ( (vec[1]+vec[3]) > (0.9 * (sum)) )
                {

                    bit.set(1);
                    bit.set(3);
                }
                else if ( (vec[0]+vec[2]) > (0.9 * (sum)) )
                {

                    bit.set(0);
                    bit.set(2);
                }
                else if ( (vec[0]+ vec[1] +vec[2]) > (0.9 * (sum)) )
                {

                    bit.set(0);
                    bit.set(1);
                    bit.set(2);
                }
                else if ( (vec[1]+ vec[2] +vec[3]) > (0.9 * (sum)) )
                {

                    bit.set(1);
                    bit.set(2);
                    bit.set(3);
                }
                else if ( (vec[0]+ vec[2] +vec[3]) > (0.9 * (sum)) )
                {

                    bit.set(0);
                    bit.set(2);
                    bit.set(3);
                }
                else if ( (vec[0]+ vec[1] +vec[3]) > (0.9 * (sum)) )
                {

                    bit.set(0);
                    bit.set(1);
                    bit.set(3);
                }
                else
                {

                    bit.set(0);
                    bit.set(1);
                    bit.set(2);
                    bit.set(3);
                }

                if (bit.test(0))
                    temp.set(j*4);

                if (bit.test(1))
                    temp.set(j*4 + 1);

                if (bit.test(2))
                    temp.set(j*4 + 2);

                if (bit.test(3))
                    temp.set(j*4 + 3);

            }

            bshot_descriptors[i].bits = temp;
        }
    }


};




