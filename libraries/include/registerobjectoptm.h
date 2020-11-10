#ifndef REGISTEROBJECTOPTM_H
#define REGISTEROBJECTOPTM_H

#include <string>
#include <math.h>
#include <cstdlib>
#include <boost/accumulators/accumulators.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/surface/mls.h>
#include <pcl_ros/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/gicp6d.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_rejection_distance.h>
#include <pcl/point_types_conversion.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>

/// Definicoes e namespaces
///
using namespace pcl;
using namespace pcl::io;
using namespace pcl::registration;
using namespace cv;
using namespace std;
using namespace Eigen;

typedef PointXYZRGB       PointT ;
typedef PointXYZRGBNormal PointTN;

class RegisterObjectOptm
{
public:
    RegisterObjectOptm();
    virtual ~RegisterObjectOptm();
    void readCloudAndPreProcess(string name, PointCloud<PointT>::Ptr cloud);
    Matrix4f icp(PointCloud<PointTN>::Ptr ctgt, PointCloud<PointTN>::Ptr csrc, float vs, int its);
    Matrix4f gicp6d(PointCloud<PointTN>::Ptr ctgt, PointCloud<PointTN>::Ptr csrc, float vs, int its);
    void matchFeaturesAndFind3DPoints(Mat imref, Mat imnow, PointCloud<PointTN>::Ptr cref, PointCloud<PointTN>::Ptr cnow,
                                      PointCloud<PointTN>::Ptr cmr, PointCloud<PointTN>::Ptr cmn);
    Matrix4f optmizeTransformSVD(PointCloud<PointTN>::Ptr cref, PointCloud<PointTN>::Ptr cnow);
    void searchNeighborsKdTree(PointCloud<PointTN>::Ptr cnow, PointCloud<PointTN>::Ptr cobj, float radius, float rate);
    void readNVM(string folder, string nome, vector<string> &clouds, vector<string> &images, vector<Matrix4f> &poses, float &foco);

};

#endif // REGISTEROBJECTOPTM_H
