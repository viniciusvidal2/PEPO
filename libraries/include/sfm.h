#ifndef SFM_H
#define SFM_H

#include "ros/ros.h"

#include <math.h>

#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>

#include <laser_geometry/laser_geometry.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <nav_msgs/Odometry.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/videoio.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/photo.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
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
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_distance.h>
#include <pcl/point_types_conversion.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>

#include <string>
#include <iostream>
#include <ros/ros.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <cstdlib>
#include <csignal>
#include <ctime>
#include <math.h>

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace pcl;
using namespace pcl::io;
using namespace pcl::registration;

typedef PointXYZRGB       PointT ;
typedef PointXYZRGBNormal PointTN;

class SFM
{
public:
  SFM(string p1, string p2);
  virtual ~SFM();
  void obter_dados(vector<string> linhas_src, vector<string> linhas_tgt);
  void calcular_features_orb();
  void calcular_features_surf();
  void orb_matches_matrix_encontrar_melhor();
  void surf_matches_matrix_encontrar_melhor();
  void calcular_pose_relativa();
  void set_debug(bool b);
  void obter_transformacao_final(Matrix4f &T, PointCloud<PointTN>::Ptr tgt, PointCloud<PointTN>::Ptr src);
  void get_matched_keypoints(vector<Point2f> &kptgt, vector<Point2f> &kpsrc);
  void get_indices_imagens(int &t, int &s);
  void obter_correspondencias_3D_e_T();
  Matrix4f icp(PointCloud<PointTN>::Ptr ctgt, PointCloud<PointTN>::Ptr csrc, float vs, int its);

private:
  void filterMatchesLineCoeff(vector<DMatch> &matches, vector<KeyPoint> kpref, vector<KeyPoint> kpnow, float width, float n);
  void ler_nuvens_correspondentes();
  void estimar_escala_translacao();
  void filtrar_ruidos_inpaint(MatrixXf &dt, MatrixXf &ds);
  void filtrar_matches_keypoints_repetidos(vector<KeyPoint> kt, vector<KeyPoint> ks, vector<DMatch> &m);

  string pasta_src, pasta_tgt;
  vector<string> imagens_src, imagens_tgt;
  Mat K;
  int imcols, imrows;

  vector<Matrix3f> rots_src, rots_tgt;
  Vector3f t_laser_cam;
  Matrix3f Rrel;
  Vector3f trel;
  Matrix3f R_src_tgt;

  vector< vector<KeyPoint> > kpts_tgt, kpts_src;
  vector<Mat> descp_tgt, descp_src;

  vector<KeyPoint> best_kptgt, best_kpsrc;
  vector<DMatch> best_matches;
  int im_src_indice, im_tgt_indice;

  bool debug;

  vector<string> nomes_nuvens;
  PointCloud<PointTN>::Ptr cloud_src;
  PointCloud<PointTN>::Ptr cloud_tgt;
  Matrix4f Tsvd;

};

#endif // SFM_H
