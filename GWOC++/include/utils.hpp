//  utils.hpp
//
//  Author:
//       Ahmad Dajani <eng.adajani@gmail.com>
//
//  Copyright (c) 2020 Ahmad Dajani
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU Lesser General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU Lesser General Public License for more details.
//
//  You should have received a copy of the GNU Lesser General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __UTILS_H
#define __UTILS_H
#include "benchmark.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>
class Utils {

public:
	static double GenerateRandomNumber();
	static double *Create1DZeroArray(unsigned int columnCount);
	static double **Create2DRandomArray(unsigned int rowCount, unsigned int columnCount, Boundaries boundaries[]);
	static void Clip1DArray(double array[], unsigned int columnCount, Boundaries boundaries[]);
	static std::vector<std::vector<int> > FindPoseRaw();
	static float raw2deg(int raw, std::string motor);
	static int deg2raw(double deg, std::string motor);
	static void calcular_features_surf(std::vector<cv::Mat>  &descp_src, std::vector<std::vector<cv::KeyPoint> >  &kpts_src, std::vector<std::string> imagens_src);
	static std::vector<std::vector<std::vector<cv::KeyPoint>>> surf_matches_matrix_encontrar_melhor(std::vector<std::vector<  std::vector<cv::DMatch> >> matriz_matches, std::vector<cv::Mat>  descp_src, std::vector< std::vector<cv::KeyPoint> >  kpts_src, std::vector<std::string> imagens_src, std::vector<std::vector<int>> &indices, std::vector<int> ind_vazios);
	static void filtrar_matches_keypoints_repetidos(std::vector<cv::KeyPoint> &kt, std::vector<cv::KeyPoint> &ks, std::vector<cv::DMatch> &m);
	static Eigen::Vector3d pointAngle(std::vector<int> pose, double R, Eigen::Vector2f center, int cols, int rows, Eigen::Vector2f foco);
};
#endif
