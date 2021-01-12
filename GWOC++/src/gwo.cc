//  gwo.cc
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

#include <iostream>
#include <cmath>
#include <chrono>
#include <algorithm> //std::copy
#include <iomanip> //std::setprecision 
#include "gwo.hpp"
#include <iostream>
#include <string>
#include <math.h>
#include <sys/stat.h>
#include <ostream>
#include <chrono>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl/kdtree/kdtree_flann.h>
#include <dirent.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/mat.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include <opencv2/stitching/detail/blenders.hpp>

#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>
GWO::GWO(Benchmark *benchmark, unsigned int searchAgentsCount,
	unsigned int maximumIterations) {
	benchmark_m = benchmark;
	searchAgentsCount_m = searchAgentsCount;
	maximumIterations_m = maximumIterations;
	boundaries_m = benchmark_m->getBoundaries();
	dimension_m = benchmark_m->GetDimension();
	convergenceCurve_m = Utils::Create1DZeroArray(maximumIterations_m);

	//Initialize alpha, beta, and delta_pos
	alphaPosition_m = Utils::Create1DZeroArray(dimension_m);
	alphaScore_m = std::numeric_limits<double>::infinity();
	betaPosition_m = Utils::Create1DZeroArray(dimension_m);
	betaScore_m = std::numeric_limits<double>::infinity();
	deltaPosition_m = Utils::Create1DZeroArray(dimension_m);
	deltaScore_m = std::numeric_limits<double>::infinity();

	//Initialize the positions of search agents
	positions_m = Utils::Create2DRandomArray(searchAgentsCount_m, dimension_m, boundaries_m);
}

GWO::~GWO() {
	for (register unsigned int agentId = 0; agentId < searchAgentsCount_m; agentId++) {
		delete positions_m[agentId];
	}
	delete positions_m;
	delete alphaPosition_m;
	delete betaPosition_m;
	delete deltaPosition_m;
	delete convergenceCurve_m;
}

void GWO::calculateFitness(std::vector<std::vector<std::vector<cv::KeyPoint>>> bestKey, std::vector<std::string> imagens_src, cv::Mat im360, cv::Mat image1, cv::Mat image2, std::vector<std::vector<int>> indices) {
	double fitness;
	//auto start_time = std::chrono::high_resolution_clock::now();
//#pragma omp parallel for
	for (register int agentIndex = 0; agentIndex < searchAgentsCount_m; agentIndex++) {
		Utils::Clip1DArray(positions_m[agentIndex], dimension_m, boundaries_m);

		//Calculate objective function for each search agent
		fitness = benchmark_m->fitness(positions_m[agentIndex], bestKey, imagens_src, im360, image1, image2, indices);

		//Update Alpha, Beta, and Delta
		if (fitness < alphaScore_m) {
			alphaScore_m = fitness; // Update alpha
			std::copy(&positions_m[agentIndex][0], &positions_m[agentIndex][dimension_m], &alphaPosition_m[0]);
		}

		if ((fitness > alphaScore_m) && (fitness < betaScore_m)) {
			betaScore_m = fitness;  // Update beta
			std::copy(&positions_m[agentIndex][0], &positions_m[agentIndex][dimension_m], &betaPosition_m[0]);
		}

		if ((fitness > alphaScore_m) && (fitness > betaScore_m) && (fitness < deltaScore_m)) {
			deltaScore_m = fitness; //Update delta
			std::copy(&positions_m[agentIndex][0], &positions_m[agentIndex][dimension_m], &deltaPosition_m[0]);
		}
	}
	/*auto finish_time = std::chrono::high_resolution_clock::now();
	auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(finish_time - start_time).count() * 1e-9;
	std::cout <<"calculateFitness: " <<time << "\n";*/
}

void GWO::updateWolves(double a) {
	//auto start_time = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
	for (register int agentIndex = 0; agentIndex < searchAgentsCount_m; agentIndex++)
	{
		for (register unsigned int variable = 0; variable < dimension_m; variable++) {
			double r1 = Utils::GenerateRandomNumber(); //r1 is a random number in [0,1]
			double r2 = Utils::GenerateRandomNumber(); //r2 is a random number in [0,1]

			double A1 = 2.0 * a * r1 - a; //Equation (3.3)
			double C1 = 2.0 * r2; //Equation (3.4)

			double D_alpha = std::abs(C1 * alphaPosition_m[variable] - positions_m[agentIndex][variable]); //Equation (3.5)-part 1
			double X1 = alphaPosition_m[variable] - A1 * D_alpha; //Equation (3.6)-part 1

			r1 = Utils::GenerateRandomNumber(); //r1 is a random number in [0,1]
			r2 = Utils::GenerateRandomNumber(); //r2 is a random number in [0,1]

			double A2 = 2 * a * r1 - a; //Equation (3.3)
			double C2 = 2 * r2; //Equation (3.4)

			double D_beta = std::abs(C2 * betaPosition_m[variable] - positions_m[agentIndex][variable]); //Equation (3.5)-part 2
			double X2 = betaPosition_m[variable] - A2 * D_beta; //Equation (3.6)-part 2

			r1 = Utils::GenerateRandomNumber(); //r1 is a random number in [0,1]
			r2 = Utils::GenerateRandomNumber(); //r2 is a random number in [0,1]

			double A3 = 2.0 * a * r1 - a; //Equation (3.3)
			double C3 = 2.0 * r2; //Equation (3.4)

			double D_delta = std::abs(C3 * deltaPosition_m[variable] - positions_m[agentIndex][variable]); //Equation (3.5)-part 3
			double X3 = deltaPosition_m[variable] - A3 * D_delta; //Equation (3.5)-part 3

			positions_m[agentIndex][variable] = (X1 + X2 + X3) / 3.0;  //Equation (3.7)
		}
	}
	/*auto finish_time = std::chrono::high_resolution_clock::now();
	auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(finish_time - start_time).count() * 1e-9;
	std::cout << "updateWolves: " << time << "\n";*/
}

double* GWO::Evaluate(bool debug, std::vector<std::vector<std::vector<cv::KeyPoint>>> bestKey, std::vector<std::string> imagens_src, cv::Mat im360, std::vector<std::vector<int>> indices) {
	double a;
	cv::Mat image1 = cv::imread(imagens_src[0]);
	cv::Mat image2 = cv::imread(imagens_src[1]);
	auto start_time = std::chrono::high_resolution_clock::now();
	//#pragma omp parallel for
	for (register int iteration = 0; iteration < maximumIterations_m; iteration++) {
		auto start_time = std::chrono::high_resolution_clock::now();
		calculateFitness(bestKey, imagens_src, im360, image1, image2, indices);

		//a decreases linearly from 2 to 0
		a = 2.0 - iteration * (2.0 / maximumIterations_m);

		updateWolves(a);
		convergenceCurve_m[iteration] = alphaScore_m;

		if (debug && (iteration % 1 == 0)) {
			std::cout << "At iteration " << iteration << " the best fitness is "
				<< std::setprecision(10)
				<< alphaScore_m << std::endl;
		}
		/*	auto finish_time = std::chrono::high_resolution_clock::now();
			executionTime_m = std::chrono::duration_cast<std::chrono::nanoseconds>(finish_time - start_time).count() * 1e-9;*/
	}

	auto finish_time = std::chrono::high_resolution_clock::now();
	executionTime_m = std::chrono::duration_cast<std::chrono::nanoseconds>(finish_time - start_time).count() * 1e-9;

	return convergenceCurve_m;
}

double* GWO::GetAlphaPosition() {
	return alphaPosition_m;
}

double GWO::GetAlphaScore() {
	return alphaScore_m;
}

double* GWO::GetBetaPosition() {
	return betaPosition_m;
}

double GWO::GetBetaScore() {
	return betaScore_m;
}

double* GWO::GetDeltaPosition() {
	return deltaPosition_m;
}

double GWO::GetDeltaScore() {
	return deltaScore_m;
}

double GWO::GetExecutionTime() {
	return executionTime_m;
}

std::ostream& operator << (std::ostream& os, const GWO *gwo) {
	os << std::scientific
		<< std::setprecision(9)
		<< "Benchmark: " << gwo->benchmark_m->GetName() << std::endl
		<< "Alpha position = ";

	for (register unsigned int variable = 0; variable < gwo->dimension_m; variable++) {
		os << gwo->alphaPosition_m[variable] << " ";
	}

	os << std::endl
		<< "Alpha score (Fitness) = " << gwo->alphaScore_m << std::endl
		<< "Time = " << gwo->executionTime_m << " seconds";


	//Writing in sfm file;

	std::ofstream cam{ "C:/dataset3/cameras_otimizado.sfm" };
	int teste = 0;
	int variable = 0;
	for (int indice_posicao = 0; indice_posicao < gwo->dimension_m / 6; indice_posicao++)
	{
		std::string nome_imagem_atual;
		if (indice_posicao + 1 < 10)
			nome_imagem_atual = "imagem_00" + std::to_string(indice_posicao + 1);
		else if (indice_posicao + 1 < 100)
			nome_imagem_atual = "imagem_0" + std::to_string(indice_posicao + 1);
		else
			nome_imagem_atual = "imagem_" + std::to_string(indice_posicao + 1);
		teste = variable + 6;
		cam << "C:/dataset3/" + nome_imagem_atual + ".png" << " "<<"0" << " " ;
		for (variable; variable < teste; variable++)
		{
			//	os << gwo->alphaPosition_m[variable] << " ";

							/*	Matrix3f R360 = euler2matrix(current_roll, -DEG2RAD(raw2deg(tilts_raw[indice_posicao], "tilt")), -DEG2RAD(raw2deg(pans_raw[indice_posicao], "pan")));
					cout << pasta + nome_imagem_atual + ".png";*/
			cam <<  gwo->alphaPosition_m[variable] << " ";
		}
		cam << "\n";
	}

	cam.close();
	return os;
}
