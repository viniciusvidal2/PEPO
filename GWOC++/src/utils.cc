//  utils.cc
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
#include <algorithm>
#include <iostream>
#include <cstring>

#include <time.h> 

#include "utils.hpp"

// random number between 0 and 1
double Utils::GenerateRandomNumber() {
	static bool init = false;
	if (!init) {
		init = true;
		srand(time(NULL));
	}
	return (double)rand() / RAND_MAX;
}

// create 1D long double array with value zero
double* Utils::Create1DZeroArray(unsigned int columnCount) {
	double *array = new double[columnCount];
	std::fill_n(array, columnCount, 0.0);
	return array;
}

// create 2d long double array, its value is between (0,1) * (ub-lb)+lb
double** Utils::Create2DRandomArray(unsigned int rowCount, unsigned int columnCount, Boundaries boundaries[]) {
	//double* ini = [ 1427.099976, 1449.400024, 987.900024, 579.400024, 1427.099976, 1449.400024, 987.900024, 579.400024,34, 2276, 375, 2276 ];
	//double data[] { 1427.099976, 1449.400024, 987.900024, 579.400024, 1427.099976, 1449.400024, 987.900024, 579.400024, 34, 2276, 375, 2276 };

	//double teste = boundaries[0].lowerBound;
	double **array = new double *[rowCount];

	for (register unsigned int y = 0; y < rowCount; y++) {
		array[y] = new double[columnCount];
		if (y == 0) {
			for (register unsigned int x = 0; x < columnCount; x++) {
				//array[y][x] = data[x];
				array[y][x] = boundaries[x].lowerBound + 10;
				array[y][x + 1] = boundaries[x + 1].lowerBound + 10;
				array[y][x + 2] = boundaries[x + 2].lowerBound + 100;
				array[y][x + 3] = boundaries[x + 3].lowerBound + 100;
				array[y][x + 4] = boundaries[x + 4].lowerBound + 20;
				array[y][x + 5] = boundaries[x + 5].lowerBound + 20;
				x = x + 5;
			}
		}
		else {
			// randomize data and apply between (lower,upper) bound
			for (register unsigned int x = 0; x < columnCount; x++) {
				array[y][x] = boundaries[x].lowerBound + (boundaries[x].upperBound - boundaries[x].lowerBound) * GenerateRandomNumber();
			}
		}
	}


	return array;
}

/*
	Return back the search agents that go beyond the boundaries of the search space
	Given an interval, values outside the interval are clipped to the interval edges.
	For example, if an interval of [0, 1] is specified, values smaller than 0 become 0,
	and values larger than 1 become 1. ... If None, clipping is not performed on lower interval edge.
*/
void Utils::Clip1DArray(double array[], unsigned int columnCount, Boundaries boundaries[]) {
	for (register unsigned int column = 0; column < columnCount; column++) {
		double value = array[column];
		if (value < boundaries[column].lowerBound) {
			array[column] = boundaries[column].lowerBound;
		}
		if (value > boundaries[column].upperBound) {
			array[column] = boundaries[column].upperBound;
		}
	}
}
int Utils::deg2raw(double deg, std::string motor) {

	float raw_min_tilt = 2595, raw_hor_tilt = 2280, raw_max_tilt = 1595;
	float deg_min_tilt = 28, deg_hor_tilt = 0, deg_max_tilt = -60.9;
	float raw_deg = 11.37777, deg_raw = 1 / raw_deg;
	if (motor == "pan")
		return int(deg*raw_deg);//int((deg - deg_min_pan )*raw_deg + raw_min_pan);
	else
		return int((deg - deg_min_tilt)*raw_deg + raw_min_tilt);
}
float Utils::raw2deg(int raw, std::string motor) {
	float raw_min_tilt = 2595, raw_hor_tilt = 2280, raw_max_tilt = 1595;
	float deg_min_tilt = 28, deg_hor_tilt = 0, deg_max_tilt = -60.9;
	float raw_deg = 11.37777, deg_raw = 1 / raw_deg;
	if (motor == "pan")
		return float(raw)*deg_raw;//(float(raw) - raw_min_pan )*deg_raw + deg_min_pan;
	else
		return (float(raw) - raw_max_tilt)*deg_raw + deg_max_tilt;
}
std::vector<std::vector<int> > Utils::FindPoseRaw() {
	float raw_min_pan = 35, raw_max_pan = 4077;
	float deg_min_pan = 3, deg_max_pan = 358;
	float raw_min_tilt = 2595, raw_hor_tilt = 2280, raw_max_tilt = 1595;
	float deg_min_tilt = 28, deg_hor_tilt = 0, deg_max_tilt = -60.9;
	float raw_deg = 11.37777, deg_raw = 1 / raw_deg;
	// Pontos de observacao em pan
	int step = 30; // [DEG]
	//Os limites em pan pode considerar os deg_min e deg_max pra pan
	float inicio_scanner_deg_pan, final_scanner_deg_pan;
	inicio_scanner_deg_pan = deg_min_pan;
	final_scanner_deg_pan = deg_max_pan;
	int vistas_pan = int(final_scanner_deg_pan - inicio_scanner_deg_pan) / step + 2; // Vistas na horizontal, somar inicio e final do range
	std::vector<float> pans_deg, tilts_deg; // [DEG]
	std::vector<float> pans_camera_deg;
	std::vector<int>   pans_raw, tilts_raw;
	std::vector<float> tilts_camera_deg{ deg_min_tilt, deg_hor_tilt, -30.0f, deg_max_tilt };

	for (int j = 0; j < vistas_pan - 1; j++)
		pans_camera_deg.push_back(inicio_scanner_deg_pan + float(j*step));

	cv::Mat poseRaw(pans_camera_deg.size(), 3, CV_32F);
	// Enchendo vetores de waypoints de imagem em deg e raw globais
	std::vector<std::vector<int> > pose;
	for (int j = 0; j < pans_camera_deg.size(); j++) {
		for (int i = 0; i < tilts_camera_deg.size(); i++) {
			int tilt;
			if (remainder(j, 2) == 0) {
				tilts_deg.push_back(tilts_camera_deg[i]);
				tilts_raw.push_back(Utils::deg2raw(tilts_camera_deg[i], "tilt"));
				tilt = Utils::deg2raw(tilts_camera_deg[i], "tilt");
			}
			else {
				tilts_deg.push_back(tilts_camera_deg[tilts_camera_deg.size() - 1 - i]);
				tilts_raw.push_back(Utils::deg2raw(tilts_camera_deg[tilts_camera_deg.size() - 1 - i], "tilt"));
				tilt = Utils::deg2raw(tilts_camera_deg[tilts_camera_deg.size() - 1 - i], "tilt");
			}
			pans_deg.push_back(pans_camera_deg[j]);
			pans_raw.push_back(Utils::deg2raw(pans_camera_deg[j], "pan"));

			std::vector<int> pos{ 0, tilt, Utils::deg2raw(pans_camera_deg[j], "pan") };
			pose.push_back(pos);
		}
	}
	return pose;//roll,tilt,pan

}
Eigen::Vector3d Utils::pointAngle(std::vector<int> pose, double R, Eigen::Vector2f center, int cols, int rows, Eigen::Vector2f foco) {
	// Definir o foco em dimensoes fisicas do frustrum
	double F = 1;
	double minX, minY, maxX, maxY;
	double dx = center[0] - double(cols) / 2, dy = center[1] - double(rows) / 2;
	//    double dx = 0, dy = 0;
	maxX = F * (float(cols) - 2 * dx) / (2.0*foco[0]);
	minX = -F * (float(cols) + 2 * dx) / (2.0*foco[0]);
	maxY = F * (float(rows) - 2 * dy) / (2.0*foco[1]);
	minY = -F * (float(rows) + 2 * dy) / (2.0*foco[1]);
	//		// Calcular os 4 pontos do frustrum
	//		/*
	//								origin of the camera = p1
	//								p2--------p3
	//								|          |
	//								|  pCenter |<--- Looking from p1 to pCenter
	//								|          |
	//								p5--------p4
	//		*/

	Eigen::Matrix3d r1;
	r1 = Eigen::AngleAxisd(-DEG2RAD(Utils::raw2deg(pose[1], "pan")), Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(pose[0], Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(-DEG2RAD(Utils::raw2deg(pose[2], "tilt")), Eigen::Vector3d::UnitX());

	Eigen::Vector3d p, p1, p2, p3, p4, p5, pCenter;
	p << 0, 0, 0;
	p1 = r1 * p;
	p << minX, minY, F;
	p2 = r1 * p;
	p << maxX, minY, F;
	p3 = r1 * p;
	p << maxX, maxY, F;
	p4 = r1 * p;
	p << minX, maxY, F;
	p5 = r1 * p;
	p << 0, 0, F;
	pCenter = r1 * p;

	Eigen::Vector3d pontos = (pCenter - p1);
	return pontos;
}

void Utils::calcular_features_surf(std::vector<cv::Mat>  &descp_src, std::vector<std::vector<cv::KeyPoint> >  &kpts_src, std::vector<std::string> imagens_src)
{

#pragma omp parallel for
	for (int i = 0; i < descp_src.size(); i++) {
		// Iniciando Keypoints e Descritores atuais
		std::vector<cv::KeyPoint> kpsrc;
		cv::Mat  dsrc;

		// Ler a imagem inicial
		cv::Mat imsrc = cv::imread(imagens_src[i], cv::IMREAD_GRAYSCALE);

		// Descritores SIFT calculados
		cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();

		sift->detectAndCompute(imsrc, cv::Mat(), kpsrc, dsrc);
		// Calculando somatorio para cada linha de descritores
		cv::Mat dsrcsum;

		reduce(dsrc, dsrcsum, 1, CV_16UC1);
		// Normalizando e passando raiz em cada elementos de linha nos descritores da src
#pragma omp parallel for
		for (int i = 0; i < dsrc.rows; i++) {
			for (int j = 0; j < dsrc.cols; j++) {
				dsrc.at<float>(i, j) = sqrt(dsrc.at<float>(i, j) / (dsrcsum.at<float>(i, 0) + std::numeric_limits<float>::epsilon()));
			}
		}
		//Salvando no vetor de keypoints
		kpts_src[i] = kpsrc;
		// Salvando no vetor de cada um os descritores
		descp_src[i] = dsrc;
	}

}



void Utils::filtrar_matches_keypoints_repetidos(std::vector<cv::KeyPoint> &kt, std::vector<cv::KeyPoint> &ks, std::vector<cv::DMatch> &m) {
	// Matriz de bins para keypoints de target e source

	const int w = 1920 / 5, h = 1080 / 5;

	std::vector<std::vector<cv::DMatch>>matriz_matches[w];

#pragma omp parallel for
	for (int i = 0; i < w; i++) {
		matriz_matches[i].resize(h);
	}

	// Itera sobre os matches pra colocar eles nos bins certos

	for (int i = 0; i < m.size(); i++) {
		cv::KeyPoint ktt = kt[m[i].trainIdx];
		int u = ktt.pt.x / 5, v = ktt.pt.y / 5;
		matriz_matches[u][v].push_back(m[i]);
	}
	// Vetor auxiliar de matches que vao passar no teste de melhor distancia
	std::vector<cv::DMatch> boas_matches;
	// Procurando na matriz de matches
	for (int i = 0; i < w; i++) {
		for (int j = 0; j < h; j++) {
			if (matriz_matches[i][j].size() > 0) {
				// Se ha matches e for so uma, adicionar ela mesmo
				if (matriz_matches[i][j].size() == 1) {
					boas_matches.push_back(matriz_matches[i][j][0]);
				}
				else { // Se for mais de uma comparar a distancia com as outras
					cv::DMatch mbest = matriz_matches[i][j][0];
					for (int k = 1; k < matriz_matches[i][j].size(); k++) {
						if (matriz_matches[i][j][k].distance < mbest.distance)
							mbest = matriz_matches[i][j][k];
					}
					// Adicionar ao vetor a melhor opcao para aquele bin
					boas_matches.push_back(mbest);
				}
			}
			matriz_matches[i][j].clear(); // Ja podemos limpar aquele vetor, ja trabalhamos
		}
	}
	m = boas_matches;
	// Fazer o mesmo agora para as matches que sobraram e kpts da src
	// Itera sobre os matches pra colocar eles nos bins certos
	for (int i = 0; i < boas_matches.size(); i++) {
		cv::KeyPoint kst = ks[m[i].queryIdx];
		int u = kst.pt.x / 5, v = kst.pt.y / 5;
		matriz_matches[u][v].push_back(m[i]);
	}
	// Vetor auxiliar de matches que vao passar no teste de melhor distancia
	std::vector<cv::DMatch> otimas_matches;
	// Procurando na matriz de matches
	for (int i = 0; i < w; i++) {
		for (int j = 0; j < h; j++) {
			if (matriz_matches[i][j].size() > 0) {
				// Se ha matches e for so uma, adicionar ela mesmo
				if (matriz_matches[i][j].size() == 1) {
					otimas_matches.push_back(matriz_matches[i][j][0]);
				}
				else { // Se for mais de uma comparar a distancia com as outras
					cv::DMatch mbest = matriz_matches[i][j][0];
					for (int k = 1; k < matriz_matches[i][j].size(); k++) {
						if (matriz_matches[i][j][k].distance < mbest.distance)
							mbest = matriz_matches[i][j][k];
					}
					// Adicionar ao vetor a melhor opcao para aquele bin
					otimas_matches.push_back(mbest);
				}
			}
			matriz_matches[i][j].clear(); // Ja podemos limpar aquele vetor, ja trabalhamos
		}
	}
	
	// Retornando as matches que restaram
	m = otimas_matches;
}


std::vector<std::vector<std::vector<cv::KeyPoint>>> Utils::surf_matches_matrix_encontrar_melhor(std::vector<std::vector<  std::vector<cv::DMatch> >> matriz_matches, std::vector<cv::Mat>  descp_src, std::vector< std::vector<cv::KeyPoint> >  kpts_src, std::vector<std::string> imagens_src, std::vector<std::vector<int>> &indices) {
	// Matcher de FLANN
	cv::Ptr<cv::DescriptorMatcher> matcher;
	matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

	// Para cada combinacao de imagens, fazer match e salvar quantidade final para ver qual
	// a melhor depois
	auto start_time = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
	for (int frame0 = 0; frame0 < descp_src.size(); frame0++) {
		auto start_time = std::chrono::high_resolution_clock::now();
		for (int frame1 = 0; frame1 < indices[frame0].size(); frame1++)
		{

			std::vector<std::vector<cv::DMatch>> matches;
			std::vector<cv::DMatch> good_matches;
			if (!descp_src[frame0].empty() && !descp_src[indices[frame0][frame1]].empty()) {
				matcher->knnMatch(descp_src[frame0], descp_src[indices[frame0][frame1]], matches, 2);
			/*	auto finish_time = std::chrono::high_resolution_clock::now();
				auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(finish_time - start_time).count() * 1e-9;
				std::cout << "matches " << time << "\n";*/
				for (size_t k = 0; k < matches.size(); k++) 
				{
					if (matches.at(k).size() >= 2) 
					{
						if (matches.at(k).at(0).distance < 0.7*matches.at(k).at(1).distance) // Se e bastante unica frente a segunda colocada
							good_matches.push_back(matches.at(k).at(0));
					}
				}
				if (good_matches.size() > 0)
				{
					// Filtrar keypoints repetidos
					Utils::filtrar_matches_keypoints_repetidos(kpts_src[indices[frame0][frame1]], kpts_src[frame0], good_matches);
					// Filtrar por matches que nao sejam muito horizontais
					//filterMatchesLineCoeff(good_matches, kpts_tgt[i], kpts_src[j], imcols, DEG2RAD(50));

					// Anota quantas venceram nessa combinacao
					//matches_count(frame0, frame1) = good_matches.size();
					matriz_matches.at(frame0).at(frame1) = good_matches;

				}

			}
			
		}
	}
	

	bool debug = false;
	std::vector<std::vector<std::vector<cv::KeyPoint>>> bestKey;
	std::vector<cv::DMatch> best_matches;
	std::vector<cv::KeyPoint> best_kptgt, best_kpsrc;
	bestKey.resize(descp_src.size());

	indices.resize(descp_src.size());
	auto start_time1 = std::chrono::high_resolution_clock::now();
	for (int frame0 = 0; frame0 < descp_src.size(); frame0++)
	{
		for (int frame1 = 0; frame1 < indices[frame0].size(); frame1++)
		{
			std::vector<cv::KeyPoint> curr_kpts_tgt = kpts_src[indices[frame0][frame1]], curr_kpts_src = kpts_src[frame0];
			best_matches = matriz_matches.at(frame0).at(frame1);

			for (auto m : best_matches) {
				best_kptgt.emplace_back(curr_kpts_tgt[m.trainIdx]);
				best_kpsrc.emplace_back(curr_kpts_src[m.queryIdx]);
			}
			
			// Converter os pontos para o formato certo
			std::vector<cv::Point2d> kptgt(best_kptgt.size()), kpsrc(best_kpsrc.size());
#pragma omp parallel for
			for (int i = 0; i < best_kptgt.size(); i++) {
				kptgt[i] = best_kptgt[i].pt;
				kpsrc[i] = best_kpsrc[i].pt;
			}
			if (best_matches.size() > 15) {
				// Calcular matriz fundamental
				cv::Mat F = findFundamentalMat(kpsrc, kptgt); // Transformacao da src para a tgt
				// Calcular pontos que ficam por conferencia da matriz F
				Eigen::Matrix3d F_;
				cv::cv2eigen(F, F_);
				std::vector<cv::Point2d> tempt, temps;
				std::vector<int> indices_inliers;
				for (int i = 0; i < kpsrc.size(); i++) {
					Eigen::Vector3d pt{ kptgt[i].x, kptgt[i].y, 1 }, ps = { kpsrc[i].x, kpsrc[i].y, 1 };
					Eigen::MatrixXd erro = pt.transpose()*F_*ps;
					if (abs(erro(0, 0)) < 0.2) {
						tempt.push_back(kptgt[i]); temps.push_back(kpsrc[i]);
						indices_inliers.push_back(i);
					}
				}
				kpsrc = temps; kptgt = tempt;

				// Segue so com os inliers dentre os best_kpts
				std::vector<cv::KeyPoint> temp_kptgt, temp_kpsrc;
				std::vector<cv::DMatch> temp_matches;
				for (auto i : indices_inliers) {
					temp_kptgt.push_back(best_kptgt[i]); temp_kpsrc.push_back(best_kpsrc[i]);
					temp_matches.push_back(best_matches[i]);
				}
				best_kptgt = temp_kptgt; best_kpsrc = temp_kpsrc;
				best_matches = temp_matches;
			}
			if (debug) {
				cv::Mat im1 = cv::imread(imagens_src[indices[frame0][frame1]], cv::IMREAD_COLOR);
				cv::Mat im2 = cv::imread(imagens_src[frame0], cv::IMREAD_COLOR);
				for (int i = 0; i < best_kpsrc.size(); i++) {
					int r = rand() % 255, b = rand() % 255, g = rand() % 255;
					circle(im1, cv::Point(best_kptgt[i].pt.x, best_kptgt[i].pt.y), 8, cv::Scalar(r, g, b), cv::FILLED, cv::LINE_8);
					circle(im2, cv::Point(best_kpsrc[i].pt.x, best_kpsrc[i].pt.y), 8, cv::Scalar(r, g, b), cv::FILLED, cv::LINE_8);
				}
				imwrite("C:/dataset3/im_tgt1.png", im1);
				imwrite("C:/dataset3/im_src1.png", im2);

			}
			//indices[frame0].push_back(frame1);
			bestKey[frame0].push_back(best_kpsrc);
			bestKey[frame0].push_back(best_kptgt);
			best_kptgt.clear();
			best_kpsrc.clear();


		}
	}
	auto finish_time1 = std::chrono::high_resolution_clock::now();
	auto time1= std::chrono::duration_cast<std::chrono::nanoseconds>(finish_time1 - start_time1).count() * 1e-9;
	std::cout << "matches " << time1 << "\n";
	return bestKey;
}











//std::vector<std::vector<std::vector<cv::KeyPoint>>> Utils::surf_matches_matrix_encontrar_melhor(std::vector<std::vector<  std::vector<cv::DMatch> >> matriz_matches, std::vector<cv::Mat>  descp_src, std::vector< std::vector<cv::KeyPoint> >  kpts_src, std::vector<std::string> imagens_src, std::vector<std::vector<int>> &indices){
//	// Matcher de FLANN
//	cv::Ptr<cv::DescriptorMatcher> matcher;
//	matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
//
//	// Para cada combinacao de imagens, fazer match e salvar quantidade final para ver qual
//	// a melhor depois
//	//#pragma omp parallel for
//	int currentCamera = 0, checkedCamera = 0, nMaxMatch = 0;
//	size_t frameInit = 0; int cont = 0;
//	for (size_t frame0 = currentCamera; frame0 < descp_src.size(); frame0++) {
//		frameInit = 0;
//		cont = 0;
//		//	if (frame0 == currentCamera)frameInit = checkedCamera + 1;
//		if (frame0 == frameInit)frameInit = checkedCamera + 1;
//
//		for (size_t frame1 = frameInit; frame1 < descp_src.size(); frame1++)
//		{
//
//			if (frame0 == frame1 && frame1 != descp_src.size() - 1)frame1 = frame1 + 1;
//			if (frame0 == descp_src.size() - 1 && frame1 == descp_src.size() - 1) {
//				break;
//			}
//			std::vector<std::vector<cv::DMatch>> matches;
//			std::vector<cv::DMatch> good_matches;
//			if (!descp_src[frame0].empty() && !descp_src[frame1].empty()) {
//				matcher->knnMatch(descp_src[frame0], descp_src[frame1], matches, 2);
//				for (size_t k = 0; k < matches.size(); k++) {
//					if (matches.at(k).size() >= 2) {
//						if (matches.at(k).at(0).distance < 0.7*matches.at(k).at(1).distance) // Se e bastante unica frente a segunda colocada
//							good_matches.push_back(matches.at(k).at(0));
//					}
//				}
//				if (good_matches.size() > 20)
//				{
//					// Filtrar keypoints repetidos
//					Utils::filtrar_matches_keypoints_repetidos(kpts_src[frame1], kpts_src[frame0], good_matches);
//					// Filtrar por matches que nao sejam muito horizontais
//					//filterMatchesLineCoeff(good_matches, kpts_tgt[i], kpts_src[j], imcols, DEG2RAD(50));
//
//					// Anota quantas venceram nessa combinacao
//					//matches_count(frame0, frame1) = good_matches.size();
//					matriz_matches.at(frame0).at(cont) = good_matches;
//
//				}
//				else {
//					for (int m = 0; m < good_matches.size(); m++) {
//						good_matches[m].trainIdx = -1;
//						good_matches[m].queryIdx = -1;
//					}
//
//					matriz_matches.at(frame0).at(cont) = good_matches;
//				}
//			}
//			cont++;
//		}
//	}
//
//	
//	bool debug = false;
//	std::vector<std::vector<std::vector<cv::KeyPoint>>> bestKey;
//	std::vector<cv::DMatch> best_matches;
//	std::vector<cv::KeyPoint> best_kptgt, best_kpsrc;
//	bestKey.resize(descp_src.size());
//	currentCamera = 0, checkedCamera = 0, nMaxMatch = 0;
//	frameInit = 0;
//	indices.resize(descp_src.size());
//	for (size_t frame0 = currentCamera; frame0 < descp_src.size(); frame0++)
//	{
//		frameInit = 0;
//		cont = 0;
//		//	if (frame0 == currentCamera)frameInit = checkedCamera + 1;
//		if (frame0 == frameInit)frameInit = checkedCamera + 1;
//
//		for (size_t frame1 = frameInit; frame1 < descp_src.size(); frame1++)
//		{
//
//			if (frame0 == frame1 && frame1 != descp_src.size() - 1)frame1 = frame1 + 1;
//			if (frame0 == descp_src.size() - 1 && frame1 == descp_src.size() - 1) {
//				break;
//			}
//			std::vector<cv::KeyPoint> curr_kpts_tgt = kpts_src[frame1], curr_kpts_src = kpts_src[frame0];
//			/*descp_tgt.clear(); descp_src.clear();*/
//			best_matches = matriz_matches.at(frame0).at(cont);
//			if (best_matches.size() > 20) {
//				for (auto m : best_matches) {
//					best_kptgt.emplace_back(curr_kpts_tgt[m.trainIdx]);
//					best_kpsrc.emplace_back(curr_kpts_src[m.queryIdx]);
//				}
//
//				
//				// Converter os pontos para o formato certo
//				std::vector<cv::Point2d> kptgt(best_kptgt.size()), kpsrc(best_kpsrc.size());
//#pragma omp parallel for
//				for (int i = 0; i < best_kptgt.size(); i++) {
//					kptgt[i] = best_kptgt[i].pt;
//					kpsrc[i] = best_kpsrc[i].pt;
//				}
//
//				// Calcular matriz fundamental
//				cv::Mat F = findFundamentalMat(kpsrc, kptgt); // Transformacao da src para a tgt
//				// Calcular pontos que ficam por conferencia da matriz F
//				Eigen::Matrix3d F_;
//				cv::cv2eigen(F, F_);
//				std::vector<cv::Point2d> tempt, temps;
//				std::vector<int> indices_inliers;
//				for (int i = 0; i < kpsrc.size(); i++) {
//					Eigen::Vector3d pt{ kptgt[i].x, kptgt[i].y, 1 }, ps = { kpsrc[i].x, kpsrc[i].y, 1 };
//					Eigen::MatrixXd erro = pt.transpose()*F_*ps;
//					if (abs(erro(0, 0)) < 0.2) {
//						tempt.push_back(kptgt[i]); temps.push_back(kpsrc[i]);
//						indices_inliers.push_back(i);
//					}
//				}
//				kpsrc = temps; kptgt = tempt;
//
//				// Segue so com os inliers dentre os best_kpts
//				std::vector<cv::KeyPoint> temp_kptgt, temp_kpsrc;
//				std::vector<cv::DMatch> temp_matches;
//				for (auto i : indices_inliers) {
//					temp_kptgt.push_back(best_kptgt[i]); temp_kpsrc.push_back(best_kpsrc[i]);
//					temp_matches.push_back(best_matches[i]);
//				}
//				best_kptgt = temp_kptgt; best_kpsrc = temp_kpsrc;
//				best_matches = temp_matches;
//
//				if (debug) {
//					cv::Mat im1 = cv::imread(imagens_src[frame1], cv::IMREAD_COLOR);
//					cv::Mat im2 = cv::imread(imagens_src[frame0], cv::IMREAD_COLOR);
//					for (int i = 0; i < best_kpsrc.size(); i++) {
//						int r = rand() % 255, b = rand() % 255, g = rand() % 255;
//						circle(im1, cv::Point(best_kptgt[i].pt.x, best_kptgt[i].pt.y), 8, cv::Scalar(r, g, b), cv::FILLED, cv::LINE_8);
//						circle(im2, cv::Point(best_kpsrc[i].pt.x, best_kpsrc[i].pt.y), 8, cv::Scalar(r, g, b), cv::FILLED, cv::LINE_8);
//					}
//					imwrite("C:/dataset3/im_tgt1.png", im1);
//					imwrite("C:/dataset3/im_src1.png", im2);
//
//				}
//
//			}
//			else
//			{
//				cv::KeyPoint temp;
//				temp.pt.x = -1;
//				temp.pt.y = -1;
//				for (auto m : best_matches) {
//					best_kptgt.emplace_back(temp);
//					best_kpsrc.emplace_back(temp);
//				}
//			}
//			
//			indices[frame0].push_back(frame1);
//			bestKey[frame0].push_back(best_kpsrc);
//			bestKey[frame0].push_back(best_kptgt);
//			best_kptgt.clear();
//			best_kpsrc.clear();
//			cont++;
//
//		}
//	}
//	return bestKey;
//}
