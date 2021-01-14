//  main.cc
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

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/mat.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include <opencv2/stitching/detail/blenders.hpp>
#include <cstdlib> //atexit
#include <iostream> //cerr, cout
#include "argument.hpp"
#include "gwo.hpp"
#include "GWOException.hpp"
/// Definicoes e namespaces

using namespace pcl;
using namespace pcl::io;
using namespace cv;
using namespace std;
using namespace Eigen;
using namespace cv::xfeatures2d;
typedef PointXYZRGBNormal PointTN;
typedef PointXYZRGB       PointC;
Argument *argument = nullptr;
GWO *gwo = nullptr;

void freeMemory() {
	if (argument) {
		delete argument;
		argument = nullptr;
	}
	if (gwo) {
		delete gwo;
		gwo = nullptr;
	}
}

int main(int argc, char **argv) {
	char* arguments[] = { "--dir", "-name","fob","-population_size","30","-iterations","1000","-debug","true" };

	argv = arguments;
	argc = sizeof(arguments) / sizeof(arguments[0]) - 1;

	//Localização arquivo NVM/SFM

	char* home;
	home = getenv("HOME");
	std::string pasta = "C:/Users/julia/Pictures/gerador_tomada2/";
	std::string arquivo_nvm = pasta + "cameras.sfm";
	ifstream nvm(arquivo_nvm);
	int contador_linhas = 1;
	std::vector<std::string> nomes_imagens, linhas, linhas_organizadas;
	std::string linha;
	int flag = 0;
	if (arquivo_nvm.substr(arquivo_nvm.find_last_of(".") + 1) == "sfm")
	{
		flag = 1;
		if (nvm.is_open()) {
			while (getline(nvm, linha)) {
				if (contador_linhas > 2 && linha.size() > 4)
					linhas.push_back(linha);

				contador_linhas++;
			}
		}
		else {
			printf("Arquivo de cameras nao encontrado. Desligando ...\n");
			return 0;
		}
	}
	else {

		if (nvm.is_open()) {
			while (getline(nvm, linha)) {
				if (contador_linhas > 3 && linha.size() > 4)
					linhas.push_back(linha);

				contador_linhas++;
			}
		}
		else {
			printf("Arquivo de cameras nao encontrado. Desligando ...\n");
			return 0;
		}
	}
	
	//Salvar o path das Imagens
	std::vector<cv::Mat>images;
	vector<string> imagens_src, imagens_tgt;
	// Supoe a esfera com resolucao em graus de tal forma - resolucao da imagem final
	float R = 1; // Raio da esfera [m]

	Vector2f foco, center;// foco e centro otico
	vector<Eigen::Vector3d>pontos; //ponto (pCenter - p1) para encontrar angulos entre imagens
	//Find position
	std::vector<std::vector<int> > pose = Utils::FindPoseRaw();

	if (arquivo_nvm.substr(arquivo_nvm.find_last_of(".") + 1) == "sfm")
	{
		for (int i = 0; i < linhas.size(); i++) {
			istringstream iss(linhas[i]);
			vector<string> splits(istream_iterator<string>{iss}, istream_iterator<string>());
			// Nome
			string nome_fim = splits[0].substr(splits[0].find_last_of('/') + 1, splits[0].size() - 1);
			imagens_src.push_back(pasta + nome_fim);
			Mat image = imread(pasta + nome_fim);
			// Foco
			foco << stof(splits[13]), stof(splits[14]);
			//centro otico			
			center << stof(splits[15]), stof(splits[16]);
			//pontos
			pontos.push_back(Utils::pointAngle(pose[i], R, center, image.cols, image.rows, foco));
		}
	}
//Features Matching - Keypoint and descriptors
	vector<vector<cv::KeyPoint>>  kpts_src;
	vector<cv::Mat>  descp_src;
	descp_src.resize(imagens_src.size());
	kpts_src.resize(imagens_src.size());
	

	//Find Fetaures
	Utils::calcular_features_surf(descp_src, kpts_src, imagens_src);
	
	//Find neighboring images

	int currentCamera = 0, checkedCamera = 0, nMaxMatch = 0;
	size_t frameInit = 0;
	std::vector<vector<int>>indices_vizinhos;
	indices_vizinhos.resize(imagens_src.size());
	for (size_t frame0 = currentCamera; frame0 < imagens_src.size(); frame0++) {
		frameInit = 0;

		double angle;//angle between images
		if (frame0 == frameInit)frameInit = checkedCamera + 1;

		for (size_t frame1 = frameInit; frame1 < imagens_src.size(); frame1++)
		{

			if (frame0 == frame1 && frame1 != imagens_src.size() - 1)frame1 = frame1 + 1;
			if (frame0 == imagens_src.size() - 1 && frame1 == imagens_src.size() - 1)
			{
				break;
			}

			double product = 0;//Scalar product 
			for (int i = 0; i < 3; i++) {
				product = product + (pontos[frame0][i] * pontos[frame1][i]);

			}
			//modulos
			double magA = pontos[frame0].norm();
			double magB = pontos[frame1].norm();
			angle = RAD2DEG(acos(product / (magA * magB))); // Angle

			if (angle < 40)
			{
				indices_vizinhos[frame0].push_back(frame1);

			}
		}
	}

	//Find Matches
	auto start_time = std::chrono::high_resolution_clock::now();

	// Ajustar matriz de quantidade de matches
	Eigen::MatrixXi matches_count = Eigen::MatrixXi::Zero(descp_src.size() - 1, descp_src.size() - 1);
	vector<vector<  vector<cv::DMatch> >> matriz_matches(descp_src.size());
	for (int i = 0; i < matriz_matches.size(); i++)
		matriz_matches.at(i).resize(indices_vizinhos[i].size());
	auto finish_time = std::chrono::high_resolution_clock::now();
	auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(finish_time - start_time).count() * 1e-9;
	std::cout << "Matches " << time << "\n";

	//Match pair of all images (Keypoints)
	std::vector<std::vector<std::vector<cv::KeyPoint>>> bestKey = Utils::surf_matches_matrix_encontrar_melhor(matriz_matches, descp_src, kpts_src, imagens_src, indices_vizinhos);

	float step_deg = 0.1; // [DEGREES]
	int raios_360 = int(360.0 / step_deg), raios_180 = raios_360 / 2.0; // Quantos raios sairao do centro para formar 360 e 180 graus de um circulo 2D

	//Size of final panoramic
	cv::Mat im360 = cv::Mat::zeros(cv::Size(raios_360, raios_180), CV_8UC3); // Imagem 360 ao final de todas as fotos passadas sem blending 
	
	try
	{
		atexit(freeMemory);
		argument = new Argument(argc, argv);
		argument->Parse();

		gwo = new GWO(argument->GetBenchmark(), argument->GetPopulationSize(), argument->GetIterations());
		(void)gwo->Evaluate(argument->IsDebug(), bestKey, imagens_src, im360, indices_vizinhos);
		
		std::cout << "Result:" << std::endl
			<< gwo << std::endl;

		freeMemory();
	}
	catch (GWOException &e) {
		std::cerr << "Grey wolf optimizer exception : " << e.what() << std::endl;
		return EXIT_FAILURE;
	}



	return EXIT_SUCCESS;
}
