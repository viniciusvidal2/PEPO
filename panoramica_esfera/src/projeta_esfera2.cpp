//#include <ros/ros.h>
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



#include "RANSAC_algo.hpp"
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>

///Definicoes e namespaces
///
using namespace pcl;
using namespace pcl::io;
using namespace cv;
using namespace std;
using namespace Eigen;
using namespace cv::xfeatures2d;
typedef PointXYZRGBNormal PointT;
typedef PointXYZRGB       PointC;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Mat GaussianFilter(Mat ori, float sigma)
{
	// A good size for kernel support
	int ksz = (int)((sigma - 0.35) / 0.15);
	if (ksz < 0) ksz = 1;
	if (ksz % 2 == 0) ksz++; // Must be odd

	Size kernel_size(ksz, ksz);

	Mat fin;
	GaussianBlur(ori, fin, kernel_size, sigma, sigma);

	return fin;
}
bool findMinMaxRows(Point const& a, Point const& b)
{
	return a.y < b.y;
}
bool findMinMaxcols(Point const& a, Point const& b)
{
	return a.x < b.x;
}
Mat createMask(Mat img, vector<vector<Point>> contours, int k)
{
	//verificando os pontos pertencentes ao contorno
	vector<Point> pts;

	for (int cC = 0; cC < contours.size(); cC++)
	{
		for (int cP = 0; cP < contours[cC].size(); cP++)
		{
			Point currentContourPixel = contours[cC][cP];
			pts.push_back(currentContourPixel);

		}
	}

	//Blending Vertical

	if (k == 73)
	{
		auto valV = std::minmax_element(pts.begin(), pts.end(), findMinMaxRows);
		int sizeV = abs(valV.first->y - valV.second->y);
#pragma omp parallel for
		for (int i = 0; i < img.cols; i++)
		{
			for (int j = valV.first->y; j < valV.first->y + sizeV / 2 - 11; j++)
			{
				Vec3b color1(0, 0, 0);
				img.at< Vec3b>(Point(i, j)) = color1;
			}
		}
	}

	//Encontrando Pontos maximos e mininmos - linha
	auto val = std::minmax_element(pts.begin(), pts.end(), findMinMaxcols);
	int size = abs(val.first->x - val.second->x); //  tamanho

	//Blending horizontal - Possibilidades :
	// �ltima Imagem Tem comportamento diferente
	if (k == 11)
	{
		//Se tiver peda�os de imagem nos 2 extremos da imagem
		if (val.first->x == 0 && val.second->x == img.cols - 1)
		{
#pragma omp parallel for
			for (int i = val.first->x; i < img.cols / 2; i++)
			{
				for (int j = 0; j < img.rows; j++)
				{

					Vec3b color1(0, 0, 0);
					img.at< Vec3b>(Point(i, j)) = color1;


				}
			}

			vector<Point>pts_cols, points;
			for (int i = val.second->x / 2; i < val.second->x; i++)
			{
				for (int j = 0; j < img.rows; j++)
				{
					if (img.at< Vec3b>(Point(i, j))[0] != 0 && img.at< Vec3b>(Point(i, j))[1] != 0 && img.at< Vec3b>(Point(i, j))[2] != 0) {

						Point p;
						p.x = i; p.y = j;
						pts_cols.push_back(p);
					}
				}
			}
			//Encontrando min e max em x e y
			auto valH = minmax_element(pts_cols.begin(), pts_cols.end(), findMinMaxcols);
			auto valVert = minmax_element(pts_cols.begin(), pts_cols.end(), findMinMaxRows);

			int size1 = abs(valH.first->x - valH.second->x);
#pragma omp parallel for
			for (int i = valH.first->x; i < valH.first->x + size1 / 3; i++)
			{
				for (int j = valH.first->y; j < valH.second->y; j++)
				{
					Vec3b color1(0, 0, 0);
					img.at< Vec3b>(Point(i, j)) = color1;

				}
			}
#pragma omp parallel for
			for (int i = valH.first->x - size1 / 3; i < valH.second->x; i++)
			{
				for (int j = 0; j < img.rows; j++)
				{

					Vec3b color1(0, 0, 0);
					img.at< Vec3b>(Point(i, j)) = color1;

				}
			}
		}
		//Caso contrario tira peda�o dos dois extremos para n�o aparecer as bordas no blending
		else {
#pragma omp parallel for
			for (int i = val.first->x; i < val.first->x + size / 3; i++)
			{
				for (int j = 0; j < img.rows; j++)
				{
					Vec3b color1(0, 0, 0);
					img.at< Vec3b>(Point(i, j)) = color1;

				}
			}
#pragma omp parallel for
			for (int i = val.second->x - size / 3; i < val.second->x; i++)
			{
				for (int j = 0; j < img.rows; j++)
				{

					Vec3b color1(0, 0, 0);
					img.at< Vec3b>(Point(i, j)) = color1;
				}
			}
		}
	}
	// Outras imagens
	if (k != 11 && k != 73)
	{

		if (k == 9)
		{

			vector<Point>pts_Teste, points;
			if (val.first->x == 0 && val.second->x == img.cols - 1)
			{
				for (int i = val.first->x; i < img.cols / 2; i++)
				{
					for (int j = 0; j < img.rows; j++)
					{
						if (img.at< Vec3b>(Point(i, j))[0] != 0 && img.at< Vec3b>(Point(i, j))[1] != 0 && img.at< Vec3b>(Point(i, j))[2] != 0) {

							Point p;
							p.x = i; p.y = j;
							pts_Teste.push_back(p);
						}
					}
				}
				auto valTeste = std::minmax_element(pts_Teste.begin(), pts_Teste.end(), findMinMaxcols);
				int sizeTeste = abs(valTeste.first->x - valTeste.second->x);

#pragma omp parallel for
				for (int i = valTeste.first->x + sizeTeste / 2 + 5; i < valTeste.second->x + 1; i++)
				{
					for (int j = 0; j < img.rows; j++)
					{


						Vec3b color1(0, 0, 0);
						img.at< Vec3b>(Point(i, j)) = color1;


					}
				}

				vector<Point>pts_cols, points;

				for (int i = val.second->x / 2; i < val.second->x; i++)
				{
					for (int j = 0; j < img.rows; j++)
					{
						if (img.at< Vec3b>(Point(i, j))[0] != 0 && img.at< Vec3b>(Point(i, j))[1] != 0 && img.at< Vec3b>(Point(i, j))[2] != 0) {
							Point p;
							p.x = i; p.y = j;
							pts_cols.push_back(p);
						}
					}
				}
				//Enocntrando min e max em x e y
				auto val1 = minmax_element(pts_cols.begin(), pts_cols.end(), findMinMaxcols);
				auto val2 = minmax_element(pts_cols.begin(), pts_cols.end(), findMinMaxRows);
				int size2 = abs(val2.first->x - val1.first->x);
#pragma omp parallel for
				for (int i = val1.first->x; i < val1.first->x + size2 / 3; i++)
				{
					for (int j = 0; j < img.rows; j++)
					{
						Vec3b color1(0, 0, 0);
						img.at< Vec3b>(Point(i, j)) = color1;

					}
				}
			}
		}
		else
		{
			//Extremos de imagens
			vector<Point>pts_Teste, points;
			if (val.first->x == 0 && val.second->x == img.cols - 1)
			{
#pragma omp parallel for
				for (int i = val.first->x; i < img.cols / 2; i++)
				{
					for (int j = 0; j < img.rows; j++)
					{
						if (img.at< Vec3b>(Point(i, j))[0] != 0 && img.at< Vec3b>(Point(i, j))[1] != 0 && img.at< Vec3b>(Point(i, j))[2] != 0) {

							Vec3b color1(0, 0, 0);
							img.at< Vec3b>(Point(i, j)) = color1;
						}
					}
				}


				vector<Point>pts_cols, points;
				for (int i = val.second->x / 2; i < val.second->x; i++)
				{
					for (int j = 0; j < img.rows; j++)
					{
						if (img.at< Vec3b>(Point(i, j))[0] != 0 && img.at< Vec3b>(Point(i, j))[1] != 0 && img.at< Vec3b>(Point(i, j))[2] != 0) {
							Point p;
							p.x = i; p.y = j;
							pts_cols.push_back(p);
						}
					}
				}
				//Enocntrando min e max em x e y
				auto val1 = minmax_element(pts_cols.begin(), pts_cols.end(), findMinMaxcols);
				auto val2 = minmax_element(pts_cols.begin(), pts_cols.end(), findMinMaxRows);
				int size2 = abs(val2.first->x - val1.first->x);
#pragma omp parallel for
				for (int i = val1.first->x; i < val1.first->x + size2 / 3; i++)
				{
					for (int j = 0; j < img.rows; j++)
					{

						Vec3b color1(0, 0, 0);
						img.at< Vec3b>(Point(i, j)) = color1;

					}
				}
			}

			else
			{
				vector<Point> pt;
				//A imagem n�o se encontra exatamente nos extremos mas  est�o distantes
				if (size > img.cols / 2)
				{
					for (int i = val.first->x; i < img.cols / 2; i++)
					{
						for (int j = 0; j < img.rows; j++)
						{
							if (img.at< Vec3b>(Point(i, j))[0] != 0 && img.at< Vec3b>(Point(i, j))[1] != 0 && img.at< Vec3b>(Point(i, j))[2] != 0) {
								Point p;
								p.x = i;	p.y = j;
								pt.push_back(p);

							}
						}
					}
					auto val4 = std::minmax_element(pt.begin(), pt.end(), findMinMaxRows);
					int size2 = abs(val4.first->x - val4.second->x);
#pragma omp parallel for
					for (int i = val4.first->x + size2 / 2; i < val.second->x + 1; i++)
					{
						for (int j = 0; j < img.rows; j++)
						{
							Vec3b color1(0, 0, 0);
							img.at< Vec3b>(Point(i, j)) = color1;

						}
					}
				}
				// Imagem normal sem ser cortada - pega um pouco mais da metade e tira;
				else
				{
#pragma omp parallel for
					for (int i = val.first->x + size / 2 + 5; i < val.second->x + 1; i++)
					{
						for (int j = 0; j < img.rows; j++)
						{

							Vec3b color1(0, 0, 0);
							img.at< Vec3b>(Point(i, j)) = color1;

						}
					}
				}
			}
		}

	}

	return img;

}
Mat multiband_blending(Mat &a, const Mat &b, int k) {

	int level_num = 4;//numero de niveis
	/*int w = a.cols, h = a.rows;*/

	Mat* a_pyramid = new Mat[level_num];
	Mat* b_pyramid = new Mat[level_num];
	Mat* mask = new Mat[level_num];


	a_pyramid[0] = a;
	b_pyramid[0] = b;

	//Contorno imagem 1
	Mat src_gray;
	cvtColor(a, src_gray, CV_BGR2GRAY);
	src_gray.convertTo(src_gray, CV_8UC3, 255);
	Mat dst(src_gray.rows, src_gray.cols, CV_8UC3, Scalar::all(0));
	vector<vector<Point> > contours; // Vector for storing contour
	vector<Vec4i> hierarchy;

	findContours(src_gray, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE); // Find the contours in the image
#pragma omp parallel for
	for (int i = 0; i < contours.size(); i++) // iterate through each contour.
	{
		Scalar color(255, 255, 255);
		drawContours(dst, contours, i, color, CV_FILLED);
	}


	Mat src_gray1;
	cvtColor(b, src_gray1, CV_BGR2GRAY);
	src_gray1.convertTo(src_gray1, CV_8UC3, 255);
	Mat dst1(src_gray1.rows, src_gray1.cols, CV_8UC3, Scalar::all(0));
	vector<vector<Point> > contours1; //
	vector<Vec4i> hierarchy1;

	findContours(src_gray1, contours1, hierarchy1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE); // Encontrando contorno
#pragma omp parallel for
	for (int i = 0; i < contours1.size(); i++)
	{
		Scalar color(255, 255, 255);
		drawContours(dst1, contours1, i, color, CV_FILLED);
	}

	//Parte comum entre as imagens
	Mat out(src_gray1.rows, src_gray1.cols, CV_8UC3, Scalar::all(0));
	bitwise_and(dst1, dst, out);
	/*imwrite("C:/dataset3/areaComum.png", out);*/
	/////////////Contorno Parte comum
	Mat src_gray3;
	//src_gray3 = out;
	cvtColor(out, src_gray3, CV_BGR2GRAY);
	src_gray3.convertTo(src_gray3, CV_8UC3, 255);
	Mat dst3(src_gray3.rows, src_gray3.cols, CV_8UC3, Scalar::all(0));
	vector<vector<Point> > contours3; // Vector for storing contour
	vector<Vec4i> hierarchy3;

	findContours(src_gray3, contours3, hierarchy3, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE); // Find the contours in the image
#pragma omp parallel for
	for (int i = 0; i < contours3.size(); i++) // iterate through each contour.
	{
		Scalar color(255, 255, 255);
		drawContours(dst3, contours3, i, color, -1, 8, hierarchy3, 0, Point());
	}

	//Encontrando a m�scara
	
	Mat mask_out = createMask(dst3, contours3, k);
	
	cv::subtract(dst, mask_out, dst);
		
	dst.convertTo(dst, CV_32FC3, 1.0 / 255.0);
	mask[0] = dst;


	//Filtro Gaussiano e o resultado � uma imagem reduzida com a metade do tamanho de cada dimens�o

	for (int i = 1; i < level_num; ++i)
	{
		/*int wp = a_pyramid[i - 1].rows / 2;
		int hp = a_pyramid[i - 1].cols / 2;*/

		Mat new_a, new_b, new_mask;

		// a imagem � inicialmente desfocada e depois reduzida
		pyrDown(a_pyramid[i - 1], new_a, Size(a_pyramid[i - 1].cols / 2, a_pyramid[i - 1].rows / 2));
		pyrDown(b_pyramid[i - 1], new_b, Size(a_pyramid[i - 1].cols / 2, a_pyramid[i - 1].rows / 2));
		pyrDown(mask[i - 1], new_mask, Size(a_pyramid[i - 1].cols / 2, a_pyramid[i - 1].rows / 2));

		a_pyramid[i] = new_a;
		b_pyramid[i] = new_b;
		mask[i] = new_mask;	
	}

	//Computando a piramide Laplaciana das imagens e da m�scara
	//Expande as imagens, fazendo elas maiores de forma que seja possivel subtrai-las
	//Subtrair cada nivel da pir�mide


	for (int i = 0; i < level_num - 1; ++i) {
	/*	int wp = a_pyramid[i].cols;
		int hp = a_pyramid[i].rows;*/

		cv::Mat dst_a, dst_b, new_a, new_b;

		cv::resize(a_pyramid[i + 1], dst_a, cv::Size(a_pyramid[i].cols, a_pyramid[i].rows));
		cv::resize(b_pyramid[i + 1], dst_b, cv::Size(a_pyramid[i].cols, a_pyramid[i].rows));

		cv::subtract(a_pyramid[i], dst_a, a_pyramid[i]);
		cv::subtract(b_pyramid[i], dst_b, b_pyramid[i]);
	}


	// Cria��o da imagem "misturada" em cada n�vel da piramide


	Mat* blend_pyramid = new Mat[level_num];
	for (int i = 0; i < level_num; ++i)
	{
		Mat fin(a_pyramid[i].rows, a_pyramid[i].cols, CV_32FC3, cv::Scalar(0, 0, 0));
		blend_pyramid[i] = fin;

		//Mat A = a_pyramid[i].mul(mask[i]);

		Mat antiMask = Scalar(1.0, 1.0, 1.0) - mask[i];

		//Mat B = b_pyramid[i].mul(antiMask);
		
		//blend_pyramid[i] = A+ B;
		blend_pyramid[i] = a_pyramid[i].mul(mask[i]) + b_pyramid[i].mul(antiMask);
		/*Mat fin(a_pyramid[i].rows, a_pyramid[i].cols, CV_32FC3, cv::Scalar(0, 0, 0));
		blend_pyramid[i] = fin;*/

//	
//#pragma omp parallel for 
//		for (int x = 0; x < blend_pyramid[i].cols; x++)
//		{
//			for (int y = 0; y < blend_pyramid[i].rows; y++)
//			{
//				for (int k = 0; k < 3; k++) {
//					blend_pyramid[i].at<Vec3f>(Point(x, y))[k] = a_pyramid[i].at<Vec3f>(Point(x, y))[k] * mask[i].at<Vec3f>(Point(x, y))[k] + b_pyramid[i].at<Vec3f>(Point(x, y))[k] * (1.0 - mask[i].at<Vec3f>(Point(x, y))[k]);
//				}
//			}
//		}

	}

	//Reconstruir a imagem completa
	//O n�vel mais baixo da nova pir�mide gaussiana d� o resultado final

	Mat expand = blend_pyramid[level_num - 1];
	for (int i = level_num - 2; i >= 0; --i)
	{
		cv::resize(expand, expand, cv::Size(blend_pyramid[i].cols, blend_pyramid[i].rows));

		add(blend_pyramid[i], expand, expand);

	}
	
	
	return expand;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix4f calculateCameraPose(Quaternion<float> q, Vector3f &C, int i) {
	Matrix3f r = q.matrix();
	Vector3f t = C;

	Matrix4f T = Matrix4f::Identity();
	T.block<3, 3>(0, 0) = r.transpose(); T.block<3, 1>(0, 3) = t;

	return T;
}
Matrix4f calculateCameraPoseSFM(Matrix3f rot, Vector3f &C, int i) {
	Matrix3f r = rot;
	Vector3f t = C;

	Matrix4f T = Matrix4f::Identity();
	T.block<3, 3>(0, 0) = r.transpose(); T.block<3, 1>(0, 3) = t;
	
	
	return T;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void doTheThing(float sd, Vector3f p2, Vector3f p4, Vector3f p5, Mat im, Mat &img, Mat &im360) {
	// A partir de frustrum, calcular a posicao de cada pixel da imagem fonte em XYZ, assim como quando criavamos o plano do frustrum
	Vector3f hor_step, ver_step; // Steps pra se andar de acordo com a resolucao da imagem
	hor_step = (p4 - p5) / float(im.cols);
	ver_step = (p2 - p5) / float(im.rows);
#pragma omp parallel for
	for (int i = 0; i < im.rows; i++) { // Vai criando o frustrum a partir dos cantos da imagem
		for (int j = 0; j < im.cols; j++) {
			Vector3f ponto;
			ponto = p5 + hor_step * j + ver_step * i;

			// Calcular latitude e longitude da esfera de volta a partir de XYZ
			float lat = RAD2DEG(acos(ponto[1] / ponto.norm()));
			float lon = -RAD2DEG(atan2(ponto[2], ponto[0]));
			lon = (lon < 0) ? lon += 360.0 : lon; // Ajustar regiao do angulo negativo, mantendo o 0 no centro da imagem

			// Pelas coordenadas, estimar posicao do pixel que deve sair na 360 final e pintar - da forma como criamos a esfera
			int u = int(lon / sd);
			u = (u >= im360.cols) ? im360.cols - 1 : u; // Nao deixar passar do limite de colunas por seguranca
			u = (u < 0) ? 0 : u;
			int v = im360.rows - 1 - int(lat / sd);
			v = (v >= im360.rows) ? im360.rows - 1 : v; // Nao deixar passar do limite de linhas por seguranca
			v = (v < 0) ? 0 : v;
			// Pintar a imagem final com as cores encontradas
			im360.at<Vec3b>(Point(u, v)) = im.at<Vec3b>(Point(j, im.rows - 1 - i));
			img.at<Vec3b>(Point(u, v)) = im.at<Vec3b>(Point(j, im.rows - 1 - i));
		}
	}
}
Mat RootSift(Mat image, vector < KeyPoint> kps)
{
	Ptr<DescriptorExtractor> extractor = SIFT::create();
	Mat descs;
	extractor->compute(image, kps, descs);
	for (int i = 0; i < descs.rows; ++i) {
		// Perform L1 normalization
		normalize(descs.row(i), descs.row(i), 1.0, 0.0, cv::NORM_L1);
		//descs.row(i) /= (cv::norm(descs.row(i), cv::NORM_L1) + eps);
	}
	// Perform sqrt on the whole descriptor matrix
	sqrt(descs, descs);
	return (descs);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) 
{
	// Coisa do ros, apagar
	/*ros::init(argc, argv, "otimiza_bat");
	ros::NodeHandle nh;*/

	//Localiza��o arquivo NVM

	char* home;
	home = getenv("HOME");
	std::string pasta = "C:/Users/julia/Pictures/gerador_tomada2/";
	std::string arquivo_nvm = pasta + "cameras.sfm";

	ifstream nvm(arquivo_nvm);
	int contador_linhas = 1;
	vector<Quaternion<float>> rots;
	vector<Vector3f> Cs;
	vector<Matrix3f> rot;
	vector<std::string> nomes_imagens, linhas, linhas_organizadas;
	std::string linha;
	int flag = 0;
	printf("Abrindo e lendo arquivo NVM ...\n");
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
	// Reorganizando nvm para facilitar para mim o blending -  Colocando em linhas
	int i = 0;
	int cont = 0;

while (cont < 6) {

		linhas_organizadas.push_back(linhas[i]);
		i = i + 7;

		linhas_organizadas.push_back(linhas[i]);
		i = i + 1;

		cont++;
	}


	i = 1;
	cont = 0;

	while (cont < 6)
	{
		linhas_organizadas.push_back(linhas[i]);
		i = i + 5;

		linhas_organizadas.push_back(linhas[i]);
		i = i + 3;
		cont++;

	}

	i = 2;
	cont = 0;

	while (cont < 6)
	{
		linhas_organizadas.push_back(linhas[i]);
		i = i + 3;

		linhas_organizadas.push_back(linhas[i]);
		i = i + 5;
		cont++;

	}

	i = 3;
	cont = 0;

	while (cont < 6)
	{
		linhas_organizadas.push_back(linhas[i]);
		i = i + 1;

		linhas_organizadas.push_back(linhas[i]);
		i = i + 7;
		cont++;

	}

	linhas = linhas_organizadas;
	int index = 0;
	// Alocar nos respectivos vetores
	Vector2f	center;
	rots.resize(linhas.size()); Cs.resize(linhas.size()), nomes_imagens.resize(linhas.size()), rot.resize(linhas.size());// center.resize(linhas.size());
	Vector2f foco;

	// Para cada imagem, obter valores
	if (arquivo_nvm.substr(arquivo_nvm.find_last_of(".") + 1) == "sfm")
	{
		for (int i = 0; i < linhas.size(); i++) {
			istringstream iss(linhas[i]);
			vector<string> splits(istream_iterator<string>{iss}, istream_iterator<string>());
			// Nome
			string nome_fim = splits[0].substr(splits[0].find_last_of('/') + 1, splits[0].size() - 1);
			nomes_imagens[i] = pasta + nome_fim;

			//rotation
			Matrix3f r;
			r << stof(splits[1]), stof(splits[2]), stof(splits[3]),
				stof(splits[4]), stof(splits[5]), stof(splits[6]),
				stof(splits[7]), stof(splits[8]), stof(splits[9]);
			rot[i] = r;
			//transla��o
			Vector3f t(stof(splits[10]), stof(splits[11]), stof(splits[12]));

			// Foco
			foco << stof(splits[13]), stof(splits[14]);

			// Centro
			Vector2f C(stof(splits[15]), stof(splits[16]));
			center << stof(splits[15]), stof(splits[16]);
			Cs[i] = t;
		}
	}
	else {
		for (int i = 0; i < linhas.size(); i++) {
			istringstream iss(linhas[i]);
			vector<string> splits(istream_iterator<string>{iss}, istream_iterator<string>());
			// Nome
			string nome_fim = splits[0].substr(splits[0].find_last_of('/') + 1, splits[0].size() - 1);
			nomes_imagens[i] = pasta + nome_fim;
			// Foco
			foco << stof(splits[1]), stof(splits[1]);//*(0.6667); // AH MARIA!
			// Quaternion
			Quaternion<float> q;
			q.w() = stof(splits[2]); q.x() = stof(splits[3]); q.y() = stof(splits[4]); q.z() = stof(splits[5]);
			rots[i] = q;
			// Centro
			Vector3f C(stof(splits[6]), stof(splits[7]), stof(splits[8]));
			Cs[i] = C;
		}
	}
	/// Ler todas as nuvens, somar e salvar
	struct stat buffer;

	// Supoe a esfera com resolucao em graus de tal forma - resolucao da imagem final
	float R = 5; // Raio da esfera [m]
	// Angulos para lat e lon, 360 de range para cada, resolucao a definir no step_deg
	float step_deg = 0.05; // [DEGREES]
	int raios_360 = int(360.0 / step_deg), raios_180 = raios_360 / 2.0; // Quantos raios sairao do centro para formar 360 e 180 graus de um circulo 2D

	/// Para cada imagem
	int img_num = linhas.size();
	Mat* imagem_esferica = new Mat[img_num];
	std::vector <cv::Mat> vect;
	
	Mat result1 = Mat::zeros(Size(raios_360, raios_180), CV_8UC3);
	Mat result2 = Mat::zeros(Size(raios_360, raios_180), CV_8UC3);
	Mat result3 = Mat::zeros(Size(raios_360, raios_180), CV_8UC3);
	Mat result4 = Mat::zeros(Size(raios_360, raios_180), CV_8UC3);
	Mat anterior = Mat::zeros(Size(raios_360, raios_180), CV_8UC3);
	/*Mat anterior2 = Mat::zeros(Size(raios_360, raios_180), CV_8UC3);
	Mat anterior3 = Mat::zeros(Size(raios_360, raios_180), CV_8UC3);
	Mat anterior4 = Mat::zeros(Size(raios_360, raios_180), CV_8UC3);*/

	int contador = 0;
	Mat im360 = Mat::zeros(Size(raios_360, raios_180), CV_8UC3); // Imagem 360 ao final de todas as fotos passadas
	//ros::Time tempo = ros::Time::now();
	auto start = chrono::steady_clock::now();
	printf("Processando cada foto, sem print nenhum pra ir mais rapido ...\n");
	for (int i = 0; i < nomes_imagens.size(); i++)
	{
		printf("Processando foto %d...\n", i + 1);
		// Ler a imagem a ser usada
		Mat image = imread(nomes_imagens[i]);
		if (image.cols < 3)
			cout << ("Imagem nao foi encontrada, checar NVM ...");
		
//		//Mat params = (Mat_<double>(1, 5) << 0.0723, -0.1413, -0.0025 ,- 0.0001, 0.0000);
//		//Mat K = (Mat_<double>(3, 3) << 1427.099976, 0.0, 987.900024,
//		//	0.0, 1449.400024, 579.400024,
//		//	0.0, 0.0, 1.0);
//		//// Tirar distorcao da imagem - fica melhor sim o resultado ou proximo
//		//Mat temp;
//		//undistort(image, temp, K, params);
//		//temp.copyTo(image);
		// Calcular a vista da camera pelo Rt inverso - rotacionar para o nosso mundo, com Z para cima
		Matrix4f T;
		if (flag == 1) {
			T = calculateCameraPoseSFM(rot[i], Cs[i], i);
		}
		else {

			T = calculateCameraPose(rots[i], Cs[i], i);
		}
		
				// Definir o foco em dimensoes fisicas do frustrum
		float F = R;
		Vector3f C = Cs[i];
		double minX, minY, maxX, maxY;
		maxX = F * (float(image.cols) / (2.0*foco[0]));
		minX = -maxX;
		maxY = F * (float(image.rows) / (2.0*foco[1]));
		minY = -maxY;
//		// Calcular os 4 pontos do frustrum
//		/*
//								origin of the camera = p1
//								p2--------p3
//								|          |
//								|  pCenter |<--- Looking from p1 to pCenter
//								|          |
//								p5--------p4
//		*/
		Vector4f p, p1, p2, p3, p4, p5, pCenter;
		p << 0, 0, 0, 1;
		p1 = T * p;
		p << minX, minY, F, 1;
		p2 = T * p;
		p << maxX, minY, F, 1;
		p3 = T * p;
		p << maxX, maxY, F, 1;
		p4 = T * p;
		p << minX, maxY, F, 1;
		p5 = T * p;
		p << 0, 0, F, 1;
		pCenter = T * p;
		// Fazer tudo aqui nessa nova funcao, ja devolver a imagem esferica inclusive nesse ponto
		imagem_esferica[i] = Mat::zeros(Size(raios_360, raios_180), CV_8UC3); // Imagem 360 ao final de todas as fotos passadas
		
		doTheThing(step_deg, p2.block<3, 1>(0, 0), p4.block<3, 1>(0, 0), p5.block<3, 1>(0, 0), image, im360, imagem_esferica[i]);		//imagem_esferica[i] = im360;
		//imwrite("C:/dataset3/im360.png", im360 );
		if (i == 0) {
			index = 0;
			anterior = imagem_esferica[i];
			anterior.convertTo(anterior, CV_32F, 1.0 / 255.0);
		}
		if (i > 0 && i < 12)
		{
			//imwrite("C:/dataset3/atual.png", imagem_esferica[i]);
			imagem_esferica[i].convertTo(imagem_esferica[i], CV_32F, 1.0 / 255.0);
			result1 = multiband_blending(anterior, imagem_esferica[i], index);
			anterior = result1;
			imwrite("C:/dataset3/imagem_esferica_result.png", result1 * 255);
		}
		if (i == 12) {
			index = 0;
			anterior.release();
			anterior = imagem_esferica[i];
			anterior.convertTo(anterior, CV_32F, 1.0 / 255.0);
		}

		if (i > 12 && i < 24)
		{
			//imwrite("C:/dataset3/atual.png", imagem_esferica[i]);
			imagem_esferica[i].convertTo(imagem_esferica[i], CV_32F, 1.0 / 255.0);
			result2 = multiband_blending(anterior, imagem_esferica[i], index);
			anterior = result2;
			//imwrite("C:/dataset3/imagem_esferica_result.png", result2 * 255);

		}
		if (i == 24)
		{
			index = 0;
			anterior.release();
			anterior = imagem_esferica[i];
			anterior.convertTo(anterior, CV_32F, 1.0 / 255.0);
		}

		if (i > 24 && i < 36)
		{
			//imwrite("C:/dataset3/atual.png", imagem_esferica[i]);
			imagem_esferica[i].convertTo(imagem_esferica[i], CV_32F, 1.0 / 255.0);
			result3 = multiband_blending(anterior, imagem_esferica[i], index);
			anterior = result3;
			//imwrite("C:/dataset3/imagem_esferica_result.png", result3 * 255);


		}
		if (i == 36) {
			index = 0;
			anterior.release();
			anterior = imagem_esferica[i];
			anterior.convertTo(anterior, CV_32F, 1.0 / 255.0);
		}

		if (i > 36 && i < 48)
		{
			//imwrite("C:/dataset3/atual.png", imagem_esferica[i]);
			imagem_esferica[i].convertTo(imagem_esferica[i], CV_32F, 1.0 / 255.0);
			result4 = multiband_blending(anterior, imagem_esferica[i], index);
			anterior = result4;
			//imwrite("C:/dataset3/imagem_esferica_result.png", result4 * 255);

		//	//result4.convertTo(result4, CV_8UC3, 255);
		}
		index++;
	} // Fim do for imagens;

		 //Resultado Final - Juntando os blendings horizontais
	Mat result;
	index = 73;
	result = multiband_blending(result4, result3, index);
	result = multiband_blending(result, result2, index);
	result = multiband_blending(result, result1, index);
	result.convertTo(result, CV_8UC3, 255);
	imwrite(pasta + "imagem_esferica_Blending_Res005.png", result);

	auto end = chrono::steady_clock::now();
	//// Store the time difference between start and end
	auto diff = end - start;
	cout << chrono::duration <double, milli>(diff).count() << " ms" << endl;
	//ROS_WARN("Tempo para processar: %.2f segundos.", (ros::Time::now() - tempo).toSec());
	// Salvando imagem esferica final
	imwrite(pasta + "imagem_esferica_result_Homografia.png", im360);
	printf("Processo finalizado.");

	return 0;

}
