#include <ros/ros.h>
#include <iostream>
#include <string>
#include <math.h>
#include <sys/stat.h>
#include <ostream>

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

///Definicoes e namespaces
///
using namespace pcl;
using namespace pcl::io;
using namespace cv;
using namespace std;
using namespace Eigen;
using namespace cv::xfeatures2d;
typedef PointXYZRGBNormal PointTN;
typedef PointXYZRGB       PointC;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
Mat createMask(Mat img, vector<vector<Point>> contours, int k)
{
        //verificando os pontos pertencentes ao contorno
        vector<Point> pts;
        for (size_t cC = 0; cC < contours.size(); cC++)
        {
                for (size_t cP = 0; cP < contours[cC].size(); cP++)
                {
                        Point currentContourPixel = contours[cC][cP];
                        pts.push_back(currentContourPixel);

                }
        }
        //Encontrando Pontos maximos - linha
        auto val = std::minmax_element(pts.begin(), pts.end(), [](Point const& a, Point const& b) {
                return a.x < b.x;
        });
        int size = abs(val.first->x - val.second->x); //  tamanho

                                                                                                  //Blending Vertical
        auto valV = std::minmax_element(pts.begin(), pts.end(), [](Point const& a, Point const& b) {
                return a.y < b.y;
        });
        int sizeV = abs(valV.first->y - valV.second->y);
        if (k == 12)
        {
                for (int i = 0; i < img.cols; i++)
                {
                        for (int j = valV.first->y; j < valV.first->y + sizeV / 2 - 5; j++)
                        {
                                Vec3b color1(0, 0, 0);
                                img.at< Vec3b>(Point(i, j)) = color1;
                        }
                }
        }
        //Blending horizontal:
        //Possibilidades :
        // última Imagem Tem comportamento diferente
        if (k == 11)
        {
                //Se tiver pedaços de imagem nos 2 extremos da imagem
                if (val.first->x == 0 && val.second->x == img.cols - 1)
                {
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
                                                p.x = i;
                                                p.y = j;
                                                pts_cols.push_back(p);
                                        }
                                }
                        }
                        //Encontrando min e max em x e y
                        auto val1 = std::minmax_element(pts_cols.begin(), pts_cols.end(), [](Point const& a, Point const& b) {
                                return a.x < b.x;
                        });
                        auto val2 = std::minmax_element(pts_cols.begin(), pts_cols.end(), [](Point const& a, Point const& b) {
                                return a.y < b.y;
                        });
                        int size1 = abs(val1.first->x - val1.second->x);
                        int size2 = abs(val2.first->x - val1.first->x);

                        for (int i = val1.first->x; i < val1.first->x + size2 / 2 + 10; i++)
                        {
                                for (int j = 0; j < img.rows; j++)
                                {
                                        if (img.at< Vec3b>(Point(i, j))[0] != 0 && img.at< Vec3b>(Point(i, j))[1] != 0 && img.at< Vec3b>(Point(i, j))[2] != 0) {
                                                Vec3b color1(0, 0, 0);
                                                img.at< Vec3b>(Point(i, j)) = color1;
                                        }
                                }
                        }
                        for (int i = val1.second->x - size2 / 2 - 10; i < val1.second->x; i++)
                        {
                                for (int j = 0; j < img.rows; j++)
                                {
                                        if (img.at< Vec3b>(Point(i, j))[0] != 0 && img.at< Vec3b>(Point(i, j))[1] != 0 && img.at< Vec3b>(Point(i, j))[2] != 0) {
                                                Vec3b color1(0, 0, 0);
                                                img.at< Vec3b>(Point(i, j)) = color1;
                                        }
                                }
                        }
                }
                //Caso contrario tira pedaço dos dois extremos para não aparecer as bordas no blending
                else {
                        for (int i = val.first->x; i < val.first->x + size / 4; i++)
                        {
                                for (int j = 0; j < img.rows; j++)
                                {
                                        if (img.at< Vec3b>(Point(i, j))[0] != 0 && img.at< Vec3b>(Point(i, j))[1] != 0 && img.at< Vec3b>(Point(i, j))[2] != 0) {

                                                Vec3b color1(0, 0, 0);
                                                img.at< Vec3b>(Point(i, j)) = color1;

                                        }
                                }
                        }
                        for (int i = val.second->x - size / 4; i < val.second->x; i++)
                        {
                                for (int j = 0; j < img.rows; j++)
                                {
                                        if (img.at< Vec3b>(Point(i, j))[0] != 0 || img.at< Vec3b>(Point(i, j))[1] != 0 || img.at< Vec3b>(Point(i, j))[2] != 0) {

                                                Vec3b color1(0, 0, 0);
                                                img.at< Vec3b>(Point(i, j)) = color1;
                                        }
                                }
                        }
                }
        }
        // Outras imagens
        if (k != 11 && k != 12) {
                //Extremos de imagens
                if (val.first->x == 0 && val.second->x == img.cols - 1)
                {
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
                        auto val1 = std::minmax_element(pts_cols.begin(), pts_cols.end(), [](Point const& a, Point const& b) {
                                return a.x < b.x;
                        });
                        auto val2 = std::minmax_element(pts_cols.begin(), pts_cols.end(), [](Point const& a, Point const& b) {
                                return a.y < b.y;
                        });
                        int size1 = abs(val1.first->x - val1.second->x);
                        int size2 = abs(val2.first->x - val1.first->x);
                        for (int i = val1.first->x; i < val1.first->x + size2 / 2 + 10; i++)
                        {
                                for (int j = 0; j < img.rows; j++)
                                {
                                        if (img.at< Vec3b>(Point(i, j))[0] != 0 && img.at< Vec3b>(Point(i, j))[1] != 0 && img.at< Vec3b>(Point(i, j))[2] != 0) {
                                                Vec3b color1(0, 0, 0);
                                                img.at< Vec3b>(Point(i, j)) = color1;
                                        }
                                }
                        }

                }
                else
                {
                        vector<Point> pt;
                        //A imagem não se encontra exatamente nos extremos mas pertence estão distantes
                        if (size > img.cols / 2)
                        {
                                for (int i = val.first->x; i < img.cols / 2; i++)
                                {
                                        for (int j = 0; j < img.rows; j++)
                                        {
                                                if (img.at< Vec3b>(Point(i, j))[0] != 0 && img.at< Vec3b>(Point(i, j))[1] != 0 && img.at< Vec3b>(Point(i, j))[2] != 0) {
                                                        Point p;
                                                        p.x = i;							p.y = j;
                                                        pt.push_back(p);

                                                }
                                        }
                                }
                                auto val4 = std::minmax_element(pt.begin(), pt.end(), [](Point const& a, Point const& b) {
                                        return a.y < b.y;				});
                                int size2 = abs(val4.first->x - val4.second->x);

                                for (int i = val4.first->x + size2 / 2; i < val.second->x + 1; i++)
                                {
                                        for (int j = 0; j < img.rows; j++)
                                        {
                                                if (img.at< Vec3b>(Point(i, j))[0] != 0 && img.at< Vec3b>(Point(i, j))[1] != 0 && img.at< Vec3b>(Point(i, j))[2] != 0) {

                                                        Vec3b color1(0, 0, 0);
                                                        img.at< Vec3b>(Point(i, j)) = color1;
                                                }
                                        }
                                }
                        }
                        // Imagem normal sem ser cortada - pega um pouco mais da metade e tira;
                        else
                        {
                                for (int i = val.first->x + size / 2 + 5; i < val.second->x + 1; i++)
                                {
                                        for (int j = 0; j < img.rows; j++)
                                        {
                                                if (img.at< Vec3b>(Point(i, j))[0] != 0 || img.at< Vec3b>(Point(i, j))[1] != 0 || img.at< Vec3b>(Point(i, j))[2] != 0) {
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

        Mat a_pyramid[4];
        Mat b_pyramid[4];

        Mat mask[4];

        a_pyramid[0] = a;
        b_pyramid[0] = b;

        int w = a.cols, h = a.rows; //

        Mat teste(h, w, CV_32FC3, Scalar(0, 0, 0));
        teste.convertTo(teste, CV_32F, 1.0 / 255.0);
        mask[0] = teste;

        //Contorno imagem 1
        Mat src_gray;
        cvtColor(a, src_gray, CV_BGR2GRAY);
        src_gray.convertTo(src_gray, CV_8UC3, 255);
        Mat dst(src_gray.rows, src_gray.cols, CV_8UC3, Scalar::all(0));
        vector<vector<Point> > contours; // Vector for storing contour
        vector<Vec4i> hierarchy;

        findContours(src_gray, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE); // Find the contours in the image

        for (int i = 0; i < contours.size(); i++) // iterate through each contour.
        {
                Scalar color(255, 255, 255);
                drawContours(dst, contours, i, color, CV_FILLED);
        }
//	imwrite("C:/dataset3/Imagem1.png", dst);

        //Contorno imagem 2

        Mat src_gray1;
        cvtColor(b, src_gray1, CV_BGR2GRAY);
        src_gray1.convertTo(src_gray1, CV_8UC3, 255);
        Mat dst1(src_gray1.rows, src_gray1.cols, CV_8UC3, Scalar::all(0));
        vector<vector<Point> > contours1; //
        vector<Vec4i> hierarchy1;

        findContours(src_gray1, contours1, hierarchy1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE); // Encontrando contorno
        for (int i = 0; i < contours1.size(); i++)
        {

                Scalar color(255, 255, 255);
                drawContours(dst1, contours1, i, color, CV_FILLED);
        }

        //Parte comum entre as imagens
        Mat out(dst1.rows, dst1.cols, CV_8UC3, Scalar::all(0));
        bitwise_and(dst, dst1, out);

        /////////////Contorno Parte comum
        Mat src_gray3;
        cvtColor(out, src_gray3, CV_BGR2GRAY);
        src_gray3.convertTo(src_gray3, CV_8UC3, 255);
        Mat dst3(src_gray3.rows, src_gray3.cols, CV_8UC3, Scalar::all(0));
        vector<vector<Point> > contours3; // Vector for storing contour
        vector<Vec4i> hierarchy3;

        findContours(src_gray3, contours3, hierarchy3, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE); // Find the contours in the image

        for (int i = 0; i < contours3.size(); i++) // iterate through each contour.
        {
                Scalar color(255, 255, 255);
                drawContours(dst3, contours3, i, color, -1, 8, hierarchy3, 0, Point());
        }

        //Encontrando a máscara
        Mat mask_out = createMask(out, contours3, k);

        for (int x = 0; x < dst.cols; x++)
        {
                for (int y = 0; y < dst.rows; y++)
                {
                        for (int c = 0; c < 3; c++)
                        {
                                dst.at<Vec3b>(Point(x, y))[c] = dst.at<Vec3b>(Point(x, y))[c] - mask_out.at<Vec3b>(Point(x, y))[c];
                        }
                }
        }

        dst.convertTo(dst, CV_32FC3, 1.0 / 255.0);
        mask[0] = dst;

        //Filtro Gaussiano e o resultado é uma imagem reduzida com a metade do tamanho de cada dimensão
        for (int i = 1; i < level_num; ++i)
        {
                int wp = a_pyramid[i - 1].rows / 2;
                int hp = a_pyramid[i - 1].cols / 2;
                int wp1 = b_pyramid[i - 1].rows / 2;
                int hp1 = b_pyramid[i - 1].cols / 2;

                Mat a_pic, b_pic, mask_pic;

                a_pic = GaussianFilter(a_pyramid[i - 1], 2);
                b_pic = GaussianFilter(b_pyramid[i - 1], 2);
                mask_pic = GaussianFilter(mask[i - 1], 2);

                Mat new_a(wp, hp, CV_32FC3, cv::Scalar(0, 0, 0));
                Mat new_b(wp1, hp1, CV_32FC3, cv::Scalar(0, 0, 0));
                Mat new_mask(wp1, hp1, CV_32FC3, cv::Scalar(0, 0, 0));

                omp_set_dynamic(0);
#pragma omp parallel for num_threads(100)
                for (int r = 0; r < new_a.cols; r++) {
                        for (int c = 0; c < new_a.rows; c++)
                        {
                                for (int k = 0; k < 3; k++)
                                {
                                        new_a.at<Vec3f>(Point(r, c))[k] = a_pic.at<Vec3f>(Point(2 * r, 2 * c))[k];
                                        new_b.at<Vec3f>(Point(r, c))[k] = b_pic.at<Vec3f>(Point(2 * r, 2 * c))[k];
                                        new_mask.at<Vec3f>(Point(r, c))[k] = mask_pic.at<Vec3f>(Point(2 * r, 2 * c))[k];
                                }
                        }
                }
                a_pyramid[i] = new_a;
                b_pyramid[i] = new_b;
                mask[i] = new_mask;
        }

        //Computando a piramide Laplaciana das imagens e da máscara
        //Expande as imagens, fazendo elas maiores de forma que seja possivel subtrai-las
        //Subtrair cada nivel da piramide

        for (int i = 0; i < level_num - 1; ++i) {
                int wp = a_pyramid[i].cols;
                int hp = a_pyramid[i].rows;

                cv::Mat dst_a, dst_b;
                cv::resize(a_pyramid[i + 1], dst_a, cv::Size(wp, hp));
                cv::resize(b_pyramid[i + 1], dst_b, cv::Size(wp, hp));

                omp_set_dynamic(0);
#pragma omp parallel for num_threads(100)
                for (int x = 0; x < a_pyramid[i].cols; x++)
                {
                        for (int y = 0; y < a_pyramid[i].rows; y++)
                        {
                                for (int k = 0; k < 3; k++) {
                                        a_pyramid[i].at<Vec3f>(Point(x, y))[k] -= dst_a.at<Vec3f>(Point(x, y))[k];
                                        b_pyramid[i].at<Vec3f>(Point(x, y))[k] -= dst_b.at<Vec3f>(Point(x, y))[k];

                                }
                        }
                }
        }

        // Criação da imagem "misturada" em cada nivel da piramide
        Mat blend_pyramid[4];
        for (int i = 0; i < level_num; ++i)
        {

                Mat fin(a_pyramid[i].rows, a_pyramid[i].cols, CV_32FC3, cv::Scalar(0, 0, 0));
                blend_pyramid[i] = fin;

                omp_set_dynamic(0);
#pragma omp parallel for num_threads(100)
                for (int x = 0; x < blend_pyramid[i].cols; x++)
                {
                        for (int y = 0; y < blend_pyramid[i].rows; y++)
                        {
                                for (int k = 0; k < 3; k++) {
                                        blend_pyramid[i].at<Vec3f>(Point(x, y))[k] = a_pyramid[i].at<Vec3f>(Point(x, y))[k] * mask[i].at<Vec3f>(Point(x, y))[k] + b_pyramid[i].at<Vec3f>(Point(x, y))[k] * (1.0 - mask[i].at<Vec3f>(Point(x, y))[k]);
                                }
                        }
                }
        }
        //Reconstruir a imagem completa
//O nível mais baixo da nova pirâmide gaussiana dá o resultado final
        Mat expand = blend_pyramid[level_num - 1];
        for (int i = level_num - 2; i >= 0; --i)
        {
                cv::resize(expand, expand, cv::Size(blend_pyramid[i].cols, blend_pyramid[i].rows));
                omp_set_dynamic(0);
#pragma omp parallel for num_threads(100)
                for (int x = 0; x < blend_pyramid[i].cols; x++)
                {
                        for (int y = 0; y < blend_pyramid[i].rows; y++)
                        {
                                expand.at<Vec3f>(Point(x, y))[0] = blend_pyramid[i].at<Vec3f>(Point(x, y))[0] + expand.at<Vec3f>(Point(x, y))[0];
                                expand.at<Vec3f>(Point(x, y))[1] = blend_pyramid[i].at<Vec3f>(Point(x, y))[1] + expand.at<Vec3f>(Point(x, y))[1];
                                expand.at<Vec3f>(Point(x, y))[2] = blend_pyramid[i].at<Vec3f>(Point(x, y))[2] + expand.at<Vec3f>(Point(x, y))[2];

                                for (int c = 0; c < 3; c++)
                                {
                                        if (expand.at<Vec3f>(Point(x, y))[c] > 255)
                                        {
                                                expand.at<Vec3f>(Point(x, y))[c] = 255;
                                        }
                                        else if (expand.at<Vec3f>(Point(x, y))[c] < 0)
                                        {
                                                expand.at<Vec3f>(Point(x, y))[c] = 0;

                                        }
                                }
                        }
                }
        }
//	cout << "Blending \n";
        return expand;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix4f calculateCameraPose(Quaternion<float> q, Vector3f &C, int i) {
	Matrix3f r = q.matrix();
	Vector3f t = C;

	Matrix4f T = Matrix4f::Identity();
	T.block<3, 3>(0, 0) = r.transpose(); T.block<3, 1>(0, 3) = t;

	return T;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void doTheThing(float sd, Vector3f p2, Vector3f p4, Vector3f p5, Mat im, Mat &im360){
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
            float lat =  RAD2DEG(acos( ponto[1]/ponto.norm()));
            float lon = -RAD2DEG(atan2(ponto[2], ponto[0]   ));
            lon = (lon < 0) ? lon += 360.0 : lon; // Ajustar regiao do angulo negativo, mantendo o 0 no centro da imagem

            // Pelas coordenadas, estimar posicao do pixel que deve sair na 360 final e pintar - da forma como criamos a esfera
            int u = int(lon/sd);
            u = (u >= im360.cols) ? im360.cols - 1 : u; // Nao deixar passar do limite de colunas por seguranca
            u = (u < 0) ? 0 : u;
            int v = im360.rows - 1 - int(lat/sd);
            v = (v >= im360.rows) ? im360.rows - 1 : v; // Nao deixar passar do limite de linhas por seguranca
            v = (v < 0) ? 0 : v;
            // Pintar a imagem final com as cores encontradas
            im360.at<Vec3b>(Point(u, v)) = im.at<Vec3b>(Point(j, im.rows - 1 - i));
        }
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
        // Coisa do ros, apagar
        ros::init(argc, argv, "otimiza_bat");
        ros::NodeHandle nh;

	//Localização arquivo NVM
	char* home;
	home = getenv("HOME");
        std::string pasta = "/home/vinicius/Desktop/gerador_tomada1/";
	std::string arquivo_nvm = pasta + "cameras.nvm";

	ifstream nvm(arquivo_nvm);
	int contador_linhas = 1;
	vector<Quaternion<float>> rots;
	vector<Vector3f> Cs;
	vector<std::string> nomes_imagens, linhas, linhas_organizadas;
	std::string linha;
	printf("Abrindo e lendo arquivo NVM ...\n");
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
	rots.resize(linhas.size()); Cs.resize(linhas.size()), nomes_imagens.resize(linhas.size());
	float foco;
        // Para cada imagem, obter valores
	for (int i = 0; i < linhas.size(); i++) {
		istringstream iss(linhas[i]);
		vector<string> splits(istream_iterator<string>{iss}, istream_iterator<string>());
		// Nome
                string nome_fim = splits[0].substr(splits[0].find_last_of('\\') + 1, splits[0].size() - 1);
		nomes_imagens[i] = pasta + nome_fim;
		// Foco
                foco = stof(splits[1]);//*(0.6667); // AH MARIA!
		// Quaternion
		Quaternion<float> q;
		q.w() = stof(splits[2]); q.x() = stof(splits[3]); q.y() = stof(splits[4]); q.z() = stof(splits[5]);
		rots[i] = q;
		// Centro
		Vector3f C(stof(splits[6]), stof(splits[7]), stof(splits[8]));
		Cs[i] = C;
	}
	/// Ler todas as nuvens, somar e salvar
	struct stat buffer;

        // Supoe a esfera com resolucao em graus de tal forma - resolucao da imagem final
        float R = 5; // Raio da esfera [m]
        // Angulos para lat e lon, 360 de range para cada, resolucao a definir no step_deg
        float step_deg = 0.1; // [DEGREES]
        int raios_360 = int(360.0 / step_deg), raios_180 = raios_360 / 2.0; // Quantos raios sairao do centro para formar 360 e 180 graus de um circulo 2D

        /// Para cada imagem
        Mat imagem_esferica[100];
        Mat result1 = Mat::zeros(Size(raios_360, raios_180), CV_8UC3);
        Mat result2 = Mat::zeros(Size(raios_360, raios_180), CV_8UC3);
        Mat result3 = Mat::zeros(Size(raios_360, raios_180), CV_8UC3);
        Mat result4 = Mat::zeros(Size(raios_360, raios_180), CV_8UC3);
        Mat anterior1 = Mat::zeros(Size(raios_360, raios_180), CV_8UC3);
        Mat anterior2 = Mat::zeros(Size(raios_360, raios_180), CV_8UC3);
        Mat anterior3 = Mat::zeros(Size(raios_360, raios_180), CV_8UC3);
        Mat anterior4 = Mat::zeros(Size(raios_360, raios_180), CV_8UC3);

        int contador = 0;
        Mat im360 = Mat::zeros(Size(raios_360, raios_180), CV_8UC3); // Imagem 360 ao final de todas as fotos passadas
        ros::Time tempo = ros::Time::now();
        printf("Processando cada foto, sem print nenhum pra ir mais rapido ...\n");
        for (int i = 0; i < nomes_imagens.size(); i++)
        {
            printf("Processando foto %d...\n", i+1);
                // Ler a imagem a ser usada
		Mat image = imread(nomes_imagens[i]);
                if (image.cols < 3)
                        ROS_ERROR("Imagem nao foi encontrada, checar NVM ...");

                // Calcular a vista da camera pelo Rt inverso - rotacionar para o nosso mundo, com Z para cima
		Matrix4f T = calculateCameraPose(rots[i], Cs[i], i);
		// Definir o foco em dimensoes fisicas do frustrum
                float F = R;
		Vector3f C = Cs[i];
		double minX, minY, maxX, maxY;
		maxX = F * (float(image.cols) / (2.0*foco));
		minX = -maxX;
		maxY = F * (float(image.rows) / (2.0*foco));
		minY = -maxY;
		// Calcular os 4 pontos do frustrum
		/*
								origin of the camera = p1
								p2--------p3
								|          |
								|  pCenter |<--- Looking from p1 to pCenter
								|          |
								p5--------p4
		*/
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
                doTheThing(step_deg, p2.block<3, 1>(0, 0), p4.block<3, 1>(0, 0), p5.block<3, 1>(0, 0), image, im360);

                imagem_esferica[i] = im360;

                if (i == 0) {
                        index = 0;
                        anterior1 = imagem_esferica[i];
                        anterior1.convertTo(anterior1, CV_32F, 1.0 / 255.0);
                }
                if (i > 0 && i < 12)
                {
                        imagem_esferica[i].convertTo(imagem_esferica[i], CV_32F, 1.0 / 255.0);
                        result1 = multiband_blending(anterior1, imagem_esferica[i], index);
                        anterior1 = result1;

                }
                if (i == 12) {
                        index = 0;
                        anterior2 = imagem_esferica[i];
                        anterior2.convertTo(anterior2, CV_32F, 1.0 / 255.0);
                }

                if (i > 12 && i < 24)
                {
                        imagem_esferica[i].convertTo(imagem_esferica[i], CV_32F, 1.0 / 255.0);
                        result2 = multiband_blending(anterior2, imagem_esferica[i], index);
                        anterior2 = result2;
                        //result2.convertTo(result2, CV_8UC3, 255);
                }
                if (i == 24)
                {
                        index = 0;

                        anterior3 = imagem_esferica[i];
                        anterior3.convertTo(anterior3, CV_32F, 1.0 / 255.0);
                }

                if (i > 24 && i < 36)
                {
                        imagem_esferica[i].convertTo(imagem_esferica[i], CV_32F, 1.0 / 255.0);
                        result3 = multiband_blending(anterior3, imagem_esferica[i], index);
                        anterior3 = result3;
                        //result3.convertTo(result3, CV_8UC3, 255);

                }
                if (i == 36) {
                        index = 0;
                        anterior4 = imagem_esferica[i];
                        anterior4.convertTo(anterior4, CV_32F, 1.0 / 255.0);
                }

                if (i > 36 && i < 48)
                {
                        imagem_esferica[i].convertTo(imagem_esferica[i], CV_32F, 1.0 / 255.0);
                        result4 = multiband_blending(anterior4, imagem_esferica[i], index);
                        anterior4 = result4;
                        //result4.convertTo(result4, CV_8UC3, 255);
                }
                index++;
	} // Fim do for imagens;

        // Resultado Final - Juntando os blendings horizontais
        Mat result;
        index = 12;
        result = multiband_blending(result4, result3, index);
        result = multiband_blending(result, result2, index);
        result = multiband_blending(result, result1, index);
        result.convertTo(result, CV_8UC3, 255);
        imwrite(pasta + "imagem_esferica_Blending.png", result);

        ROS_WARN("Tempo para processar: %.2f segundos.", (ros::Time::now() - tempo).toSec());
	// Salvando imagem esferica final
        imwrite(pasta + "imagem_esferica_result.png", im360);
	printf("Processo finalizado.");

	return 0;
}

