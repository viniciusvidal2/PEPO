#include <iostream>
#include <ros/ros.h>
#include <string>
#include <math.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "../../libraries/include/processcloud.h"
#include "../../libraries/include/registerobjectoptm.h"

/// Namespaces
///
using namespace pcl;
using namespace pcl::io;
using namespace cv;
using namespace std;
using namespace message_filters;

/// Definiçoes
///
typedef PointXYZRGB       PointT ;
typedef PointXYZRGBNormal PointTN;

int main(int argc, char **argv)
{
  ros::init(argc, argv, "imagem_virtual");
  ros::NodeHandle nh;
  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
  ROS_INFO("Criar imagem virtual ...");

  // Ler a nuvem
  string pasta = "/home/vinicius/Desktop/gerador_tomada1/";
  PointCloud<PointTN>::Ptr nuvem  (new PointCloud<PointTN>);
  loadPLYFile(pasta+"acumulada.ply", *nuvem);

  // Ler NVM e so pegar as horizontais
  vector<string> linhas;
  string linha;
  int contador_linhas = 0;
  vector<int> linhas_horizontal{4, 9, 12, 17, 20, 25, 28, 33, 36, 41, 44, 49};
  // Lendo pasta 1
  string arquivo = pasta + "cameras.sfm";
  ifstream fotos_horizontais(arquivo);
  if (fotos_horizontais.is_open()) {
      while (getline(fotos_horizontais, linha)) {
          for(auto i:linhas_horizontal){
              if (contador_linhas > 2 && linha.size() > 4 && (contador_linhas+1) == i)
                  linhas.push_back(linha);
          }
          contador_linhas++;
      }
  }
  fotos_horizontais.close();
  // Dados de cada camera
  Vector2f foco, centro_otico;
  vector<string> imagens;
  vector<Matrix3f> rots;
  for (auto s:linhas) {
    istringstream iss(s);
    vector<string> splits(istream_iterator<string>{iss}, istream_iterator<string>());
    // Nome
    string nome_fim = splits[0].substr(splits[0].find_last_of('/') + 1, splits[0].size() - 1);
    imagens.push_back(pasta + nome_fim);

    // Rotation
    Matrix3f r;
    r << stof(splits[1]), stof(splits[2]), stof(splits[3]),
        stof(splits[4]), stof(splits[5]), stof(splits[6]),
        stof(splits[7]), stof(splits[8]), stof(splits[9]);
    rots.push_back(r);

    // Foco e centro para matriz K - igual, sempre mesma camera
    foco << stof(splits[13]), stof(splits[14]);
    centro_otico << stof(splits[15]), stof(splits[16]);
  }

  // Matriz intrinseca da camera
  Matrix3f K_cam;
  const int w = 1920, h = 1080;
  K_cam << foco(0),   0.0472, centro_otico(0),
            0.0157,  foco(1), centro_otico(1),
            0     ,   0     ,   1            ;
  int indice_imagem = 4;
//  K_cam << 375.29  ,   0.0472, 245.18,
//      0.0157, 374.50  , 142.36,
//      0     ,   0     ,   1   ;

  /// Rotacionar a gosto a nuvem para variar a imagem
  ///
  Quaternion<float> q(rots[indice_imagem]);
  transformPointCloudWithNormals(*nuvem, *nuvem, Vector3f::Zero(), q);
  PassThrough<PointTN> pass;
  pass.setInputCloud(nuvem);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(0, 200);
  pass.filter(*nuvem);

  // Separa a nuvem que se ve naquela vista
  PointCloud<PointTN>::Ptr temp (new PointCloud<PointTN>);
  for(size_t i = 0; i < nuvem->size(); i++){
    /// Pegar ponto em coordenadas normais
    MatrixXf X_(3, 1);
    X_ << nuvem->points[i].x,
        nuvem->points[i].y,
        nuvem->points[i].z;
    MatrixXf X = K_cam*X_;
    X = X/X(2, 0);
    /// Adicionando ponto na imagem se for o caso de projetado corretamente (para otimizacao de foco so funciona a menos de 65 metros)
    if(floor(X(0,0)) >= 0 && floor(X(0,0)) < w && floor(X(1,0)) >= 0 && floor(X(1,0)) < h)
      temp->push_back(nuvem->points[i]);
  }
  *nuvem  = *temp;
  temp->clear();

  // Varre nuvem que ficou atras da maior e menor distancia
  float mindist = 1000, maxdist = 0;
  vector<int> distancias_16bit(nuvem->size());
  for(size_t i=0; i < nuvem->size(); i++){
    float d = nuvem->points[i].z;
    if(d> maxdist)
      maxdist = d;
    if(d < mindist)
      mindist = d;
  }
  // Calcula distancias de cada ponto, regra sobre a distancia e atribui cor
#pragma omp parallel for
  for(size_t i=0; i < nuvem->size(); i++)
    distancias_16bit[i] = 65536 * (1 - nuvem->points[i].z/(maxdist - mindist));

//  savePLYFileBinary<PointTN>("/home/vinicius/Desktop/teste.ply", *nuvem);

  /// Projetar nuvem para criar imagem virtual
  ///

  octree::OctreePointCloudSearch<PointTN> octs(0.05);
  octs.setInputCloud(nuvem);
  octs.addPointsFromInputCloud();
  Mat fl(Size(w, h), CV_16UC1, Scalar(0));
  float fx = K_cam(0, 0), fy = K_cam(1, 1);
  float cx = K_cam(0, 2), cy = K_cam(1, 2);
#pragma omp parallel for
  for(int u=0; u<w; u++){
    for(int v=0; v<h; v++){
      // Calcular a direcao no frame da camera para eles
      Vector3f dirs;
      dirs << (u - cx)/(fx), (v - cy)/(fy), 1;
//      dirs = rots[indice_imagem].inverse() * dirs;
      // Aplicar ray casting para saber em que parte da nuvem vai bater
      octree::OctreePointCloudSearch<PointTN>::AlignedPointTVector aligns;
      octs.getIntersectedVoxelCenters(Vector3f::Zero(), dirs.normalized(), aligns, 1);
      // Se achou, colorir aquele pixel ou colocar distancia ali dele
      if(aligns.size() > 0){
        // Achar o ponto da nuvem que diz respeito a aquele voxel
        vector<int> id;
        vector<float> distances;
        octs.nearestKSearch(aligns[0], 1, id, distances);
        // Pegar os dados desse ponto
        if(id.size() > 0)
          fl.at<uint16_t>(Point(u, v)) = distancias_16bit[id[0]];
      }
    }
  }

  // Salva de uma vez a foto do laser
  vector<int> params;
  params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  params.push_back(9);
  imwrite(pasta+"Imagem_depth.png", fl, params);
  // Salva a de cor em JPG
  Mat fc = imread(imagens[indice_imagem]);
  imwrite(pasta+"Imagem_cor.jpg", fc);
  ROS_WARN("Terminado.");

}
