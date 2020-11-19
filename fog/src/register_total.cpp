#include <ros/ros.h>
#include "../../libraries/include/registerobjectoptm.h"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "register_total");
  ros::NodeHandle nh;
  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

  RegisterObjectOptm ro;

  // Pasta geral
  char* home;
  home = getenv("HOME");
  // Checando se ha a pasta spaces, senao criar
  string pasta = string(home)+"/Desktop/SANTOS_DUMONT_2/sala/";

  // Ler arquivo de transformadas
  ROS_INFO("Lendo arquivo de transformadas ...");
  vector<Matrix4f> ts;
  vector<string> scans;
  string linha_atual, scan_base;
  ifstream transfs;
  transfs.open((pasta+"transformadas_inicial.txt").c_str());
  if(transfs.is_open()){
    while(getline(transfs, linha_atual)){
      istringstream ss(linha_atual);
      vector<string> results((std::istream_iterator<string>(ss)), std::istream_iterator<string>());
      scan_base = results.at(1);
      scans.emplace_back(results.at(0));
      Matrix4f T;
      T << stof(results.at( 2)), stof(results.at( 3)), stof(results.at( 4)), stof(results.at( 5)),
           stof(results.at( 6)), stof(results.at( 7)), stof(results.at( 8)), stof(results.at( 9)),
           stof(results.at(10)), stof(results.at(11)), stof(results.at(12)), stof(results.at(13)),
           stof(results.at(14)), stof(results.at(15)), stof(results.at(16)), stof(results.at(17));
      ts.emplace_back(T);
    }
  }

  // Iniciar nuvem acumulada com nuvem base
  ROS_INFO("Inicia nuvem base ...");
  string caminho_nuvem = pasta + scan_base + "/acumulada.ply";
  PointCloud<PointTN>::Ptr acc (new PointCloud<PointTN>);
  loadPLYFile<PointTN>(caminho_nuvem, *acc);

  // Para cada pasta filha, ler nuvem, transformar, evitar vizinhos e somar
  ROS_INFO("Soma cada nuvem ...");
  PointCloud<PointTN>::Ptr temp (new PointCloud<PointTN>);
  for(int i=0; i<scans.size(); i++){
    ROS_INFO("Somando nuvem %d de %zu ...", i+1, scans.size());
    caminho_nuvem = pasta + scans[i] + "/acumulada.ply";
    loadPLYFile<PointTN>(caminho_nuvem, *temp);
    transformPointCloudWithNormals(*temp, *temp, ts[i]);
//    cout << "Realizando icp ..." << endl;
//    ro.icp(acc, temp, 0, 15);
    cout << "Buscando vizinhos ..." << endl;
    ro.searchNeighborsKdTree(temp, acc, 0.1, 40);
    *acc += *temp;
    temp->clear();
  }

  ROS_INFO("Retira outliers ...");
  StatisticalOutlierRemoval<PointTN> sor;
  sor.setInputCloud(acc);
  sor.setMeanK(50);
  sor.setStddevMulThresh(2.5);
  sor.filter(*acc);

  // Salvar nuvem final
  ROS_INFO("Salvando registro final ...");
  savePLYFileBinary(pasta+"registro_final.ply", *acc);

  ROS_WARN("Tudo finalizado.");

  return 0;
}
