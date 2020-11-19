#include <ros/ros.h>
#include <string>
#include "../../libraries/include/sfm.h"
#include "../../libraries/include/processcloud.h"

using namespace std;

void escrever_t_final(string ps, string pt, Matrix4f T){
  // Onde salvar
  string destino;
  char* home;
  home = getenv("HOME");
  destino = string(home) + "/Desktop/"+ ps + "/transformada_inicial.txt";
  ofstream ts(destino);
  if(ts.is_open()){
    ts << ps + " ";
    ts << pt + " ";
    for(int i=0; i<4; i++){
      for(int j=0; j<4; j++){
        ts << std::to_string(T(i, j)) + " ";
      }
    }
    ts << "\n";
  }
  ts.close();
}

///////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
  ros::init(argc, argv, "sfm_horizontal");
  ros::NodeHandle n_("~");
  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
  ROS_INFO("Iniciando o no de SFM para encontrar movimentacao relativa entre duas aquisicoes ...");

  /// Parametros de entrada
  ///
  string pasta_src, pasta_tgt;
  n_.param<string>("pasta_src", pasta_src, "src");
  n_.param<string>("pasta_tgt", pasta_tgt, "tgt");

  char* home;
  home = getenv("HOME");
  pasta_src = string(home)+"/Desktop/"+pasta_src+"/";
  pasta_tgt = string(home)+"/Desktop/"+pasta_tgt+"/";

  /// Classe de trabalho de SfM e de registro
  ///
  SFM sfm(pasta_src, pasta_tgt);
  sfm.set_debug(true);
  ProcessCloud pc(pasta_src);

  /// Iniciar leitura das imagens da vista anterior e da atual
  ///
  ROS_INFO("Lendo ambas as pastas os arquivos .sfm ...");
  vector<string> linhas_tgt, linhas_src;
  string linha;
  int contador_linhas = 0;
  vector<int> linhas_horizontal{4, 9, 12, 17, 20, 25, 28, 33, 36, 41, 44, 49, 52, 57, 60};
  // Lendo pasta 1
  string arquivo_sfm = pasta_tgt + "cameras_ok.sfm";
  ifstream sfm_tgt(arquivo_sfm);
  if (sfm_tgt.is_open()) {
      while (getline(sfm_tgt, linha)) {
          for(auto i:linhas_horizontal){
              if (contador_linhas > 2 && linha.size() > 4 && (contador_linhas+1) == i)
                  linhas_tgt.push_back(linha);
          }
          contador_linhas++;
      }
  }
  sfm_tgt.close();
  // Lendo pasta 2
  arquivo_sfm = pasta_src + "cameras_ok.sfm";
  contador_linhas = 0;
  ifstream sfm_src(arquivo_sfm);
  if (sfm_src.is_open()) {
      while (getline(sfm_src, linha)) {
          for(auto i:linhas_horizontal){
              if (contador_linhas > 2 && linha.size() > 4 && (contador_linhas+1) == i)
                  linhas_src.push_back(linha);
          }
          contador_linhas++;
      }
  }
  sfm_src.close();
  // Nomes das imagens e dados
  ROS_INFO("Lendo tudo e reduzindo dados ...");
  PointCloud<PointTN>::Ptr cloud_tgt (new PointCloud<PointTN>), cloud_src (new PointCloud<PointTN>);
  loadPLYFile<PointTN>(pasta_tgt+"acumulada.ply", *cloud_tgt);
  loadPLYFile<PointTN>(pasta_src+"acumulada.ply", *cloud_src);
  pc.filterRayCasting(cloud_tgt, 90.0, 180.0, 180.0, 360.0);
  pc.filterRayCasting(cloud_src, 90.0, 180.0, 180.0, 360.0);
  sfm.set_clouds(cloud_tgt, cloud_src);

  sfm.obter_dados(linhas_src, linhas_tgt);

  /// Calcular matriz de features e matches entre as imagens
  ///
  ROS_INFO("Calculando features ...");
  sfm.calcular_features_surf();
  ROS_INFO("Calculando matches e melhor combinacao de imagens ...");
  sfm.surf_matches_matrix_encontrar_melhor();

  /// Transformacao relativa por imagens
  ///
  ROS_INFO("Calculando pose relativa por imagens correspondentes ...");
  sfm.calcular_pose_relativa();

  /// Transformar o frames das duas nuvens com a transformacao relativa
  ///
  ROS_INFO("Pose da transformacao final ...");
  sfm.obter_transformacao_final_sfm();

  /// Transformacao estimada entre as nuvens
  ///
  ROS_INFO("Fechando com ICP ...");
  ros::Time tempo = ros::Time::now();
  Matrix4f Tfinal = sfm.icp(8.0, 200);
  ROS_INFO("Somando spaces ...");
  sfm.somar_spaces(0.10, 40);
  ROS_WARN("Tempo para o ICP e soma das nuvens: %.2f segundos.", (ros::Time::now() - tempo).toSec());

  /// Escrever arquivo com a transformada final
  ///
  n_.param<string>("pasta_src", pasta_src, "src");
  n_.param<string>("pasta_tgt", pasta_tgt, "tgt");
  escrever_t_final(pasta_src, pasta_tgt, Tfinal);

  ROS_INFO("Processo terminado.");
  ros::spinOnce();
  return 0;
}
