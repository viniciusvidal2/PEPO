#include <ros/ros.h>
#include <string>
#include "../../libraries/include/sfm.h"

using namespace std;

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
  Matrix4f Ticp = sfm.icp(4.0, 100);
  ROS_INFO("Somando spaces ...");
  sfm.somar_spaces(Ticp, 0.10, 200);
  ROS_WARN("Tempo para o ICP e soma das nuvens: %.2f segundos.", (ros::Time::now() - tempo).toSec());

  ROS_INFO("Processo terminado.");
  ros::spinOnce();
  return 0;
}
