#include <ros/ros.h>
#include <string>
#include "../../libraries/include/sfm.h"

using namespace std;

///////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
  ros::init(argc, argv, "sfm_horizontal");
  ros::NodeHandle nh;
  ros::NodeHandle n_("~");
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
  vector<int> linhas_horizontal{4, 9, 12, 17, 20, 25, 28, 33, 36, 41, 44, 49};
  // Lendo pasta 1
  string arquivo_sfm = pasta_tgt + "cameras.sfm";
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
  arquivo_sfm = pasta_src + "cameras.sfm";
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

  /// Pegar pontos correspondentes em 2D para procurar na nuvem certa em 3D
  ///
  sfm.obter_correspondencias_3D_e_T();

  /// Transformacao estimada entre as nuvens
  ///

  ROS_INFO("Processo terminado.");
  ros::spinOnce();
  return 0;
}
