/// Includes
///
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
typedef sync_policies::ApproximateTime<sensor_msgs::PointCloud2, nav_msgs::Odometry> MySyncPolicy;

/// Variaveis globais
///
// Limites em raw e deg para os servos de pan e tilt
double raw_min_pan = 35, raw_max_pan = 4077;
double deg_min_pan =  3, deg_max_pan =  358;
float raw_min_tilt = 2595.0 , raw_hor_tilt = 2280.0, raw_max_tilt = 1595.0 ;
float deg_min_tilt =   27.68, deg_hor_tilt =    0.0, deg_max_tilt =  -60.20;
float raw_deg_pan, deg_raw_pan, raw_deg_tilt, deg_raw_tilt;
// Publicador de imagem, nuvem parcial e odometria
ros::Publisher cl_pub;
ros::Publisher od_pub;
// Classe de processamento de nuvens
ProcessCloud *pc;
RegisterObjectOptm *roo;
// Nuvens de pontos acumulada e anterior
PointCloud<PointTN>::Ptr acc;
PointCloud<PointTN>::Ptr anterior;
PointCloud<PointTN>::Ptr parcial;
PointCloud<PointTN>::Ptr parcial_esq_anterior; // Parte esquerda ja filtrada e armazenada para otimizar o algoritmo
// Valor do servo naquele instante em variavel global para ser acessado em varios callbacks
int pan, tilt;
// Nome da pasta que vamos trabalhar
string pasta;
// Poses das cameras para aquela aquisicao [DEG]
vector<float> pan_cameras, pitch_cameras, roll_cameras;
// Vetor com linhas do arquivo NVM
vector<string> linhas_sfm;
// Quantos tilts estao ocorrendo por pan, e contador de quantos ja ocorreram
int ntilts = 4, contador_nuvens = 0;
// Parametros para filtros
float voxel_size, depth;
int filter_poli;
// Braco do centro ao laser
Vector3f off_laser{0, 0, 0.056};

// Vetores para resultados de tempo
vector<float > tempos_transito_msg, tempos_filtra_cor, tempos_octree, tempos_demaisfiltros, tempos_normais, tempos_vizinhos_lastview, tempos_rcast;
vector<size_t> pontos_demaisfiltros, pontos_covariancia, pontos_kdtree, pontos_rcast;

///////////////////////////////////////////////////////////////////////////////////////////
int deg2raw(float deg, string motor){
  if(motor == "pan")
    return int(deg*raw_deg_pan);// int((deg - deg_min_pan )*raw_deg_pan  + raw_min_pan);
  else
    return int((deg - deg_min_tilt)*raw_deg_tilt + raw_min_tilt);
}
float raw2deg(int raw, string motor){
  if(motor == "pan")
    return float(raw)*deg_raw_pan;// (float(raw) - raw_min_pan )*deg_raw_pan  + deg_min_pan;
  else
    return (float(raw) - raw_max_tilt)*deg_raw_tilt + deg_max_tilt;
}
string create_folder(string p){
    struct stat buffer;
    for(int i=1; i<200; i++){ // Tentar criar ate 200 pastas - impossivel
        string nome_atual = p + std::to_string(i);
        if(stat(nome_atual.c_str(), &buffer)){ // Se nao existe a pasta
            mkdir(nome_atual.c_str(), 0777);
            return nome_atual;
        }
    }
}
///////////////////////////////////////////////////////////////////////////////////////////
void saveTempJuliano(PointCloud<PointTN>::Ptr cloud, int n, float p, float t){
  // Salva a parcial daquela vista
  ROS_INFO("Salvando a nuvem do juliano ...");
  if(n < 10){
    pc->saveCloud(cloud, "pj_00"+std::to_string(n));
  } else if(n < 100) {
    pc->saveCloud(cloud, "pj_0" +std::to_string(n));
  } else {
    pc->saveCloud(cloud, "pj_"  +std::to_string(n));
  }
}
///////////////////////////////////////////////////////////////////////////////////////////
void saveTimeFiles(){
  // Abre os arquivos todos
  ofstream t_msg(pasta+"tempos_msg.txt");
  ofstream t_cor(pasta+"tempos_cor.txt");
  ofstream t_octree(pasta+"tempos_octree.txt");
  ofstream t_df(pasta+"tempos_df.txt");
  ofstream t_normais(pasta+"tempos_normais.txt");
  ofstream t_vlv(pasta+"tempos_vizinhoslastview.txt");
  ofstream t_rcast(pasta+"tempos_raycasting.txt");

  // Escreve uma linha para cada valor
  if(t_msg.is_open()){
    for(auto t:tempos_transito_msg){
      t_msg << t;
      t_msg << "\n";
    }
  }
  if(t_cor.is_open()){
    for(auto t:tempos_filtra_cor){
      t_cor << t;
      t_cor << "\n";
    }
  }
  if(t_octree.is_open()){
    for(auto t:tempos_octree){
      t_octree << t;
      t_octree << "\n";
    }
  }
  if(t_df.is_open()){
    for(auto t:tempos_demaisfiltros){
      t_df << t;
      t_df << "\n";
    }
  }
  if(t_normais.is_open()){
    for(auto t:tempos_normais){
      t_normais << t;
      t_normais << "\n";
    }
  }
  if(t_vlv.is_open()){
    for(auto t:tempos_vizinhos_lastview){
      t_vlv << t;
      t_vlv << "\n";
    }
  }
  if(t_rcast.is_open()){
    for(auto t:tempos_rcast){
      t_rcast << t;
      t_rcast << "\n";
    }
  }

  // Fecha arquivos
  t_msg.close(); t_cor.close(); t_octree.close();
  t_df.close(); t_normais.close(); t_vlv.close(); t_rcast.close();
}
///////////////////////////////////////////////////////////////////////////////////////////
void savePointFiles(){
  // Abre os arquivos todos
  ofstream p_df(pasta+"pontos_demaisfiltros.txt");
  ofstream p_cov(pasta+"pontos_covariancia.txt");
  ofstream p_kdt(pasta+"pontos_vizinhos_kdtree.txt");
  ofstream p_rcast(pasta+"pontos_rcast.txt");

  // Escreve uma linha para cada valor
  if(p_df.is_open()){
    for(auto p:pontos_demaisfiltros){
      p_df << p;
      p_df << "\n";
    }
  }
  if(p_cov.is_open()){
    for(auto p:pontos_covariancia){
      p_cov << p;
      p_cov << "\n";
    }
  }
  if(p_kdt.is_open()){
    for(auto p:pontos_kdtree){
      p_kdt << p;
      p_kdt << "\n";
    }
  }
  if(p_rcast.is_open()){
    for(auto p:pontos_rcast){
      p_rcast << p;
      p_rcast << "\n";
    }
  }

  // Fecha arquivos
  p_df.close(); p_cov.close(); p_kdt.close(); p_rcast.close();
}
///////////////////////////////////////////////////////////////////////////////////////////

/// Callback das poses
///
//void posesCallback(const nav_msgs::OdometryConstPtr& msg){
//  // As mensagens trazem angulos em unidade RAD, exceto pan - salvar em DEG
//  roll_cameras.push_back(msg->pose.pose.position.x);
//  pan_cameras.push_back(DEG2RAD(raw2deg(int(msg->pose.pose.position.z), "pan")));
//  pitch_cameras.push_back(msg->pose.pose.position.y);
//  ROS_INFO("Poses recebidas: %zu.", pan_cameras.size());
//  // Atualiza a quantidade de tilts que estamos esperando
//  ntilts = int(msg->pose.pose.orientation.x);
//}

/// Callback do laser e odometria sincronizado
///
void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg_cloud, const nav_msgs::OdometryConstPtr& msg_angle){
  // As mensagens trazem angulos em unidade RAD, exceto pan
  roll_cameras.push_back(msg_angle->pose.pose.position.x);
  pan_cameras.push_back(DEG2RAD(raw2deg(int(msg_angle->pose.pose.position.z), "pan")));
  pitch_cameras.push_back(msg_angle->pose.pose.position.y);
  // Atualiza a quantidade de tilts que estamos esperando
  ntilts = int(msg_angle->pose.pose.orientation.x);

  // Tempo de transito da mensagem na rede
  ros::Time tempo = ros::Time::now();
//  cout << "\nTEMPO AGORA: " << tempo << "   TEMPO DA MENSAGEM: " << msg_cloud->header.stamp << endl << endl;
  tempos_transito_msg.push_back((tempo - msg_cloud->header.stamp).toSec());
  tempo = ros::Time::now();

  // As mensagens trazem angulos em unidade RAW
  pan = int(msg_angle->pose.pose.position.x);
  int cont_aquisicao = msg_angle->pose.pose.orientation.y + 1;

  // Atualiza contagem de nuvens que chegaram na vista parcial de pan
  contador_nuvens++;

  // Converter mensagem em nuvem
  PointCloud<PointT>::Ptr cloud (new PointCloud<PointT>);
  fromROSMsg(*msg_cloud, *cloud);
  ROS_INFO("Recebendo nuvem %d com %zu pontos, contador %d e filtrando ...", cont_aquisicao, cloud->size(), contador_nuvens);
  // Aplica filtro raycasting
  ROS_INFO("Filtro raycasting ...");
  float clat = 90.0 - RAD2DEG(pitch_cameras.back()), clon = RAD2DEG(-pan_cameras.back()), fov = 38;
  if(clon < 0) clon += 360;
  ros::Time trcast = ros::Time::now();
//  pc->filterRayCasting(cloud, clat, clon, fov, fov);
  tempos_rcast.emplace_back((ros::Time::now() - trcast).toSec());
  pontos_rcast.emplace_back(cloud->size());
  // Realizar pre-processamento
  ROS_INFO("Pre-processamento com %zu pontos ...", cloud->size());
  PointCloud<PointTN>::Ptr cloud_normals (new PointCloud<PointTN>);
  float tempo_cor, tempo_octree, tempo_demaisfiltros, tempo_normais;
  size_t p_df, p_cov;
  pc->preprocess(cloud, cloud_normals, voxel_size/100.0f, depth, filter_poli,
                 tempo_cor, tempo_octree, tempo_demaisfiltros, tempo_normais,
                 p_df, p_cov);
  tempos_filtra_cor.push_back(tempo_cor);
  tempos_octree.push_back(tempo_octree);
  tempos_demaisfiltros.push_back(tempo_demaisfiltros);
  tempos_normais.push_back(tempo_normais);
  pontos_demaisfiltros.push_back(p_df);
  pontos_covariancia.push_back(p_cov);

  cloud->clear();
  ROS_INFO("Apos preprocessamento, com %zu pontos ...", cloud_normals->size());
  // Adiciona somente uma parcial daquela vista em pan - somente pontos novos!
  float raio_vizinhos = (voxel_size > 0) ? 5*voxel_size/100.0f : 0.02;
  //    roo->searchNeighborsKdTree(cloud_normals, parcial, raio_vizinhos, 200.0);
  *parcial += *cloud_normals;

  ////////// Salva as coisas temporarias para o juliano
  //    saveTempJuliano(cloud_normals, cont_aquisicao, pan, tilt);

  if(contador_nuvens == ntilts){

    tempo = ros::Time::now();
    // Transformacao inicial, antes de sofrer otimizacao, devido aos angulos de servo em PAN
//    Matrix3f R = pc->euler2matrix(0, 0, -DEG2RAD(raw2deg(pan, "pan")));
//    Quaternion q(R);
//    transformPointCloudWithNormals<PointTN>(*parcial, *parcial, R*off_laser, q);

    ///// ADICIONANDO CENTRO DA CAMERA EM PAN NA NUVEM PARA TESTE
    //        PointTN pteste;
    //        pteste.x = off(0); pteste.y = off(1); pteste.z = off(2);
    //        pteste.r =     0 ; pteste.g =    250; pteste.b =     0 ;
    //        parcial->push_back(pteste);

    ////////////////
    // Se for primeira nuvem, guardar e iniciar
    ROS_INFO("Acumulando parcial com %zu pontos ...", parcial->size());
    PointCloud<PointTN>::Ptr parcial_pontos_novos (new PointCloud<PointTN>);
    *parcial_pontos_novos = *parcial;
    if(acc->size() < 10){
      // Inicia nuvem acumulada
      *acc = *parcial;
      *anterior = *parcial;
    } else {
      // Procurar por pontos ja existentes e retirar nesse caso
      // Se nao e a ultima
      if(abs(cont_aquisicao - msg_angle->pose.pose.orientation.w) > ntilts)
        roo->searchNeighborsKdTree(parcial_pontos_novos, parcial_esq_anterior, raio_vizinhos, 10.0); // quanto maior o ultimo valor, maior o raio que eu aceito ter vizinhos
      else // Se for, comparar com a acumulada pra nao repetir pontos do inicio tambem
        roo->searchNeighborsKdTree(parcial_pontos_novos, acc                 , raio_vizinhos, 10.0);

      pontos_kdtree.push_back(parcial_pontos_novos->size());

      // Acumulando pontos novos
      *acc += *parcial_pontos_novos;
      ROS_WARN("Nuvem acumulada com %zu pontos.", acc->size());
      // Salvando nuvem atual para a proxima iteracao
      *anterior = *parcial_pontos_novos;
    }

    // Salvar a vista da esquerda para a proxima iteracao
    Matrix3f R = pc->euler2matrix(0, 0, *pan_cameras.end());
    Quaternion<float> q(R);
    transformPointCloudWithNormals(*parcial, *parcial_esq_anterior, Vector3f::Zero(), q);
    PassThrough<PointTN> pass;
    pass.setFilterFieldName("x");
    pass.setFilterLimits(0, 100); // Negativo de tudo no eixo X
    pass.setNegative(true);
    pass.setInputCloud(parcial_esq_anterior);
    pass.filter(*parcial_esq_anterior);
    transformPointCloudWithNormals(*parcial_esq_anterior, *parcial_esq_anterior, Vector3f::Zero(), q.inverse());

    tempos_vizinhos_lastview.push_back((ros::Time::now() - tempo).toSec());

    // Salvar nuvem parcia somente considerando os pontos novos
//    ROS_INFO("Salvando nuvem de pontos %d ...", cont_aquisicao);
//    if(cont_aquisicao < 10){
//      pc->saveCloud(parcial_pontos_novos, "pf_00"+std::to_string(cont_aquisicao));
//    } else if(cont_aquisicao < 100) {
//      pc->saveCloud(parcial_pontos_novos, "pf_0" +std::to_string(cont_aquisicao));
//    } else {
//      pc->saveCloud(parcial_pontos_novos, "pf_"  +std::to_string(cont_aquisicao));
//    }
    // Limpa a parcial atual vista em PAN
    parcial->clear();
    // Reseta o contador de nuvens
    contador_nuvens = 0;

  } // Fim do if else de contador nuvens

  // Publicar nuvem acumulada
  sensor_msgs::PointCloud2 msg_out;
  toROSMsg(*acc, msg_out);
  msg_out.header.stamp = ros::Time::now();
  msg_out.header.frame_id = "map";
  cl_pub.publish(msg_out);

  // Fazendo processos finais
  if(cont_aquisicao >= msg_angle->pose.pose.orientation.w){
    ROS_INFO("Processando todo o SFM otimizado ...");
    for(int i=0; i<pan_cameras.size(); i++){
      // Calcula a matriz de rotacao da camera
      float r = -roll_cameras[i], t = -pitch_cameras[i], p = -pan_cameras[i]; // [RAD]
//      Matrix3f Rt = pc->euler2matrix(0, -DEG2RAD(t), 0);
      Matrix3f Rp = pc->euler2matrix(0, 0, -p);
      Matrix3f Rcam = pc->euler2matrix(0, t, p).inverse();//(Rp*Rt).transpose();

      // Calcula centro da camera
      Vector3f C = Rp*off_laser;
      C = -Rcam.transpose()*pc->gettCam() + C;

      // Calcula vetor de translacao da camera por t = -R'*C
      Vector3f tcam = C;

      ///// ADICIONANDO CENTRO DA CAMERA NA NUVEM PARA TESTE
      //            PointTN pteste;
      //            pteste.x = C(0); pteste.y = C(1); pteste.z = C(2);
      //            pteste.r =   0 ; pteste.g =   0; pteste.b =   250;
      //            acc->push_back(pteste);

      // Escreve a linha e anota no vetor de linhas SFM
      string nome_imagem;
      if(i+1 < 10)
        nome_imagem = "imagem_00"+std::to_string(i+1)+".png";
      else if(i+1 < 100)
        nome_imagem = "imagem_0" +std::to_string(i+1)+".png";
      else
        nome_imagem = "imagem_"  +std::to_string(i+1)+".png";
      linhas_sfm.push_back(pc->escreve_linha_sfm(nome_imagem, Rcam, tcam));
    }
    ROS_INFO("Salvando nuvem acumulada final ...");
    pc->saveCloud(acc, "acumulada");
    ROS_INFO("Salvando SFM e planta baixa final ...");
    pc->compileFinalSFM(linhas_sfm);
    Mat blueprint;
    float side_area = 20, side_res = 0.04;
    pc->blueprint(acc, side_area, side_res, blueprint);
    // Salva em arquivos os vetores de tempo e pontos para serem observados no futuro
    saveTimeFiles();
    savePointFiles();
    ROS_INFO("Processo finalizado.");
    ros::shutdown();
  }
  ////////////////
}

/// Main
///
int main(int argc, char **argv)
{
  ros::init(argc, argv, "acc_space2");
  ros::NodeHandle nh;
  ros::NodeHandle n_("~");
  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
  ROS_INFO("Iniciando o processo em FOG de registro de nuvens ...");

  int a=0;
  ros::Time tem;
  vector<uint64_t> temps(500);
  for(int k=0; k<500; k++){
    tem = ros::Time::now();
    for(int i=0; i<10; i++)
      a = 10+10;
    temps[k] = (ros::Time::now() - tem).toNSec();
  }
  float avg = float(accumulate(temps.begin(), temps.end(), 0))/float(temps.size());
  ROS_WARN("TEMPO PU DA FOG: %.2f", avg);
  cout << endl << endl << endl;

  // Pegando o nome da pasta por parametro
  string nome_param;
  n_.param<string>("pasta", nome_param , string("Dados_PEPO"));
  n_.param<float >("vs"   , voxel_size , 2    );
  n_.param<float >("df"   , depth      , 10   );
  n_.param<int   >("fp"   , filter_poli, 1    );

  // Apagando pasta atual e recriando a mesma na area de trabalho
  char* home;
  home = getenv("HOME");
  // Checando se ha a pasta spaces, senao criar
  pasta = string(home)+"/Desktop/ambientes/";
  struct stat buffer;
  if(stat(pasta.c_str(), &buffer)) // Se nao existe a pasta
      mkdir(pasta.c_str(), 0777);
  // Criando pasta mae
  pasta = pasta + nome_param.c_str();
  if(stat(pasta.c_str(), &buffer)) // Se nao existe a pasta
      mkdir(pasta.c_str(), 0777);
  // Criando pastas filhas
  pasta = create_folder(pasta + "/scan") + "/";

  // Inicia classe de processo de nuvens
  pc  = new ProcessCloud(pasta);
  roo = new RegisterObjectOptm();

  // Calculando taxas exatas entre deg e raw para os motores de pan e tilt
  deg_raw_pan  = 0.08764648;
  deg_raw_tilt = deg_raw_pan; //(deg_max_tilt - deg_min_tilt)/(raw_max_tilt - raw_min_tilt);
  raw_deg_pan  = 1.0/deg_raw_pan ;
  raw_deg_tilt = 1.0/deg_raw_tilt;

  // Iniciando a nuvem parcial acumulada de cada pan
  acc = (PointCloud<PointTN>::Ptr) new PointCloud<PointTN>();
  acc->header.frame_id = "map";
  anterior = (PointCloud<PointTN>::Ptr) new PointCloud<PointTN>();
  anterior->header.frame_id = "map";
  parcial = (PointCloud<PointTN>::Ptr) new PointCloud<PointTN>();
  parcial->header.frame_id = "map";
  parcial_esq_anterior = (PointCloud<PointTN>::Ptr) new PointCloud<PointTN>();
  parcial_esq_anterior->header.frame_id = "map";

  // Publicadores
  cl_pub = nh.advertise<sensor_msgs::PointCloud2>("/cloud_user" , 50);
  od_pub = nh.advertise<nav_msgs::Odometry      >("/vista_space", 50);

  // Subscriber de poses das cameras
//  ros::Subscriber poses_sub = nh.subscribe("/poses_space", 100, &posesCallback);

  // Iniciar subscritor da nuvem sincronizado com odometria
  message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub(nh, "/cloud_space", 100);
  message_filters::Subscriber<nav_msgs::Odometry      > angle_sub(nh, "/angle_space", 100);
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(100), cloud_sub, angle_sub);
  sync.registerCallback(boost::bind(&cloudCallback, _1, _2));

  // Esperando ligar a camera e vir imagens - poses
//  ros::Rate r(10);
//  ROS_INFO("Aguardando vinda de poses ...");
//  while(poses_sub.getNumPublishers() == 0){
//    r.sleep();
//    ros::spinOnce();
//  }
  ROS_INFO("Comecando a reconstrucao do space ...");

  // Aguardar todo o processamento e ir publicando
  ros::spin();

  return 0;
}
