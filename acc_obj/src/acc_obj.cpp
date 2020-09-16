#include <ros/ros.h>
#include <iostream>
#include <string>
#include <math.h>
#include <sys/stat.h>
#include <ostream>
#include <iterator>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>

#include <sensor_msgs/PointCloud2.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>

#include "../../libraries/include/processcloud.h"
#include "../../libraries/include/registerobjectoptm.h"

using namespace pcl;
using namespace pcl::io;
using namespace Eigen;
using namespace std;
using namespace cv;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int getdir(string dir, vector<string> &imgs, vector<string> &nuvens){
    // Abrindo a pasta raiz para contar os arquivos de imagem e nuvem que serao lidos e enviados
    DIR *dp;
    struct dirent *dirp;
    string nome_temp;
    if((dp  = opendir(dir.c_str())) == NULL)
        ROS_ERROR("Nao foi possivel abrir o diretorio");

    while ((dirp = readdir(dp)) != NULL) {
        nome_temp = string(dirp->d_name);
        // Todas as imagens na pasta
        if(nome_temp.substr(nome_temp.find_last_of(".")+1) == "png")
            imgs.push_back(nome_temp);
        // Todas as nuvens na pasta
        if(nome_temp.substr(nome_temp.find_last_of(".")+1) == "ply")
            nuvens.push_back(nome_temp);
    }
    sort(imgs.begin()  , imgs.end()  );
    sort(nuvens.begin(), nuvens.end());

    closedir(dp);

    return 0;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    ros::init(argc, argv, "acc_obj");
    ros::NodeHandle nh;
    ros::NodeHandle n_("~");

    pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

    // Inicia publicador da nuvem parcial e do objeto
    ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>("/acc_obj/obj", 10);
    sensor_msgs::PointCloud2 msg;

    // Ler pasta com os dados
    string nome_param;
    n_.param("pasta", nome_param, string("Dados_PEPO"));
    char* home;
    home = getenv("HOME");
    string pasta = string(home)+"/Desktop/"+nome_param+"/";

    // Objetos de classe usados no processo
    ProcessCloud pc(pasta);
    RegisterObjectOptm roo;

    // Se ja houver objeto salvo, deletar ele nesse ponto e comecar outro
    struct stat buffer;
    string nome_objeto_final = pasta + "objeto_final.ply";
    if(!stat(nome_objeto_final.c_str(), &buffer)){
        if(remove(nome_objeto_final.c_str()) == 0)
            ROS_INFO("Deletamos nuvem final anterior.");
        else
            ROS_ERROR("Nuvem final anterior nao foi deletada.");
    }

    // Vasculhar a pasta em busca de todas as nuvens e imagens
    vector<string> nomes_nuvens, nomes_imagens;
    getdir(pasta, nomes_imagens, nomes_nuvens);
    ROS_INFO("Temos um total de %zu nuvens a processar ...", nomes_nuvens.size());

    /// Inicia as variaveis - aqui comeca a pancadaria
    ///
    // Nuvem de agora, nuvem referencia, nuvem do objeto acumulado,
    // Matriz de pixels auxiliar referencia, Matriz de pixels auxiliar atual
    PointCloud<PointTN>::Ptr cnow (new PointCloud<PointTN>), cref (new PointCloud<PointTN>), cobj (new PointCloud<PointTN>);
    PointCloud<PointTN>::Ptr cpixref (new PointCloud<PointTN>), cpixnow (new PointCloud<PointTN>);
    MatrixXi imrefpix(1080, 1920), imnowpix(1080, 1920);
    cref->header.frame_id = "map";
    cnow->header.frame_id = "map";
    cobj->header.frame_id = "map";
    // Imagem de agora e da referencia
    Mat imnow, imref;
    // Pose da nuvem agora e na referencia
    Matrix4f Tnow, Tref;
    // Centro e orientacao da camera naquela aquisicao
    Vector3f C;
    Quaternion<float> q;
    // Vetor de offset entre centro do laser e da camera - desenho solid, e foco
    Vector3f t_off_lc(0.00, 0.0443, 0.023);
    float f = 1130;
    // Vetor de linhas para NVM
    vector<string> linhas_nvm;


    /// Inicia primeiro a nuvem referencia e outros dados com o primeiro dado lido
    ///
    if(cref->empty()){
        ROS_INFO("Iniciando os dados de referencia ...");
        roo.readCloudAndPreProcess(pasta+nomes_nuvens[0], cref);
        pc.calculateNormals(cref);
        pc.filterCloudDepthCovariance(cref, 15, 1);
        imref = imread(pasta+nomes_imagens[0]);
        imrefpix = MatrixXi::Constant(imref.rows, imref.cols, -1);

        // Projetar a nuvem na imagem de forma calibrada, otimizar a cor e salvar em matriz de
        // pontos auxiliar os pixels correspondentes a cada ponto
        roo.projectCloudAndAnotatePixels(cref, imref, cpixref, f, t_off_lc, imrefpix);
        savePLYFileBinary<PointTN>(pasta+nomes_nuvens[0], *cref);
        *cobj = *cref;
        toROSMsg(*cobj, msg);
        msg.header.frame_id = "map";
        msg.header.stamp = ros::Time::now();
        pub.publish(msg);

        // Inicia pose e escreve para o NVM
        Tref = Matrix4f::Identity();
        Tnow = Tref;        
        C = -t_off_lc;
        q = Quaternion<float>::Identity();
        linhas_nvm.push_back(pc.escreve_linha_imagem(f, nomes_imagens[0], C, q));
    }

    /// A partir da segunda nuvem e imagem, processo de match e anotacao
    ///
    for(size_t i=1; i<nomes_nuvens.size(); i++){
        ROS_INFO("Acumulando nuvem de objeto %zu ...", i+1);
        // Le dados, filtra e guarda na memoria
        ROS_INFO("Pre-processando nuvem %zu ...", i+1);
        roo.readCloudAndPreProcess(pasta+nomes_nuvens[i], cnow);
        pc.calculateNormals(cnow);
        pc.filterCloudDepthCovariance(cnow, 20, 1);
        imnow = imread(pasta+nomes_imagens[i]);
        imnowpix = MatrixXi::Constant(imnow.rows, imnow.cols, -1);

        // Projetar nuvem e salvar nuvem auxiliar de pixels
        ROS_INFO("Projetando para otimizar cores e anotar os pixels ...");
        roo.projectCloudAndAnotatePixels(cnow, imnow, cpixnow, f, t_off_lc, imnowpix);
        savePLYFileBinary<PointTN>(pasta+nomes_nuvens[i], *cnow);

        // Levar aonde paramos no processo de reconstrucao do objeto
        transformPointCloudWithNormals<PointTN>(*cnow, *cnow, Tref);

        // Calcular features e matches e retornar os N melhores e seus pontos em 3D
        // correspondentes a partir da nuvem auxiliar de pixels
        ROS_INFO("Match de features 2D e obtendo correspondencias em 3D ...");
        vector<Point2d> matches3Dindices;
        roo.matchFeaturesAndFind3DPoints(imref, imnow, cpixref, cpixnow, 90, matches3Dindices, imrefpix, imnowpix, 5);
        ROS_INFO("Foram obtidas %zu correspondencias 3D.", matches3Dindices.size());

        // Continuamos somente se a imagem forneceu matches
        Matrix4f Ticp;
        if(matches3Dindices.size() > 5){
            // Rodar a otimizacao da transformada por SVD
            ROS_INFO("Otimizando a transformacao relativa das nuvens por SVD ...");
            Tnow = roo.optmizeTransformSVD(cref, cnow, matches3Dindices);

            // Transforma a nuvem atual com a transformacao encontrada
            transformPointCloudWithNormals<PointTN>(*cnow, *cnow, Tnow);

            // Refina a transformacao por ICP com poucas iteracoes
            ROS_INFO("Refinando registro por ICP ...");
            Ticp = roo.icp(cobj, cnow, 15);
        } else {
            // Transforma a nuvem atual com a transformacao encontrada
            transformPointCloudWithNormals<PointTN>(*cnow, *cnow, Tref);
            // Refina a transformacao por ICP com mais iteracoes
            ROS_INFO("Nao encontrou match por features, refinando registro por ICP ...");
            roo.icp(cobj, cnow, 50);
        }

        // Soma a nuvem transformada e poe no lugar certo somente pontos "novos"
        ROS_INFO("Registrando nuvem atual no objeto final ...");
        Matrix4f Tobj = Ticp*Tnow*Tref;
        PointCloud<PointTN>::Ptr cnowtemp (new PointCloud<PointTN>);
        *cnowtemp = *cnow;
        roo.searchNeighborsKdTree(cnowtemp, cobj, 0.6);
        *cobj += *cnowtemp;
        StatisticalOutlierRemoval<PointTN> sor;
        sor.setInputCloud(cobj);
        sor.setMeanK(30);
        sor.setStddevMulThresh(2);
        sor.filter(*cobj);

        // Publicando o resultado atual para visualizacao
        toROSMsg(*cobj, msg);
        msg.header.frame_id = "map";
        msg.header.stamp = ros::Time::now();
        pub.publish(msg);

        // Calcula a pose da camera e escreve no NVM
        ROS_INFO("Escrevendo no NVM ...");
        Matrix4f Tcam = Matrix4f::Identity();
        Tcam.block<3,1>(0, 3) = t_off_lc;
        Tcam = Tobj*Tcam;
        C = Tcam.block<3,1>(0, 3);
        q = Tcam.block<3,3>(0, 0).transpose();
        linhas_nvm.push_back(pc.escreve_linha_imagem(f, nomes_imagens[i], C, q));
        pc.compileFinalNVM(linhas_nvm);

        // Atualiza as referencias e parte para a proxima aquisicao
        *cref = *cnow;
        *cpixref = *cpixnow; // Nuvem de indices ja esta referenciado a origem
        imrefpix = imnowpix;
        Tref = Tobj;
        imnow.copyTo(imref);
    }

    // Filtrando novamente objeto final
    ROS_INFO("Filtrando e salvando nuvem de objeto ...");
    StatisticalOutlierRemoval<PointTN> sor;
    sor.setMeanK(10);
    sor.setStddevMulThresh(2);
    sor.setInputCloud(cobj);
    sor.filter(*cobj);

    // Publicando o resultado atual para visualizacao
    toROSMsg(*cobj, msg);
    msg.header.frame_id = "map";
    msg.header.stamp = ros::Time::now();
    pub.publish(msg);

    // Salvando a nuvem final do objeto
    savePLYFileBinary<PointTN>(nome_objeto_final, *cobj);
    pc.compileFinalNVM(linhas_nvm);

    ROS_INFO("Processo finalizado.");

    ros::spinOnce();
    return 0;
}
