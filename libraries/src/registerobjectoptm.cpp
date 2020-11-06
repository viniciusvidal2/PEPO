#include "../include/registerobjectoptm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
RegisterObjectOptm::RegisterObjectOptm()
{

}
/////////////////////////////////////////////////////////////////////////////////////////////////
RegisterObjectOptm::~RegisterObjectOptm(){
    ros::shutdown();
    ros::waitForShutdown();
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void RegisterObjectOptm::readCloudAndPreProcess(string name, PointCloud<PointT>::Ptr cin){
    // Le a nuvem
    loadPLYFile<PointT>(name, *cin);
    // Retirando indices NaN se existirem
    vector<int> indicesNaN;
    removeNaNFromPointCloud(*cin, *cin, indicesNaN);
    // Filtro de voxels para aliviar a entrada
    VoxelGrid<PointT> voxel;
    float lfsz = 0.03;
    voxel.setLeafSize(lfsz, lfsz, lfsz);
    // Filtro de profundidade para nao pegarmos muito fundo
    PassThrough<PointT> pass;
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0, 10); // Z metros de profundidade
    // Filtro de ruidos aleatorios
    StatisticalOutlierRemoval<PointT> sor;
    sor.setMeanK(10);
    sor.setStddevMulThresh(2.5);
    sor.setNegative(false);
    // Passando filtros
    sor.setInputCloud(cin);
    sor.filter(*cin);
    voxel.setInputCloud(cin);
//    voxel.filter(*cin);
    pass.setInputCloud(cin);
    pass.filter(*cin);
    sor.setInputCloud(cin);
//    sor.filter(*cin);
//    // Passando polinomio pra suavizar a parada
//    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>());
//    MovingLeastSquares<PointT, PointTN> mls;
//    mls.setComputeNormals(true);
//    mls.setInputCloud(cin);
//    mls.setPolynomialOrder(1);
//    mls.setSearchMethod(tree);
//    mls.setSearchRadius(0.05);
//    mls.process(*cloud);
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void RegisterObjectOptm::projectCloudAndAnotatePixels(PointCloud<PointTN>::Ptr cloud, Mat im, PointCloud<PointTN>::Ptr cloud_pix, float f, Vector3f t, MatrixXi &impix){
    // Matriz intrinseca e extrinseca
    Matrix3f K;
    K << f, 0, 982,//im.cols/2.0,
            0, f, 540,//im.rows/2.0,
            0, 0,  1 ;
    MatrixXf Rt(3, 4); // Desenho do antonio - diferenca no frame da camera do laser para a camera
    Rt << 1, 0, 0, t(0),
            0, 1, 0, t(1),
            0, 0, 1, t(2);
    MatrixXf P(3, 4);
    P = K*Rt;
    // Preparando nuvem de pixels
    cloud_pix->resize(cloud->size());
    for(size_t i = 0; i < cloud->size(); i++){
        // Pegar ponto em coordenadas homogeneas
        MatrixXf X_(4, 1);
        X_ << cloud->points[i].x,
                cloud->points[i].y,
                cloud->points[i].z,
                1       ;
        MatrixXf X(3, 1);
        X = P*X_;
        if(X(2, 0) > 0){
            X = X/X(2, 0);
            // Colorindo ponto se for o caso de projetado corretamente
            if(floor(X(0,0)) > 0 && floor(X(0,0)) < im.cols && floor(X(1,0)) > 0 && floor(X(1,0)) < im.rows){
                cv::Vec3b cor = im.at<Vec3b>(Point(X(0,0), X(1,0)));
                PointTN point = cloud->points[i];
                point.b = cor.val[0]; point.g = cor.val[1]; point.r = cor.val[2];
                cloud->points[i] = point;
                // Anotando que aquele ponto 3D cai naquele pixel da imagem, e usando a normal pra guardar a coordenada 3D em si - debug
                cloud_pix->points[i].x = X(0,0); cloud_pix->points[i].y = X(1,0);
                cloud_pix->points[i].r = cor.val[2]; cloud_pix->points[i].g = cor.val[1]; cloud_pix->points[i].b = cor.val[0];
                cloud_pix->points[i].normal_x = cloud->points[i].x;
                cloud_pix->points[i].normal_y = cloud->points[i].y;
                cloud_pix->points[i].normal_z = cloud->points[i].z;
                impix(X(1,0), X(0,0)) = i;
            } else {
                cloud_pix->points[i].x = 1000000; cloud_pix->points[i].y = 1000000;
            }
        } else {
            cloud_pix->points[i].x = 1000000; cloud_pix->points[i].y = 1000000;
        }
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////
Matrix4f RegisterObjectOptm::icp(PointCloud<PointTN>::Ptr ctgt, PointCloud<PointTN>::Ptr csrc, float vs, int its){
    Matrix4f Ticp = Matrix4f::Identity();

    // Reduzindo ainda mais as nuvens pra nao dar trabalho assim ao icp
    PointCloud<PointTN>::Ptr tgttemp(new PointCloud<PointTN>);
    PointCloud<PointTN>::Ptr srctemp(new PointCloud<PointTN>);
    VoxelGrid<PointTN> voxel;
    voxel.setLeafSize(vs, vs, vs);
    voxel.setInputCloud(ctgt);
    voxel.filter(*tgttemp);
    voxel.setInputCloud(csrc);
    voxel.filter(*srctemp);
    StatisticalOutlierRemoval<PointTN> sor;
    sor.setMeanK(30);
    sor.setStddevMulThresh(2);
    sor.setNegative(false);
    sor.setInputCloud(srctemp);
    sor.filter(*srctemp);
    sor.setInputCloud(tgttemp);
    sor.filter(*tgttemp);
//    *tgttemp = *ctgt;
//    *srctemp = *csrc;

    // Criando o otimizador de ICP comum
    GeneralizedIterativeClosestPoint<PointTN, PointTN> icp;
    //    IterativeClosestPoint<PointTN, PointTN> icp;
    icp.setUseReciprocalCorrespondences(true);
    icp.setInputTarget(tgttemp);
    icp.setInputSource(srctemp);
    //    icp.setRANSACIterations(30);
    icp.setMaximumIterations(its); // Chute inicial bom 10-100
    icp.setTransformationEpsilon(1*1e-8);
    icp.setEuclideanFitnessEpsilon(1*1e-10);
    icp.setMaxCorrespondenceDistance(vs*3);
    // Alinhando
    PointCloud<PointTN> dummy;
    icp.align(dummy, Matrix4f::Identity());
    // Obtendo a transformacao otimizada e aplicando
    if(icp.hasConverged())
        Ticp = icp.getFinalTransformation();
//    transformPointCloudWithNormals<PointTN>(*csrc, *csrc, Ticp);

    return Ticp;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void RegisterObjectOptm::matchFeaturesAndFind3DPoints(Mat imref, Mat imnow, PointCloud<PointTN>::Ptr cref, PointCloud<PointTN>::Ptr cnow,
                                                      PointCloud<PointTN>::Ptr cmr, PointCloud<PointTN>::Ptr cmn){
    /// Calculando descritores SIFT ///
    // Keypoints e descritores para astra e zed
    vector<KeyPoint> kpref, kpnow;
    Mat dref, dnow;
    /// Comparando e filtrando matchs ///
    cv::Ptr<DescriptorMatcher> matcher;
    matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    vector<vector< DMatch > > matches;
    vector<DMatch> good_matches;

    // Descritores SIFT calculados
    Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create();
    sift->detectAndCompute(imref, Mat(), kpref, dref);
    sift->detectAndCompute(imnow, Mat(), kpnow, dnow);
    // Calculando somatorio para cada linha de descritores
    Mat drefsum, dnowsum;
    reduce(dref, drefsum, 1, CV_16UC1);
    reduce(dnow, dnowsum, 1, CV_16UC1);
    // Normalizando e passando raiz em cada elementos de linha nos descritores da src
#pragma omp parallel for
    for(int i=0; i<dnow.rows; i++){
      for(int j=0; j<dnow.cols; j++){
        dnow.at<float>(i, j) = sqrt(dnow.at<float>(i, j) / (dnowsum.at<float>(i, 0) + numeric_limits<float>::epsilon()));
      }
    }
    // Normalizando e passando raiz em cada elementos de linha nos descritores da tgt
#pragma omp parallel for
    for(int i=0; i<dref.rows; i++){
      for(int j=0; j<dref.cols; j++){
        dref.at<float>(i, j) = sqrt(dref.at<float>(i, j) / (drefsum.at<float>(i, 0) + numeric_limits<float>::epsilon()));
      }
    }

    matcher->knnMatch(dnow, dref, matches, 2);
    for (size_t k = 0; k < matches.size(); k++){
      if (matches.at(k).size() >= 2){
        if (matches.at(k).at(0).distance < 0.8*matches.at(k).at(1).distance) // Se e bastante unica frente a segunda colocada
          good_matches.push_back(matches.at(k).at(0));
      }
    }

    for(int i=0; i<good_matches.size(); i++){
      int r = rand()%255, b = rand()%255, g = rand()%255;
      circle(imref, Point(kpref[good_matches[i].trainIdx].pt.x, kpref[good_matches[i].trainIdx].pt.y), 8, Scalar(r, g, b), FILLED, LINE_8);
      circle(imnow, Point(kpnow[good_matches[i].queryIdx].pt.x, kpnow[good_matches[i].queryIdx].pt.y), 8, Scalar(r, g, b), FILLED, LINE_8);
    }
    imshow("targetc", imref);
    imshow("sourcec", imnow);
    waitKey(0);

    // Dados intrinsecos da camera
    float fx = 1130.000000, fy = 1130.000000, cx = 1920.0/2, cy = 1080.0/2;

    // Se houveram matches suficientes
    if(good_matches.size() > 6){
      // Aloca espaco para nuvens de saida de match 3D
      cmr->resize(good_matches.size()); cmn->resize(good_matches.size());
      // Calcular pontos 3D de cada match em now
      vector<int> istherepoint_now(good_matches.size());
      octree::OctreePointCloudSearch<PointTN> oct(0.02);
      oct.setInputCloud(cnow);
      oct.addPointsFromInputCloud();
#pragma omp parallel for
      for(int i=0; i<good_matches.size(); i++){
        float u = kpnow[good_matches[i].queryIdx].pt.x, v = kpnow[good_matches[i].queryIdx].pt.y;
        Vector3f dir;
        dir << (u - cx)/fx, (v - cy)/fy, 1;
        octree::OctreePointCloudSearch<PointTN>::AlignedPointTVector align;
        oct.getIntersectedVoxelCenters(Vector3f::Zero(), dir.normalized(), align, 1);
        // Se achou, adicionar na lista de pontos 3D naquele local
        if(align.size() > 0){
          vector<int> ind;
          vector<float> dists;
          oct.nearestKSearch(align[0], 1, ind, dists);
          if(ind.size() > 0){
            istherepoint_now[i] = 1;
            cmn->points[i] = cnow->points[ind[0]];
          } else {
            istherepoint_now[i] = 0;
          }
        } else {
          istherepoint_now[i] = 0;
        }
      }

      // Calcular pontos 3D de cada match em ref
      vector<int> istherepoint_ref(good_matches.size());
      octree::OctreePointCloudSearch<PointTN> oct2(0.02);
      oct2.setInputCloud(cref);
      oct2.addPointsFromInputCloud();
#pragma omp parallel for
      for(int i=0; i<good_matches.size(); i++){
        float u = kpref[good_matches[i].trainIdx].pt.x, v = kpref[good_matches[i].trainIdx].pt.y;
        Vector3f dir;
        dir << (u - cx)/fx, (v - cy)/fy, 1;
        octree::OctreePointCloudSearch<PointTN>::AlignedPointTVector align;
        oct2.getIntersectedVoxelCenters(Vector3f::Zero(), dir.normalized(), align, 1);
        // Se achou, adicionar na lista de pontos 3D naquele local
        if(align.size() > 0){
          vector<int> ind;
          vector<float> dists;
          oct2.nearestKSearch(align[0], 1, ind, dists);
          if(ind.size() > 0){
            istherepoint_ref[i] = 1;
            cmr->points[i] = cref->points[ind[0]];
          } else {
            istherepoint_ref[i] = 0;
          }
        } else {
          istherepoint_ref[i] = 0;
        }
      }

      // Preencher nuvens de pontos correspondentes
      PointCloud<PointTN>::Ptr tempn (new PointCloud<PointTN>), tempr (new PointCloud<PointTN>);
      for(int i=0; i<good_matches.size(); i++){
        if(istherepoint_now[i] == 1 && istherepoint_ref[i] == 1){
          tempn->push_back(cmn->points[i]); tempr->push_back(cmr->points[i]);
        }
      }
      *cmn = *tempn; *cmr = *tempr;
    }

    // Plotando os dados para debugar
//    this->plotDebug(imref, imnow, cref, cnow, goodimrefpts, goodimnowpts, matches3d);
}
/////////////////////////////////////////////////////////////////////////////////////////////////
Matrix4f RegisterObjectOptm::optmizeTransformLeastSquares(PointCloud<PointTN>::Ptr cref, PointCloud<PointTN>::Ptr cnow, vector<Point2d> matches3d){
    /// Conta dos minimos quadrados
    /// Theta = inv(Xt*X)*Xt*y
    /// Theta sao os coeficientes finais
    /// X sao em linhas as coordenadas dos pontos na nuvem now
    /// y sao em linhas cada hora uma das coordenadas nos pontos na nuvem ref, a depender do sistema que estamos tratando (eixos, X, Y ou Z)

    // Iniciando vetores
    MatrixXf X(matches3d.size(), 4), Xt, y(matches3d.size(), 1), Theta; // minimos quadrados
    Matrix4f Tf = Matrix4f::Identity(); // Matriz que sera otimizada
    int indice_ref, indice_now; // Indices dos pontos 3D correspondentes nas nuvens
    // Para cada sistema (eixo 3D)
    for(int s=0; s<3; s++){
        // Preenchendo vetores least squares //
        for(int i=0; i<matches3d.size(); i++){
            indice_ref = matches3d[i].x;
            indice_now = matches3d[i].y;
            X.block<1,4>(i, 0) << cnow->points[indice_now].x, cnow->points[indice_now].y, cnow->points[indice_now].z, 1;
            if(s == 0)
                y(i, 0) = cref->points[indice_ref].x;
            else if(s == 1)
                y(i, 0) = cref->points[indice_ref].y;
            else if(s == 2)
                y(i, 0) = cref->points[indice_ref].z;
        }
        // Obtendo a transposta para fazer isso so uma vez
        Xt = X.transpose();
        // Operacao de minimos quadrados para obter resultado para o sistema atual
        Theta = (Xt*X).inverse()*Xt*y;
        // Inserindo resultado na matriz de transformacao final
        Tf.block<1,4>(s, 0) << Theta.transpose();
    }

    // Retorna a transformacao otimizada
    return Tf;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void RegisterObjectOptm::plotDebug(Mat imref, Mat imnow, PointCloud<PointTN>::Ptr cref, PointCloud<PointTN>::Ptr cnow, vector<Point2f> pref, vector<Point2f> pnow, vector<Point2d> match){
    // Imagens temporarias para salvar
    Mat r, n;
    imref.copyTo(r); imnow.copyTo(n);
    // Nuvens temporarias para salvar
    PointCloud<PointTN>::Ptr cr (new PointCloud<PointTN>);
    PointCloud<PointTN>::Ptr cn (new PointCloud<PointTN>);
    PointCloud<PointTN>::Ptr crt (new PointCloud<PointTN>);
    PointCloud<PointTN>::Ptr cnt (new PointCloud<PointTN>);
    PointTN p;
    // Para cada ponto de match
    for(int i=0; i<match.size(); i++){
        // Criar uma cor aleatoria
        Scalar color = Scalar(rand() % 255, rand() % 255, rand() % 255);
        // Desenha circulo na imagem
        circle(r, pref[i], 9, color, 4);
        circle(n, pnow[i], 9, color, 4);
        // Colore ponto das nuvens com a cor tambem
        p.r = color[2]; p.g = color[1]; p.b = color[0];
        // Adiciona o ponto na nuvem temporaria
        p.x = cref->points[match[i].x].normal_x; p.y = cref->points[match[i].x].normal_y; p.z = cref->points[match[i].x].normal_z;
        cr->push_back(p);
        p.x = cnow->points[match[i].y].normal_x; p.y = cnow->points[match[i].y].normal_y; p.z = cnow->points[match[i].y].normal_z;
        cn->push_back(p);
    }
    // Toda a nuvem
    crt->resize(cref->size());
#pragma omp parallel for
    for(int i=0; i<cref->size(); i++){
        crt->points[i].x = cref->points[i].normal_x; crt->points[i].y = cref->points[i].normal_y; crt->points[i].z = cref->points[i].normal_z;
    }
    cnt->resize(cnow->size());
#pragma omp parallel for
    for(int i=0; i<cnow->size(); i++){
        cnt->points[i].x = cnow->points[i].normal_x; cnt->points[i].y = cnow->points[i].normal_y; cnt->points[i].z = cnow->points[i].normal_z;
    }
    // Salvando arquivos
    char* home;
    home = getenv("HOME");
    string pasta = string(home)+"/Desktop/";
    imwrite(pasta+"debugref.png", r);
    imwrite(pasta+"debugnow.png", n);
    savePLYFileBinary<PointTN>(pasta+"debugref.ply", *cr);
    savePLYFileBinary<PointTN>(pasta+"debugnow.ply", *cn);
    savePLYFileBinary<PointTN>(pasta+"debugreft.ply", *crt);
    savePLYFileBinary<PointTN>(pasta+"debugnowt.ply", *cnt);
}
/////////////////////////////////////////////////////////////////////////////////////////////////
Matrix4f RegisterObjectOptm::optmizeTransformSVD(PointCloud<PointTN>::Ptr cref, PointCloud<PointTN>::Ptr cnow){
    // Inicia estimador
    registration::TransformationEstimationSVD<PointTN, PointTN> svd;
    Matrix4f Tsvd;
    // Estima a transformacao
    svd.estimateRigidTransformation(*cnow, *cref, Tsvd);

    return Tsvd;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void RegisterObjectOptm::filterMatchesLineCoeff(vector<DMatch> &matches, vector<KeyPoint> kpref, vector<KeyPoint> kpnow, float width, float n){
    // Fazer e calcular vetor de coeficientes para cada ponto correspondente do processo de match
    vector<float> coefs(matches.size());
#pragma omp parallel for
    for(int i=0; i<matches.size(); i++){
        float xr, yr, xn, yn;
        xr = kpref[matches[i].queryIdx].pt.x;
        yr = kpref[matches[i].queryIdx].pt.y;
        xn = kpnow[matches[i].trainIdx].pt.x + width;
        yn = kpnow[matches[i].trainIdx].pt.y;
        // Calcular os coeficientes angulares
        coefs[i] = (yn - yr)/(xn - xr);
    }
    // Media e desvio padrao dos coeficientes
    float average = accumulate(coefs.begin(), coefs.end(), 0.0)/coefs.size();
    float accum = 0.0;
    for_each(coefs.begin(), coefs.end(), [&](const float d){
        accum += (d - average) * (d - average);
    });
    float stdev = sqrt(accum/(coefs.size()-1));
    // Filtrar o vetor de matches na posicao que os coeficientes estejam fora
    vector<DMatch> temp;
    for(int i=0; i<coefs.size(); i++){
        if(abs(coefs[i]-average) < n*stdev)
            temp.push_back(matches[i]);
    }
    matches = temp;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
Matrix4f RegisterObjectOptm::optmizeTransformP2P(PointCloud<PointTN>::Ptr cref, PointCloud<PointTN>::Ptr cnow, vector<Point2d> matches3d){
    // Inicia estimador
    registration::TransformationEstimationPointToPlaneLLS<PointTN, PointTN, float> p2p;
    PointIndices pref, pnow;
    Matrix4f Tp2p;
    // Anota os indices vindos dos matches para os pontos correspondentes em cada nuvem
    for(int i=0; i<matches3d.size(); i++){
        pref.indices.push_back(matches3d[i].x);
        pnow.indices.push_back(matches3d[i].y);
    }
    // Estima a transformacao
    p2p.estimateRigidTransformation(*cnow, pnow.indices, *cref, pref.indices, Tp2p);

    return Tp2p;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
int RegisterObjectOptm::searchNeighbors(MatrixXi im, int r, int c, int l){
    // Aumentando a janela de busca de 2 em dois nos lados ate o limite proposto
    for(int k=1; k < l; k = k+2){
        // Duplo for sobre a vizinhanca pra ver se encontra, e o primeiro de todos leva
        for(int i=r-k; i<r+k; i++){
            if(i >= 0 && i < im.rows()){
                for(int j=c-k; j<c+k; j++){
                    if(j > 0 && j < im.cols()){
                        // Se nesse lugar esta algo diferente de -1
                        if(im(i, j) != -1)
                            return im(i, j);
                    }
                }
            }
        }
    }

    return -1;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void RegisterObjectOptm::searchNeighborsKdTree(PointCloud<PointTN>::Ptr cnow, PointCloud<PointTN>::Ptr cobj, float radius, float rate){
    if(cnow->size() > 200 && cobj->size() > 200){
        // Iniciar kdtree de busca
        KdTreeFLANN<PointTN> kdtree;
        kdtree.setInputCloud(cobj);
        vector<int> pointIdxRadiusSearch;
        vector<float> pointRadiusSquaredDistance;
        // Nuvem de pontos de indices bons
        PointIndices::Ptr indices (new PointIndices);
        // Retirando indices NaN se existirem
        vector<int> indicesNaN;
        removeNaNFromPointCloud(*cnow, *cnow, indicesNaN);
        removeNaNFromPointCloud(*cobj, *cobj, indicesNaN);
        // Para cada ponto, se ja houver vizinhos, nao seguir
        for(size_t i=0; i<cnow->size(); i++){
            if(kdtree.radiusSearch(cnow->points[i], radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) <= rate)
                indices->indices.emplace_back(i);
        }
        // Filtrar na nuvem now so os indices que estao sem vizinhos na obj
        ExtractIndices<PointTN> extract;
        extract.setInputCloud(cnow);
        extract.setIndices(indices);
        extract.setNegative(false);
        extract.filter(*cnow);
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////
Matrix4f RegisterObjectOptm::estimate3DcorrespondenceAndTransformation(PointCloud<PointTN>::Ptr cnow, PointCloud<PointTN>::Ptr cobj){
    // Separando as nuvens em pontos, cor e normais
    PointCloud<PointT>::Ptr cnowp (new PointCloud<PointT>);
    PointCloud<PointT>::Ptr cobjp (new PointCloud<PointT>);
    PointCloud<Normal>::Ptr cnown (new PointCloud<Normal>);
    PointCloud<Normal>::Ptr cobjn (new PointCloud<Normal>);
    cnowp->resize(cnow->size()); cobjp->resize(cobj->size());
    cnown->resize(cnow->size()); cobjn->resize(cobj->size());
#pragma omp parallel for
    for(size_t i=0; i<cnow->size(); i++){
        cnowp->points[i].x = cnow->points[i].x; cnowp->points[i].y = cnow->points[i].y; cnowp->points[i].z = cnow->points[i].z;
        cnowp->points[i].r = cnow->points[i].r; cnowp->points[i].g = cnow->points[i].g; cnowp->points[i].b = cnow->points[i].b;
        cnown->points[i].normal_x = cnow->points[i].normal_x; cnown->points[i].normal_y = cnow->points[i].normal_y; cnown->points[i].normal_z = cnow->points[i].normal_z;
    }
#pragma omp parallel for
    for(size_t i=0; i<cobj->size(); i++){
        cobjp->points[i].x = cobj->points[i].x; cobjp->points[i].y = cobj->points[i].y; cobjp->points[i].z = cobj->points[i].z;
        cobjp->points[i].r = cobj->points[i].r; cobjp->points[i].g = cobj->points[i].g; cobjp->points[i].b = cobj->points[i].b;
        cobjn->points[i].normal_x = cobj->points[i].normal_x; cobjn->points[i].normal_y = cobj->points[i].normal_y; cobjn->points[i].normal_z = cobj->points[i].normal_z;
    }

    // Encontrando Keypoints para as duas nuvens
    float radius = 0.15;
    PointCloud<PointXYZI>::Ptr keypoints_now (new PointCloud<PointXYZI>);
    PointCloud<PointXYZI>::Ptr keypoints_obj (new PointCloud<PointXYZI>);
    HarrisKeypoint3D<PointT, PointXYZI> harris;
    harris.setNumberOfThreads(8);
    harris.setRefine(true);
    harris.setThreshold(1e-5);
    harris.setRadius(radius);

    harris.setInputCloud(cnowp);
    harris.setNormals(cnown);
    harris.compute(*keypoints_now);
    harris.setInputCloud(cobjp);
    harris.setNormals(cobjn);
    harris.compute(*keypoints_obj);

    // Estimando descritores para os Keypoints
    FPFHEstimation<PointXYZI, Normal, FPFHSignature33> fpfh_est;
    search::KdTree<PointXYZI>::Ptr tree (new search::KdTree<PointXYZI>);
    fpfh_est.setSearchMethod(tree);

    fpfh_est.setInputCloud(keypoints_now);
    fpfh_est.setInputNormals(cnown);
    PointCloud<PointXYZI>::Ptr now_xyzi (new PointCloud<PointXYZI> ());
    PointCloudXYZRGBtoXYZI(*cnowp, *now_xyzi);
    fpfh_est.setSearchSurface(now_xyzi);
    fpfh_est.setRadiusSearch(radius);
    PointCloud<FPFHSignature33>::Ptr fpfhsnow (new PointCloud<FPFHSignature33>);
    fpfh_est.compute(*fpfhsnow);

    fpfh_est.setInputCloud(keypoints_obj);
    fpfh_est.setInputNormals(cobjn);
    PointCloud<PointXYZI>::Ptr obj_xyzi (new PointCloud<PointXYZI> ());
    PointCloudXYZRGBtoXYZI(*cobjp, *obj_xyzi);
    fpfh_est.setSearchSurface(obj_xyzi);
    fpfh_est.setRadiusSearch(radius);
    PointCloud<FPFHSignature33>::Ptr fpfhsobj (new PointCloud<FPFHSignature33>);
    fpfh_est.compute(*fpfhsobj);

    // Computando as correspondencias
    CorrespondenceEstimation<FPFHSignature33, FPFHSignature33> est;
    est.setInputSource(fpfhsnow);
    est.setInputTarget(fpfhsobj);
    CorrespondencesPtr all_correspondences (new Correspondences);
    est.determineReciprocalCorrespondences(*all_correspondences);

    // Rejeitando correspondencias por distancia
    CorrespondenceRejectorDistance rej;
    rej.setInputSource<PointXYZI>(keypoints_now);
    rej.setInputTarget<PointXYZI>(keypoints_obj);
    rej.setMaximumDistance(6*radius);
    rej.setInputCorrespondences(all_correspondences);
    CorrespondencesPtr remaining_correspondences (new Correspondences);
    rej.getCorrespondences(*remaining_correspondences);

    // Estimando a transformacao
    Matrix4f transf;
    TransformationEstimationSVD<PointXYZI, PointXYZI> trans_est;
    trans_est.estimateRigidTransformation(*keypoints_now, *keypoints_obj, *remaining_correspondences, transf);

    if(remaining_correspondences->size() < 6)
        transf = Matrix4f::Identity();

    return transf;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void RegisterObjectOptm::readNVM(string folder, string nome, vector<string> &clouds, vector<string> &images, vector<Matrix4f> &poses, float &foco){
    // Objetos para leitura das linhas
    ifstream nvm;
    string linha_atual;
    int conta_linha = 0;
    // Abrindo o arquivo
    string arquivo = folder + nome + ".nvm";
    nvm.open(arquivo);
    // Varrendo o arquivo
    if(nvm.is_open()){
        // For ao longo das linhas, ler as poses
        while(getline(nvm, linha_atual)){
            if(!linha_atual.empty()){
                conta_linha++; // atualiza aqui para pegar o numero 1 na primeira e assim por diante

                if(conta_linha >= 3){ // A partir daqui tem cameras
                    // Separando a string de entrada em nome da imagem e os dados numericos
                    int pos = linha_atual.find_first_of(' ');
                    string path = linha_atual.substr(0, pos);
                    string numericos = linha_atual.substr(pos+1);
                    // Adicionando o nome da imagem no vetor de nomes de imagem
                    images.push_back(path.substr(path.find_last_of("/")+1));
                    // Pegando o mesmo nome para a nuvem e adicionando no vetor de nuvens
                    string nome_imagem_basear = path.substr(path.find_last_of("/")+1);
                    string identidade_numerica_nuvem = nome_imagem_basear.substr(nome_imagem_basear.find_last_of("_")+1);
                    string nome_nuvem = "pf_"+identidade_numerica_nuvem.substr(0, identidade_numerica_nuvem.find_last_of("."))+".ply";
                    clouds.push_back(nome_nuvem);
                    // Elementos numericos divididos por espaços
                    istringstream ss(numericos);
                    vector<string> results((std::istream_iterator<string>(ss)), std::istream_iterator<string>());
                    // Elementos da camera e matriz para transformar a nuvem em si
                    Quaternion<float> q;
                    Matrix3f rot;
                    Vector3f C;
                    Matrix4f T = Matrix4f::Identity();
                    // Foco da camera
                    foco = stof(results.at(0));
                    // Quaternion da orientacao da camera
                    q.w() = stof(results.at(1)); q.x() = stof(results.at(2));
                    q.y() = stof(results.at(3)); q.z() = stof(results.at(4));
                    rot = q.matrix();
                    // Centro da camera antigo
                    C(0) = stof(results.at(5)); C(1) = stof(results.at(6)); C(2) = stof(results.at(7));
                    // Vetor de translaçao
                    T.block<3,1>(0, 3) = rot*C;
                    // Matriz de rotacao
                    T.block<3,3>(0, 0) = rot.transpose();
                    // Adicionando no vetor de poses
                    poses.push_back(T);
                } // Fim do if para iteracao de linhas

            }
        } // Fim do while linhas
    }
}
