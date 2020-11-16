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
//  pass.filter(*cin);
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
//  *tgttemp = *ctgt;
//  *srctemp = *csrc;

  // Criando o otimizador de ICP
  GeneralizedIterativeClosestPoint<PointTN, PointTN> icp;
  //    IterativeClosestPoint<PointTN, PointTN> icp;
  icp.setUseReciprocalCorrespondences(true);
  icp.setInputTarget(tgttemp);
  icp.setInputSource(srctemp);
  //    icp.setRANSACIterations(30);
  icp.setMaximumIterations(its); // Chute inicial bom 10-100
  icp.setTransformationEpsilon(1*1e-8);
  icp.setEuclideanFitnessEpsilon(1*1e-11);
  icp.setMaxCorrespondenceDistance(vs*3);
  // Alinhando
  PointCloud<PointTN> dummy;
  icp.align(dummy, Matrix4f::Identity());
  // Obtendo a transformacao otimizada e aplicando
  if(icp.hasConverged())
    Ticp = icp.getFinalTransformation();

  return Ticp;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
Matrix4f RegisterObjectOptm::gicp6d(PointCloud<PointTN>::Ptr ctgt, PointCloud<PointTN>::Ptr csrc, float vs, int its){
  Matrix4f Ticp = Matrix4f::Identity();

  // Reduzindo ainda mais as nuvens pra nao dar trabalho assim ao icp
  PointCloud<PointXYZRGBA>::Ptr tgttemp(new PointCloud<PointXYZRGBA>);
  PointCloud<PointXYZRGBA>::Ptr srctemp(new PointCloud<PointXYZRGBA>);
//  tgttemp->resize(ctgt->size()); srctemp->resize(csrc->size());
//#pragma omp parallel for
//  for(size_t i=0; i<ctgt->size(); i++){
//    tgttemp->points[i].x = ctgt->points[i].x; tgttemp->points[i].y = ctgt->points[i].y; tgttemp->points[i].z = ctgt->points[i].z;
//    tgttemp->points[i].r = ctgt->points[i].r; tgttemp->points[i].b = ctgt->points[i].b; tgttemp->points[i].g = ctgt->points[i].g;
//  }
//#pragma omp parallel for
//  for(size_t i=0; i<csrc->size(); i++){
//    srctemp->points[i].x = csrc->points[i].x; srctemp->points[i].y = csrc->points[i].y; srctemp->points[i].z = csrc->points[i].z;
//    srctemp->points[i].r = csrc->points[i].r; srctemp->points[i].b = csrc->points[i].b; srctemp->points[i].g = csrc->points[i].g;
//  }
  copyPointCloud(*ctgt, *tgttemp);
  copyPointCloud(*csrc, *srctemp);
  VoxelGrid<PointXYZRGBA> voxel;
  voxel.setLeafSize(vs, vs, vs);
  voxel.setInputCloud(tgttemp);
  voxel.filter(*tgttemp);
  voxel.setInputCloud(srctemp);
  voxel.filter(*srctemp);
  StatisticalOutlierRemoval<PointXYZRGBA> sor;
  sor.setMeanK(30);
  sor.setStddevMulThresh(2);
  sor.setNegative(false);
  sor.setInputCloud(srctemp);
  sor.filter(*srctemp);
  sor.setInputCloud(tgttemp);
  sor.filter(*tgttemp);

  // Criando o otimizador de ICP
  GeneralizedIterativeClosestPoint6D icp;
  icp.setUseReciprocalCorrespondences(true);
  icp.setInputTarget(tgttemp);
  icp.setInputSource(srctemp);
  //    icp.setRANSACIterations(30);
  icp.setMaximumIterations(its); // Chute inicial bom 10-100
  icp.setTransformationEpsilon(1*1e-6);
  icp.setEuclideanFitnessEpsilon(1*1e-6);
  icp.setMaxCorrespondenceDistance(vs*3);
  // Alinhando
  PointCloud<PointXYZRGBA> dummy;
  icp.align(dummy, Matrix4f::Identity());
  // Obtendo a transformacao otimizada e aplicando
  if(icp.hasConverged())
    Ticp = icp.getFinalTransformation();

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
      if (matches.at(k).at(0).distance < 0.7*matches.at(k).at(1).distance) // Se e bastante unica frente a segunda colocada
        good_matches.push_back(matches.at(k).at(0));
    }
  }

  /// Filtrar os matches por matriz fundamental ///

  // Converter os pontos para o formato certo
  vector<Point2f> ppref(good_matches.size()), ppnow(good_matches.size());
#pragma omp parallel for
  for(int i=0; i<good_matches.size(); i++){
    ppref[i] = kpref[good_matches[i].trainIdx].pt;
    ppnow[i] = kpnow[good_matches[i].queryIdx].pt;
  }
  // Calcular matriz fundamental
  Mat F = findFundamentalMat(ppnow, ppref);
  // Calcular pontos que ficam por conferencia da matriz F
  Matrix3f F_;
  cv2eigen(F, F_);
  vector<Point2f> tempt, temps;
  for(int i=0; i<ppnow.size(); i++){
    Vector3f pt{ppref[i].x, ppref[i].y, 1}, ps = {ppnow[i].x, ppnow[i].y, 1};
    MatrixXf erro = pt.transpose()*F_*ps;
    if(abs(erro(0, 0)) < 0.2){
      tempt.push_back(ppref[i]); temps.push_back(ppnow[i]);
    }
  }
  ppnow = temps; ppref = tempt;
  /////////////////////////////////////////////////

//  Mat imnow2, imref2;
//  imnow.copyTo(imnow2);
//  imref.copyTo(imref2);
//  for(int i=0; i<ppref.size(); i++){
//    int r = rand()%255, b = rand()%255, g = rand()%255;
//    circle(imref2, Point(ppref[i].x, ppref[i].y), 8, Scalar(r, g, b), FILLED, LINE_8);
//    circle(imnow2, Point(ppnow[i].x, ppnow[i].y), 8, Scalar(r, g, b), FILLED, LINE_8);
//  }
//  imshow("targetc", imref2);
//  imshow("sourcec", imnow2);
//  waitKey(0);

  // Dados intrinsecos da camera
  float fx = 1427.1, fy = 1449.4, cx = 987.9, cy = 579.4;

  // Se houveram matches suficientes
  if(ppref.size() > 6){
    // Aloca espaco para nuvens de saida de match 3D
    cmr->resize(ppref.size()); cmn->resize(ppref.size());
    // Calcular pontos 3D de cada match em now
    vector<int> istherepoint_now(ppnow.size());
    octree::OctreePointCloudSearch<PointTN> oct(0.02);
    oct.setInputCloud(cnow);
    oct.addPointsFromInputCloud();
#pragma omp parallel for
    for(int i=0; i<ppnow.size(); i++){
      float u = ppnow[i].x, v = ppnow[i].y;
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
    vector<int> istherepoint_ref(ppref.size());
    octree::OctreePointCloudSearch<PointTN> oct2(0.02);
    oct2.setInputCloud(cref);
    oct2.addPointsFromInputCloud();
#pragma omp parallel for
    for(int i=0; i<ppref.size(); i++){
      float u = ppref[i].x, v = ppref[i].y;
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

}
/////////////////////////////////////////////////////////////////////////////////////////////////
Matrix4f RegisterObjectOptm::optmizeTransformSVD(PointCloud<PointTN>::Ptr cref, PointCloud<PointTN>::Ptr cnow){
  pcl::Correspondences corresp;
  for(size_t i=0; i<cref->size(); i++){
    pcl::Correspondence corr;
    corr.index_query = i;
    corr.index_match = i;
    corresp.push_back(corr);
  }

  /// RANSAC BASED Correspondence Rejection
  pcl::CorrespondencesConstPtr correspond = boost::make_shared< pcl::Correspondences >(corresp);

  pcl::Correspondences corr;
  pcl::registration::CorrespondenceRejectorSampleConsensus< PointTN > Ransac_based_Rejection;
  Ransac_based_Rejection.setInputSource(cnow);
  Ransac_based_Rejection.setInputTarget(cref);
  double sac_threshold = 0.5;// default PCL value..can be changed and may slightly affect the number of correspondences
  Ransac_based_Rejection.setInlierThreshold(sac_threshold);
  Ransac_based_Rejection.setInputCorrespondences(correspond);
  Ransac_based_Rejection.getCorrespondences(corr);

  Matrix4f mat = Ransac_based_Rejection.getBestTransformation();

  return mat;

//  // Inicia estimador
//  registration::TransformationEstimationSVD<PointTN, PointTN> svd;
//  Matrix4f Tsvd;
//  // Estima a transformacao
//  svd.estimateRigidTransformation(*cnow, *cref, Tsvd);

//  return Tsvd;
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
