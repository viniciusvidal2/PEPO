#include "../include/sfm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
SFM::SFM(string p1, string p2):pasta_src(p1), pasta_tgt(p2){
  // Vamos debugar ou nao
  debug = false;
  // Calibracao da camera
  K = Mat::zeros(3, 3, CV_64FC1);
  // Nome das nuvens
  nomes_nuvens = {"004", "008", "012", "016", "020", "024",
                  "028", "032", "036", "040", "044", "048"};
#pragma omp parallel for
  for(int i=0; i<nomes_nuvens.size(); i++)
    nomes_nuvens[i] = "pf_" + nomes_nuvens[i] + ".ply";
  // Iniciando ponteiros de nuvem
  cloud_src = (PointCloud<PointTN>::Ptr) new PointCloud<PointTN>();
  cloud_tgt = (PointCloud<PointTN>::Ptr) new PointCloud<PointTN>();
  // Inicia transformacao estimada
  Tsvd = Matrix4f::Identity();
}
/////////////////////////////////////////////////////////////////////////////////////////////////
SFM::~SFM(){
  ros::shutdown();
  ros::waitForShutdown();
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void SFM::obter_dados(vector<string> linhas_src, vector<string> linhas_tgt){
  if(linhas_src.size() < 2 || linhas_tgt.size() < 2){
    ROS_ERROR("Os arquivos .sfm nao foram bem lidos!");
    return;
  }
  Vector2f foco, centro_otico;
  for (auto s:linhas_tgt) {
    istringstream iss(s);
    vector<string> splits(istream_iterator<string>{iss}, istream_iterator<string>());
    // Nome
    string nome_fim = splits[0].substr(splits[0].find_last_of('/') + 1, splits[0].size() - 1);
    imagens_tgt.push_back(pasta_tgt + nome_fim);

    // Rotation
    Matrix3f r;
    r << stof(splits[1]), stof(splits[2]), stof(splits[3]),
        stof(splits[4]), stof(splits[5]), stof(splits[6]),
        stof(splits[7]), stof(splits[8]), stof(splits[9]);
    rots_tgt.push_back(r);

    // Foco e centro para matriz K - igual, sempre mesma camera
    foco << stof(splits[13]), stof(splits[14]);
    centro_otico << stof(splits[15]), stof(splits[16]);
  }
//  K.at<double>(0, 0) = foco(0)        ; K.at<double>(1, 1) = foco(1);
//  K.at<double>(0, 2) = centro_otico(0); K.at<double>(1, 2) = centro_otico(1);
//  K.at<double>(2, 2) = 1;
  K.at<double>(0, 0) = 375; K.at<double>(1, 1) = 374;
  K.at<double>(0, 2) = 245; K.at<double>(1, 2) = 142;
  K.at<double>(2, 2) = 1;
  for (auto s:linhas_src) {
    istringstream iss(s);
    vector<string> splits(istream_iterator<string>{iss}, istream_iterator<string>());
    // Nome
    string nome_fim = splits[0].substr(splits[0].find_last_of('/') + 1, splits[0].size() - 1);
    imagens_src.push_back(pasta_src + nome_fim);

    //rotation
    Matrix3f r;
    r << stof(splits[1]), stof(splits[2]), stof(splits[3]),
        stof(splits[4]), stof(splits[5]), stof(splits[6]),
        stof(splits[7]), stof(splits[8]), stof(splits[9]);
    rots_src.push_back(r);
  }
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void SFM::calcular_features_surf(){
  // Resize do vetor geral de kpts e descritores para a quantidade de imagens de cada caso
  descp_src.resize(imagens_src.size()); descp_tgt.resize(imagens_tgt.size());
  kpts_src.resize( imagens_src.size()); kpts_tgt.resize( imagens_tgt.size());

#pragma omp parallel for
  for(int i=0; i<descp_src.size(); i++){
    // Iniciando Keypoints e Descritores atuais
    vector<KeyPoint> kptgt, kpsrc;
    Mat dtgt, dsrc;

    // Ler a imagem inicial
    Mat imtgt = imread(imagens_tgt[i], IMREAD_COLOR);
    Mat imsrc = imread(imagens_src[i], IMREAD_COLOR);
    resize(imtgt, imtgt, Size(imtgt.cols/4, imtgt.rows/4));
    resize(imsrc, imsrc, Size(imsrc.cols/4, imsrc.rows/4));

    // Salvar aqui as dimensoes da imagem para a sequencia do algoritmo
    imcols = imtgt.cols; imrows = imtgt.rows;

    // Descritores SURF calculados
    float min_hessian = 1500;
    Ptr<xfeatures2d::SURF> surf = xfeatures2d::SURF::create(min_hessian);
    surf->detectAndCompute(imtgt, Mat(), kptgt, dtgt);
    surf->detectAndCompute(imsrc, Mat(), kpsrc, dsrc);

    // Salvando no vetor de keypoints
    kpts_tgt[i] = kptgt;
    kpts_src[i] = kpsrc;

    // Salvando no vetor de cada um os descritores
    descp_tgt[i] = dtgt;
    descp_src[i] = dsrc;
  }
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void SFM::surf_matches_matrix_encontrar_melhor(){
  // Ajustar matriz de quantidade de matches
  MatrixXi matches_count(descp_tgt.size(), descp_src.size());
  vector<vector<  vector<DMatch> >> matriz_matches(descp_tgt.size());
  for(int i=0; i<matriz_matches.size(); i++)
    matriz_matches.at(i).resize(descp_src.size());

  // Matcher de FLANN
  cv::Ptr<DescriptorMatcher> matcher;
  matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);

  // Para cada combinacao de imagens, fazer match e salvar quantidade final para ver qual
  // a melhor depois
#pragma omp parallel for
  for(int i=0; i<descp_tgt.size(); i++){
    for(int j=0; j<descp_src.size(); j++){
      vector<vector<DMatch>> matches;
      vector<DMatch> good_matches;
      matcher->knnMatch(descp_src[j], descp_tgt[i], matches, 2);
      for (size_t k = 0; k < matches.size(); k++){
        if (matches.at(k).size() >= 2){
          if (matches.at(k).at(0).distance < 0.75*matches.at(k).at(1).distance) // Se e bastante unica frente a segunda colocada
            good_matches.push_back(matches.at(k).at(0));
        }
      }
      // Filtrar por matches que nao sejam muito horizontais
      this->filterMatchesLineCoeff(good_matches, kpts_tgt[i], kpts_src[j], imcols, DEG2RAD(20));
      // Anota quantas venceram nessa combinacao
      matches_count(i, j)        = good_matches.size();
      matriz_matches.at(i).at(j) = good_matches;
    }
  }

  if(debug)
    cout << "\nMatriz de matches:\n" << matches_count << endl << "\nMaximo de matches: " << matches_count.maxCoeff() << endl;

  // Atraves do melhor separar matches daquelas vistas
  int max_matches = matches_count.maxCoeff();
  for(int i=0; i<descp_tgt.size(); i++){
    for(int j=0; j<descp_src.size(); j++){
      if(matches_count(i, j) == max_matches){
        best_matches = matriz_matches.at(i).at(j);
        im_tgt_indice = i; im_src_indice = j;
        break;
      }
    }
  }

  // Libera memoria
  descp_tgt.clear(); descp_src.clear();

  // Pegar somente bons kpts
  vector<KeyPoint> curr_kpts_tgt = kpts_tgt[im_tgt_indice], curr_kpts_src = kpts_src[im_src_indice];
  for(auto m:best_matches){
    best_kptgt.emplace_back(curr_kpts_tgt[m.trainIdx]);
    best_kpsrc.emplace_back(curr_kpts_src[m.queryIdx]);
    cout << best_kptgt[best_kptgt.size()-1].pt << " " << best_kpsrc[best_kpsrc.size()-1].pt << " " << m.trainIdx << " " << curr_kpts_tgt.size() << " " << m.queryIdx << " " << curr_kpts_src.size() << endl;
  }

  // Plotar imagens
//  if(debug){
//    Mat im_matches;
//    Mat im1 = imread(imagens_tgt[im_tgt_indice], IMREAD_COLOR);
//    Mat im2 = imread(imagens_src[im_src_indice], IMREAD_COLOR);
//    resize(im1, im1, Size(im1.cols/4, im1.rows/4));
//    resize(im2, im2, Size(im2.cols/4, im2.rows/4));
////    drawMatches(im1, kpts_tgt[im_tgt_indice], im2, kpts_src[im_src_indice], best_matches, im_matches);
////    imshow("Best Matches", im_matches);
////    waitKey();
//    imwrite(pasta_src+"matches.png", im_matches);
//  }

  // Libera memoria
  kpts_tgt.clear(); kpts_src.clear();
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void SFM::obter_transformacao_final(Matrix4f &T, PointCloud<PointTN>::Ptr tgt, PointCloud<PointTN>::Ptr src){
  // Lendo as nuvens correspondentes
  this->ler_nuvens_correspondentes();

  if(debug)
    cout << "\nAplicando transformacao final ..." << endl;
  // Calcular rotacao relativa entre o frame src e tgt, src -> tgt
  // Conta necessaria: 2_R^1 = inv(in_R^2)*in_R^1
  R_src_tgt = rots_src[im_src_indice]*rots_tgt[im_tgt_indice].inverse();
  // Aplicar a rotacao relativa pela esquerda
  R_src_tgt = Rrel * R_src_tgt;

  // Transformacao final (so rotacao)
  T.block<3,3>(0, 0) = R_src_tgt;
  // Transformacao final (chute em translacao)
  trel << 0, 0, 1;
  trel = (Rrel*rots_tgt[im_tgt_indice].inverse()).transpose()*trel;
//  trel = R_src_tgt*Rrel.inverse()*trel;
  T.block<3,1>(0, 3) = trel*12;

  // Transformar a nuvem source com a transformacao estimada
  transformPointCloudWithNormals<PointTN>(*cloud_src, *cloud_src, T);

  if(debug){
    // Salvar ambas as nuvens na pasta source pra comparar
    savePLYFileBinary<PointTN>(pasta_src+"debug_tgt.ply", *cloud_tgt);
    savePLYFileBinary<PointTN>(pasta_src+"debug_src.ply", *cloud_src);
  }

  *tgt = *cloud_tgt;
  *src = *cloud_src;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void SFM::ler_nuvens_correspondentes(){
  if(debug)
    cout << "\nLendo nuvens ..." << endl;
  string ntgt = "acumulada.ply";
  string nsrc = "acumulada.ply";
  loadPLYFile<PointTN>(pasta_tgt+ntgt, *cloud_tgt);
  loadPLYFile<PointTN>(pasta_src+nsrc, *cloud_src);
  // Rotacionar a nuvem de acordo com a entrada - frame da camera
  // e filtrar o que esta nas costas
  Quaternion<float> qt(rots_tgt[im_tgt_indice]);
  transformPointCloudWithNormals<PointTN>(*cloud_tgt, *cloud_tgt, Vector3f::Zero(), qt);
  PassThrough<PointTN> pass;
  pass.setFilterFieldName("z");
  pass.setFilterLimits(1, 100);
  pass.setInputCloud(cloud_tgt);
  pass.filter(*cloud_tgt);
  Quaternion<float> qs(rots_src[im_src_indice]);
  transformPointCloudWithNormals<PointTN>(*cloud_src, *cloud_src, Vector3f::Zero(), qs);
  pass.setInputCloud(cloud_src);
  pass.filter(*cloud_src);

  // Filtrar pelo angulo de visao da camera, considerando ai thresh graus
  // redondos por exemplo em ambas as direcoes
  float thresh = 80.0/2.0;
  PointIndices::Ptr indt (new PointIndices);
  float d;
  for(size_t i=0; i<cloud_tgt->size(); i++){
    d = sqrt( pow((*cloud_tgt)[i].x, 2) + pow((*cloud_tgt)[i].y, 2) + pow((*cloud_tgt)[i].z, 2) );
    if(abs(acos( (*cloud_tgt)[i].z / d )) < DEG2RAD(thresh))
      indt->indices.push_back(i);
  }
  ExtractIndices<PointTN> extract;
  extract.setIndices(indt);
  extract.setInputCloud(cloud_tgt);
  extract.filter(*cloud_tgt);
  PointIndices::Ptr inds (new PointIndices);
  for(size_t i=0; i<cloud_src->size(); i++){
    d = sqrt( pow((*cloud_src)[i].x, 2) + pow((*cloud_src)[i].y, 2) + pow((*cloud_src)[i].z, 2) );
    if(abs(acos( (*cloud_src)[i].z / d )) < DEG2RAD(thresh))
      inds->indices.push_back(i);
  }
  extract.setIndices(inds);
  extract.setInputCloud(cloud_src);
  extract.filter(*cloud_src);

  // Trazer as nuvens de volta aos seus respectivos frames
  transformPointCloudWithNormals<PointTN>(*cloud_tgt, *cloud_tgt, Vector3f::Zero(), qt.inverse());
  transformPointCloudWithNormals<PointTN>(*cloud_src, *cloud_src, Vector3f::Zero(), qs.inverse());
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void SFM::obter_correspondencias_3D_e_T(){
  // Ler nuvens referentes as imagens correspondentes
  this->ler_nuvens_correspondentes();
  // Thresh para aceitar o ponto (pixels)
  float thresh = 10;
  // Matriz intrinseca em Eigen
  Matrix3f K_;
  cv2eigen(K, K_);
  /// Para nuvem tgt
  ///
  if(debug)
    cout << "\nProcurando pontos na tgt ..." << endl;
  // Nuvem suporte com mesma quantidade de pontos das correspondencias
  PointCloud<PointTN>::Ptr ct (new PointCloud<PointTN>);
  ct->resize(best_kptgt.size());
  // Projetar cada ponto em paralelo - se chegar perto o suficiente do pixel, e isso ai mesmo
#pragma omp parallel for
  for(size_t i = 0; i < cloud_tgt->size(); i++){
    Vector3f X_{cloud_tgt->points[i].x, cloud_tgt->points[i].y, cloud_tgt->points[i].z};
    Vector3f X;
    X = rots_tgt[im_tgt_indice]*X_;
    X = K_*X;
    if(X(2) > 0){
      X = X/X(2);
      // Se caiu dentro da imagem
      if(floor(X(0)) > 0 && floor(X(0)) < imcols && floor(X(1)) > 0 && floor(X(1)) < imrows){
        // Procura no vetor de correspondencias se perto de alguma, se sim anota esse ponto naquela posicao do pixel no vetor
#pragma omp parallel for
        for(int j=0; j<best_kptgt.size(); j++){
          if(sqrt(pow(best_kptgt[j].pt.x - X(0), 2) + pow(best_kptgt[j].pt.y - X(1), 2)) < thresh){
            ct->points[j] = cloud_tgt->points[i];
            ct->points[j].g = 255; ct->points[j].r = 0; ct->points[j].b = 0;
          }
        }
      }
    }
  }
  /// Para nuvem src
  ///
  if(debug)
    cout << "\nProcurando pontos na src ..." << endl;
  // Nuvem suporte com mesma quantidade de pontos das correspondencias
  PointCloud<PointTN>::Ptr cs (new PointCloud<PointTN>);
  cs->resize(best_kpsrc.size());
  // Projetar cada ponto em paralelo - se chegar perto o suficiente do pixel, e isso ai mesmo
#pragma omp parallel for
  for(size_t i = 0; i < cloud_src->size(); i++){
    Vector3f X_{cloud_src->points[i].x, cloud_src->points[i].y, cloud_src->points[i].z};
    Vector3f X;
    X = rots_src[im_src_indice]*X_;
    X = K_*X;
    if(X(2) > 0){
      X = X/X(2);
      // Se caiu dentro da imagem
      if(floor(X(0)) > 0 && floor(X(0)) < imcols && floor(X(1)) > 0 && floor(X(1)) < imrows){
        // Procura no vetor de correspondencias se perto de alguma, se sim anota esse ponto naquela posicao do pixel no vetor
#pragma omp parallel for
        for(int j=0; j<best_kpsrc.size(); j++){
          if(sqrt(pow(best_kpsrc[j].pt.x - X(0), 2) + pow(best_kpsrc[j].pt.y - X(1), 2)) < thresh){
            cs->points[j] = cloud_src->points[i];
            cs->points[j].g = 0; cs->points[j].r = 255; cs->points[j].b = 255;
          }
        }
      }
    }
  }
  /// Estimar a transformacao entre os pontos correspondentes
  ///
  // Filtrar por pontos na origem nao vistos
  PointIndices::Ptr indices (new PointIndices);
  for(size_t i=0; i<cs->size(); i++){
    if((*cs)[i].x != 0 && (*cs)[i].y != 0 && (*cs)[i].z != 0 &&
       (*ct)[i].x != 0 && (*ct)[i].y != 0 && (*ct)[i].z != 0)
      indices->indices.push_back(i);
  }
  if(debug)
    cout << "\nMatches que vingaram nas duas nuvens: " << indices->indices.size() << endl;
  ExtractIndices<PointTN> extract;
  extract.setIndices(indices);
  extract.setInputCloud(ct);
  extract.filter(*ct);
  extract.setInputCloud(cs);
  extract.filter(*cs);
  // Estimar transformacao por SVD se houver pontos suficientes
  if(indices->indices.size() >= 5){
    TransformationEstimationSVD<PointTN, PointTN> svd;
    svd.estimateRigidTransformation(*cs, *ct, Tsvd);
  }

  if(debug){
    // Mostrar transformacao aproximada
    cout << "\nTransformacao por svd:\n" << Tsvd << endl;
    // Transformar a nuvem source com a transformacao estimada
    transformPointCloudWithNormals<PointTN>(*cloud_src, *cloud_src, Tsvd);
    transformPointCloudWithNormals<PointTN>(*cs, *cs, Tsvd);
    // Salvar ambas as nuvens na pasta source pra comparar
    savePLYFileASCII<PointTN>(pasta_src+"debug_tgt.ply", *cloud_tgt);
    savePLYFileASCII<PointTN>(pasta_src+"debug_src.ply", *cloud_src);
    savePLYFileASCII<PointTN>(pasta_src+"kpts_tgt.ply", *ct);
    savePLYFileASCII<PointTN>(pasta_src+"kpts_src.ply", *cs);
  }

}
/////////////////////////////////////////////////////////////////////////////////////////////////
void SFM::set_debug(bool b){
  debug = b;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void SFM::filterMatchesLineCoeff(vector<DMatch> &matches, vector<KeyPoint> kpref, vector<KeyPoint> kpnow, float width, float n){
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
  // Filtrar o vetor de matches na posicao que os coeficientes estejam fora por ngraus
  vector<DMatch> temp;
  for(int i=0; i<coefs.size(); i++){
    if(abs(coefs[i]) < n)
      temp.push_back(matches[i]);
  }
  matches = temp;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void SFM::calcular_pose_relativa(){
  // Converter os pontos para o formato certo
  vector<Point2f> kptgt(best_kptgt.size()), kpsrc(best_kpsrc.size());
#pragma omp parallel for
  for(int i=0; i<best_kptgt.size(); i++){
    kptgt[i] = best_kptgt[i].pt;
    kpsrc[i] = best_kpsrc[i].pt;
  }
  // Calcular matriz fundamental
  Mat F = findFundamentalMat(kpsrc, kptgt); // Transformacao da src para a tgt
  // Matriz Essencial
  Mat E = K.t()*F*K;
  // Recupera pose - matriz de rotacao e translacao
  Mat r, t;
  int inliers;
  inliers = recoverPose(E, kpsrc, kptgt, K, r, t);
  if(debug)
    cout << "\nInliers:  " << inliers << " de " << best_kpsrc.size() << endl;
  // Passar para Eigen e seguir processo
  cv2eigen(r, Rrel);
  cv2eigen(t, trel);
}
/////////////////////////////////////////////////////////////////////////////////////////////////
Matrix4f SFM::icp(PointCloud<PointTN>::Ptr ctgt, PointCloud<PointTN>::Ptr csrc, float vs, int its){
    Matrix4f Ticp = Matrix4f::Identity();
    vs = vs/100.0;

    // Reduzindo ainda mais as nuvens pra nao dar trabalho assim ao icp
    PointCloud<PointTN>::Ptr tgttemp(new PointCloud<PointTN>);
    PointCloud<PointTN>::Ptr srctemp(new PointCloud<PointTN>);
    if(vs > 0){
      VoxelGrid<PointTN> voxel;
      voxel.setLeafSize(vs, vs, vs);
      voxel.setInputCloud(ctgt);
      voxel.filter(*tgttemp);
      voxel.setInputCloud(csrc);
      voxel.filter(*srctemp);
    } else {
      *tgttemp = *ctgt;
      *srctemp = *csrc;
    }

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
    icp.setMaxCorrespondenceDistance(vs*20);
    // Alinhando
    PointCloud<PointTN> dummy;
    icp.align(dummy, Matrix4f::Identity());
    // Obtendo a transformacao otimizada e aplicando
    if(icp.hasConverged())
        Ticp = icp.getFinalTransformation();
    transformPointCloudWithNormals<PointTN>(*csrc, *csrc, Ticp);

    if(debug)
      savePLYFileBinary<PointTN>(pasta_src+"src_final.ply", *csrc);

    return Ticp;
}
