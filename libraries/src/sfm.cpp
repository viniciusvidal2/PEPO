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
  // Inicia a translacao calibrada laser -> camera
  t_laser_cam << 0.0226, 0.0938, 0.0221;
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
    float min_hessian = 2000;
    Ptr<xfeatures2d::SURF> surf = xfeatures2d::SURF::create(min_hessian);
    surf->detectAndCompute(imtgt, Mat(), kptgt, dtgt);
    surf->detectAndCompute(imsrc, Mat(), kpsrc, dsrc);
//    Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create();
//    sift->detectAndCompute(imtgt, Mat(), kptgt, dtgt);
//    sift->detectAndCompute(imsrc, Mat(), kpsrc, dsrc);

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
  MatrixXi matches_count = MatrixXi::Zero(descp_tgt.size(), descp_src.size());
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
      if(!descp_src[j].empty() && !descp_tgt[i].empty()){
        matcher->knnMatch(descp_src[j], descp_tgt[i], matches, 2);
        for (size_t k = 0; k < matches.size(); k++){
          if (matches.at(k).size() >= 2){
            if (matches.at(k).at(0).distance < 0.7*matches.at(k).at(1).distance) // Se e bastante unica frente a segunda colocada
              good_matches.push_back(matches.at(k).at(0));
          }
        }
        if(good_matches.size() > 0){
          // Filtrar keypoints repetidos
          this->filtrar_matches_keypoints_repetidos( kpts_tgt[i], kpts_src[j], good_matches);
          // Filtrar por matches que nao sejam muito horizontais
          this->filterMatchesLineCoeff(good_matches, kpts_tgt[i], kpts_src[j], imcols, DEG2RAD(30));

          // Anota quantas venceram nessa combinacao
          matches_count(i, j)        = good_matches.size();
          matriz_matches.at(i).at(j) = good_matches;
        }
      }
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
//    cout << best_kptgt[best_kptgt.size()-1].pt << " " << best_kpsrc[best_kpsrc.size()-1].pt << " " << m.trainIdx << " " << curr_kpts_tgt.size() << " " << m.queryIdx << " " << curr_kpts_src.size() << endl;
  }

  // Plotar imagens
  Mat im1 = imread(imagens_tgt[im_tgt_indice], IMREAD_COLOR);
  Mat im2 = imread(imagens_src[im_src_indice], IMREAD_COLOR);
  resize(im1, im1, Size(im1.cols/4, im1.rows/4));
  resize(im2, im2, Size(im2.cols/4, im2.rows/4));
  for(int i=0; i<best_kpsrc.size(); i++){
    int r = rand()%255, b = rand()%255, g = rand()%255;
    circle(im1, Point(best_kptgt[i].pt.x, best_kptgt[i].pt.y), 3, Scalar(r, g, b), FILLED, LINE_8);
    circle(im2, Point(best_kpsrc[i].pt.x, best_kpsrc[i].pt.y), 3, Scalar(r, g, b), FILLED, LINE_8);
  }
  imshow("targetc", im1);
  imshow("sourcec", im2);
  waitKey();

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

  // Transformacao final (so rotacao)
  T.block<3,3>(0, 0) = Rrel * R_src_tgt;
  // Transformacao final (em translacao)
  this->estimar_escala_translacao();
  T.block<3,1>(0, 3) = trel;

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

  if(debug)
    cout << "\nPronto a leitura." << endl;
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
  // Calcular pontos que ficam por conferencia da matriz F
  Matrix3f F_;
  cv2eigen(F, F_);
  vector<Point2f> tempt, temps;
  vector<int> indices_inliers;
  for(int i=0; i<kpsrc.size(); i++){
    Vector3f pt{kptgt[i].x, kptgt[i].y, 1}, ps = {kpsrc[i].x, kpsrc[i].y, 1};
    MatrixXf erro = pt.transpose()*F_*ps;
    if(abs(erro(0, 0)) < 0.2){
      tempt.push_back(kptgt[i]); temps.push_back(kpsrc[i]);
      indices_inliers.push_back(i);
    }
  }
  kpsrc = temps; kptgt = tempt;

  // Segue so com os inliers dentre os best_kpts
  vector<KeyPoint> temp_kptgt, temp_kpsrc;
  for(auto i:indices_inliers){
    temp_kptgt.push_back(best_kptgt[i]); temp_kpsrc.push_back(best_kpsrc[i]);
  }
  best_kptgt = temp_kptgt; best_kpsrc = temp_kpsrc;

  // Matriz Essencial
  //  Mat E = K.t()*F*K;
  Mat E = findEssentialMat(kpsrc, kptgt, K);
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
/////////////////////////////////////////////////////////////////////////////////////////////////
void SFM::estimar_escala_translacao(){
  // Parametros intrinsecos da camera explicitados para facilitar
  float fx = K.at<double>(0, 0), fy = K.at<double>(1, 1), cx = K.at<double>(0, 2), cy = K.at<double>(1, 2);
  // Lista de pontos 3D correspondentes a partir de cada match
  vector<Vector3f> corresp_3d_src(best_kpsrc.size()), corresp_3d_tgt(best_kptgt.size());

  PointCloud<PointTN>::Ptr temps (new PointCloud<PointTN>);
  PointCloud<PointTN>::Ptr tempt (new PointCloud<PointTN>);
  Quaternion<float> qs(rots_src[im_src_indice]);
  Quaternion<float> qt(rots_tgt[im_tgt_indice]);
  transformPointCloudWithNormals(*cloud_src, *temps, Vector3f::Zero(), qs);
  transformPointCloudWithNormals(*cloud_tgt, *tempt, Vector3f::Zero(), qt);
  cout << "\nComecando a vasculhar os matches procurando por octree seus pontos na nuvem ..." << endl;
  // Para cada match
  for(int i=0; i<best_kpsrc.size(); i++){
    // Pegar o keypoint na imagem em questao
    float us, vs, ut, vt;
    us = best_kpsrc[i].pt.x; vs = best_kpsrc[i].pt.y;
    ut = best_kptgt[i].pt.x; vt = best_kptgt[i].pt.y;
    // Calcular a direcao no frame da camera para eles
    Vector3f dirs, dirt;
    dirs << (us - cx)/fx, -(vs - cy)/fy, 1;
//    cout << dirs << endl << endl;
    dirt << (ut - cx)/fx, -(vt - cy)/fy, 1;
    // Rotacionar o vetor para o frame local
//    dirs = rots_src[im_src_indice].transpose() * dirs;
//    dirt = rots_tgt[im_tgt_indice].transpose() * dirt;
    // Aplicar ray casting para saber em que parte da nuvem vai bater
    octree::OctreePointCloudSearch<PointTN> oct(0.2);
    octree::OctreePointCloudSearch<PointTN>::AlignedPointTVector aligns, alignt;

    oct.setInputCloud(temps);
    oct.addPointsFromInputCloud();
    oct.getIntersectedVoxelCenters(Vector3f::Zero(), dirs.normalized(), aligns, 1);
    // Se achou, adicionar na lista de pontos 3D naquele local
    if(aligns.size() > 0)
      corresp_3d_src[i] << aligns[0].x, aligns[0].y, aligns[0].z;
    else
      corresp_3d_src[i] << 0, 0, 0;

    oct.setInputCloud(tempt);
    oct.addPointsFromInputCloud();
    oct.getIntersectedVoxelCenters(Vector3f::Zero(), dirt.normalized(), alignt, 1);
    // Se achou, adicionar na lista de pontos 3D naquele local
    if(alignt.size() > 0)
      corresp_3d_tgt[i] << alignt[0].x, alignt[0].y, alignt[0].z;
    else
      corresp_3d_tgt[i] << 0, 0, 0;
  }

  // Para cada ponto resultante do ray cast
  vector<Vector3f> boas_translacoes;
  for(int i=0; i < corresp_3d_src.size(); i++){
    if(!(corresp_3d_src[i].norm() == 0) && !(corresp_3d_tgt[i].norm() == 0)){
      // Rotacionar ponto para o frame de src->tgt e obter translacao necessaria
      Vector3f t;
      t = corresp_3d_tgt[i] - Rrel*corresp_3d_src[i];
      // Se direcao casa com a inicial proposta pelo match das imagens, ok
      Vector3f Z{0, 0, 1}; // Eixos das imagens alinhados apos a rotacao src->tgt
      float theta = RAD2DEG( acos(t.dot(Z)/(t.norm()*Z.norm())) );
      cout << theta << endl;
      if(abs(theta) < 15 || abs(theta - 180) < 15){
        t = rots_tgt[im_tgt_indice].inverse()*t; // Levar para a orientacao original da tgt no espaco
        boas_translacoes.push_back(t);
      }
    }
  }
  cout << "\nQuantos pontos foram bem  " << boas_translacoes.size() << endl;
  // Tira a media do somatorio dos vetores e atribui ao que antes era a translacao
  // so em escala, agora vai para o mundo real
  Vector3f acc;
  for(auto t:boas_translacoes) acc += t;
  if(boas_translacoes.size() > 0) trel = acc/boas_translacoes.size(); else trel = 10*trel;

  if(debug){
//    for(int i=0; i<boas_translacoes.size(); i++)
//      cout << boas_translacoes[i] << endl;
    cout <<"\nT final: " << trel << endl;
  }

//  /// Projetar ambas as imagens na matriz - imagem de profundidade
//  ///
//  MatrixXf ds = MatrixXf::Zero(imrows, imcols);
//  MatrixXf dt = MatrixXf::Zero(imrows, imcols);
//  Matrix3f K_;
//  cv2eigen(K, K_);
//  // Imagem profundidade nuvem source
//#pragma omp parallel for
//  for(size_t i=0; i<cloud_src->size(); i++){
//    PointTN p = (*cloud_src)[i];
//    Vector3f X_{p.x, p.y, p.z};
//    Vector3f X, p_;
//    p_ = rots_src[im_src_indice]*X_ + t_laser_cam; // Somente o ponto rotacionado para o frame da camera e tomarmos assim a profundidade em Z
//    X = K_*p_;
//    if(X(2) > 0){
//      X = X/X(2);
//      // Se caiu dentro da imagem
//      if(floor(X(0)) > 0 && floor(X(0)) < imcols && floor(X(1)) > 0 && floor(X(1)) < imrows)
//        ds(X(1), X(0)) = p_(2);
//    }
//  }
//  // Imagem profundidade nuvem target
//#pragma omp parallel for
//  for(size_t i=0; i<cloud_tgt->size(); i++){
//    PointTN p = (*cloud_tgt)[i];
//    Vector3f X_{p.x, p.y, p.z};
//    Vector3f X, p_;
//    p_ = rots_tgt[im_tgt_indice]*X_ + t_laser_cam; // Somente o ponto rotacionado para o frame da camera e tomarmos assim a profundidade em Z
//    X = K_*p_;
//    if(X(2) > 0){
//      X = X/X(2);
//      // Se caiu dentro da imagem
//      if(floor(X(0)) > 0 && floor(X(0)) < imcols && floor(X(1)) > 0 && floor(X(1)) < imrows)
//        dt(X(1), X(0)) = p_(2);
//    }
//  }

//  this->filtrar_ruidos_inpaint(dt, ds);

//  /// Encontrar Keypoints nas imagens de profundidade - transf. a partir da RGB
//  /// Calular a translacao de sorce->target que ajusta melhor aqueles pontos
//  ///
//  vector<Vector3f> tts_vec;
//  for(int i=0; i<best_kpsrc.size(); i++){
//    if(dt(best_kptgt[i].pt.y, best_kptgt[i].pt.x) > 0 && ds(best_kpsrc[i].pt.y, best_kpsrc[i].pt.x) > 0){
//      float x, y, z;
//      Vector3f Ps, Pt, tts;

//      z = ds(best_kpsrc[i].pt.y, best_kpsrc[i].pt.x);
//      x = ((best_kpsrc[i].pt.x - K.at<double>(0, 2))*z)/K.at<double>(0, 0);
//      y = ((best_kpsrc[i].pt.y - K.at<double>(1, 2))*z)/K.at<double>(1, 1);
//      Ps << x, y, z;
//      z = dt(best_kptgt[i].pt.y, best_kptgt[i].pt.x);
//      x = ((best_kptgt[i].pt.x - K.at<double>(0, 2))*z)/K.at<double>(0, 0);
//      y = ((best_kptgt[i].pt.y - K.at<double>(1, 2))*z)/K.at<double>(1, 1);
//      Pt << x, y, z;

//      tts = Pt - Rrel*Ps;
//      // Angulo em relacao a estimativa tida pela imagens
//      float cos_theta = (tts.dot(trel))/(tts.norm()*trel.norm());
//      float theta = RAD2DEG(acos(cos_theta));
//      cout << theta << endl;
//      // Se esta com um angulo pequeno, temos bom chute
//      if(abs(theta) < 15.0 || abs(theta - 180.0) < 15.0)
//        tts_vec.emplace_back(tts);
//    }
//  }

}
/////////////////////////////////////////////////////////////////////////////////////////////////
void SFM::filtrar_ruidos_inpaint(MatrixXf &dt, MatrixXf &ds){
  // Dimensao da janela pra olhar pela minima distancia - lado = 2l+1
  const int l = 2, lado = 2*l + 1;

  // Varrer em busca de falhas na imagem de distancia
  MatrixXf tempt = dt, temps = ds;
#pragma omp parallel for
  for(int u=l; u<imrows - l; u++){
    for(int v=l; v<imcols - l; v++){
      if(dt(u, v) == 0){
        MatrixXf mask(lado, lado);
        mask = dt.block<lado,lado>(u-l, v-l);
        tempt(u, v) = (mask.minCoeff() != 0) ? mask.minCoeff() : mask.maxCoeff();
      }
      if(ds(u, v) == 0){
        MatrixXf mask(lado, lado);
        mask = ds.block<lado,lado>(u-l, v-l);
        temps(u, v) = (mask.minCoeff() != 0) ? mask.minCoeff() : mask.maxCoeff();
      }
    }
  }
  dt = tempt; ds = temps;

  // Maiores distancias para corresponder a 255 e criar escala
  float tma = dt.maxCoeff(), sma = ds.maxCoeff();

  // Criar imagens em 1 canal 8 bits e mascaras de uma vez
  Mat t8c(imrows, imcols, CV_8UC1)  , s8c(imrows, imcols, CV_8UC1)  ;
#pragma omp parallel for
  for(int u=0; u<imrows; u++){
    for(int v=0; v<imcols; v++){
      t8c.at<u_int8_t>(u, v) = int(255.0/tma*(dt(u, v)));
      s8c.at<u_int8_t>(u, v) = int(255.0/sma*(ds(u, v)));
    }
  }

  if(debug){
    Mat debugt(imrows, imcols, CV_8UC3), debugs(imrows, imcols, CV_8UC3);
    cvtColor(t8c, debugt, CV_GRAY2BGR);
    cvtColor(s8c, debugs, CV_GRAY2BGR);
    for(int i=0; i<best_kpsrc.size(); i++){
      int r = rand()*255, b = rand()*255, g = rand()*255;
      circle(debugt, Point(best_kptgt[i].pt.x, best_kptgt[i].pt.y), 3, Scalar(r, g, b), FILLED, LINE_8);
      circle(debugs, Point(best_kpsrc[i].pt.x, best_kpsrc[i].pt.y), 3, Scalar(r, g, b), FILLED, LINE_8);
    }
    imshow("target", debugt);
    imshow("source", debugs);
    waitKey(0);
  }
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void SFM::filtrar_matches_keypoints_repetidos(vector<KeyPoint> kt, vector<KeyPoint> ks, vector<DMatch> &m){
  // Matriz de bins para keypoints de target e source
  int w = imcols/5, h = imrows/5;
  vector<DMatch> matriz_matches[w][h];

  // Itera sobre os matches pra colocar eles nos bins certos
  for(int i=0; i<m.size(); i++){
    KeyPoint ktt = kt[m[i].trainIdx];
    int u = ktt.pt.x/5, v = ktt.pt.y/5;
    matriz_matches[u][v].push_back(m[i]);
  }
  // Vetor auxiliar de matches que vao passar no teste de melhor distancia
  vector<DMatch> boas_matches;
  // Procurando na matriz de matches
  for(int i=0; i<w; i++){
    for(int j=0; j<h; j++){
      if(matriz_matches[i][j].size() > 0){
        // Se ha matches e for so uma, adicionar ela mesmo
        if(matriz_matches[i][j].size() == 1){
          boas_matches.push_back(matriz_matches[i][j][0]);
        } else { // Se for mais de uma comparar a distancia com as outras
          DMatch mbest = matriz_matches[i][j][0];
          for(int k=1; k<matriz_matches[i][j].size(); k++){
            if(matriz_matches[i][j][k].distance < mbest.distance)
              mbest = matriz_matches[i][j][k];
          }
          // Adicionar ao vetor a melhor opcao para aquele bin
          boas_matches.push_back(mbest);
        }
      }
      matriz_matches[i][j].clear(); // Ja podemos limpar aquele vetor, ja trabalhamos
    }
  }
  m = boas_matches;
//  // Fazer o mesmo agora para as matches que sobraram e kpts da src
//  // Itera sobre os matches pra colocar eles nos bins certos
//  for(int i=0; i<boas_matches.size(); i++){
//    KeyPoint kst = ks[m[i].queryIdx];
//    int u = kst.pt.x/5, v = kst.pt.y/5;
//    matriz_matches[u][v].push_back(m[i]);
//  }
//  // Vetor auxiliar de matches que vao passar no teste de melhor distancia
//  vector<DMatch> otimas_matches;
//  // Procurando na matriz de matches
//  for(int i=0; i<w; i++){
//    for(int j=0; j<h; j++){
//      if(matriz_matches[i][j].size() > 0){
//        // Se ha matches e for so uma, adicionar ela mesmo
//        if(matriz_matches[i][j].size() == 1){
//          otimas_matches.push_back(matriz_matches[i][j][0]);
//        } else { // Se for mais de uma comparar a distancia com as outras
//          DMatch mbest = matriz_matches[i][j][0];
//          for(int k=1; k<matriz_matches[i][j].size(); k++){
//            if(matriz_matches[i][j][k].distance < mbest.distance)
//              mbest = matriz_matches[i][j][k];
//          }
//          // Adicionar ao vetor a melhor opcao para aquele bin
//          otimas_matches.push_back(mbest);
//        }
//      }
//      matriz_matches[i][j].clear(); // Ja podemos limpar aquele vetor, ja trabalhamos
//    }
//  }

//  // Retornando as matches que restaram
//  m = otimas_matches;
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
