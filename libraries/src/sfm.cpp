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
  K.at<double>(0, 0) = foco(0); K.at<double>(1, 1) = foco(1);
  K.at<double>(0, 2) = centro_otico(0); K.at<double>(1, 2) = centro_otico(1);
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

    // Salvar aqui as dimensoes da imagem para a sequencia do algoritmo
    imcols = imtgt.cols; imrows = imtgt.rows;

    // Descritores SIFT calculados
    Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create();
    sift->detectAndCompute(imtgt, Mat(), kptgt, dtgt);
    sift->detectAndCompute(imsrc, Mat(), kpsrc, dsrc);
    // Calculando somatorio para cada linha de descritores
    Mat dtgtsum, dsrcsum;
    reduce(dtgt, dtgtsum, 1, CV_16UC1);
    reduce(dsrc, dsrcsum, 1, CV_16UC1);
    // Normalizando e passando raiz em cada elementos de linha nos descritores da src
#pragma omp parallel for
    for(int i=0; i<dsrc.rows; i++){
      for(int j=0; j<dsrc.cols; j++){
        dsrc.at<float>(i, j) = sqrt(dsrc.at<float>(i, j) / (dsrcsum.at<float>(i, 0) + numeric_limits<float>::epsilon()));
      }
    }
    // Normalizando e passando raiz em cada elementos de linha nos descritores da tgt
#pragma omp parallel for
    for(int i=0; i<dtgt.rows; i++){
      for(int j=0; j<dtgt.cols; j++){
        dtgt.at<float>(i, j) = sqrt(dtgt.at<float>(i, j) / (dtgtsum.at<float>(i, 0) + numeric_limits<float>::epsilon()));
      }
    }

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
  //#pragma omp parallel for
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
          this->filterMatchesLineCoeff(good_matches, kpts_tgt[i], kpts_src[j], imcols, DEG2RAD(25));

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
  }

  // Plotar imagens
  if(debug){
    Mat im1 = imread(imagens_tgt[im_tgt_indice], IMREAD_COLOR);
    Mat im2 = imread(imagens_src[im_src_indice], IMREAD_COLOR);
    for(int i=0; i<best_kpsrc.size(); i++){
      int r = rand()%255, b = rand()%255, g = rand()%255;
      circle(im1, Point(best_kptgt[i].pt.x, best_kptgt[i].pt.y), 8, Scalar(r, g, b), FILLED, LINE_8);
      circle(im2, Point(best_kpsrc[i].pt.x, best_kpsrc[i].pt.y), 8, Scalar(r, g, b), FILLED, LINE_8);
    }
    imshow("targetc", im1);
    imshow("sourcec", im2);
    imwrite(pasta_src+"im_tgt.png", im1);
    imwrite(pasta_src+"im_src.png", im2);
    waitKey();
  }

  // Libera memoria
  kpts_tgt.clear(); kpts_src.clear();
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
void SFM::obter_transformacao_final_sfm(){
  // Lendo as nuvens correspondentes
  this->ler_nuvens_correspondentes();

  if(debug)
    cout << "\nAplicando transformacao final ..." << endl;
  // Calcular rotacao relativa entre o frame src e tgt, src -> tgt
  // Conta necessaria: 2_R^1 = inv(in_R^2)*in_R^1
  R_src_tgt = rots_src[im_src_indice]*rots_tgt[im_tgt_indice].inverse();

  // Transformacao final (so rotacao)
  Tsvd.block<3,3>(0, 0) = Rrel * R_src_tgt;
  // Transformacao final (em translacao)
  this->estimar_escala_translacao();
//  Tsvd.block<3,1>(0, 3) = trel;

  // Transformar a nuvem source com a transformacao estimada
//  transformPointCloudWithNormals<PointTN>(*cloud_src, *cloud_src, Tsvd);

  if(debug){
    // Salvar ambas as nuvens na pasta source pra comparar
    savePLYFileBinary<PointTN>(pasta_src+"debug_tgt.ply", *cloud_tgt);
    savePLYFileBinary<PointTN>(pasta_src+"debug_src.ply", *cloud_src);
  }
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void SFM::estimar_escala_translacao(){
  // Parametros intrinsecos da camera explicitados para facilitar
  float fx = K.at<double>(0, 0), fy = K.at<double>(1, 1), cx = K.at<double>(0, 2), cy = K.at<double>(1, 2);
  // Lista de pontos 3D correspondentes a partir de cada match
  vector<Vector3f> corresp_3d_src(best_kpsrc.size()), corresp_3d_tgt(best_kptgt.size());
  PointCloud<PointT>::Ptr corresp_src (new PointCloud<PointT>);
  PointCloud<PointT>::Ptr corresp_tgt (new PointCloud<PointT>);

  cout << "\nComecando a vasculhar os matches procurando por octree seus pontos na nuvem ..." << endl;
  octree::OctreePointCloudSearch<PointTN> octs(0.02);
  octs.setInputCloud(cloud_src);
  octs.addPointsFromInputCloud();
  // Para cada match na src
  for(int i=0; i<best_kpsrc.size(); i++){
    // Pegar o keypoint na imagem em questao
    float us, vs;
    us = best_kpsrc[i].pt.x; vs = best_kpsrc[i].pt.y;
    // Calcular a direcao no frame da camera para eles
    Vector3f dirs;
    dirs << (us - cx)/(fx), (vs - cy)/(fy), 1;
    // Rotacionar o vetor para o frame local
    dirs = rots_src[im_src_indice].inverse() * dirs;
    // Aplicar ray casting para saber em que parte da nuvem vai bater
    octree::OctreePointCloudSearch<PointTN>::AlignedPointTVector aligns;
    octs.getIntersectedVoxelCenters(Vector3f::Zero(), dirs.normalized(), aligns, 1);
    // Se achou, adicionar na lista de pontos 3D naquele local
    if(aligns.size() > 0)
      corresp_3d_src[i] << aligns[0].x, aligns[0].y, aligns[0].z;
    else
      corresp_3d_src[i] << 0, 0, 0;
  }

  octree::OctreePointCloudSearch<PointTN> octt(0.02);
  octt.setInputCloud(cloud_tgt);
  octt.addPointsFromInputCloud();
  // Para cada match na tgt
  for(int i=0; i<best_kptgt.size(); i++){
    // Pegar o keypoint na imagem em questao
    float ut, vt;
    ut = best_kptgt[i].pt.x; vt = best_kptgt[i].pt.y;
    // Calcular a direcao no frame da camera para eles
    Vector3f dirt;
    dirt << (ut - cx)/(fx), (vt - cy)/(fy), 1;
    // Rotacionar o vetor para o frame local
    dirt = rots_tgt[im_tgt_indice].inverse() * dirt;
    // Aplicar ray casting para saber em que parte da nuvem vai bater
    octree::OctreePointCloudSearch<PointTN>::AlignedPointTVector alignt;
    octt.getIntersectedVoxelCenters(Vector3f::Zero(), dirt.normalized(), alignt, 1);
    // Se achou, adicionar na lista de pontos 3D naquele local
    if(alignt.size() > 0)
      corresp_3d_tgt[i] << alignt[0].x, alignt[0].y, alignt[0].z;
    else
      corresp_3d_tgt[i] << 0, 0, 0;
  }

  // NUVEM DE PONTOS DE TESTE DOS KEYPOINTS EM 3D NA TGT
  PointCloud<PointT>::Ptr teste (new PointCloud<PointT>);
  for(auto p:corresp_3d_tgt){
    PointT pp;
    pp.x = p(0); pp.y = p(1); pp.z = p(2);
    pp.g = 255; pp.r = 0; pp.b = 0;
    teste->push_back(pp);
  }
  savePLYFileBinary(pasta_src+"kpts_tgt.ply", *teste);
  teste->clear();

  // Para cada ponto resultante do ray cast
  vector<float> angulos;
  vector<Vector3f> translacoes;
  for(int i=0; i < corresp_3d_src.size(); i++){
    if(!(corresp_3d_src[i].norm() == 0) && !(corresp_3d_tgt[i].norm() == 0)){
      // Adicionar nas nuvens de correspondencias os pontos
      PointT pp;
      pp.x = corresp_3d_tgt[i](0); pp.y = corresp_3d_tgt[i](1); pp.z = corresp_3d_tgt[i](2);
      corresp_tgt->push_back(pp);
      pp.x = corresp_3d_src[i](0); pp.y = corresp_3d_src[i](1); pp.z = corresp_3d_src[i](2);
      corresp_src->push_back(pp);
      // Rotacionar ponto para o frame de src->tgt e obter translacao necessaria
      Vector3f t;
      t = corresp_3d_tgt[i] - Rrel*R_src_tgt*corresp_3d_src[i];
      // Se direcao casa com a inicial proposta pelo match das imagens, ok
      float theta = RAD2DEG( acos(t.dot(trel)/(t.norm()*trel.norm())) );
      angulos.push_back(theta);
      translacoes.push_back(t);
    }
  }

  // Acha melhores translacoes segundo a media e desvio padrao das orientacoes
  vector<Vector3f> boas_translacoes;
  accumulator_set<double, stats<tag::variance> > a_acc;
  for_each(angulos.begin(), angulos.end(), bind<void>(ref(a_acc), _1));
  for(int i=0; i<translacoes.size(); i++){
    if(abs(angulos[i] - extract::mean(a_acc)) < sqrt(extract::variance(a_acc)))
      boas_translacoes.push_back(translacoes[i]);
  }

  // Tira a media do somatorio dos vetores e atribui ao que antes era a translacao
  // so em escala, agora vai para o mundo real
  Vector3f acc;
  for(auto t:boas_translacoes) acc += t;
  if(boas_translacoes.size() > 0) trel = acc/boas_translacoes.size(); else trel = 10*trel;

  for(auto p:corresp_3d_src){
    PointT pp;
    p = Tsvd.block<3,3>(0, 0)*p + trel;
    pp.x = p(0); pp.y = p(1); pp.z = p(2);
    pp.g = 255; pp.r = 0; pp.b = 255;
    teste->push_back(pp);
  }
  savePLYFileBinary(pasta_src+"kpts_src.ply", *teste);
  teste->clear();

  // A partir das correspondencias, pega por ransac a transformacao 3D de uma vez
  pcl::Correspondences corresp;
  for(size_t i=0; i<corresp_tgt->size(); i++){
    pcl::Correspondence corr;
    corr.index_query = i;
    corr.index_match = i;
    corresp.push_back(corr);
  }

  /// RANSAC BASED Correspondence Rejection
  pcl::CorrespondencesConstPtr correspond = boost::make_shared< pcl::Correspondences >(corresp);

  pcl::Correspondences corr;
  pcl::registration::CorrespondenceRejectorSampleConsensus< PointT > Ransac_based_Rejection;
  Ransac_based_Rejection.setInputSource(corresp_src);
  Ransac_based_Rejection.setInputTarget(corresp_tgt);
  double sac_threshold = 0.5;// default PCL value..can be changed and may slightly affect the number of correspondences
  Ransac_based_Rejection.setInlierThreshold(sac_threshold);
  Ransac_based_Rejection.setInputCorrespondences(correspond);
  Ransac_based_Rejection.getCorrespondences(corr);

  Tsvd = Ransac_based_Rejection.getBestTransformation();
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
  // Fazer o mesmo agora para as matches que sobraram e kpts da src
  // Itera sobre os matches pra colocar eles nos bins certos
  for(int i=0; i<boas_matches.size(); i++){
    KeyPoint kst = ks[m[i].queryIdx];
    int u = kst.pt.x/5, v = kst.pt.y/5;
    matriz_matches[u][v].push_back(m[i]);
  }
  // Vetor auxiliar de matches que vao passar no teste de melhor distancia
  vector<DMatch> otimas_matches;
  // Procurando na matriz de matches
  for(int i=0; i<w; i++){
    for(int j=0; j<h; j++){
      if(matriz_matches[i][j].size() > 0){
        // Se ha matches e for so uma, adicionar ela mesmo
        if(matriz_matches[i][j].size() == 1){
          otimas_matches.push_back(matriz_matches[i][j][0]);
        } else { // Se for mais de uma comparar a distancia com as outras
          DMatch mbest = matriz_matches[i][j][0];
          for(int k=1; k<matriz_matches[i][j].size(); k++){
            if(matriz_matches[i][j][k].distance < mbest.distance)
              mbest = matriz_matches[i][j][k];
          }
          // Adicionar ao vetor a melhor opcao para aquele bin
          otimas_matches.push_back(mbest);
        }
      }
      matriz_matches[i][j].clear(); // Ja podemos limpar aquele vetor, ja trabalhamos
    }
  }

  // Retornando as matches que restaram
  m = otimas_matches;
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
  vector<DMatch> temp;
  for(int i=0; i<coefs.size(); i++){
    // Se os matches estao na mesma regiao da foto
    if( (kpref[matches[i].queryIdx].pt.x < width/2 && kpnow[matches[i].trainIdx].pt.x < width/2) ||
        (kpref[matches[i].queryIdx].pt.x > width/2 && kpnow[matches[i].trainIdx].pt.x > width/2)){
      // Filtrar o vetor de matches na posicao que os coeficientes estejam fora por ngraus
      if(abs(coefs[i]) < n)
        temp.push_back(matches[i]);
    }
  }
  matches = temp;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
Matrix4f SFM::icp(float vs, int its){
  // Atribuindo FOV para as nuvens de acordo com a orientacao (usando fov da camera ~80 graus)
  PointCloud<PointTN>::Ptr csrc (new PointCloud<PointTN>);
  PointCloud<PointTN>::Ptr ctgt (new PointCloud<PointTN>);
  if(debug)
    cout << "\nTirando FOV ..." << endl;
  float thresh = 50.0/2.0;
  float d;
  for(size_t i=0; i<cloud_tgt->size(); i++){
    Vector3f p{(*cloud_tgt)[i].x, (*cloud_tgt)[i].y, (*cloud_tgt)[i].z};
    p = rots_tgt[im_tgt_indice]*p;
    d = p.norm();
    if(abs(acos( p(2)/d )) < DEG2RAD(thresh) && p(2) > 0)
      ctgt->push_back((*cloud_tgt)[i]);
  }
  for(size_t i=0; i<cloud_src->size(); i++){
    Vector3f p{(*cloud_src)[i].x, (*cloud_src)[i].y, (*cloud_src)[i].z};
    p = rots_src[im_src_indice]*p;
    d = p.norm();
    if(abs(acos( p(2)/d )) < DEG2RAD(thresh) && p(2) > 0)
      csrc->push_back((*cloud_src)[i]);
  }
  transformPointCloudWithNormals(*csrc, *csrc, Tsvd);
  transformPointCloudWithNormals(*cloud_src, *cloud_src, Tsvd);

  Matrix4f Ticp = Matrix4f::Identity();
  vs = vs/100.0;
  if(debug)
    cout << "\nPerformando ICP ..." << endl;
  // Reduzindo ainda mais as nuvens pra nao dar trabalho assim ao icp
  PointCloud<PointXYZRGBA>::Ptr tgttemp(new PointCloud<PointXYZRGBA>);
  PointCloud<PointXYZRGBA>::Ptr srctemp(new PointCloud<PointXYZRGBA>);
  copyPointCloud(*cloud_tgt, *tgttemp);
  copyPointCloud(*cloud_src, *srctemp);
  if(vs > 0){
    VoxelGrid<PointXYZRGBA> voxel;
    voxel.setLeafSize(vs, vs, vs);
    voxel.setInputCloud(tgttemp);
    voxel.filter(*tgttemp);
    voxel.setInputCloud(srctemp);
    voxel.filter(*srctemp);
  }
  StatisticalOutlierRemoval<PointXYZRGBA> sor;
  sor.setMeanK(30);
  sor.setStddevMulThresh(2);
  sor.setNegative(false);
  sor.setInputCloud(srctemp);
  sor.filter(*srctemp);
  sor.setInputCloud(tgttemp);
  sor.filter(*tgttemp);
  savePLYFileBinary<PointTN>(pasta_src+"src_antes_icp.ply", *csrc);

  // Criando o otimizador de ICP comum
  GeneralizedIterativeClosestPoint6D icp;
//  IterativeClosestPoint<PointTN, PointTN> icp;
  icp.setUseReciprocalCorrespondences(true);
  icp.setInputTarget(tgttemp);
  icp.setInputSource(srctemp);
  //    icp.setRANSACIterations(30);
  icp.setMaximumIterations(its); // Chute inicial bom 10-100
  icp.setTransformationEpsilon(1*1e-10);
  icp.setEuclideanFitnessEpsilon(1*1e-13);
  icp.setMaxCorrespondenceDistance(0.04);
  // Alinhando
  PointCloud<PointXYZRGBA> dummy;
  icp.align(dummy, Matrix4f::Identity());
  // Obtendo a transformacao otimizada e aplicando
  if(icp.hasConverged()){
    Ticp = icp.getFinalTransformation();
    cout << "\nICP convergiu !!!" << endl;
  }
  transformPointCloudWithNormals<PointTN>(*csrc, *csrc, Ticp);

  savePLYFileBinary<PointTN>(pasta_src+"src_final.ply", *csrc);
  savePLYFileBinary<PointTN>(pasta_src+"tgt_final.ply", *ctgt);

  return Ticp;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void SFM::somar_spaces(Matrix4f T, float radius, int rate){
  // Trazer nuvem source finalmente para a posicao
  transformPointCloudWithNormals<PointTN>(*cloud_src, *cloud_src, T);
  // Iniciar kdtree de busca
  KdTreeFLANN<PointTN> kdtree;
  kdtree.setInputCloud(cloud_tgt);
  vector<int> pointIdxRadiusSearch;
  vector<float> pointRadiusSquaredDistance;
  // Nuvem de pontos de indices bons
  PointIndices::Ptr indices (new PointIndices);
  // Retirando indices NaN se existirem
  vector<int> indicesNaN;
  removeNaNFromPointCloud(*cloud_src, *cloud_src, indicesNaN);
  removeNaNFromPointCloud(*cloud_tgt, *cloud_tgt, indicesNaN);
  // Para cada ponto, se ja houver vizinhos, nao seguir
  for(size_t i=0; i<cloud_src->size(); i++){
    if(kdtree.radiusSearch(cloud_src->points[i], radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) <= rate)
      indices->indices.emplace_back(i);
  }
  // Filtrar na nuvem now so os indices que estao sem vizinhos na obj
  ExtractIndices<PointTN> extract;
  extract.setInputCloud(cloud_src);
  extract.setIndices(indices);
  extract.setNegative(false);
  extract.filter(*cloud_src);

  // Somar as duas nuvens e salvar resultado
  *cloud_tgt += *cloud_src;
  savePLYFileBinary<PointTN>(pasta_src+"registro_final.ply", *cloud_tgt);
}
/////////////////////////////////////////////////////////////////////////////////////////////////
