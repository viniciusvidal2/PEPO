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
    K.at<double>(0, 0) = foco(0)        ; K.at<double>(1, 1) = foco(1);
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
void SFM::calcular_features_orb(){
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

        // Passar pra escala de cinza cada imagem
        Mat imtgtGray, imsrcGray;
        cvtColor(imtgt, imtgtGray, CV_BGR2GRAY);
        cvtColor(imsrc, imsrcGray, CV_BGR2GRAY);

        // Descritores ORB calculados
        Ptr<ORB> orb = ORB::create(300);
        orb->detectAndCompute(imtgtGray, Mat(), kptgt, dtgt);
        orb->detectAndCompute(imsrcGray, Mat(), kpsrc, dsrc);

        // Salvando no vetor de keypoints
        kpts_tgt[i] = kptgt;
        kpts_src[i] = kpsrc;

        // Salvando no vetor de cada um os descritores
        descp_tgt[i] = dtgt;
        descp_src[i] = dsrc;
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

        // Descritores SURF calculados
        float min_hessian = 2500;
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
void SFM::orb_matches_matrix_encontrar_melhor(){
    // Ajustar matriz de quantidade de matches
    MatrixXi matches_count(descp_tgt.size(), descp_src.size());

    // Matcher de forca bruta
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    // Para cada combinacao de imagens, fazer match e salvar quantidade final para ver qual
    // a melhor depois
#pragma omp parallel for
    for(int i=0; i<descp_tgt.size(); i++){
        for(int j=0; j<descp_src.size(); j++){
            vector<DMatch> matches, matches_fwd, matches_bck;
            matcher->match(descp_tgt[i], descp_src[j], matches_fwd, Mat());
            matcher->match(descp_src[j], descp_tgt[i], matches_bck, Mat());
            for(size_t k=0; k<matches_fwd.size(); k++ ){
                DMatch forward = matches_fwd[k];
                DMatch backward = matches_bck[forward.trainIdx];
                if(backward.trainIdx == forward.queryIdx)
                    matches.push_back(forward);
            }
            std::sort(matches.begin(), matches.end());
            const int numGoodMatches = matches.size() * 0.5f;
            matches.erase(matches.begin()+numGoodMatches, matches.end());
            matches_count(i, j) = matches.size();
        }
    }

    if(debug)
        cout << "Matriz de matches:\n" << matches_count << endl << "\nMaximo de matches: " << matches_count.maxCoeff() << endl;

    // Atraves do melhor separar descritores e matches daquelas vistas
    Mat best_descp_tgt, best_descp_src;
    int max_matches = matches_count.maxCoeff();
    for(int i=0; i<descp_tgt.size(); i++){
        for(int j=0; j<descp_src.size(); j++){
            if(matches_count(i, j) == max_matches){
                best_descp_tgt = descp_tgt[i]; best_descp_src = descp_src[j];
                best_kptgt     = kpts_tgt[i] ; best_kpsrc     = kpts_src[j] ;
                break;
            }
        }
    }
    descp_tgt.clear(); descp_src.clear(); // Libera memoria
    kpts_tgt.clear(); kpts_src.clear();
    vector<DMatch> matches;
    matcher->match(best_descp_tgt, best_descp_src, matches, Mat());
    std::sort(matches.begin(), matches.end());
    const int numGoodMatches = matches.size() * 0.5f;
    matches.erase(matches.begin()+numGoodMatches, matches.end());
    best_matches = matches;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void SFM::surf_matches_matrix_encontrar_melhor(){
    // Ajustar matriz de quantidade de matches
    MatrixXi matches_count(descp_tgt.size(), descp_src.size());

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
                    if (matches.at(k).at(0).distance < 0.65*matches.at(k).at(1).distance) // Se e bastante unica frente a segunda colocada
                        good_matches.push_back(matches.at(i).at(0));
                }
            }
            // Filtrando por distancia media entre os matches
            vector<float> distances (good_matches.size());
            for (int i=0; i < good_matches.size(); i++)
                distances[i] = good_matches[i].distance;
            float average = accumulate(distances.begin(), distances.end(), 0.0) / distances.size();
            for(vector<DMatch>::iterator it = good_matches.begin(); it!=good_matches.end();){
                if(it->distance > average)
                    good_matches.erase(it);
                else
                    ++it;
            }
            // Anota quantas venceram nessa combinacao
            matches_count(i, j) = good_matches.size();
        }
    }

    if(debug)
        cout << "Matriz de matches:\n" << matches_count << endl << "\nMaximo de matches: " << matches_count.maxCoeff() << endl;

    // Atraves do melhor separar descritores e matches daquelas vistas
    Mat best_descp_tgt, best_descp_src;
    int max_matches = matches_count.maxCoeff();
    for(int i=0; i<descp_tgt.size(); i++){
        for(int j=0; j<descp_src.size(); j++){
            if(matches_count(i, j) == max_matches){
                best_descp_tgt = descp_tgt[i]; best_descp_src = descp_src[j];
                im_tgt_indice = i; im_src_indice = j;
                break;
            }
        }
    }

    // Libera memoria
    descp_tgt.clear(); descp_src.clear();

    // Recalcular os matches aqui, e separar somente ao final os bons keypoints
    vector<vector<DMatch>> matches;
    vector<DMatch> good_matches;
    matcher->knnMatch(best_descp_tgt, best_descp_src, matches, 2);
    for (size_t k = 0; k < matches.size(); k++){
        if (matches.at(k).size() >= 2){
            if (matches.at(k).at(0).distance < 0.8*matches.at(k).at(1).distance) // Se e bastante unica frente a segunda colocada
                good_matches.push_back(matches.at(k).at(0));
        }
    }
    // Filtrando por distancia media entre os matches
    vector<float> distances (good_matches.size());
    for (int i=0; i < good_matches.size(); i++)
        distances[i] = good_matches[i].distance;
    float average = accumulate(distances.begin(), distances.end(), 0.0) / distances.size();
    for(vector<DMatch>::iterator it = good_matches.begin(); it!=good_matches.end();){
        if(it->distance > average)
            good_matches.erase(it);
        else
            ++it;
    }
    best_matches = good_matches;

    // Pegar somente bons kpts
    vector<KeyPoint> curr_kpts_tgt = kpts_tgt[im_tgt_indice], curr_kpts_src = kpts_src[im_src_indice];
    for(auto m:good_matches){
        best_kptgt.emplace_back(curr_kpts_tgt[m.trainIdx]);
        best_kpsrc.emplace_back(curr_kpts_src[m.queryIdx]);
    }

    // Plotar imagens
    if(debug){
        Mat im_matches;
        Mat im1 = imread(imagens_tgt[im_tgt_indice], IMREAD_COLOR);
        Mat im2 = imread(imagens_src[im_src_indice], IMREAD_COLOR);
        drawMatches(im1, kpts_tgt[im_tgt_indice], im2, kpts_src[im_src_indice], good_matches, im_matches, Scalar::all(-1),
                    Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        imshow("Good Matches", im_matches);
        waitKey();
    }

    // Libera memoria
    kpts_tgt.clear(); kpts_src.clear();
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
void SFM::obter_transformacao_final(Matrix4f &T){
    // Acertar eixos de rotacao para a transformacao relativa ficar no mesmo
    // frame que a nuvem esta disposta

    // Calcular rotacao relativa entre o frame src e tgt, src -> tgt
    // Conta necessaria: 2_R^1 = inv(in_R^2)*in_R^1

    // Aplicar a rotacao relativa pela esquerda

    // Finaliza transformacao

}
/////////////////////////////////////////////////////////////////////////////////////////////////
void SFM::get_matched_keypoints(vector<Point2f> &kptgt, vector<Point2f> &kpsrc){
    kptgt.resize(best_kptgt.size()); kpsrc.resize(best_kpsrc.size());
#pragma omp parallel for
    for(int i=0; i<best_kptgt.size(); i++){
        kptgt[i] = best_kptgt[i].pt;
        kpsrc[i] = best_kpsrc[i].pt;
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void SFM::get_indices_imagens(int &t, int &s){
    t = im_tgt_indice;
    s = im_src_indice;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void SFM::ler_nuvens_correspondentes(){
    string ntgt = "acumulada.ply";//nomes_nuvens[im_tgt_indice];
    string nsrc = "acumulada.ply";//nomes_nuvens[im_src_indice];
    loadPLYFile<PointTN>(pasta_tgt+ntgt, *cloud_tgt);
    loadPLYFile<PointTN>(pasta_src+nsrc, *cloud_src);
    // Rotacionar a nuvem de acordo com a entrada e filtrar o que esta nas costas
    Quaternion<float> q(rots_tgt[im_tgt_indice]);
    transformPointCloudWithNormals<PointTN>(*cloud_tgt, *cloud_tgt, Vector3f::Zero(), q);
    PassThrough<PointTN> pass;
    pass.setFilterFieldName("z");
    pass.setFilterLimits(1, 100);
    pass.setInputCloud(cloud_tgt);
    pass.filter(*cloud_tgt);
    transformPointCloudWithNormals<PointTN>(*cloud_tgt, *cloud_tgt, Vector3f::Zero(), q.inverse());

    q = rots_src[im_src_indice];
    transformPointCloudWithNormals<PointTN>(*cloud_src, *cloud_src, Vector3f::Zero(), q);
    pass.setInputCloud(cloud_src);
    pass.filter(*cloud_src);
    transformPointCloudWithNormals<PointTN>(*cloud_src, *cloud_src, Vector3f::Zero(), q.inverse());
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void SFM::obter_correspondencias_3D_e_T(){
    // Ler nuvens referentes as imagens correspondentes
    this->ler_nuvens_correspondentes();
    // Thresh para aceitar o ponto (pixels)
    float thresh = 2;
    /// Para nuvem tgt
    ///
    // Nuvem suporte com mesma quantidade de pontos das correspondencias
    PointCloud<PointTN>::Ptr ct (new PointCloud<PointTN>);
    ct->resize(best_kptgt.size());
    // Rotacao dessa camera - matriz da camera P - nao vamos considerar translacao, irrelevante
    Matrix3f K_, P;
    cv2eigen(K, K_);
    P = K_*rots_tgt[im_tgt_indice];
    // Projetar cada ponto em paralelo - se chegar perto o suficiente do pixel, e isso ai mesmo
//#pragma omp parallel for
    for(size_t i = 0; i < ct->size(); i++){
        Vector3f X_{cloud_tgt->points[i].x, cloud_tgt->points[i].y, cloud_tgt->points[i].z};
        Vector3f X;
        X = P*X_;
        if(X(2) > 0){
            X = X/X(2);
            // Se caiu dentro da imagem
            if(floor(X(0)) > 0 && floor(X(0)) < imcols && floor(X(1)) > 0 && floor(X(1)) < imrows){
                // Procura no vetor de correspondencias se perto de alguma, se sim anota esse ponto naquela posicao do pixel no vetor
//#pragma omp parallel for
                for(int j=0; j<best_kptgt.size(); j++){
                    if(sqrt(pow(best_kptgt[j].pt.x - X(0), 2) + pow(best_kptgt[j].pt.y - X(1), 2)) < thresh){
                        ct->points[j] = cloud_tgt->points[i];
                        break;
                    }
                }
            }
        }
    }
    /// Para nuvem src
    ///
    // Nuvem suporte com mesma quantidade de pontos das correspondencias
    PointCloud<PointTN>::Ptr cs (new PointCloud<PointTN>);
    cs->resize(best_kpsrc.size());
    // Rotacao dessa camera - matriz da camera P - nao vamos considerar translacao, irrelevante
    P = K_*rots_src[im_src_indice];
    // Projetar cada ponto em paralelo - se chegar perto o suficiente do pixel, e isso ai mesmo
//#pragma omp parallel for
    for(size_t i = 0; i < ct->size(); i++){
        Vector3f X_{cloud_src->points[i].x, cloud_src->points[i].y, cloud_src->points[i].z};
        Vector3f X;
        X = P*X_;
        if(X(2) > 0){
            X = X/X(2);
            // Se caiu dentro da imagem
            if(floor(X(0)) > 0 && floor(X(0)) < imcols && floor(X(1)) > 0 && floor(X(1)) < imrows){
                // Procura no vetor de correspondencias se perto de alguma, se sim anota esse ponto naquela posicao do pixel no vetor
//#pragma omp parallel for
                for(int j=0; j<best_kpsrc.size(); j++){
                    if(sqrt(pow(best_kpsrc[j].pt.x - X(0), 2) + pow(best_kpsrc[j].pt.y - X(1), 2)) < thresh){
                        cs->points[j] = cloud_src->points[i];
                        break;
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
    ExtractIndices<PointTN> extract;
    extract.setIndices(indices);
    extract.setInputCloud(ct);
    extract.filter(*ct);
    extract.setInputCloud(cs);
    extract.filter(*cs);
    // Estimar transformacao por SVD
    TransformationEstimationSVD<PointTN, PointTN> svd;
    svd.estimateRigidTransformation(*cs, *ct, Tsvd);

}
