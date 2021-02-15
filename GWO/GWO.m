%___________________________________________________________________%
%  Grey Wolf Optimizer (GWO) source codes version 1.0               %
%                                                                   %
%  Developed in MATLAB R2011b(7.13)                                 %
%                                                                   %
%  Author and programmer: Seyedali Mirjalili                        %
%                                                                   %
%         e-Mail: ali.mirjalili@gmail.com                           %
%                 seyedali.mirjalili@griffithuni.edu.au             %
%                                                                   %
%       Homepage: http://www.alimirjalili.com                       %
%                                                                   %
%   Main paper: S. Mirjalili, S. M. Mirjalili, A. Lewis             %
%               Grey Wolf Optimizer, Advances in Engineering        %
%               Software , in press,                                %
%               DOI: 10.1016/j.advengsoft.2013.12.007               %
%                                                                   %
%___________________________________________________________________%

% Grey Wolf Optimizer
function [Alpha_score,Alpha_pos,Convergence_curve] = GWO(SearchAgents_no,Max_iter,lb,ub,dim,fobj)
%% Ler as imagens
% Pegando os nomes das imagens 
pose = position();%Posição tilt e pan de cada imagem (tilt - vertical, pan - horizontal)
nomes_imagens = cell(1, length(pose));
for n=1:length(pose)
    if n < 10
        nome = ['C:/dataset3/imagem_00',num2str(n),'.png'];
    else
        nome = ['C:/dataset3/imagem_0',num2str(n),'.png'];
    end
    nomes_imagens{n} = nome;
end


% Ler imagem em double - Pegando 3 imagens
im2 = double(imread(nomes_imagens{3}))/255;
im3 = double(imread(nomes_imagens{8}))/255;
im4 = double(imread(nomes_imagens{13}))/255;


step_deg = 0.1;%step em graus para determinar tamanho da imagem 360 final
%Criando a imagem 360 final
im360 = zeros(180/step_deg, 360/step_deg, 3);

%Detectar features points com SURF 
points2 = detectSURFFeatures(rgb2gray(im2));
points3 = detectSURFFeatures(rgb2gray(im3));
points4 = detectSURFFeatures(rgb2gray(im4));

[features2,valid_points2] = extractFeatures(rgb2gray(im2),points2);
[features3,valid_points3] = extractFeatures(rgb2gray(im3),points3);
[features4,valid_points4] = extractFeatures(rgb2gray(im4),points4);


% Match - Encontrando Matches entre imagens
%Considerando Imagem 8 como referencia e pegando as vizinhas 2 e 13 e
%encontrando os matches entre elas
[indexPairs1, matchmetric1] =   matchFeatures(features3, features2);
[indexPairs2, matchmetric2] =   matchFeatures(features3, features4);

matchedPoints3_2 = valid_points3(indexPairs1(:,1),:);
matchedPoints2 = valid_points2(indexPairs1(:,2),:);

matchedPoints3_4 = valid_points3(indexPairs2(:,1),:);
matchedPoints4 = valid_points4(indexPairs2(:,2),:);

% Filtrar matches-  Pegando x melhores matches para tornar o coidgo mais
% rapido
[~, bests1] = sort(matchmetric1);
[~, bests2] = sort(matchmetric2);
melhores = 30;
matchedPoints3_2 = matchedPoints3_2(bests1(1:melhores));
matchedPoints2 = matchedPoints2(bests1(1:melhores));

matchedPoints3_4 = matchedPoints3_4(bests2(1:melhores));
matchedPoints4 = matchedPoints4(bests2(1:melhores));


%% Pontos de feature para ver erro
features_referencia3_4 = [matchedPoints3_4.Location'; ones(1, size(matchedPoints3_4.Location, 1))];
features_esquerda   = [matchedPoints4.Location'; ones(1, size(matchedPoints4.Location, 1))];

features_referencia3_2 = [matchedPoints3_2.Location'; ones(1, size(matchedPoints3_2.Location, 1))];
features_direita   = [matchedPoints2.Location'; ones(1, size(matchedPoints2.Location, 1))];

features_referencia = [features_referencia3_2',features_referencia3_4'];
features_vizinhos = [features_direita',features_esquerda'];
% initialize alpha, beta, and delta_pos
Alpha_pos=zeros(1,dim);
Alpha_score = inf; %change this to -inf for maximization problems

Beta_pos = zeros(1,dim);
Beta_score = inf; %change this to -inf for maximization problems

Delta_pos = zeros(1,dim);
Delta_score = inf; %change this to -inf for maximization problems

%Initialize the positions of search agents
Positions = initialization(SearchAgents_no,dim,ub,lb,pose);

Convergence_curve = zeros(1,Max_iter);

l=0;% Loop counter

% Main loop
while l<Max_iter
   
    for i=1:size(Positions,1)
        
        %        Return back the search agents that go beyond the boundaries of the search space
        %         Flag4ub = Positions(i,:)>ub
        %         Flag4lb = Positions(i,:)<lb
        %         Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        %
        for j=1:dim
            if Positions(i,j)>ub(j)
                ub_i = ub(j);
                lb_i=lb(j);
               
                Positions(i,j)=rand(1).*(ub_i-lb_i)+lb_i;
            end
            if Positions(i,j)<lb(j)
                ub_i=ub(j);
                lb_i=lb(j);
                Positions(i,j)=rand(1).*(ub_i-lb_i)+lb_i;
            end
        end
        
        

        % Calculate objective function for each search agent
        fitness = fobj(Positions(i,:),features_referencia,features_vizinhos);
        
        % Update Alpha, Beta, and Delta
        if fitness < Alpha_score
            Alpha_score=fitness; % Update alpha
            Alpha_pos=Positions(i,:);
        end
        
        if fitness>Alpha_score && fitness<Beta_score
            Beta_score=fitness; % Update beta
            Beta_pos=Positions(i,:);
        end
        
        if fitness>Alpha_score && fitness>Beta_score && fitness<Delta_score
            Delta_score=fitness; % Update delta
            Delta_pos=Positions(i,:);
        end
    end
    
    
    a= 2-l*((2)/Max_iter); % a decreases linearly fron 2 to 0
    
    % Update the Position of search agents including omegas
    for i=1:size(Positions,1)
        for j=1:size(Positions,2)
            
            r1=rand(); % r1 is a random number in [0,1]
            r2=rand(); % r2 is a random number in [0,1]
            
            A1=2*a*r1-a; % Equation (3.3)
            C1=2*r2; % Equation (3.4)
            
            D_alpha=abs(C1*Alpha_pos(j)-Positions(i,j)); % Equation (3.5)-part 1
            X1=Alpha_pos(j)-A1*D_alpha; % Equation (3.6)-part 1
            
            r1=rand();
            r2=rand();
            
            A2=2*a*r1-a; % Equation (3.3)
            C2=2*r2; % Equation (3.4)
            
            D_beta=abs(C2*Beta_pos(j)-Positions(i,j)); % Equation (3.5)-part 2
            X2=Beta_pos(j)-A2*D_beta; % Equation (3.6)-part 2
            
            r1=rand();
            r2=rand();
            
            A3=2*a*r1-a; % Equation (3.3)
            C3=2*r2; % Equation (3.4)
            
            D_delta=abs(C3*Delta_pos(j)-Positions(i,j)); % Equation (3.5)-part 3
            X3=Delta_pos(j)-A3*D_delta; % Equation (3.5)-part 3
            
            Positions(i,j)=(X1+X2+X3)/3;% Equation (3.7)
            
        end
    end
    l=l+1;
    Convergence_curve(l) = Alpha_score;
    
end



