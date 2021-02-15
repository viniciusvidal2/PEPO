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

% This function containts full information and implementations of the benchmark
% functions in Table 1, Table 2, and Table 3 in the paper

% lb is the lower bound: lb=[lb_1,lb_2,...,lb_d]
% up is the uppper bound: ub=[ub_1,ub_2,...,ub_d]
% dim is the number of variables (dimension of the problem)

function [lb,ub,dim,fobj] = Get_Functions_details(F,pose)


switch F
    
    case 'Fob'
        fobj = @Fob;
        lb = [pose(8,1)-5,pose(8,2)-5,pose(3,1)-5,pose(3,2)-5,pose(13,1)-5,pose(13,2)-5];
        ub = [pose(8,1)+5,pose(8,2)+5,pose(3,1)+5,pose(3,2)+5,pose(13,1)+5,pose(13,2)+5];
        dim = 6;%tilt e pan de cada imagem
        
end
end

function o = Fob(x,features_referencia,features_esquerda)
% focos e centros oticos da camera
fx = 951.4;
fy = 966.2;
cx = 658.6;
cy = 386.6;
%Matriz intrisica da camera
K = [fx 0 cx; 0 fy cy; 0 0 1];
%Tamanho final da imagem 360
w = 3599;
h = 1799;
[m,n] = size(features_referencia);
l = 3;
for i =1:3:n
    for j = 1:m
        % Rotacao da imagem tida como referencia
        Ri = roty(x(2))*rotx(x(1));
        % Para cada imagem que desejamos, criar homografia pela equacao do artigo e
        % projetar na 360
        
        % Para imagem de referencia
        Rj = Ri;%% Para imagem de Referencia a homografia sera igual a Identidade
        Hr = K*Ri'*Rj*inv(K);
        uki = Hr*features_referencia(j,i:i+2)';
        % Projetar cada pixel, com offset para o centro da imagem
        uki = ceil(uki/uki(3)) + [w/2; h/2; 0];
        %Verificando os limites dos pontos
        uki(uki <= 0) = 1;
        if (uki(1) >= w)
            uki(1) = w;
        end
        if (uki(2) >= h)
            uki(2)= h;
        end
        features_referencia(j,i:i+2)= uki';
        %Imagem vizinha a referencia
        Rj = roty(x(l+1))*rotx(x(l));
        Hr = K*Ri'*Rj*inv(K);
        pij = Hr*features_esquerda(j,i:i+2)';
        % Projetar cada pixel, com offset para o centro da imagem
        pij = ceil(pij/pij(3)) + [w/2; h/2; 0];
        pij(pij <= 0) = 1;
        %Verificando os limites dos pontos
        if(pij(2) >= h)
            pij(2)= h;
        end
        if(pij(1) >= w)
            pij(1)= w;
        end
        features_esquerda(j,i:i+2) = pij';
    end
    l=l+2;
end
erro = features_referencia - features_esquerda;
o =  norm((erro))^2;

end


