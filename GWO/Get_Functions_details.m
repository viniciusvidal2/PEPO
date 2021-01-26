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

function [lb,ub,dim,fobj] = Get_Functions_details(F,roll,pitch,yaw)


switch F
    case 'Fob'
        fobj = @Fob;
%          lb = [ roll(2,1), yaw(2,1)-10, pitch(2,1)-10, roll(7,1), yaw(7,1)-10,pitch(7,1)-10];
%         ub = [ roll(2,1), yaw(2,1)+10, pitch(2,1)+10, roll(7,1), yaw(7,1)+10,pitch(7,1)+10];
        %%%%% fx1,  fy1, cx1,  cy1, fx2, fy2,  cx2, cy2, roll1, pitch1,    yaw1,     roll2, pitch2,   yaw2
        lb = [1377, 1399, 930, 510, 1377, 1399, 930, 510, roll(7,1), yaw(7,1)-10, pitch(7,1)-10, roll(10,1), yaw(10,1)-10, pitch(10,1)-10];
        ub = [1477, 1499, 990, 570, 1477, 1499, 990, 570, roll(7,1), yaw(7,1)+10, pitch(7,1)+10, roll(10,1), yaw(10,1)+10, pitch(10,1)+10];
        dim = 14;%fx1, fy1, cx1, cy1, fx2, fy2, cx2,cy2,roll1,pitch1,yaw1,roll2,pitch2,yaw2
    case 'Fob2'
         fobj = @Fob2;
        
        %%%%% fx1,  fy1, cx1,  cy1, fx2, fy2,  cx2, cy2, roll1, pitch1,    yaw1,     roll2, pitch2,   yaw2
        lb = [1377, 1399, 930, 510, 1377, 1399, 930, 510,  yaw(2,1)-10, pitch(2,1)-10,  yaw(7,1)-10, pitch(7,1)-10];
        ub = [1477, 1499, 990, 570, 1477, 1499, 990, 570,  yaw(2,1)+10, pitch(2,1)+10,  yaw(7,1)+10, pitch(7,1)+10];
        dim = 12;%fx1, fy1, cx1, cy1, fx2, fy2, cx2,cy2,roll1,pitch1,yaw1,roll2,pitch2,yaw2
case 'Fob3'
         fobj = @Fob3;
        
        %%%%% fx1,  fy1, cx1,  cy1, fx2, fy2,  cx2, cy2, roll1, pitch1,    yaw1,     roll2, pitch2,   yaw2
        lb = [1377, 1399, 930, 510, 1377, 1399, 930, 510, 1377, 1399, 930, 510,  yaw(2,1)-10, pitch(2,1)-10,  yaw(7,1)-10, pitch(7,1)-10, yaw(10,1)-10, pitch(10,1)-10];
        ub = [1477, 1499, 990, 570, 1477, 1499, 990, 570, 1477, 1499, 990, 570,  yaw(2,1)+10, pitch(2,1)+10,  yaw(7,1)+10, pitch(7,1)+10,yaw(10,1)+10, pitch(10,1)+10];
        dim = 18;%fx1, fy1, cx1, cy1, fx2, fy2, cx2,cy2,roll1,pitch1,yaw1,roll2,pitch2,yaw2

end

end
function o = Fob(x,matches,kpt1,kpt2,image1,image2,im360)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%FOB%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Match em questao

[m,n] = size(kpt1);

for j = 1:m
    
    a = matches(j,2)+1; % no matlab começa no index 1, pegou index do c++ que começa no zero;
    b = matches(j,1)+1;
    
    % Keypoints
    kp1 = kpt1(j,:);
    kp2 = kpt2(j,:);
    %% Pose da CAMERA 1, so existe aqui rotacao, vamos suprimir as translacoes
    % pois serao irrelevantes e serao compensadas por outros dados
    
    step_deg = 0.1;
  
    r1 = eul2rotm([x(9),- deg2rad(raw2deg(x(10), 'pan')),- deg2rad(raw2deg(x(11), 'tilt'))] );
    % // Vamos criar o frustrum da CAMERA 1, assim como nas nossas funcoes, como o desenho do github
    % // Supondo a variavel F e o raio da esfera como F = R = 1, nao interferiu nas experiencias
    [rows1 cols1,numberOfColorChannels] = size(image1);
    dx1 = x(3) - double(cols1) / 2;
    dy1 = x(4) - double(rows1) / 2;

    maxX = ((cols1) - 2*dx1) / (2.0 * x(1));
    minX = ((cols1) + 2*dx1) / (2.0 * x(1));
    maxY = ((rows1) - 2*dy1) / (2.0 * x(2));
    minY = ((rows1) + 2*dy1) / (2.0 * x(2));
    
    F = 1;
    p =[ 0, 0, 0]';
    p1 = r1 * p; % Nao usado a principio, pode omitir
    p =[ minX, minY, F]';
    p2 = r1 * p;
    p =[ maxX, minY, F]';
    p3 = r1 * p; % Nao usado a principio, pode omitir
    p =[maxX, maxY, F]';
    p4 = r1 * p;
    p =[ minX, maxY, F]';
    p5 = r1 * p;
    p =[ 0, 0, F]';
    pCenter = r1 * p; % Nao usado a principio, pode omitir
    
    %  Ponto no frustrum 3D correspondente a feature na imagem 1 em 2D
    ponto3d = p5 + (p4 - p5) * kp1(1) / cols1 + (p2 - p5) * kp1(2) / rows1;
    % Latitude e longitude no 360
    lat = 180 / 3.1415 * (acos(ponto3d(2) / norm(ponto3d)));
    lon = -180 / 3.1415 * (atan2(ponto3d(3), ponto3d(1)));
    if(lon < 0)
        lon = lon + 360.0;
    else
        lon = lon;
    end
    
    [rows360 cols360,numberOfColorChannels] = size(im360);
    u = (lon / step_deg);
    v = rows360 - 1 - (lat / step_deg);
    if(u >= cols360)
        u = cols360 - 1;
    else
        u = u;
    end % Nao deixar passar do limite de colunas por seguranca
    if(u < 0)
        u = 0;
    else
        u = u;
    end
    if(v >= rows360)
        u = rows360 - 1;
    else
        v = v;
    end % Nao deixar passar do limite de linhas por seguranca
    if(v < 0)
        v = 0;
    else
        v = v;
    end
   

 % Ponto na imagem 360 devido a camera 2, finalmente apos as contas, armazenar
    ponto_fc1(j,:) = [ u, v ];
    %% Pose da CAMERA 2, so existe aqui rotacao, vamos suprimir as translacoes
    % pois serao irrelevantes e serao compensadas por outros dados
       r2 = eul2rotm([x(12),- deg2rad(raw2deg(x(13), 'pan')),- deg2rad(raw2deg(x(14), 'tilt'))] );
    % // Vamos criar o frustrum da CAMERA 1, assim como nas nossas funcoes, como o desenho do github
    % // Supondo a variavel F e o raio da esfera como F = R = 1, nao interferiu nas experiencias
    [rows2 cols2,numberOfColorChannels] = size(image2);
    dx2 = x(7) - double(cols2) / 2;
    dy2 = x(8) - double(rows2) / 2;
    
    maxX = ((cols2) - 2*dx2) / (2.0 * x(5));
    minX = ((cols2) + 2*dx2) / (2.0 * x(5));
    maxY = ((rows2) - 2*dy2) / (2.0 * x(6));
    minY = ((rows2) + 2*dy2) / (2.0 * x(6));
   
    F = 1;
    p =[ 0, 0, 0]';
    p1 = r2 * p; % Nao usado a principio, pode omitir
    p =[ minX, minY, F]';
    p2 = r2 * p;
    p =[ maxX, minY, F]';
    p3 = r2 * p; % Nao usado a principio, pode omitir
    p =[maxX, maxY, F]';
    p4 = r2 * p;
    p =[ minX, maxY, F]';
    p5 = r2 * p;
    p =[ 0, 0, F]';
    pCenter = r2 * p; % Nao usado a principio, pode omitir
    
    %%  Ponto no frustrum 3D correspondente a feature na imagem 1 em 2D
    ponto3d = p5 + (p4 - p5) * kp2(1) / cols2 + (p2 - p5) * kp2(2) / rows2;
    %% Latitude e longitude no 360
    lat = 180 / 3.1415 * (acos(ponto3d(2) / norm(ponto3d)));
    lon = -180 / 3.1415 * (atan2(ponto3d(3), ponto3d(1)));
    if(lon < 0)
        lon = lon + 360.0;
    else
        lon = lon;
    end
    
    [rows360 cols360,numberOfColorChannels] = size(im360);
    u = (lon / step_deg);
    v = rows360 - 1 - (lat / step_deg);
    if(u >= cols360)
        u = cols360 - 1;
    else
        u = u;
    end % Nao deixar passar do limite de colunas por seguranca
    if(u < 0)
        u = 0;
    else
        u = u;
    end
    if(v >= rows360)
        u = rows360 - 1;
    else
        v = v;
    end % Nao deixar passar do limite de linhas por seguranca
    if(v < 0)
        v = 0;
    else
        v = v;
    end

    %% Ponto na imagem 360 devido a camera 2, finalmente apos as contas, armazenar
    ponto_fc2(j,:) =[ u, v ];
    
end
p1_norm = normalize(ponto_fc1,'range');
p2_norm = normalize(ponto_fc2,'range');
o =  norm((p1_norm - p2_norm));

end


function o = Fob2(x,matches,kpt1,kpt2,image1,image2,im360)
[m,n] = size(kpt1);

for j = 1:m
% a = matches(j,2)+1; % no matlab começa no index 1, pegou index do c++ que começa no zero;
% b = matches(j,1)+1;
   
    % Keypoints
    kp1 = kpt1(j,:);
    kp2 = kpt2(j,:);
    %% Pose da CAMERA 1, so existe aqui rotacao, vamos suprimir as translacoes
    % pois serao irrelevantes e serao compensadas por outros dados
    
    step_deg = 0.1;
  
%     r1 = eul2rotm([x(9),- deg2rad(raw2deg(x(10), 'pan')),- deg2rad(raw2deg(x(11), 'tilt'))] );
    % // Vamos criar o frustrum da CAMERA 1, assim como nas nossas funcoes, como o desenho do github
    % // Supondo a variavel F e o raio da esfera como F = R = 1, nao interferiu nas experiencias
    [rows1 cols1,numberOfColorChannels] = size(image1);
    dx1 = x(3) - double(cols1) / 2;
    dy1 = x(4) - double(rows1) / 2;

    F = 1;
    maxX = ((cols1) - 2*dx1) / (2.0 * x(1));
    minX = ((cols1) + 2*dx1) / (2.0 * x(1));
    maxY = ((rows1) - 2*dy1) / (2.0 * x(2));
    minY = ((rows1) + 2*dy1) / (2.0 * x(2));
    p2 = [ minX, minY, F]';
    p4 = [maxX, maxY, F]';
    p5 = [ minX, maxY, F]';
    
    %  Ponto no frustrum 3D correspondente a feature na imagem 1 em 2D
    ponto3d = p5 + (p4 - p5) * kp1(1) / cols1 + (p2 - p5) * kp1(2) / rows1;
    % Latitude e longitude no 360
    lat = 180 / 3.1415 * (acos(ponto3d(2) / norm(ponto3d)));
    lon = -180 / 3.1415 * (atan2(ponto3d(3), ponto3d(1)));
    if(lon < 0)
        lon = lon + 360.0;
    else
        lon = lon;
    end
   
    lat = lat -  deg2rad(raw2deg(x(9),'pan'));
    lon = lon + deg2rad(raw2deg(x(10),'tilt'));
    
    [rows360 cols360,numberOfColorChannels] = size(im360);
    u = (lon / step_deg);
    v = rows360 - 1 - (lat / step_deg);
    if(u >= cols360)
        u = cols360 - 1;
    else
        u = u;
    end % Nao deixar passar do limite de colunas por seguranca
    if(u < 0)
        u = 0;
    else
        u = u;
    end
    if(v >= rows360)
        u = rows360 - 1;
    else
        v = v;
    end % Nao deixar passar do limite de linhas por seguranca
    if(v < 0)
        v = 0;
    else
        v = v;
    end
   

 % Ponto na imagem 360 devido a camera 2, finalmente apos as contas, armazenar
    ponto_fc1(j,:) = [ u, v ];
    
      %% Pose da CAMERA 2, so existe aqui rotacao, vamos suprimir as translacoes
    % pois serao irrelevantes e serao compensadas por outros dados
%        r2 = eul2rotm([x(12),- deg2rad(raw2deg(x(13), 'pan')),- deg2rad(raw2deg(x(14), 'tilt'))] );
    % // Vamos criar o frustrum da CAMERA 1, assim como nas nossas funcoes, como o desenho do github
    % // Supondo a variavel F e o raio da esfera como F = R = 1, nao interferiu nas experiencias
    [rows2 cols2,numberOfColorChannels] = size(image2);
    dx2 = x(7) - double(cols2) / 2;
    dy2 = x(8) - double(rows2) / 2;
    
    maxX = ((cols2) - 2*dx2) / (2.0 * x(5));
    minX = ((cols2) + 2*dx2) / (2.0 * x(5));
    maxY = ((rows2) - 2*dy2) / (2.0 * x(6));
    minY = ((rows2) + 2*dy2) / (2.0 * x(6));
   
    
     F = 1;
    p2 = [ minX, minY, F]';
    p4 = [maxX, maxY, F]';
    p5 = [ minX, maxY, F]';
     
    
   
    
    %%  Ponto no frustrum 3D correspondente a feature na imagem 1 em 2D
    ponto3d = p5 + (p4 - p5) * kp2(1) / cols2 + (p2 - p5) * kp2(2) / rows2;
    %% Latitude e longitude no 360
    lat = 180 / 3.1415 * (acos(ponto3d(2) / norm(ponto3d)));
    lon = -180 / 3.1415 * (atan2(ponto3d(3), ponto3d(1)));
    if(lon < 0)
        lon = lon + 360.0;
    else
        lon = lon;
    end
      lat = lat -  deg2rad(raw2deg(x(11),'pan'));
    lon = lon + deg2rad(raw2deg(x(12),'tilt'));
    [rows360 cols360,numberOfColorChannels] = size(im360);
    u = (lon / step_deg);
    v = rows360 - 1 - (lat / step_deg);
    if(u >= cols360)
        u = cols360 - 1;
    else
        u = u;
    end % Nao deixar passar do limite de colunas por seguranca
    if(u < 0)
        u = 0;
    else
        u = u;
    end
    if(v >= rows360)
        u = rows360 - 1;
    else
        v = v;
    end % Nao deixar passar do limite de linhas por seguranca
    if(v < 0)
        v = 0;
    else
        v = v;
    end

    %% Ponto na imagem 360 devido a camera 2, finalmente apos as contas, armazenar
    ponto_fc2(j,:) =[ u, v ];
end
p1_norm = normalize(ponto_fc1,'range');
p2_norm = normalize(ponto_fc2,'range');
o =  norm((p1_norm - p2_norm));

end

function o = Fob3(x,matches,kpt1,kpt2,image1,image2,image3,im360)

kpt1_12 = load('C:/dataset3/1keypt12.txt');
kpt1_13 = load('C:/dataset3/1keypt13.txt');
kpt1_21 = load('C:/dataset3/1keypt21.txt');
kpt1_31 = load('C:/dataset3/1keypt31.txt');

erro1_12 =erro(x(1),x(2),x(3),x(4),x(5),x(6),x(7),x(8), deg2rad(raw2deg(x(14),'pan')),deg2rad(raw2deg(x(13),'tilt')),deg2rad(raw2deg(x(16),'pan')),deg2rad(raw2deg(x(15),'tilt')),kpt1_12,kpt1_21,image1,image2,im360);
erro1_13 =erro(x(1),x(2),x(3),x(4),x(9),x(10),x(11),x(12), deg2rad(raw2deg(x(14),'pan')),deg2rad(raw2deg(x(13),'tilt')),deg2rad(raw2deg(x(18),'pan')),deg2rad(raw2deg(x(17),'tilt')),kpt1_13,kpt1_31,image1,image3,im360);

kpt2_21 = load('C:/dataset3/2keypt21.txt');
kpt2_23 = load('C:/dataset3/2keypt23.txt');
kpt2_12= load('C:/dataset3/2keypt12.txt');
kpt2_32 = load('C:/dataset3/2keypt32.txt');
erro2_21 =erro(x(5),x(6),x(7),x(8),x(1),x(2),x(3),x(4), deg2rad(raw2deg(x(16),'pan')),deg2rad(raw2deg(x(15),'tilt')),deg2rad(raw2deg(x(14),'pan')),deg2rad(raw2deg(x(13),'tilt')),kpt2_21,kpt2_12,image2,image1,im360);
erro2_23 =erro(x(5),x(6),x(7),x(8),x(9),x(10),x(11),x(12),deg2rad(raw2deg(x(16),'pan')),deg2rad(raw2deg(x(15),'tilt')),deg2rad(raw2deg(x(18),'pan')),deg2rad(raw2deg(x(17),'tilt')),kpt2_23,kpt2_32,image2,image3,im360);

kpt3_31 = load('C:/dataset3/3keypt31.txt');
kpt3_32 = load('C:/dataset3/3keypt32.txt');
kpt3_13= load('C:/dataset3/3keypt13.txt');
kpt3_23 = load('C:/dataset3/3keypt23.txt');

erro3_31 =erro(x(9),x(10),x(11),x(12),x(1),x(2),x(3),x(4), deg2rad(raw2deg(x(18),'pan')),deg2rad(raw2deg(x(17),'tilt')),deg2rad(raw2deg(x(14),'pan')),deg2rad(raw2deg(x(13),'tilt')),kpt3_31,kpt3_13,image3,image1,im360);
erro3_32 =erro(x(9),x(10),x(11),x(12),x(5),x(6),x(7),x(8),deg2rad(raw2deg(x(18),'pan')),deg2rad(raw2deg(x(17),'tilt')),deg2rad(raw2deg(x(16),'pan')),deg2rad(raw2deg(x(15),'tilt')),kpt3_32,kpt3_23,image3,image2,im360);

o = (erro1_12 + erro1_13+erro2_21+erro2_23+erro3_31+erro3_32)/6;
end