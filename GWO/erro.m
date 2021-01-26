function [res] = erro(fx1,fy1,cx1,cy1,fx2,fy2,cx2,cy2,pitch1,yaw1,pitch2,yaw2,kpt1,kpt2,image1,image2,im360)
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
    dx1 = cx1 - double(cols1) / 2;
    dy1 = cy1 - double(rows1) / 2;

    F = 1;
    maxX = ((cols1) - 2*dx1) / (2.0 * fx1);
    minX = ((cols1) + 2*dx1) / (2.0 * fx1);
    maxY = ((rows1) - 2*dy1) / (2.0 * fy1);
    minY = ((rows1) + 2*dy1) / (2.0 * fy1);
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
   
    lat = lat -  pitch1;
    lon = lon + yaw1;
    
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
    dx2 = cx2 - double(cols2) / 2;
    dy2 = cy2 - double(rows2) / 2;
    
    maxX = ((cols2) - 2*dx2) / (2.0 * fx2);
    minX = ((cols2) + 2*dx2) / (2.0 * fx2);
    maxY = ((rows2) - 2*dy2) / (2.0 * fy2);
    minY = ((rows2) + 2*dy2) / (2.0 * fy2);
   
    
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
      lat = lat -  pitch2;
    lon = lon + yaw2;
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
res =  norm((p1_norm - p2_norm));
end