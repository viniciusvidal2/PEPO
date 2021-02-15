%%Função para projetar na esfera - panoramica 360

function  [im] = dothething(tilt,pan,image1,im360)
%focos e centro otico da camera
fx = 951.4;
fy = 966.2;
cx = 658.6;
cy = 386.6;

step_deg = 0.1;% step em graus para formar o tamanho da 360 fial
rot = roty(pan)*rotx(tilt);% formula que calcula matriz de rotação considerando os valores em tilt e pan em graus


[rows1 cols1,numberOfColorChannels] = size(image1);
step_deg = 0.1;
F =1;
dx = cx - cols1/2;
dy = cy - rows1/2;
% Calcular os 4 pontos do frustrum - Imagem no 3d
% 		
% 										origin of the camera = p1
% 										p2--------p3
% 										|          |
% 										|  pCenter |<--- Looking from p1 to pCenter
% 										|          |
% 										p5--------p4
% 		
maxX = ((cols1) - 2*dx) / (2.0 * fx);
minX = -((cols1) + 2*dx) / (2.0 * fx);
maxY = ((rows1) - 2*dy) / (2.0 * fy);
minY = -((rows1) + 2*dy) / (2.0 * fy);
p = [0, 0, 0]';
p1 = rot * p;
p= [ minX, minY, F]';
p2 = rot * p;
p = [maxX, minY, F]';
p3 = rot * p;
p = [maxX, maxY, F]';
p4 = rot * p;
p = [minX, maxY, F]';
p5 = rot * p;
p  =[0, 0, F]';
pCenter = rot * p;
% A partir de frustrum, calcular a posicao de cada pixel da imagem fonte em XYZ, assim como quando criavamos o plano do frustrum
hor_step = (p4 - p5) / cols1;
ver_step = (p2 - p5) / rows1;
for v = 1:size(image1,1)
    for u = 1:size(image1,2)% Vai criando o frustrum a partir dos cantos da imagem
        
        % Ponto no frustrum 3D
        ponto3d =  p5 + hor_step * u + ver_step * v;
        if norm(pCenter - ponto3d)< norm(p4-p5)/2
            %  Latitude e longitude no 360       
            %  Calcular latitude e longitude da esfera de volta a partir de XYZ
           
            lat = 180 / 3.1415 * (acos(ponto3d(2) / norm(ponto3d)));
            lon = -180 / 3.1415 * (atan2(ponto3d(3), ponto3d(1)));
            % Ajustar regiao do angulo negativo, mantendo o 0 no centro da imagem
            if(lon < 0)
                lon = lon + 360.0;
            else
                lon = lon;
            end
            % Pelas coordenadas, estimar posicao do pixel que deve sair na 360 final e pintar - da forma como criamos a esfera
            [rows360 cols360,numberOfColorChannels] = size(im360);
            xo  = ceil(lon / step_deg);
            yo = ceil(rows360 +1 - (lat / step_deg));
            if(xo > cols360)
                xo = cols360 ;
            end
            % Nao deixar passar do limite de colunas por seguranca
            if(xo <= 0)
                xo = 1;
            end
            if(yo > rows360)
                yo = rows360 ;
            end
            % Nao deixar passar do limite de linhas por seguranca
            if(yo <= 0)
                yo = 1;
            end
            % Pintar a imagem final com as cores encontradas
            im360(yo, xo , :) = image1(size(image1, 1) - v + 1,  u , :);
        end
    end
end

im = im360;
end