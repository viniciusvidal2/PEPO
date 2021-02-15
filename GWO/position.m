function [pose] = position()
%% Criar angulos para as imagens
% Aqui so estou replicando as orientacoes como estao no codigo interno do
% PEPO
tilts_camera_deg = [50.1, 28, 0, -30.0, -60.2];
ntilts = length(tilts_camera_deg);
step = 25;
inicio_scanner_deg_pan = step/2;
final_scanner_deg_pan = 360 - step/2;

vistas_pan = floor(final_scanner_deg_pan - inicio_scanner_deg_pan)/step; % Vistas na horizontal, somar inicio e final do range
pans_camera_deg = [];

for j=0:vistas_pan
    pans_camera_deg = [pans_camera_deg inicio_scanner_deg_pan + j*step];
end
vistas_pan = vistas_pan + 1;

% Enchendo vetores de waypoints de imagem em deg e raw globais
pans_deg = [];
tilts_deg = [];
for j=1:length(pans_camera_deg)
    for i=1:length(tilts_camera_deg)
        if rem(j, 2) == 1
            tilts_deg = [tilts_deg tilts_camera_deg(i)];
        else
            tilts_deg = [tilts_deg tilts_camera_deg(length(tilts_camera_deg) - i + 1)];
        end
        pans_deg = [pans_deg pans_camera_deg(j)];
    end
end

tilts_im_deg = -tilts_deg;
pans_im_deg = -pans_deg;
pose =[tilts_im_deg;pans_im_deg]';
end