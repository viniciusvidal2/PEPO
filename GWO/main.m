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

% You can simply define your cost in a seperate file and load its handle to fobj
% The initial parameters that you need are:
%__________________________________________
% fobj = @YourCostFunction
% dim = number of your variables
% Max_iteration = maximum number of generations
% SearchAgents_no = number of search agents
% lb=[lb1,lb2,...,lbn] where lbn is the lower bound of variable n
% ub=[ub1,ub2,...,ubn] where ubn is the upper bound of variable n
% If all the variables have equal lower bound you can just
% define lb and ub as two single number numbers

% To run GWO: [Best_score,Best_pos,GWO_cg_curve]=GWO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj)
%__________________________________________

clear all ;clc;close all
tic

SearchAgents_no = 30; % Number of search agents

Function_name = 'Fob'; % Name of the test function that can be from F1 to F23 (Table 1,2,3 in the paper)

Max_iteration = 1000; % Maximum numbef of iterations

% Posição de cada imagem - Tilt(Vertical) e pan(Horizontal) 
pose = position();


% Load details of the selected benchmark function
[lb,ub,dim,fobj] = Get_Functions_details(Function_name,pose);

[Best_score,Best_pos,GWO_cg_curve] = GWO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);

figure('Position',[500 500 660 290])
%Draw search space
subplot(1,2,1);
% func_plot(Function_name);
title('Parameter space')
xlabel('x_1');
ylabel('x_2');
zlabel([Function_name,'( x_1 , x_2 )'])
%Draw objective space
subplot(1,2,2);
semilogy(GWO_cg_curve,'Color','r')
title('Objective space')
xlabel('Iteration');
ylabel('Best score obtained so far');
axis tight
grid on
box on
legend('GWO')
im360 = zeros(180/0.1, 360/0.1, 3);


%%Plot imagens - Posição otimizada
t = size(Best_pos,2);
nomes_imagens = cell(1, length(2));
nomes_imagens{1} = 'C:/dataset3/imagem_008.png';
nomes_imagens{2} = 'C:/dataset3/imagem_003.png';
nomes_imagens{3} = 'C:/dataset3/imagem_013.png';
im = im360;
l=1;
for j = 1:2:t
    
   im = dothething(Best_pos(j),Best_pos(j+1),double(imread(nomes_imagens{l}))/255,im);
   
  l=l+1;
end
    
figure,imshow(im)
title('Otimizado');
%% Plot imagem - Posição original - Vinda do Robô
nomes_imagens{1} = 'C:/dataset3/imagem_003.png';
nomes_imagens{2} = 'C:/dataset3/imagem_008.png';
nomes_imagens{3} = 'C:/dataset3/imagem_013.png';
imoriginal= zeros(180/0.1, 360/0.1, 3);
l=1;
for j = 3:5:13
    
   imoriginal = dothething(pose(j,1),pose(j,2),double(imread(nomes_imagens{l}))/255,imoriginal);
   
  l=l+1;
end
figure,imshow(imoriginal)
title('Original');
display(['The best solution obtained by GWO is : ', num2str(Best_pos)]);
display(['The best optimal value of the objective funciton found by GWO is : ', num2str(Best_score)]);

toc



