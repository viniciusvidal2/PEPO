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

% This function initialize the first population of search agents
function Positions=initialization(SearchAgents_no,dim,ub,lb,pose)

Boundary_no= size(ub,2); % numnber of boundaries

% If the boundaries of all variables are equal and user enter a signle
% number for both ub and lb
if Boundary_no==1
    Positions=rand(SearchAgents_no,dim).*(ub-lb)+lb;
end
%Inicializando o primeiro agente com os valores vindos do robô -
%Inicializei de acordo com as imagens escolhidas - 8,3 e 13
ini = [pose(8,1),pose(8,2),pose(3,1),pose(3,2),pose(13,1),pose(13,2)]';
% If each variable has a different lb and ub
 Positions(1,:)=ini ;
if Boundary_no>1
    for j=2:SearchAgents_no
    for i=1:dim
        ub_i = ub(i);
        lb_i = lb(i);
        Positions(j,i) = rand(1,1).*(ub_i-lb_i)+lb_i;
    end
    end
end