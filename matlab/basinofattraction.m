
function [FINV,colors] = basinofattraction(beta,N,M)

%beta=2.9;
pInter=1/2;

Samples = 1000;
T = 500;
Tmodel = 1000;
offset = 0;%1/5000;
samplesOA = 800;%400;
samplesOB = 800;%400;
oAMin = -0.65;
oAMax = 0.025;
oBMin = -0.025;
oBMax = 0.65;

oAMin = -0.8;
oAMax = 0.2;
oBMin = -0.2;
oBMax = 0.8;

FIN = zeros(samplesOA+1,samplesOB+1,3)+0.5;
FINV=zeros(samplesOA+1,samplesOB+1);
OAs = zeros(samplesOA+1,1);
OBs = zeros(samplesOB+1,1);
%FINB = zeros(samplesOA+1,samplesOB+1);

for sOA = 0:samplesOA
    for sOB = 0:samplesOB

        %beta = bMin + (bMax-bMin)*sB/samplesB

        oA = oAMin + (oAMax-oAMin)*(sOA/samplesOA * 2 - 1);
        OAs(sOA+1) = oA;
        oB = oBMin + (oBMax-oBMin)*(sOB/samplesOB *2 - 1);
        OBs(sOB+1) = oB;       
        for t = 1:T
            dA = (1-pInter)*IRF(oA,oA,beta,M) + pInter * IRF(oA,oB,beta,M);
            dB = (1-pInter)*IRF(oB,oB,beta,M) + pInter * IRF(oB,oA,beta,M);
            oA = oA + dA + offset*(rand - 0.5);
            oB = oB + dB + offset*(rand - 0.5);
        end

        FIN(sOA+1,sOB+1,1)=oA;
        FIN(sOA+1,sOB+1,2)=oB;
        FINV(sOA+1,sOB+1) = var([oA,oB])/2;
        
    end

end



% imshow(FIN,'InitialMagnification', 'fit')
%     xlabel('oA', 'FontSize', 16);
%     ylabel('oB', 'FontSize', 16);
% 
%     set(gca,'FontSize',12)
% 
% pcolor(FINV)  
figure
%subplot(1,2,1)
contour(OAs,OBs,FINV',1,'LineWidth', 2);
        xlabel('o_A', 'FontSize', 16);
        ylabel('o_B', 'FontSize', 16);
        set(gca,'FontSize',16)
        grid on;
        xlim([oAMin,oAMax]);
        ylim([oBMin,oBMax]);

hold on

if(0)
ic_mode = 0; % binomial arguments
[oA,oB,colors] = testoninitialconditions(N,M,Tmodel,Samples,pInter,beta,ic_mode,0);

scatter(oA,oB,8, colors, 'filled') % 50 is the marker size
    colormap('jet'); % Set the colormap (optional)
    %colorbar; % Show colorbar (optional)
    xlabel('opinion group A', 'FontSize', 20);
    ylabel('opinion group B', 'FontSize', 20);
    %xlim([-1,1]);
    %ylim([-1,1]);
    
    set(gca,'FontSize',16)
    grid on;

% Larger M    
[oA,oB,colorsM8] = testoninitialconditions(N,8,Tmodel,Samples,pInter,beta,ic_mode,0);

scatter(oA,oB,8, colorsM8, 'filled') % 50 is the marker size
    colormap('jet'); % Set the colormap (optional)
    %colorbar; % Show colorbar (optional)
    xlabel('opinion group A', 'FontSize', 20);
    ylabel('opinion group B', 'FontSize', 20);
    %xlim([-1,1]);
    %ylim([-1,1]);
    
    set(gca,'FontSize',16)
    grid on;

% Larger M    
[oA,oB,colorsM16] = testoninitialconditions(N,16,Tmodel,Samples,pInter,beta,ic_mode,0);

scatter(oA,oB,8, colorsM16, 'filled') % 50 is the marker size
    colormap('jet'); % Set the colormap (optional)
    %colorbar; % Show colorbar (optional)
    xlabel('opinion group A', 'FontSize', 20);
    ylabel('opinion group B', 'FontSize', 20);
    %xlim([-1,1]);
    %ylim([-1,1]);
    
    set(gca,'FontSize',16)
    grid on;     

ic_mode = 1; % random opinions
[oA,oB,colorsU] = testoninitialconditions(N,M,Tmodel,Samples,pInter,beta,ic_mode,0);

scatter(oA,oB,8, colorsU, 'filled') % 50 is the marker size
    colormap('jet'); % Set the colormap (optional)
    %colorbar; % Show colorbar (optional)
    xlabel('opinion group A', 'FontSize', 20);
    ylabel('opinion group B', 'FontSize', 20);
    %xlim([-1,1]);
    %ylim([-1,1]);
    
    set(gca,'FontSize',16)
    grid on;    

hold off

figure
subplot(4,1,2)
%histogram(colors, 'Normalization', 'probability');
hist(colors)
    xlabel('variance', 'FontSize', 20);
    ylabel('frequency', 'FontSize', 20);
    %ylim([0,10000]);
    xlim([0,1]);
    set(gca,'FontSize',12)
    grid on;

subplot(4,1,3)
%histogram(colors, 'Normalization', 'probability');
hist(colorsM8)
    xlabel('variance', 'FontSize', 20);
    ylabel('frequency', 'FontSize', 20);
    %ylim([0,10000]);
    xlim([0,1]);
    set(gca,'FontSize',12)
    grid on;   

subplot(4,1,4)
%histogram(colors, 'Normalization', 'probability');
hist(colorsM16)
    xlabel('variance', 'FontSize', 20);
    ylabel('frequency', 'FontSize', 20);
    %ylim([0,10000]);
    xlim([0,1]);
    set(gca,'FontSize',12)
    grid on;     

subplot(4,1,1)
%histogram(colorsU, 'Normalization', 'probability');
hist(colorsU)
    xlabel('variance', 'FontSize', 20);
    ylabel('frequency', 'FontSize', 20);
    %ylim([0,10000]);
    xlim([0,1]);
    set(gca,'FontSize',12)
    grid on;
   


% scatter(OAs, OBs, [], FIN, 'filled');
% xlabel('X');
% ylabel('Y');
% title('Scatter Plot with RGB Colors');
% axis([-1 1 -1 1]); % Set axis limits to match the range [-1, 1]    

end % if 0

end



