
function [FINV,colors] = basinofattraction(beta,N)

%beta=2.9;
p=1/2;


T = 1000;
offset = 1/5000;
samplesOA = 200;
samplesOB = 200;
oAMin = -0.5;
oAMax = 0.25;
oBMin = -0.25;
oBMax = 0.5;


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
            dA = (1-p)*IRF(oA,oA,beta) + p * IRF(oA,oB,beta);
            dB = (1-p)*IRF(oB,oB,beta) + p * IRF(oB,oA,beta);
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
figure(4)
subplot(1,2,1)
contour(OAs,OBs,FINV',1,'LineWidth', 2);
        xlabel('o_A', 'FontSize', 16);
        ylabel('o_B', 'FontSize', 16);
        set(gca,'FontSize',16)
        grid on;
        xlim([oAMin,oAMax]);
        ylim([oBMin,oBMax]);

hold on

mode = 0; % random arguments
[oA,oB,colors] = testoninitialconditions(N,p,beta,mode,0);

scatter(oA,oB,8, colors, 'filled') % 50 is the marker size
    colormap('jet'); % Set the colormap (optional)
    %colorbar; % Show colorbar (optional)
    xlabel('opinion group A', 'FontSize', 20);
    ylabel('opinion group B', 'FontSize', 20);
    %xlim([-1,1]);
    %ylim([-1,1]);
    
    set(gca,'FontSize',16)
    grid on;


mode = 1; % random opinions
[oA,oB,colorsU] = testoninitialconditions(N,p,beta,mode,0);

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
subplot(1,2,1)
%histogram(colors, 'Normalization', 'probability');
hist(colors)
    xlabel('variance', 'FontSize', 20);
    ylabel('frequency', 'FontSize', 20);
    ylim([0,10000]);
    xlim([0,1]);
    set(gca,'FontSize',12)
    grid on;
    

subplot(1,2,2)
%histogram(colorsU, 'Normalization', 'probability');
hist(colorsU)
    xlabel('variance', 'FontSize', 20);
    ylabel('frequency', 'FontSize', 20);
    ylim([0,10000]);
    xlim([0,1]);
    set(gca,'FontSize',12)
    grid on;
   


% scatter(OAs, OBs, [], FIN, 'filled');
% xlabel('X');
% ylabel('Y');
% title('Scatter Plot with RGB Colors');
% axis([-1 1 -1 1]); % Set axis limits to match the range [-1, 1]    

end



