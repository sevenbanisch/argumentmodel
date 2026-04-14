
% New script for prediction experiment
% 25/03/2025

RUN_ABM = 1;
RUN_MF = 1;

% fourth parameter set (convergence in transition test)
N = 1000; % 100
M = 4;

samplesB = 40;
samplesIN = 1000;
bMin = 2.5;  % 2
bMax = 4.5; % 6
Tmodel = 1000 * M/4;
Ttheory = 1000 * M/4;
Samples = 10000;
%steps = 4000;
%steps = 16000;
%T = 10;


%M = 16;
%M = 64;
pN = 0;

ic_mode = 0; %'randargs';
%ic_mode = 1; % 'uniform';

V = zeros(2*M,1);
V(1:M) = -1/M;
V(M+1:2*M) = 1/M;

if ic_mode == 0
    titleStr = sprintf("random arguments (N = %d, M = %d, T = %d, S = %d)", N, M, Tmodel,Samples);
else
    titleStr = sprintf("random opinions (N = %d, M = %d, T = %d, S = %d)", N, M, Tmodel,Samples);
end  

if RUN_ABM
    finVarABM = zeros(samplesIN,samplesB+1);
end
%finVarMF2D = zeros(Samples,samplesB+1);
finVarMF2Dgs = zeros(Samples,samplesB+1);
finVarMF4D = zeros(Samples,samplesB+1);
%finVarMF4Dmed = zeros(Samples,samplesB+1);
BETA = zeros(1,samplesB+1);

for sB = 0:samplesB
    beta = bMin + (bMax-bMin)*sB/samplesB
    BETA(1,sB+1) = beta;
    resultABM = zeros(samplesIN,1);
    %resultMF2D = zeros(Samples,1);
    resultMF2Dgs = zeros(Samples,1);
    resultMF4D = zeros(Samples,1);
    %resultMF4Dmed = zeros(Samples,1);

    if RUN_ABM
        parfor sIN = 1:samplesIN       
    
            A_t = ReducedArgumentModel(Tmodel,N,M,beta,pN,ic_mode,0);
            resultABM(sIN,1) = var(A_t(:,Tmodel),1);
    
            % Ainit = A_t(:,1);
            % GS = 0
            % O2D_t = MFmodel2Dgs(steps,M,beta,Ainit,GS);
            % resultMF2D(sIN,1) = var(O2D_t(:,Tmodel),1);
            % 
            % GS = 1
            % O2Dgs_t = MFmodel2Dgs(steps,M,beta,Ainit,GS);
            % resultMF2Dgs(sIN,1) = var(O2Dgs_t(:,Tmodel),1);
            % 
            % O4D_t = MFmodel4D(steps,M,beta,Ainit);
            % resultMF4D(sIN,1) = var(O4D_t(:,Tmodel),1);
    
        end
        finVarABM(:,sB+1) = resultABM;
    end % end if RUN_ABM

    if RUN_MF
        parfor sIN = 1:Samples      
    
            if(ic_mode == 1)
                % uniformly at random
                Ainit = 2*(rand(N,1)-0.5);
            else
                % as in the argument model
                args = randi(2,N,2*M)-1;
                Ainit = args*V; 
            end
    
            %GS = 0
            %O2D_t = MFmodel2Dgs(Ttheory,M,beta,Ainit,GS);
            %resultMF2D(sIN,1) = var(O2D_t(:,Ttheory),1);
    
            GS = 1;
            O2Dgs_t = MFmodel2Dgs(Ttheory,M,beta,Ainit,GS);
            resultMF2Dgs(sIN,1) = var(O2Dgs_t(:,Ttheory),1);
    
            splitting = 0; % split second time at mean
            O4D_t = MFmodel4D(Ttheory,M,beta,Ainit,splitting);
            resultMF4D(sIN,1) = var(O4D_t(:,Ttheory),1);
    
            %splitting = 1; % split second time at median
            %O4Dmed_t = MFmodel4D(Ttheory,M,beta,Ainit,splitting);
            %resultMF4Dmed(sIN,1) = var(O4Dmed_t(:,Ttheory),1);
    
        end
    end % RUN_MF

    %finVarMF2D(:,sB+1) = resultMF2D;
    finVarMF2Dgs(:,sB+1) = resultMF2Dgs;
    finVarMF4D(:,sB+1) = resultMF4D;
    %finVarMF4Dmed(:,sB+1) = resultMF4Dmed;

end   

figure
    varthreshh=0.25;
    alpha_value = 0.1;
    hold on

    CR = finVarABM < varthreshh;
    plot(BETA,mean(CR),'b','LineWidth', 3)

    %CR = finVarMF2D < varthreshh;
    %plot(BETA,mean(CR),'r','LineWidth', 2)

    CR = finVarMF2Dgs < varthreshh;
    plot(BETA,mean(CR),'r--','LineWidth', 2)

    CR = finVarMF4D < varthreshh;
    plot(BETA,mean(CR),'g','LineWidth', 2)
    
    %CR = finVarMF4Dmed < varthreshh;
    %plot(BETA,mean(CR),'g--','LineWidth', 2)


    

    %plot(BETA,PEC,'LineWidth', 2)
        xlabel('\beta', 'FontSize', 20);
        ylabel('consensus rate', 'FontSize', 20);
        xlim([bMin,bMax])

        title(titleStr)
        set(gca,'FontSize',16)
        grid on;

    %legend('ABM','theory (2D)', 'theory (2D, GS)', 'theory (4D,mean)','theory (4D,median)', 'FontSize', 14, 'FontWeight', 'bold');    
    legend('ABM', 'theory (2D, GS)', 'theory (4D,mean)', 'FontSize', 14, 'FontWeight', 'bold');    

    hold off




