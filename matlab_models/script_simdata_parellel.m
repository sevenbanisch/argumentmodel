% Sample model for beta
% Script to produce the simulation data for later analyses

tic


%% 1. Data Generation
if(1)


% 
% % fifth parameter set (Figure 1, basic bifurcation)
% samplesB = 80;
% samplesIN = 100;
% 
% N = 1000; 
% 
% M = 4;
% pN = 0;
% 
% bMin = 0;
% bMax = 8;
% 
% basesteps = 1000;
% steps = M * basesteps;
% 
% T = M * 100; % just to not store for all times
% OBSperiod = 25; % take temporal average over last 25 % of the runs
% 

% sixth parameter set (Figure 1, insets)
samplesB = 60;
samplesIN = 100;

samplesB = 30;
samplesIN = 3;

N = 100; 

M = 4;
pN = 0;

bMin = 2;
bMax = 8;

basesteps = 1000;
steps = M * basesteps * 4; % 64000 steps

T = M * 100; % just to not store for all times
OBSperiod = 25; % take temporal average over last 25 % of the runs

SIMDATA = zeros(samplesIN,samplesB+1,steps/T);
OBS01 = zeros(samplesIN,samplesB+1);
BETA = zeros(1,samplesB+1);

% 3. Prepare for all parallel
numruns = samplesIN+samplesB+1;
par_list = zeros(2,numruns);
numrun = 1;

for sB = 0:samplesB
    beta = bMin + (bMax-bMin)*sB/samplesB;
    BETA(1,sB+1) = beta;
    for sIN = 1:samplesIN
        par_list(:,numrun) = [sB,sIN];
        numrun=numrun+1;
    end
end
%par_list
for run = 1:numruns

    sB = par_list(1,run);
    beta = bMin + (bMax-bMin)*sB/samplesB;
    sIN = par_list(2,run);  
        
    %A_t = NormalizedArgumentModel(steps,N,M,beta,pN,0);
        
    A_t = ReducedArgumentModel(steps,N,M,beta,pN,0);

        % first version with normalization by N
    SIMDATA(sIN,sB+1,:) = var(A_t(:,T:T:steps),1);
    OBS01(sIN,sB+1) = mean(var(A_t(:,steps-steps/OBSperiod:steps),1));

    %fprintf('.')
       
end

end % if 1. Sim Data

%% 2. Scatter plot of variance in (quasi-)stationary state
% 

if(0)
figure
alpha_value = 0.1;
scatter(BETA,OBS01,'o','MarkerEdgeColor',[0.2 0.5450 0.7330],...
                       'MarkerFaceColor',[0.2 0.5450 0.7330],...
                       'MarkerFaceAlpha', alpha_value,...
                       'MarkerEdgeAlpha', alpha_value);
    xlabel('\beta', 'FontSize', 20);
    ylabel('variance', 'FontSize', 20);
    set(gca,'FontSize',20)
    grid on;

end % if 2.Scatter plot


%% 2.2 Inset
% 

if(0)

rmSIMDATA=SIMDATA;
rmBETA = BETA;

figure
alpha_value = 0.1;

CR = emSIMDATA(:,:,steps/T) > 0.2; % 16000 steps
plot(emBETA,mean(CR))
hold on
CR = emSIMDATA(:,:,3) > 0.2; % 1200 steps
plot(emBETA,mean(CR))

CR = emSIMDATA(:,:,10) > 0.2; % 4000 steps
plot(emBETA,mean(CR))


CR = rmSIMDATA(:,:,steps/T) > 0.2; % 16000 steps
plot(rmBETA,mean(CR))

CR = rmSIMDATA(:,:,3) > 0.2; % 1200 steps
plot(rmBETA,mean(CR))

CR = rmSIMDATA(:,:,10) > 0.2; % 4000 steps
plot(rmBETA,mean(CR))
    
    xlabel('\beta', 'FontSize', 20);
    ylabel('polarization probability', 'FontSize', 20);
    set(gca,'FontSize',20)
    xlim([2.5,6])
    grid on;

hold off

end % if 2.Scatter plot


toc