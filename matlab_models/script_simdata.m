% Sample model for beta
% Script to produce the simulation data for later analyses

tic


%% 1. Data Generation
if(0)

% % first parameter set
% samplesB = 100;
% samplesIN = 25;
% bMin = 0;
% bMax = 6;
% steps = 50000;
% T = 100;
% N = 500; % 100
% M = 4;
% pN = 0;
% 
% % second parameter set
% samplesB = 20;
% samplesIN = 25;
% bMin = 0;
% bMax = 1;
% steps = 50000;
% T = 10;
% N = 500; % 100
% M = 4;
% pN = 0;


% third parameter set
% samplesB = 60;
% samplesIN = 100;
% bMin = 2;
% bMax = 5;
% steps = 10000;
% T = 10;
% N = 1000; % 100
% M = 4;
% pN = 0;


% % fourth parameter set (convergence in transition test)
% samplesB = 60;
% samplesIN = 100;
% bMin = 2;
% bMax = 5;
% steps = 1000;
% steps = 4000;
% T = 10;
% N = 100; % 100
% M = 4;
% M = 16;
% pN = 0;


% fifth parameter set (Figure 1, basic bifurcation)
samplesB = 80;
samplesIN = 100;

N = 1000; 

M = 4;
pN = 0;

bMin = 0;
bMax = 8;

basesteps = 1000;
steps = M * basesteps;

T = M * 100; % just to not store for all times
OBSperiod = 25; % take temporal average over last 25 % of the runs


% sixth parameter set (Figure 1, insets)
samplesB = 60;
samplesIN = 100;

N = 1000; 

M = 4;
pN = 0;

bMin = 2;
bMax = 8;

basesteps = 1000;
steps = M * basesteps * 4; % 16000 steps

T = M * 100; % just to not store for all times
OBSperiod = 25; % take temporal average over last 25 % of the runs


SIMDATA = zeros(samplesIN,samplesB+1,steps/T);
OBS01 = zeros(samplesIN,samplesB+1);
BETA = zeros(1,samplesB+1);

parfor sB = 0:samplesB
    beta = bMin + (bMax-bMin)*sB/samplesB
    BETA(1,sB+1) = beta;
    for sIN = 1:samplesIN
        
        %A_t = NormalizedArgumentModel(steps,N,M,beta,pN,0);
        A_t = ReducedArgumentModel(steps,N,M,beta,pN,0);
         
        %SIMDATA(sIN,sB+1,:) = mean(A_t(:,T:T:steps));

        % first version with normalization by N
        SIMDATA(sIN,sB+1,:) = var(A_t(:,T:T:steps),1);
        OBS01(sIN,sB+1) = mean(var(A_t(:,steps-steps/OBSperiod:steps),1));

        % default version with normalization by N-1
        %SIMDATA(sIN,sB+1,:) = var(A_t(:,T:T:steps));
               

        fprintf('.')
    end
    
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

if(1)

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






% single run
%plot(reshape(SIMDATA(1,10,:),[1,steps/T]))
% mean over samplesIN runs
%plot(reshape(mean(SIMDATA(:,10,:)),[1,steps/T]))
% all samplesIN runs
%plot(reshape(SIMDATA(:,6,:),[samplesIN,steps/T])')

% scatter plot variance at fin
%scatter(BETA,SIMDATA(:,:,steps/T))




% scatter(BETA,SIMDATA(:,:,steps/T),'*','MarkerEdgeColor',[0.2 0.5450 0.7330])
% hold on
% plot(BETA,mean(SIMDATA(:,:,steps/T)))
% hold off

%end % if sample


%% For Convergence Rate

% if(0)
%     nconvR = SIMDATA >= 0.01;
%     %nconvO = SIMDATAOM >= 0.01;

%     CR = SIMDATA(:,:,steps/T) < 0.2;
% 
%     figure
%     scatter(BETA,sum(nconvR(:,:,:),3),'*','MarkerEdgeColor',[0.2 0.5450 0.7330])
%     hold on
%     %scatter(BETA,sum(nconvO(:,:,:),3),'*','MarkerEdgeColor',[0.7330 0.2 0.5450])
%     plot(BETA,mean(sum(nconvR(:,:,:),3)))
%     %plot(BETA,mean(sum(nconvO(:,:,:),3)))
%     hold off
%     set(gca,'Yscale','log')
% 
% end

toc