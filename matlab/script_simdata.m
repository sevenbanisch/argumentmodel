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


% % fifth parameter set (Figure: basic bifurcation)
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


% sixth parameter set (Figure: runtime)
samplesB = 60;
samplesIN = 1000;

%samplesB = 10;
%samplesIN = 3;

N = 100; 

M = 4;
pN = 0;

bMin = 2;
bMax = 8;

basesteps = 1000;
steps = M * basesteps * 16; % for N = 1000, 16000 is ok. 64000 steps is toooo much!

T = M * 100; % just to not store for all times
OBSperiod = 25; % take temporal average over last 25 % of the runs


rmSIMDATA = zeros(samplesIN,samplesB+1,steps/T);
rmOBS01 = zeros(samplesIN,samplesB+1);
rmBETA = zeros(1,samplesB+1);

emSIMDATA = zeros(samplesIN,samplesB+1,steps/T);
emOBS01 = zeros(samplesIN,samplesB+1);
emBETA = zeros(1,samplesB+1);

parfor sB = 0:samplesB
    beta = bMin + (bMax-bMin)*sB/samplesB
    rmBETA(1,sB+1) = beta;
    emBETA(1,sB+1) = beta;
    for sIN = 1:samplesIN
        
        A_t = NormalizedArgumentModel(steps,N,M,beta,pN,0);

        emSIMDATA(sIN,sB+1,:) = var(A_t(:,T:T:steps),1);
        emOBS01(sIN,sB+1) = mean(var(A_t(:,steps-steps/OBSperiod:steps),1));        
        
        A_t = ReducedArgumentModel(steps,N,M,beta,pN,0);

        % normalization by N
        rmSIMDATA(sIN,sB+1,:) = var(A_t(:,T:T:steps),1);
        rmOBS01(sIN,sB+1) = mean(var(A_t(:,steps-steps/OBSperiod:steps),1));

        fprintf('.')
    end
    
end

end % if 1. Sim Data

%% 2. Scatter plot of variance in (quasi-)stationary state
% 

if(0)
figure
alpha_value = 0.1;
scatter(rmBETA,rmOBS01,'o','MarkerEdgeColor',[0.2 0.5450 0.7330],...
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

% emSIMDATA=SIMDATA;
% emBETA = BETA;
% rmBETA = BETA;

varthreshh=0.5;

figure
subplot(1,2,1)
alpha_value = 0.1;
hold on

 CR = emSIMDATA(:,:,steps/T) > varthreshh; % 16000 steps
 plot(emBETA,mean(CR))
 
 CR = emSIMDATA(:,:,steps/(2*T)) > varthreshh; % 1200 steps
 plot(emBETA,mean(CR))
 
 CR = emSIMDATA(:,:,steps/(4*T)) > varthreshh; % 4000 steps
 plot(emBETA,mean(CR))

 CR = emSIMDATA(:,:,steps/(8*T)) > varthreshh; % 4000 steps
 plot(emBETA,mean(CR))
  
 CR = emSIMDATA(:,:,steps/(16*T)) > varthreshh; % 4000 steps
 plot(emBETA,mean(CR))
steps/(32*T)
 CR = emSIMDATA(:,:,steps/(32*T)) > varthreshh; % 4000 steps
 plot(emBETA,mean(CR))

 CR = emSIMDATA(:,:,2) > varthreshh; % 4000 steps
 plot(emBETA,mean(CR),'--')

    xlabel('\beta', 'FontSize', 20);
    ylabel('polarization probability', 'FontSize', 20);
    set(gca,'FontSize',20)
    xlim([2.5,6])
    grid on;

 hold off
 % ---------

 subplot(1,2,2)
 hold on

    CR = rmSIMDATA(:,:,steps/T) > varthreshh; % 16000 steps
    plot(rmBETA,mean(CR))
    
    CR = rmSIMDATA(:,:,steps/(2*T)) > varthreshh; % 1200 steps
    plot(rmBETA,mean(CR))
    % 
    CR = rmSIMDATA(:,:,steps/(4*T)) > varthreshh; % 1200 steps
    plot(rmBETA,mean(CR))
    % 
    CR = rmSIMDATA(:,:,steps/(8*T)) > varthreshh; % 4000 steps
    plot(rmBETA,mean(CR))
    
    CR = rmSIMDATA(:,:,steps/(16*T)) > varthreshh; % 1200 steps
    plot(rmBETA,mean(CR))
    % 
    CR = rmSIMDATA(:,:,steps/(32*T)) > varthreshh; % 4000 steps
    plot(rmBETA,mean(CR))
    
    CR = rmSIMDATA(:,:,2) > varthreshh; % 800 steps
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