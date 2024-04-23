% Sample model for beta
% Script to produce the simulation data for later analyses

if(1)

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
samplesB = 60;
samplesIN = 50;
bMin = 0;
bMax = 6;
steps = 20000;
T = 10;
N = 1000; % 100
M = 4;
pN = 0;


SIMDATA = zeros(samplesIN,samplesB+1,steps/T);
BETA = zeros(1,samplesB+1);

for sB = 0:samplesB
    beta = bMin + (bMax-bMin)*sB/samplesB
    BETA(1,sB+1) = beta;
    for sIN = 1:samplesIN
        
        %A_t = NormalizedArgumentModel(steps,N,M,beta,pN,0);
        A_t = ReducedArgumentModel(steps,N,M,beta,pN,0);
         
        %SIMDATA(sIN,sB+1,:) = mean(A_t(:,T:T:steps));

        % first version with normalization by N
        SIMDATA(sIN,sB+1,:) = var(A_t(:,T:T:steps),1);
        % default version with normalization by N-1
        %SIMDATA(sIN,sB+1,:) = var(A_t(:,T:T:steps));
               

        fprintf('.')
    end
    
end



figure
% single run
%plot(reshape(SIMDATA(1,10,:),[1,steps/T]))
% mean over samplesIN runs
%plot(reshape(mean(SIMDATA(:,10,:)),[1,steps/T]))
% all samplesIN runs
%plot(reshape(SIMDATA(:,6,:),[samplesIN,steps/T])')

% scatter plot variance at fin
%scatter(BETA,SIMDATA(:,:,steps/T))
scatter(BETA,SIMDATA(:,:,steps/T),'*','MarkerEdgeColor',[0.2 0.5450 0.7330])
hold on
plot(BETA,mean(SIMDATA(:,:,steps/T)))
hold off

end % if sample

if(0)
    nconvR = SIMDATARM >= 0.01;
    nconvO = SIMDATAOM >= 0.01;

    figure
    scatter(BETA,sum(nconvR(:,:,:),3),'*','MarkerEdgeColor',[0.2 0.5450 0.7330])
    hold on
    scatter(BETA,sum(nconvO(:,:,:),3),'*','MarkerEdgeColor',[0.7330 0.2 0.5450])
    plot(BETA,mean(sum(nconvR(:,:,:),3)))
    plot(BETA,mean(sum(nconvO(:,:,:),3)))
    hold off
    set(gca,'Yscale','log')

end

