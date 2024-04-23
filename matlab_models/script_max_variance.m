

samplesB = 100;
samplesIN = 25;

bMax = 6;
steps = 10000;
N = 1000;
M = 4;
pN = 0;

x = zeros(1,samplesB+1);
MXVAR_o = zeros(samplesIN,samplesB+1);
MXVAR_r = zeros(samplesIN,samplesB+1);

FNVAR_o = zeros(samplesIN,samplesB+1);
FNVAR_r = zeros(samplesIN,samplesB+1);

T1000VAR_o = zeros(samplesIN,samplesB+1);
T1000VAR_r = zeros(samplesIN,samplesB+1);

T3000VAR_o = zeros(samplesIN,samplesB+1);
T3000VAR_r = zeros(samplesIN,samplesB+1);

T5000VAR_o = zeros(samplesIN,samplesB+1);
T5000VAR_r = zeros(samplesIN,samplesB+1);

for sB = 0:samplesB
    beta = bMax*sB/samplesB
    x(1,sB+1) = beta;
    for sIN = 1:samplesIN
        
        A_to = NormalizedArgumentModel(steps,N,M,beta,pN,0);
        A_tr = ReducedArgumentModel(steps,N,M,beta,pN,0);

        MXVAR_o(sIN,sB+1) = max(var(A_to));
        MXVAR_r(sIN,sB+1) = max(var(A_tr));

        FNVAR_o(sIN,sB+1) = var(A_to(:,steps));
        FNVAR_r(sIN,sB+1) = var(A_tr(:,steps));

        T1000VAR_o(sIN,sB+1) = var(A_to(:,1000));
        T1000VAR_r(sIN,sB+1) = var(A_tr(:,1000));

        T3000VAR_o(sIN,sB+1) = var(A_to(:,3000));
        T3000VAR_r(sIN,sB+1) = var(A_tr(:,3000));

        T5000VAR_o(sIN,sB+1) = var(A_to(:,5000));
        T5000VAR_r(sIN,sB+1) = var(A_tr(:,5000));

        fprintf('.')
    end
    
end

%figure
%plot(x,mean(MXVAR_o))
%hold on
%plot(x,mean(MXVAR_r))
%hold off
%xlim([0,bMax]);


figure
plot(x,mean(FNVAR_o))
hold on
plot(x,mean(FNVAR_r))
hold off
xlim([0,bMax]);
