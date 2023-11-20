

samplesB = 50;
samplesIN = 25;

bMax = 8;
steps = 4000;
N = 100;
M = 4;
pN = 0;

x = zeros(1,samplesB+1);
DATA_o = zeros(samplesIN,samplesB+1);
DATA_r = zeros(samplesIN,samplesB+1);

for sB = 0:samplesB
    beta = bMax*sB/samplesB
    x(1,sB+1) = beta;
    for sIN = 1:samplesIN
        
        A_to = NormalizedArgumentModel(steps,N,M,beta,pN,0);
        A_tr = ReducedArgumentModel(steps,N,M,beta,pN,0);

        DATA_o(sIN,sB+1) = max(var(A_to));
        DATA_r(sIN,sB+1) = max(var(A_tr));

        fprintf('.')
    end
    
end

figure
plot(x,mean(DATA_o))
hold on
plot(x,mean(DATA_r))
hold off
xlim([0,bMax]);



