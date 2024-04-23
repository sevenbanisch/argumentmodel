

p=1/2;
N = 1000;
samplesB = 60;
bMin = 2;
bMax = 5;

BETA = zeros(1,samplesB+1);
PEC = zeros(1,samplesB+1);

for sB = 0:samplesB
    beta = bMin + (bMax-bMin)*sB/samplesB
    BETA(1,sB+1) = beta;
    %for sIN = 1:samplesIN

    [oA,oB,colors] = testoninitialconditions(N,p,beta,0);
    PEC(sB+1) = mean(colors < 0.2);

end

figure
plot(BETA,PEC)
    xlabel('\beta', 'FontSize', 20);
    ylabel('consensus rate', 'FontSize', 20);
    
    set(gca,'FontSize',16)
    grid on;

