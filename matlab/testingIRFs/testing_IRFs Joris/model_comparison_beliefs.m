K = 8;
M = K/2;
V = zeros(K,1);
V(1:M) = 1/M;
V(M+1:2*M) = -1/M;
beta = 2;

bi = [0,0,0,0,0,0,0,0];
bj = [1,1,1,1,1,1,1,1];

oi = bi*V;
oj = bj*V;

fprintf('opinions: %f, %f \n',oi,oj)

% new argument in favor
Pplus = 1/(2*M)*( sum( bj(1:M) ) - sum( bi(1:M).*bj(1:M) ) + sum( bi(M+1:2*M) ) - sum( bj(M+1:2*M).*bi(M+1:2*M) ));

% new argument against
Pminus = 1/(2*M)*(sum( bi(1:M) ) - sum( bj(1:M).*bi(1:M) ) + sum( bj(M+1:2*M) ) - sum( bi(M+1:2*M).*bj(M+1:2*M) ));

fprintf('Probs: %f, %f \n',Pplus,Pminus)

% expected change
pBetaPlus = 1/(1+exp(-oi*beta));
pBetaMinus = 1/(1+exp(oi*beta));
Echange = 1/M * (Pplus*pBetaPlus - Pminus*pBetaMinus);

Echange_naive = 1/(4*M) * (oj-oi + tanh(oi*beta/2)*(1-oi*oj));

samplesIN = 1000000;
data = zeros(2,samplesIN);

for sIN = 1:samplesIN
    [deltaO,cnew] = irf_argmodel(bi,bj,V,beta);
    data(1,sIN) = deltaO;
    data(2,sIN) = cnew;
end

fprintf('Echange: %f, %f \n',Echange, mean(data(1,:)))

irf_belief_comparison(bi,bj,V,beta,1000)

figure
hold on
histogram(data(1,:), ...
    "Normalization","probability", ...
    "BinWidth", 0.02, ...
    "BinLimits",[-0.25,0.25]);
pat = xline(mean(data(1,:)),"k", "LineWidth",2, "Label","PAT", "LabelVerticalAlignment","top");
pat.FontSize = 15;

irf = xline(Echange,"-.k", "LineWidth",2, "Label","IRF", "LabelVerticalAlignment","middle");
irf.FontSize = 15;
naiveirf = xline(Echange_naive,"--k", "LineWidth",2, "Label","Naive IRF", "LabelVerticalAlignment","bottom");
naiveirf.FontSize = 15;
%annotation('doublearrow',[0.5+2*Echange 0.5+2*mean(data(1,:))],[0.5 0.5])
legend("Histogram of Opinion Change in PAT Model", "Mean of PAT Model", "Mean of IRF Model", "Mean of Naive IRF Model")

bi_str = sprintf('%d ', bi);
bj_str = sprintf('%d ', bj);

title(["Opinion Change for", ...
    sprintf('b_{i} = [%s]', bi_str), ...
    "and", ...
    sprintf('b_{j} = [%s]', bj_str)]);

