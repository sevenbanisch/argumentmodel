function [error, oi, oj] = irf_belief_comparison(bi, bj, V, beta, samplesIN)

K = length(bi);
M = K/2;

oi = bi*V;
oj = bj*V;

% new argument in favor
Pplus = 1/(2*M)*( sum( (1-bi(1:M)).*bj(1:M) ) + sum( (1-bj(M+1:2*M)).*bi(M+1:2*M) ));

% new argument against
Pminus = 1/(2*M)*(sum( (1-bj(1:M)).*bi(1:M) ) + sum( (1-bi(M+1:2*M)).*bj(M+1:2*M) ));

% expected change
pBetaPlus = 1/(1+exp(-oi*beta));
pBetaMinus = 1/(1+exp(oi*beta));
Echange = 1/M * (Pplus*pBetaPlus - Pminus*pBetaMinus);

data = zeros(2,samplesIN);

for sIN = 1:samplesIN
    [deltaO,cnew] = irf_argmodel(bi,bj,V,beta);
    data(1,sIN) = deltaO;
    data(2,sIN) = cnew;
end

error = (mean(data(1,:))-Echange);

end
