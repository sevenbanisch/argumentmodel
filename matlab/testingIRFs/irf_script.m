
K = 8;
M = K/2;
V = zeros(K,1);
V(1:M) = 1/M;
V(M+1:2*M) = -1/M;
beta = 0;

% anti-correlation
bi = [1,0,0,0,0,0,0,0];
bj = [0,1,1,1,1,1,1,1];

% correlation
bi = [1,0,0,0,0,0,0,0];
bj = [0,0,0,0,0,0,0,1];

%bi = [1,0,0,1,1,0,1,1];
%bj = [1,1,1,1,1,1,0,0];

%bi = [1,1,0,0,1,1,1,0];
%bj = [1,1,1,1,1,1,0,0];

oi = bi*V;
oj = bj*V;

fprintf('opinions: %f, %f \n',oi,oj)

% new argument in favor
Pplus = 1/(2*M)*( sum( (1-bi(1:M)).*bj(1:M) ) + sum( (1-bj(M+1:2*M)).*bi(M+1:2*M) ));

% new argument against
Pminus = 1/(2*M)*(sum( (1-bj(1:M)).*bi(1:M) ) + sum( (1-bi(M+1:2*M)).*bj(M+1:2*M) ));

fprintf('Probs: %f, %f \n',Pplus,Pminus)

% expected change
pBetaPlus = 1/(1+exp(-oi*beta));
pBetaMinus = 1/(1+exp(oi*beta));
Echange = 1/M * (Pplus*pBetaPlus - Pminus*pBetaMinus);


%b1 = [1,0,0,0,0,0,0,0];
%b2 = [0,0,0,0,0,0,1,1];

samplesIN = 1000000;
data = zeros(2,samplesIN);

for sIN = 1:samplesIN
    [deltaO,cnew] = irf_argmodel(bi,bj,V,beta);
    data(1,sIN) = deltaO;
    data(2,sIN) = cnew;
end

fprintf('Echange: %f, %f \n',Echange,mean(data(1,:)))


%mean(data(2,:))