% 4D mean field model

function [O_t] = MFmodel2Dgs(T,M,beta,Ainit,GS)

oA = mean( Ainit(Ainit<0));
oB = mean( Ainit(Ainit>0));

if GS
    gA = mean(Ainit<0);
    gB = mean(Ainit>0);
else
    gA=1/2;
    gB=1/2;
end


O_t = zeros(2,T);
O_t(1,1) = oA;
O_t(2,1) = oB;

oAt = oA;
oBt = oB;

for step = 2:T
    dA = gA*IRF(oAt,oAt,beta,M) + gB * IRF(oAt,oBt,beta,M);
    dB = gB*IRF(oBt,oBt,beta,M) + gA * IRF(oBt,oAt,beta,M);
    oAt = oAt + dA; 
    oBt = oBt + dB; 
    O_t(1,step) = oAt;
    O_t(2,step) = oBt;
end

end