% 4D mean field model

function [O_t] = MFmodel4D(T,M,beta,Ainit,splitting)

if splitting == 1
    oA = median( Ainit(Ainit<0));
    oB = median( Ainit(Ainit>0));
else
    oA = mean( Ainit(Ainit<0));
    oB = mean( Ainit(Ainit>0));
end

oAstrong = mean( Ainit( Ainit < 0 & Ainit <= oA ));
gAs = mean( Ainit < 0 & Ainit <= oA );
oAweak = mean( Ainit( Ainit < 0 & Ainit >= oA ));
gAw = mean( Ainit < 0 & Ainit >= oA );

oBstrong = mean( Ainit(Ainit > 0 & Ainit >= oB));
gBs = mean(Ainit > 0 & Ainit >= oB);
oBweak = mean( Ainit(Ainit > 0 & Ainit <= oB));
gBw = mean(Ainit > 0 & Ainit <= oB);

O_t = zeros(4,T);
O_t(1,1) = oAstrong;
O_t(2,1) = oAweak;
O_t(3,1) = oBweak;
O_t(4,1) = oBstrong;
oAst = oAstrong;
oAwt = oAweak;
oBwt = oBweak;
oBst = oBstrong;


for step = 2:T
    dAs = gAs*IRF(oAst,oAst,beta,M) + gAw*IRF(oAst,oAwt,beta,M) + gBw*IRF(oAst,oBwt,beta,M)+ gBs*IRF(oAst,oBst,beta,M);
    dAw = gAs*IRF(oAwt,oAst,beta,M) + gAw*IRF(oAwt,oAwt,beta,M) + gBw*IRF(oAwt,oBwt,beta,M)+ gBs*IRF(oAwt,oBst,beta,M);
    dBs = gAs*IRF(oBst,oAst,beta,M) + gAw*IRF(oBst,oAwt,beta,M) + gBw*IRF(oBst,oBwt,beta,M)+ gBs*IRF(oBst,oBst,beta,M);
    dBw = gAs*IRF(oBwt,oAst,beta,M) + gAw*IRF(oBwt,oAwt,beta,M) + gBw*IRF(oBwt,oBwt,beta,M)+ gBs*IRF(oBwt,oBst,beta,M);

    oAst = oAst + dAs;
    oAwt = oAwt + dAw;   
    oBwt = oBwt + dBw;
    oBst = oBst + dBs;

    % oAst = oAst + dAs*gAs;
    % oAwt = oAwt + dAw*gAw;   
    % oBwt = oBwt + dBw*gBw;
    % oBst = oBst + dBs*gBs;

    O_t(1,step) = oAst;
    O_t(2,step) = oAwt;
    O_t(3,step) = oBwt;
    O_t(4,step) = oBst;
end

end