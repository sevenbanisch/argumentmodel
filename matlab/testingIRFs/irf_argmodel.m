function [deltaO,cnew] = irf_argmodel(bi,bj,V,beta)


K = length(bi);
M = K/2;
%V

oOld = bi*V;


kx = randi(K);
arg = bj(kx);

if bj(kx)==bi(kx)
    cnew = 0;
else
    cnew = 1;
end

dcoh = (2*arg-1) * V(kx) * oOld;
pAdopt = exp(beta * dcoh) / (1 + exp(beta * dcoh));
if(rand < pAdopt)
    bi(kx)=arg;
end  

oNew = bi*V;

deltaO = oNew-oOld;




end