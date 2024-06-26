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
% why is there a exp(beta*dcoh) above the denominator as well?
pAdopt = exp(beta * dcoh) / (1 + exp(beta * dcoh));
% shouldn't this only be 1?
pAdopt = 1 / (1 + exp(beta * dcoh));

if(rand < pAdopt)
    bi(kx)=arg;
end  

oNew = bi*V;

deltaO = oNew-oOld;

end