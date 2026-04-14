function [irf] = IRF(oi,oj,beta,M)
     
    irf = (oj-oi+(1-oi*oj)*tanh(beta*oi/2))/(4*M);

end