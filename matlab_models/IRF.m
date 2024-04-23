function [irf] = IRF(oi,oj,beta)
     
    M=4;
    irf = (oj-oi+(1-oi*oj)*tanh(beta*oi/2))/(4*M);

end