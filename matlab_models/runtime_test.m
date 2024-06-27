beta = 3.8;
N = 100;
M = 16;
pN = 0;
steps = 4000;
reps = 600;

datetime("now", Format = "HH:mm:ss")
for rep = 1:reps
    A_tn = NormalizedArgumentModel(steps,N,M,beta,pN,0);
    A_tr = ReducedArgumentModel(steps,N,M,beta,pN,0);
end
datetime("now", Format = "HH:mm:ss")