
N = 100;
M = 4;

obs_V=zeros(1,100);

for i = 1:100
V = zeros(2*M,1);
V(1:M) = -1/M;
V(M+1:2*M) = 1/M;

% 2. Initial Conditions

args = randi(2,N,2*M)-1;
A = args*V;

obs_V(1,i) = var(A);

end

mean(obs_V)