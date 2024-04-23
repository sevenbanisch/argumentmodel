function [oA,oB,colors] = testoninitialconditions(N,p,beta,mode,visualize)

%N = 100;
T = 1000;
samples = 10000;
%p = 1/2;

if(mode == 1)
    ICs = 2*(rand(samples,N)-0.5);  % uniformly in [-1,1]
else
    M = 4;
    V = zeros(2*M,1);
    V(1:M) = -1/M;
    V(M+1:2*M) = 1/M;
    
    
    ICs=zeros(samples,N);
    for i = 1:samples
        args = randi(2,N,2*M)-1;
        A = args*V;     % binomial 
        ICs(i,:) = A;  
    end
end

ICA = ICs<0;
ICB = ICs>0;

% rMat = rand(samples,N);
% ICA =  rMat < 0.5;
% ICB = rMat > 0.5;

oA = mean(ICs.*ICA,2);

oB = mean(ICs.*ICB,2);

% convergence test
colors = zeros(samples,1)+0.05;
if(1)
    offset = 1/10000;
    for s = 1:samples
        oAt = oA(s);
        oBt = oB(s);
        for t = 1:T
            dA = (1-p)*IRF(oAt,oAt,beta) + p * IRF(oAt,oBt,beta);
            dB = (1-p)*IRF(oBt,oBt,beta) + p * IRF(oBt,oAt,beta);
            oAt = oAt + dA + offset*(rand - 0.5);
            oBt = oBt + dB + offset*(rand - 0.5);
        end
        colors(s)=var([oAt,oBt])/2;

    end
end

if(visualize)
%figure
scatter(oA,oB,5, colors, 'filled') % 50 is the marker size
    colormap('jet'); % Set the colormap (optional)
    colorbar; % Show colorbar (optional)
    xlabel('opinion group A', 'FontSize', 20);
    ylabel('opinion group B', 'FontSize', 20);
    %xlim([-1,1]);
    %ylim([-1,1]);
    
    set(gca,'FontSize',12)
    grid on;

    figure
    hist(colors)
end

end