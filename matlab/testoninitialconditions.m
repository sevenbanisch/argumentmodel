function [oA,oB,colors] = testoninitialconditions(N,M,T,Samples,pInter,beta,mode,visualize)

%N = 100;
%T = 1000;
%samples = 1000;
%p = 1/2;

if(mode == 1)
    ICs = 2*(rand(Samples,N)-0.5);  % uniformly in [-1,1]
else
    %M = 4;
    V = zeros(2*M,1);
    V(1:M) = -1/M;
    V(M+1:2*M) = 1/M;
    
    
    ICs=zeros(Samples,N);
    for i = 1:Samples
        args = randi(2,N,2*M)-1;
        A = args*V;     % binomial 
        ICs(i,:) = A;  
    end
end

% 1. first trail, splitting by 0
 ICA = ICs<0;
 ICB = ICs>0;

% 2. second trail splitting by mean.

% trsh = mean(ICs,2);
% 
% ICA = ICs<trsh;
% ICB = ICs>trsh;

% 3. third trail splitting half/half

% ICA = ICs * false;
% ICB = ICs * false;
% for s = 1:samples
%     [osorted,ix] = sort(ICs(s,:));
%     trsh = osorted(N/2);
%     ICA(s,ix(1:N/2)) = true;
%     ICB(s,ix(N/2+1:N)) = true;
%     %sum(ICA(s,:))
% end


% Compute mean opinion (wrong version)
%oA = mean(ICs.*ICA,2);
%oB = mean(ICs.*ICB,2);

% Compute means using sum and count
%oA = sum(ICs .* ICA, 2) ./ sum(ICA, 2); % Mean of positive values
%oB = sum(ICs .* ICB, 2) ./ sum(ICB, 2); % Mean of negative values

for i = 1:Samples
    oA(i) = mean(ICs(i, ICs(i, :) < 0));
    oB(i) = mean(ICs(i, ICs(i, :) > 0));
end

% rMat = rand(samples,N);
% ICA =  rMat < 0.5;
% ICB = rMat > 0.5;



% convergence test
colors = zeros(Samples,1)+0.05;
if(1)
    %offset = 1/10000;
    for s = 1:Samples
        oAt = oA(s);
        oBt = oB(s);
        for t = 1:T
            dA = (1-pInter)*IRF(oAt,oAt,beta,M) + pInter * IRF(oAt,oBt,beta,M);
            dB = (1-pInter)*IRF(oBt,oBt,beta,M) + pInter * IRF(oBt,oAt,beta,M);
            oAt = oAt + dA;% + offset*(rand - 0.5);
            oBt = oBt + dB;% + offset*(rand - 0.5);
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