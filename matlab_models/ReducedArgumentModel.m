
%%%%%%%%%%%%%%%%%
% Baseline Model
% 
% steps: number of time steps
% N: number of agents
% beta: strength of biased processing
% pN: noise level
% visualize: toggle on and off the visualization of the model outcome


function [A_t] = ReducedArgumentModel(steps,N,M,beta,pN,visualize)

% 1. Model Parameters

I = 1;
V = zeros(2*M,1);
V(1:M) = -1/M;
V(M+1:2*M) = 1/M;

% 2. Initial Conditions

args = randi(2,N,2*M)-1;
A = args*V;

% 3. Observables

A_t = zeros(N,steps);
A_t(:,1)=A;

% 4. Simulation Loop 

for step = 1:steps
    
    aList = randperm(N);

    for pair = 1:2:N
        sx = aList(pair);
        rx = aList(pair+1);
        %kx = randi(M);
        %arg = args(sx,kx);
        
        % Random arguments from external sources.
        %if(rand < pN)
        %     arg = randi(2)-1;
        %end    

        %ix=1;
        %dcoh = (2*arg-1) * sign(V(kx,ix)) * A(rx,ix);
        %pAdopt = exp(beta * dcoh) / (1 + exp(beta * dcoh));
        %if(rand < pAdopt)
        %    args(rx,kx)=arg;
        %end    

        dA = (A(sx,1)-A(rx,1)+(1-A(sx,1)*A(rx,1))*tanh(beta*A(rx,1)/2))/(4*M);
        A(rx,1) = A(rx,1) + dA;

    end
    
    %A = args*V;
    A_t(:,step)=A;

end     % steps

if(visualize)

    figure;
    
    colormap bone;
    cmap = colormap;
    cmap = flipud(cmap);
    for i = 1:20
        k = (i-1)/20;
        map(i,:) = [1 1-k 1-k];
    end
    colormap(map);

    imagesc(flipud(hist(A_t(:,:),9)));
    %hold on;
    %yticks(1:9);
    %yticklabels([4,3,2,1,0,-1,-2,-3,-4]*c);

    %plot(1:steps,5-mean(reshape(A_t(:,:),[],steps)),'Linewidth', 3);
    %plot(1:steps,9-2*(std(reshape(A_t(:,:),[],steps))),'Linewidth', 3);
    
    xlabel('time', 'FontSize', 20);
    ylabel('opinion', 'FontSize', 20);
    set(gca,'FontSize',20)
    grid on;
    hold off;

end

end

