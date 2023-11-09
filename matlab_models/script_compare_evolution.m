
beta = 3.2;
N = 500;
M = 5;
pN = 0;
steps = 5000;

A_t = ArgumentModel(steps,N,beta,pN,0);
A_tn = NormalizedArgumentModel(steps,N,M,beta,pN,0);
A_tr = ReducedArgumentModel(steps,N,M,beta,pN,0);



if(1)

    figure(1);

    
    colormap bone;
    cmap = colormap;
    cmap = flipud(cmap);
    for i = 1:20
        k = (i-1)/20;
        map(i,:) = [1 1-k 1-k];
    end
    colormap(map);



    %yticks(1:9);
    %yticklabels([4,3,2,1,0,-1,-2,-3,-4]*c);

    %plot(1:steps,mean(reshape(A_t(:,:),[],steps)),'Linewidth', 3);
    %plot(1:steps,9-2*(std(reshape(A_t(:,:),[],steps))),'Linewidth', 3);
    


    subplot(2,1,1)
    imagesc(flipud(hist(A_tn(:,:),2*M+1)));
    hold on
    plot(1:steps,(2*M+1)-(2*M+1)*std(reshape(A_tn(:,:),[],steps)),'Linewidth', 3);
    hold off
    xlabel('time', 'FontSize', 20);
    ylabel('opinion', 'FontSize', 20);
    set(gca,'FontSize',12)
    grid on;
    yticks(1:2*M+1);
    yticklabels([1:-1/M:-1]);
    %yticklabels([4,3,2,1,0,-1,-2,-3,-4]);

    subplot(2,1,2)
    imagesc(flipud(hist(A_tr(:,:),2*M+1)));
    hold on
    plot(1:steps,(2*M+1)-(2*M+1)*std(reshape(A_tr(:,:),[],steps)),'Linewidth', 3);
    hold off
    xlabel('time', 'FontSize', 20);
    ylabel('opinion', 'FontSize', 20);
    set(gca,'FontSize',12)
    grid on;
    yticks(1:2*M+1);
    yticklabels([1:-1/M:-1]);

    % figure(2);
    % subplot(3,1,1)
    % hist(A_t(:,steps))
    % subplot(3,1,2)
    % hist(A_tn(:,steps))
    % subplot(3,1,3)
    % hist(A_tr(:,steps))

end




%% For all three
if(0)

    figure(1);
    subplot(3,1,1)
    
    colormap bone;
    cmap = colormap;
    cmap = flipud(cmap);
    for i = 1:20
        k = (i-1)/20;
        map(i,:) = [1 1-k 1-k];
    end
    colormap(map);

    imagesc(flipud(hist(A_t(:,:),9)));
    hold on;

    %yticks(1:9);
    %yticklabels([4,3,2,1,0,-1,-2,-3,-4]*c);

    %plot(1:steps,mean(reshape(A_t(:,:),[],steps)),'Linewidth', 3);
    %plot(1:steps,9-2*(std(reshape(A_t(:,:),[],steps))),'Linewidth', 3);
    
    xlabel('time', 'FontSize', 20);
    ylabel('opinion', 'FontSize', 20);
    set(gca,'FontSize',20)
    grid on;
    hold off;

    subplot(3,1,2)
    imagesc(flipud(hist(A_tn(:,:),9)));
    xlabel('time', 'FontSize', 20);
    ylabel('opinion', 'FontSize', 20);
    set(gca,'FontSize',20)
    grid on;

    subplot(3,1,3)
    imagesc(flipud(hist(A_tr(:,:),9)));
    xlabel('time', 'FontSize', 20);
    ylabel('opinion', 'FontSize', 20);
    set(gca,'FontSize',20)
    grid on;

    figure(2);
    subplot(3,1,1)
    hist(A_t(:,steps))
    subplot(3,1,2)
    hist(A_tn(:,steps))
    subplot(3,1,3)
    hist(A_tr(:,steps))

end