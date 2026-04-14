
beta = 3.6;
N = 500;
M = 16;
pN = 0;
mode = 0; % random arguments
steps = 4000;

%A_t = ArgumentModel(steps,N,beta,pN,0);
A_tn = NormalizedArgumentModel(steps,N,M,beta,pN,0);
A_tr = ReducedArgumentModel(steps,N,M,beta,pN,mode,0);

GS = 0;
AinitN = A_tn(:,1);
[A_MFN_t] = MFmodel2Dgs(steps,M,beta,AinitN,GS);
AinitR = A_tr(:,1);
[A_MFR_t] = MFmodel2Dgs(steps,M,beta,AinitR,GS);

if(1)

    figure(1);

    
    colormap sky;
    cmap = colormap;
    cmap = flipud(cmap);
    for i = 1:20
        k = (i-1)/20;
        map(i,:) = [1 1-k 1-k];
    end
    colormap(map);

    darkBlue = [0 0 0.5];    % Darker blue
    darkGreen = [0 0.5 0];   % Darker green

    %yticks(1:9);
    %yticklabels([4,3,2,1,0,-1,-2,-3,-4]*c);

    %plot(1:steps,mean(reshape(A_t(:,:),[],steps)),'Linewidth', 3);
    %plot(1:steps,9-2*(std(reshape(A_t(:,:),[],steps))),'Linewidth', 3);
    


    subplot(2,1,1)
    %imagesc(flipud(hist(A_tn(:,:),2*M+1)));
    alpha = 0.15;
    
    ix0 = AinitN == 0;
    AinitN(ix0) = AinitN(ix0)+(rand(sum(ix0),1)-0.5).*0.001;
    ixA = AinitN < 0;
    ixB = AinitN > 0;
    plot(A_tn(ixA,:)','Color', [0, 0, 0.7, alpha],'LineWidth', 0.25);
    hold on
    plot(A_tn(ixB,:)','Color', [0, 0.7, 0, alpha],'LineWidth', 0.25);

    %plot(1:steps,(2*M+1)-(2*M+1)*std(reshape(A_tn(:,:),[],steps)),'Linewidth', 3);
    %plot(1:steps,(2*M+1)-(2*M+1)*std(reshape(A_tn(:,:),[],steps))+0.5,'Linewidth', 3);
    O_vis = (-A_MFN_t * M + (M+1));
    O_vis = A_MFN_t;
    plot(O_vis(1,:),'--','Color',darkBlue,'LineWidth', 2)
    plot(O_vis(2,:),'--','Color',darkGreen,'LineWidth', 2)
    hold off
    xlabel('time', 'FontSize', 20);
    ylabel('opinion', 'FontSize', 20);
    
    set(gca,'FontSize',12)
    grid on;
    %yticks(1:2*M+1);
    %yticklabels([1:-1/M:-1]);
    %yticklabels([4,3,2,1,0,-1,-2,-3,-4]);

    subplot(2,1,2)
    resolution = 1; %% doen not work correctly
    binEdges = -1:1/M/resolution:1;


    %imagesc(flipud(hist(A_tr(:,:),binEdges)));
    %imagesc(flipud(hist(A_tr(:,:),2*M+1)));
    ix0 = AinitR == 0;
    AinitR(ix0) = AinitR(ix0)+(rand(sum(ix0),1)-0.5).*0.001;
    ixA = AinitR < 0;
    ixB = AinitR > 0;
    plot(A_tr(ixA,:)','Color', [0, 0, 0.7, alpha],'LineWidth', 0.25);
    hold on
    plot(A_tr(ixB,:)','Color', [0, 0.7, 0, alpha],'LineWidth', 0.25);
    %plot(1:steps,(2*M+1)-(2*M+1)*std(reshape(A_tr(:,:),[],steps))+0.5,'Linewidth', 3);
    O_vis = (-A_MFR_t * resolution * M + (M+1)); %+ 12; for resulution = 4
    O_vis = A_MFR_t;
    plot(O_vis(1,:),'--','Color',darkBlue,'LineWidth', 2)
    plot(O_vis(2,:),'--','Color',darkGreen,'LineWidth', 2)
    hold off
    xlabel('time', 'FontSize', 20);
    ylabel('opinion', 'FontSize', 20);
    %ylim([1,2*M+1])
    set(gca,'FontSize',12)
    grid on;
    %yticks(1:resolution:resolution*2*M+1);
    %yticklabels([1:-1/M:-1]);

    figure(2);
    edges = linspace(-1, 1, 2*M+1);
    subplot(2,1,1)
    histogram(A_tn(:,steps),edges,'FaceColor', 'r')
    hold on
    xline(A_MFN_t(1,steps), '--','Color',darkBlue, 'LineWidth', 3);
    xline(A_MFN_t(2,steps), '--','Color',darkGreen,'LineWidth', 3);
    hold off
    xlabel('final opinions');
    %xticks([-1:1/M:1]);
    ylabel('frequency');
    xlim([-1,1])
    set(gca,'FontSize',16)
    grid on;

    subplot(2,1,2)
    edges = linspace(-1, 1, 2*resolution * M+1);
    histogram(A_tr(:,steps),edges,'FaceColor', 'r')
    hold on
    xline(A_MFR_t(1,steps), '--','Color',darkBlue, 'LineWidth', 3);
    xline(A_MFR_t(2,steps), '--','Color',darkGreen, 'LineWidth', 3);
    hold off
    xlabel('final opinions');
    ylabel('frequency');
    xlim([-1,1])
    set(gca,'FontSize',16)
    grid on;

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