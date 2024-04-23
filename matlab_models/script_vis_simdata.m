
%% just plotting

%% variance after T steps
if(1)

    figure
    time = 10;  % 
    %time = steps/T
    subplot(2,1,1)
    SIMDATA = SIMDATAOM500;
    scatter(BETA,SIMDATA(:,:,time),'*','MarkerEdgeColor',[0.2 0.5450 0.7330])
    hold on
    plot(BETA,mean(SIMDATA(:,:,time)))
    plot(BETA,bftheory)
    hold off
    xlabel('\beta', 'FontSize', 16);
    ylabel('variance', 'FontSize', 16);
    
    set(gca,'FontSize',12)
    grid on;

    subplot(2,1,2)
    SIMDATA = SIMDATARM500;
    scatter(BETA,SIMDATA(:,:,time),'*','MarkerEdgeColor',[0.2 0.5450 0.7330])
    hold on
    plot(BETA,mean(SIMDATA(:,:,time)))
    plot(BETA,bftheory)
    hold off
    xlabel('\beta', 'FontSize', 16);
    ylabel('variance', 'FontSize', 16);
    
    set(gca,'FontSize',12)
    grid on;

end




if(0)

%% convergence after T steps
    if(0)
        convO = SIMDATAOM500 <= 0.01;
        convR = SIMDATARM500 <= 0.01;
    else
        convO = SIMDATAOM1000 <= 0.01;
        convR = SIMDATARM1000 <= 0.01;
        BETA = BETA1000;

    end
    figure
    time = 2000;
    subplot(2,1,1)
    plot(BETA,mean(convO(:,:,time)))
    % plot(BETA,mean(convO(:,:,500)))
    % hold on
    % plot(BETA,mean(convO(:,:,400)))
    % plot(BETA,mean(convO(:,:,300)))
    % plot(BETA,mean(convO(:,:,200)))
    % plot(BETA,mean(convO(:,:,10)))
    % hold off
    
    subplot(2,1,2)
    plot(BETA,mean(convR(:,:,time)))
    % plot(BETA,mean(convR(:,:,500)))
    % hold on
    % plot(BETA,mean(convR(:,:,400)))
    % plot(BETA,mean(convR(:,:,300)))
    % plot(BETA,mean(convR(:,:,200)))
    % plot(BETA,mean(convR(:,:,100)))
    % hold off
end

if(0)
%% convergence times
    if(0)
        nconvO = SIMDATAOM1000 >= 0.01;
        nconvR = SIMDATARM1000 >= 0.01;
        BETA = BETA1000;
    else
        nconvO = SIMDATAOM100 >= 0.01;
        nconvR = SIMDATARM100 >= 0.01;
        BETA = BETA500;
    end
    figure
    scatter(BETA,sum(nconvR(:,:,:),3),'*','MarkerEdgeColor',[0.2 0.5450 0.7330])
    hold on
    scatter(BETA,sum(nconvO(:,:,:),3),'*','MarkerEdgeColor',[0.7330 0.2 0.5450])
    plot(BETA,mean(sum(nconvR(:,:,:),3)))
    plot(BETA,mean(sum(nconvO(:,:,:),3)))
    hold off
    %set(gca,'Yscale','log')

end

