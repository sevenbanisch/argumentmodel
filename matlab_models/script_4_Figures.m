%% Use this script for reproduction of the figures in the main paper

computation = 0; % toggle whether to run the systematic simulations (time consuming)
figurenumber = 4;

%% Figure 1: runtime comparison

if(figurenumber == 1)

    % parameter set (Figure: runtime)
    samplesB = 60;
    samplesIN = 1000;

    samplesIN = 100;
    N = 10;
    
    %N = 100;    
    M = 4;
    pN = 0;
    
    bMin = 2;
    bMax = 8;
    
    basesteps = 1000;
    steps = M * basesteps * 16; % for N = 1000, 16000 is ok. 64000 steps is toooo much!
    
    T = M * 100; % just to not store for all times
    OBSperiod = 25; % take temporal average over last 25 % of the runs

    if(computation)
        
        rmSIMDATA = zeros(samplesIN,samplesB+1,steps/T);
        rmOBS01 = zeros(samplesIN,samplesB+1);
        rmBETA = zeros(1,samplesB+1);
        
        emSIMDATA = zeros(samplesIN,samplesB+1,steps/T);
        emOBS01 = zeros(samplesIN,samplesB+1);
        emBETA = zeros(1,samplesB+1);

        parfor sB = 0:samplesB
            beta = bMin + (bMax-bMin)*sB/samplesB;
            rmBETA(1,sB+1) = beta;
            emBETA(1,sB+1) = beta;
            for sIN = 1:samplesIN
                
                A_t = NormalizedArgumentModel(steps,N,M,beta,pN,0);
        
                emSIMDATA(sIN,sB+1,:) = var(A_t(:,T:T:steps),1);
                emOBS01(sIN,sB+1) = mean(var(A_t(:,steps-steps/OBSperiod:steps),1));        
                
                A_t = ReducedArgumentModel(steps,N,M,beta,pN,0);
        
                % normalization by N
                rmSIMDATA(sIN,sB+1,:) = var(A_t(:,T:T:steps),1);
                rmOBS01(sIN,sB+1) = mean(var(A_t(:,steps-steps/OBSperiod:steps),1));
        
                fprintf('.')
            end
    
        end
    else
        load('./SIMDATA/runtimefigure/N100.mat')
    end % if computation


    varthreshh=0.5;

    figure
    subplot(1,2,1)
    alpha_value = 0.1;
    hold on
    
     CR = emSIMDATA(:,:,steps/T) > varthreshh; % 16000 steps
     plot(emBETA,mean(CR))
     
     CR = emSIMDATA(:,:,steps/(2*T)) > varthreshh; % 1200 steps
     plot(emBETA,mean(CR))
     
     CR = emSIMDATA(:,:,steps/(4*T)) > varthreshh; % 4000 steps
     plot(emBETA,mean(CR))
    
     CR = emSIMDATA(:,:,steps/(8*T)) > varthreshh; % 4000 steps
     plot(emBETA,mean(CR))
      
     CR = emSIMDATA(:,:,steps/(16*T)) > varthreshh; % 4000 steps
     plot(emBETA,mean(CR))
    
     CR = emSIMDATA(:,:,steps/(32*T)) > varthreshh; % 4000 steps
     plot(emBETA,mean(CR))
    
     CR = emSIMDATA(:,:,2) > varthreshh; % 4000 steps
     plot(emBETA,mean(CR))
    
        xlabel('\beta', 'FontSize', 20);
        ylabel('polarization probability', 'FontSize', 20);
        set(gca,'FontSize',20)
        xlim([2.5,6])
        grid on;
    
     hold off
     % ---------
    
    subplot(1,2,2)
    hold on
    
        CR = rmSIMDATA(:,:,steps/T) > varthreshh; % 16000 steps
        plot(rmBETA,mean(CR))
        
        CR = rmSIMDATA(:,:,steps/(2*T)) > varthreshh; % 1200 steps
        plot(rmBETA,mean(CR))
        % 
        CR = rmSIMDATA(:,:,steps/(4*T)) > varthreshh; % 1200 steps
        plot(rmBETA,mean(CR))
        % 
        CR = rmSIMDATA(:,:,steps/(8*T)) > varthreshh; % 4000 steps
        plot(rmBETA,mean(CR))
        
        CR = rmSIMDATA(:,:,steps/(16*T)) > varthreshh; % 1200 steps
        plot(rmBETA,mean(CR))
        % 
        CR = rmSIMDATA(:,:,steps/(32*T)) > varthreshh; % 4000 steps
        plot(rmBETA,mean(CR))
        
        CR = rmSIMDATA(:,:,2) > varthreshh; % 800 steps
        plot(rmBETA,mean(CR))
        
            xlabel('\beta', 'FontSize', 20);
            ylabel('polarization probability', 'FontSize', 20);
            set(gca,'FontSize',20)
            xlim([2.5,6])
            grid on;
        
        hold off


end % Figure 1


%% 2. Bifurcation and Scatter plot of variance in (quasi-)stationary state
% 
if(figurenumber == 2)

    load('./SIMDATA/theory/theoryP05B0to8.mat')

    figure
    hold on
    alpha_value = 0.3;
    %myblue = [0.2 0.5450 0.7330];
    %myred = [0.7330 0.2 0.5450];

    myblue = [0.6 0.60 0.60];
    myred = [0.4 0.4 0.40];
    

    scatter(rmBETA,rmOBS01,100,'*','MarkerEdgeColor',myblue,...
                       'MarkerFaceColor',myblue,...
                       'MarkerFaceAlpha', alpha_value,...
                       'MarkerEdgeAlpha', alpha_value,...
                       'LineWidth',1.5);
    scatter(emBETA,emOBS01,'+','MarkerEdgeColor',myred,...
                       'MarkerFaceColor',myred,...
                       'MarkerFaceAlpha', alpha_value,...
                       'MarkerEdgeAlpha', alpha_value,...
                       'LineWidth',1.5);

    plot(theoryBETA,theoryP05B0to8);
    xlabel('\beta', 'FontSize', 20);
    ylabel('variance', 'FontSize', 20);
    xlim([0,6])
    set(gca,'FontSize',20)
    grid on;

    hold off
end % if 2.Scatter plot


% 
% figure
% 
% plot(theoryBETA,theoryP05B0to8(:,1:3),'black');
% 
% hold on
% 
% alpha_value = 0.1;
% marker_size = 20;
% scatter(emBETA,emOBS01,'o','MarkerEdgeColor',[0.2 0.5450 0.7330],...
%                        'MarkerFaceColor',[0.2 0.5450 0.7330],...
%                        'MarkerFaceAlpha', alpha_value,...
%                        'MarkerEdgeAlpha', alpha_value);
%     xlabel('\beta', 'FontSize', 20);
%     ylabel('variance', 'FontSize', 20);
%     set(gca,'FontSize',20)
%     grid on;
% 
% 
% alpha_value = 0.2;
% marker_size = 14;
% color = [0.7330 0.1450 0.2];
% scatter(rmBETA,rmOBS01,marker_size,'o','MarkerEdgeColor',color,...
%                        'MarkerFaceColor',color,...
%                        'MarkerFaceAlpha', alpha_value,...
%                        'MarkerEdgeAlpha', alpha_value);
%     xlabel('\beta', 'FontSize', 20);
%     ylabel('variance', 'FontSize', 20);
%     set(gca,'FontSize',20)
%     grid on;
%     ylim([-0.02,1.02])
% 
% hold off




