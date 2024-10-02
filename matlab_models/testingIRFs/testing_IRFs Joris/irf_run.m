
if 1
    % set parameters
    K = 8;
    M = K/2;
    beta = 2;
    
    % Create normalized connection matrix
    V = zeros(K,1);
    V(1:M) = 1/M;
    V(M+1:2*M) = -1/M;
    
    % create all possible belief strings
    bi_s = (dec2bin(0:2^K-1)' - '0').';
    bj_s = (dec2bin(0:2^K-1)' - '0').';
    
    % create data structure that saves the results in a long format
    comb_result = zeros(size(bi_s,1)^2, 3);
    
    % iterate over all possible belief string combinations
    for row_i = 1:size(bi_s,1)
        for row_j = 1:size(bj_s,1)
            bi = bi_s(row_i,:);
            bj = bj_s(row_j,:);
            
            % returns error in the expected opinion change compared to
            % original argument model
            [error, oi, oj] = irf_belief_comparison(bi, bj, V, beta, 1000);
            
            k = (row_i-1)*2^K + row_j;
            % get oi
            comb_result(k, 1) = round(oi,3);
            % get oj
            comb_result(k, 2) = round(oj,3);
            % get error in prediction
            comb_result(k, 3) = error;
        end
    end
end
   
if 0
    % create a 3d plot between oi, oj, and the error. shows the mean error and
    % its confidence interval
    figure
    hold on
    % iterate over all possible opinion pairs
    for oi = unique(comb_result(:,1)).'
        for oj = unique(comb_result(:,2)).'
               
            % get all indexes corresponding to the opinion pair
            idx = comb_result(:,1)==oi & comb_result(:,2)==oj;
    
            z_error = comb_result(idx,3);
            mean_z = mean(z_error);
            lb_confidence = mean_z - 1.96 * (std(z_error)/size(z_error,1));
            ub_confidence = mean_z + 1.96 * (std(z_error)/size(z_error,1));
    
            plot3([oi;oi], [oj;oj], [lb_confidence;ub_confidence], '-b');
            scatter3(oi, oj, mean(z_error), 100,'red','square', 'filled');
            
        end
    end
    xlabel("oi");
    ylabel("oj");
    zlabel("mean error in expected opinion change");
    title("Mean error and confidence intervals in expected opinion change");
    hold off
end
% create two heatmaps, first between oi, oj, and the mean error, second
% between oi, oj and the variance of the error
figure

% get the unique possible opinions 
uni_oi = unique(comb_result(:,1));
uni_oj = unique(comb_result(:,2));
% structure to save the data in a heatmap ready format
heatmap_data = zeros(size(uni_oi, 1), size(uni_oj, 1), 2);

% iterate over all possible opinion pairs and save the mean/variance in the
% corresponding matrix index
for oi = uni_oi.'
    for oj = uni_oj.'

        % get all indexes corresponding to the opinion pair
        idx = comb_result(:,1)==oi & comb_result(:,2)==oj;

        z_error = comb_result(idx,3);
        mean_z = mean(z_error);
        variance = var(z_error);
        
        heatmap_data(cast((oi*M+M+1),"uint8"), cast((oj*M+M+1),"uint8"), 1) = mean_z;
        heatmap_data(cast((oi*M+M+1),"uint8"), cast((oj*M+M+1),"uint8"), 2) = variance;

    end
end

% create the heatmaps
heatmap(uni_oi, uni_oj, heatmap_data(:,:,1), "Title","Mean of Error", "XLabel","oi","YLabel","oj");
figure
heatmap(uni_oi, uni_oj, heatmap_data(:,:,2), "Title","Variance of the Error", "XLabel","oi","YLabel","oj");


% this creates a 2d scatterplot between the difference of oi and oj with
% the error of the expected opinion change. The mean error is added too
figure
hold on

% find unique delta(oi,oj) values to later use them as bins
x_vals = unique(round(abs(comb_result(:,1)-comb_result(:,2)),3))-0.01;
x_vals(end + 1) = max(x_vals + 0.2);
% get the index in which bin each datapoint is
groups = discretize(abs(comb_result(:,1)-comb_result(:,2)), x_vals);

% iterate over each group
for group = unique(groups.')

    x = x_vals(group)+0.01;
    y = comb_result(groups == group, 3);
    
    scatter(x, y, 'blue'); 
    scatter(x,mean(y),100,'red','square', 'filled');

end

xlabel("delta(oi,oj)");
ylabel("Error in expected opinion change");
title("Error and mean error in expected opinion change");
hold off







