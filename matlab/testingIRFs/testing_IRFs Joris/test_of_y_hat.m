

if 0
    % set parameters
    K = 8;
    M = K/2;
    precision = 16;

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
            
            y = y_hat(bi, bj, M, precision);
    	    oi = bi*V;
            oj = bj*V;
            
            k = (row_i-1)*2^K + row_j;
            % get oi
            comb_result(k, 1) = round(oi,3);
            % get oj
            comb_result(k, 2) = round(oj,3);
            % get error in prediction
            comb_result(k, 3) = y;
        end
    end
end

% create two heatmaps, first between oi, oj, and the mean y_hat, second
% between oi, oj and the variance of y_hat
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
heatmap(uni_oi, uni_oj, heatmap_data(:,:,1), "Title","Mean of $$\hat{y}$$",'Interpreter','Latex', "XLabel","$o_i$","YLabel","$o_{j}$",'Interpreter','Latex');
figure
heatmap(uni_oi, uni_oj, heatmap_data(:,:,2), "Title","Variance of $$\hat{y}$$",'Interpreter','Latex', "XLabel","$o_i$","YLabel","$o_j$",'Interpreter','Latex');
