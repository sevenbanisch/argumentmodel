function [y_dach] = y_hat(bi, bj, M, precision)
%Calculation of y_hat as described in appendix.tex

sigma_p = (vpa(sum( bi(1:M).*bj(1:M) )/M - sum( bi(1:M) )/M * sum( bj(1:M) )/M ,precision));

sigma_c = (vpa(sum( bi(M+1:2*M).*bj(M+1:2*M) )/M - sum( bi(M+1:2*M) )/M * sum( bj(M+1:2*M) )/M ,precision));

sums = (1/M) * (vpa( sum( bi ) + sum( bj ) , precision));

cross_sums = (1/M^2) * (vpa( sum(bi(1:M))     * sum(bj(1:M)) + ...
                             sum(bi(M+1:2*M)) * sum(bj(M+1:2*M)) + ...
                             sum(bi(1:M))     * sum(bj(M+1:2*M)) + ...
                             sum(bi(M+1:2*M)) * sum(bj(1:M)) , precision));

y_dach = 1 - sums + cross_sums + 2 * ( sigma_p + sigma_c );
end

