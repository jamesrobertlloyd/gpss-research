function [ alpha ] = solve_sdp( A, B )
%SOLVE_SDP Basic binary search for really simple semi-definite program
%   Detailed explanation goes here
    lower = 0;
    upper = 1;
    for dummy = 1:20
        alpha = (lower + upper) / 2;
        failed = false;
        try
            K = A - alpha*B;
            chol(K + (max(max(K))-min(min(K)))*1e-6*eye(size(K)));
        catch
            failed = true;
        end
        if failed
            upper = alpha;
        else
            lower = alpha;
        end
    end
    alpha = lower;
end

