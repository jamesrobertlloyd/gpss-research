function [ new_K ] = non_singular( K )
%NON_SINGULAR Summary of this function goes here
%   Detailed explanation goes here
    success = false;
    i = 9;
    while ~success
        success = true;
        try
            new_K = K + (10^-i)*eye(size(K))*max(max(K));
            chol(new_K);
        catch
            success = false;
            i = i - 1;
        end
    end
end

