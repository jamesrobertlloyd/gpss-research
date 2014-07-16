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
            if i == -1
                % Things have gone very wrong
                % Return identity to prevent errors
                new_K = eye(size(K));
                success = true;
                warning('Could not make matrix non-singular - aborting');
            end
        end
    end
end

