function order_by_mae_reduction(X, y, mean_family, mean_params, ...
           complete_covfunc, complete_hypers, decomp_list, ...
           decomp_hypers, lik_family, lik_params, figname)
% Orders additive components of kernel by reduction in MAE

%%%%%%%%%%%
% WARNING %
%%%%%%%%%%%
% - Ignores lik_family - assumes provides a noise parameter

if isempty(lik_params)
    log_noise = -inf;
else
    log_noise = lik_params(0);
end
noise_var = exp(2*log_noise);

% Convert to double in case python saved as integers
X = double(X);
y = double(y);

% Subtract the mean function
y = y - feval(mean_family{:}, mean_params, X);

%%%% TODO - does this ever want to be a parameter?
folds = 10;

X_train = cell(folds,1);
y_train = cell(folds,1);
X_valid = cell(folds,1);
y_valid = cell(folds,1);

%%%% TODO - Check me for overlap
for fold = 1:folds
    range = max(1,floor(length(X)*(fold-1)/folds)):floor(length(X)*(fold)/folds);
    X_valid{fold} = X(range);
    y_valid{fold} = y(range);
    range = [1:min(length(X),floor(length(X)*(fold-1)/folds)-1),...
            max(1,floor(length(X)*(fold)/folds)+1):length(X)];
    X_train{fold} = X(range);
    y_train{fold} = y(range);
end

idx = [];

cum_kernel = cell(0);
cum_hyp = [];

MAEs = zeros(numel(decomp_list), 1);
MAE_reductions = zeros(numel(decomp_list), 1);
MAV_data = mean(abs(y));
previous_MAE = MAV_data;

for i = 1:numel(decomp_list)
    best_MAE = Inf;
    for j = 1:numel(decomp_list)
        if ~sum(j == idx)
            kernels = cum_kernel;
            kernels{i} = decomp_list{j};
            hyps = cum_hyp;
            hyps = [hyps, decomp_hypers{j}]; %#ok<AGROW>
            hyp.mean = [];
            hyp.cov = hyps;
            cur_cov = {@covSum, kernels};
            e = NaN(length(X_train), 1);
            for fold = 1:length(X_train)
              K = feval(complete_covfunc{:}, complete_hypers, X_train{fold}) + ...
                  noise_var*eye(length(y_train{fold}));
              Ks = feval(cur_cov{:}, hyp.cov, X_train{fold}, X_valid{fold});

              ymu = Ks' * (K \ y_train{fold});

              e(fold) = mean(abs(y_valid{fold} - ymu));
            end
            
            my_MAE = mean(e);
            if my_MAE < best_MAE
                best_j  = j;
                best_MAE = my_MAE;
            end
        end
    end
    MAEs(i) = best_MAE;
    MAE_reductions(i) = (1 - best_MAE / previous_MAE)*100;
    previous_MAE = best_MAE;
    idx = [idx, best_j]; %#ok<AGROW>
    cum_kernel{i} = decomp_list{best_j};
    cum_hyp = [cum_hyp, decomp_hypers{best_j}]; %#ok<AGROW>
end

% Save data to file

save(sprintf('%s_mae_data.mat', figname), 'idx', 'MAEs', 'MAV_data', ...
     'MAE_reductions');
 
end


