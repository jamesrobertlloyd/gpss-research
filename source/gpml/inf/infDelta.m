function [post nlZ dnlZ] = infDelta(hyp, mean, cov, lik, x, y)

% Exact inference for a GP with Gaussian likelihood. Compute a parametrization
% of the posterior, the negative log marginal likelihood and its derivatives
% w.r.t. the hyperparameters. See also "help infMethods".
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2011-02-18
%
% Modified by James Robert Lloyd 25 September 2013 to allow the likDelta
% likelihood function (likGauss with zero noise)
%
% See also INFMETHODS.M.

likstr = lik; if ~ischar(lik), likstr = func2str(lik); end 
if ~strcmp(likstr,'likDelta') && ~delta_mode % NOTE: no explicit call to likDelta
  error('Delta inference only possible with Delta likelihood');
end
 
[n, D] = size(x);
K = feval(cov{:}, hyp.cov, x);                      % evaluate covariance matrix
m = feval(mean{:}, hyp.mean, x);                          % evaluate mean vector

sn2 = 0;
L = chol(K);               % Cholesky factor of covariance
alpha = solve_chol(L,y-m);

post.alpha = alpha;                            % return the posterior parameters
% post.sW = ones(n,1)/sqrt(sn2);                  % sqrt of noise precision vector
post.sW = Inf;
post.L  = L;                                        % L = chol(eye(n)+sW*sW'.*K)

if nargout>1                               % do we want the marginal likelihood?
  nlZ = (y-m)'*alpha/2 + sum(log(diag(L))) + n*log(2*pi)/2;  % -log marg lik
  if nargout>2                                         % do we want derivatives?
    dnlZ = hyp;                                 % allocate space for derivatives
    K_inv = solve_chol(L,eye(n));
%     Q = solve_chol(L,eye(n))/sn2 - alpha*alpha';    % precompute for convenience
    for i = 1:numel(hyp.cov)
%       dnlZ.cov(i) = sum(sum(Q.*feval(cov{:}, hyp.cov, x, [], i)))/2;
      dK = feval(cov{:}, hyp.cov, x, [], i);
      dnlZ.cov(i) = 0.5*(-alpha'*dK*alpha + trace(K_inv * dK));
    end
    dnlZ.lik = [];
    for i = 1:numel(hyp.mean)
      dnlZ.mean(i) = -feval(mean{:}, hyp.mean, x, i)'*alpha;
    end
  end
end
