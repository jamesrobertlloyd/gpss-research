function K = covCosUnit(hyp, x, z, i)

% Stationary covariance function for a sinusoid with period p:
%
% k(x,y) = cos(pi * (x - x') / p)
%
% where the hyperparameters are:
%
% hyp = [ log(p) ]
%
% Copyright (c) by James Robert Lloyd, 2013-08-05.

%%%% Warning - assumes 1d x and z

if nargin<2, K = '1'; return; end                  % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

n = size(x,1);
p = exp(hyp(1));

% precompute distances
if dg                                                               % vector kxx
  K = zeros(size(x,1),1);
else
  if xeqz                                                 % symmetric matrix Kxx
    K = repmat(x, 1, n) - repmat(x', n, 1);
  else                                                   % cross covariances Kxz
    K = repmat(x, 1, size(z, 1)) - repmat(z', n, 1);
  end
end

K = 2*pi*K/p;
if nargin<4                                                        % covariances
    K = cos(K);
else                                                               % derivatives
  if i==1
    K = sin(K).*K;
  else
    error('Unknown hyperparameter')
  end
end