function K = covZero(hyp, x, z, i)

% covariance function for the zero function:
%
% Copyright (c) by James Robert Lloyd, 2013-08-18.

if nargin<2, K = '0'; return; end                  % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

n = size(x,1);

if dg                                                               % vector kxx
  K = zeros(n,1);
else
  if xeqz                                                 % symmetric matrix Kxx
    K = zeros(n);
  else                                                   % cross covariances Kxz
    K = zeros(n,size(z,1));
  end
end

if nargin>3      
  error('This covariance is constant')
end
