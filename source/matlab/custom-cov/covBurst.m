function K = covBurst(cov, hyp, x, z, i)

% covBurst - OMG another kernel.
%
% Copyright (c) by James Robert Lloyd 2013-08-16.
%
% See also COVFUNCTIONS.M.

% if numel(cov)==0, error('We require at least one summand.'), end
% for ii = 1:numel(cov)                        % iterate over covariance functions
%   f = cov(ii); if iscell(f{:}), f = f{:}; end   % expand cell array if necessary
%   j(ii) = cellstr(feval(f{:}));                          % collect number hypers
% end
% 
% if nargin<3                                        % report number of parameters
%   K = char(j(1)); for ii=2:length(cov), K = [K, '+', char(j(ii))]; end, return
% end
% if nargin<4, z = []; end                                   % make sure, z exists
% [n,D] = size(x);
% 
% v = [];               % v vector indicates to which covariance parameters belong
% for ii = 1:length(cov), v = [v repmat(ii, 1, eval(char(j(ii))))]; end
% 
% if nargin<5                                                        % covariances
%   K = 0; if nargin==3, z = []; end                                 % set default
%   for ii = 1:length(cov)                      % iteration over summand functions
%     f = cov(ii); if iscell(f{:}), f = f{:}; end % expand cell array if necessary
%     K = K + feval(f{:}, hyp(v==ii), x, z);              % accumulate covariances
%   end
% else                                                               % derivatives
%   if i<=length(v)
%     vi = v(i);                                       % which covariance function
%     j = sum(v(1:i)==vi);                    % which parameter in that covariance
%     f  = cov(vi);
%     if iscell(f{:}), f = f{:}; end         % dereference cell array if necessary
%     K = feval(f{:}, hyp(v==vi), x, z, j);                   % compute derivative
%   else
%     error('Unknown hyperparameter')
%   end
  if nargin<3 
      K = feval(cov{:});
  end
  ax = repmat(4 * (1 ./ (1 + exp(-x))) ./ (1 + exp(x)), 1, length(x));
  az = ax';
  K = ax .* feval(cov{:}, hyp, x) .* az;
end
