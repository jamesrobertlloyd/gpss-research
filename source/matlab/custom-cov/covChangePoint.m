function K = covChangePoint(cov, hyp, x, z, i)

% 1d change point kernel
%
% k(x^p,x^q) = xxx
%
% hyp = [ location
%         log(steepness)
%         hyp_1
%         hyp_2 ]
%
% Copyright (c) by James Robert Lloyd, 2013-08-16.

if ~numel(cov)==2, error('Change point uses two covariances.'), end
for ii = 1:numel(cov)                        % iterate over covariance functions
  f = cov(ii); if iscell(f{:}), f = f{:}; end   % expand cell array if necessary
  j(ii) = cellstr(feval(f{:}));                          % collect number hypers
end

if nargin<3                                        % report number of parameters
  K = ['2' '+' char(j(1))]; for ii=2:length(cov), K = [K, '+', char(j(ii))]; end, return
end
if nargin<4, z = []; end                                   % make sure, z exists
[n,D] = size(x);
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode
if xeqz
    z = x;
end

v = [];               % v vector indicates to which covariance parameters belong
for ii = 1:length(cov), v = [v repmat(ii, 1, eval(char(j(ii))))]; end

location = hyp(1);
steepness = exp(hyp(2));

ax = 1 ./ (1 + exp(-(x-location)*steepness));
if ~dg
    ax = repmat(ax, 1, length(z));
end
if ~dg
    az = 1 ./ (1 + exp(-(z-location)*steepness));
    az = repmat(az', length(x), 1);
else
    az = ax;
end

ax = 1 - ax; % Switching the order of base kernels to match intuition
az = 1 - az;

if nargin<5                                                        % covariances
  K = 0; if nargin==3, z = x; end                                 % set default
  for ii = 1:length(cov)                              % iteration over functions
    f = cov(ii); if iscell(f{:}), f = f{:}; end % expand cell array if necessary
    if ii == 1
        K = K + ax .* feval(f{:}, hyp([false false (v==ii)]), x, z) .* az;
    else
        K = K + (1-ax) .* feval(f{:}, hyp([false false (v==ii)]), x, z) .* (1-az);
    end
  end
else                                                               % derivatives
  if i==1
    dx = -steepness * repmat(exp(+(x-location)*steepness), 1, length(z)) .* ...
         (1 + repmat(exp(+(x-location)*steepness), 1, length(z))).^(-2);
    dx(isnan(dx)) = 0; % infty * 0 = 0 for this function
    dx(isinf(dx)) = 0; % infty * eps = 0 for this function
    dz = -steepness * repmat(exp(+(z-location)*steepness)', length(x), 1) .* ...
         (1 + repmat(exp(+(z-location)*steepness)', length(x), 1)).^(-2);
    dz(isnan(dz)) = 0; % infty * 0 = 0 for this function
    dz(isinf(dz)) = 0; % infty * eps = 0 for this function
    dx = -dx; % Switching the order of base kernels to match intuition 
    dz = -dz;
    K = 0;
    for ii = 1:length(cov)                              % iteration over functions
        f = cov(ii); if iscell(f{:}), f = f{:}; end % expand cell array if necessary
        if ii == 1
            K = K + dx .* feval(f{:}, hyp([false false (v==ii)]), x, z) .* az;
            K = K + ax .* feval(f{:}, hyp([false false (v==ii)]), x, z) .* dz;
        else
            K = K + -dx .* feval(f{:}, hyp([false false (v==ii)]), x, z) .* (1-az);
            K = K + (1-ax) .* feval(f{:}, hyp([false false (v==ii)]), x, z) .* (-dz);
        end
    end
  elseif i==2
    dx = steepness * repmat(exp(+(x-location)*steepness).*(x-location), 1, length(z)) .* ...
         (1 + repmat(exp(+(x-location)*steepness), 1, length(z))).^(-2);
    dx(isnan(dx)) = 0; % infty * 0 = 0 for this function
    dx(isinf(dx)) = 0; % infty * eps = 0 for this function
    dz = steepness * repmat((exp(+(z-location)*steepness).*(z-location))', length(x), 1) .* ...
         (1 + repmat(exp(+(z-location)*steepness)', length(x), 1)).^(-2);
    dz(isnan(dz)) = 0; % infty * 0 = 0 for this function
    dz(isinf(dz)) = 0; % infty * eps = 0 for this function
    dx = -dx; % Switching the order of base kernels to match intuition 
    dz = -dz;
    K = 0;
    for ii = 1:length(cov)                              % iteration over functions
        f = cov(ii); if iscell(f{:}), f = f{:}; end % expand cell array if necessary
        if ii == 1
            K = K + dx .* feval(f{:}, hyp([false false (v==ii)]), x, z) .* az;
            K = K + ax .* feval(f{:}, hyp([false false (v==ii)]), x, z) .* dz;
        else
            K = K + -dx .* feval(f{:}, hyp([false false (v==ii)]), x, z) .* (1-az);
            K = K + (1-ax) .* feval(f{:}, hyp([false false (v==ii)]), x, z) .* (-dz);
        end
    end
  elseif i<=length(v)+2
    i = i - 2;
    vi = v(i);                                       % which covariance function
    j = sum(v(1:i)==vi);                    % which parameter in that covariance
    f  = cov(vi);
    if iscell(f{:}), f = f{:}; end         % dereference cell array if necessary
    K = feval(f{:}, hyp([false false (v==vi)]), x, z, j);                   % compute derivative
    if vi == 1
        K = ax .* K .* az;
    elseif vi ==2
        K = (1-ax) .* K .* (1-az);
    end
  else
    error('Unknown hyperparameter')
  end
end
