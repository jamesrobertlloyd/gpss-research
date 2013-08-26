function K = covChangePointTanh(cov, hyp, x, z, i)

% 1d change point kernel using tanh
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

tx = tanh((x-location)*steepness);
ax = 0.5 + 0.5 * tx;
if ~dg
    ax = repmat(ax, 1, length(z));
end
if ~dg
    tz = tanh((z-location)*steepness);
    az = 0.5 + 0.5 * tz;
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
    dx = -0.5*(1-tx.^2)*steepness;
    dz = -0.5*(1-tz.^2)*steepness;
    dx = repmat(dx, 1, length(z));
    dz = repmat(dz', length(x), 1);
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
    dx = -0.5*(1-tx.^2).*(x-location).*steepness;
    dz = -0.5*(1-tz.^2).*(z-location).*steepness;
    dx = repmat(dx, 1, length(z));
    dz = repmat(dz', length(x), 1);
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
