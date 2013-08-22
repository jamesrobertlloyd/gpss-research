function K = covBurst(cov, hyp, x, z, i)

% covBurst - OMG another kernel.
%
% Copyright (c) by James Robert Lloyd 2013-08-16.

if ~numel(cov)==1, error('Burst uses one covariance.'), end
for ii = 1:numel(cov)                        % iterate over covariance functions
  f = cov(ii); if iscell(f{:}), f = f{:}; end   % expand cell array if necessary
  j(ii) = cellstr(feval(f{:}));                          % collect number hypers
end

if nargin<3                                        % report number of parameters
  K = ['3' '+' char(j(1))]; for ii=2:length(cov), K = [K, '+', char(j(ii))]; end, return
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
width = exp(hyp(3));
ax = (1 ./ (1 + exp(-(x-(location-0.5*width))*steepness))) .* ...
     (1 ./ (1 + exp(+(x-(location+0.5*width))*steepness)));
if ~dg
    ax = repmat(ax, 1, length(z));
end
if ~dg
    az = (1 ./ (1 + exp(-(z-(location-0.5*width))*steepness))) .* ...
         (1 ./ (1 + exp(+(z-(location+0.5*width))*steepness)));
    az = repmat(az', length(x), 1);
else
    az = ax;
end

if nargin<5                                                        % covariances
  K = 0; if nargin==3, z = x; end                                 % set default
  for ii = 1:length(cov)                              % iteration over functions
    f = cov(ii); if iscell(f{:}), f = f{:}; end % expand cell array if necessary
    if ii == 1
        K = K + ax .* feval(f{:}, hyp([false false false (v==ii)]), x, z) .* az;
    end
  end
else                                                               % derivatives
  if i==1
    xl = repmat(x - location, 1, length(z));
    zl = repmat((z - location)', length(x), 1);
    dx = steepness * exp(steepness*(xl+0.5*width)) .* (exp(steepness*2*xl) - 1) .* ...
         (1 + exp(steepness*(xl+0.5*width))).^(-2) .* ...
         (1 + exp(steepness*(xl-0.5*width))).^(-2);
    dx(isnan(dx)) = 0; % infty * 0 = 0 for this function
    dx(isinf(dx)) = 0; % infty * eps = 0 for this function
    dz = steepness * exp(steepness*(zl+0.5*width)) .* (exp(steepness*2*zl) - 1) .* ...
         (1 + exp(steepness*(zl+0.5*width))).^(-2) .* ...
         (1 + exp(steepness*(zl-0.5*width))).^(-2);
    dz(isnan(dz)) = 0; % infty * 0 = 0 for this function
    dz(isinf(dz)) = 0; % infty * eps = 0 for this function
    K = 0;
    for ii = 1:length(cov)                              % iteration over functions
        f = cov(ii); if iscell(f{:}), f = f{:}; end % expand cell array if necessary
        if ii == 1
            K = K + dx .* feval(f{:}, hyp([false false false (v==ii)]), x, z) .* az;
            K = K + ax .* feval(f{:}, hyp([false false false (v==ii)]), x, z) .* dz;
        end
    end
  elseif i==2
    xl = repmat(x - location, 1, length(z));
    zl = repmat((z - location)', length(x), 1);
    dx = steepness * (xl + 0.5*width) .* ...
         exp(-steepness*(xl+0.5*width)) .* ...
         (1 + exp(-steepness*(xl+0.5*width))).^(-2) .* ...
         (1 + exp(steepness*(xl-0.5*width))).^(-1) ...
         - ...
         steepness * (xl - 0.5*width) .* ...
         exp(steepness*(xl-0.5*width)) .* ...
         (1 + exp(steepness*(xl-0.5*width))).^(-2) .* ...
         (1 + exp(-steepness*(xl+0.5*width))).^(-1);
    dx(isnan(dx)) = 0; % infty * 0 = 0 for this function
    dx(isinf(dx)) = 0; % infty * eps = 0 for this function
    dz = steepness * (zl + 0.5*width) .* ...
         exp(-steepness*(zl+0.5*width)) .* ...
         (1 + exp(-steepness*(zl+0.5*width))).^(-2) .* ...
         (1 + exp(steepness*(zl-0.5*width))).^(-1) ...
         - ...
         steepness * (zl - 0.5*width) .* ...
         exp(steepness*(zl-0.5*width)) .* ...
         (1 + exp(steepness*(zl-0.5*width))).^(-2) .* ...
         (1 + exp(-steepness*(zl+0.5*width))).^(-1);
    dz(isnan(dz)) = 0; % infty * 0 = 0 for this function
    dz(isinf(dz)) = 0; % infty * eps = 0 for this function
    %dx = steepness * exp(steepness*(xl+0.5*width)) .* ...
    %     (1 + exp(steepness*(xl+0.5*width))).^(-2) .* ...
    %     (1 + exp(steepness*(xl-0.5*width))).^(-2) .* ...
    %     (width * (0.5 + exp(steepness*xl-0.5*width)) + ...
    %      xl + ...
    %      (-xl+0.5*width).*(exp(steepness*2*xl)));
    %dz = steepness * exp(steepness*(zl+0.5*width)) .* ...
    %     (1 + exp(steepness*(zl+0.5*width))).^(-2) .* ...
    %     (1 + exp(steepness*(zl-0.5*width))).^(-2) .* ...
    %     (width * (0.5 + exp(steepness*zl-0.5*width)) + ...
    %      zl + ...
    %      (-zl+0.5*width).*(exp(steepness*2*zl)));
    K = 0;
    for ii = 1:length(cov)                              % iteration over functions
        f = cov(ii); if iscell(f{:}), f = f{:}; end % expand cell array if necessary
        if ii == 1
            K = K + dx .* feval(f{:}, hyp([false false false (v==ii)]), x, z) .* az;
            K = K + ax .* feval(f{:}, hyp([false false false (v==ii)]), x, z) .* dz;
        end
    end
  elseif i==3
    xl = repmat(x - location, 1, length(z));
    zl = repmat((z - location)', length(x), 1);
    dx = steepness * width * exp(steepness*(xl+0.5*width)) .* (0.5 + 0.5*exp(steepness*2*xl) + exp(steepness*(xl-0.5*width)) ) .* ...
         (1 + exp(steepness*(xl+0.5*width))).^(-2) .* ...
         (1 + exp(steepness*(xl-0.5*width))).^(-2);
    dx(isnan(dx)) = 0; % infty * 0 = 0 for this function
    dx(isinf(dx)) = 0; % infty * eps = 0 for this function
    dz = steepness * width * exp(steepness*(zl+0.5*width)) .* (0.5 + 0.5*exp(steepness*2*zl) + exp(steepness*(zl-0.5*width)) ) .* ...
         (1 + exp(steepness*(zl+0.5*width))).^(-2) .* ...
         (1 + exp(steepness*(zl-0.5*width))).^(-2);
    dz(isnan(dz)) = 0; % infty * 0 = 0 for this function
    dz(isinf(dz)) = 0; % infty * eps = 0 for this function
    K = 0;
    for ii = 1:length(cov)                              % iteration over functions
        f = cov(ii); if iscell(f{:}), f = f{:}; end % expand cell array if necessary
        if ii == 1
            K = K + dx .* feval(f{:}, hyp([false false false (v==ii)]), x, z) .* az;
            K = K + ax .* feval(f{:}, hyp([false false false (v==ii)]), x, z) .* dz;
        end
    end
  elseif i<=length(v)+3
    i = i - 3;
    vi = v(i);                                       % which covariance function
    j = sum(v(1:i)==vi);                    % which parameter in that covariance
    f  = cov(vi);
    if iscell(f{:}), f = f{:}; end         % dereference cell array if necessary
    K = feval(f{:}, hyp([false false false (v==vi)]), x, z, j);                   % compute derivative
    if vi == 1
        K = ax .* K .* az;
    end
  else
    error('Unknown hyperparameter')
  end
end
