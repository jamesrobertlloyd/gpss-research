function K = covIBM(hyp, x, z, i)

% Integrated Brownian motion
%
% k(x,y) = xxx
%
% where the hyperparameters are:
%
% hyp = [ log(r)
%         location
%         log(offset_sf)
%         log(grad_sf)]
%
% Copyright (c) by James Robert Lloyd, 2013-08-05.

%%%% Warning - assumes 1d x and z

if nargin<2, K = '4'; return; end                  % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

n = size(x,1);
r = exp(hyp(1));
location = hyp(2);
offset_scale = exp(2*hyp(3));
lin_scale = exp(2*hyp(4));

if dg
    a = x - location;
    b = x - location;
else
    if xeqz 
        a = repmat(x - location, 1, n);
        b = a';
    else
        a = repmat(x - location, 1, length(z));
        b = repmat((z - location)', length(x), 1);
    end
end

%zero_pattern = ((b >= a) .* (a > 0) | (b <= a) .* (a < 0));

if nargin<4                                                        % covariances
    %K = abs(-r/6 * a .* a .* (a - 3 * b));
    K = -r/6 * a .* a .* (a - 3 * b) .* ((a > 0) & (b >= a)) + ...
        +r/6 * a .* a .* (a - 3 * b) .* ((a < 0) & (b <= a)) + ...
        -r/6 * b .* b .* (b - 3 * a) .* ((b > 0) & (a > b)) + ...
        +r/6 * b .* b .* (b - 3 * a) .* ((b < 0) & (a < b));
    G = a .* b * lin_scale;
    H = offset_scale * ones(size(K));
else                                                               % derivatives
  if i == 1
    %K = abs(-r/6 * a .* a .* (a - 3 * b));
    K = -r/6 * a .* a .* (a - 3 * b) .* ((a > 0) & (b >= a)) + ...
        +r/6 * a .* a .* (a - 3 * b) .* ((a < 0) & (b <= a)) + ...
        -r/6 * b .* b .* (b - 3 * a) .* ((b > 0) & (a > b)) + ...
        +r/6 * b .* b .* (b - 3 * a) .* ((b < 0) & (a < b));
    G = zeros(size(K));
    H = zeros(size(K));
  elseif i == 2
    %K = -abs(r * a .* b) .* ((a>0)*2 - 1);
    K = -r * a .* b .* ((a > 0) & (b >= a)) + ...
        +r * a .* b .* ((a < 0) & (b <= a)) + ...
        -r * b .* a .* ((b > 0) & (a > b)) + ...
        +r * b .* a .* ((b < 0) & (a < b));
    G = -(a + b) * lin_scale;
    H = zeros(size(K));
  elseif i == 3
    H = 2 * offset_scale * ones(size(a));
    K = zeros(size(H));
    G = zeros(size(H));
  elseif i == 4
    G = 2 * a .* b * lin_scale;
    K = zeros(size(G));
    H = zeros(size(K));
  else
    error('Unknown hyperparameter')
  end
end
%K = K .* zero_pattern;
%if (~dg) & (xeqz)
%    K = K + K' - diag(diag(K));
%end
K = K + G + H;
end