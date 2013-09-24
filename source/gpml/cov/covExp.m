function K = covCos(hyp, x, z, i)

% Simple exponential function - for windowing other functions
% Initially trying an over-parametrised version
%
% k(x,y) = sf2 * k(x)k(y)
% k(x)   = exp(rate*(x-location))
%
% where the hyperparameters are:
%
% hyp = [ rate
%         location
%         log(sqrt(sf2)) ]
%
% Copyright (c) by James Robert Lloyd, 2013-09-24.

%%%% Warning - assumes 1d x and z

if nargin<2, K = '3'; return; end                  % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

n = size(x,1);
rate = hyp(1);
location = hyp(2);
sf2 = exp(2*hyp(3));

if dg
    a = exp(rate*(x - location));
    b = a;
else
    if xeqz 
        a = repmat(exp(rate*(x - location)), 1, n);
        b = a';
    else
        a = repmat(exp(rate*(x - location)), 1, length(z));
        b = repmat(exp(rate*(z - location)'), length(x), 1);
    end
end

if nargin<4                                                        % covariances
    K = sf2*a.*b;
else                                                               % derivatives
  if i==1
    K = sf2*a.*b.*(repmat((x - location), 1, length(z)) + ...
                   repmat((z - location)', length(z), 1));
  elseif i==2
    K = -2*rate*sf2*a.*b;
  elseif i==3
    K = 2*sf2*a.*b;
  else
    error('Unknown hyperparameter')
  end
end
