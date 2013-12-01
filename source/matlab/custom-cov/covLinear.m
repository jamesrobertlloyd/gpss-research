function K = covLinear(hyp, x, z, i)

% Assumes 1d
%
% hyp = [ log(sf)
%         shift      ]

% Based on covLINard by
% Carl Edward Rasmussen and Hannes Nickisch, 2010-09-10.
% Adapted by James Robert Lloyd, 2013
%
% N.B. Reparametrised to have a scale factor rather than inverse
%
% See also COVFUNCTIONS.M.

if nargin<2, K = '2'; return; end                  % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

[n,~] = size(x);
sf = exp(hyp(1));
shift = hyp(2);
x = (x-repmat(shift',n,1))*sf;

% precompute inner products
if dg                                                               % vector kxx
  K = sum(x.*x,2);
else
  if xeqz                                                 % symmetric matrix Kxx
    K = x*x';
  else                                                   % cross covariances Kxz
    [nz,~] = size(z); 
    z = (z-repmat(shift',nz,1))*sf;
    K = x*z';
  end
end

if nargin>3                                                        % derivatives
  if i == 1
    if dg
      K = 2*x(:,i).*x(:,i);
    else
      if xeqz
        K = 2*x(:,i)*x(:,i)';
      else
        K = 2*x(:,i)*z(:,i)';
      end
    end
  elseif i == 2 
    %%%% Not tested with D > 1
    %%%% Also really ugly!
    if dg
      K = -2 * x * sf;
    else
      if xeqz
        K = -sf * (repmat(x, 1, n) + repmat(x', n, 1));
      else
        K = -sf * (repmat(x, 1, n) + repmat(z', n, 1));
      end
    end
  else
    error('Unknown hyperparameter')
  end
end