function K = covFourier(hyp, x, z, i)

% Stationary covariance function for a smooth periodic function, with period p:
% Now with constant removed!
% And now scaled to make it have sf2 marginal variance
% covFourier is just a code name until the code is refactored
%
% k(x,y) = see fourier.nb
%
% where the hyperparameters are:
%
% hyp = [ log(ell)
%         log(p)
%         log(sqrt(sf2)) ]
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2011-01-05.
% Modified by James Robert Lloyd 6 Sept 2013
%
% See also COVFUNCTIONS.M.

%%%% WARNING - this is numerically unstable as ell -> infty
%%%%         - the limit is the cosine kernel - can this limit be used
%%%%         - somehow to prevent numerical instability?        

if nargin<2, K = '3'; return; end                  % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

% The limit of this function is cosine - switch to it when close for
% numerical stability
if hyp(1) > 4
    if nargin < 4  
        K = covCos(hyp(2:3), x, z);
    else
        if i == 1
            % Returning zeros means the parameter will get stuck here
            % Not too undesirable
            if xeqz
                K = zeros(size(x,1));
            else
                K = zeros(size(x,1),size(z,1));
            end
        else
            K = covCos(hyp(2:3), x, z, i-1);
        end
    end
else
    % Compute covariance exactly
    n = size(x,1);
    ell = exp(hyp(1));
    p   = exp(hyp(2));
    sf2 = exp(2*hyp(3));

    % precompute distances
    if dg                                                               % vector kxx
      K = zeros(size(x,1),1);
    else
      if xeqz                                                 % symmetric matrix Kxx
        K = sqrt(sq_dist(x'));
      else                                                   % cross covariances Kxz
        K = sqrt(sq_dist(x',z'));
      end
    end

    K = 2*pi*K/p;
    if nargin<4                                                        % covariances
        K = exp(cos(K)/(ell^2));
        Bessel0 = besseli(0,ell^-2);
        K = K - Bessel0;
        K = K / (exp(ell^-2)-Bessel0);
        K = sf2*K;
    else
      if i==1
        sin_K2 = sin(K/2).^2;
        cos_K = cos(K);
        exp_cos_K2 = exp(2*(cos(K/2)/(ell)).^2);
        exp_cos_K = exp(cos_K/(ell^2));
        exp_ell_2 = exp(ell^-2);
        Bessel0 = besseli(0,ell^-2);
        Bessel1 = besseli(1,ell^-2);
        K = -2*((-exp_ell_2+exp_cos_K)*Bessel1+Bessel0*(exp_ell_2-exp_cos_K.*cos_K)-2*exp_cos_K2.*sin_K2);
        K = K / ((ell*(exp_ell_2-Bessel0))^2);
        K = sf2*K;
      elseif i==2
        K = exp(cos(K)/(ell^2)).*sin(K).*K;
        Bessel0 = besseli(0,ell^-2);
        K = K / (ell^2*(exp(ell^-2)-Bessel0));
        K = sf2*K;
      elseif i==3
        K = exp(cos(K)/(ell^2));
        Bessel0 = besseli(0,ell^-2);
        K = K - Bessel0;
        K = K / (exp(ell^-2)-Bessel0);
        K = 2*sf2*K;
      else
        error('Unknown hyperparameter')
      end
    end
end