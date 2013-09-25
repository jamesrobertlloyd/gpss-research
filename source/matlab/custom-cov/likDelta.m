function [varargout] = likDelta(hyp, y, mu, s2, inf)

% This is a hack of likGauss - I will keep the redundant sn2 for the moment
%
% likGauss - Gaussian likelihood function for regression. The expression for the 
% likelihood is 
%   likGauss(t) = exp(-(t-y)^2/2*sn^2) / sqrt(2*pi*sn^2),
% where y is the mean and sn is the standard deviation.
%
% The hyperparameters are:
%
% hyp = [  log(sn)  ]
%
% Several modes are provided, for computing likelihoods, derivatives and moments
% respectively, see likelihoods.m for the details. In general, care is taken
% to avoid numerical issues when the arguments are extreme.
%
% See also likFunctions.m.
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2011-02-18

if nargin<2, varargout = {'0'}; return; end   % report number of hyperparameters

% sn2 = exp(2*hyp);
sn2 = 0;

if nargin<5                              % prediction mode if inf is not present
  if numel(y)==0,  y = zeros(size(mu)); end
  s2zero = 1; if nargin>3, if norm(s2)>0, s2zero = 0; end, end         % s2==0 ?
  if s2zero                                                    % log probability
%     lp = -(y-mu).^2./sn2/2-log(2*pi*sn2)/2; s2 = 0;
    error('Delta likelihood needs non-zero noise upstream')
  else
    lp = likDelta(hyp, y, mu, s2, 'infEP');                         % prediction
  end
  ymu = {}; ys2 = {};
  if nargout>1
    ymu = mu;                                                   % first y moment
    if nargout>2
      ys2 = s2 + sn2;                                          % second y moment
    end
  end
  varargout = {lp,ymu,ys2};
else
  switch inf 
  case 'infEP'
    if nargin<6                                             % no derivative mode
      lZ = -(y-mu).^2./(sn2+s2)/2 - log(2*pi*(sn2+s2))/2;    % log part function
      dlZ = {}; d2lZ = {};
      if nargout>1
        dlZ  = (y-mu)./(sn2+s2);                    % 1st derivative w.r.t. mean
        if nargout>2
          d2lZ = -1./(sn2+s2);                      % 2nd derivative w.r.t. mean
        end
      end
      varargout = {lZ,dlZ,d2lZ};
    else                                                       % derivative mode
%       dlZhyp = ((y-mu).^2./(sn2+s2)-1) ./ (1+s2./sn2);   % deriv. w.r.t. hyp.lik
        % This does not appeart to be called ever
      dlZhyp = 0;
      varargout = {dlZhyp};
    end
  end
end
