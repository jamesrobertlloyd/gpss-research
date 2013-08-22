function K = covIMT3(hyp, x, z, i)

% Integrated MT3
%
% k(x,y) = xxx
%
% where the hyperparameters are:
%
% hyp = [ log(l)
%         location ]
%
% Copyright (c) by James Robert Lloyd, 2013-08-05.

%%%% Warning - assumes 1d x and z

if nargin<2, K = '2'; return; end                  % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

n = size(x,1);
l = exp(hyp(1));
location = hyp(2);

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

if nargin<4                                                        % covariances
    K = ((a > 0) & (b >= a)) .* ...
        (1/3) .* l .* exp(-sqrt(3).*(a+b)/l) .* (...
        -3.*l.*(-1+exp(sqrt(3).*a/l)).*(exp(sqrt(3).*a/l)+exp(sqrt(3).*b/l)) + ...
        sqrt(3).*(-b.*exp(sqrt(3).*a/l).*(-1+exp(sqrt(3).*a/l)) + ...
        a.*(exp(2.*sqrt(3).*a/l)+exp(2.*sqrt(3).*b/l)+...
        4.*exp(2.*sqrt(3).*(a+b)/l))));
    K = K + K' - diag(diag(K));
else                                                               % derivatives
%   if i == 1
%     K = -r/6 * a .* a .* (a - 3 * b) .* ((a > 0) & (b >= a)) + ...
%         +r/6 * a .* a .* (a - 3 * b) .* ((a < 0) & (b <= a)) + ...
%         -r/6 * b .* b .* (b - 3 * a) .* ((b > 0) & (a > b)) + ...
%         +r/6 * b .* b .* (b - 3 * a) .* ((b < 0) & (a < b));
%   elseif i == 2
%     K = -r * a .* b .* ((a > 0) & (b >= a)) + ...
%         +r * a .* b .* ((a < 0) & (b <= a)) + ...
%         -r * b .* a .* ((b > 0) & (a > b)) + ...
%         +r * b .* a .* ((b < 0) & (a < b));
%   else
%     error('Unknown hyperparameter')
%   end
  error('Derivatives nto implemented')
end
end