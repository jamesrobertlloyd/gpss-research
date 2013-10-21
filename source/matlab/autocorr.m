function varargout = autocorr(Series , nLags , Q , nSTDs)
[rows , columns]  =  size(Series);
if (rows ~= 1) & (columns ~= 1)
   error('GARCH:autocorr:NonVectorInput' , ' Input ''Series'' must be a vector.');
end
rowSeries   =  size(Series,1) == 1;
Series      =  Series(:);       % Ensure a column vector
n           =  length(Series);  % Sample size.
defaultLags =  20;              % BJR recommend about 20 lags for ACFs.
if (nargin >= 2) & ~isempty(nLags)
  if prod(size(nLags)) > 1
     error('GARCH:autocorr:NonScalarLags' , ' Number of lags ''nLags'' must be a scalar.');
  end
  if (round(nLags) ~= nLags) | (nLags <= 0)
     error('GARCH:autocorr:NonPositiveInteger' , ' Number of lags ''nLags'' must be a positive integer.');
  end
  if nLags > (n - 1)
     error('GARCH:autocorr:LagsTooLarge' , ' Number of lags ''nLags'' must not exceed ''Series'' length - 1.');
  end
else
  nLags  =  min(defaultLags , n - 1);
end
if (nargin >= 3) & ~isempty(Q)
  if prod(size(Q)) > 1
     error('GARCH:autocorr:NonScalarQ' , ' Number of lags ''Q'' must be a scalar.');
  end
  if (round(Q) ~= Q) | (Q < 0)
     error('GARCH:autocorr:NegativeInteger' , ' Number of lags ''Q'' must be a non-negative integer.');
  end
  if Q >= nLags
     error('GARCH:autocorr:QTooLarge' , ' ''Q'' must be less than ''nLags''.');
  end
else
  Q  =  0;     % Default is 0 (Gaussian white noise hypothisis).
end
if (nargin >= 4) & ~isempty(nSTDs)
  if prod(size(nSTDs)) > 1
     error('GARCH:autocorr:NonScalarSTDs' , ' Number of standard deviations ''nSTDs'' must be a scalar.');
  end
  if nSTDs < 0
     error('GARCH:autocorr:NegativeSTDs' , ' Number of standard deviations ''nSTDs'' must be non-negative.');
  end
else
  nSTDs =  2;     % Default is 2 standard errors (95% condfidence interval).
end
nFFT =  2^(nextpow2(length(Series)) + 1);
F    =  fft(Series-mean(Series) , nFFT);
F    =  F .* conj(F);
ACF  =  ifft(F);
ACF  =  ACF(1:(nLags + 1));         % Retain non-negative lags.
ACF  =  ACF ./ ACF(1);     % Normalize.
ACF  =  real(ACF);
sigmaQ  =  sqrt((1 + 2*(ACF(2:Q+1)'*ACF(2:Q+1)))/n);
bounds  =  sigmaQ * [nSTDs ; -nSTDs];
Lags    =  [0:nLags]';
if nargout == 0                     % Make plot if requested.
  lineHandles  =  stem(Lags , ACF , 'filled' , 'r-o');
  set   (lineHandles(1) , 'MarkerSize' , 4)
  grid  ('on')
  xlabel('Lag')
  ylabel('Sample Autocorrelation')
  title ('Sample Autocorrelation Function (ACF)')
  hold  ('on')
  plot([Q+0.5 Q+0.5 ; nLags nLags] , [bounds([1 1]) bounds([2 2])] , '-b');
  plot([0 nLags] , [0 0] , '-k');
  hold('off')
  a  =  axis;
  axis([a(1:3) 1]);
else
  if rowSeries
     ACF     =  ACF.';
     Lags    =  Lags.';
     bounds  =  bounds.';
  end
  varargout  =  {ACF , Lags , bounds};
end