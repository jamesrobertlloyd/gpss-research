function set_pagewidth( fraction, pagewidth )
% Sets the pagewidth of a figure such that the whole thing, when cropped,
% is exactly as wide as fraction * pagewidth, in points.
%
% David Duvenaud

if nargin < 1; fraction = 1; end
if nargin < 2; pagewidth = 412.5649; end % The pagewidth of a CUED thesis, in points.

set(gca, 'Units', 'points');
position = get(gca, 'Position');          % Width of the figure
tightinset = get(gca, 'TightInset');      % Width of the surrounding text
total_width = position(3) + tightinset(3) + tightinset(1);
scale_factor = (pagewidth*fraction)/total_width;
set(gca, 'Position', [position(1:2), position(3:4).*scale_factor]);
