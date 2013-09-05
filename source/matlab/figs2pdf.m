% Turns all figures in a directory into pdfs.
%
% Should be run from source/matlab/
%
% David Duvenaud
% Feb 2013


%topdir = '../../figures/decomposition';
%latexdir = '../../latex/figures/decomposition';
%topdir = '../../figures/decomposition/temp';
%latexdir = '../../latex/figures/decomposition/temp';
%topdir = '../../figures/decomposition/no-rq-test';
%latexdir = '../../latex/figures/decomposition/no-rq-test';
% topdir = '../../analyses/time-series/figures';
% latexdir = '../../analyses/time-series/figures';
% topdir = '../../analyses/2013-08-14-no-rq/figures';
% latexdir = '../../analyses/2013-08-14-no-rq/figures';
% topdir = '../../analyses/2013-08-16-IBM/figures';
% latexdir = '../../analyses/2013-08-16-IBM/figures';
% topdir = '../../analyses/2013-08-16-spectral/figures';
% latexdir = '../../analyses/2013-08-16-spectral/figures';
% topdir = '../../analyses/2013-08-16-CP/figures';
% latexdir = '../../analyses/2013-08-16-CP/figures';
% topdir = '../../analyses/2013-08-19-time-series/figures';
% latexdir = '../../analyses/2013-08-19-time-series/figures';
% topdir = '../../analyses/2013-08-20-time-series/figures';
% latexdir = '../../analyses/2013-08-20-time-series/figures';
% topdir = '../../analyses/2013-08-21-test1/figures';
% latexdir = '../../analyses/2013-08-21-test1/figures';
% topdir = '../../analyses/2013-08-22-time-series/figures';
% latexdir = '../../analyses/2013-08-22-time-series/figures';
% topdir = '../../analyses/2013-08-27-time-series/figures';
% latexdir = '../../analyses/2013-08-27-time-series/figures';
% topdir = '../../analyses/2013-08-28-time-series/figures';
% latexdir = '../../analyses/2013-08-28-time-series/figures';
% topdir = '../../analyses/2013-08-29-time-series/figures';
% latexdir = '../../analyses/2013-08-29-time-series/figures';
% topdir = '../../analyses/2013-08-30-time-series/figures';
% latexdir = '../../analyses/2013-08-30-time-series/figures';
% topdir = '../../analyses/2013-08-31-time-series/figures';
% latexdir = '../../analyses/2013-08-31-time-series/figures';
% topdir = '../../analyses/2013-09-01-time-series/figures';
% latexdir = '../../analyses/2013-09-01-time-series/figures';
% topdir = '../../analyses/2013-09-02-summary/figures';
% latexdir = '../../analyses/2013-09-02-summary/figures';
% topdir = '../../analyses/2013-09-05/figures';
% latexdir = '../../analyses/2013-09-05/figures';
topdir = '../../analyses/2013-09-02-projective/figures';
latexdir = '../../analyses/2013-09-02-projective/figures';
% topdir = '../../analyses/2013-09-05-projective/figures';
% latexdir = '../../analyses/2013-09-05-projective/figures';
dirnames = dir(topdir);
isub = [dirnames(:).isdir]; %# returns logical vector
dirnames = {dirnames(isub).name}';
dirnames(ismember(dirnames,{'.','..'})) = [];

%dirnames = [];
%dirnames{end+1} = '11-Feb-02-solar-s';
%dirnames{end+1} = '11-Feb-03-mauna2003-s';
%dirnames{end+1} = '31-Jan-v301-airline-months';

%dirnames{end+1} = '11-Feb-v4-03-mauna2003-s_max_level_0';
%dirnames{end+1} = '11-Feb-v4-03-mauna2003-s_max_level_1';
%dirnames{end+1} = '11-Feb-v4-03-mauna2003-s_max_level_2';
%dirnames{end+1} = '11-Feb-v4-03-mauna2003-s_max_level_3';

for i = 1:length(dirnames)
    dirname = dirnames{i};
    files = dir([topdir, '/', dirname, '/*.fig']);
    for f_ix = 1:numel(files)
        curfile = [topdir, '/', dirname, '/', files(f_ix).name];
        h = open(curfile);
        outfile = [topdir, '/', dirname, '/', files(f_ix).name];
        pdfname = strrep(outfile, '.fig', '')
        save2pdf( pdfname, gcf, 600, true );
        %export_fig(pdfname, '-pdf');
        close all
    end
end
