% Turns all figures in a directory into pdfs.
%
% Should be run from source/matlab/
%
% David Duvenaud
% Feb 2013
 
% topdir = '../../analyses/2014-02-19-depths';
% latexdir = '../../analyses/2014-02-19-depths';
% topdir = '../../analyses/2014-02-20-pl2';
% latexdir = '../../analyses/2014-02-20-pl2';
% topdir = '../../analyses/2014-04-21-motorcycle';
% latexdir = '../../analyses/2014-04-21-motorcycle';
% topdir = '../../analyses/2014-05-19-GPSS-add-mmd';
% latexdir = '../../analyses/2014-05-19-GPSS-add-mmd';
% topdir = '../../analyses/2014-05-19-SE-mmd';
% latexdir = '../../analyses/2014-05-19-SE-mmd';
% topdir = '../../analyses/2014-05-19-TCI-mmd';
% latexdir = '../../analyses/2014-05-19-TCI-mmd';
% topdir = '../../analyses/2014-05-19-SP-mmd';
% latexdir = '../../analyses/2014-05-19-SP-mmd';
% topdir = '../../analyses/2014-05-28-prejudice';
% latexdir = '../../analyses/2014-05-28-prejudice';
% topdir = '../../analyses/2014-11-10-gefcom-revisited';
% latexdir = '../../analyses/2014-11-10-gefcom-revisited';
% topdir = '../../analyses/2014-11-11-gefcom-revisited-v2';
% latexdir = '../../analyses/2014-11-11-gefcom-revisited-v2';
topdir = '../../analyses/2014-11-11-gefcom-revisited-hint';
latexdir = '../../analyses/2014-11-11-gefcom-revisited-hint';
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
% for i = 14:20
%for i = 53:length(dirnames)
%for i = 1
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
