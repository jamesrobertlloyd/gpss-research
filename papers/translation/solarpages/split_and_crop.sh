pdftk $1 burst
ls pg_*.pdf | xargs -n 1 pdfcrop
