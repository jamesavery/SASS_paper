.PHONY: SN-main.pdf
SN-main.pdf: SN-main.tex
	pdflatex SN-main
	pdflatex SN-main
	bibtex SN-main
	pdflatex SN-main


clean:
	rm -f *.aux *.log *.toc *.spl *.out *.bbl *.blg *~ \#*\#
