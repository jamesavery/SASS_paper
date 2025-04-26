.PHONY: sn-article.pdf
SN-main.pdf: sn-article.tex
	pdflatex sn-article
	bibtex sn-article
	pdflatex sn-article
	pdflatex sn-article

fast: main.tex
	pdflatex sn-article

clean:
	rm -f *.aux *.log *.toc *.spl *.out *.bbl *.blg *~ \#*\#
