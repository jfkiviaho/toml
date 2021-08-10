# Makefile for ML/TO paper


# Commands
make     = make
pdflatex = pdflatex
bibtool  = bibtex
rm       = rm -f


# File definitions
source = draft.tex
style  = common.sty
refs   = references.bib

target = $(source:.tex=.pdf)
root   = $(source:.tex=)
files  = $(shell ls)
output = $(filter-out %.tex %.pdf, $(shell ls $(root)*))


# Phony targets
.PHONY: tidy clean


# Rules
default: $(target)

$(target): $(source) $(style) $(refs)
	$(pdflatex) $(source)
	$(bibtool)  $(root)
	$(pdflatex) $(source)
	$(pdflatex) $(source)

tidy: 
	$(rm) $(output)

clean: tidy
	$(rm) $(target)
