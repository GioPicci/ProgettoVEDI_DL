@echo off
echo Compilazione relazione in LaTeX (pdflatex + bibtex)...
cd /d "%~dp0"
pdflatex -interaction=nonstopmode --disable-installer main.tex
bibtex main
pdflatex -interaction=nonstopmode --disable-installer main.tex
pdflatex -interaction=nonstopmode --disable-installer main.tex
echo.
echo Compilazione completata! Il file PDF generato e': main.pdf
pause
