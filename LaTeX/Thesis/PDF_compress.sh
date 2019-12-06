#setting=[/screen, /ebook, /printer, /prepress]
#Quality level settings are "/screen," the lowest 
#resolution and lowest file size, but fine for viewing 
#on a screen; "/ebook," a mid-point in resolution and 
#file size; "/printer" and "/prepress," high-quality 
#settings used for printing PDFs.

#REPLACE: setting, output.pdf, input.pdf

#https://www.techwalla.com/articles/reduce-pdf-file-size-linux

gs -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/printer -sOutputFile=Gilbert_Eaton_THESIS_final.pdf Gilbert_Eaton_THESIS.pdf
