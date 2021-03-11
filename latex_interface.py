
default_header=r'''\documentclass[11pt, oneside]{{article}}   	% use amsart instead of "article" for AMSLaTeX format
\usepackage{{geometry}}                		% See geometry.pdf to learn the layout options. There are lots.
\geometry{{letterpaper}}                   		% ... or a4paper or a5paper or ... 
%\geometry{{landscape}}              		% Activate for rotated page geometry
%\usepackage[parfill]{{parskip}}    		% Activate to begin paragraphs with an empty line rather than an indent
\usepackage{{graphicx}}				        % Use pdf, png, jpg, or eps§ with pdflatex; use eps in DVI mode
								            % TeX will automatically convert eps --> pdf in pdflatex		
\usepackage{{amssymb}}
\usepackage{{amsmath}}


%SetFonts

%SetFonts

'''
d2 = r'''
\title{{{t1}}}
\author{{{a1}}}
'''

d3=r'''
\begin{{document}}
\maketitle'''



def header(title, author):
    return default_header.format() + d2.format(t1=title, a1=author) +d3.format()

def footer():
    return '''
    
            \end{document}'''