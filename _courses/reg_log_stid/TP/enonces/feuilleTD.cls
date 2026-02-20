%%
%% This is file `feuilleTD.cls',
%%
%% The original source files were:
%%
%% article.cls
%% 
%% 
%% Copyright 1993 1994 1995 1996 1997 1998 1999 2000 2001
%% The LaTeX3 Project and any individual authors listed elsewhere
%% in this file.
%%
%% author : Stéphane Labbé, stephane.labbe@math.u-psud.fr
%% 
%% This file was generated from file(s) of the LaTeX base system.
%% --------------------------------------------------------------
%% 
%% It may be distributed and/or modified under the
%% conditions of the LaTeX Project Public License, either version 1.2
%% of this license or (at your option) any later version.
%% The latest version of this license is in
%%    http://www.latex-project.org/lppl.txt
%% and version 1.2 or later is part of all distributions of LaTeX
%% version 1999/12/01 or later.
%% 
%% This file may only be distributed together with a copy of the LaTeX
%% base system. You may however distribute the LaTeX base system without
%% such generated files.
%% 
%% The list of all files belonging to the LaTeX base distribution is
%% given in the file `manifest.txt'. See also `legal.txt' for additional
%% information.
%% 
%% \CharacterTable
%%  {Upper-case    \A\B\C\D\E\F\G\H\I\J\K\L\M\N\O\P\Q\R\S\T\U\V\W\X\Y\Z
%%   Lower-case    \a\b\c\d\e\f\g\h\i\j\k\l\m\n\o\p\q\r\s\t\u\v\w\x\y\z
%%   Digits        \0\1\2\3\4\5\6\7\8\9
%%   Exclamation   \!     Double quote  \"     Hash (number) \#
%%   Dollar        \$     Percent       \%     Ampersand     \&
%%   Acute accent  \'     Left paren    \(     Right paren   \)
%%   Asterisk      \*     Plus          \+     Comma         \,
%%   Minus         \-     Point         \.     Solidus       \/
%%   Colon         \:     Semicolon     \;     Less than     \<
%%   Equals        \=     Greater than  \>     Question mark \?
%%   Commercial at \@     Left bracket  \[     Backslash     \\
%%   Right bracket \]     Circumflex    \^     Underscore    \_
%%   Grave accent  \`     Left brace    \{     Vertical bar  \|
%%   Right brace   \}     Tilde         \~}
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{feuilleTD}
              [2001/09/02 v0.1
 Standard LaTeX document class]
\RequirePackage{epsf}
\RequirePackage{t1enc}
\RequirePackage[latin1]{inputenc}
\RequirePackage[french]{babel}
\RequirePackage{amsfonts}
\RequirePackage{fancyhdr}
\newcommand\@ptsize{}
\newif\if@restonecol
\newif\if@titlepage
\DeclareOption{a4paper}
   {\setlength\paperheight {297mm}%
    \setlength\paperwidth  {210mm}}
\DeclareOption{oneside}{\@twosidefalse \@mparswitchfalse}
\DeclareOption{onecolumn}{\@twocolumnfalse}
\ExecuteOptions{a4paper,oneside,onecolumn,final}
\ProcessOptions
%modif taille police
\input{size11.clo}
\setlength\lineskip{1\p@}
\setlength\normallineskip{1\p@}
\renewcommand\baselinestretch{}
\setlength\parskip{0\p@ \@plus \p@}
\@lowpenalty   51
\@medpenalty  151
\@highpenalty 301
\setcounter{topnumber}{2}
\renewcommand\topfraction{.7}
\setcounter{bottomnumber}{1}
\renewcommand\bottomfraction{.3}
\setcounter{totalnumber}{3}
\renewcommand\textfraction{.2}
\renewcommand\floatpagefraction{.5}
\setcounter{dbltopnumber}{2}
\renewcommand\dbltopfraction{.7}
\renewcommand\dblfloatpagefraction{.5}
\def\ps@headings{%
  \let\@oddfoot\@empty
  \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
  \let\@mkboth\markboth
  \def\sectionmark##1{%
    \markright {\MakeUppercase{%
        \ifnum \c@secnumdepth >\m@ne
        \thesection\quad
        \fi
        ##1}}}}
\newcommand\maketitle{\par
  \begingroup
    \renewcommand\thefootnote{\@fnsymbol\c@footnote}%
    \def\@makefnmark{\rlap{\@textsuperscript{\normalfont\@thefnmark}}}%
    \long\def\@makefntext##1{\parindent 1em\noindent
            \hb@xt@1.8em{%
                \hss\@textsuperscript{\normalfont\@thefnmark}}##1}%
    \if@twocolumn
      \ifnum \col@number=\@ne
        \@maketitle
      \else
        \twocolumn[\@maketitle]%
      \fi
    \else
      \newpage
      \global\@topnum\z@   % Prevents figures from going at top of page.
      \@maketitle
    \fi
    \thispagestyle{plain}\@thanks
  \endgroup
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@date\@empty
  \global\let\@author\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\date\relax
  \global\let\author\relax
  \global\let\and\relax
}
\def\@maketitle{%
  \newpage
  \null
  \vskip 2em%
  \begin{center}%
  \let \footnote \thanks
    {\LARGE \@title \par}%
    \vskip 1.5em%
    {\large
      \lineskip .5em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
    \vskip 1em%
    {\large \@date}%
  \end{center}%
  \par
  \vskip 1.5em}

\leftmargin  \leftmargini
\setlength\leftmarginii  {2.2em}
\setlength\leftmarginiii {1.87em}
\setlength\leftmarginiv  {1.7em}
\setlength  \labelsep  {.5em}
\setlength  \labelwidth{\leftmargini}
\addtolength\labelwidth{-\labelsep}
\@beginparpenalty -\@lowpenalty
\@endparpenalty   -\@lowpenalty
\@itempenalty     -\@lowpenalty
\renewcommand\theenumi{\@arabic\c@enumi}
\renewcommand\theenumii{\@alph\c@enumii}
\renewcommand\theenumiii{\@roman\c@enumiii}
\renewcommand\theenumiv{\@Alph\c@enumiv}
\newcommand\labelenumi{\theenumi.}
\newcommand\labelenumii{(\theenumii)}
\newcommand\labelenumiii{\theenumiii.}
\newcommand\labelenumiv{\theenumiv.}
\renewcommand\p@enumii{\theenumi}
\renewcommand\p@enumiii{\theenumi(\theenumii)}
\renewcommand\p@enumiv{\p@enumiii\theenumiii}
\newcommand\labelitemi{\textbullet}
\newcommand\labelitemii{\normalfont\bfseries \textendash}
\newcommand\labelitemiii{\textasteriskcentered}
\newcommand\labelitemiv{\textperiodcentered}
\newcommand*\descriptionlabel[1]{\hspace\labelsep
                                \normalfont\bfseries #1}
\newcommand\appendix{\par
  \setcounter{section}{0}%
  \setcounter{subsection}{0}%
  \gdef\thesection{\@Alph\c@section}}
\setlength\arraycolsep{5\p@}
\setlength\tabcolsep{6\p@}
\setlength\arrayrulewidth{.4\p@}
\setlength\doublerulesep{2\p@}
\setlength\tabbingsep{\labelsep}
\skip\@mpfootins = \skip\footins
\setlength\fboxsep{3\p@}
\setlength\fboxrule{.4\p@}
\renewcommand \theequation {\@arabic\c@equation}
\newcounter{figure}
\renewcommand \thefigure {\@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{\figurename~\thefigure}
\newenvironment{figure}
               {\@float{figure}}
               {\end@float}
\newenvironment{figure*}
               {\@dblfloat{figure}}
               {\end@dblfloat}
\newcounter{table}
\renewcommand\thetable{\@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{\tablename~\thetable}
\newenvironment{table}
               {\@float{table}}
               {\end@float}
\newenvironment{table*}
               {\@dblfloat{table}}
               {\end@dblfloat}
\newlength\abovecaptionskip
\newlength\belowcaptionskip
\setlength\abovecaptionskip{10\p@}
\setlength\belowcaptionskip{0\p@}
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{#1: #2}%
  \ifdim \wd\@tempboxa >\hsize
    #1: #2\par
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*\cal{\@fontswitch\relax\mathcal}
\DeclareRobustCommand*\mit{\@fontswitch\relax\mathnormal}
\newcommand\@pnumwidth{1.55em}
\newcommand\@tocrmarg{2.55em}
\newcommand\@dotsep{4.5}
\setcounter{tocdepth}{3}
\let\l@table\l@figure
\newcommand\newblock{\hskip .11em\@plus.33em\@minus.07em}
\renewcommand\footnoterule{%
  \kern-3\p@
  \hrule\@width.4\columnwidth
  \kern2.6\p@}
\newcommand\@makefntext[1]{%
    \parindent 1em%
    \noindent
    \hb@xt@1.8em{\hss\@makefnmark}#1}
\newcommand\figurename{Figure}
\def\today{\ifcase\month\or
  Janvier\or Février\or Mars\or Avril\or Mai\or Juin\or
  Juillet\or Août\or Septembre\or Octobre\or Novembre\or Décembre\fi
  \space\number\day, \number\year}
\setlength\columnsep{10\p@}
\setlength\columnseprule{0\p@}
\pagestyle{plain}
\pagenumbering{arabic}
\if@twoside
\else
  \raggedbottom
\fi
\if@twocolumn
  \twocolumn
  \sloppy
  \flushbottom
\else
  \onecolumn
\fi
%-------Réglage de la taille de la page------
\textwidth  15 cm
\textheight 25 cm
\voffset -2 cm
\hoffset -1,5 cm
%-------Commandes spéciales------------------
%=========Definitions==================== 
\newcommand{\Matlab}{{\sc Matlab }}
\newcommand{\com}[1]{{\tt #1}}
\newcommand {\Cc}  {\mathbb{C}} 
\newcommand {\Rr}  {\mathbb{R}} 
\newcommand {\Nn}  {\mathbb{N}} 
\newcommand {\Zz}  {\mathbb{Z}} 
\newcommand {\Rrp}  {{\mathbb{R}}} 
\newcommand {\ds} {\displaystyle} 
\newcommand {\derivp}  [2] {\frac{\partial #1}{\partial #2}} 
\newcommand {\derivpn}  [3] {\frac{\partial^{#3} #1}{\partial #2^{#3}}} 
\newcommand {\derivt}  [2] {\frac{d\ #1}{d\ #2}} 
\newcommand {\derivpt} [1] {\displaystyle\frac{\partial #1}{\partial t}} 
\newcommand {\derivpT} [1] {\displaystyle\frac{\partial #1}{\partial T}} 
\newcommand {\vcth}  [1] {\hat{\mbox{\bf #1}}} 
\newcommand {\vct}   [1] {\mbox{\bf #1}} 
\newcommand {\ope}   [1] {\mbox{#1}} 
\newcommand {\CpO} [2] {{\cal C}^{#1}(#2)}
\newcommand {\Cp} [1] {{\cal C}^{#1}(\Rr)}
\newcommand {\CpnO} [3] {{\cal C}^{#1}_{#2}(#3)}
\newcommand {\normL} [3] {|\!|#1|\!|_{#2,#3}}
\newcounter{numexo}\setcounter{numexo}{0}
\newcounter{numquestdevoir}
\newcounter{numquest}
\newcounter{numsousquest}
\newcommand{\EnteteExo}[1]{
        \addtocounter{numexo}{1}
        \noindent
        \begin{center}\rule{\textwidth}{1pt}\end{center}
        {\bf Exercice - } \arabic{numexo}\hspace{0,3cm}{\it #1}\vspace{-0,2cm}
        \begin{center}\rule{\textwidth}{1pt}\end{center}
        \setcounter{numquestdevoir}{1}
        \setcounter{numquest}{1}
        }
\newcommand{\EnteteNotes}[1]{
        \noindent
        \begin{center}\rule{\textwidth}{1pt}\end{center}
        {\bf #1}\vspace{-0,2cm}
        \begin{center}\rule{\textwidth}{1pt}\end{center}
        \setcounter{numquestdevoir}{1}
        \setcounter{numquest}{1}
        }
\newcommand{\questionDevoir}{\noindent {\bf\large\sc \alph{numquestdevoir}- }\addtocounter{numquestdevoir}{1}\setcounter{numquest}{1}}
\newcommand{\question}{\noindent {\bf \arabic{numquest}- }\addtocounter{numquest}{1}\setcounter{numsousquest}{1}}
\newcommand{\sousquestion}{\noindent {\bf \alph{numsousquest}- }\addtocounter{numsousquest}{1}}
\newcommand{\matiereTitre}[1]{\def\DmatiereTitre{#1}}
\newcommand{\matiereEntete}[1]{\def\DmatiereEntete{#1}}
\newcommand{\annee}[1]{\def\Dannee{#1}}
\newcommand{\feuilleTD}[1]{\def\DfeuilleTD{#1}}
\newcommand{\unitevaleur}[1]{\def\Dunitevaleur{#1}}
\newcommand{\etablissement}[1]{\def\Detablissement{#1}}
\global\let\@DfeuilleTD\@empty
\global\let\DfeuilleTD\relax
%-------Définition par défaut de variables---
\title{\DmatiereTitre\\ \Dunitevaleur\\ \DfeuilleTD}
\pagestyle{fancy}
\fancyhead[L]{\DfeuilleTD}
\fancyhead[C]{\bf \Dunitevaleur}
\fancyhead[R]{\DmatiereEntete$\;$\Dannee}
\fancyfoot[L]{\it \Detablissement}
\fancyfoot[C]{\thepage}
\endinput
%%
%% End of file `feuilleTD.cls'.
