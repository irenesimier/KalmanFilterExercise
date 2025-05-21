\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{decar-wsd}[DECAR WSD Template]
% Pass all unrecognized options to article class (including font size)
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[letterpaper]{article}

\RequirePackage{times}  % TODO Obselete package. Find replacement
\RequirePackage[top=1.0in, bottom=1.0in, left=1.0in, right=1.0in]{geometry}
\RequirePackage{fancyhdr}
\RequirePackage{lastpage}
\RequirePackage{titling}
\RequirePackage[toc,page]{appendix}

% Header setup
\pagestyle{fancy}
\lhead{}
\chead{}
\rhead{}
\lfoot{\small \thetitle{} --- \thedate{}}
\cfoot{}
\rfoot{\small Page \thepage{}\ of \pageref{LastPage}}
\renewcommand{\headrulewidth}{0.0pt}
\renewcommand{\footrulewidth}{0.75pt}

% Header setup for first page
\fancypagestyle{plain}
{
  \fancyhf{}
  \lhead{}
  \chead{}
  \rhead{}
  \lfoot{\small \thetitle{} --- \thedate{}}
  \cfoot{}
  \rfoot{\small Page \thepage{}\ of \pageref{LastPage}}
  \renewcommand{\headrulewidth}{0.0pt}
  \renewcommand{\footrulewidth}{0.75pt}
}

\makeatletter
\def\@maketitle{%
  \newpage
  \null
  \vskip 2em%
  \begin{center}%
    \let \footnote \thanks
    {\LARGE \textbf{\@title} \par}%
    \vskip 1.5em%
      {\large
        \lineskip .5em%
        \begin{tabular}[t]{c}%
          \emph{\@author}                                                 \\
          \small{Department of Mechanical Engineering, McGill University} \\
          \small{817 Sherbrooke Street West, Montreal QC H3A 0C3}
        \end{tabular}\par}%
    \vskip 1em%
      {\large \@date}%
  \end{center}%
  \par
  \begin{center}%
    \rule{\linewidth}{1.5pt}%
  \end{center}%
  \raggedright}
\makeatother

% Make \subsubsubsection be numbered and appear in TOC
\setcounter{secnumdepth}{5}
\setcounter{tocdepth}{5}

% TODO Fix title in footer

% Default document title
\title{Weekly Summary Document}
\date{\today{}}
