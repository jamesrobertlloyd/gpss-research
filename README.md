This is part of the [automatic statistician](http://www.automaticstatistician.com/) project
========

Automatic Bayesian Covariance Discovery
=====================

<img src="https://raw.githubusercontent.com/jamesrobertlloyd/gpss-research/master/logo.png" width="700">

This repo contains the source code to run the system described in the paper

[Automatic Construction and Natural-Language Description of Nonparametric Regression Models](http://arxiv.org/pdf/1402.4304.pdf)
by James Robert Lloyd, David Duvenaud, Roger Grosse, Joshua B. Tenenbaum and Zoubin Ghahramani,
appearing in [AAAI 2014](http://www.aaai.org/Conferences/AAAI/aaai14.php).


### Abstract

This paper presents the beginnings of an automatic statistician, focusing on regression problems. Our system explores an open-ended space of statistical models to discover a good explanation of a data set, and then produces a detailed report with figures and natural-language text. Our approach treats unknown regression functions nonparametrically using Gaussian processes, which has two important consequences. First, Gaussian processes can model functions in terms of high-level properties (e.g. smoothness, trends, periodicity, changepoints). Taken together with the compositional structure of our language of models this allows us to automatically describe functions in simple terms. Second, the use of flexible nonparametric models and a rich language for composing them in an open-ended manner also results in state-of-the-art extrapolation performance evaluated over 13 real time series data sets from various domains.

Feel free to email the authors with any questions:  
[James Lloyd](http://mlg.eng.cam.ac.uk/Lloyd/) (jrl44@cam.ac.uk)  
[David Duvenaud](http://people.seas.harvard.edu/~dduvenaud/) (dduvenaud@seas.harvard.edu)  
[Roger Grosse](http://www.cs.toronto.edu/~rgrosse/) (rgrosse@cs.toronto.edu)  


### Data used in the paper

 - [Full time series](https://github.com/jamesrobertlloyd/gpss-research/tree/master/data/tsdlr-renamed)
 - [Extrapolation](https://github.com/jamesrobertlloyd/gpss-research/tree/master/data/tsdlr_9010)
 - [Interpolation](https://github.com/jamesrobertlloyd/gpss-research/tree/master/data/tsdlr_5050)


Related Repo
------------------

Source code to run an earlier version of the system, appearing in 
[Structure Discovery in Nonparametric Regression through Compositional Kernel Search](http://arxiv.org/abs/1302.4922)
by David Duvenaud, James Robert Lloyd, Roger Grosse, Joshua B. Tenenbaum, Zoubin Ghahramani  
can be found at

[github.com/jamesrobertlloyd/gp-structure-search/](www.github.com/jamesrobertlloyd/gp-structure-search/).

