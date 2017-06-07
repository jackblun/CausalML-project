This file gives a short description of each file's rule in the Final Econ293 project by Luis Armona, Jack Blundell, and Karthik Rajkumar, for the Deep IV section.

Main Files:
-mdn.py: a library containing functions to estimate/run both the first stage distribution of deepIV as a mixture density network (MDN) or as a multinomial (MN) for the case of a discrete endogenous variable (in our setting, years of education)
-deepiv.py: a library containing functions to estimate/run the second stage of deepIV, given a distribution for each observation estimated w/ mdn.py. In particular, it contains functions for the loss function and gradient outlined in hartford et al., along with the training / CV functions for the 2nd stage. It also contains functions to perform frequentist counterfactuals, including estimating treatments/instruments, and the IV coefs.
-deepiv_plots.r: plots the CV performance for our estimators


paper-specific files files (AK/AJR):
-train_first_stage.py/train_first_stage_mdn.py: estimate the first stage instrument-driven distribution of the endogenous variable (using MN in AK, MDN in AJR).
-train_second_stage_mp: use parallelization to cross-validate # of nodes for second stage response network.
-predict_counterfactuals.py: estimates frequentist IV coefs corresponding to 2nd stage NN and does some counterfactuals for each paper

