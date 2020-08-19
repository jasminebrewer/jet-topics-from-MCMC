# jet-topics-from-MCMC

This code is designed to use the Markov Chain Monte Carlo (MCMC) emcee to calculate jet topics in proton-proton and heavy-ion collisions. It was developed by Jasmine Brewer and Andrew P. Turner with conceptual oversight from Jesse Thaler. It is based on OUR PREPRINT. If you use this code in part or in its entirety for scientific work, please cite OUR PREPRINT.

Along with the code, get_topics_from_MCMC.py, we also provide sample dijet and photon+jet histograms in proton-proton ('PP_JEWEL_etamax1_constmult.pickle') and heavy-ion collisions ('HI_JEWEL_etamax1_constmult_13invnbYJ.pickle') which can be used as an example to run the code. To run the code in its current form, the pickle files containing the sample histograms should be saved in the same directory as the .py file.


The syntax to run the code is

./get_topics_from_MCMC.py system ptindex nwalkers nsamples burn_in nkappa


Parameters:

"system" -- allowed values are PP or HI, depending on whether you want to process the PP or HI sample histograms.

"ptindex" -- allowed values are 0, 1, or 2. These correspond to three different pT bins, [100,120] GeV (0), [120,140] GeV (1), and [140,160] GeV (2), for which we provide sample histograms in each case.

"nwalkers": number of walkers for the MCMC

"nsamples": number of samples taken by each walker in the MCMC

"burn_in": number of samples after which the MCMC is thought to have converged to the posterior. It is required that burn_in < nsamples.

"nkappa": number of samples from the posterior at which to sample kappa. It is required that nkappa < (nsamples - burn_in)*nwalkers.


Example parameter sets:

An example parameter set to get qualitative results with reasonable computational resources is

./get_topics_from_MCMC.py HI 0 100 8000 7000 1000


Output:

The output of the code is a sequence of plots, saved in the current directory, and a pickle file containing the extracted values of kappa. 

*least-squares_fit.png: shows the input histograms along with the least-squares fit obtained by the code. The MCMC walkers are started in a gaussian ball around these least squares fit parameters, so it is important that these fits are good. "trytimes" specifies how many times to attempt a least-squares fit and the best one is kept; increasing trytimes may improve the fit if it is not converging.

*MCMC_samples.png: shows the parameters of the fit obtained by the MCMC walkers as a function of the time step. The vertical blue line is the value of burn_in specified as an input to the function. It is important that the walkers have converged by the burn_in time or results may be biased.

*kappas_or.png: shows the ratios pdf1/pdf2 and pdf2/pdf1 for the fits extracted from the MCMC (red) compared to the input histograms (blue). The extracted values and locations of kappa12 (left) and kappa21 (right) are shown as black dots.

*topics.png: shows the extracted topics (colored bands) compared to sample distributions of photon+quark and photon+gluon.

*fractions.png: shows histograms of the extracted fractions of topic1 and topic2.

kappas*.pickle: a .pickle file containing the kappas extracted from this run of the MCMC. Along with the input distributions, these kappas can be used to reproduce the topics and fractions plots.


See https://emcee.readthedocs.io/en/stable/tutorials/line/ for a valuable tutorial on fitting with MCMC. 

