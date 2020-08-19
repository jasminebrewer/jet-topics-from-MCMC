# jet-topics-from-MCMC

###This code is designed to use the Markov Chain Monte Carlo emcee to calculate jet topics in proton--proton and heavy-ion collisions.

Along with the code, we provide sample dijet and photon+jet histograms in proton--proton ('PP') and heavy-ion ('HI') collisions which can be used to run the code. To run the code in its current form, the sample histograms should be saved in the same folder as the .py file.


The syntax to run the code is
./get_topics_from_MCMC.py system ptindex nwalkers nsamples burn_in nkappa

Parameters:
"system" -- allowed values are PP or HI, depending on whether you want to process the PP or HI sample histograms. (note that on windows, these may need to be in quotes, 'PP' or 'HI')

"ptindex" -- allowed values are 0, 1, or 2. These correspond to three different pT bins, [100,120] GeV (0), [120,140] GeV (1), and [140,160] GeV (2), for which we provide sample histograms in each case.

"nwalkers": number of walkers for the MCMC

"nsamples": number of samples taken by each walker in the MCMC

"burn_in": number of samples after which the MCMC is thought to have converged to the posterior. It is required that burn_in < nsamples.

"nkappa": number of samples from the posterior at which to sample kappa. It is required that nkappa < (nsamples - burn_in)*nwalkers.





See https://emcee.readthedocs.io/en/stable/tutorials/line/ for a valuable tutorial on fitting with MCMC. 

