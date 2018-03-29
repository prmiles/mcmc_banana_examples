#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 13:34:00 2018
% Example from Marko Laine's website: http://helios.fmi.fi/~lainema/mcmc/
% 
% This technical example constructs a non Gaussian target
% distribution by twisting two first dimensions of Gaussian
% distribution. The Jacobian of the transformation is 1, so it is
% easy to calculate the right probability regions for the banana
% and study different adaptive methods.
%
% We demonstrate sampling using the following algorithms:
    MH - Metropolis Hastings
    AM - Adaptive Metropolis
    DR - Delayed Rejection
    DRAM - Delayed Rejection Adaptive Metropolis
%%
@author: prmiles
"""

# import required packages
from __future__ import division
import numpy as np
from pymcmcstat.MCMC import MCMC
from pymcmcstat.MCMCPlotting import MCMCPlotting

# Define model object
def bananafunction(x, a, b):
    response = x
    response[:,0] = a*x[:,0]
    response[:,1] = x[:,1]*a**(-1) - b*((a*x[:,0])**(2) + a**2)
    return response

def bananainverse(x, a, b):
    response = x
    response[0] = x[0]*a**(-1)
    response[1] = x[1]*a + a*b*(x[0]**2 + a**2)
    return response

def bananass(theta, data):
    udobj = data.user_defined_object[0]
    lam = udobj.lam
    mu = udobj.mu
    a = udobj.a
    b = udobj.b
    npar = udobj.npar
    
    x = np.array([theta])
    x = x.reshape(npar,1)
    
    baninv = bananainverse(x-mu, a, b)
    
    stage1 = np.matmul(baninv.transpose(),lam)
    stage2 = np.matmul(stage1, baninv)
    
    return stage2

class Banana_Parameters:
    def __init__(self, rho = 0.9, npar = 12, a = 1, b = 1, mu = None):
        self.rho = rho
        self.npar = npar
        self.a = a
        self.b = b
        
        self.sig = np.eye(npar)
        self.sig[0,1] = rho
        self.sig[1,0] = rho
        self.lam = np.linalg.inv(self.sig)
        
        if mu is None:
            self.mu = np.zeros([npar, 1])
            
npar = 2 # number of model parameters
udobj = Banana_Parameters(npar = npar) # user defined object

# Initialize MCMC objects
mh = MCMC()
am = MCMC()
dr = MCMC()
dram = MCMC()

# initialize data within each MCMC object
mh.data.add_data_set(np.zeros(1),np.zeros(1), user_defined_object = udobj)
am.data.add_data_set(np.zeros(1),np.zeros(1), user_defined_object = udobj)
dr.data.add_data_set(np.zeros(1),np.zeros(1), user_defined_object = udobj)
dram.data.add_data_set(np.zeros(1),np.zeros(1), user_defined_object = udobj)

# Add model parameters
for ii in xrange(npar):
    mh.parameters.add_model_parameter(name = str('$x_{}$'.format(ii+1)), theta0 = 0.0)
    am.parameters.add_model_parameter(name = str('$x_{}$'.format(ii+1)), theta0 = 0.0)
    dr.parameters.add_model_parameter(name = str('$x_{}$'.format(ii+1)), theta0 = 0.0)
    dram.parameters.add_model_parameter(name = str('$x_{}$'.format(ii+1)), theta0 = 0.0)

# Define options - include sampling algorithm!
mh.simulation_options.define_simulation_options(nsimu = int(2.0e3), qcov = np.eye(npar)*5, method='mh')
am.simulation_options.define_simulation_options(nsimu = int(2.0e3), qcov = np.eye(npar)*5, method='am', adaptint=100)
dr.simulation_options.define_simulation_options(nsimu = int(2.0e3), qcov = np.eye(npar)*5, method='dr', ntry = 2)
dram.simulation_options.define_simulation_options(nsimu = int(2.0e3), qcov = np.eye(npar)*5, method='dram', adaptint=100, ntry = 2)

# Define model settings
mh.model_settings.define_model_settings(sos_function = bananass, N = 1)
am.model_settings.define_model_settings(sos_function = bananass, N = 1)
dr.model_settings.define_model_settings(sos_function = bananass, N = 1)
dram.model_settings.define_model_settings(sos_function = bananass, N = 1)

# Run simulation
mh.run_simulation()
am.run_simulation()
dr.run_simulation()
dram.run_simulation()

# Extract results
mh_results = mh.simulation_results.results
am_results = am.simulation_results.results
dr_results = dr.simulation_results.results
dram_results = dram.simulation_results.results

# MCMC Plotting Routines
mcpl = MCMCPlotting() 

# plot chain panel
mcpl.plot_chain_panel(dram_results['chain'],dram_results['names'])

# plot pairwise correlation
mcpl.plot_pairwise_correlation_panel(dram_results['chain'][:,0:2], dram_results['names'][0:2])

# Print acceptance statistics
print('\n----------------\n')
print('MH: Number of accepted runs: {} out of {} ({})'.format(len(np.unique(mh_results['chain'][:,0])), mh.simulation_options.nsimu, 100*(1-mh_results['total_rejected'])))
print('AM: Number of accepted runs: {} out of {} ({})'.format(len(np.unique(am_results['chain'][:,0])), am.simulation_options.nsimu, 100*(1-am_results['total_rejected'])))
print('DR: Number of accepted runs: {} out of {} ({})'.format(len(np.unique(dr_results['chain'][:,0])), dr.simulation_options.nsimu, 100*(1-dr_results['total_rejected'])))
print('DRAM: Number of accepted runs: {} out of {} ({})'.format(len(np.unique(dram_results['chain'][:,0])), dram.simulation_options.nsimu, 100*(1-dram_results['total_rejected'])))

# save chains for comparison
np.savetxt('mh.txt', mh_results['chain'], delimiter=',')
np.savetxt('am.txt', am_results['chain'], delimiter=',')
np.savetxt('dr.txt', dr_results['chain'], delimiter=',')
np.savetxt('dram.txt', dram_results['chain'], delimiter=',')