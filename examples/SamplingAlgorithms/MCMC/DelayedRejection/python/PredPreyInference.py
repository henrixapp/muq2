import muq.Modeling as mm
import muq.SamplingAlgorithms as ms
import muq.Approximation as ma
import muq.Utilities as mu

import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../../../../Modeling/CustomModPiece/python')

# Import a DiffusionEquation ModPiece from the CustomModPiece example
from LotkaModPiece import LotkaEquation

# The standard deviation of the additive Gaussian noise
noiseStd = 0.05

# Number of cells to use in Finite element discretization
timeSteps = 400

mod = LotkaEquation(timeSteps,1.0,.5,.5,0.5,0.5)

# Create an exponential operator to transform log-conductivity into conductivity

# Combine the two models into a graph
graph = mm.WorkGraph()
graph.AddNode(mm.IdentityOperator(1), "in1")
#graph.AddNode(cond, "Conductivity")
#graph.AddNode(recharge, "Recharge")
graph.AddNode(mod,"Forward Model")


graph.AddEdge("in1", 0, "Forward Model", 0)
#graph.AddEdge("Conductivity",0,"Forward Model",0)
#graph.AddEdge("Recharge",0,"Forward Model",1)


# Set up the Gaussian process prior on the log-conductivity
priorMean = [1.0]
priorStd = 0.2
prior =  mm.Gaussian(priorMean, priorStd*priorStd*np.ones((1,)))
graph.AddNode(prior.AsDensity(), "Prior")
graph.AddEdge("in1",0,"Prior",0)

# Generate a "true" log conductivity
#trueLogK = prior.Sample()
truePredPop = [0.8]
truePreyPop = mod.Evaluate([truePredPop]) # after 20 timesteps
obsPreyPop = truePreyPop + noiseStd*mu.RandomGenerator.GetNormal(1)[:,0]
print(obsPreyPop.shape)
# Set up the likelihood and posterior
likely = mm.Gaussian(obsPreyPop, noiseStd*noiseStd*np.ones((1,)))
graph.AddNode(likely.AsDensity(),"Likelihood")
graph.AddNode(mm.DensityProduct(2), "Posterior")

graph.AddEdge("Forward Model",0,"Likelihood",0)
graph.AddEdge("Prior",0,"Posterior",0)
graph.AddEdge("Likelihood",0,"Posterior",1)

graph.Visualize('ModelGraph2.png')

#### MCMC
opts = dict()
opts['NumSamples'] = 20000 # Number of MCMC steps to take
opts['BurnIn'] = 20 # Number of steps to throw away as burn in
opts['PrintLevel'] = 3 # in {0,1,2,3} Verbosity of the output
opts['Beta'] = 0.02 # Crank Nicholson parameter
opts['StepSize'] = 1e-5 # MALA Step Size

postDens = graph.CreateModPiece("Posterior")
problem = ms.SamplingProblem(postDens)

# Construct a prior-preconditioned Crank-Nicolson proposal
pcnProp = ms.CrankNicolsonProposal(opts, problem, prior)
malaProp = ms.MALAProposal(opts,problem,prior)

# Use the proposal to construct a Metropolis-Hastings kernel
kern = ms.MHKernel(opts,problem,pcnProp) # <- USE THIS FOR PRIOR-PRECONDITIONED MALA
#kern = ms.MHKernel(opts,problem,malaProp) # <- USE THIS FOR PRIOR-PRECONDITIONED Crank-Nicolson
#kern = ms.DRKernel(opts, problem, [pcnProp], [1.0]) # <- USE THIS FOR Delayed Rejection

# Construct the MCMC sampler using this transition kernel
sampler = ms.SingleChainMCMC(opts, [kern])

# Run the MCMC sampler
x0 = [[1.5]]
samps = sampler.Run(x0)

# Extract the posteior samples as a matrix and compute some posterior statistics
sampMat = samps.AsMatrix()

print(sampMat)
postMean = np.mean(sampMat,axis=1)
print(postMean)
q05 = np.percentile(sampMat,5,axis=1)
q95 = np.percentile(sampMat,95,axis=1)

# Plot the results
fig, axs = plt.subplots(ncols=3)
axs[0].plot(trueHead, label='True Head')
axs[0].plot(obsHead,'.k',label='Observed Head')
axs[0].legend()
axs[0].set_title('Data')
axs[0].set_xlabel('Position, $x$')
axs[0].set_ylabel('Hydraulic head $h(x)$')

axs[1].fill_between(mod.xs[0:-1], q05, q95, alpha=0.5,label='5%-95% CI')
axs[1].plot(mod.xs[0:-1],postMean, label='Posterior Mean')
axs[1].plot(mod.xs[0:-1],trueLogK,label='Truth')
axs[1].legend()
axs[1].set_title('Posterior on log(K)')
axs[1].set_xlabel('Position, $x$')
axs[1].set_ylabel('Log-Conductivity $log(K(x))$')

axs[2].plot(sampMat[0,:], label='$log K_0$')
axs[2].plot(sampMat[100,:],label='$log K_{100}$')
axs[2].plot(sampMat[190,:],label='$log K_{190}')
axs[2].set_title('MCMC Trace')
axs[2].set_xlabel('MCMC Iteration (after burnin)')
axs[2].set_ylabel('Log-Conductivity')
plt.show()
