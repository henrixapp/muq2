#ifndef GMHKERNEL_H_
#define GMHKERNEL_H_

#include "MUQ/SamplingAlgorithms/MCMCProposal.h"
#include "MUQ/SamplingAlgorithms/MHKernel.h"

namespace muq {
  namespace SamplingAlgorithms {
    /// A kernel for the generalized Metropolis-Hastings kernel
    /**
       Reference: "A general construction for parallelizing Metropolis-Hastings algorithms" (Calderhead, 2014)
     */
    class GMHKernel : public MHKernel {
    public:

      /**
	 @param[in] pt Options for this kenel and the standard Metropolis-Hastings kernel
	 @param[in] problem The problem we want to sample
       */
      GMHKernel(boost::property_tree::ptree const& pt, std::shared_ptr<AbstractSamplingProblem> problem);

      /**
	 @param[in] pt Options for this kenel and the standard Metropolis-Hastings kernel
	 @param[in] problem The problem we want to sample
	 @param[in] proposalIn The proposal for the MCMC chain
       */
      GMHKernel(boost::property_tree::ptree const& pt, std::shared_ptr<AbstractSamplingProblem> problem, std::shared_ptr<MCMCProposal> proposalIn);

      virtual ~GMHKernel();

      /**
	 Propose GMHKernel::N points and compute the cumulative distribution of the stationary distribution for the acceptance probability (GMHKernel::proposedStates and GMHKernel::stationaryAcceptance, respectively)
	 @param[in] t The current step in the MCMC chain
	 @param[in] state The current MCMC state
       */
      virtual void PreStep(unsigned int const t, std::shared_ptr<SamplingState> state) override;


      /**
	 Sample GMHKernel::M states from the proposed states GMHKernel::proposedStates.
	 @param[in] t The current step in the MCMC chain
	 @param[in] state The current MCMC state
	 \return The accepted states
       */
      virtual std::vector<std::shared_ptr<SamplingState> > Step(unsigned int const t, std::shared_ptr<SamplingState> state) override;

      /// Get the cumulative stationary acceptance probability
      /**
	 Must be called after GMHKernel::PreStep.
	 \return The cumulative stationary acceptance probability
       */
      Eigen::VectorXd CumulativeStationaryAcceptance() const;

    private:

      /// Compute the dominate eigenvalue
      /**
	 Populate the GMHKernel::stationaryAcceptance with the dominate eigen value of the transition matrix for the finite-state Markov chain over the proposals.
	 @param[in] A The Markov transition matrix for the finite-state Markov chain over the proposals
	 \return The dominate eigenvalue (it better be 1)
       */
      double PowerIteration(Eigen::MatrixXd const& A);

      /// Number of proposals
      const unsigned int N;

      /// Number of proposals plus one
      /**
	 Defined so we don't aways have to compute \f$N+1\f$.
       */
      const unsigned int Np1;

      /// Number of accepted points (number of points added to the chain)
      const unsigned int M;

      /// Tolerance for the power iteration
      const double tol = 1.0e-12;

      /// Max iterations for the power iteration
      const unsigned int maxIt = 1000;

      /// The cumulative stationary accepatnce probability 
      Eigen::VectorXd stationaryAcceptance;

      /// Proposed states
      std::vector<std::shared_ptr<SamplingState> > proposedStates;
    };
  } // namespace SamplingAlgorithms
} // namespace muq

#endif