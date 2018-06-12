#ifndef TRANSITIONKERNEL_H_
#define TRANSITIONKERNEL_H_

#include <iostream>
#include <map>

#include <boost/function.hpp>
#include <boost/property_tree/ptree.hpp>

#include "MUQ/Utilities/RegisterClassName.h"

#include "MUQ/Modeling/WorkPiece.h"

#include "MUQ/SamplingAlgorithms/SamplingProblem.h"
#include "MUQ/SamplingAlgorithms/SamplingState.h"

namespace muq {
  namespace SamplingAlgorithms {
    class TransitionKernel { //: public muq::Modeling::WorkPiece {
    public:

      TransitionKernel(boost::property_tree::ptree const& pt, std::shared_ptr<AbstractSamplingProblem> problem);

      virtual ~TransitionKernel() = default;

      /// Static constructor for the transition kernel
      /**
	 @param[in] pt Parameters for the kernel
	 @param[in] problem The sampling problem that evaluates/samples the distribution we are trying to characterize
	 \return The transition kernel
       */
      static std::shared_ptr<TransitionKernel> Construct(boost::property_tree::ptree const& pt, std::shared_ptr<AbstractSamplingProblem> problem);
      typedef boost::function<std::shared_ptr<TransitionKernel>(boost::property_tree::ptree, std::shared_ptr<AbstractSamplingProblem>)> TransitionKernelConstructor;
      typedef std::map<std::string, TransitionKernelConstructor> TransitionKernelMap;
      static std::shared_ptr<TransitionKernelMap> GetTransitionKernelMap();


      virtual void PreStep(unsigned int const t, std::shared_ptr<SamplingState> state) {};

      /// Allow the kernel to adapt given a new state
      /**
	 By default this function does nothing but children can override it to adapt the kernel
	 @param[in] t The current step
	 @param[in] state The current state
       */
      virtual void PostStep(unsigned int const t, std::vector<std::shared_ptr<SamplingState>> const& state) {};

      virtual std::vector<std::shared_ptr<SamplingState>> Step(unsigned int const t, std::shared_ptr<SamplingState> prevState) = 0;

      virtual void PrintStatus() const {PrintStatus("");};
      virtual void PrintStatus(std::string prefix) const {};

      // What block of the state does this kernel work on?
      const int blockInd = 0;

    protected:

      /// The sampling problem that evaluates/samples the target distribution
      std::shared_ptr<AbstractSamplingProblem> problem;

    private:
    };
  } // namespace SamplingAlgorithms
} // namespace muq

#define REGISTER_TRANSITION_KERNEL(NAME) static auto reg ##NAME		\
  = muq::SamplingAlgorithms::TransitionKernel::GetTransitionKernelMap()->insert(std::make_pair(#NAME, muq::Utilities::shared_factory<NAME>()));

#endif
