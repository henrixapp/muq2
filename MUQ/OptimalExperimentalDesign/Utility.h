#ifndef UTILITY_H_
#define UTILITY_H_

#include <boost/property_tree/ptree.hpp>

#include "MUQ/config.h"

#if MUQ_HAS_PARCER==1
#include <parcer/Communicator.h>
#endif

#include "MUQ/Modeling/ModPiece.h"
#include "MUQ/Modeling/WorkGraph.h"
#include "MUQ/Modeling/Distributions/Distribution.h"

#include "MUQ/OptimalExperimentalDesign/OEDResidual.h"

namespace muq {
  namespace OptimalExperimentalDesign {
    class Utility : public muq::Modeling::ModPiece {
    public:
      /// Use the prior as the biasing distribution---Monte Carlo estimate
      Utility(std::shared_ptr<muq::Modeling::Distribution> const& prior, std::shared_ptr<muq::Modeling::Distribution> const& likelihood, std::shared_ptr<muq::Modeling::Distribution> const& evidence, boost::property_tree::ptree pt);

      Utility(std::shared_ptr<muq::Modeling::Distribution> const& prior, std::shared_ptr<muq::Modeling::Distribution> const& likelihood, std::shared_ptr<muq::Modeling::Distribution> const& evidence, std::shared_ptr<muq::Modeling::Distribution> const& biasing, boost::property_tree::ptree pt);

      /// Use the prior as the biasing distribution---Monte Carlo estimate
      Utility(std::shared_ptr<muq::Modeling::Distribution> const& prior, std::shared_ptr<OEDResidual> const& resid, boost::property_tree::ptree pt);

      Utility(std::shared_ptr<muq::Modeling::Distribution> const& prior, std::shared_ptr<OEDResidual> const& resid, std::shared_ptr<muq::Modeling::Distribution> const& biasing, boost::property_tree::ptree pt);

#if MUQ_HAS_PARCER==1
      /// Use the prior as the biasing distribution---Monte Carlo estimate
      Utility(std::shared_ptr<muq::Modeling::Distribution> const& prior, std::shared_ptr<muq::Modeling::Distribution> const& likelihood, std::shared_ptr<muq::Modeling::Distribution> const& evidence, boost::property_tree::ptree pt, std::shared_ptr<parcer::Communicator> const& comm);

      Utility(std::shared_ptr<muq::Modeling::Distribution> const& prior, std::shared_ptr<muq::Modeling::Distribution> const& likelihood, std::shared_ptr<muq::Modeling::Distribution> const& evidence, std::shared_ptr<muq::Modeling::Distribution> const& biasing, boost::property_tree::ptree pt, std::shared_ptr<parcer::Communicator> const& comm);

      /// Use the prior as the biasing distribution---Monte Carlo estimate
      Utility(std::shared_ptr<muq::Modeling::Distribution> const& prior, std::shared_ptr<OEDResidual> const& resid, boost::property_tree::ptree pt, std::shared_ptr<parcer::Communicator> const& comm);

      Utility(std::shared_ptr<muq::Modeling::Distribution> const& prior, std::shared_ptr<OEDResidual> const& resid, std::shared_ptr<muq::Modeling::Distribution> const& biasing, boost::property_tree::ptree pt, std::shared_ptr<parcer::Communicator> const& comm);
#endif

      virtual ~Utility() = default;

      unsigned int TotalRefinements() const;

      Eigen::VectorXi RunningRefinements() const;
    private:
      void CreateGraph(std::shared_ptr<muq::Modeling::Distribution> const& prior, std::shared_ptr<muq::Modeling::Distribution> const& likelihood, std::shared_ptr<muq::Modeling::Distribution> const& evidence);

      void CreateGraph(std::shared_ptr<muq::Modeling::Distribution> const& prior, std::shared_ptr<OEDResidual> const& resid, boost::property_tree::ptree pt);

      virtual void EvaluateImpl(muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs) override;

      void EvaluateBruteForce(muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs);

      void EvaluateSurrogate(muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs);

      void RandomlyRefineNear(Eigen::VectorXd const& xd, double const radius);

      void RefineAt(Eigen::VectorXd const& pnt, double const radius);

      const unsigned int numImportanceSamples;
      const double gamma0 = 1.0;
      const double radius0 = 1.0;

      std::shared_ptr<muq::Modeling::Distribution> biasing;

      const bool runningEstimate = false;

#if MUQ_HAS_PARCER==1
      std::shared_ptr<parcer::Communicator> comm;
#endif

      std::shared_ptr<muq::Modeling::WorkGraph> graph;

      std::shared_ptr<muq::Approximation::LocalRegression> reg;

      unsigned int totalRefinements = 0;
      Eigen::VectorXi runningRefinements;
    };
  } // namespace OptimalExperimentalDesign
} // namespace muq

#endif
