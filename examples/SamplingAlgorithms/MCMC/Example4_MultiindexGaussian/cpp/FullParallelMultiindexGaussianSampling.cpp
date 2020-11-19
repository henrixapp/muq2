#include "MUQ/SamplingAlgorithms/SLMCMC.h"
#include "MUQ/SamplingAlgorithms/GreedyMLMCMC.h"
#include "MUQ/SamplingAlgorithms/MIMCMC.h"

#include "MUQ/Modeling/Distributions/Gaussian.h"
#include "MUQ/Modeling/Distributions/Density.h"

#include "MUQ/SamplingAlgorithms/MHKernel.h"
#include "MUQ/SamplingAlgorithms/MHProposal.h"
#include "MUQ/SamplingAlgorithms/CrankNicolsonProposal.h"
#include "MUQ/SamplingAlgorithms/SamplingProblem.h"
#include "MUQ/SamplingAlgorithms/SubsamplingMIProposal.h"

#include "MUQ/SamplingAlgorithms/MIComponentFactory.h"

#include <boost/property_tree/ptree.hpp>

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;

#include "MUQ/SamplingAlgorithms/ParallelMIMCMCWorker.h"
#include "MUQ/SamplingAlgorithms/ParallelFixedSamplesMIMCMC.h"

#include "ParallelProblem.h"


int main(int argc, char **argv){

  spdlog::set_level(spdlog::level::debug);

  MPI_Init(&argc, &argv);

  pt::ptree pt;
  pt.put("NumSamples_0", 1e3);
  pt.put("NumSamples_1", 5e2);
  pt.put("NumSamples_2", 1e2);
  pt.put("MCMC.burnin", 1e1);
  pt.put("MLMCMC.Subsampling", 5);

  auto comm = std::make_shared<parcer::Communicator>();


  auto componentFactory = std::make_shared<MyMIComponentFactory>(pt);
  StaticLoadBalancingMIMCMC parallelMIMCMC (pt, componentFactory);

  if (comm->GetRank() == 0) {
    spdlog::info("Starting parallel run");
    parallelMIMCMC.Run();
    spdlog::info("Parallel run finished");
    Eigen::VectorXd meanQOI = parallelMIMCMC.MeanQOI();
    std::cout << "mean QOI: " << meanQOI.transpose() << std::endl;
  }
  parallelMIMCMC.Finalize();

  MPI_Finalize();
}