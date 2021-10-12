#include "MUQ/Modeling/Distributions/Gaussian.h"
#include "MUQ/Modeling/Distributions/Density.h"
#include "MUQ/Modeling/Distributions/DensityProduct.h"
#include "MUQ/Modeling/WorkGraph.h"
#include "MUQ/Modeling/ModGraphPiece.h"

#include "MUQ/SamplingAlgorithms/SamplingProblem.h"
#include "MUQ/SamplingAlgorithms/SingleChainMCMC.h"
#include "MUQ/SamplingAlgorithms/MHProposal.h"
#include "MUQ/SamplingAlgorithms/MHKernel.h"
#include "MUQ/SamplingAlgorithms/Diagnostics.h"

#include "MUQ/Utilities/RandomGenerator.h"
#include "MUQ/Modeling/LinearAlgebra/IdentityOperator.h"
#include <boost/property_tree/ptree.hpp>
#include  "ModPiecePredPrey.h"
namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;

int main(){

  Eigen::VectorXd mux(1), varx(1),trueobs(1),noiseStd(1);
  trueobs<<0.8;
  noiseStd<<0.05;
  mux << 1.0;
  varx << 0.2;

  auto prior = std::make_shared<Gaussian>(mux,varx)->AsDensity();

//   Eigen::VectorXd muy(2), vary(2);
//   muy << -1.0, -1.0;
//   vary << 1.0, 2.0;

//   auto py = std::make_shared<Gaussian>(muy,vary)->AsDensity();
  auto mod = std::make_shared<PredPreyModel>(800);
  trueobs = mod->Evaluate(trueobs).at(0);//maybe add some gaussian noise
  auto graph = std::make_shared<WorkGraph>();
  auto likely = std::make_shared<Gaussian>(trueobs,noiseStd*noiseStd)->AsDensity();
  auto in1 = std::make_shared<IdentityOperator>(1);
  graph->AddNode(in1,"in1");
  graph->AddNode(prior, "Prior");
  graph->AddNode(likely, "Likelihood");
  graph->AddNode(mod, "Forward Model");
  graph->AddNode(std::make_shared<DensityProduct>(2), "Posterior");
  graph->AddEdge("in1",0,"Forward Model", 0);
  graph->AddEdge("in1",0,"Prior", 0);
  graph->AddEdge("Prior",0,"Posterior",0);
  graph->AddEdge("Forward Model",0,"Likelihood",0);
  graph->AddEdge("Likelihood",0,"Posterior",1);
  //graph->AddEdge("p(y)",0,"p(x,y)",1);
  graph->Visualize("graph.png");

  auto pxy = graph->CreateModPiece("Posterior");

  // Define the sampling problem as normal
  auto problem = std::make_shared<SamplingProblem>(pxy);

  // Construct two kernels: one for x and one for y
  boost::property_tree::ptree opts;

  // A vector to holding the two transition kernels
  std::vector<std::shared_ptr<TransitionKernel>> kernels(1);

  // Construct the kernel on x
  opts.put("ProposalVariance", 3.0);
  opts.put("BlockIndex", 0); // Specify that this proposal should target x
  auto propx = std::make_shared<MHProposal>(opts, problem);
  kernels.at(0) = std::make_shared<MHKernel>(opts, problem, propx);

//   // Construct the kernel on y
//   opts.put("ProposalVariance", 5.0);
//   opts.put("BlockIndex", 1); // Specify that this proposal should target y
//   auto propy = std::make_shared<MHProposal>(opts, problem);
//   kernels.at(1) = std::make_shared<MHKernel>(opts, problem, propy);

  // Construct the MCMC sampler using this transition kernel
  opts.put("NumSamples", 10000);
  opts.put("BurnIn", 100);
  opts.put("PrintLevel", 3);


  // Run 4 independent chains to help assess convergence
  unsigned int numChains = 4;
  std::vector<std::shared_ptr<SampleCollection>> chains(numChains);
  Eigen::VectorXd startPt(1);

  for(int i=0; i<numChains;++i){
    startPt <<0.5;
   // startPt.at(1) = 2.0*RandomGenerator::GetNormal(2); // Initial point for y

    auto sampler = std::make_shared<SingleChainMCMC>(opts, kernels);
    auto res = sampler->Run(startPt);
    chains.at(i) = res;
    auto sampMat = res->AsMatrix();
    //std::cout<<sampMat<<std::endl;
    std::cout<<"Size: "<<res->size()<<std::endl;
    for(auto & e : res->at(100)->state)
    std::cout<< e<<std::endl;
  }
for(auto & chain:chains){
    std::cout<<chain->Mean()<<std::endl;
}
  // Compute the Rhat convergence diagnostic
  Eigen::VectorXd rhat = Diagnostics::Rhat(chains);
  std::cout << "Rhat = " << rhat.transpose() << std::endl;

  // Estimate the total effective sample size
  Eigen::VectorXd ess = chains.at(0)->ESS();
  for(int i=1; i<numChains; ++i)
    ess += chains.at(i)->ESS();

  std::cout << "ESS: " << ess.transpose() << std::endl;



  return 0;
}
