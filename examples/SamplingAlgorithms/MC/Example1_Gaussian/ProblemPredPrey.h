#include "MUQ/SamplingAlgorithms/ParallelizableMIComponentFactory.h"
#include "ModPiecePredPrey.h"
#include "MUQ/Modeling/WorkGraph.h"
#include "MUQ/Modeling/ModGraphPiece.h"
#include "MUQ/Modeling/LinearAlgebra/IdentityOperator.h"
#include "MUQ/Modeling/Distributions/Density.h"
#include "MUQ/Modeling/Distributions/DensityProduct.h"

class MySamplingProblem : public AbstractSamplingProblem {
public:
  MySamplingProblem(std::shared_ptr<muq::Modeling::ModGraphPiece> targetIn)
   : AbstractSamplingProblem(Eigen::VectorXi::Constant(1,1), Eigen::VectorXi::Constant(1,1)),
     target(targetIn){
       //std::cout<<"TEST";

     }

  virtual ~MySamplingProblem() = default;


  virtual double LogDensity(std::shared_ptr<SamplingState> const& state) override {
    lastState = state;
    //std::cout<<"THIS"<<target->Evaluate(state->state).at(0)(0)<<std::endl;
    return target->Evaluate(state->state).at(0)(0);
  };

  virtual std::shared_ptr<SamplingState> QOI() override {
    assert (lastState != nullptr);
    return std::make_shared<SamplingState>(lastState->state, 1.0);
  }

private:
  std::shared_ptr<SamplingState> lastState = nullptr;

  std::shared_ptr<muq::Modeling::ModGraphPiece> target;

};


class MyInterpolation : public MIInterpolation {
public:
  std::shared_ptr<SamplingState> Interpolate (std::shared_ptr<SamplingState> const& coarseProposal, std::shared_ptr<SamplingState> const& fineProposal) {
    return std::make_shared<SamplingState>(coarseProposal->state);
  }
};

class MyMIComponentFactory : public MIComponentFactory {
public:
  MyMIComponentFactory (pt::ptree pt)
   : pt(pt)
  { }

  virtual std::shared_ptr<MCMCProposal> Proposal (std::shared_ptr<MultiIndex> const& index, std::shared_ptr<AbstractSamplingProblem> const& samplingProblem) override {
    pt::ptree pt;
    pt.put("BlockIndex",0);

    Eigen::VectorXd mu(1);
    mu << 1.0;
    Eigen::MatrixXd cov(1,1);
    cov << 0.04;
   // std::cerr<<"P1"<<std::endl;
    auto prior = std::make_shared<Gaussian>(mu, cov);

    return std::make_shared<MHProposal>(pt, samplingProblem);//Changed this line,removed prior
  }

  virtual std::shared_ptr<MultiIndex> FinestIndex() override {
    auto index = std::make_shared<MultiIndex>(1);
    index->SetValue(0, 3);
    return index;
  }

  virtual std::shared_ptr<MCMCProposal> CoarseProposal (std::shared_ptr<MultiIndex> const& fineIndex,
                                                        std::shared_ptr<MultiIndex> const& coarseIndex,
                                                        std::shared_ptr<AbstractSamplingProblem> const& coarseProblem,
                                                           std::shared_ptr<SingleChainMCMC> const& coarseChain) override {
    pt::ptree ptProposal = pt;
    ptProposal.put("BlockIndex",0);
    return std::make_shared<SubsamplingMIProposal>(ptProposal, coarseProblem, coarseIndex, coarseChain);
  }

  virtual std::shared_ptr<AbstractSamplingProblem> SamplingProblem (std::shared_ptr<MultiIndex> const& index) override {
    Eigen::VectorXd mu(1);
    mu << 1.0;
    Eigen::MatrixXd cov(1,1);
    cov << 0.04;
/*
    if (index->GetValue(0) == 0) {
      mu *= 0.8;
      cov *= 2.0;
    } else if (index->GetValue(0) == 1) {
      mu *= 0.9;
      cov *= 1.5;
    } else if (index->GetValue(0) == 2) {
      mu *= 0.99;
      cov *= 1.1;
    } else if (index->GetValue(0) == 3) {
      mu *= 1.0;
      cov *= 1.0;
    } else {
      std::cerr << "Sampling problem not defined!" << std::endl;
      assert (false);
    }
*/
//std::cout<<"index->GetValue(0):"<<index->GetValue(0)<<std::endl;
 // std::cout<<"Resolution: "<<pow(2,index->GetValue(0))*50<<std::endl;
    auto  mod = std::make_shared<PredPreyModel>(pow(2,index->GetValue(0))*50);
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
  //auto mod = std::make_shared<PredPreyModel>(800);
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
  //graph->Visualize("graph.png");

  auto pxy = graph->CreateModPiece("Posterior");

  // Define the sampling problem as normal

    return std::make_shared<MySamplingProblem>(pxy);
  }

  virtual std::shared_ptr<MIInterpolation> Interpolation (std::shared_ptr<MultiIndex> const& index) override {
    return std::make_shared<MyInterpolation>();
  }

  virtual Eigen::VectorXd StartingPoint (std::shared_ptr<MultiIndex> const& index) override {
    Eigen::VectorXd mu(1);
    mu << 0.5;
    return mu;
  }

private:
  pt::ptree pt;
};


