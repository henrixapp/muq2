#include "AllClassWrappers.h"

#include "MUQ/config.h"

#include "MUQ/SamplingAlgorithms/MIMCMC.h"
#include "MUQ/SamplingAlgorithms/SingleChainMCMC.h"
#include "MUQ/SamplingAlgorithms/SamplingAlgorithm.h"
#include "MUQ/SamplingAlgorithms/MCMCFactory.h"
#include "MUQ/SamplingAlgorithms/MHKernel.h"

#include "MUQ/Utilities/PyDictConversion.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>

#include <string>

#include <functional>
#include <vector>

using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;
namespace py = pybind11;

#include "MUQ/Modeling/Distributions/Gaussian.h"
#include "MUQ/SamplingAlgorithms/CrankNicolsonProposal.h"
#include "MUQ/SamplingAlgorithms/SubsamplingMIProposal.h"

using namespace muq::Modeling;


// In the long run, this is to be replaced by ConcatenatingInterpolation to be introduced along with parallel MIMCMC
class PyConcatenatingInterpolation : public MIInterpolation {
public:
  PyConcatenatingInterpolation(std::shared_ptr<MultiIndex> const& index) : index(index) {
  }

  virtual std::shared_ptr<SamplingState> Interpolate (std::shared_ptr<SamplingState> const& coarseProposal, std::shared_ptr<SamplingState> const& fineProposal) override {
    int fine_part_size = fineProposal->state[0].size() - coarseProposal->state[0].size();

    Eigen::VectorXd interpolatedState(fineProposal->state[0].size());
    interpolatedState << coarseProposal->state[0], fineProposal->state[0].tail(fine_part_size);

    return std::make_shared<SamplingState>(interpolatedState);
  }

private:
  std::shared_ptr<MultiIndex> index;
};


class PythonMIComponentFactory : public MIComponentFactory {
public:
  PythonMIComponentFactory(pt::ptree pt, Eigen::VectorXd startingPoint, std::vector<std::shared_ptr<AbstractSamplingProblem>> const& pySamplingProblems)
   : pt(pt), startingPoint(startingPoint), pySamplingProblems(pySamplingProblems) {}

  /*PythonMIComponentFactory(pt::ptree pt, Eigen::VectorXd startingPoint, py::array_t<std::shared_ptr<AbstractSamplingProblem>> const& pySamplingProblems)
   : pt(pt), startingPoint(startingPoint) {
     auto buf_problems = pySamplingProblems.request();
  }*/


  virtual std::shared_ptr<MCMCProposal> Proposal (std::shared_ptr<MultiIndex> const& index, std::shared_ptr<AbstractSamplingProblem> const& samplingProblem) override {


    boost::property_tree::ptree subTree = pt.get_child("Proposal");
    subTree.put("BlockIndex",0);
    //subTree.put("BlockIndex", blockInd);

    // Construct the proposal
    std::shared_ptr<MCMCProposal> proposal = MCMCProposal::Construct(subTree, samplingProblem);
    assert(proposal);
    return proposal;
  }

  virtual std::shared_ptr<MultiIndex> FinestIndex() override {
    auto index = std::make_shared<MultiIndex>(1);
    index->SetValue(0, pySamplingProblems.size() - 1);
    //index->SetValue(1, 2);
    return index;
  }

  virtual std::shared_ptr<MCMCProposal> CoarseProposal (std::shared_ptr<MultiIndex> const& index,
                                                        std::shared_ptr<AbstractSamplingProblem> const& coarseProblem,
                                                           std::shared_ptr<SingleChainMCMC> const& coarseChain) override {
    pt::ptree ptProposal;
    ptProposal.put("BlockIndex",0);
    ptProposal.put("subsampling", pt.get<int>("Subsampling"));
    return std::make_shared<SubsamplingMIProposal>(ptProposal, coarseProblem, coarseChain);
  }

  virtual std::shared_ptr<AbstractSamplingProblem> SamplingProblem (std::shared_ptr<MultiIndex> const& index) override {
    return pySamplingProblems[index->GetValue(0)];
  }

  virtual std::shared_ptr<MIInterpolation> Interpolation (std::shared_ptr<MultiIndex> const& index) override {
    return std::make_shared<PyConcatenatingInterpolation>(index);
  }

  virtual Eigen::VectorXd StartingPoint (std::shared_ptr<MultiIndex> const& index) override {
    return startingPoint;
  }
private:
  pt::ptree pt;
  Eigen::VectorXd startingPoint;
  std::vector<std::shared_ptr<AbstractSamplingProblem>> pySamplingProblems;
};


void PythonBindings::MCMCWrapper(py::module &m) {
  py::class_<SamplingAlgorithm, std::shared_ptr<SamplingAlgorithm>> sampAlg(m, "SamplingAlgorithm");
  sampAlg
    .def("Run", (std::shared_ptr<SampleCollection>  (SamplingAlgorithm::*)(std::vector<Eigen::VectorXd> const&)) &SamplingAlgorithm::Run,
                 py::call_guard<py::scoped_ostream_redirect,py::scoped_estream_redirect>())
    .def("GetSamples", &SamplingAlgorithm::GetSamples);

  py::class_<SingleChainMCMC, SamplingAlgorithm, std::shared_ptr<SingleChainMCMC>> singleMCMC(m, "SingleChainMCMC");
  singleMCMC
    .def(py::init( [](py::dict d, std::shared_ptr<AbstractSamplingProblem> problem) {return new SingleChainMCMC(ConvertDictToPtree(d), problem);}))
    .def(py::init( [](py::dict d, std::vector<std::shared_ptr<TransitionKernel>> kernels) {return new SingleChainMCMC(ConvertDictToPtree(d), kernels);}))
    .def("Kernels", &SingleChainMCMC::Kernels)
    .def("RunImpl", &SingleChainMCMC::RunImpl)
    .def("AddNumSamps", &SingleChainMCMC::AddNumSamps)
    .def("NumSamps", &SingleChainMCMC::NumSamps);

  py::class_<MIMCMC, SamplingAlgorithm, std::shared_ptr<MIMCMC>> multiindexMCMC(m, "MIMCMC");
  multiindexMCMC
    .def(py::init( [](py::dict d, Eigen::VectorXd startingPoint, std::vector<std::shared_ptr<AbstractSamplingProblem>> problems) {return new MIMCMC(ConvertDictToPtree(d), std::make_shared<PythonMIComponentFactory>(ConvertDictToPtree(d), startingPoint, problems)); }))
    //.def(py::init( [](py::dict d, Eigen::VectorXd startingPoint, py::array_t<std::shared_ptr<AbstractSamplingProblem>> problems) {return new MIMCMC(ConvertDictToPtree(d), std::make_shared<PythonMIComponentFactory>(ConvertDictToPtree(d), startingPoint, problems)); }))

    .def("RunImpl", &MIMCMC::RunImpl)
    .def("MeanParam", &MIMCMC::MeanParam);

  py::class_<MCMCFactory, std::shared_ptr<MCMCFactory>> fact(m, "MCMCFactory");
  fact
    .def_static("CreateSingleChain", [](py::dict d, std::shared_ptr<AbstractSamplingProblem> problem) {return MCMCFactory::CreateSingleChain(ConvertDictToPtree(d), problem);},
                py::call_guard<py::scoped_ostream_redirect,py::scoped_estream_redirect>() );


}
