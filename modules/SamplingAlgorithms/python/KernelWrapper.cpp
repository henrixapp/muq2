#include "AllClassWrappers.h"

//#include "MUQ/SamplingAlgorithms/ISKernel.h"
//#include "MUQ/SamplingAlgorithms/MCKernel.h"
#include "MUQ/SamplingAlgorithms/MHKernel.h"
#include "MUQ/SamplingAlgorithms/TransitionKernel.h"

#include "MUQ/Utilities/PyDictConversion.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <string>

#include <functional>
#include <vector>

using namespace muq::SamplingAlgorithms::PythonBindings;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;
namespace py = pybind11;


void muq::SamplingAlgorithms::PythonBindings::KernelWrapper(py::module &m)
{


  py::class_<TransitionKernel, std::shared_ptr<TransitionKernel>> transKern(m, "TransitionKernel");
  transKern
    .def_static("Construct", [](py::dict d, std::shared_ptr<AbstractSamplingProblem> problem)->std::shared_ptr<TransitionKernel>{return TransitionKernel::Construct(ConvertDictToPtree(d), problem);})
    .def("PreStep", &TransitionKernel::PreStep)
    .def("PostStep", &TransitionKernel::PostStep)
    .def("Step", &TransitionKernel::Step)
    .def_readonly("blockInd", &TransitionKernel::blockInd);


  py::class_<MHKernel, TransitionKernel, std::shared_ptr<MHKernel>> mhKern(m, "MHKernel");
  mhKern
    .def("__init__", [](MHKernel &instance, py::dict d, std::shared_ptr<AbstractSamplingProblem> problem) {new (&instance) MHKernel(ConvertDictToPtree(d), problem);})
    .def("Proposal", &MHKernel::Proposal)
    .def("PostStep", &MHKernel::PostStep)
    .def("Step", &MHKernel::Step);

}
