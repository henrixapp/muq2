#include <gtest/gtest.h>

#include <boost/property_tree/ptree.hpp>

#include "MUQ/SamplingAlgorithms/MonteCarlo.h"

namespace pt = boost::property_tree;
using namespace muq::SamplingAlgorithms;

TEST(MonteCarlo, Setup) {
  // create an instance of Monte Carlo
  auto mc = std::make_shared<MonteCarlo>();

  // parameters for the sampler
  pt::ptree pt;
  pt.put<unsigned int>("SamplingAlgorithm.NumSamples", 100); // number of Monte Carlo samples

  // evaluate
  mc->Evaluate(pt);
}
