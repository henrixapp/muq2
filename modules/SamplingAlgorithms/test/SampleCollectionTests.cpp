#include <gtest/gtest.h>

#include "MUQ/Utilities/RandomGenerator.h"
#include "MUQ/Utilities/AnyHelpers.h"

#include "MUQ/SamplingAlgorithms/SampleCollection.h"


using namespace muq::Utilities;
using namespace muq::SamplingAlgorithms;

class SampleCollectionTest : public::testing::Test {
protected:

    virtual void SetUp() override {

      L.resize(2,2);
      L << 1.0, 0.0,
           1.0, 2.0;

      samps = L * RandomGenerator::GetNormal(2,numSamps);
      weights = RandomGenerator::GetUniform(numSamps);
      weights /= weights.sum();

      for(int i=0; i<numSamps; ++i)
        collection.Add(std::make_shared<SamplingState>(Eigen::VectorXd(samps.col(i)), weights(i)));

    }

    virtual void TearDown() override {
    }

    const int numSamps = 1e4;
    Eigen::MatrixXd L;

    Eigen::MatrixXd samps;
    Eigen::VectorXd weights;
    SampleCollection collection;
};

TEST_F(SampleCollectionTest, Mean)
{
  Eigen::VectorXd mu = collection.Mean(0);

  Eigen::VectorXd trueMu = samps*weights;

  double mcStd = L(0,0)*L(0,0)/sqrt(double(numSamps));
  EXPECT_NEAR(0.0, mu(0), 3*mcStd);
  mcStd = (L(1,0)*L(1,0) + L(1,1)*L(1,1))/sqrt(double(numSamps));
  EXPECT_NEAR(0.0, mu(1), 3*mcStd);

  EXPECT_NEAR(trueMu(0), mu(0), 1e-13);
  EXPECT_NEAR(trueMu(1), mu(1), 1e-13);

  Eigen::VectorXd mu2 = collection.Mean();

  mcStd = L(0,0)*L(0,0)/sqrt(double(numSamps));
  EXPECT_NEAR(0.0, mu2(0), 3*mcStd);
  mcStd = (L(1,0)*L(1,0) + L(1,1)*L(1,1))/sqrt(double(numSamps));
  EXPECT_NEAR(0.0, mu2(1), 3*mcStd);

  EXPECT_NEAR(trueMu(0), mu2(0), 1e-13);
  EXPECT_NEAR(trueMu(1), mu2(1), 1e-13);
}

// Used for timing comparison
TEST_F(SampleCollectionTest, SampMean)
{
  Eigen::VectorXd mu = samps * weights;

  double mcStd = L(0,0)*L(0,0)/sqrt(double(numSamps));
  EXPECT_NEAR(0.0, mu(0), 3*mcStd);
  mcStd = (L(1,0)*L(1,0) + L(1,1)*L(1,1))/sqrt(double(numSamps));
  EXPECT_NEAR(0.0, mu(1), 3*mcStd);
}

TEST_F(SampleCollectionTest, ToMatrix)
{
  Eigen::MatrixXd sampMat = collection.AsMatrix();

  for(int j=0; j<samps.cols(); ++j){
    for(int i=0; i<samps.rows(); ++i)
      EXPECT_DOUBLE_EQ(samps(i,j), sampMat(i,j));
  }
}

TEST_F(SampleCollectionTest, ToWeights)
{
  Eigen::VectorXd sampWeights = collection.Weights();
  for(int i=0; i<weights.size(); ++i)
    EXPECT_DOUBLE_EQ(weights(i), sampWeights(i));
}

TEST_F(SampleCollectionTest, ESS)
{
  double ess = collection.ESS()(0);
  EXPECT_NEAR(std::pow(weights.sum(),2.0) / weights.array().square().sum(), ess, 5e-11);
}


TEST_F(SampleCollectionTest, Variance)
{
  Eigen::VectorXd var = collection.Variance(0);

  Eigen::VectorXd sampMu = samps * weights;
  Eigen::VectorXd sampVar = (samps.colwise() - sampMu).array().pow(2.0).matrix() * weights;

  EXPECT_NEAR(L(0,0)*L(0,0), var(0), 5.0/sqrt(double(numSamps)));
  EXPECT_NEAR(L(1,0)*L(1,0) + L(1,1)*L(1,1), var(1), 50.0/sqrt(double(numSamps)));

  EXPECT_NEAR(sampVar(0), var(0), 1e-13);
  EXPECT_NEAR(sampVar(1), var(1), 1e-13);

  Eigen::VectorXd var2 = collection.Variance();
  EXPECT_NEAR(L(0,0)*L(0,0), var2(0), 5.0/sqrt(double(numSamps)));
  EXPECT_NEAR(L(1,0)*L(1,0) + L(1,1)*L(1,1), var2(1), 50.0/sqrt(double(numSamps)));

}

TEST_F(SampleCollectionTest, Covariance)
{
  Eigen::MatrixXd cov = collection.Covariance(0);

  Eigen::VectorXd sampMu  = samps * weights;
  Eigen::MatrixXd sampCov = (samps.colwise()-sampMu) * weights.asDiagonal() * (samps.colwise()-sampMu).transpose();

  Eigen::MatrixXd trueCov = L*L.transpose();

  EXPECT_NEAR(trueCov(0,0), cov(0,0), 5.0/sqrt(double(numSamps)));
  EXPECT_NEAR(trueCov(0,1), cov(0,1), 10.0/sqrt(double(numSamps)));
  EXPECT_NEAR(trueCov(1,0), cov(1,0), 10.0/sqrt(double(numSamps)));
  EXPECT_NEAR(trueCov(1,1), cov(1,1), 50.0/sqrt(double(numSamps)));

  EXPECT_NEAR(sampCov(0,0), cov(0,0), 1e-13);
  EXPECT_NEAR(sampCov(0,1), cov(0,1), 1e-13);
  EXPECT_NEAR(sampCov(1,0), cov(1,0), 1e-13);
  EXPECT_NEAR(sampCov(1,1), cov(1,1), 1e-13);

  Eigen::MatrixXd cov2 = collection.Covariance();

  EXPECT_NEAR(trueCov(0,0), cov2(0,0), 5.0/sqrt(double(numSamps)));
  EXPECT_NEAR(trueCov(0,1), cov2(0,1), 10.0/sqrt(double(numSamps)));
  EXPECT_NEAR(trueCov(1,0), cov2(1,0), 10.0/sqrt(double(numSamps)));
  EXPECT_NEAR(trueCov(1,1), cov2(1,1), 50.0/sqrt(double(numSamps)));

  EXPECT_NEAR(sampCov(0,0), cov2(0,0), 1e-13);
  EXPECT_NEAR(sampCov(0,1), cov2(0,1), 1e-13);
  EXPECT_NEAR(sampCov(1,0), cov2(1,0), 1e-13);
  EXPECT_NEAR(sampCov(1,1), cov2(1,1), 1e-13);
}
