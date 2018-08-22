#include "MUQ/config.h"

#if MUQ_HAS_MPI

#include <gtest/gtest.h>

#include "MUQ/Utilities/RandomGenerator.h"

#include "MUQ/SamplingAlgorithms/DistributedCollection.h"

using namespace muq::Utilities;
using namespace muq::SamplingAlgorithms;

class DistributedCollectionTest : public::testing::Test {
protected:

  inline virtual void SetUp() override {
    L.resize(2,2);
    L << 1.0, 0.0,
      1.0, 2.0;

    {
      // create the local collection
      auto local = std::make_shared<SampleCollection>();

      // the default communicator is MPI_COMM_WORLD
      auto comm = std::make_shared<parcer::Communicator>();
      rank = comm->GetRank();
      nproc = comm->GetSize();
      
      // create the global collection
      collection = std::make_shared<DistributedCollection>(local, comm);
    }

    expectedSize = 0;
    for( unsigned int i=0; i<numSamps; ++i ) {
      if( i%nproc!=rank ) { continue; }

      auto state = std::make_shared<SamplingState>(L*RandomGenerator::GetNormal(2), RandomGenerator::GetUniform());
      state->meta["rank"] = rank;
      state->meta["id"] = i;
      
      collection->Add(state);
      ++expectedSize;
    }
  }
  
  inline virtual void TearDown() override {
    int cnt = 0;
    for( unsigned int i=0; i<numSamps; ++i ) {
      if( i%nproc!=rank ) { continue; }
	    
      auto state = collection->at(cnt++);
      EXPECT_EQ(boost::any_cast<unsigned int const>(state->meta["id"]), i);
      EXPECT_EQ(boost::any_cast<int const>(state->meta["rank"]), rank);
    }
  }
  
  const int numSamps = 1.0e5;
  Eigen::MatrixXd L;

  int rank;
  int nproc;

  int expectedSize;
  
  std::shared_ptr<DistributedCollection> collection;
};

TEST_F(DistributedCollectionTest, SizeTest) {
  EXPECT_EQ(collection->LocalSize(), expectedSize);
  EXPECT_EQ(collection->GlobalSize(), numSamps);
  EXPECT_EQ(collection->size(), numSamps);
}

TEST_F(DistributedCollectionTest, CentralMoment) {
  EXPECT_NEAR(collection->LocalCentralMoment(1).norm(), 0.0, 1.0e-5);
  EXPECT_NEAR(collection->GlobalCentralMoment(1).norm(), 0.0, 1.0e-5);
  EXPECT_NEAR(collection->CentralMoment(1).norm(), 0.0, 1.0e-5);
}

TEST_F(DistributedCollectionTest, Mean) {
  EXPECT_NEAR(collection->LocalMean().norm(), 0.0, 1.0e-1);
  EXPECT_NEAR(collection->GlobalMean().norm(), 0.0, 1.0e-1);
  EXPECT_NEAR(collection->Mean().norm(), 0.0, 1.0e-1);
}

TEST_F(DistributedCollectionTest, Variance) {
  const Eigen::VectorXd& truth = (L*L.transpose()).diagonal();
  
  EXPECT_NEAR((collection->LocalVariance()-truth).norm(), 0.0, 1.0e-1);
  EXPECT_NEAR((collection->GlobalVariance()-truth).norm(), 0.0, 1.0e-1);
  EXPECT_NEAR((collection->Variance()-truth).norm(), 0.0, 1.0e-1);
}

TEST_F(DistributedCollectionTest, Covariance) {
  const Eigen::MatrixXd& truth = L*L.transpose();
  
  EXPECT_NEAR((collection->LocalCovariance()-truth).norm(), 0.0, 1.0e-1);
  EXPECT_NEAR((collection->GlobalCovariance()-truth).norm(), 0.0, 1.0e-1);
  EXPECT_NEAR((collection->Covariance()-truth).norm(), 0.0, 1.0e-1);
}

TEST_F(DistributedCollectionTest, ESS) {
  const Eigen::VectorXd& less = collection->LocalESS();
  const Eigen::VectorXd& gess = collection->GlobalESS();
  const Eigen::VectorXd& ess = collection->ESS();

  EXPECT_EQ(less.size(), ess.size());
  EXPECT_EQ(gess.size(), ess.size());

  for( unsigned int i=0; i<ess.size(); ++i ) {
    EXPECT_TRUE(less(i)>0);
    EXPECT_TRUE(gess(i)>0);
    EXPECT_TRUE(ess(i)>0);
  }
}

TEST_F(DistributedCollectionTest, AsMatrix) {
  const Eigen::MatrixXd& localMat = collection->AsLocalMatrix();
  EXPECT_EQ(localMat.rows(), 2);
  EXPECT_EQ(localMat.cols(), expectedSize);

  const Eigen::MatrixXd& globalMat = collection->AsGlobalMatrix();
  EXPECT_EQ(globalMat.rows(), 2);
  EXPECT_EQ(globalMat.cols(), numSamps);

  const Eigen::MatrixXd& mat = collection->AsMatrix();
  EXPECT_EQ(mat.rows(), 2);
  EXPECT_EQ(mat.cols(), numSamps);
}

TEST_F(DistributedCollectionTest, Weights) {
  const Eigen::VectorXd& localWeights = collection->LocalWeights();
  EXPECT_EQ(localWeights.size(), expectedSize);

  const Eigen::MatrixXd& globalWeights = collection->GlobalWeights();
  EXPECT_EQ(globalWeights.size(), numSamps);

  const Eigen::MatrixXd& weights = collection->Weights();
  EXPECT_EQ(weights.size(), numSamps);
}

TEST_F(DistributedCollectionTest, WriteToFile) {
  const std::string filename = "output.h5";
  collection->WriteToFile(filename);

  if( rank==0 ) { 
    auto hdf5file = std::make_shared<HDF5File>(filename);
    
    const Eigen::MatrixXd samples = hdf5file->ReadMatrix("/samples");
    const Eigen::RowVectorXd weights = hdf5file->ReadMatrix("/weights");
    const Eigen::RowVectorXd id = hdf5file->ReadMatrix("/id");
    const Eigen::RowVectorXd rnk = hdf5file->ReadMatrix("/rank");
    
    EXPECT_EQ(id.size(), numSamps);
    EXPECT_EQ(rnk.size(), numSamps);
    EXPECT_EQ(weights.size(), numSamps);
    EXPECT_EQ(samples.cols(), numSamps);
    EXPECT_EQ(samples.rows(), 2);

    std::remove(filename.c_str());
  }
}

#endif // end MUQ_HAS_MPI