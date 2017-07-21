#include "MUQ/Approximation/GaussianProcesses/CovarianceKernels.h"
#include "MUQ/Utilities/Exceptions.h"

#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <iostream>

using namespace muq::Approximation;

TEST(Approximation_GP, VectorNorm)
{

    int dim = 100;
    Eigen::VectorXd v1 = Eigen::VectorXd::Random(dim);
    Eigen::VectorXd v2 = Eigen::VectorXd::Random(dim);

    EXPECT_DOUBLE_EQ((v2-v1).norm(), CalcDistance(v1,v2));
}


TEST(Approximation_GP, LinearTransformKernel)
{
    
    const unsigned dim = 2;
    Eigen::MatrixXd sigma2(2,2);
    sigma2 << 1.0, 0.9,
	      0.9, 1.5;
    
    auto kernel = ConstantKernel(dim, sigma2) * SquaredExpKernel(dim, 2.0, 0.35 );

    EXPECT_EQ(2, kernel.coDim);
    
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(3,2);

    auto kernel2 = A*kernel;
    auto kernel3 = A * ConstantKernel(dim, sigma2) * SquaredExpKernel(dim, 2.0, 0.35 );

    Eigen::MatrixXd x1(dim,1);
    x1 << 0.1, 0.4;

    Eigen::MatrixXd x2(dim,1);
    x2 << 0.2, 0.7;
    
    Eigen::MatrixXd result2 = kernel2.Evaluate(x1,x2);
    Eigen::MatrixXd result3 = kernel3.Evaluate(x1,x2);

    Eigen::MatrixXd expected = A * kernel.Evaluate(x1,x2) * A.transpose();

    for(int j=0; j<A.rows(); ++j)
    {
	for(int i=0; i<A.rows(); ++i)
	{
	    EXPECT_NEAR(expected(i,j), result2(i,j), 1e-15);
	    EXPECT_NEAR(expected(i,j), result3(i,j), 1e-15);
	}
    }
}



TEST(Approximation_GP, Clone)
{
    
    const unsigned dim = 2;
    auto kernel = ConstantKernel(dim, 2.0) * SquaredExpKernel(dim, 2.0, 0.35 );

    std::shared_ptr<KernelBase> kernel_copy = kernel.Clone();

    
    EXPECT_DOUBLE_EQ(kernel.inputDim, kernel_copy->inputDim);
    EXPECT_DOUBLE_EQ(kernel.coDim, kernel_copy->coDim);
    EXPECT_DOUBLE_EQ(kernel.numParams, kernel_copy->numParams);

    
    Eigen::VectorXd x1(dim);
    x1 << 0.1, 0.4;
    
    Eigen::VectorXd x2(dim);
    x2 << 0.2, 0.7;

    Eigen::MatrixXd result = kernel.Evaluate(x1,x2);
    Eigen::MatrixXd result_ptr = kernel_copy->Evaluate(x1,x2);
    
    EXPECT_DOUBLE_EQ(result(0,0), result_ptr(0,0));
}


// TEST(Approximation_GP, SeperableProduct)
// {
//     {
        
//         const unsigned dim = 2;
//         auto kernel1 = SquaredExpKernel(dim, 2.0, 3.5) * SquaredExpKernel(dim, 1.0, 0.5);
        
//         auto comps1 = kernel1.GetSeperableComponents();
//         EXPECT_EQ(1, comps1.size());
//     }

//     {
//         std::vector<unsigned> inds1{0};
//         std::vector<unsigned> inds2{1};
//         const unsigned dim = 2;
//         auto kernel2 = SquaredExpKernel(dim, inds1, 2.0, 0.35, {0.1,10} ) * SquaredExpKernel(dim, inds2, 2.0, 0.35, {0.1,10} );
        
//         auto comps2 = kernel2.GetSeperableComponents();
//         EXPECT_EQ(2, comps2.size());
//     }

//     {
//         std::vector<unsigned> inds1{0};
//         std::vector<unsigned> inds2{1,2};
        
//         const unsigned dim = 3;
//         auto kernel2 = SquaredExpKernel(dim, inds1, 2.0, 0.35, {0.1,10} ) * SquaredExpKernel(dim, inds2, 2.0, 0.35, {0.1,10} );

//         auto comps2 = kernel2.GetSeperableComponents();
//         EXPECT_EQ(2, comps2.size());
//     }

//     {
//         std::vector<unsigned> inds1{0,1};
//         std::vector<unsigned> inds2{1,2};
        
//         const unsigned dim = 3;
//         auto kernel2 = SquaredExpKernel(dim, inds1, 2.0, 0.35, {0.1,10} ) * SquaredExpKernel(dim, inds2, 2.0, 0.35, {0.1,10} );

//         auto comps2 = kernel2.GetSeperableComponents();
//         EXPECT_EQ(1, comps2.size());
//     }
        
// }


TEST(Approximation_GP, StateSpaceError)
{
    std::vector<std::shared_ptr<KernelBase>> kernels;
    kernels.push_back( std::make_shared<SquaredExpKernel>(1, 1.0, 1.0) );
    kernels.push_back( std::make_shared<SquaredExpKernel>(2, 1.0, 1.0) );
    
    CoregionalKernel kernel(2, Eigen::MatrixXd::Identity(2,2), kernels);

    EXPECT_THROW(kernel.GetStateSpace(), muq::NotImplementedError);
        
}
