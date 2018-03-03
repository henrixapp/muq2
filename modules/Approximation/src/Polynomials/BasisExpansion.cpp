#include "MUQ/Approximation/Polynomials/BasisExpansion.h"

#include "MUQ/Utilities/MultiIndices/MultiIndexFactory.h"

using namespace muq::Approximation;
using namespace muq::Utilities;

BasisExpansion::BasisExpansion(std::vector<std::shared_ptr<IndexedScalarBasis>> const& basisCompsIn) :
                              BasisExpansion(basisCompsIn,
                                             MultiIndexFactory::CreateTotalOrder(basisCompsIn.size(),0))
{
}

BasisExpansion::BasisExpansion(std::vector<std::shared_ptr<IndexedScalarBasis>> const& basisCompsIn,
                               std::shared_ptr<muq::Utilities::MultiIndexSet>          multisIn) :
                               BasisExpansion(basisCompsIn,
                                              multisIn,
                                              Eigen::MatrixXd::Zero(1,multisIn->Size()))
{
};

BasisExpansion::BasisExpansion(std::vector<std::shared_ptr<IndexedScalarBasis>> const& basisCompsIn,
                               std::shared_ptr<muq::Utilities::MultiIndexSet>          multisIn,
                               Eigen::MatrixXd                                  const& coeffsIn) :
                               basisComps(basisCompsIn),
                               multis(multisIn),
                               coeffs(coeffsIn)
{
  assert(basisComps.size() == multis->GetMultiLength());
  assert(multis->Size() == coeffs.cols());
}

Eigen::VectorXd BasisExpansion::GetAllTerms(Eigen::VectorXd const& x) const{

  // Get the maximum orders
  Eigen::VectorXi maxOrders = multis->GetMaxOrders();

  // Evaluate each dimension up to the maximum order
  std::vector<std::vector<double>> uniEvals(basisComps.size());
  assert(uniEvals.size() == maxOrders.size());

  for(int i=0; i<uniEvals.size(); ++i){
    uniEvals.at(i).resize(maxOrders(i)+1);
    for(int j=0; j<=maxOrders(i); ++j){
      uniEvals.at(i).at(j) = basisComps.at(i)->BasisEvaluate(j, x(i));
    }
  }

  // Now that we have all the univariate terms evaluated, evaluate the expansion
  Eigen::VectorXd allTerms = Eigen::VectorXd::Ones(multis->Size());
  for(int i=0; i<multis->Size(); ++i){

    for(auto it = multis->at(i)->GetNzBegin(); it != multis->at(i)->GetNzEnd(); ++it)
      allTerms(i) *= uniEvals.at(it->first).at(it->second);
  }

  return allTerms;
}

Eigen::MatrixXd BasisExpansion::GetAllDerivs(Eigen::VectorXd const& x) const{

  // Get the maximum orders
  Eigen::VectorXi maxOrders = multis->GetMaxOrders();

  // Evaluate each dimension up to the maximum order
  std::vector<std::vector<double>> uniEvals(basisComps.size());
  std::vector<std::vector<double>> uniDerivs(basisComps.size());
  assert(uniEvals.size() == maxOrders.size());

  for(int i=0; i<uniEvals.size(); ++i){
    uniEvals.at(i).resize(maxOrders(i)+1);
    uniDerivs.at(i).resize(maxOrders(i)+1);

    for(int j=0; j<=maxOrders(i); ++j){
      uniEvals.at(i).at(j) = basisComps.at(i)->BasisEvaluate(j, x(i));
      uniDerivs.at(i).at(j) = basisComps.at(i)->DerivativeEvaluate(j, 1, x(i));
    }
  }

  // Now that we have all the univariate terms evaluated, evaluate the expansion
  Eigen::MatrixXd allDerivs = Eigen::MatrixXd::Ones(multis->Size(),x.size());
  for(int i=0; i<multis->Size(); ++i){

    // Loop over each dimension
    for(int j=0; j<x.size(); ++j){
      if(multis->at(i)->GetValue(j)==0){
        allDerivs(i,j) = 0;
      }else{
        for(auto it = multis->at(i)->GetNzBegin(); it != multis->at(i)->GetNzEnd(); ++it){

          if(it->first == j){
            allDerivs(i,j) *= uniDerivs.at(it->first).at(it->second);
          }else{
            allDerivs(i,j) *= uniEvals.at(it->first).at(it->second);
          }

        }
      }
    }
  }

  return allDerivs;
}

std::vector<Eigen::MatrixXd> BasisExpansion::GetHessians(Eigen::VectorXd const& x) const{

  // Get the maximum orders
  Eigen::VectorXi maxOrders = multis->GetMaxOrders();

  // Evaluate each dimension up to the maximum order
  std::vector<std::vector<double>> uniEvals(basisComps.size());
  std::vector<std::vector<double>> uniD1(basisComps.size()); // first derivatives
  std::vector<std::vector<double>> uniD2(basisComps.size()); // second derivatives

  assert(uniEvals.size() == maxOrders.size());

  for(int i=0; i<uniEvals.size(); ++i){
    uniEvals.at(i).resize(maxOrders(i)+1);
    uniD1.at(i).resize(maxOrders(i)+1);
    uniD2.at(i).resize(maxOrders(i)+1);

    for(int j=0; j<=maxOrders(i); ++j){
      uniEvals.at(i).at(j) = basisComps.at(i)->BasisEvaluate(j, x(i));
      uniD1.at(i).at(j) = basisComps.at(i)->DerivativeEvaluate(j, 1, x(i));
      uniD2.at(i).at(j) = basisComps.at(i)->DerivativeEvaluate(j, 2, x(i));
    }
  }

  std::vector<Eigen::MatrixXd> hessians(coeffs.rows(), Eigen::MatrixXd::Zero(x.size(),x.size()));

  // Loop over each term in the expansion
  for(int i=0; i<multis->Size(); ++i){

    // Loop over each dimension
    for(int j=0; j<x.size(); ++j){
      for(int k=0; k<=j; ++k){
        if((multis->at(i)->GetValue(j)!=0) && ((multis->at(i)->GetValue(k)!=0))){

          double tempVal = 1.0;

          for(auto it = multis->at(i)->GetNzBegin(); it != multis->at(i)->GetNzEnd(); ++it){

            if((j==k) && (it->first == j)){
              tempVal *= uniD2.at(it->first).at(it->second);
            }else if((it->first == j)||(it->first == k)){
              tempVal *= uniD1.at(it->first).at(it->second);
            }else{
              tempVal *= uniEvals.at(it->first).at(it->second);
            }
          }

          // Add the results into each of the hessians matrices
          for(int kk=0; kk<coeffs.rows(); ++kk)
            hessians.at(kk)(j,k) += coeffs(kk,i)*tempVal;
        }
      }
    }
  }

  // make sure all the hessians are symmetric
  for(int kk=0; kk<coeffs.rows(); ++kk)
    hessians.at(kk).triangularView<Eigen::Upper>() =  hessians.at(kk).triangularView<Eigen::Lower>().transpose();

  return hessians;

}

Eigen::VectorXd const& BasisExpansion::ProcessInputs(muq::Modeling::ref_vector<boost::any> const& inputs)
{
    // If there are two inputs, then the second input will reset the coefficients
  if(inputs.size()>1){
      Eigen::MatrixXd const& newCoeffs = *boost::any_cast<Eigen::MatrixXd>(&inputs.at(1).get());
      assert(newCoeffs.rows() == coeffs.rows());
      assert(newCoeffs.cols() == coeffs.cols());

      coeffs = newCoeffs;
  }else if(inputs.size()==0){
    throw std::logic_error("Could not evaluate BasisExpansion because no input point was provided.  BasisExpansion::EvaluateImpl requires at least 1 input.");
  }

  // Extract the point where we want to evaluate the expansion

  Eigen::VectorXd const& x = *boost::any_cast<Eigen::VectorXd>(&inputs.at(0).get());
  return x;
}


void BasisExpansion::EvaluateImpl(muq::Modeling::ref_vector<boost::any> const& inputs) {

  Eigen::VectorXd const& x = ProcessInputs(inputs);

  // Compute the output
  outputs.resize(1);
  outputs.at(0) = (coeffs*GetAllTerms(x)).eval();

}

void BasisExpansion::JacobianImpl(unsigned int const                           wrtIn,
                                  unsigned int const                           wrtOut,
                                  muq::Modeling::ref_vector<boost::any> const& inputs)
{
  assert(wrtOut==0);

  Eigen::VectorXd const& x = ProcessInputs(inputs);

  if(wrtIn==0){
    jacobian = Eigen::MatrixXd(coeffs*GetAllDerivs(x));
  }else if(wrtIn==1){
    jacobian = Eigen::MatrixXd(GetAllTerms(x).transpose().replicate(coeffs.rows(),1));
  }
}


Eigen::MatrixXd BasisExpansion::SecondDerivative(unsigned                       outputDim,
                                                 unsigned                       derivDim1,
                                                 unsigned                       derivDim2,
                                                 Eigen::VectorXd         const& x,
                                                 Eigen::MatrixXd         const& newCoeffs)
{
  SetCoeffs(newCoeffs);
  return SecondDerivative(outputDim, derivDim1, derivDim2, x);
}

Eigen::MatrixXd BasisExpansion::SecondDerivative(unsigned                       outputDim,
                                                 unsigned                       derivDim1,
                                                 unsigned                       derivDim2,
                                                 Eigen::VectorXd         const& x)
{
  if((derivDim1==0) && (derivDim2==1)){
    return GetAllDerivs(x).transpose();
  }else if((derivDim1==1) && (derivDim2==0)){
    return GetAllDerivs(x);
  }else if(derivDim1==0){
    return GetHessians(x).at(outputDim);
  }else{
    return Eigen::MatrixXd::Zero(coeffs.cols(), coeffs.cols());
  }
}


Eigen::MatrixXd BasisExpansion::GetCoeffs() const{
  return coeffs;
}

void BasisExpansion::SetCoeffs(Eigen::MatrixXd const& allCoeffs){
  assert(coeffs.rows()==allCoeffs.rows());
  assert(coeffs.cols()==allCoeffs.cols());
  coeffs = allCoeffs;
}