#include "MUQ/Approximation/Quadrature/GaussQuadrature.h"

using namespace muq::Approximation;

GaussQuadrature::GaussQuadrature() {}

GaussQuadrature::GaussQuadrature(std::shared_ptr<OrthogonalPolynomial> polyIn,
                                 int polyOrderIn)
                                : poly(polyIn), polyOrder(polyOrderIn)
{
  Compute();
}

void GaussQuadrature::Compute() {

  // Create diagonal and subdiagonal vectors
  Eigen::VectorXd diag = Eigen::VectorXd::Zero(polyOrder);
  Eigen::VectorXd subdiag = Eigen::VectorXd::Zero(polyOrder);

  for (unsigned int i=1; i<polyOrder+1; i++) {

    // Calling ak, bk, ad ck correctly?
    double alpha_i = -poly->bk(i)/poly->ak(i);
    double beta_i = std::sqrt(poly->ck(i+1)/(poly->ak(i)*poly->ak(i+1)));

    // Diagonal entry of J
    diag(i-1) = alpha_i;

    // Off diagonal entries of J
    if (i < polyOrder)
      subdiag(i-1) = beta_i;

  }

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
  es.computeFromTridiagonal(diag, subdiag);

  // Set gauss points
  gaussPts = es.eigenvalues();

  // Get mu_0 value (integral of weighting function)
  double mu0 = poly->Normalization(0);

  // Set gauss weights
  gaussWts = mu0*es.eigenvectors().row(0).array().square();

}

Eigen::VectorXd const& GaussQuadrature::Points() const {

  return gaussPts;

}

Eigen::VectorXd const& GaussQuadrature::Weights() const {

  return gaussWts;

}
