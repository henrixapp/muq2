#ifndef GAUSSIAN_H_
#define GAUSSIAN_H_

#include "MUQ/Utilities/LinearAlgebra/AnyAlgebra.h"

#include "MUQ/Modeling/Distributions/Distribution.h"

namespace muq {
  namespace Modeling {
    class Gaussian : public Distribution {
    public:

      /// Are we specifying the mean, covariance matrix, or precision matrix
      enum Mode {

        /// We are specifying the covariance
        Covariance,

        /// We are specifying the precision
	      Precision
      };

      enum ExtraInputs {
        None           = 1 << 0,
        Mean           = 1 << 1,
        DiagCovariance = 1 << 2,
        DiagPrecision  = 1 << 3,
        FullCovariance = 1 << 4,
        FullPrecision  = 1 << 5,
      };
      typedef uint8_t InputMask;

      Gaussian(unsigned int dim,
               InputMask    extraInputs = ExtraInputs::None);

      /// Construct a Gaussian with scaled identity covariance/precision
      /**
	      @param[in] mu The mean
      */
      Gaussian(Eigen::VectorXd const& mu,
               InputMask              extraInputs = ExtraInputs::None);

      /// Construct a Gaussian by specifying both the mean and covariance or precision matrix
      /**
      @param[in] mu The mean vector
      @param[in] obj A matrix holding either the covariance or precision
      @param[in] mode A flag indicating whether the matrix should be treated as the covariance or precision
      */
      Gaussian(Eigen::VectorXd const& mu,
               Eigen::MatrixXd const& obj,
               Gaussian::Mode         mode = Gaussian::Mode::Covariance,
               InputMask              extraInputs = ExtraInputs::None);


      virtual ~Gaussian() = default;

      Mode GetMode() const{return mode;};

      unsigned int Dimension() const;

      /// Get the covariance
      /**
	     @return The covariance
      */
      Eigen::MatrixXd GetCovariance() const;
      Eigen::MatrixXd GetPrecision() const;

      /// Get the mean
      /**
        @return A vector holding the distribution mean.
      */
      Eigen::VectorXd const& GetMean() const{return mean;};

      /// Set the mean value
      /**
        @param[in] newMu A vector containing the new mean.  Must be the same size as the current mean.
      */
      void SetMean(Eigen::VectorXd const& newMu);

      /// Set the covariance matrix
      /**
        @param[in] newcov The new covariance
      */
      void SetCovariance(Eigen::MatrixXd const& newCov);

      /// Set the precision matrix
      /**
        @param[in] newprec The new precision
      */
      void SetPrecision(Eigen::MatrixXd const& newPrec);

    private:


      virtual double LogDensityImpl(ref_vector<Eigen::VectorXd> const& inputs) override;

      /// Sample the distribution
      virtual Eigen::VectorXd SampleImpl(ref_vector<Eigen::VectorXd> const& inputs) override;


      /// Compute the distribution's scaling constant
      /**
	      @return Scaling constant
      */
      void ComputeNormalization();

      void ResetHyperparameters(ref_vector<Eigen::VectorXd> const& params);

      static Eigen::VectorXi GetExtraSizes(unsigned dim, InputMask extraInputs);

      static Gaussian::Mode ModeFromExtras(InputMask extraInputs);
      static void CheckInputTypes(InputMask extraInputs, Mode mode);

      /// Have we specified the covariance or the precision
      Gaussian::Mode mode;

      /// What form do the extra inputs take? Just the mean, or the mean and covariance?
      Gaussian::InputMask inputTypes;

      // Space to store the mean of the distribution
      Eigen::VectorXd mean;

      // Space to store either the covariance or precision matrix (depending on mode)
      Eigen::MatrixXd covPrec;

      // Space to tore the matrix square root of the covariance or precision
      Eigen::LLT<Eigen::MatrixXd> sqrtCovPrec;

      /// The scaling constant for the density
      double logNormalization;

    };
  } // namespace Modeling
} // namespace muq

#endif
