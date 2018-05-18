#ifndef DENSITY_H
#define DENSITY_H

#include "MUQ/Modeling/Distributions/Distribution.h"
#include "MUQ/Modeling/ModPiece.h"

namespace muq{
  namespace Modeling{

    class Density : public Distribution, public ModPiece{

    public:
      Density(std::shared_ptr<Distribution> distIn);

      virtual ~Density() = default;

    protected:
      std::shared_ptr<Distribution> dist;

      virtual void EvaluateImpl(ref_vector<Eigen::VectorXd> const& inputs) override;

      virtual double LogDensityImpl(ref_vector<Eigen::VectorXd> const& inputs) override;

      virtual Eigen::VectorXd GradLogDensityImpl(unsigned int wrt, ref_vector<Eigen::VectorXd> const& inputs) override;
      virtual Eigen::VectorXd SampleImpl(ref_vector<Eigen::VectorXd> const& inputs) override;

      static Eigen::VectorXi GetInputSizes(std::shared_ptr<Distribution> distIn);

      void GradientImpl(unsigned int                const  outputDimWrt,
                        unsigned int                const  inputDimWrt,
                        ref_vector<Eigen::VectorXd> const& input,
                        Eigen::VectorXd             const& sensitivity) override;

      void JacobianImpl(unsigned int                const  outputDimWrt,
                        unsigned int                const  inputDimWrt,
                        ref_vector<Eigen::VectorXd> const& input) override;

      void ApplyJacobianImpl(unsigned int                const  outputDimWrt,
                             unsigned int                const  inputDimWrt,
                             ref_vector<Eigen::VectorXd> const& input,
                             Eigen::VectorXd             const& vec) override;
    }; // class Density

  } // namespace Modeling
} // namespace muq



#endif // #ifndef DENSITY_H