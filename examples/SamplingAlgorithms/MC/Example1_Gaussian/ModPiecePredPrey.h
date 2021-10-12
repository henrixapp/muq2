#include "MUQ/Modeling/ModPiece.h"
#include "Euler.hh"
const double initialPreyPop = 1.0;
class PredPreyModel : public muq::Modeling::ModPiece
{
    int numSteps_;
public:
  PredPreyModel(int numSteps) : muq::Modeling::ModPiece({1},{1}),
  numSteps_(numSteps) {
  };//one dim

protected:
  void EvaluateImpl(muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs) override
  {
    //std::cout<<"THaT"<<std::endl;
    Eigen::VectorXd const& theta = inputs.at(0).get();
    Eigen::VectorXd   input(2);
    //std::cout<<"THaT"<<theta[0]<<std::endl;
    input<<initialPreyPop,theta;
    double a,b,c,d;
    a=0.5;
    b=c=d=a;
    auto F = [a,b,c,d](const State& s) -> State{
        State s2(2);
        s2<<a*s[0]-b*s[1]*s[0],d*s[0]*s[1]-c*s[1];
        return s2;
    };
    auto res = explicit_euler(F,input,15.0/(double)numSteps_,numSteps_);
    outputs.resize(1);
    State oneD(1);
    oneD<<res[0];
    outputs.at(0) = oneD;
  };

  virtual void JacobianImpl(unsigned int outWrt,
                            unsigned int inWrt,
                            muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs) override
  {
    Eigen::VectorXd const& x = inputs.at(0).get();
    Eigen::VectorXd const& c = inputs.at(1).get();

    // Jacobian wrt x
    if(inWrt==0){
      jacobian = c(0)*Eigen::VectorXd::Identity(x.size(), x.size());

    // Jacobian wrt c
    }else{
      jacobian = Eigen::MatrixXd::Ones(outputSizes(0), inputSizes(inWrt));
      jacobian.col(0) = x;
    }
  }

  virtual void GradientImpl(unsigned int outWrt,
                            unsigned int inWrt,
                            muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs,
                            Eigen::VectorXd const& sens) override
  {
    Eigen::VectorXd const& x = inputs.at(0).get();
    Eigen::VectorXd const& c = inputs.at(1).get();

    // Gradient wrt x
    if(inWrt==0){
      gradient = c(0) * sens;

    // Gradient wrt c
    }else{
      gradient.resize(2);
      gradient(0) = x.dot(sens);
      gradient(1) = sens.sum();
    }
  }

  virtual void ApplyJacobianImpl(unsigned int outWrt,
                                 unsigned int inWrt,
                                 muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs,
                                 Eigen::VectorXd const& vec) override
  {
    Eigen::VectorXd const& x = inputs.at(0).get();
    Eigen::VectorXd const& c = inputs.at(1).get();

    // Jacobian wrt x
    if(inWrt==0){
      jacobianAction = c(0)*vec;

    // Jacobian wrt c
    }else{
      jacobianAction = vec(0)*x + vec(1)*Eigen::VectorXd::Ones(x.size());
    }
  }

  virtual void ApplyHessianImpl(unsigned int outWrt,
                                 unsigned int inWrt1,
                                 unsigned int inWrt2,
                                 muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs,
                                 Eigen::VectorXd const& sens,
                                 Eigen::VectorXd const& vec) override
  {
    Eigen::VectorXd const& x = inputs.at(0).get();
    Eigen::VectorXd const& c = inputs.at(1).get();

    // Apply d^2 / dxdc
    if((inWrt1==0)&&(inWrt2==1)){
      hessAction = vec(0) * sens;

    // Apply d^2 / dcdx
    }else if((inWrt2==0)&&(inWrt1==1)){
      hessAction.resize(2);
      hessAction(0) = sens.dot(vec);
      hessAction(1) = 0;

    // Apply d^2 / dxds
    }else if((inWrt1==0)&&(inWrt2==2)){
      hessAction = c(0) * vec;

    // Apply d^2 / dcds
    }else if((inWrt1==1)&&(inWrt2==2)){

      hessAction.resize(2);
      hessAction(0) = x.dot(vec);
      hessAction(1) = vec.sum();

    // Apply d^2/dx^2  or  d^2/dc^2  or  d^2/ds^2 or d^2 / dsdx or  d^2 / dsdc
    }else{
      hessAction = Eigen::VectorXd::Zero(inputSizes(inWrt1));
    }
  }
}; // end of class PredPreyModel