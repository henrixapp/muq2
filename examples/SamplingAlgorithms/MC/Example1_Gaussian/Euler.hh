#ifndef EULER_HH
#define EULER_HH
#include <array>
#include <vector>
#include <functional>
using State = Eigen::VectorXd;
template< class VectorFunction>
State explicit_eulerStep(const VectorFunction &F,double deltaT,const State& state) {
    return  state+deltaT*F(state);
}
template<class VectorFunction>
State explicit_euler(const VectorFunction &F,State initialState, double stepSize, size_t numSteps){
    State currState=initialState;
    for(size_t i= 0;i<numSteps;i++){
        currState = explicit_eulerStep(F,stepSize,currState);
    }
    return currState;
}
#endif