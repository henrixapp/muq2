import muq.Modeling as mm

import numpy as np
import copy

import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.integrate as iscint
import matplotlib.pyplot as plt

class LotkaEquation(mm.PyModPiece):
    """
    This class solves the 1D elliptic ODE of the form
    $$
    -\frac{\partial}{\partial x}\cdot(K(x) \frac{\partial h}{\partial x}) = f(x).
    $$
    over $x\in[0,1]$ with boundary conditions $h(0)=0$ and
    $\partial h/\partial x =0$ at $x=1$.  This equation is a basic model of steady
    state fluid flow in a porous media, where $h(x)$ is the hydraulic head, $K(x)$
    is the hydraulic conductivity, and $f(x)$ is the recharge.

    This ModPiece uses linear finite elements on a uniform grid. There are two inputs,
    the conductivity $K(x)$, which is represented as piecewise constant within each
    of the $N$ cells, and the recharge $f(x)$, which is also piecewise constant in
    each cell.   There is a single output of this ModPice: the head $h(x)$ at the
    $N+1$ nodes in the discretization.

    """

    def __init__(self, timeSteps, x0_1, alpha,beta,gamma,delta):
        super(LotkaEquation,self).__init__([1],[1]) # (inputSizes, outputSizes)
        self.timeSteps = timeSteps
        self.x0_1 = x0_1
        self.a = alpha
        self.b = beta
        self.g = gamma
        self.d = delta

    def EvaluateImpl(self, inputs):
        """ Constructs the stiffness matrix and solves the resulting linear system.

            INPUTS:
                inputs: A list of vector-valued inputs.  Here, inputs has two
                        components. The first component contains a vector of
                        conductivity values for each cell and the second contains
                        recharge values for each cell.

            RETURNS:
                This function returns nothing.  It stores the result in the private
                ModPiece::outputs list that is then returned by the `Evaluate` function.
        """

        theta = inputs[0]


        # Solve the sparse linear system
        def lotkavolterra(t,y):
            return np.array([self.a*y[0]-self.b*y[0]*y[1],
                            self.d*y[0]*y[1]-self.g*y[1]])
        sol = iscint.solve_ivp(lotkavolterra,[0,15],[self.x0_1,theta],method="Radau",dense_output=True)
        
        t = np.linspace(0, 15, self.timeSteps)
        z = sol.sol(t)
        """
        print(z.T)
        plt.plot( t,z.T)
        plt.xlabel('t')
        plt.legend(['x', 'y'], shadow=True)
        plt.title('Lotka-Volterra System')
        plt.show()
        """
        # Set the output list using the solution
        self.outputs = [[z[-1][0]]]

    def GradientImpl(self, outWrt, inWrt, inputs, sensitivity):
        print("Shouldn't be called")
        """ Given the gradient of some objective function to the output of this
            ModPiece, this function computes one step of the chain rule to provide
            the gradient with respect to one of the inputs of the model.  The
            gradient with respect to the conductivity field by solving the forward
            model, solving the adjoint system, and then combining the results to
            obtain the gradient.

            INPUTS:
                outWrt: For a model with multiple outputs, this would be the index
                        of the output list that corresponds to the sensitivity vector.
                        Since this ModPiece only has one output, the outWrt argument
                        is not used in the GradientImpl function.

                inWrt: Specifies the index of the input for which we want to compute
                       the gradient.  If inWrt==0, then the gradient with respect
                       to the conductivity is returned.  If inWrt==1, the gradient
                       with respect to the recharge is returned.

                inputs: A list of vector-valued inputs.  Here, inputs has two
                        components. The first component contains a vector of
                        conductivity values for each cell and the second contains
                        recharge values for each cell.

                sensitivity: A vector containing the gradient of some function
                             with respect to the output of this ModPiece.

            RETURNS:
                This function returns nothing.  It stores the result in the private
                ModPiece::gradient variable that is then returned by the `Gradient` function.

        """
        condVals = inputs[0]
        recharge = inputs[1]

        # Construct the adjoint system
        K = self.BuildStiffness(condVals)

        # If the gradient is with respect to the conductivity...
        if(inWrt==0):
            adjRhs = self.BuildAdjointRhs(sensitivity)
            rhs = self.BuildRhs(recharge)

            sol = spla.spsolve(K,rhs)
            adjSol = spla.spsolve(K.T,adjRhs)
            self.gradient = -1.0*(-adjSol[0:-1]/self.dx + adjSol[1:]/self.dx)*(-sol[0:-1]/self.dx + sol[1:]/self.dx)

        # Otherwise, if the gradient is with respect to the recharge...
        elif(inWrt==1):

            gradRhs = spla.spsolve(K.T,sensitivity)

            grad = np.zeros((self.numCells,))
            grad[0:-1] += 0.5*self.dx*gradRhs[1:-1]
            grad[1:]   += 0.5*self.dx*gradRhs[1:-1]

            self.gradient = grad


    # def ApplyHessianImpl(self, outWrt, inWrt1, inWrt2, inputs, sensitivity, vec):
    #
    #     condVals = inputs[0]
    #     recharge = inputs[1]
    #
    #     # If this is the second derivative wrt recharge, return 0
    #     if((inWrt1==1)&(inWrt2==1)):
    #         self.hessAction = np.zeros(vec.shape)
    #
    #     # if it's a mixed derivative (conductivity then recharge)
    #     elif((inWrt1==0)&(inWrt2==1)):
    #         assert(False)
    #
    #     # if it's a mixed derivative (recharge then conductivity)
    #     elif((inWrt1==1)&(inWrt2==0)):
    #         ApplyHessianImpl(self,outWrt,inWrt2,inWrt1, inputs,sensitivity,vec)
    #
    #     # Both wrt conductivity
    #     elif((inWrt1==0)&(inWrt2==0)):
    #
    #         # Solve the forward system
    #         K = self.BuildStiffness(condVals)
    #         rhs = self.BuildRhs(recharge)
    #         sol = spla.spsolve(K,rhs)
    #
    #         # Solve the adjoint system
    #         adjRhs = self.BuildAdjointRhs(sensitivity)
    #         adjSol = spla.spsolve(K,adjRhs) # Because K is symmetric
    #
    #         # Solve the incremental forward system
    #         incrForRhs = -self.BuildIncrementalRhs(adjSol)
    #         incrForSol = spla.spsolve(K,incrForRhs)
    #
    #         # Solve the incremental adjoint system
    #         incrAdjRhs = self.BuildIncrementalRhs(sol)
    #         incrAdjSol = spla.spsolve(K,incrAdjRhs) # Because K is symmetric
    #
    #         # Construct the Hessian action
    #         solDeriv = -sol[0:-1]/self.dx + sol[1:]/self.dx
    #         adjDeriv = -adjSol[0:-1]/self.dx + adjSol[1:]/self.dx
    #         incrForDeriv = -incrForSol[0:-1]/self.dx + incrForSol[1:]/self.dx
    #         incrAdjDeriv = -incrAdjSol[0:-1]/self.dx + incrAdjSol[1:]/self.dx
    #
    #         self.hessAction = (vec*incrAdjDeriv * solDeriv - vec*incrForDeriv * adjDeriv)#*2.
    #
    #     else:
    #         print('Have not implement Hessian wrt sensitivity yet.')
    #         assert(False)



    def BuildStiffness(self, condVals):
        """ Constructs the stiffness matrix for the conductivity defined within each cell. """

        rows = []
        cols = []
        vals = []

        dx2 = self.dx*self.dx

        # Left Dirichlet BC
        rows.append(0)
        cols.append(0)
        vals.append(1e10) # Trick to approximately enforce Dirichlet BC at x=0 while keeping matrix symmetric

        rows.append(0)
        cols.append(1)
        vals.append(-condVals[0]/dx2)

        # Interior nodes
        for i in range(1,self.numNodes-1):
            rows.append(i)
            cols.append(i-1)
            vals.append(-condVals[i-1]/dx2)

            rows.append(i)
            cols.append(i)
            vals.append(condVals[i-1]/dx2 + condVals[i]/dx2)

            rows.append(i)
            cols.append(i+1)
            vals.append(-condVals[i]/dx2)

        # Last cell on right
        rows.append(self.numCells)
        cols.append(self.numCells-1)
        vals.append(-condVals[self.numCells-1]/self.dx)

        rows.append(self.numCells)
        cols.append(self.numCells)
        vals.append(1e10)
        #vals.append(condVals[self.numCells-1]/self.dx)

        return sp.csr_matrix((vals,(rows,cols)), shape=(self.numNodes, self.numNodes))

    def BuildRhs(self, recharge):
        """ Constructs the right hand side vector. """

        rhs = np.zeros((self.numNodes,))

        rhs[1:-1] = 0.5*self.dx*(recharge[0:-1] + recharge[1:])
        rhs[-1] = 0.0
        return rhs

    def BuildAdjointRhs(self,sensitivity):
        rhs = copy.deepcopy(sensitivity)
        rhs[0] = 0.0
        rhs[-1] = 0.0
        return rhs

    def BuildIncrementalRhs(self, sol):
        # The derivative of the solution in each cell
        solGrad = -sol[0:-1]/self.dx + sol[1:]/self.dx

        rhs = np.zeros((self.numNodes,))
        rhs[0:-1] = -1.0*solGrad #/ self.dx * self.dx
        rhs[1:] += solGrad #* self.dx * self.dx
        rhs[0]=0.0
        rhs[-1]=0.0
        return rhs

if __name__ == "__main__":
    numCells = 2
    def lotkavolterra(t,y,a=0.5,b=0.5,c=0.5,d=0.5):
        return np.array([a*y[0]-b*y[0]*y[1],
                            d*y[0]*y[1]-c*y[1]])
    sol = iscint.solve_ivp(lotkavolterra,[0,15],[1.0,0.8],method="Radau",dense_output=True)
    sol2 = iscint.solve_ivp(lotkavolterra,[0,15],[1.0,0.85],method="Radau",dense_output=True)
    t = np.linspace(0, 15, 400)
    z = sol.sol(t)
    z2 = sol2.sol(t)
    print(z.T-z2.T)
    plt.plot( t,z.T)
    plt.plot( t,z2.T)
    plt.xlabel('t')
    plt.legend(['x', 'y','x2','y2'], shadow=True)
    plt.title('Lotka-Volterra System')
    plt.show()
    mod = LotkaEquation(numCells,1,.5,.5,0.5,0.5)

    cond = np.linspace(1,2,numCells)
    rchg = 1.0*np.ones(numCells)
    sens = np.ones(numCells+1)
    dir = np.ones(numCells)
    mod.EvaluateImpl([0.5])
    hessActFD = mod.ApplyHessianByFD(0,0,0, [cond,rchg], sens, dir)
    hessActFD2 = (mod.Gradient(0,0,[cond+1e-6*dir,rchg],sens) - mod.Gradient(0,0,[cond,rchg],sens))/1e-6
    hessAct = mod.ApplyHessian(0,0,0, [cond,rchg], sens, dir)
    print(hessActFD)
    print('vs')
    print(hessAct)
    plt.plot(hessActFD,label='fd1')
    plt.plot(hessActFD2,'--',label='fd2')
    plt.plot(hessAct,label='adjoint')
    plt.legend()
    plt.show()
    quit()

    # Create an exponential operator to transform log-conductivity into conductivity
    cond = mm.ExpOperator(numCells)

    # Combine the two models into a graph
    graph = mm.WorkGraph()
    graph.AddNode(mm.IdentityOperator(numCells), "Log-Conductivity")
    graph.AddNode(cond, "Conductivity")
    graph.AddNode(mod,"Forward Model")
    graph.AddEdge("Log-Conductivity", 0, "Conductivity", 0)
    graph.AddEdge("Conductivity",0,"Forward Model",0)
    graph.Visualize("ModelGraph.png")

    # Create a ModGraphPiece with two inputs: [log-conductivity, recharge]
    fullMod = graph.CreateModPiece("Forward Model")
    print('Full model input sizes  = ', fullMod.inputSizes)
    print('Full model output sizes = ', fullMod.outputSizes)

    # Create an arbitrary log-conductivity field
    logCond = np.linspace(-1,2,numCells)

    # Set the recharge to be constant across the domain
    recharge = np.ones(numCells)

    # Solve for the hydraulic head
    hydHead = fullMod.Evaluate([logCond,recharge])

    # Compute the gradient with respect to the conductivity field.
    sens = np.ones((numCells+1))
    condGrad = mod.Gradient(0,0,[logCond,recharge],sens)
    condGradFD = mod.GradientByFD(0,0,[logCond,recharge],sens)

    # Compute the gradient with respect to the recharge field
    rchgGrad = mod.Gradient(0,1,[logCond,recharge],sens)
    rchgGradFD = mod.GradientByFD(0,1,[logCond,recharge],sens)

    # Plot the gradients for comparison of adjoint and FD solutions
    fig, axs = plt.subplots(ncols=3,figsize=(14,5))
    axs[0].plot(mod.xs, hydHead[0])
    axs[0].set_title('PDE Solution: Hydraulic Head')

    axs[1].plot(mod.xs[0:-1], condGrad, label='Adjoint')
    axs[1].plot(mod.xs[0:-1], condGradFD, label='Finite Difference')
    axs[1].set_title('Gradient wrt Conductivity')
    axs[1].legend()

    axs[2].plot(mod.xs[0:-1], rchgGrad, label='Adjoint')
    axs[2].plot(mod.xs[0:-1], rchgGradFD, label='Finite Difference')
    axs[2].set_title('Gradient wrt Recharge')
    axs[2].legend()

    plt.show()
