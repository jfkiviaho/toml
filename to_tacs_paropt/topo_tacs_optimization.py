"""
Perform a 2D plane stress analysis for topology optimization
"""

import numpy as np
import argparse
from mpi4py import MPI
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt
from tacs import TACS, functions, elements, constitutive
from paropt import ParOpt

class TopoAnalysis(ParOpt.Problem):
    def __init__(self, comm, fname='test.bdf', r0=1.5, p=3.0,
                 E0=1.0, nu=0.3):
        """
        The constructor for the topology optimization class.

        This function sets up the data that is requried to perform a
        plane stress analysis of a square, plane stress structure.
        This is probably only useful for topology optimization.
        """
        self.comm = comm
        # Load in the mesh
        struct_mesh = TACS.MeshLoader(self.comm)
        struct_mesh.scanBDFFile(fname)
        # Get the number of elements
        num_elems = struct_mesh.getNumElements()
        super(TopoAnalysis, self).__init__(self.comm, num_elems, 1)

        # Create the isotropic material class
        props = constitutive.MaterialProperties(E=E0, rho=1.0, nu=0.3)
        
        self.r0 = r0
        self.xfilter = None
        # Loop over components, creating stiffness and element object for each
        for i in range(num_elems):
            # Set the design variable index
            design_variable_index = i
            stiff = constitutive.SimpPSConstitutive(props, t=1.0, tNum=i,q=p,
                                                    tlb=1e-3,tub=1.0)
            # Get the element model class
            model = elements.LinearElasticity2D(stiff)
            # Get the basis functions
            basis = elements.LinearQuadBasis()
            # Create the element object
            element = elements.Element2D(model,basis)
            # Set the element into the mesh
            struct_mesh.setElement(i, element)

        # Create tacs assembler object from mesh loader
        self.assembler = struct_mesh.createTACS(2)
        # # Now, compute the filter weights and store them as a sparse
        # # matrix
        # F = sparse.lil_matrix((self.nxelems*self.nyelems,
        #                        self.nxelems*self.nyelems))

        # # Compute the inter corresponding to the filter radius
        # ri = int(np.ceil(self.r0))

        # for j in range(self.nyelems):
        #     for i in range(self.nxelems):
        #         w = []
        #         wvars = []

        #         # Compute the filtered design variable: xfilter
        #         for jj in range(max(0, j-ri), min(self.nyelems, j+ri+1)):
        #             for ii in range(max(0, i-ri), min(self.nxelems, i+ri+1)):
        #                 r = np.sqrt((i - ii)**2 + (j - jj)**2)
        #                 if r < self.r0:
        #                     w.append((self.r0 - r)/self.r0)
        #                     wvars.append(ii + jj*self.nxelems)

        #         # Normalize the weights
        #         w = np.array(w)
        #         w /= np.sum(w)

        #         # Set the weights into the filter matrix W
        #         F[i + j*self.nxelems, wvars] = w

        # # Covert the matrix to a CSR data format
        # self.F = F.tocsr()

        return

    def mass(self, x):
        """
        Compute the mass of the structure
        """

        area = (self.Lx/self.nxelems)*(self.Ly/self.nyelems)

        return area*np.sum(x)

    def mass_grad(self, x):
        """
        Compute the derivative of the mass
        """

        area = (self.Lx/self.nxelems)*(self.Ly/self.nyelems)
        dmdx = area*np.ones(x.shape)

        return dmdx

    def compliance(self, x):
        """
        Compute the structural compliance
        """

        # Compute the filtered compliance. Note that 'dot' is scipy
        # matrix-vector multiplicataion
        xfilter = self.F.dot(x)

        # Compute the Young's modulus in each element
        E = self.E0*xfilter**self.p
        self.analyze_structure(E)

        # Return the compliance
        return 0.5*np.dot(self.f, self.u)

    def analyze_structure(self, E):
        """
        Given the elastic modulus variable values, perform the
        analysis and update the state variables.

        This function sets up and solves the linear finite-element
        problem with the given set of elastic moduli. Note that E > 0
        (component wise).

        Args:
           E: An array of the elastic modulus for every element in the
              plane stress domain
        """

        # Assemble the Jacobian and factor the matrix
        alpha = 1.0
        beta = 0.0
        gamma = 0.0
        self.assembler.zeroVariables()
        self.assembler.assembleJacobian(alpha, beta, gamma, 
                                        self.res, self.mat)
        self.pc.factor()

        # Solve the linear system and set the variables into TACS
        self.gmres.solve(self.forces, self.ans)
        self.assembler.setVariables(self.ans)

        return

    def compliance_grad(self, x):
        """
        Compute the gradient of the compliance using the adjoint
        method.

        Since the governing equations are self-adjoint, and the
        function itself takes a special form:

        K*psi = 0.5*f => psi = 0.5*u

        So we can skip the adjoint computation itself since we have
        the displacement vector u from the solution.

        d(compliance)/dx = - 0.5*u^{T}*d(K*u - f)/dx = - 0.5*u^{T}*dK/dx*u
        """

        # Compute the filtered variables
        self.xfilter = self.F.dot(x)

        # First compute the derivative with respect to the filtered
        # variables
        dcdxf = np.zeros(x.shape)

        if self.thermal_problem:
            # Sum up the contributions from each
            kelem = self.compute_element_thermal()

            for i in range(self.nelems):
                evars = self.u[self.elem_vars[i, :]]
                dxfdE = self.kappa0*self.p*self.xfilter[i]**(self.p - 1.0)
                dcdxf[i] = -0.5*np.dot(evars, np.dot(kelem, evars))*dxfdE
        else:
            # Sum up the contributions from each
            kelem = self.compute_element_stiffness()

            for i in range(self.nelems):
                evars = self.u[self.elem_vars[i, :]]
                dxfdE = self.E0*self.p*self.xfilter[i]**(self.p - 1.0)
                dcdxf[i] = -0.5*np.dot(evars, np.dot(kelem, evars))*dxfdE

        # Now evaluate the effect of the filter
        dcdx = (self.F.transpose()).dot(dcdxf)

        return dcdx

    def compliance_negative_hessian(self, s):
        """
        Compute the product of the negative
        """

        # Compute the filtered variables
        sfilter = self.F.dot(s)

        # First compute the derivative with respect to the filtered
        # variables
        Hsf = np.zeros(s.shape)

        if self.thermal_problem:
            # Sum up the contributions from each
            kelem = self.compute_element_thermal()

            scale = self.kappa0*self.p*(self.p - 1.0)
            for i in range(self.nelems):
                evars = self.u[self.elem_vars[i, :]]
                dxfdE = scale*sfilter[i]*self.xfilter[i]**(self.p - 2.0)
                Hsf[i] = 0.5*np.dot(evars, np.dot(kelem, evars))*dxfdE
        else:
            # Sum up the contributions from each
            kelem = self.compute_element_stiffness()

            scale = self.E0*self.p*(self.p - 1.0)
            for i in range(self.nelems):
                evars = self.u[self.elem_vars[i, :]]
                dxfdE = scale*sfilter[i]*self.xfilter[i]**(self.p - 2.0)
                Hsf[i] = -0.5*np.dot(evars, np.dot(kelem, evars))*dxfdE

        # Now evaluate the effect of the filter
        Hs = (self.F.transpose()).dot(Hsf)

        return Hs

    def computeQuasiNewtonUpdateCorrection(self, s, y):
        """
        The exact Hessian of the compliance is composed of the difference
        between two contributions:

        H = P - N

        Here P is a positive-definite term while N is positive semi-definite.
        Since the true Hessian is a difference between the two, the quasi-Newton
        Hessian update can be written as:

        H*s = y = P*s - N*s

        This often leads to damped update steps as the optimization converges.
        Instead, we want to approximate just P, so  we modify y so that

        ymod ~ P*s = (H + N)*s ~ y + N*s
        """
        Ns = self.compliance_negative_hessian(s[:])
        y[:] += Ns[:]
        return

    def getVarsAndBounds(self, x, lb, ub):
        """Get the variable values and bounds"""
        lb[:] = 1e-3
        ub[:] = 1.0
        # Make the initial point random
        x[:] = 0.95
        return

    def evalObjCon(self, x):
        """
        Return the objective, constraint and fail flag
        """

        fail = 0
        # Set the new design variable values
        filtr_x = self.F.dot(x)
        self.assembler.setDesignVars(x)

        


        

        obj = self.compliance(x[:])
        con = np.array([0.4*self.Lx*self.Ly - self.mass(x[:])])

        self.itr += 1

        return fail, obj, con

    def evalObjConGradient(self, x, g, A):
        """
        Return the objective, constraint and fail flag
        """

        fail = 0
        g[:] = self.compliance_grad(x[:])
        A[0][:] = -self.mass_grad(x[:])

        self.write_output(x[:])

        return fail

    def write_output(self, x):
        """
        Write out something to the screen
        """
        
    
        return

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fname', type=str)
    args = p.parse_args()
    comm = MPI.COMM_WORLD
    problem = TopoAnalysis(comm, args.fname,
                            E0=70e3, r0=3)
    asd
    problem.checkGradients()

    options = {
        'algorithm': 'tr',
        'tr_init_size': 0.05,
        'tr_min_size': 1e-6,
        'tr_max_size': 10.0,
        'tr_eta': 0.25,
        'tr_infeas_tol': 1e-6,
        'tr_l1_tol': 1e-3,
        'tr_linfty_tol': 0.0,
        'tr_adaptive_gamma_update': True,
        'tr_max_iterations': 1000,
        'penalty_gamma': 10.0,
        'qn_subspace_size': 10,
        'qn_type': 'bfgs',
        'qn_diag_type': 'yts_over_sts',
        'abs_res_tol': 1e-8,
        'starting_point_strategy': 'affine_step',
        'barrier_strategy': 'mehrotra_predictor_corrector',
        'tr_steering_barrier_strategy':
            'mehrotra_predictor_corrector',
        'tr_steering_starting_point_strategy': 'affine_step',
        'use_line_search': False}

    # Set up the optimizer
    opt = ParOpt.Optimizer(problem, options)

    # Set a new starting point
    opt.optimize()
    x, z, zw, zl, zu = opt.getOptimizedPoint()

