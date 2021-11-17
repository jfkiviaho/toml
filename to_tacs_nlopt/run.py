from pathlib import Path
import os
import numpy as np
import argparse
from mpi4py import MPI
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt
from tacs import TACS, functions, elements, constitutive
import nlopt

class TopoAnalysis(object):
    def __init__(self, comm, fname='test.bdf', prefix='results',r0=1.5, p=3.0,
                 E0=1.0, nu=0.3, vol_con=0.3):
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
        # Get the model information
        ptr, conn, comps, Xpts = struct_mesh.getConnectivity()
        # Get the number of elements
        self.num_elems = struct_mesh.getNumElements()

        # Create the isotropic material class
        props = constitutive.MaterialProperties(E=E0, rho=1.0, nu=0.3)
        self.prefix = prefix
        # Get the centroid location
        centroid = np.zeros((self.num_elems,3))
        # Loop over components, creating stiffness and element object for each
        for i in range(self.num_elems):
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
            # Get the nodal ids for this element
            local_xpts = np.zeros(3)
            for nid in range(ptr[i], ptr[i+1]):
                # Get the nodal id
                node_id = conn[nid]
                local_xpts[0] += Xpts[3*node_id]
                local_xpts[1] += Xpts[3*node_id+1]
                local_xpts[2] += Xpts[3*node_id+2]
            # Find the centroid location for element i
            local_xpts /= ptr[i+1]-ptr[i]
            # Add to the global centroid array
            centroid[i,:] = local_xpts
        # Create tacs assembler object from mesh loader
        self.assembler = struct_mesh.createTACS(2)
        # Now, compute the filter weights and store them as a sparse
        # matrix
        # First check if the CSR matrix exists
        csr_matrix_name = Path(self.prefix+'/sparse-filter-rmin'+str(r0)+'.npz')
        if csr_matrix_name.exists():
            self.F = sparse.load_npz(csr_matrix_name)
        else:
            # Create the filtering matrix
            F = sparse.lil_matrix((self.num_elems, self.num_elems))

            # Do O(N^2) loop since I am lazy
            for i in range(self.num_elems):
                wt = 0.0
                for j in range(self.num_elems):
                    dist = np.linalg.norm(centroid[i,:]- centroid[j,:])
                    # Add to matrix
                    if dist <= r0:
                        w = 1.-dist/r0
                        F[i,j] = w
                        wt += w
                # Normalize the weights
                F[i,:] /= wt
            # Convert to a sparse CSR form
            self.F = F.tocsr()
            # Save the CSR matrix
            sparse.save_npz(csr_matrix_name, self.F)
        print("Done filtering")
        # Create the function call (evaluate compliance manually)
        self.funcs = [functions.StructuralMass(self.assembler)]
        # Create the forces
        self.forces = self.assembler.createVec()
        force_array = self.forces.getArray()
        # Apply the y nodal force on node 2
        force_array[1713*2+1] = -100.0
        self.assembler.applyBCs(self.forces)
        # Set up the objects needed for the solver
        self.ans = self.assembler.createVec()
        self.res = self.assembler.createVec()
        self.mat = self.assembler.createMat()
        self.dfdx = self.assembler.createDesignVec()
        self.dgdx = self.assembler.createDesignVec()
        self.pc = TACS.Pc(self.mat)
        subspace = 100
        restarts = 2
        self.gmres = TACS.KSM(self.mat, self.pc, subspace, restarts)
        # Scale the objective
        self.obj_scale = 1.e0
        self.dvs = self.assembler.createDesignVec()
        dvs_array = self.dvs.getArray()
        self.assembler.getDesignVars(self.dvs)
        # Set the inequality options for this problem in ParOpt:
        # The dense constraints are inequalities c(x) >= 0 and
        # use both the upper/lower bounds
        self.setInequalityOptions(sparse_ineq=False,
                                  use_lower=True, use_upper=True)
        # Set the volume constraint
        self.vol_con = vol_con
        self.itr = 0
        # Set visualization
        # Output for visualization
        flag = (TACS.OUTPUT_CONNECTIVITY |
                TACS.OUTPUT_NODES |
                TACS.OUTPUT_DISPLACEMENTS |
                TACS.OUTPUT_EXTRAS)
        self.f5 = TACS.ToFH5(self.assembler, TACS.PLANE_STRESS_ELEMENT, flag)

        return

    def setupAndRun(self):
        # Setup the NLOPT optimizer
        num_design_vars = self.num_elems
        opt = nlopt.opt(nlopt.LD_MMA, num_design_vars)

        opt.set_lower_bounds(np.ones(num_design_vars)*1e-3)
        opt.set_upper_bounds(np.ones(num_design_vars)*1.0)

        opt.set_min_objective(self.evalObj)

        opt.add_inequality_constraint(self.evalCon)

        opt.set_xtol_rel(1e-4)
        x = opt.optimize(np.random.rand(num_design_vars))
        minf = opt.last_optimum_value()

        return 

    def evalObj(self, x, grad):
        """
        Return the objective and objective gradient
        """

        con = np.zeros(1)
        # Set the new design variable values
        x = self.F.dot(x)
        x_array = self.dvs.getArray()
        np.copyto(x_array,x)
        self.assembler.setDesignVars(self.dvs)

        # Assemble the Jacobian and factor the matrix
        alpha = 1.0
        beta = 0.0
        gamma = 0.0
        self.assembler.zeroVariables()
        self.assembler.assembleJacobian(alpha, beta, gamma,
            self.res, self.mat)
        self.pc.factor()

        # Solve the linear system and set the varaibles into TACS
        self.gmres.solve(self.forces, self.ans)
        self.assembler.setVariables(self.ans)

        # Evaluate the function
        fvals = self.assembler.evalFunctions(self.funcs)

        # Set the compliance as the objective
        fobj = self.obj_scale*self.ans.dot(self.forces)

        # Compute the objective gradient
        if grad.size() > 0:
            self.assembler.setVariables(self.ans)
            self.assembler.addAdjointResProducts([self.ans], [self.dfdx],\
                alpha=-self.obj_scale)
            f_array = self.dfdx.getArray()
            grad[:] = (self.F.transpose()).dot(f_array)

            # Output the numpy array
            self.write_output(x[:])
            self.itr += 1

        return fobj

    def evalCon(self, x, grad):
        """
        Return the constraint and constraint gradient
        """

        con = np.zeros(1)
        # Set the new design variable values
        x = self.F.dot(x)
        x_array = self.dvs.getArray()
        np.copyto(x_array,x)
        self.assembler.setDesignVars(self.dvs)

        # Assemble the Jacobian and factor the matrix
        alpha = 1.0
        beta = 0.0
        gamma = 0.0
        self.assembler.zeroVariables()
        self.assembler.assembleJacobian(alpha, beta, gamma,
            self.res, self.mat)
        self.pc.factor()

        # Solve the linear system and set the varaibles into TACS
        self.gmres.solve(self.forces, self.ans)
        self.assembler.setVariables(self.ans)

        # Evaluate the function
        fvals = self.assembler.evalFunctions(self.funcs)

        # Set the volume fraction to be the constraint
        con[0] = fvals[0] - self.vol_con*2.0

        if grad.size() > 0:
            # Compute the objective gradient
            self.assembler.setVariables(self.ans)

            # Compute the mass gradient
            self.assembler.addDVSens([self.funcs[0]], [self.dgdx])
            Ax = self.dgdx.getArray()

            grad[:] = (self.F.transpose()).dot(Ax)

        return con

    def evalObjCon(self, x):
        """
        Return the objective, constraint and fail flag
        """

        fail = 0
        con = np.zeros(1)
        # Set the new design variable values
        x = self.F.dot(x)
        x_array = self.dvs.getArray()
        np.copyto(x_array,x)
        self.assembler.setDesignVars(self.dvs)

        # Assemble the Jacobian and factor the matrix
        alpha = 1.0
        beta = 0.0
        gamma = 0.0
        self.assembler.zeroVariables()
        self.assembler.assembleJacobian(alpha, beta, gamma,
            self.res, self.mat)
        self.pc.factor()

        # Solve the linear system and set the varaibles into TACS
        self.gmres.solve(self.forces, self.ans)
        self.assembler.setVariables(self.ans)

        # Evaluate the function
        fvals = self.assembler.evalFunctions(self.funcs)
        # Set the compliance as the objective
        fobj = self.obj_scale*self.ans.dot(self.forces)
        # Set the volume fraction to be the constraint
        con[0] = self.vol_con*2.0-fvals[0]

        return fail, fobj, con

    def evalObjConGradient(self, x, g, A):
        """
        Return the objective, constraint and fail flag
        """

        fail = 0
        # Compute the objective gradient
        self.assembler.setVariables(self.ans)
        self.assembler.addAdjointResProducts([self.ans], [self.dfdx],\
            alpha=-self.obj_scale)
        f_array = self.dfdx.getArray()
        g[:] = (self.F.transpose()).dot(f_array)
        # Compute the mass gradient
        self.assembler.addDVSens([self.funcs[0]], [self.dgdx])
        Ax = self.dgdx.getArray()
        A[0][:] = -(self.F.transpose()).dot(Ax)
        # Output the numpy array
        self.write_output(x[:])
        self.itr += 1
        return fail

    def write_output(self, x):
        """
        Write out the design variables
        """
        dv_file = self.prefix+'/design-vars-%04d'%(self.itr)+'.npy'
        np.save(dv_file, x)
        # Write to f5 file
        f5_file = self.prefix+'/tacs-output-%04d'%(self.itr)+'.f5'
        self.f5.writeToFile(f5_file)

        return

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fname', type=str)
    p.add_argument('--rmin', type=float, default=0.5)
    p.add_argument('--vol_con', type=float, default=0.5)
    p.add_argument('--prefix', type=str, default='results')
    p.add_argument('--check_grad', type=bool, default=False)

    args = p.parse_args()
    # Root communicator
    comm = MPI.COMM_WORLD
    if not Path(args.prefix).exists():
        os.mkdir(Path(args.prefix))
    # Create the problem class
    problem = TopoAnalysis(comm, args.fname, args.prefix,
                            E0=70e3, r0=args.rmin, vol_con=args.vol_con)
    problem.setupAndRun()
