"""
Perform a 2D plane stress analysis for topology optimization
"""
from pathlib import Path
import os
import numpy as np
import argparse
from mpi4py import MPI
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt
from tacs import TACS, functions, elements, constitutive
from paropt import ParOpt

class TopoAnalysis(ParOpt.Problem):
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
        super(TopoAnalysis, self).__init__(self.comm, self.num_elems, 1)

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
        self.obj_scale = 1.e-1/4.
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

    def getVarsAndBounds(self, x, lb, ub):
        """Get the variable values and bounds"""
        lb[:] = 1e-3
        ub[:] = 1.0
        # Make the initial point random
        x[:] = self.vol_con#np.random.rand(self.num_elems)
        return

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

def modify_PID(input_file, output_file):
    """
    Modify the PID of the elements
    """
    # Modify the PID
    input_data = open(input_file).readlines()
    index = 0
    new_list = []
    while index < len(input_data):
        line = input_data[index]
        if line.startswith('CQ'):
            eid = int(line[8:16])
            line = line[:16]+f"{eid:>8}"+line[24:]
        new_list.append(line)
        index += 1
    fout = open(output_file, 'w')
    fout.writelines(new_list)
    fout.close()

def modify_orientation(input_file, output_file, elem_list=None):
    """
    Rotate some of the CQUAD elements
    """
    input_data = open(input_file).readlines()
    index = 0
    new_list = []
    while index < len(input_data):
        line = input_data[index]
        if line.startswith('CQ'):
            eid = int(line[8:16])
            # Modify the orientation if needed
            if eid in elem_list:
                elem_order = []
                for i in range(4):
                    elem_order.append(int(line[24+i*8:32+i*8]))
                # Reverse the orientation of the connectivity
                elem_order.reverse()
                line = line[:24] + ("{:>8d}"*4).format(*elem_order)+"\n"
        new_list.append(line)
        index += 1
    fout = open(output_file, 'w')
    fout.writelines(new_list)
    fout.close()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fname', type=str)
    p.add_argument('--rmin', type=float, default=0.03)
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
    options = {
        'algorithm': 'tr',
        'tr_init_size': 0.01,
        'tr_min_size': 1e-3,
        'tr_max_size': 0.1,
        'tr_eta': 0.25,
        'tr_infeas_tol': 1e-6,
        'tr_l1_tol': 1e-3,
        'tr_linfty_tol': 0.0,
        'tr_adaptive_gamma_update': True,
        'tr_max_iterations': 2500,
        'penalty_gamma': 10.0,
        'qn_subspace_size': 25,
        'qn_type': 'bfgs',
        'qn_diag_type': 'yty_over_yts',
        'abs_res_tol': 1e-8,
        'starting_point_strategy': 'affine_step',
        'barrier_strategy': 'mehrotra_predictor_corrector',
        'tr_steering_barrier_strategy':
            'mehrotra_predictor_corrector',
        'tr_steering_starting_point_strategy': 'affine_step',
        'use_line_search': False}
    if args.check_grad:
        problem.checkGradients()
    else:
        # Set up the optimizer
        opt = ParOpt.Optimizer(problem, options)

        # Set a new starting point
        opt.optimize()
        x, z, zw, zl, zu = opt.getOptimizedPoint()

