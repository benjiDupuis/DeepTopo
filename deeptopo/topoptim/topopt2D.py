"""
Implementation of 2D topology optimization using SIMP method
Freely inspired by...
pass ft=2 to remove filtering. With ft=0 or ft=1,
it can be used without neural networks as a classical SIMP implementation
"""
import numpy as np
import cvxopt
import cvxopt.cholmod
from scipy.sparse import coo_matrix
from tqdm import tqdm

from deeptopo.topoptim.topopt_utils import lk, deleterowcol
from deeptopo.topoptim.loadcase import LoadCase


class Topopt2D(object):

    gradient: np.ndarray = None
    vol: float = None
    compliance: float = None
    KE: np.ndarray = lk()
    rmin: float = 5.4  # Filtering radius, if filtering is used
    g: int

    def __init__(self, load: LoadCase, volfrac: float,
                 penal: float = 3., method: str = "OC", ft: int = 0):

        assert 0. <= volfrac and volfrac <= 1., "Volume fraction should be in [0,1]"
        assert ft in [0, 1, 2], "Unknown filtering method"
        assert method in ["OC", "lagrangian", None], "Unknown optimization method"

        self.shape = (load.shape[0] - 1, load.shape[1] - 1)
        self.volfrac = volfrac
        self.penal = penal
        self.method = method
        self.ft = ft  # Filtering mode. 0: sensitivity; 1: density; other: no filtering
        self.Emin = 1.e-9
        self.Emax = 1.
        self.learning_rate = 1.e-2
        ndof = 2*(1+self.shape[0])*(1+self.shape[1])  # Degrees of freedom
        nele = self.shape[0]*self.shape[1]
        self.dc = np.ones(nele)
        self.ce = np.ones(nele)
        self.dv = np.ones(nele)
        self.u = np.zeros((ndof, 1))
        self.iK, self.jK, self.edofMat, self.H, self.Hs = \
            Topopt2D.construct_matrix(self.shape, self.rmin)
        self.f, self.fixed = load()
        dofs = np.arange(ndof)
        self.free = np.setdiff1d(dofs, self.fixed)

    def __str__(self):
        return "SIMP topology optimization using" + str(self.method)

    @staticmethod
    def construct_matrix(shape, rmin):

        (nelx, nely) = shape

        edofMat = np.zeros((nelx*nely, 8), dtype=int)
        for elx in range(nelx):
            for ely in range(nely):
                el = ely+elx*nely
                n1 = (nely+1)*elx+ely
                n2 = (nely+1)*(elx+1)+ely
                edofMat[el, :] = np.array(
                    [2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3, 2*n2, 2*n2+1, 2*n1, 2*n1+1])
        # Construct the index pointers for the coo format
        iK = np.kron(edofMat, np.ones((8, 1))).flatten()
        jK = np.kron(edofMat, np.ones((1, 8))).flatten()

        # Filter: Build (and assemble) the index+data vectors for the coo matrix format
        nfilter = int(nelx*nely*((2*(np.ceil(rmin)-1)+1)**2))
        iH = np.zeros(nfilter)
        jH = np.zeros(nfilter)
        sH = np.zeros(nfilter)
        cc = 0
        for i in range(nelx):
            for j in range(nely):
                row = i*nely+j
                kk1 = int(np.maximum(i-(np.ceil(rmin)-1), 0))
                kk2 = int(np.minimum(i+np.ceil(rmin), nelx))
                ll1 = int(np.maximum(j-(np.ceil(rmin)-1), 0))
                ll2 = int(np.minimum(j+np.ceil(rmin), nely))
                for k in range(kk1, kk2):
                    for ll in range(ll1, ll2):
                        col = k*nely+ll
                        fac = rmin-np.sqrt(((i-k)*(i-k)+(j-ll)*(j-ll)))
                        iH[cc] = row
                        jH[cc] = col
                        sH[cc] = np.maximum(0.0, fac)
                        cc = cc+1
        # Finalize assembly and convert to csc format
        H = coo_matrix((sH, (iH, jH)), shape=(nelx*nely, nelx*nely)).tocsc()
        Hs = H.sum(1)

        return iK, jK, edofMat, H, Hs

    # Optimality criteria method
    def __OC(self, x, dc, dv, g):
        (nelx, nely) = self.shape
        l1 = 1.e-9
        l2 = 1.e9
        move = 0.2
        xnew = np.zeros(nelx*nely, dtype=np.float)

        while (l2 - l1)/(l1 + l2) > 1.e-3:
            lmid = 0.5*(l2+l1)
            xnew[:] = np.maximum(0.0, np.maximum(
                x-move, np.minimum(1.0, np.minimum(x+move, x*np.sqrt(-dc/dv/lmid)))))
            gt = g+np.sum((dv*(xnew-x)))
            if gt > 0:
                l1 = lmid
            else:
                l2 = lmid
        self.g = gt
        return xnew

    # Apply density or sensitivity filtering if ft=0 or ft=1
    def __filtering(self, x):
        (nelx, nely) = self.shape
        nele = nelx*nely
        xPhys = np.zeros(nele)
        # Density filtering
        if self.ft == 0:
            xPhys[:] = x
        elif self.ft == 1:
            xPhys[:] = np.asarray(self.H*x[np.newaxis].T/self.Hs)[:, 0]
        else:
            xPhys[:] = x  # no filtering
        return xPhys

    # Warning: this method does not apply filtering
    def __step_class(self, xPhys, x, ce, dc, dv, u):

        assert self.free is not None, "No boundary conditions"
        assert self.f is not None, "Forces not determined"

        (nelx, nely) = self.shape
        ndof = 2*(1+nelx)*(1+nely)
        Emin, Emax = self.Emin, self.Emax
        penal = self.penal

        young = Emin + (xPhys)**penal*(Emax - Emin)
        d_young = penal*xPhys**(penal-1)*(Emax - Emin)

        sK = ((self.KE.flatten()[np.newaxis]).T*(young)).flatten(order='F')
        K = coo_matrix((sK, (self.iK, self.jK)), shape=(ndof, ndof)).tocsc()

        # Remove constrained dofs from matrix and convert to coo
        K = deleterowcol(K, self.fixed, self.fixed).tocoo()

        # Solve system
        K = cvxopt.spmatrix(K.data, K.row.astype(np.int), K.col.astype(np.int))
        B = cvxopt.matrix(self.f[self.free, 0])
        cvxopt.cholmod.linsolve(K, B)
        u[self.free, 0] = np.array(B)[:, 0]

        # Objective and sensitivity
        ce[:] = (np.dot(u[self.edofMat].reshape(nelx*nely, 8), self.KE)
                 * u[self.edofMat].reshape(nelx*nely, 8)).sum(1)
        obj = ((young)*ce).sum()
        dc[:] = (-d_young)*ce
        dv[:] = np.ones(nely*nelx)
        self.compliance = obj
        self.gradient = dc.copy()

        # Sensitivity filtering:
        if self.ft == 0:
            dc[:] = np.asarray((self.H*(x*dc))[np.newaxis].T/self.Hs)[:, 0] / np.maximum(0.001, x)
        elif self.ft == 1:
            dc[:] = np.asarray(self.H*(dc[np.newaxis].T/self.Hs))[:, 0]
            dv[:] = np.asarray(self.H*(dv[np.newaxis].T/self.Hs))[:, 0]

        # Volume
        vol = np.sum(xPhys)
        self.vol = vol
        xold = x.copy()

        # Optimization step, either with OC or None (In case of neural parameterization)
        assert self.method in [
            "OC", None], "Unknown optimization method,\
                 possibilities are OC or None (in case of neural parameterization)"
        if self.method == "OC":
            x[:] = self.__OC(x, dc, dv, self.g)

        change = np.linalg.norm(x.reshape(nelx*nely, 1)-xold.reshape(nelx*nely, 1), np.inf)

        return x, change

    def step(self, loop, x):
        if loop == 0:
            xPhys = x.copy()
        else:
            xPhys = self.__filtering(x)
        return self.__step_class(xPhys, x, self.ce, self.dc, self.dv, self.u)

    # return the final density field
    # TODO: ce, dc, dv, ... doivent etre défini au début de _call__
    # et ne pas en faire des attributs
    def __call__(self, iter_max: int, display: bool = True) -> np.ndarray:

        compliance_tab = []
        (nelx, nely) = self.shape
        nele = nelx*nely
        x = self.volfrac * np.ones(nelx*nely, dtype=np.float64)
        self.g = 0

        for k in tqdm(range(iter_max)):
            x[:], _ = self.step(k, x)
            compliance_tab.append(self.compliance)
            if display or (k == iter_max - 1):
                print("Iteration : ", k, "  Compliance : ", round(
                    self.compliance, 2), "   Volume : ", round(self.vol/nele, 2))
        xPhys = self.__filtering(x)

        return xPhys.reshape(nelx, nely).T, compliance_tab
