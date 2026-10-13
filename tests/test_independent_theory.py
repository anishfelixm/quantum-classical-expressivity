"""
INDEPENDENT CHECK OF THE THEORY - plain numpy, no PennyLane, no project code.

    python -m pytest tests/test_independent_theory.py -v -s      (or: python tests/test_independent_theory.py)

Why: tests/test_fourier_equivalence.py checks the dequantization claim against
the project's own VQCHead and classical_fourier modules. If both shared a
mistake, that test would still pass. This file rebuilds the circuit from first
principles - 2x2 gate matrices, Kronecker products, density matrices - with
PennyLane's gate conventions (RY, Rot = RZ RY RZ, StronglyEntanglingLayers'
CNOT ranges r = (l mod (d-1)) + 1, DepolarizingChannel's Pauli form) and checks
every theoretical statement the paper makes:

  1. single-encoding <X_i> lies in the 3^d trigonometric span; a wrong-frequency
     basis does not fit;
  2. two-qubit observables <X_i X_j> lie in the same span;
  3. re-uploading twice needs the 5^d span and does not fit the 3^d span;
  4. depolarizing noise applied ONCE per qubit after the circuit scales every
     output by exactly (1 - 4p/3), so a binary decision score keeps its ranking;
  5. depolarizing noise after EVERY layer does not - the scaling then depends on
     the input, so the AUC-invariance claim is specific to end-of-circuit noise.
     (This is the model 07_hardware_noise.py simulates for the single-block
     VQC; the paper must name it.)
"""
import itertools
from itertools import combinations

import numpy as np

rng = np.random.default_rng(0)
I2=np.eye(2); X=np.array([[0,1],[1,0]],complex); Y=np.array([[0,-1j],[1j,0]]); Z=np.diag([1,-1]).astype(complex)
def RY(t): c,s=np.cos(t/2),np.sin(t/2); return np.array([[c,-s],[s,c]],complex)
def RZ(t): return np.diag([np.exp(-1j*t/2),np.exp(1j*t/2)])
def Rot(a,b,c): return RZ(c)@RY(b)@RZ(a)          # PennyLane Rot(phi,theta,omega)
def op(g,w,d):
    m=np.array([[1]],complex)
    for k in range(d): m=np.kron(m,g if k==w else I2)
    return m
def cnot(c,t,d):
    P0=np.diag([1,0]); P1=np.diag([0,1])
    a=np.array([[1]],complex); b=np.array([[1]],complex)
    for k in range(d):
        a=np.kron(a,P0 if k==c else I2); b=np.kron(b,P1 if k==c else (X if k==t else I2))
    return a+b
def sel_layer(W,l,d):                                # StronglyEntanglingLayers, PennyLane ranges
    U=np.eye(2**d,dtype=complex)
    for w in range(d): U=op(Rot(*W[l,w]),w,d)@U
    r=(l%(d-1))+1
    for w in range(d): U=cnot(w,(w+r)%d,d)@U
    return U
def depol(rho,p,w,d):
    out=(1-p)*rho
    for P in (X,Y,Z): Pw=op(P,w,d); out+=p/3*Pw@rho@Pw.conj().T
    return out
def run(z,W,d,L,R=1,p=0.0,where="end",pairs=False):
    psi=np.zeros(2**d,complex); psi[0]=1; rho=np.outer(psi,psi.conj())
    per=L//R
    for r in range(R):
        for w in range(d): U=op(RY(z[w]),w,d); rho=U@rho@U.conj().T
        for l in range(r*per,(r+1)*per):
            U=sel_layer(W,l,d); rho=U@rho@U.conj().T
            if p>0 and where=="layer":
                for w in range(d): rho=depol(rho,p,w,d)
    if p>0 and where=="end":
        for w in range(d): rho=depol(rho,p,w,d)
    out=[np.real(np.trace(op(X,i,d)@rho)) for i in range(d)]
    if pairs: out+=[np.real(np.trace(op(X,i,d)@op(X,j,d)@rho)) for i in range(d) for j in range(i+1,d)]
    return np.array(out)
def basis(Zs,K):
    om=np.array(list(itertools.product(range(-K,K+1),repeat=Zs.shape[1])),float)
    return np.concatenate([np.cos(Zs@om.T),np.sin(Zs@om.T)],1)   # spans all; redundancy harmless for lstsq
def resid(F,V):
    c,*_=np.linalg.lstsq(F,V,rcond=None); return np.abs(F@c-V).max()


D, L = 4, 2
_W = rng.normal(0, 1, (L, D, 3))
_Z = (rng.random((400, D)) - 0.5) * np.pi
_V = np.array([run(z, _W, D, L) for z in _Z])


def test_single_encoding_in_3d_span():
    assert resid(basis(_Z, 1), _V) < 1e-10
    bad = np.array([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2], [3, 0, 0, 0], [2, 2, 0, 0]], float)
    wrong = np.concatenate([np.cos(_Z @ bad.T), np.sin(_Z @ bad.T)], 1)
    assert resid(wrong, _V) > 0.1


def test_two_local_readout_in_same_span():
    Vp = np.array([run(z, _W, D, L, pairs=True) for z in _Z[:200]])
    assert resid(basis(_Z[:200], 1), Vp) < 1e-10


def test_reuploading_needs_5d_span():
    Vr = np.array([run(z, _W, D, L, R=2) for z in _Z])
    assert resid(basis(_Z, 2), Vr) < 1e-10
    assert resid(basis(_Z, 1), Vr) > 0.05


def test_end_of_circuit_depolarizing_is_uniform_contraction():
    p = 0.05
    Ve = np.array([run(z, _W, D, L, p=p, where="end") for z in _Z[:40]])
    assert np.abs(Ve - (1 - 4 * p / 3) * _V[:40]).max() < 1e-12


def test_layerwise_depolarizing_is_not_uniform():
    p = 0.05
    Vl = np.array([run(z, _W, D, L, p=p, where="layer") for z in _Z[:60]])
    w = np.random.default_rng(1).normal(size=D)
    s0, s1 = _V[:60] @ w, Vl @ w
    flips = sum(1 for i, j in combinations(range(60), 2) if (s0[i] - s0[j]) * (s1[i] - s1[j]) < 0)
    ratio = Vl / _V[:60]
    assert np.nanmax(ratio) - np.nanmin(ratio) > 0.01 and flips > 0


if __name__ == "__main__":
    for f in (test_single_encoding_in_3d_span, test_two_local_readout_in_same_span,
              test_reuploading_needs_5d_span, test_end_of_circuit_depolarizing_is_uniform_contraction,
              test_layerwise_depolarizing_is_not_uniform):
        f()
        print("PASS", f.__name__)
