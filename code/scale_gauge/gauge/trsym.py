"""Does reversing amplitude damping remain a gauge for rz (and any) encoding axis, via
spin time reversal Theta = (x)(-iY) K?  Independent numpy sim of the classifier layout:
per layer: trainable SU(2)^n -> CNOT ring -> noise -> encoding R_n(a) -> noise; trailing SU(2)^n -> noise; Z readout."""
import numpy as np, itertools
from functools import reduce
rng=np.random.default_rng(3); n,L,p=3,3,0.3
I2=np.eye(2); X=np.array([[0,1],[1,0]],complex); Y=np.array([[0,-1j],[1j,0]]); Z=np.diag([1.,-1.]).astype(complex)
kron=lambda ms: reduce(np.kron, ms)
def haar():
    q,r=np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))); return q*(np.diag(r)/abs(np.diag(r)))
def cnot(c,t):
    P0=np.diag([1,0]).astype(complex); P1=np.diag([0,1]).astype(complex)
    return kron([P0 if k==c else I2 for k in range(n)])+kron([P1 if k==c else (X if k==t else I2) for k in range(n)])
E=reduce(lambda A,B: B@A,[cnot(j,(j+1)%n) for j in range(n)])
def Rn(axis,a):
    G=axis[0]*X+axis[1]*Y+axis[2]*Z; return np.cos(a/2)*I2-1j*np.sin(a/2)*G
AD=[np.diag([1,np.sqrt(1-p)]).astype(complex), np.sqrt(p)*np.array([[0,1],[0,0]],complex)]
FLIP=[X@k@X for k in AD]
def noise(rho,K):
    for j in range(n):
        rho=sum(kron([k if m==j else I2 for m in range(n)])@rho@kron([k if m==j else I2 for m in range(n)]).conj().T for k in K)
    return rho
def feats(a,blocks,K,axis):
    rho=np.zeros((2**n,)*2,complex); rho[0,0]=1; S=kron([Rn(axis,ai) for ai in a])
    for l in range(L):
        U=E@kron(blocks[l]); rho=noise(U@rho@U.conj().T,K); rho=noise(S@rho@S.conj().T,K)
    U=kron(blocks[L]); rho=noise(U@rho@U.conj().T,K)
    return np.array([np.trace(kron([Z if m==j else I2 for m in range(n)])@rho).real for j in range(n)])
def pauli_string(Pm):
    paulis=[I2,X,Y,Z]
    for combo in itertools.product(range(4),repeat=n):
        Q=kron([paulis[c] for c in combo]); ov=np.trace(Q.conj().T@Pm)/2**n
        if np.isclose(abs(ov),1): return [paulis[c] for c in combo]
    raise AssertionError("not a Pauli string")
blocks=[[haar() for _ in range(n)] for _ in range(L+1)]
def time_reversed_blocks(blocks):
    Yn=kron([Y]*n); Pr=pauli_string(E.conj().T@Yn@E.conj()@Yn)    # Y conj(E) Y = E . P''  (E real)
    new=[]
    for l in range(L+1):
        b=[Y@blocks[l][j].conj()@Y for j in range(n)]
        if l<L: b=[Pr[j]@b[j] for j in range(n)]
        new.append(b)
    new[0]=[new[0][j]@X for j in range(n)]                         # T(|0><0|)=|1><1|: absorb X into first block
    return new
def unitary_X_blocks(blocks):
    Xn=kron([X]*n); Pr=pauli_string(E.conj().T@Xn@E@Xn)
    new=[]
    for l in range(L+1):
        b=[X@blocks[l][j]@X for j in range(n)]
        if l<L: b=[Pr[j]@b[j] for j in range(n)]
        new.append(b)
    new[0]=[new[0][j]@X for j in range(n)]
    return new
TR=time_reversed_blocks(blocks); UX=unitary_X_blocks(blocks)
gen=np.array([0.3,-0.5,0.8]); gen/=np.linalg.norm(gen)
print(f"{'encoding axis':22s} {'unitary X-frame gauge':>24s} {'time-reversal gauge':>22s}")
for name,axis in [('rx',(1,0,0)),('ry',(0,1,0)),('rz',(0,0,1)),('generic (.3,-.5,.8)',tuple(gen))]:
    ux=tr=0
    for _ in range(20):
        a=rng.uniform(0,2*np.pi,size=n); zf=feats(a,blocks,FLIP,axis)
        ux=max(ux,np.abs(zf+feats(a,UX,AD,axis)).max()); tr=max(tr,np.abs(zf+feats(a,TR,AD,axis)).max())
    print(f"{name:22s} {ux:24.2e} {tr:22.2e}")
