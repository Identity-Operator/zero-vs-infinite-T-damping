"""Interleaved-ordering density-matrix simulator: rho stored as tensor (r0,c0,r1,c1,...,r_{n-1},c_{n-1}).
A per-qubit superop (4x4 on index 2r+c) for a block of G consecutive qubits is kron of 4x4s, applied by matmul without transposes."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ.setdefault(k,'1')
import sys; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import sim12 as S12
zbits=S12.zbits; Ediag=S12.Ediag; layer_super=S12.layer_super; sv_energy=S12.sv_energy
_cache={}
def interleave_index(n):
    """returns flat interleaved index for (r,c) pairs: idx[r,c]."""
    if n in _cache: return _cache[n]
    N=2**n; r=np.arange(N)
    # spread bits: bit (n-1-q) of r goes to bit 2(n-1-q)+1, of c to 2(n-1-q)
    def spread(x):
        o=np.zeros_like(x,dtype=np.int64)
        for k in range(n): o|=((x>>k)&1).astype(np.int64)<<(2*k)
        return o
    sr=spread(r); _cache[n]=(sr,)
    return _cache[n]
class Sim:
    def __init__(self,n,hd,G=None,dt=np.complex64):
        self.n=n; self.N=2**n; self.dt=dt; self.G=G or (2 if n<=8 else 3)
        sr,=interleave_index(n)
        self.diag_idx=(sr<<1)|sr
        # D[interleaved]=hd[r]-hd[c]
        rdt=np.float32 if dt==np.complex64 else np.float64
        idx=((sr[:,None]<<1)|sr[None,:]).ravel(); N=self.N
        self.ri=np.empty(N*N,np.int16); self.ci=np.empty(N*N,np.int16)
        self.ri[idx]=np.repeat(np.arange(N),N); self.ci[idx]=np.tile(np.arange(N),N)
        self.hd=hd
        self.groups=[list(range(k,min(k+self.G,n))) for k in range(0,n,self.G)]
    def run(self,g,b,p,svec):
        n,N,dt=self.n,self.N,self.dt
        rho=np.full(N*N,1.0/N,dt)
        for l in range(len(g)):
            ph=np.exp(-1j*g[l]*self.hd).astype(dt)
            rho*=ph[self.ri]; rho*=ph.conj()[self.ci]
            Ss={s:layer_super(p,s,b[l]) for s in set(svec)}
            for grp in self.groups:
                T=np.ones((1,1))
                for q in grp: T=np.kron(T,Ss[svec[q]])
                T=T.astype(dt); m=T.shape[0]
                L=4**grp[0]; R=4**(n-grp[-1]-1)
                if R==1: rho=(rho.reshape(L,m)@T.T).ravel()
                elif L==1: rho=(T@rho.reshape(m,R)).ravel()
                else: rho=np.matmul(T,rho.reshape(L,m,R)).ravel()
        return np.real(rho[self.diag_idx]).astype(float)
