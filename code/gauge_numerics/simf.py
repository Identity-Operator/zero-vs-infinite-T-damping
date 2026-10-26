import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ.setdefault(k,'1')
import sys; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import sim12 as S12
zbits=S12.zbits; Ediag=S12.Ediag; layer_super=S12.layer_super; sv_energy=S12.sv_energy
def run(n,hd,g,b,p,svec,dt=np.complex128):
    """Per-qubit superop application. Returns diag probabilities. svec: per-qubit s in {+1,-1,0}."""
    N=2**n
    d=np.full(N,1/np.sqrt(N),dt); rho=np.outer(d,d.conj())
    for l in range(len(g)):
        ph=np.exp(-1j*g[l]*hd).astype(dt)
        rho*=ph[:,None]; rho*=ph.conj()[None,:]
        Ss={s:layer_super(p,s,b[l]).reshape(2,2,2,2).astype(dt) for s in set(svec)}
        for q in range(n):
            A=2**q; B=2**(n-q-1); T=Ss[svec[q]]
            r=rho.reshape(A,2,B,A,2,B)
            src=[[r[:,a,:,:,c,:] for c in (0,1)] for a in (0,1)]
            out=np.empty_like(r)
            for x in (0,1):
                for y in (0,1):
                    o=out[:,x,:,:,y,:]
                    np.multiply(src[0][0],T[x,y,0,0],out=o)
                    for a,c in ((0,1),(1,0),(1,1)):
                        if T[x,y,a,c]!=0: o+=T[x,y,a,c]*src[a][c]
            rho=out.reshape(N,N)
    return np.real(np.diagonal(rho)).astype(float)
