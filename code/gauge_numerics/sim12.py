import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ.setdefault(k,'1')
import numpy as np
DT=np.complex128
def zbits(n):
    k=np.arange(2**n); return np.array([1-2*((k>>(n-1-q))&1) for q in range(n)]).T
def Ediag(n,J):
    z=zbits(n); return sum(J[u,v]*z[:,u]*z[:,v] for u in range(n) for v in range(u+1,n) if J[u,v]!=0)
def gad(p,s):
    # superop on vec(rho) index 2a+b (rho_ab), N=(1-s)/2
    N=(1-s)/2; c=np.sqrt(1-p); S=np.zeros((4,4),complex)
    S[0,0]=1-p*N; S[0,3]=p*(1-N); S[3,3]=1-p*(1-N); S[3,0]=p*N; S[1,1]=S[2,2]=c; return S
def rx(b):
    r=np.array([[np.cos(b),-1j*np.sin(b)],[-1j*np.sin(b),np.cos(b)]]); return np.kron(r,r.conj())
def layer_super(p,s,b): return gad(p,s)@rx(b)@gad(p,s)
def run(n,hd,g,b,p,svec,G=3,dt=None):
    """returns diag probabilities. svec per-qubit direction (0=twirl). qubits grouped G at a time."""
    dt=dt or DT
    d=np.ones(2**n,dt)/np.sqrt(2**n); rho=np.outer(d,d.conj()).astype(dt)
    groups=[list(range(k,min(k+G,n))) for k in range(0,n,G)]
    for l in range(len(g)):
        ph=np.exp(-1j*g[l]*hd).astype(dt)
        rho*=ph[:,None]; rho*=ph.conj()[None,:]
        S1={s:layer_super(p,s,b[l]).reshape(2,2,2,2) for s in set(svec)}
        t=rho.reshape([2]*(2*n)); order=list(range(2*n))
        for grp in groups:
            T=np.ones([1]*0)
            T=S1[svec[grp[0]]]
            for q in grp[1:]:
                k=T.ndim//2
                T=np.multiply.outer(T,S1[svec[q]])  # out axes (o..., i..., o2, i2)
                T=np.moveaxis(T,[2*k,2*k+1],[k,k+1])
            T=T.astype(dt)
            ax=[q for q in grp]+[n+q for q in grp]
            Tm=T  # current order: (r1,c1,r2,c2..., in r1,c1,...)
            k=len(grp); perm=[2*a for a in range(k)]+[2*a+1 for a in range(k)]
            Tm=np.transpose(T,perm+[2*k+x for x in perm])  # (r..., c..., in r..., in c...)
            pos=[order.index(a) for a in ax]
            t=np.tensordot(Tm,t,axes=(list(range(2*k,4*k)),pos))
            order=ax+[o for o in order if o not in ax]
        t=np.ascontiguousarray(np.transpose(t,np.argsort(order)))
        rho=t.reshape(2**n,2**n)
    return np.real(np.diag(rho)).astype(float)
def sv_energy(n,hd,g,b):
    psi=np.ones(2**n,complex)/np.sqrt(2**n)
    for l in range(len(g)):
        psi=psi*np.exp(-1j*g[l]*hd)
        t=psi.reshape([2]*n); c,s=np.cos(b[l]),-1j*np.sin(b[l])
        for q in range(n):
            t=np.moveaxis(t,q,0); t=np.stack([c*t[0]+s*t[1],s*t[0]+c*t[1]]); t=np.moveaxis(t,0,q)
        psi=t.reshape(-1)
    return float(np.real(np.abs(psi)**2@hd))
