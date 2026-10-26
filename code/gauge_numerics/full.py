"""Gauge-dependence study, tiers 1-3.  usage: python3 full.py TIER NWORKERS
Model: H=sum J_uv Z_u Z_v, |+>^n, per layer e^{-i g H}, AD stage A, X mixer, AD stage B (uniform p).
Gauge y: AD toward |y_q> on qubit q (equivalent to H^y = P_y H P_y with AD toward |0>). Twirl: s=0 (GAD, N=1/2)."""
import os,sys,json,time
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[k]='1'
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, simi
from scipy.optimize import minimize
from scipy.stats import pearsonr,spearmanr
from multiprocessing import Pool

TIER=int(sys.argv[1]); NW=int(sys.argv[2]) if len(sys.argv)>2 else 5
CFG={1:dict(n=8, ng=10,Ps=(1,2,3),ps=(0.01,0.05,0.1),restarts=4,gauges='all',nrnd=None,dt='c128',twopt=True,seed=101),
     2:dict(n=10,ng=8, Ps=(2,3),ps=(0.05,0.1),restarts=2,gauges='sub',nrnd=80,dt='c128',twopt=True,seed=202),
     3:dict(n=12,ng=6, Ps=(2,3),ps=(0.05,0.1),restarts=2,gauges='sub',nrnd=60,dt='c64',twopt=True,seed=303)}[TIER]
n=CFG['n']; rng=np.random.default_rng(CFG['seed'])
DT=np.complex64 if CFG['dt']=='c64' else np.complex128
EPS=3e-5 if DT==np.complex64 else 1e-7

def g_gauss():
    while True:
        J=np.zeros((n,n))
        for u in range(n):
            for v in range(u+1,n):
                if rng.uniform()<0.5: J[u,v]=rng.normal()
        if (J!=0).sum()>=n: return J
def g_3reg():
    while True:
        stubs=np.repeat(np.arange(n),3); rng.shuffle(stubs); E=set(); ok=True
        for a,b in stubs.reshape(-1,2):
            a,b=min(a,b),max(a,b)
            if a==b or (a,b) in E: ok=False;break
            E.add((a,b))
        if ok:
            J=np.zeros((n,n))
            for a,b in E: J[a,b]=1.
            return J
insts=[('gauss',g_gauss()) for _ in range(CFG['ng']//2)]+[('3reg',g_3reg()) for _ in range(CFG['ng']//2)]
z=simi.zbits(n)

def gauge_set(hd):
    half=np.arange(2**(n-1))   # canonical reps y_0=0; E(y)=E(ybar) exactly for h=0
    if CFG['gauges']=='all':
        isg=np.abs(hd[half]-hd.min())<1e-9; return half,[('gs' if x else 'all') for x in isg]
    e=hd[half]; gs=half[np.abs(e-hd.min())<1e-9]
    order=half[np.argsort(e,kind='stable')]; sg=set(gs.tolist())
    low=np.array([k for k in order if k not in sg][:20])
    rest=np.setdiff1d(half,np.concatenate([gs,low]))
    rnd=rng.choice(rest,CFG['nrnd'],replace=False)
    return np.concatenate([gs,low,rnd]).astype(int),['gs']*len(gs)+['low']*len(low)+['rnd']*len(rnd)

def opt_noiseless(hd,P,k=4):
    res=[]
    for r in range(16):
        x0=np.concatenate([rng.uniform(-1,1,P)*0.8,rng.uniform(-1,1,P)*0.8])
        o=minimize(lambda x:simi.sv_energy(n,hd,x[:P],x[P:]),x0,method='L-BFGS-B'); res.append((o.fun,o.x))
    res.sort(key=lambda t:t[0]); out=[]
    for f,x in res:   # distinct starts
        if all(np.abs(x-y).max()>1e-3 for _,y in out): out.append((f,x))
        if len(out)==k: break
    while len(out)<k: out.append((res[0][0],res[0][1]+rng.normal(0,0.1,2*P)))
    return out

def make_tasks():
  tasks=[]
  for ii,(fam,J) in enumerate(insts):
    hd=simi.Ediag(n,J); ks,lab=gauge_set(hd)
    for P in CFG['Ps']:
        starts=opt_noiseless(hd,P,CFG['restarts'])
        for p in CFG['ps']:
            rp=[(list(rng.uniform(-np.pi,np.pi,P)),list(rng.uniform(-np.pi/2,np.pi/2,P))) for _ in range(3)]
            tasks.append(dict(ii=ii,fam=fam,J=J.tolist(),P=P,p=p,starts=[list(x) for _,x in starts],Enl=float(starts[0][0]),
                              ks=ks.tolist(),lab=lab,rp=rp))
  return tasks

def metrics(E,PG,Etw,PGtw,Hy,lab,E0,full):
    dE=E-Etw; srt=np.argsort(E,kind='stable'); isgs=np.abs(Hy-E0)<1e-9; lab=np.array(lab)
    a=float(dE@Hy/(Hy@Hy)); r2=float(1-((dE-a*Hy)**2).sum()/((dE-dE.mean())**2).sum())
    avg_set=E if full else E[lab=='rnd']
    return dict(pe=float(pearsonr(dE,Hy)[0]),sp=float(spearmanr(dE,Hy)[0]),argmin_gs=bool(isgs[srt[0]]),
        gs_rank=int(np.where(isgs[srt])[0][0]),a_eff=a,r2=r2,spread=float((E.max()-E.min())/abs(Etw)),
        gain=float((Etw-E.min())/abs(Etw)),pgratio=float(PG[srt[0]]/PGtw) if PGtw>0 else None,
        avg_minus_tw=float((avg_set.mean()-Etw)/abs(Etw)),Etw=float(Etw),Emin=float(E.min()),PGtw=float(PGtw),PGbest=float(PG[srt[0]]))

def work(t):
    t0=time.time(); J=np.array(t['J']); hd=simi.Ediag(n,J); E0=hd.min(); gsm=(np.abs(hd-E0)<1e-9).astype(float)
    sim=simi.Sim(n,hd,G=2 if n<=8 else 3,dt=DT); P=t['P']; p=t['p']; tw=[0.0]*n
    ftw=lambda x: float(sim.run(x[:P],x[P:],p,tw)@hd)
    def fg(x):
        f0=ftw(x); gr=np.zeros_like(x)
        for i in range(len(x)):
            xx=x.copy(); xx[i]+=EPS; gr[i]=(ftw(xx)-f0)/EPS
        return f0,gr
    best=None; nev=0; opt_log=[]
    for x0 in t['starts']:
        o=minimize(fg,np.array(x0),jac=True,method='L-BFGS-B',options=dict(maxiter=80))
        opt_log.append(float(o.fun)); nev+=o.nfev
        if best is None or o.fun<best.fun: best=o
    canon=lambda b:[float((x+np.pi/4)%(np.pi/2)-np.pi/4) for x in b]   # beta -> beta+pi/2 leaves E_tw and E_noiseless invariant
    graw=list(best.x[:P]); braw=list(best.x[P:])
    sets=[('opt',graw,canon(braw)),('opt_raw',graw,braw)]+[('rand',g,b) for g,b in t['rp']]+[('randc',g,canon(b)) for g,b in t['rp']]
    out=dict({k:t[k] for k in ('ii','fam','P','p','Enl')},n=n,Eground=float(E0),opt_log=opt_log,opt_nfev=nev,param_source='twirl-opt',res=[])
    Hy=hd[t['ks']]
    cache={}
    for kind,g,b in sets:
        key=tuple(np.round(g,12))+tuple(np.round(b,12))
        if key in cache:
            m=dict(cache[key]); m.update(kind=kind,dup=True); out['res'].append(m); continue
        pt=sim.run(g,b,p,tw); Etw=float(pt@hd); PGtw=float(pt@gsm)
        E=[];PG=[]
        for k in t['ks']:
            pr=sim.run(g,b,p,[float(x) for x in z[k]]); E.append(float(pr@hd)); PG.append(float(pr@gsm))
        m=metrics(np.array(E),np.array(PG),Etw,PGtw,Hy,t['lab'],E0,CFG['gauges']=='all')
        m.update(kind=kind,g=g,b=b,E=E,PG=PG); out['res'].append(m); cache[key]=m
    out.update(Hy=Hy.tolist(),lab=t['lab'],sec=time.time()-t0)
    o=out['res'][0]; r=[x for x in out['res'] if x['kind']=='rand']; rc=[x for x in out['res'] if x['kind']=='randc']
    print(n,t['ii'],t['fam'],P,p,'opt pe=%.3f sp=%.3f gs=%d rank=%d a=%.3f r2=%.3f spread=%.3f gain=%.3f pg=%.2f avg=%+.4f Etw=%.3f Enl=%.3f E0=%.3f raw_pe=%.2f | rand pe=%s randc pe=%s | %.0fs nfev=%d'%(
        o['pe'],o['sp'],o['argmin_gs'],o['gs_rank'],o['a_eff'],o['r2'],o['spread'],o['gain'],o['pgratio'] or -1,o['avg_minus_tw'],o['Etw'],t['Enl'],E0,
        out['res'][1]['pe'],','.join('%.2f'%x['pe'] for x in r),','.join('%.2f'%x['pe'] for x in rc),out['sec'],nev),flush=True)
    return out

if __name__=='__main__':
    tasks=make_tasks()
    fn=os.path.join(os.path.dirname(os.path.abspath(__file__)),f'out_tier{TIER}.json')
    print(f'tier {TIER}: n={n}, {len(tasks)} points, gauges/point={[len(t["ks"]) for t in tasks]}',flush=True)
    tasks.sort(key=lambda t:(-t['P'],-len(t['ks'])))
    res=[]
    with Pool(NW) as pool:
        for r in pool.imap_unordered(work,tasks):
            res.append(r); json.dump(dict(cfg=CFG,results=res),open(fn+'.tmp','w')); os.replace(fn+'.tmp',fn)
    json.dump(dict(cfg=CFG,results=res,complete=True),open(fn,'w'))
    print('done',flush=True)
