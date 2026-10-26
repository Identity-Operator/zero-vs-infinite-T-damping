import json,os,sys,numpy as np
D=os.path.dirname(os.path.abspath(__file__))
def load(t):
    f=os.path.join(D,f'out_tier{t}.json')
    if not os.path.exists(f): return None
    return json.load(open(f))
med=lambda a: float(np.median(a)) if len(a) else float('nan')
print('n P p    | N | opt(canonical beta): pe med(min)  sp  GSrate gsrank_med a_eff  R2   gain   spread  PGratio  avg-tw  | opt_raw pe med(min) [Ndiff] | rand(|b|<pi/2): pe med  flip | randc(|b|<pi/4): pe med flip')
for t in (1,2,3):
    d=load(t)
    if not d: continue
    R=d['results']; n=d['cfg']['n']
    for P in d['cfg']['Ps']:
        for p in d['cfg']['ps']:
            pts=[x for x in R if x['P']==P and x['p']==p]
            if not pts: continue
            O=[[r for r in x['res'] if r['kind']=='opt'][0] for x in pts]
            Raw=[[r for r in x['res'] if r['kind']=='opt_raw'][0] for x in pts]
            nd=sum(1 for r in Raw if not r.get('dup'))
            rr=[r for x in pts for r in x['res'] if r['kind']=='rand']; rc=[r for x in pts for r in x['res'] if r['kind']=='randc']
            f=lambda L,k: np.array([x[k] for x in L],dtype=float)
            print(f"{n} {P} {p:.2f} | {len(pts):2d} | {med(f(O,'pe')):.3f}({f(O,'pe').min():+.3f}) {med(f(O,'sp')):.3f} {f(O,'argmin_gs').mean():.2f} {med(f(O,'gs_rank')):3.0f} "
                  f"{med(f(O,'a_eff')):.4f} {med(f(O,'r2')):.2f} {med(f(O,'gain')):.3f} {med(f(O,'spread')):.3f} {med(f(O,'pgratio')):.2f} {med(f(O,'avg_minus_tw')):+.4f} | "
                  f"{med(f(Raw,'pe')):.3f}({f(Raw,'pe').min():+.3f}) [{nd}] | {med(f(rr,'pe')):.3f} {np.mean(f(rr,'pe')<0):.2f} | {med(f(rc,'pe')):.3f} {np.mean(f(rc,'pe')<0):.2f}")
if '-v' in sys.argv:
    for t in (1,2,3):
        d=load(t)
        if not d: continue
        for x in sorted(d['results'],key=lambda x:(x['ii'],x['P'],x['p'])):
            o=x['res'][0]
            print(t,x['ii'],x['fam'],x['P'],x['p'],'pe=%.3f gs=%d rank=%d gain=%.3f pg=%.2f a=%.4f r2=%.2f Etw=%.3f Enl=%.3f E0=%.3f'%(o['pe'],o['argmin_gs'],o['gs_rank'],o['gain'],o['pgratio'],o['a_eff'],o['r2'],o['Etw'],x['Enl'],x['Eground']))
