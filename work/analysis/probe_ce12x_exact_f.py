"""CPU diagnostic: exact independent-Bernoulli expected F0.5 versus production approximation.

Integrates the generating polynomial with Gaussian quadrature of sufficient degree.
Both selectors operate on identical exclusive-owner probabilities and owned candidates.
Missing-candidate truth is included in evaluation, but not modeled by the exact selector.
"""
from pathlib import Path
import json
import sys
import time
import itertools
import numpy as np
import polars as pl

WORK=Path(__file__).resolve().parents[1]
ART=WORK/'artifacts'
OUT=WORK/'analysis/ce12x_review'
sys.path.insert(0,str(WORK/'code/business_entity_resolution/src'))
from io_utils import load_ground_truth_pairs
from postprocess import exclusive_owner_prob,one_owner

def exact_utility(p):
    n=p.shape[1]
    mmax=min(12,n)
    # Integrand degree <= n+4*mmax-1, so 2*order-1 >= that degree.
    order=(n+4*mmax+1)//2
    gx,gw=np.polynomial.legendre.leggauss(order)
    x=(gx+1)/2
    w=gw/2
    product=np.ones((len(p),order),dtype=np.float64)
    for j in range(n):product*=1-p[:,j,None]+p[:,j,None]*x
    utility=np.empty((len(p),mmax+1))
    utility[:,0]=np.prod(1-p,axis=1)
    selected=np.zeros_like(product)
    for j in range(mmax):
        term=p[:,j,None]*x
        selected+=term/(1-p[:,j,None]+term)
        m=j+1
        utility[:,m]=5*np.sum(w*x**(4*m-1)*product*selected,axis=1)
    return utility

def verify():
    p=np.array([[.72,.51,.13],[.9,.8,.2],[.2,.1,.05]])
    ex=exact_utility(p)
    brute=np.zeros_like(ex)
    for b in itertools.product((0,1),repeat=p.shape[1]):
        mask=np.asarray(b)
        prob=np.prod(np.where(mask,p,1-p),axis=1)
        t=sum(b)
        for m in range(4):
            f=float(t==0) if m==0 else 1.25*sum(b[:m])/(m+.25*t)
            brute[:,m]+=prob*f
    np.testing.assert_allclose(ex,brute,atol=1e-12)

def main():
    verify()
    truthn=load_ground_truth_pairs().group_by('s1_id').len('n_true')
    result={}
    for c in ('India','US'):
        start=time.time()
        d=pl.scan_parquet(ART/'oof/test_blend_v7_ce_v12x_evaloof.parquet').filter(pl.col('country')==c).collect()
        ids=d.select('s1_id').unique()
        owned=one_owner(exclusive_owner_prob(d,'p'),'p').sort(['s1_id','p','cand_id'],descending=[False,True,False])
        groups=owned.group_by('s1_id',maintain_order=True).agg('p','y').join(truthn,on='s1_id',how='left').with_columns(pl.col('n_true').fill_null(0),pl.col('p').list.len().alias('n'))
        rows=groups.to_dicts()
        bylen={}
        for row in rows:bylen.setdefault((row['n']+7)//8*8,[]).append(row)
        records=[]
        for width,rs in bylen.items():
            for startrow in range(0,len(rs),512):
                chunk=rs[startrow:startrow+512]
                p=np.zeros((len(chunk),width))
                y=np.zeros_like(p)
                for j,r in enumerate(chunk):
                    p[j,:r['n']]=r['p']
                    y[j,:r['n']]=r['y']
                mmax=min(12,width)
                m=np.arange(1,mmax+1)
                ps=p.sum(axis=1)
                cs=p[:,:mmax].cumsum(axis=1)
                zero=np.prod(1-p,axis=1)[:,None]
                approx=np.concatenate([zero,1.25*cs/(m[None,:]+.25*ps[:,None]/.985)],axis=1)
                approx1=np.concatenate([zero,1.25*cs/(m[None,:]+.25*ps[:,None])],axis=1)
                exact=exact_utility(p)
                picks=np.column_stack([approx.argmax(axis=1),approx1.argmax(axis=1),exact.argmax(axis=1)])
                cum_y=np.concatenate([np.zeros((len(chunk),1)),y[:,:mmax].cumsum(axis=1)],axis=1)
                nt=np.array([r['n_true'] for r in chunk])
                for j,r in enumerate(chunk):
                    fs=[]
                    for k in picks[j]:
                        fs.append(float(nt[j]==0) if k==0 else 1.25*cum_y[j,k]/(.25*nt[j]+k))
                    records.append((r['s1_id'],*fs,*picks[j].tolist()))
        scores=pl.DataFrame(records,schema=['s1_id','production_approx','recall1_approx','exact_independent','m_prod','m_rec1','m_exact'],orient='row')
        noowned=ids.join(scores.select('s1_id'),on='s1_id',how='anti').join(truthn,on='s1_id',how='left').with_columns(pl.col('n_true').fill_null(0))
        n=ids.height
        def avg(col):return float((scores[col].sum()+noowned.filter(pl.col('n_true')==0).height)/n)
        result[c]={'n_S1':n,**{name:avg(name) for name in ('production_approx','recall1_approx','exact_independent')},
                   'changed_S1':scores.filter(pl.col('m_prod')!=pl.col('m_exact')).height,
                   'max_owned_candidates':int(groups['n'].max()),'seconds':time.time()-start}
        scores.write_parquet(OUT/f'{c}_exact_f_scores.parquet')
        print(json.dumps({c:result[c]}),flush=True)
        (OUT/'exact_f_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':main()
