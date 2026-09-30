from __future__ import annotations
from .attributes import attribute_match, passes_hard_filters

def _norm(v:float,lo:float,hi:float)->float:
    return 0.0 if hi<=lo else (v-lo)/(hi-lo)

def rerank(products:list[dict], candidate_ids:list[int], bm25:dict[int,float], dense:dict[int,float], fused:dict[int,float], attrs:dict, correction_confidence:float, k:int=20):
    bvals=[bm25.get(i,0) for i in candidate_ids] or [0]; dvals=[dense.get(i,0) for i in candidate_ids] or [0]; fvals=[fused.get(i,0) for i in candidate_ids] or [0]
    blo,bhi=min(bvals),max(bvals); dlo,dhi=min(dvals),max(dvals); flo,fhi=min(fvals),max(fvals)
    rows=[]
    for i in candidate_ids:
        p=products[i]
        if not passes_hard_filters(p,attrs): continue
        am,matched=attribute_match(p,attrs)
        b=_norm(bm25.get(i,0),blo,bhi); d=_norm(dense.get(i,0),dlo,dhi); f=_norm(fused.get(i,0),flo,fhi)
        category_bonus=0.08 if attrs.get('category') and any(t in (p['category']+' '+p['attributes']['product_type']).lower() for t in attrs['category'].lower().split()) else 0.0
        score=0.28*b+0.28*d+0.18*f+0.20*am+category_bonus+0.02*correction_confidence
        rows.append((i,float(score),matched,{'bm25':b,'dense':d,'fusion':f,'attribute':am,'category_bonus':category_bonus}))
    rows.sort(key=lambda x:(-x[1],products[x[0]]['price'],products[x[0]]['id']))
    return rows[:k]
