from __future__ import annotations

def reciprocal_rank_fusion(rankings:list[list[tuple[int,float]]],k0:int=60)->dict[int,float]:
    fused={}
    for ranking in rankings:
        for rank,(idx,_score) in enumerate(ranking,1): fused[idx]=fused.get(idx,0.0)+1.0/(k0+rank)
    return fused

def sort_fused(fused:dict[int,float],k:int=50): return sorted(fused.items(),key=lambda x:(-x[1],x[0]))[:k]
