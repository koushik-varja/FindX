from __future__ import annotations
from collections import Counter, defaultdict
import math, re
import numpy as np

def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w-]+", text.lower(), flags=re.UNICODE)

class BM25Index:
    def __init__(self,k1:float=1.5,b:float=0.75): self.k1=k1; self.b=b
    def fit(self,docs:list[str]):
        self.docs=docs; self.tokens=[tokenize(d) for d in docs]; self.n=len(docs)
        self.avgdl=sum(map(len,self.tokens))/max(1,self.n); self.df=defaultdict(int); self.tf=[]
        for toks in self.tokens:
            c=Counter(toks); self.tf.append(c)
            for t in c:self.df[t]+=1
        self.idf={t:math.log(1+(self.n-df+0.5)/(df+0.5)) for t,df in self.df.items()}
        return self
    def scores(self,query:str)->np.ndarray:
        q=tokenize(query); scores=np.zeros(self.n,dtype=np.float32)
        for i,c in enumerate(self.tf):
            dl=len(self.tokens[i]); s=0.0
            for t in q:
                if t not in c: continue
                tf=c[t]; idf=self.idf.get(t,0.0)
                s += idf*(tf*(self.k1+1))/(tf+self.k1*(1-self.b+self.b*dl/max(self.avgdl,1e-9)))
            scores[i]=s
        return scores
    def topk(self,query:str,k:int=20):
        s=self.scores(query); idx=np.argsort(-s)[:k]
        return [(int(i),float(s[i])) for i in idx if s[i]>0]
