from __future__ import annotations
from dataclasses import dataclass
from difflib import SequenceMatcher
import re

HINDI_MAP = {
    "काले":"black", "काला":"black", "ब्लैक":"black", "नीले":"blue", "नीली":"blue", "नीला":"blue",
    "रनिंग":"running", "जूते":"shoes", "जूता":"shoe", "कुर्ती":"kurti", "कॉटन":"cotton",
    "वायरलेस":"wireless", "ईयरबड्स":"earbuds", "हेडफोन":"headphones", "घड़ी":"watch",
    "स्मार्टवॉच":"smartwatch", "बैग":"bag", "बैकपैक":"backpack", "लैपटॉप":"laptop",
    "सस्ता":"cheaper", "अंदर":"under", "से":"", "कम":"under", "महिला":"women", "नीले":"blue"
}
HINGLISH_PHRASES = {
    "ke andar": "under", "k andar": "under", "se kam": "under", "se niche": "under",
    "kaale": "black", "kale": "black", "neela": "blue", "neeli": "blue", "jute": "shoes",
    "mahila": "women", "sasta": "cheaper", "saste": "cheaper"
}
ALIASES = {
    "samsoong":"samsung","samung":"samsung","samsng":"samsung","wirless":"wireless","wireles":"wireless",
    "hedphones":"headphones","hedphone":"headphones","runing":"running","addidas":"adidas","nik":"nike",
    "cotn":"cotton","earbud":"earbuds","tshirt":"t-shirt","earbuds":"earbuds"
}
STOPWORDS = {"the","a","an","for","with","and","or","to","of","in","on","but","same","style","this","like","me","show"}

@dataclass(frozen=True)
class NormalizedQuery:
    original: str
    normalized: str
    corrected: str
    corrections: dict[str,str]
    correction_confidence: float

class QueryNormalizer:
    def __init__(self, vocabulary: set[str] | None = None):
        self.vocabulary = {v.lower() for v in (vocabulary or set()) if len(v) > 2}
        self.vocabulary.update(ALIASES.values())

    @staticmethod
    def _price_k(text: str) -> str:
        def repl(m: re.Match[str]) -> str:
            return str(int(float(m.group(1)) * 1000))
        return re.sub(r"(?<!\w)(\d+(?:\.\d+)?)\s*k\b", repl, text, flags=re.I)

    def normalize(self, query: str) -> NormalizedQuery:
        original = query.strip()
        text = original.lower().replace("₹", " ₹ ")
        for hi,en in HINDI_MAP.items(): text = text.replace(hi,en)
        for src,dst in HINGLISH_PHRASES.items(): text = re.sub(rf"\b{re.escape(src)}\b", dst, text)
        text = self._price_k(text)
        text = re.sub(r"[^\w\-₹.]+", " ", text, flags=re.UNICODE)
        text = re.sub(r"\s+", " ", text).strip()
        tokens = text.split()
        out=[]; corrections={}; confs=[]
        for token in tokens:
            bare=token.strip(".-")
            if bare in ALIASES:
                corr=ALIASES[bare]; corrections[bare]=corr; out.append(corr); confs.append(0.99); continue
            if len(bare) < 4 or bare in self.vocabulary or bare in STOPWORDS or bare.isdigit() or bare == "₹":
                out.append(token); continue
            best=None; score=0.0
            for v in self.vocabulary:
                if abs(len(v)-len(bare)) > 3: continue
                s=SequenceMatcher(None,bare,v).ratio()
                if s>score: best,score=v,s
            if best and score >= 0.82:
                corrections[bare]=best; out.append(best); confs.append(score)
            else: out.append(token)
        corrected=" ".join(out)
        confidence=sum(confs)/len(confs) if confs else 1.0
        return NormalizedQuery(original, text, corrected, corrections, confidence)
