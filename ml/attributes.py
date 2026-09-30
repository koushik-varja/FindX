from __future__ import annotations
import re

COLORS={"black","blue","red","white","grey","green","navy","pink","brown","beige","silver","gold"}
BRANDS={"nike","puma","adidas","asics","campus","sparx","skechers","biba","aurelia","libas","samsung","boat","jbl","oneplus","realme","sony","logitech","hp","sandisk","anker","portronics","wildcraft","skybags","mokobara","lavie","casio","titan","fastrack","noise","timex","philips","wipro","milton","cello","prestige","bajaj","wakefit"}
MATERIALS={"cotton","rayon","mesh","polyester","denim","fleece","leather","metal","plastic","canvas","silicone","steel","wood","knit"}
CATEGORY_ALIASES=[
    ("wireless earbuds",["wireless earbuds","earbuds","buds","tws"]),
    ("wireless headphones",["wireless headphones","bluetooth headphones","headphones"]),
    ("running shoes",["running shoes","running shoe","road running","sports shoes","jogging shoes"]),
    ("walking shoes",["walking shoes","walking shoe"]),("trail shoes",["trail shoes","trail running"]),
    ("sneakers",["sneakers","sneaker"]),("kurti",["kurti","kurta"]),("sports t-shirt",["sports t-shirt","training t-shirt","gym t-shirt","sports tshirt"]),
    ("t-shirt",["t-shirt","tee"]),("shirt",["formal shirt","shirt"]),("hoodie",["hoodie"]),("jeans",["jeans","denim"]),
    ("backpack",["backpack","college bag","laptop backpack"]),("duffel bag",["duffel","gym bag"]),("tote bag",["tote"]),("sling bag",["sling"]),("laptop bag",["messenger bag","laptop bag"]),
    ("smartwatch",["smartwatch","fitness watch"]),("analog watch",["analog watch","formal watch"]),("digital watch",["digital watch"]),
    ("wireless mouse",["wireless mouse","bluetooth mouse"]),("portable SSD",["portable ssd","ssd"]),("charger",["charger"]),("power bank",["power bank"]),("wireless keyboard",["wireless keyboard","bluetooth keyboard"]),
    ("desk lamp",["desk lamp","study lamp"]),("water bottle",["water bottle","bottle"]),("electric kettle",["electric kettle","kettle"]),("pillow",["pillow"]),("bedsheet",["bedsheet"]),("storage basket",["storage basket"]),("table clock",["table clock"])
]

def parse_attributes(query: str) -> dict:
    q=query.lower(); out={}
    # price max/min
    max_patterns=[r"(?:under|below|less than|upto|up to)\s*₹?\s*(\d+)",r"₹\s*(\d+)\s*(?:or less|max)",r"(?:^|\s)(\d+)\s*(?:under|below)\b"]
    min_patterns=[r"(?:above|over|more than|at least)\s*₹?\s*(\d+)"]
    for pat in max_patterns:
        m=re.search(pat,q)
        if m: out['price_max']=int(m.group(1)); break
    for pat in min_patterns:
        m=re.search(pat,q)
        if m: out['price_min']=int(m.group(1)); break
    for c in COLORS:
        if re.search(rf"\b{re.escape(c)}\b",q): out['colour']=c; break
    for b in sorted(BRANDS,key=len,reverse=True):
        if re.search(rf"\b{re.escape(b)}\b",q): out['brand']='boAt' if b=='boat' else b.title(); break
    for canonical, aliases in CATEGORY_ALIASES:
        if any(re.search(rf"\b{re.escape(a)}\b",q) for a in aliases): out['category']=canonical; break
    for m in MATERIALS:
        if re.search(rf"\b{re.escape(m)}\b",q): out['material']=m; break
    if re.search(r"\b(women|woman|female|ladies)\b",q): out['gender']='women'
    elif re.search(r"\b(men|man|male)\b",q): out['gender']='men'
    if 'cheaper' in q: out['comparative']='cheaper'
    if 'similar' in q or 'same style' in q: out['similar']=True
    return out

def attribute_match(product: dict, attrs: dict) -> tuple[float,list[str]]:
    if not attrs: return 0.0, []
    matched=[]; checks=0; hits=0
    for key in ('colour','brand','material','gender'):
        if key in attrs:
            checks+=1
            pv = product.get(key) if key in product else product.get('attributes',{}).get(key)
            if str(pv).lower()==str(attrs[key]).lower(): hits+=1; matched.append(f"{key}={attrs[key]}")
    if 'category' in attrs:
        checks+=1
        target=attrs['category'].lower(); hay=(product.get('category','')+' '+product.get('attributes',{}).get('product_type','')).lower()
        if target in hay or any(x in hay for x in target.split()): hits+=1; matched.append(f"category={attrs['category']}")
    if 'price_max' in attrs:
        checks+=1
        if float(product.get('price',1e18)) <= attrs['price_max']: hits+=1; matched.append(f"price≤₹{attrs['price_max']}")
    if 'price_min' in attrs:
        checks+=1
        if float(product.get('price',-1)) >= attrs['price_min']: hits+=1; matched.append(f"price≥₹{attrs['price_min']}")
    return (hits/checks if checks else 0.0), matched

def passes_hard_filters(product: dict, attrs: dict) -> bool:
    if 'price_max' in attrs and product['price']>attrs['price_max']: return False
    if 'price_min' in attrs and product['price']<attrs['price_min']: return False
    return True
