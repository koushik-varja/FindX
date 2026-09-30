from ml.normalization import QueryNormalizer
from ml.attributes import parse_attributes

def test_3k_and_hinglish():
    n=QueryNormalizer({'black','running','shoes','under'}).normalize('3k ke andar black running shoes')
    assert '3000' in n.corrected and 'under' in n.corrected
    a=parse_attributes(n.corrected); assert a['price_max']==3000 and a['colour']=='black'
def test_typo_aliases():
    n=QueryNormalizer({'samsung','wireless','earbuds'}).normalize('samsoong wirless earbuds')
    assert n.corrected=='samsung wireless earbuds' and n.corrections['samsoong']=='samsung'
def test_hindi_mapping():
    n=QueryNormalizer({'black','running','shoes'}).normalize('काले रनिंग जूते')
    assert all(x in n.corrected for x in ['black','running','shoes'])
