from src.strategy import micro_price
def test_micro_price_flat():
    book = {'bid': [[90, 10]], 'ask': [[110, 10]]}
    q = micro_price(book, None)
    assert 99. < q.mid < 101.
