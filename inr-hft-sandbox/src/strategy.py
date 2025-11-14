from dataclasses import dataclass
@dataclass
class Quote:
    bid: float
    ask: float
    mid: float
def micro_price(book, trade_tick):
    # simple mid + 0.3 * (imbalance)
    bids = sum([b[1] for b in book['bid'][:5]])
    asks = sum([a[1] for a in book['ask'][:5]])
    imbalance = (bids - asks) / (bids + asks + 1e-8)
    mid = (book['bid'][0][0] + book['ask'][0][0]) / 2
    return Quote(bid=mid * (1 - 0.0005 * imbalance),
                 ask=mid * (1 + 0.0005 * imbalance),
                 mid=mid)
