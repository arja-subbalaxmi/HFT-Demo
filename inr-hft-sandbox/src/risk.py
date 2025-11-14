class RiskGuard:
    def __init__(self, risk_cfg, strategy_cfg):
        self.cfg = risk_cfg
        self.cash = risk_cfg['initial_cash']
        self.max_inv = strategy_cfg['max_inventory_btc']
    def can_quote(self, inventory_btc, side):
        if side == 'buy' and inventory_btc >= self.max_inv: return False
        return True
    def pnl_pct(self, mark_inr, cash):
        return (cash - self.cfg['initial_cash']) / self.cfg['initial_cash'] * 100
