import unittest

from trading.risk import RiskManager


class RiskManagerTests(unittest.TestCase):
    def test_same_side_cap_blocks_stacking_but_allows_opposite_hedge(self) -> None:
        risk = RiskManager(max_contracts_per_market=1, max_open_markets=10, max_total_exposure=100.0)

        risk.record_fill("TEST", "yes", 0.60, 1)

        self.assertEqual(risk.capped_contracts("TEST", "yes", 0.60, 1), 0)
        self.assertEqual(risk.capped_contracts("TEST", "no", 0.40, 1), 1)

        ok_yes, _ = risk.check_trade("TEST", "yes", 0.60, 1)
        ok_no, _ = risk.check_trade("TEST", "no", 0.40, 1)

        self.assertFalse(ok_yes)
        self.assertTrue(ok_no)

    def test_reduce_position_only_removes_one_side(self) -> None:
        risk = RiskManager(max_contracts_per_market=2, max_open_markets=10, max_total_exposure=100.0)

        risk.record_fill("TEST", "yes", 0.60, 2)
        risk.record_fill("TEST", "no", 0.35, 1)
        removed = risk.reduce_position("TEST", "yes", 1)

        self.assertEqual(removed, 1)
        self.assertEqual(risk.current_side_contracts("TEST", "yes"), 1)
        self.assertEqual(risk.current_side_contracts("TEST", "no"), 1)

    def test_locked_pnl_sign_is_positive_for_profitable_box(self) -> None:
        risk = RiskManager(max_contracts_per_market=2, max_open_markets=10, max_total_exposure=100.0)

        risk.record_fill("TEST", "yes", 0.45, 1)
        risk.record_fill("TEST", "no", 0.40, 1)

        self.assertIn("locked +0.15", risk.net_position_str("TEST"))


if __name__ == "__main__":
    unittest.main()
