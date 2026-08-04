import unittest

import pandas as pd

from src.point_in_time_universe import (
    apply_delisting_returns,
    expand_monthly_membership,
    filter_panel_to_point_in_time_universe,
    validate_security_master,
)


class PointInTimeUniverseTests(unittest.TestCase):
    def setUp(self):
        self.master = pd.DataFrame(
            [
                {
                    "security_id": "SEC1",
                    "ticker": "AAA",
                    "company_name": "Alpha plc",
                    "index_name": "NDX",
                    "entry_date": "2020-01-15",
                    "exit_date": "2020-03-20",
                    "ticker_start_date": "2020-01-15",
                    "ticker_end_date": "2020-03-20",
                    "delisting_date": "",
                    "delisting_return": "",
                    "event_type": "membership",
                    "source_name": "Official notice",
                    "source_url": "https://example.test/alpha",
                    "source_retrieved_at": "2026-08-05",
                    "source_quality": "official",
                },
                {
                    "security_id": "SEC2",
                    "ticker": "BBB",
                    "company_name": "Beta plc",
                    "index_name": "NDX",
                    "entry_date": "2020-02-01",
                    "exit_date": "",
                    "ticker_start_date": "2020-02-01",
                    "ticker_end_date": "",
                    "delisting_date": "2020-04-15",
                    "delisting_return": -0.45,
                    "event_type": "delisting",
                    "source_name": "Exchange bulletin",
                    "source_url": "https://example.test/beta",
                    "source_retrieved_at": "2026-08-05",
                    "source_quality": "official",
                },
            ]
        )

    def test_membership_is_expanded_only_inside_intervals(self):
        membership = expand_monthly_membership(
            self.master, "NDX", "2020-01-01", "2020-04-30"
        )
        alpha_months = membership.loc[membership["ticker"] == "AAA", "date"].dt.strftime("%Y-%m").tolist()
        beta_months = membership.loc[membership["ticker"] == "BBB", "date"].dt.strftime("%Y-%m").tolist()
        self.assertEqual(alpha_months, ["2020-01", "2020-02", "2020-03"])
        self.assertEqual(beta_months, ["2020-02", "2020-03", "2020-04"])

    def test_panel_filter_excludes_non_members(self):
        membership = expand_monthly_membership(
            self.master, "NDX", "2020-01-01", "2020-04-30"
        )
        panel = pd.DataFrame(
            {
                "date": ["2020-01-31", "2020-01-31", "2020-04-30"],
                "ticker": ["AAA", "BBB", "AAA"],
                "feature": [1.0, 2.0, 3.0],
            }
        )
        eligible, audit = filter_panel_to_point_in_time_universe(panel, membership)
        self.assertEqual(len(eligible), 1)
        self.assertEqual(eligible.iloc[0]["ticker"], "AAA")
        self.assertEqual(len(audit), 2)

    def test_delisting_return_replaces_vendor_return(self):
        panel = pd.DataFrame(
            {
                "date": ["2020-04-30"],
                "ticker": ["BBB"],
                "future_stock_return": [-0.10],
                "delisting_date": ["2020-04-15"],
                "delisting_return": [-0.45],
            }
        )
        adjusted = apply_delisting_returns(panel)
        self.assertAlmostEqual(adjusted.iloc[0]["future_stock_return"], -0.45)
        self.assertTrue(bool(adjusted.iloc[0]["delisting_adjustment_applied"]))

    def test_invalid_interval_is_rejected(self):
        invalid = self.master.copy()
        invalid.loc[0, "exit_date"] = "2019-12-31"
        with self.assertRaises(ValueError):
            validate_security_master(invalid)


if __name__ == "__main__":
    unittest.main()
