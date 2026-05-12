from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ARTIFACTS_PATH = PROJECT_ROOT / "trading" / "model_artifacts.npz"

# Kalshi API
KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
COINS = ["BTC", "ETH", "XRP", "SOL"]
SERIES_TICKER_TEMPLATE = "KX{coin}15M"

# Trading signal thresholds (tune from notebook PnL grid)
TAU = 0.04       # minimum edge: |p_hat - q_mid| to trigger a trade
SIGMA = 0.10     # max posterior std — rejects high-uncertainty predictions

# Post-hoc p_hat calibration. Model has a measured systematic overshoot vs
# market mid: ~+10¢ in the middle zone (0.2 < yes_mid < 0.8) and ~+1-3¢ at
# the tails. Subtracting these offsets from p_hat before the edge calc
# reduces YES-side over-firing and the resulting NO-side miscalibration.
# Set either to 0.0 to disable that regime's correction.
P_HAT_BIAS_MID_ZONE = 0.08      # subtract when yes_mid in (MID_LOW, MID_HIGH)
P_HAT_BIAS_TAIL_ZONE = 0.03     # subtract when yes_mid outside that range
P_HAT_BIAS_MID_LOW = 0.10       # lower boundary of "mid zone"
P_HAT_BIAS_MID_HIGH = 0.77      # upper boundary of "mid zone"

# Cost model: Kalshi taker fee is variance-proportional, not flat — computed
# per-trade via KalshiClient.taker_fee(price) = ceil(7*P*(1-P)) cents.
# No tunable FEE constant; edge gates use TAU + taker_fee(ask_price).

# Risk limits
N_CONTRACTS = 1                      # base contracts to buy per signal tick (Kelly-scaled if enabled)
MAX_OPEN_MARKETS = 10
MAX_TOTAL_EXPOSURE_DOLLARS = 100.0   # max total capital at risk across all open positions (contracts * price)
EXIT_THRESHOLD = 0.25               # absolute stop-loss on market price (yes_mid): held YES exits if yes_mid<0.30, held NO exits if yes_mid>0.70

# Kelly-ish edge-proportional sizing. Baseline: a trade at edge == KELLY_BASE_EDGE
# gets N_CONTRACTS contracts. Stronger edges get proportionally more, capped by
# N_CONTRACTS_MAX. Disable with KELLY_ENABLED=False to revert to flat N_CONTRACTS.
#
# Scaling:  n = N_CONTRACTS * (edge / KELLY_BASE_EDGE), clamped to [1, N_CONTRACTS_MAX].
#
# Example with N_CONTRACTS=3, KELLY_BASE_EDGE=0.04, N_CONTRACTS_MAX=10:
#   edge 0.04 → 3 contracts (baseline)
#   edge 0.06 → 4 contracts
#   edge 0.08 → 6 contracts
#   edge 0.12 → 9 contracts
#   edge 0.16 → 10 contracts (capped)
KELLY_ENABLED = True
KELLY_BASE_EDGE = 0.04
N_CONTRACTS_MAX = 6

# Relative stop-loss: scales loss tolerance with entry price so cheap entries
# and expensive entries each get a sensible ceiling on per-contract loss.
# Fires whichever triggers first with the absolute stop above.
#
# Equation:
#   upside         = 1 - avg_entry                  # max $ you could win
#   raw_loss       = FRACTION_OF_UPSIDE × upside
#   loss_tolerance = clip(raw_loss, MIN_ABS, MAX_ABS)
#   stop_price     = max(0.01, avg_entry - loss_tolerance)
#
# Behavior with defaults below:
#   entry 0.30 → loss_tol 0.25 (max cap)  → stop 0.05
#   entry 0.50 → loss_tol 0.25 (max cap)  → stop 0.25
#   entry 0.70 → loss_tol 0.15            → stop 0.55
#   entry 0.85 → loss_tol 0.15 (min floor)→ stop 0.70
#   entry 0.95 → loss_tol 0.15 (min floor)→ stop 0.80
#
# Tuning:
#   Raise FRACTION to let mid-entries ride farther before stopping.
#   Raise MAX_ABS to tolerate more dollar loss on low-priced entries.
#   Raise MIN_ABS to give high-priced entries more breathing room against noise.
REL_STOP_ENABLED = True
REL_STOP_LOSS_FRACTION_OF_UPSIDE = 0.55   # risk this fraction of potential upside
REL_STOP_MAX_LOSS_ABS = 0.30              # never more than this loss/contract
REL_STOP_MIN_LOSS_ABS = 0.20              # always at least this much breathing room
REL_STOP_MIN_PRICE_FLOOR = 0.15           # stop-sell price never goes below this (protects cheap entries)

# Post-stop cooldown. After a stop-loss fires on a ticker, block NEW entries
# (either side) on that ticker for this many seconds. Prevents same-tick
# re-entry thrash (stop → re-buy → stop cycle). TP exits do NOT trigger
# this cooldown — only absolute/relative stop-loss fires.
COOLDOWN_SECONDS_AFTER_STOP = 5

# Operational filters
MIN_TIME_TO_CLOSE_SECONDS = 5    # skip near-expiry markets
MAX_TIME_TO_CLOSE_SECONDS = 280   # matches training filter (0–450s)
MIN_ORDER_PRICE = 0.22            # skip lottery-ticket trades where market has already near-decided
POLL_INTERVAL_SECONDS = 1

# Extreme-mid early-ttc filter: reject entries when the market is already pinned
# and there is still meaningful time for mean reversion. Asymmetric payoffs at
# extreme mids make small miscalibration costly.
EXTREME_MID_THRESHOLD = 0.82                                            # skip if yes_mid > this or < 1-this
EXTREME_MID_EARLY_TTC_SECONDS = int(MAX_TIME_TO_CLOSE_SECONDS * 2 / 3)  # only while ttc exceeds this

# ---------------------------------------------------------------------------
# Take-profit: close a held position whose edge has been absorbed by the market.
# ---------------------------------------------------------------------------
# Master toggle. Set to False to disable TP entirely — the engine falls through
# to the existing stop-loss logic as if TP didn't exist.
TAKE_PROFIT_ENABLED = True

# Capital-pressure gate. TP only fires when exposure utilization
# (risk.total_exposure / MAX_TOTAL_EXPOSURE_DOLLARS) exceeds this fraction.
# Set to 0.0 to always fire TP; set to 1.01 to effectively disable TP via this
# gate without touching TAKE_PROFIT_ENABLED.
TP_EXPOSURE_UTILIZATION_THRESHOLD = 0.5

# Signal-absorbed gate. TP fires when remaining_edge = (fair - sell_bid) is
# below this. fair = p_hat for held YES, 1-p_hat for held NO. Smaller = more
# conservative (lets winners run longer); larger = more aggressive.
TAKE_PROFIT_EDGE_THRESHOLD = 0.03

# Profit-minimum gate. TP requires at least this realized net profit per
# contract (after entry fee + exit fee). 0.0 = any profit; raise to avoid
# TPing on near-breakeven where noise dominates.
TAKE_PROFIT_MIN_PROFIT = 0.02

# Close-size rule. Each tick TP fires, sell this FRACTION of the currently
# held contracts (rounded down, floor 1), capped by TP_MAX_CONTRACTS_PER_TICK.
#   1.0 = full close — dump entire held side in one order (fastest unwind,
#         best defense vs crashes where scale-out can't keep up)
#   0.5 = half close — take half the position off each trigger
#   0.1 = slow chip-away — ~10% per trigger
# The MAX cap prevents one giant limit order blowing through a thin book.
TP_CLOSE_FRACTION = 0.8
TP_MAX_CONTRACTS_PER_TICK = 200

# Book-walk rule for TP sell orders.
#   True  = walk the bid book and place a limit at the DEEPEST profitable level
#           (guarantees full fill through all levels that clear TAKE_PROFIT_MIN_PROFIT).
#           Stops walking the moment a level would make us sell at a loss.
#   False = limit at top bid only (best price, may partial fill on thin books).
# Default True is safer on thin books because the full-close TP mode wants the
# whole position out; partial fills leave remaining contracts exposed to the
# next tick's market state.
TP_USE_BOOK_WALK = True

# Feature rolling window sizes (must match training in filter.py)
YES_MID_DIFF_SHORT = 1
YES_MID_DIFF_LONG = 5
YES_MID_STD_SHORT = 30
YES_MID_STD_LONG = 60
YES_SPREAD_MEAN_WINDOW = 30
SPOT_RETURN_VOL_SHORT = 30
SPOT_RETURN_VOL_LONG = 300  # 5 minutes
SPOT_FLOW_MEAN_WINDOW = 30
