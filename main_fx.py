import json
from pathlib import Path
from datetime import datetime, date
import datetime as dt

import numpy as np
import pandas as pd
import requests
import streamlit as st


# =========================
# BASIC STYLING & CONFIG
# =========================

st.markdown("""
<style>
/* Hide the 'Press Enter to submit form' helper text everywhere */
[data-testid="InputInstructions"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="FX Portfolio Tracker",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Tighten layout a bit
st.markdown("""
<style>
    div[data-testid="stTabs"] > div:nth-child(1) {
        margin-top: -20px !important;
        padding-top: 0 !important;
    }

    .block-container {
        padding-top: 0.5rem !important;
    }

    div[data-testid="column"] {
        margin-top: -10px !important;
    }
</style>
""", unsafe_allow_html=True)

# Hide +/- on number_input globally (if we use any)
st.markdown("""
<style>
div[data-testid="stNumberInput"] button {
    display: none !important;
}
div[data-testid="stNumberInput"] input {
    padding-right: 0.5rem !important;
}
</style>
""", unsafe_allow_html=True)

# Compact KPI header styling
st.markdown("""
<style>
.mobile-header {
    margin-bottom: 0.4rem;
}
.mobile-header-title {
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 0.35rem;
}
.kpi-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 10px;
}
.kpi {
    flex: 1 1 calc(25% - 10px);
    border-radius: 10px;
    padding: 6px 10px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.02);
    text-align: center;
}
.kpi-label {
    font-size: 0.8rem;
    opacity: 0.8;
    white-space: nowrap;
}
.kpi-value {
    font-size: 1.1rem;
    font-weight: 700;
    margin-top: 2px;
    line-height: 1.2;
}
@media (min-width: 800px) {
    .kpi {
        flex: 1 1 0;
    }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

div[data-testid="stTabs"] button {
    padding-top: 0px !important;     /* default ~10px */
    padding-bottom: 6px !important;
    padding-left: 0px !important;
    padding-right: 0px !important;

    font-size: 0.85rem !important;   /* smaller label text */
    height: 30px !important;         /* reduce full tab height */
}

div[data-testid="stTabs"] button p {
    font-size: 0.85rem !important;   /* tab label text */
}

/* Optional: reduce gap between tabs a bit */
div[data-testid="stTabs"] button + button {
    margin-left: -2px !important;
}

</style>
""", unsafe_allow_html=True)


# =========================
# FILES & SETTINGS
# =========================

DATA_FILE = Path("fx_trades.csv")
SETTINGS_FILE = Path("fx_settings.json")
DEFAULT_BASE_CCY = "SEK"

# We persist only the raw inputs; all running/PL columns are recomputed
FX_INPUT_COLS = [
    "date",          # datetime
    "foreign_ccy",   # e.g. EUR, USD
    "txn_type",      # Buy, Sell, Debit, Credit
    "foreign_amount",
    "fx_rate",       # base per 1 foreign
    "fee_foreign",   # fee in foreign
    "memo",
]


def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_settings(settings: dict) -> None:
    try:
        SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    except Exception:
        pass

@st.cache_data(show_spinner=False)
def fetch_historical_fx_bulk(
    dates: list[date],
    from_ccy: str,
    to_ccy: str,
) -> dict[date, float]:
    """
    Get FX rates (from_ccy → to_ccy) for many dates using Frankfurter timeseries.
    Returns a dict mapping each requested date to 'to_ccy per 1 from_ccy'.

    Weekend/holiday handling:
    - Frankfurter only returns business days.
    - We forward-fill rates so weekends use the last available business day.
    """
    from_ccy = (from_ccy or "").upper()
    to_ccy = (to_ccy or "").upper()

    # Basic guards
    if not from_ccy or not to_ccy:
        return {d: 1.0 for d in dates}
    if from_ccy == to_ccy:
        return {d: 1.0 for d in dates}

    # Clean dates
    valid_dates = sorted(set(d for d in dates if pd.notna(d)))
    if not valid_dates:
        return {}

    start_str = valid_dates[0].strftime("%Y-%m-%d")
    end_str = valid_dates[-1].strftime("%Y-%m-%d")

    # ✅ Correct Frankfurter timeseries endpoint: /v1/start..end
    url = f"https://api.frankfurter.dev/v1/{start_str}..{end_str}"

    try:
        resp = requests.get(
            url,
            params={
                "base": from_ccy,
                "symbols": to_ccy,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get("rates", {})

        # If API gave nothing back, degrade gracefully
        if not data:
            return {d: 1.0 for d in dates}

        # Convert to Series: index = datetime.date, value = rate
        series = pd.Series(
            {
                datetime.strptime(ds, "%Y-%m-%d").date(): float(r[to_ccy])
                for ds, r in data.items()
                if to_ccy in r
            }
        ).sort_index()

        if series.empty:
            return {d: 1.0 for d in dates}

        # Build full daily index and forward-fill so weekends/holidays are covered
        full_idx = pd.date_range(valid_dates[0], valid_dates[-1], freq="D")
        series = series.reindex(full_idx.date).ffill()

        # Return rates for exactly the requested dates
        return {d: float(series.get(d, 1.0)) for d in dates}

    except Exception:
        # On error, degrade gracefully
        return {d: 1.0 for d in dates}


# =========================
# FX HELPERS (FRANKFURTER)
# =========================

@st.cache_data(show_spinner=False)
def fetch_historical_fx(date_obj: datetime, from_ccy: str, to_ccy: str) -> float:
    """
    Get FX rate (from_ccy → to_ccy) for a given date, using Frankfurter.dev.
    Returns rate as 'to_ccy per 1 from_ccy'.
    """
    from_ccy = (from_ccy or "").upper()
    to_ccy = (to_ccy or "").upper()
    if not from_ccy or not to_ccy:
        return 1.0
    if from_ccy == to_ccy:
        return 1.0
    if pd.isna(date_obj):
        return 1.0

    date_str = date_obj.strftime("%Y-%m-%d")
    url = f"https://api.frankfurter.dev/v1/{date_str}"
    try:
        resp = requests.get(
            url,
            params={"base": from_ccy, "symbols": to_ccy},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        rate = float(data["rates"][to_ccy])
        return rate if rate > 0 else 1.0
    except Exception:
        return 1.0


@st.cache_data(show_spinner=False)
def fetch_latest_fx(from_ccy: str, to_ccy: str) -> float:
    """
    Get latest FX rate (from_ccy → to_ccy).
    Returns rate as 'to_ccy per 1 from_ccy'.
    """
    from_ccy = (from_ccy or "").upper()
    to_ccy = (to_ccy or "").upper()
    if not from_ccy or not to_ccy:
        return 1.0
    if from_ccy == to_ccy:
        return 1.0

    url = "https://api.frankfurter.dev/v1/latest"
    try:
        resp = requests.get(
            url,
            params={"base": from_ccy, "symbols": to_ccy},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        rate = float(data["rates"][to_ccy])
        return rate if rate > 0 else 1.0
    except Exception:
        return 1.0


# =========================
# CORE FX CALCULATIONS
# =========================

def recompute_average_cost(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average-cost model per foreign currency.

    New rule:
    - For trades on the SAME DATE and SAME CURRENCY:
      Buy/Credit are processed BEFORE Sell/Debit so that
      same-day buys affect the average cost used for sells on that day.

    Conventions:
    - foreign_amount: +ve for Buy/Credit (inflow), -ve for Sell/Debit (outflow)
    - fee_foreign: fee in foreign currency, converted to base using fx_rate
    - fx_rate: base per 1 foreign

    Also computes per-row:
    - base_amount
    - fee_base
    - sale_proceeds_base (for sells/debits)
    - sale_cost_base (cost basis of units sold)
    """
    if trades_df.empty:
        return trades_df

    df = trades_df.copy()

    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Keep original order as a tiebreaker
    df = df.reset_index(drop=False).rename(columns={"index": "_orig_idx"})

    # Define intra-day execution order:
    # Buy/Credit (0) -> Sell/Debit (1) -> everything else (2)
    order_map = {
        "BUY": 0,
        "CREDIT": 0,
        "SELL": 1,
        "DEBIT": 1,
    }
    df["txn_order"] = (
        df["txn_type"]
        .astype(str)
        .str.upper()
        .map(order_map)
        .fillna(2)
    )

    # Sort:
    # - by date
    # - by currency (optional but nice to keep things grouped)
    # - by txn_order (Buy/Credit first, then Sell/Debit)
    # - by original index as final tiebreaker
    df = df.sort_values(["date", "foreign_ccy", "txn_order", "_orig_idx"])

    # Ensure columns exist
    for col in ["foreign_amount", "fx_rate", "fee_foreign"]:
        if col not in df.columns:
            df[col] = 0.0

    df["foreign_amount"] = (
        df["foreign_amount"]
        .astype(str).str.replace(",", ".", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )
    df["fx_rate"] = (
        df["fx_rate"]
        .astype(str).str.replace(",", ".", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )
    df["fee_foreign"] = (
        df["fee_foreign"]
        .astype(str).str.replace(",", ".", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )

    df["base_amount"] = df["foreign_amount"] * df["fx_rate"]
    df["fee_base"] = df["fee_foreign"] * df["fx_rate"]

    df["realized_pl_base"] = 0.0
    df["running_position"] = 0.0
    df["running_cost_base"] = 0.0
    df["running_avg_cost"] = 0.0

    # Track cost & proceeds for realized P/L reporting
    df["sale_proceeds_base"] = 0.0
    df["sale_cost_base"] = 0.0

    positions: dict[str, dict[str, float]] = {}

    for idx, row in df.iterrows():
        ccy = str(row["foreign_ccy"]).upper()
        if ccy not in positions:
            positions[ccy] = {"pos": 0.0, "cost": 0.0}

        pos = positions[ccy]["pos"]
        cost = positions[ccy]["cost"]

        txn_type = str(row["txn_type"]).upper()
        foreign_amount = float(row["foreign_amount"])
        fx_rate = float(row["fx_rate"])
        fee_base = float(row["fee_base"])

        realized_pl = 0.0
        sale_proceeds = 0.0
        sale_cost = 0.0

        if txn_type in ["BUY", "CREDIT"]:
            units = abs(foreign_amount)
            base_amount = units * fx_rate
            cost_increase = base_amount + fee_base
            new_pos = pos + units
            new_cost = cost + cost_increase

        elif txn_type in ["SELL", "DEBIT"]:
            units = abs(foreign_amount)
            if pos > 0:
                avg_cost = cost / pos
            else:
                avg_cost = 0.0

            sale_cost = avg_cost * units
            sale_proceeds = units * fx_rate - fee_base
            realized_pl = sale_proceeds - sale_cost

            new_pos = pos - units
            new_cost = cost - sale_cost

        else:
            new_pos = pos
            new_cost = cost

        positions[ccy]["pos"] = new_pos
        positions[ccy]["cost"] = new_cost

        df.at[idx, "realized_pl_base"] = realized_pl
        df.at[idx, "running_position"] = new_pos
        df.at[idx, "running_cost_base"] = new_cost
        df.at[idx, "running_avg_cost"] = (new_cost / new_pos) if new_pos != 0 else 0.0
        df.at[idx, "sale_proceeds_base"] = sale_proceeds
        df.at[idx, "sale_cost_base"] = sale_cost

    # (Optional) drop helper columns before returning
    df = df.drop(columns=["txn_order", "_orig_idx"])

    return df


def load_trades(base_ccy: str, recalc_fx: bool) -> pd.DataFrame:
    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE, sep=";")
    else:
        df = pd.DataFrame(columns=FX_INPUT_COLS)

    # Ensure columns exist
    for col in FX_INPUT_COLS:
        if col not in df.columns:
            df[col] = "" if col in ["foreign_ccy", "txn_type", "memo"] else 0.0

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Re-fetch FX for all rows if base currency changed
    if recalc_fx and base_ccy:
        base_ccy = base_ccy.upper()
        df["foreign_ccy"] = df["foreign_ccy"].astype(str).str.upper()

        # Only rows with a date and currency
        mask = df["date"].notna() & df["foreign_ccy"].astype(bool)
        df_valid = df[mask].copy()

        # Bulk-fetch per foreign currency
        for fccy, group in df_valid.groupby("foreign_ccy"):
            # list of datetime.date, not datetime
            trade_dates = list(group["date"].dt.date)
            rates_map = fetch_historical_fx_bulk(trade_dates, fccy, base_ccy)

            # Assign rates back to df
            for idx, d in zip(group.index, trade_dates):
                df.at[idx, "fx_rate"] = rates_map.get(d, 1.0)

    df = recompute_average_cost(df)
    return df


def save_trades(df: pd.DataFrame) -> None:
    to_save = df[FX_INPUT_COLS].copy()
    to_save["date"] = pd.to_datetime(to_save["date"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    to_save.to_csv(DATA_FILE, index=False, sep=";")


def build_positions_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-currency summary: position, total_cost_base, avg_cost, realized P/L.
    Only returns rows where position != 0 (within a tiny tolerance).
    """
    if trades_df.empty:
        return pd.DataFrame(
            columns=[
                "foreign_ccy",
                "position",
                "total_cost_base",
                "avg_cost_base",
                "realized_pl_base",
            ]
        )

    # Use the current order from recompute_average_cost (already sorted correctly)
    latest = trades_df.groupby("foreign_ccy", as_index=False).tail(1)

    realized = (
        trades_df.groupby("foreign_ccy")["realized_pl_base"]
        .sum()
        .reset_index()
    )

    res = latest[["foreign_ccy", "running_position", "running_cost_base", "running_avg_cost"]].copy()
    res = res.rename(
        columns={
            "running_position": "position",
            "running_cost_base": "total_cost_base",
            "running_avg_cost": "avg_cost_base",
        }
    )
    res = res.merge(realized, on="foreign_ccy", how="left")
    res["realized_pl_base"].fillna(0.0, inplace=True)

    # 🔑 NEW: drop zero-positions (with float tolerance)
    res = res[res["position"].abs() >= 1e-9].reset_index(drop=True)

    return res


def build_holdings_valuation(summary_df: pd.DataFrame, base_ccy: str) -> pd.DataFrame:
    """
    For each foreign currency with non-zero position, get latest FX and compute market value.
    """
    rows = []
    base_ccy = base_ccy.upper()
    for _, r in summary_df.iterrows():
        pos = float(r["position"])
        if abs(pos) < 1e-9:
            continue
        ccy = str(r["foreign_ccy"]).upper()
        fx = fetch_latest_fx(ccy, base_ccy)
        mv = pos * fx
        rows.append(
            {
                "foreign_ccy": ccy,
                "units": pos,
                "fx_rate": fx,
                "market_value_base": mv,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["foreign_ccy", "units", "fx_rate", "market_value_base"])
    return pd.DataFrame(rows)


def parse_float(text: str, field_name: str):
    text = (text or "").strip()
    if text == "":
        st.error(f"{field_name} is required.")
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        st.error(f"{field_name} must be a number.")
        return None


# =========================
# STATE & SETTINGS
# =========================

_settings = load_settings()
_last_base = _settings.get("base_ccy", DEFAULT_BASE_CCY)

if "history_edit_mode" not in st.session_state:
    st.session_state.history_edit_mode = False

with st.sidebar:
    st.header("⚙️ Settings")

    base_ccy = st.text_input("Base Currency", value=_last_base).upper()

    # Refresh latest holdings valuation (clear FX cache)
    if st.button("🔄 Fetch current FX rates"):
        fetch_latest_fx.clear()
        st.success("Latest FX rates cleared from cache; holdings will refresh on next load.")
        st.rerun()

    hide_values = st.toggle("Hide values", value=False)


    st.divider()

    # Download current FX trades
    if DATA_FILE.exists():
        st.download_button(
            label="⬇️ Download FX trades",
            data=DATA_FILE.read_bytes(),
            file_name="fx_trades.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.caption("No FX trades file yet.")

    # Upload FX trades
    uploaded_file = st.file_uploader(
        "Upload/Replace FX trades",
        type=["csv"],
        accept_multiple_files=False,
    )
    if uploaded_file is not None:
        if st.button("⬆️ Upload", use_container_width=True):
            try:
                DATA_FILE.write_bytes(uploaded_file.getvalue())
                st.success("FX trades uploaded.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save uploaded file: {e}")

    st.divider()

    st.caption(f"Data file: `{DATA_FILE}`")

recalc_fx = base_ccy != _last_base

trades_df = load_trades(base_ccy, recalc_fx=recalc_fx)
if recalc_fx:
    save_trades(trades_df)
    _settings["base_ccy"] = base_ccy
    save_settings(_settings)

positions_summary = build_positions_summary(trades_df)

# =========================
# GLOBAL HEADER (TOTAL HOLDINGS + TOTAL UNREALIZED P/L)
# =========================

if positions_summary.empty or positions_summary["position"].abs().sum() == 0:
    total_holdings_value = 0.0
    total_unrealized_pl = 0.0
else:
    holdings_header_df = build_holdings_valuation(positions_summary, base_ccy)
    merged_header = positions_summary.merge(
        holdings_header_df, on="foreign_ccy", how="left"
    )
    merged_header["market_value_base"].fillna(0.0, inplace=True)
    merged_header["total_cost_base"].fillna(0.0, inplace=True)
    merged_header["unrealized_pl"] = (
        merged_header["market_value_base"] - merged_header["total_cost_base"]
    )
    total_holdings_value = merged_header["market_value_base"].sum()
    total_unrealized_pl = merged_header["unrealized_pl"].sum()


def mask_number(val: float, mask: bool, decimals: int = 0):
    if mask:
        return "••••••"
    return f"{val:,.{decimals}f}"


header_html = f"""
<div class="mobile-header">
  <div class="mobile-header-title">💶 FX Holdings</div>
  <div class="kpi-row">
    <div class="kpi">
      <div class="kpi-label">Total holdings value ({base_ccy})</div>
      <div class="kpi-value">{mask_number(total_holdings_value, hide_values, 2)}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Total unrealized P&amp;L ({base_ccy})</div>
      <div class="kpi-value">{mask_number(total_unrealized_pl, hide_values, 2)}</div>
    </div>
  </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)
st.markdown("---")

# =========================
# MAIN TABS
# =========================

tab_holdings, tab_new, tab_history, tab_pl = st.tabs(
    ["📊 Holdings", "➕ New Transaction", "🧾 History", "💰 Realized P/L"]
)

# ---------- TAB: HOLDINGS ----------
with tab_holdings:

    if positions_summary.empty:
        st.info("No positions yet. Add an FX transaction first.")
    else:
        holdings_df = build_holdings_valuation(positions_summary, base_ccy)
        merged = positions_summary.merge(holdings_df, on="foreign_ccy", how="left")

        merged["market_value_base"].fillna(0.0, inplace=True)
        merged["total_cost_base"].fillna(0.0, inplace=True)
        merged["unrealized_pl"] = (
            merged["market_value_base"] - merged["total_cost_base"]
        )

        display = merged[
            [
                "foreign_ccy",
                "position",
                "avg_cost_base",
                "fx_rate",
                "total_cost_base",
                "market_value_base",
                "unrealized_pl",
            ]
        ].copy()

        display = display.sort_values("market_value_base", ascending=False)
        
        display = display.rename(
            columns={
                "foreign_ccy": "Currency",
                "position": "Amount",
                "avg_cost_base": f"Avg Cost ({base_ccy})",
                "fx_rate": f"Current FX rate (to {base_ccy})",
                "total_cost_base": f"Cost basis ({base_ccy})",
                "market_value_base": f"Market value ({base_ccy})",
                "unrealized_pl": f"Unrealized P&L ({base_ccy})",
            }
        )

        if hide_values:
            for col in display.columns:
                if col != "Currency":
                    display[col] = "••••••"
        else:
            for col in display.columns:
                if col == "Amount":
                    # No decimals for quantity
                    display[col] = display[col].apply(lambda x: f"{x:,.0f}")

                elif col == f"Avg Cost ({base_ccy})":
                    # Avg cost with 4 decimals
                    display[col] = display[col].apply(lambda x: f"{x:,.4f}")

                elif col == f"Current FX rate (to {base_ccy})":
                    # FX rate with 4 decimals
                    display[col] = display[col].apply(lambda x: f"{x:,.4f}")

                elif col != "Currency" and np.issubdtype(display[col].dtype, np.number):
                    # All other numeric values with 2 decimals
                    display[col] = display[col].apply(lambda x: f"{x:,.2f}")            

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )


# ---------- TAB: NEW TRANSACTION ----------
with tab_new:
    
    auto_fx = st.checkbox(
        "Auto FX",
        value=True,
        help="If checked, FX rate is fetched for the trade date, foreign currency, and current base.",
    )

    with st.form("fx_trade_form", clear_on_submit=True):
        col_l, col_r = st.columns(2)

        # Row 1
        with col_l:
            t_date = st.date_input("Date", value=date.today())
        with col_r:
            t_ccy = st.text_input("Currency").upper()

        # Row 2
        with col_l:
            t_type_ui = st.selectbox(
                "Transaction type",
                [
                    "Debit",
                    "Credit",
                    "FX Buy",
                    "FX Sell",
                ],
            )
        with col_r:
            t_amount_str = st.text_input(
                "Amount",
            )

        # Row 3
        with col_l:
            t_fx_str = st.text_input(
                f"FX rate",
                disabled=auto_fx,
            )
        with col_r:
            t_fee_str = st.text_input(
                "Exchange fee (in foreign currency)",
            )

        memo = st.text_input("Memo / Note")

        submitted = st.form_submit_button("💾 Save transaction", type="primary")

    if submitted:
        valid = True

        if not t_ccy:
            st.error("Foreign currency is required.")
            valid = False

        amount = parse_float(t_amount_str, "Amount")
        if amount is None or amount <= 0:
            valid = False

        fee_foreign = 0.0
        if t_fee_str.strip() != "":
            tmp = parse_float(t_fee_str, "Fee")
            if tmp is None or tmp < 0:
                valid = False
            else:
                fee_foreign = tmp

        if not auto_fx:
            fx_rate = parse_float(t_fx_str, "FX rate")
            if fx_rate is None or fx_rate <= 0:
                st.error("FX rate must be greater than zero.")
                valid = False
        else:
            fx_rate = None

        if valid and auto_fx:
            dt = datetime.combine(t_date, datetime.min.time())
            fx_rate = fetch_historical_fx(dt, t_ccy, base_ccy)
            if fx_rate <= 0:
                st.error("Failed to fetch a valid FX rate.")
                valid = False

        if valid:
            # Map UI types to internal type + sign of amount
            if t_type_ui == "Debit":
                txn_type = "Debit"
                signed_amount = -abs(amount)
            elif t_type_ui == "Credit":
                txn_type = "Credit"
                signed_amount = abs(amount)
            elif t_type_ui == "FX Buy":
                txn_type = "Buy"
                signed_amount = abs(amount)
            else:  # Sell foreign for base
                txn_type = "Sell"
                signed_amount = -abs(amount)

            new_row = {
                "date": datetime.combine(t_date, datetime.min.time()),
                "foreign_ccy": t_ccy.upper(),
                "txn_type": txn_type,
                "foreign_amount": signed_amount,
                "fx_rate": fx_rate,
                "fee_foreign": fee_foreign,
                "memo": memo,
            }

            # Append to existing
            if DATA_FILE.exists():
                existing = pd.read_csv(DATA_FILE, sep=";")
            else:
                existing = pd.DataFrame(columns=FX_INPUT_COLS)

            combined = pd.concat(
                [
                    existing[FX_INPUT_COLS]
                    if not existing.empty
                    else pd.DataFrame(columns=FX_INPUT_COLS),
                    pd.DataFrame([new_row]),
                ],
                ignore_index=True,
            )
            save_trades(combined)

            st.success("Transaction saved.")
            st.rerun()


# ---------- TAB: HISTORY ----------
with tab_history:
    spacer, toggle_col = st.columns([5, 1])
    with toggle_col:
        current_edit = st.session_state.get("history_edit_mode", False)
        toggle_val = st.toggle(
            "Enable Edit Mode",
            value=current_edit,
            key="history_edit_toggle",
            help="Toggle to enable editing (add/remove/change rows). Remember to save.",
        )
        st.session_state.history_edit_mode = toggle_val

    if trades_df.empty:
        st.info("No FX transactions recorded yet.")
    else:
        # Columns: Date; Currency; Transaction type; Amount; Fx Rate; Fees; Memo
        history_cols = [
            "date",
            "foreign_ccy",
            "txn_type",
            "foreign_amount",
            "fx_rate",
            "fee_foreign",
            "memo",
        ]

        if not st.session_state.history_edit_mode:
            # --- READ-ONLY VIEW ---
            display = trades_df[history_cols].copy()
            display["date"] = display["date"].dt.date
            display = display.rename(
                columns={
                    "date": "Date",
                    "foreign_ccy": "Currency",
                    "txn_type": "Transaction type",
                    "foreign_amount": "Amount",
                    "fx_rate": "Fx Rate",
                    "fee_foreign": "Fees",
                    "memo": "Memo",
                }
            )

            if hide_values:
                for col in ["Amount", "Fx Rate", "Fees"]:
                    display[col] = "••••••"

            st.dataframe(
                display.sort_values("Date", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

        else:
            # --- EDITABLE VIEW (slider ON) ---
            editable = trades_df[history_cols].copy()
            # Show date as string for editing
            editable["date"] = editable["date"].dt.strftime("%Y-%m-%d")

            editable = editable.rename(
                columns={
                    "date": "Date",
                    "foreign_ccy": "Currency",
                    "Transaction type": "txn_type",
                    "txn_type": "Transaction type",
                    "foreign_amount": "Amount",
                    "fx_rate": "Fx Rate",
                    "fee_foreign": "Fees",
                    "memo": "Memo",
                }
            )

            edited_df = st.data_editor(
                editable,
                num_rows="dynamic",  # allow add/remove rows
                use_container_width=True,
                key="history_editor",
            )

            st.caption("Tip: Use the slider above to exit edit mode after saving.")

            # Small save button under the editor
            if st.button("💾 Save changes", key="save_history_btn"):
                try:
                    # Map back to internal column names
                    updated = edited_df.rename(
                        columns={
                            "Date": "date",
                            "Currency": "foreign_ccy",
                            "Transaction type": "txn_type",
                            "Amount": "foreign_amount",
                            "Fx Rate": "fx_rate",
                            "Fees": "fee_foreign",
                            "Memo": "memo",
                        }
                    ).copy()

                    # Parse date
                    updated["date"] = pd.to_datetime(updated["date"], errors="coerce")

                    # Normalize currency & txn type
                    updated["foreign_ccy"] = (
                        updated["foreign_ccy"].astype(str).str.upper()
                    )
                    updated["txn_type"] = updated["txn_type"].astype(str)

                    # Numeric fields
                    for col in ["foreign_amount", "fx_rate", "fee_foreign"]:
                        updated[col] = (
                            updated[col]
                            .astype(str)
                            .str.replace(",", ".", regex=False)
                            .pipe(pd.to_numeric, errors="coerce")
                            .fillna(0.0)
                        )

                    # Drop completely empty / invalid rows
                    mask_valid = (
                        updated["date"].notna()
                        & updated["foreign_ccy"].astype(str).str.len().gt(0)
                        & updated["txn_type"].astype(str).str.len().gt(0)
                    )
                    updated = updated[mask_valid].copy()

                    # This becomes the new raw trades dataset
                    new_trades = updated[history_cols].copy()
                    new_trades = recompute_average_cost(new_trades)
                    save_trades(new_trades)

                    st.success("History updated.")
                    # Keep edit mode state as-is; user can slide it off when done
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to save changes: {e}")


# ---------- TAB: REALIZED P/L ----------
with tab_pl:

    if trades_df.empty:
        st.info("No transactions yet.")
    else:
        # Date range bounds from data
        min_dt = trades_df["date"].min().date()
        max_dt = trades_df["date"].max().date()

        # Use the latest data date (or today, whichever is earlier) as reference
        today = dt.date.today()
        reference_date = min(max_dt, today)

        # Date range selector (dropdown)
        range_options = ["Year to date", "Last calendar year", "Last 12 months", "Custom range"]
        selected_range = st.selectbox("Date range", range_options, index=0)

        # Compute start_date and end_date based on selection
        if selected_range == "Year to date":
            year_start = dt.date(reference_date.year, 1, 1)
            start_date = max(year_start, min_dt)
            end_date = reference_date

        elif selected_range == "Last calendar year":
            last_year = reference_date.year - 1
            start_candidate = dt.date(last_year, 1, 1)
            end_candidate = dt.date(last_year, 12, 31)

            # Clamp to available data
            start_date = max(start_candidate, min_dt)
            end_date = min(end_candidate, max_dt)

        elif selected_range == "Last 12 months":
            end_date = reference_date
            start_candidate = end_date - dt.timedelta(days=365)
            start_date = max(start_candidate, min_dt)

        else:  # "Custom range"
            col1, col2 = st.columns(2)
            start_date = col1.date_input("From", value=min_dt, min_value=min_dt, max_value=max_dt)
            end_date = col2.date_input("To", value=max_dt, min_value=min_dt, max_value=max_dt)

        if start_date > end_date:
            st.error("Start date cannot be after end date.")
        else:
            # Filter data based on computed range
            mask_range = (
                (trades_df["date"].dt.date >= start_date)
                & (trades_df["date"].dt.date <= end_date)
            )
            df_range = trades_df[mask_range].copy()

            # Only Sell/Debit rows have realized P/L
            df_real = df_range[df_range["txn_type"].isin(["Sell", "Debit"])].copy()

            if df_real.empty:
                st.info("No realized P/L in the selected period.")
            else:
                grouped = (
                    df_real.groupby("foreign_ccy")
                    .agg(
                        Qty=("foreign_amount", lambda x: np.abs(x).sum()),
                        Cost=("sale_cost_base", "sum"),
                        Proceeds=("sale_proceeds_base", "sum"),
                        Realized=("realized_pl_base", "sum"),
                    )
                    .reset_index()
                )

                total_realized_period = grouped["Realized"].sum()

                if hide_values:
                    total_str = "••••••"
                else:
                    total_str = f"{total_realized_period:,.2f}"

                st.markdown(
                    f"<div style='font-size:0.85rem; opacity:0.8;'>Total realized P&L "
                    f"({base_ccy}) from {start_date} to {end_date}</div>"
                    f"<div style='font-size:1.2rem; font-weight:700; margin-bottom:0.2rem;'>{total_str}</div>",
                    unsafe_allow_html=True,
                )

                display = grouped.copy()
                display = display.rename(
                    columns={
                        "foreign_ccy": "Currency",
                        "Qty": "Qty",
                        "Cost": f"Cost ({base_ccy})",
                        "Proceeds": f"Proceeds ({base_ccy})",
                        "Realized": f"Realized P&L ({base_ccy})",
                    }
                )

                if hide_values:
                    for col in display.columns:
                        if col != "Currency":
                            display[col] = "••••••"
                else:
                    for col in display.columns:
                        if col != "Currency" and np.issubdtype(display[col].dtype, np.number):
                            display[col] = display[col].apply(lambda x: f"{x:,.2f}")

                st.dataframe(
                    display,
                    use_container_width=True,
                    hide_index=True,
                )





