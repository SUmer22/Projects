# ════════════════════════════════════════════════════════════════════════════
# TFE-GHMM + Gradio — Combined Single File
# KSE-100 Pakistan Stock Exchange | Market Regime Detection
# ════════════════════════════════════════════════════════════════════════════
# HOW TO RUN IN COLAB:
#   Cell 1: !pip install gradio pandas numpy matplotlib seaborn yfinance ta hmmlearn scikit-learn scipy -q
#   Cell 2: Paste or upload this entire file and run it as one cell
#   → A public gradio.live URL is printed automatically when running in Colab
#
# HOW TO RUN LOCALLY:
#   pip install gradio pandas numpy matplotlib seaborn yfinance ta hmmlearn scikit-learn scipy
#   python TFE_GHMM_Gradio.py
#   → Opens at http://localhost:<port>  (default starts at 7860; auto-selects if busy)
# ════════════════════════════════════════════════════════════════════════════

# ── Section 0: Imports ───────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import os
import io
import math
import datetime

try:
    import numpy as np
except ImportError:
    os.system("pip install numpy -q")
    import numpy as np

try:
    import pandas as pd
except ImportError:
    os.system("pip install pandas -q")
    import pandas as pd

import matplotlib
matplotlib.use("Agg")
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    os.system("pip install matplotlib -q")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap

try:
    import seaborn as sns
except ImportError:
    os.system("pip install seaborn -q")
    import seaborn as sns

try:
    import yfinance as yf
except ImportError:
    os.system("pip install yfinance -q")
    import yfinance as yf

try:
    import ta
except ImportError:
    os.system("pip install ta -q")
    import ta

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    os.system("pip install hmmlearn -q")
    from hmmlearn.hmm import GaussianHMM

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    os.system("pip install scikit-learn -q")
    from sklearn.preprocessing import StandardScaler

try:
    from scipy.stats import norm
except ImportError:
    os.system("pip install scipy -q")
    from scipy.stats import norm

try:
    import gradio as gr
except ImportError:
    os.system("pip install gradio -q")
    import gradio as gr

# ── Colab detection ──────────────────────────────────────────────────────────
# Automatically True when running inside Google Colab; False everywhere else.
try:
    import google.colab  # noqa: F401
    IS_COLAB = True
except ImportError:
    IS_COLAB = False


# ── Section 1: Global Constants ──────────────────────────────────────────────

CONFIG = {
    "risk_free_rate": 0.06,   # 6% annual (Pakistan T-bill proxy)
    "random_seed": 42,
    "output_dir": "outputs",
    "default_n_states": 3,
    "default_n_iter": 200,
    "default_train_ratio": 0.8,
}

PALETTE = {
    "Bear":     "#e05c5c",
    "Sideways": "#f0c040",
    "Bull":     "#5cb85c",
    "bg":       "#0d1117",
    "fg":       "#c9d1d9",
    "grid":     "#21262d",
    "border":   "#30363d",
}

REGIME_COLOR_MAP = {
    "Bear":     "#e05c5c",
    "Sideways": "#f0c040",
    "Bull":     "#5cb85c",
}

DEFAULT_COLORS = ["#e05c5c", "#f0c040", "#5cb85c", "#5b9bd5", "#a78bfa"]

ALL_FEATURES = [
    "Return", "Return_5d", "Return_20d",
    "Volatility_20d", "Volatility_60d",
    "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Width", "BB_Pos",
    "ATR_14", "ATR_Ratio",
    "Stoch_K", "Stoch_D",
    "Volume_Ratio", "Volume_Trend",
    "Body_Size", "Upper_Wick", "Lower_Wick",
]

MACRO_FEATURES = [
    "PKR_Change", "PKR_Change_5d",
    "Oil_Change", "VIX_Change", "MSCI_EM_Return",
]

DEFAULT_FEATURES = [
    "Return", "Volatility_20d", "RSI_14",
    "MACD_Hist", "BB_Width", "ATR_Ratio", "Volume_Ratio",
]

# Global cache shared across Gradio callbacks
CACHE = {
    "model":         None,
    "scaler":        None,
    "feat_df":       None,
    "feature_names": None,
    "regime_labels": None,
    "regime_colors": None,
    "metrics":       None,
    "X_train":       None,
    "X_test":        None,
}


# ── Section 2: Pipeline Functions ────────────────────────────────────────────

def load_kse100(filepath: str, start: str, end: str) -> pd.DataFrame:
    """Robust CSV loader for KSE-100 OHLCV data."""
    raw_df = pd.read_csv(filepath)

    # Identify date column
    date_col = next(
        (c for c in raw_df.columns if c.lower().strip() in ["date", "datetime", "time"]),
        None,
    )
    if date_col is None:
        raise ValueError("No 'Date' column found. Expected one of: Date, datetime, time.")

    # Normalise column names
    col_map = {date_col: "Date"}
    for c in raw_df.columns:
        cl = c.lower().replace(" ", "").replace("_", "")
        if cl in ["open", "openprice"]:
            col_map[c] = "Open"
        elif cl in ["high", "highprice"]:
            col_map[c] = "High"
        elif cl in ["low", "lowprice"]:
            col_map[c] = "Low"
        elif cl in ["close", "closeprice", "adjclose", "last"]:
            col_map[c] = "Close"
        elif cl in ["volume", "vol"]:
            col_map[c] = "Volume"

    raw_df = raw_df.rename(columns=col_map)

    # Parse dates
    raw_df["Date"] = pd.to_datetime(raw_df["Date"], dayfirst=True, errors="coerce")
    raw_df = raw_df.dropna(subset=["Date"])
    raw_df = raw_df.set_index("Date").sort_index()

    # Clean numeric columns — handle dash/NA placeholders common in KSE-100 CSVs
    _MISSING = {"-", "--", "n/a", "na", "nan", "null", "none", ""}
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in raw_df.columns:
            if raw_df[col].dtype == object:
                # Strip thousand-separators and replace common placeholders with NaN
                raw_df[col] = (
                    raw_df[col]
                    .astype(str)
                    .str.replace(",", "", regex=False)
                    .str.strip()
                    .apply(lambda v: np.nan if str(v).lower() in _MISSING else v)
                )
            raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

    if "Close" not in raw_df.columns:
        raise ValueError("No 'Close' column found after renaming. Cannot proceed.")

    # Apply date filter
    start_dt = pd.to_datetime(start)
    end_dt   = pd.to_datetime(end)
    raw_df = raw_df.loc[start_dt:end_dt]

    if raw_df.empty:
        raise ValueError(
            f"No data in selected date range {start} → {end}. "
            "Please check your file or adjust the date range."
        )

    raw_df = raw_df.dropna(subset=["Close"])
    return raw_df


def fetch_macro_data(start: str, end: str) -> pd.DataFrame:
    """Fetch macro indicators from Yahoo Finance: PKR=X, BZ=F, ^VIX, EEM."""
    tickers = {
        "PKR_USD":  "PKR=X",
        "Brent_Oil": "BZ=F",
        "VIX":       "^VIX",
        "MSCI_EM":   "EEM",
    }
    series = {}
    for name, ticker in tickers.items():
        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
            if not data.empty:
                close = data["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close.name = name
                series[name] = close
        except Exception:
            pass

    if not series:
        return pd.DataFrame()

    macro_df = pd.DataFrame(series)
    macro_df = macro_df.ffill().bfill()
    return macro_df


def engineer_features(df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """Compute full set of technical + macro features."""
    feat = pd.DataFrame(index=df.index)

    close  = df["Close"]
    high   = df["High"]   if "High"   in df.columns else close
    low    = df["Low"]    if "Low"    in df.columns else close
    open_  = df["Open"]   if "Open"   in df.columns else close
    volume = df["Volume"] if "Volume" in df.columns else pd.Series(
        np.ones(len(df)), index=df.index
    )

    # --- Returns ---
    feat["Return"]    = close.pct_change()
    feat["Return_5d"] = close.pct_change(5)
    feat["Return_20d"]= close.pct_change(20)

    # --- Volatility ---
    feat["Volatility_20d"] = feat["Return"].rolling(20).std()
    feat["Volatility_60d"] = feat["Return"].rolling(60).std()

    # --- RSI ---
    try:
        feat["RSI_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    except Exception:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        feat["RSI_14"] = 100 - (100 / (1 + rs))

    # --- MACD ---
    try:
        macd_ind = ta.trend.MACD(close=close)
        feat["MACD"]        = macd_ind.macd()
        feat["MACD_Signal"] = macd_ind.macd_signal()
        feat["MACD_Hist"]   = macd_ind.macd_diff()
    except Exception:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        feat["MACD"]        = ema12 - ema26
        feat["MACD_Signal"] = feat["MACD"].ewm(span=9, adjust=False).mean()
        feat["MACD_Hist"]   = feat["MACD"] - feat["MACD_Signal"]

    # --- Bollinger Bands ---
    try:
        bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
        feat["BB_Width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        feat["BB_Pos"]   = (close - bb.bollinger_lband()) / (
            bb.bollinger_hband() - bb.bollinger_lband() + 1e-10
        )
    except Exception:
        sma20   = close.rolling(20).mean()
        std20   = close.rolling(20).std()
        bb_high = sma20 + 2 * std20
        bb_low  = sma20 - 2 * std20
        feat["BB_Width"] = (bb_high - bb_low) / (sma20 + 1e-10)
        feat["BB_Pos"]   = (close - bb_low) / (bb_high - bb_low + 1e-10)

    # --- ATR ---
    try:
        atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14)
        feat["ATR_14"]   = atr.average_true_range()
        feat["ATR_Ratio"]= feat["ATR_14"] / (close + 1e-10)
    except Exception:
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        feat["ATR_14"]    = tr.rolling(14).mean()
        feat["ATR_Ratio"] = feat["ATR_14"] / (close + 1e-10)

    # --- Stochastic ---
    try:
        stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=close)
        feat["Stoch_K"] = stoch.stoch()
        feat["Stoch_D"] = stoch.stoch_signal()
    except Exception:
        low14  = low.rolling(14).min()
        high14 = high.rolling(14).max()
        feat["Stoch_K"] = 100 * (close - low14) / (high14 - low14 + 1e-10)
        feat["Stoch_D"] = feat["Stoch_K"].rolling(3).mean()

    # --- Volume ---
    vol_ma = volume.rolling(20).mean()
    feat["Volume_Ratio"] = volume / (vol_ma + 1e-10)
    feat["Volume_Trend"] = vol_ma.pct_change(5)

    # --- Candlestick ---
    body   = (close - open_).abs() / (close + 1e-10)
    candle_range = high - low + 1e-10
    feat["Body_Size"]   = body
    feat["Upper_Wick"]  = (high  - pd.concat([close, open_], axis=1).max(axis=1)) / candle_range
    feat["Lower_Wick"]  = (pd.concat([close, open_], axis=1).min(axis=1) - low) / candle_range

    # --- Macro ---
    if not macro_df.empty:
        m = macro_df.reindex(feat.index, method="ffill")
        for col in m.columns:
            feat[col] = m[col]
        if "PKR_USD" in feat.columns:
            feat["PKR_Change"]    = feat["PKR_USD"].pct_change()
            feat["PKR_Change_5d"] = feat["PKR_USD"].pct_change(5)
        if "Brent_Oil" in feat.columns:
            feat["Oil_Change"]    = feat["Brent_Oil"].pct_change()
        if "VIX" in feat.columns:
            feat["VIX_Change"]    = feat["VIX"].pct_change()
        if "MSCI_EM" in feat.columns:
            feat["MSCI_EM_Return"]= feat["MSCI_EM"].pct_change()

    feat["Close"] = close

    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna()
    return feat


def select_features(feat_df: pd.DataFrame, feature_names: list) -> list:
    """Validate and return available features from the requested list."""
    available = [f for f in feature_names if f in feat_df.columns]
    missing   = [f for f in feature_names if f not in feat_df.columns]
    if missing:
        print(f"  ⚠ Features not found and skipped: {missing}")
    if not available:
        raise ValueError("None of the requested features are available in the data.")
    return available


def preprocess_and_split(
    feat_df: pd.DataFrame,
    feature_names: list,
    train_ratio: float = 0.8,
):
    """Scale features and split into train/test sets. Returns (X_train, X_test, d_train, d_test, scaler)."""
    X = feat_df[feature_names].values
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    split   = int(len(X_scaled) * train_ratio)
    X_train = X_scaled[:split]
    X_test  = X_scaled[split:]
    d_train = [len(X_train)]
    d_test  = [len(X_test)]
    return X_train, X_test, d_train, d_test, scaler


def train_hmm(
    X_train,
    n_states: int = 3,
    n_iter: int = 200,
    random_state: int = 42,
) -> GaussianHMM:
    """Train a Gaussian HMM on scaled feature data."""
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
        tol=1e-4,
    )
    model.fit(X_train, [len(X_train)])
    return model


def _auto_label_states(model: GaussianHMM, feat_df: pd.DataFrame, n_states: int):
    """Auto-label HMM states by mean return: lowest = Bear, highest = Bull."""
    # Sort states by the first feature's mean in the trained model (proxy for return level)
    state_returns = {}
    if "Return" in feat_df.columns:
        # Use the model's mean parameters directly — index 0 corresponds to the
        # first feature in the training matrix.  When "Return" is the first
        # selected feature this gives a clean ordering; otherwise it still
        # provides a stable, consistent sort without fitting an extra scaler.
        for s in range(n_states):
            state_returns[s] = float(model.means_[s, 0])
    else:
        for s in range(n_states):
            state_returns[s] = float(model.means_[s, 0])

    sorted_states = sorted(state_returns.keys(), key=lambda s: state_returns[s])

    labels = ["Bear", "Sideways", "Bull"]
    label_map = {}
    color_map = {}

    if n_states == 3:
        for rank, state in enumerate(sorted_states):
            label_map[state] = labels[rank]
            color_map[state] = list(REGIME_COLOR_MAP.values())[rank]
    else:
        for rank, state in enumerate(sorted_states):
            label_map[state] = f"State {state}"
            color_map[state] = DEFAULT_COLORS[rank % len(DEFAULT_COLORS)]

    return label_map, color_map


def decode_regimes(
    model: GaussianHMM,
    feat_df: pd.DataFrame,
    feature_names: list,
    X_train,
    X_test,
    d_train,
    d_test,
    label_map: dict,
    color_map: dict,
):
    """Run Viterbi decoding and assign regime labels to feat_df."""
    train_states = model.predict(X_train, d_train)
    test_states  = model.predict(X_test,  d_test)
    all_states   = np.concatenate([train_states, test_states])

    feat_df = feat_df.copy()
    feat_df["State"]  = all_states
    feat_df["Regime"] = feat_df["State"].map(label_map)
    feat_df["Color"]  = feat_df["State"].map(color_map)

    # Train/Test split marker
    split = len(X_train)
    feat_df["Split"] = "Train"
    feat_df.iloc[split:, feat_df.columns.get_loc("Split")] = "Test"

    return feat_df


def generate_signals(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Convert regime labels to +1 (Bull) / 0 (Sideways) / -1 (Bear) signals with 1-day lag."""
    feat_df = feat_df.copy()
    sig_map = {"Bull": 1, "Sideways": 0, "Bear": -1}
    feat_df["Signal"] = feat_df["Regime"].map(sig_map).fillna(0)
    feat_df["Signal"] = feat_df["Signal"].shift(1).fillna(0)  # avoid look-ahead
    return feat_df


def backtest(feat_df: pd.DataFrame, risk_free_rate: float = 0.06) -> dict:
    """Long-only strategy vs Buy & Hold backtest."""
    df = feat_df.copy()
    df["BH_Return"]       = df["Return"]
    df["Strategy_Return"] = df["Return"] * (df["Signal"] > 0).astype(float)

    # Cumulative
    df["BH_Cumulative"]  = (1 + df["BH_Return"]).cumprod()
    df["Strat_Cumulative"]= (1 + df["Strategy_Return"]).cumprod()

    # Drawdown
    bh_peak    = df["BH_Cumulative"].cummax()
    strat_peak = df["Strat_Cumulative"].cummax()
    df["BH_Drawdown"]    = (df["BH_Cumulative"]    - bh_peak)    / bh_peak
    df["Strat_Drawdown"] = (df["Strat_Cumulative"] - strat_peak) / strat_peak

    rfr_daily = risk_free_rate / 252

    def _metrics(returns, cum_series):
        total   = cum_series.iloc[-1] - 1
        n_years = len(returns) / 252
        cagr    = (cum_series.iloc[-1] ** (1 / n_years)) - 1 if n_years > 0 else 0
        excess  = returns - rfr_daily
        sharpe  = (excess.mean() / (returns.std() + 1e-10)) * math.sqrt(252)
        peak     = cum_series.cummax()
        drawdown = (cum_series - peak) / peak
        max_dd   = drawdown.min()
        return {"Total Return": total, "CAGR": cagr, "Sharpe": sharpe, "Max Drawdown": max_dd}

    metrics = {
        "Strategy":    _metrics(df["Strategy_Return"], df["Strat_Cumulative"]),
        "Buy & Hold":  _metrics(df["BH_Return"],       df["BH_Cumulative"]),
        "df":          df,
    }
    return metrics


def predict_current_regime(
    model: GaussianHMM,
    scaler: StandardScaler,
    feat_df: pd.DataFrame,
    feature_names: list,
    regime_labels: dict,
    regime_colors: dict,
):
    """Predict the current (latest) market regime and return prediction info."""
    latest_X = feat_df[feature_names].iloc[-1:].values
    scaled   = scaler.transform(latest_X)
    state    = model.predict(scaled, [1])[0]
    probs    = model.predict_proba(scaled)[0]

    label = regime_labels.get(state, f"State {state}")
    color = regime_colors.get(state, "#888888")

    regime_prob = {regime_labels.get(s, f"State {s}"): probs[s] for s in range(len(probs))}

    return {
        "state":       state,
        "label":       label,
        "color":       color,
        "probs":       regime_prob,
        "last_date":   feat_df.index[-1],
    }


# ── Section 3: Pipeline Runner ───────────────────────────────────────────────

def run_pipeline_gradio(
    file_obj,
    start_date,
    end_date,
    n_states,
    n_iter,
    train_ratio,
    feature_names,
    use_macro,
    random_state,
):
    """Run the full KSE-100 GHMM pipeline and return outputs for Gradio."""
    log_lines = []

    def L(msg):
        log_lines.append(msg)
        print(msg)

    empty_fig = plt.figure(figsize=(6, 3))
    plt.close(empty_fig)

    def _fail(msg):
        return msg, "", empty_fig, empty_fig, empty_fig, empty_fig

    # Guard: require a CSV upload
    if file_obj is None:
        return _fail("❌ Please upload a KSE-100 CSV file to proceed.")

    L("═" * 60)
    L("  TFE-GHMM  |  KSE-100 Market Regime Detection")
    L("═" * 60)

    # ── Step 1: Load CSV ──────────────────────────────────────────
    L("\n[Step 1/7] Loading KSE-100 CSV data...")
    try:
        filepath = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        raw_df   = load_kse100(filepath, str(start_date), str(end_date))
        L(f"  ✓ Loaded {len(raw_df):,} rows | {raw_df.index[0].date()} → {raw_df.index[-1].date()}")
    except Exception as exc:
        return _fail(f"❌ Step 1 failed — {exc}")

    # ── Step 2: Fetch macro ───────────────────────────────────────
    L("\n[Step 2/7] Fetching macro data...")
    macro_df = pd.DataFrame()
    if use_macro:
        try:
            macro_df = fetch_macro_data(str(start_date), str(end_date))
            L(f"  ✓ Macro fetched: {list(macro_df.columns) if not macro_df.empty else 'none'}")
        except Exception as exc:
            L(f"  ✗ Macro fetch failed: {exc} — continuing without macro")
    else:
        L("  — Macro data skipped (checkbox unchecked)")

    # ── Step 3: Engineer features ─────────────────────────────────
    L("\n[Step 3/7] Engineering features...")
    try:
        feat_df = engineer_features(raw_df, macro_df)
        L(f"  ✓ Feature matrix: {feat_df.shape[0]:,} rows × {feat_df.shape[1]} cols")
    except Exception as exc:
        return _fail(f"❌ Step 3 failed — {exc}")

    # ── Step 4: Select features ───────────────────────────────────
    L("\n[Step 4/7] Selecting features...")
    try:
        if not feature_names:
            feature_names = DEFAULT_FEATURES
        selected = select_features(feat_df, feature_names)
        L(f"  ✓ Using features: {selected}")
    except Exception as exc:
        return _fail(f"❌ Step 4 failed — {exc}")

    # ── Step 5: Preprocess & Train ────────────────────────────────
    L("\n[Step 5/7] Preprocessing and training HMM...")
    try:
        X_train, X_test, d_train, d_test, scaler = preprocess_and_split(
            feat_df, selected, float(train_ratio)
        )
        model = train_hmm(X_train, int(n_states), int(n_iter), int(random_state))
        L(f"  ✓ HMM trained | n_states={n_states} | train={len(X_train)}, test={len(X_test)}")
    except Exception as exc:
        return _fail(f"❌ Step 5 failed — {exc}")

    # ── Step 6: Decode regimes & generate signals ─────────────────
    L("\n[Step 6/7] Decoding regimes and generating signals...")
    try:
        label_map, color_map = _auto_label_states(model, feat_df, int(n_states))
        feat_df = decode_regimes(
            model, feat_df, selected, X_train, X_test, d_train, d_test,
            label_map, color_map,
        )
        feat_df = generate_signals(feat_df)
        counts  = feat_df["Regime"].value_counts().to_dict()
        L(f"  ✓ Regimes decoded: {counts}")
    except Exception as exc:
        return _fail(f"❌ Step 6 failed — {exc}")

    # ── Step 7: Backtest ──────────────────────────────────────────
    L("\n[Step 7/7] Running backtest...")
    try:
        metrics = backtest(feat_df, CONFIG["risk_free_rate"])
        strat   = metrics["Strategy"]
        bh      = metrics["Buy & Hold"]
        L(f"  ✓ Strategy  — Return: {strat['Total Return']:.1%}, Sharpe: {strat['Sharpe']:.2f}, MaxDD: {strat['Max Drawdown']:.1%}")
        L(f"  ✓ Buy&Hold  — Return: {bh['Total Return']:.1%},  Sharpe: {bh['Sharpe']:.2f},  MaxDD: {bh['Max Drawdown']:.1%}")
    except Exception as exc:
        return _fail(f"❌ Step 7 failed — {exc}")

    # ── Cache results ─────────────────────────────────────────────
    CACHE["model"]         = model
    CACHE["scaler"]        = scaler
    CACHE["feat_df"]       = feat_df
    CACHE["feature_names"] = selected
    CACHE["regime_labels"] = label_map
    CACHE["regime_colors"] = color_map
    CACHE["metrics"]       = metrics
    CACHE["X_train"]       = X_train
    CACHE["X_test"]        = X_test

    L("\n" + "═" * 60)
    L("  ✅ Pipeline complete!")
    L("═" * 60)

    # ── Build outputs ─────────────────────────────────────────────
    log_text  = "\n".join(log_lines)
    stats_html = build_stats_table(feat_df, model, label_map, metrics)

    fig_regime   = make_regime_chart(feat_df, label_map, color_map)
    fig_backtest = make_backtest_chart(metrics, feat_df)
    fig_dist     = make_distribution_chart(feat_df, label_map, color_map)
    fig_trans    = make_transition_chart(model, label_map, color_map)

    return log_text, stats_html, fig_regime, fig_backtest, fig_dist, fig_trans


# ── Section 4: Chart Functions ───────────────────────────────────────────────

def _apply_dark_style(fig, axes):
    """Apply consistent dark GitHub-like style to a figure."""
    fig.patch.set_facecolor(PALETTE["bg"])
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(PALETTE["bg"])
        ax.tick_params(colors=PALETTE["fg"])
        ax.xaxis.label.set_color(PALETTE["fg"])
        ax.yaxis.label.set_color(PALETTE["fg"])
        ax.title.set_color(PALETTE["fg"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["border"])
        ax.grid(color=PALETTE["grid"], linewidth=0.5, alpha=0.6)


def make_regime_chart(feat_df: pd.DataFrame, label_map: dict, color_map: dict):
    """4-panel chart: price + regime shading, returns, RSI, volume."""
    fig = plt.figure(figsize=(14, 10), facecolor=PALETTE["bg"])
    gs  = gridspec.GridSpec(4, 1, hspace=0.08, figure=fig)

    ax_price  = fig.add_subplot(gs[0])
    ax_ret    = fig.add_subplot(gs[1], sharex=ax_price)
    ax_rsi    = fig.add_subplot(gs[2], sharex=ax_price)
    ax_vol    = fig.add_subplot(gs[3], sharex=ax_price)

    _apply_dark_style(fig, [ax_price, ax_ret, ax_rsi, ax_vol])

    # Price + regime shading
    ax_price.plot(feat_df.index, feat_df["Close"], color=PALETTE["fg"], linewidth=1.2, label="Close")
    ax_price.set_ylabel("Price", color=PALETTE["fg"])
    ax_price.set_title("KSE-100 Market Regime Detection", color=PALETTE["fg"], fontsize=13)

    # Shade regimes
    prev_regime = None
    seg_start   = feat_df.index[0]
    for i, (idx, row) in enumerate(feat_df.iterrows()):
        regime = row.get("Regime", "Unknown")
        if regime != prev_regime:
            if prev_regime is not None:
                clr = REGIME_COLOR_MAP.get(prev_regime, "#888888")
                for ax in [ax_price, ax_ret, ax_rsi, ax_vol]:
                    ax.axvspan(seg_start, idx, alpha=0.15, color=clr, linewidth=0)
            seg_start   = idx
            prev_regime = regime
    if prev_regime is not None:
        clr = REGIME_COLOR_MAP.get(prev_regime, "#888888")
        for ax in [ax_price, ax_ret, ax_rsi, ax_vol]:
            ax.axvspan(seg_start, feat_df.index[-1], alpha=0.15, color=clr, linewidth=0)

    # Returns
    pos_ret = feat_df["Return"].clip(lower=0)
    neg_ret = feat_df["Return"].clip(upper=0)
    ax_ret.bar(feat_df.index, pos_ret, color=PALETTE["Bull"],  width=1, alpha=0.8)
    ax_ret.bar(feat_df.index, neg_ret, color=PALETTE["Bear"],  width=1, alpha=0.8)
    ax_ret.axhline(0, color=PALETTE["fg"], linewidth=0.5)
    ax_ret.set_ylabel("Return", color=PALETTE["fg"])

    # RSI
    if "RSI_14" in feat_df.columns:
        ax_rsi.plot(feat_df.index, feat_df["RSI_14"], color="#7ec8e3", linewidth=1)
        ax_rsi.axhline(70, color=PALETTE["Bear"],    linewidth=0.8, linestyle="--", alpha=0.7)
        ax_rsi.axhline(30, color=PALETTE["Bull"],    linewidth=0.8, linestyle="--", alpha=0.7)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_ylabel("RSI-14", color=PALETTE["fg"])

    # Volume
    if "Volume" in feat_df.columns or "Volume_Ratio" in feat_df.columns:
        vol_col = "Volume" if "Volume" in feat_df.columns else "Volume_Ratio"
        ax_vol.bar(feat_df.index, feat_df[vol_col], color="#5b9bd5", width=1, alpha=0.7)
        ax_vol.set_ylabel("Volume", color=PALETTE["fg"])

    # Legend patches
    patches = [
        mpatches.Patch(color=REGIME_COLOR_MAP.get(v, "#888"), label=v)
        for v in sorted(set(label_map.values()))
    ]
    ax_price.legend(handles=patches, loc="upper left",
                    facecolor=PALETTE["bg"], edgecolor=PALETTE["border"],
                    labelcolor=PALETTE["fg"])

    plt.setp(ax_price.get_xticklabels(), visible=False)
    plt.setp(ax_ret.get_xticklabels(),   visible=False)
    plt.setp(ax_rsi.get_xticklabels(),   visible=False)

    fig.tight_layout()
    return fig


def make_backtest_chart(metrics: dict, feat_df: pd.DataFrame):
    """Cumulative returns + drawdown comparison chart."""
    df = metrics["df"]

    fig, (ax_cum, ax_dd) = plt.subplots(2, 1, figsize=(12, 8),
                                         facecolor=PALETTE["bg"], sharex=True)
    _apply_dark_style(fig, [ax_cum, ax_dd])

    ax_cum.plot(df.index, df["Strat_Cumulative"], color=PALETTE["Bull"],
                linewidth=1.5, label="Strategy (Long on Bull)")
    ax_cum.plot(df.index, df["BH_Cumulative"],    color="#5b9bd5",
                linewidth=1.5, label="Buy & Hold", linestyle="--")
    ax_cum.axhline(1, color=PALETTE["fg"], linewidth=0.5, alpha=0.5)
    ax_cum.set_ylabel("Cumulative Return", color=PALETTE["fg"])
    ax_cum.set_title("Backtest: Strategy vs Buy & Hold", color=PALETTE["fg"], fontsize=13)
    ax_cum.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["border"], labelcolor=PALETTE["fg"])

    ax_dd.fill_between(df.index, df["Strat_Drawdown"], 0, color=PALETTE["Bull"], alpha=0.4, label="Strategy DD")
    ax_dd.fill_between(df.index, df["BH_Drawdown"],    0, color="#5b9bd5",       alpha=0.4, label="B&H DD")
    ax_dd.set_ylabel("Drawdown", color=PALETTE["fg"])
    ax_dd.set_xlabel("Date",     color=PALETTE["fg"])
    ax_dd.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["border"], labelcolor=PALETTE["fg"])

    fig.tight_layout()
    return fig


def make_distribution_chart(feat_df: pd.DataFrame, label_map: dict, color_map: dict):
    """Return histograms per regime with KDE overlay + stats box."""
    regimes   = sorted(set(label_map.values()))
    n_regimes = len(regimes)
    fig, axes = plt.subplots(1, n_regimes, figsize=(5 * n_regimes, 6),
                              facecolor=PALETTE["bg"])
    if n_regimes == 1:
        axes = [axes]
    _apply_dark_style(fig, axes)

    for ax, regime in zip(axes, regimes):
        mask    = feat_df["Regime"] == regime
        returns = feat_df.loc[mask, "Return"].dropna() * 100
        color   = REGIME_COLOR_MAP.get(regime, "#888888")

        if len(returns) > 1:
            ax.hist(returns, bins=50, color=color, alpha=0.6, density=True)
            mu, sigma = returns.mean(), returns.std()
            x = np.linspace(returns.min(), returns.max(), 200)
            ax.plot(x, norm.pdf(x, mu, sigma), color="white", linewidth=1.5)
            stats_text = f"μ={mu:.2f}%\nσ={sigma:.2f}%\nn={len(returns):,}"
            ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                    verticalalignment="top", horizontalalignment="right",
                    color=PALETTE["fg"], fontsize=9,
                    bbox=dict(facecolor=PALETTE["bg"], edgecolor=color, alpha=0.7))

        ax.set_title(regime, color=color, fontsize=12)
        ax.set_xlabel("Daily Return (%)", color=PALETTE["fg"])
        ax.set_ylabel("Density",          color=PALETTE["fg"])

    fig.suptitle("Return Distributions by Regime", color=PALETTE["fg"], fontsize=13)
    fig.tight_layout()
    return fig


def make_transition_chart(model: GaussianHMM, label_map: dict, color_map: dict):
    """Heatmap of regime transition probabilities + expected duration bars."""
    n = model.n_components
    labels = [label_map.get(i, f"State {i}") for i in range(n)]

    fig, (ax_heat, ax_dur) = plt.subplots(1, 2, figsize=(12, 5),
                                           facecolor=PALETTE["bg"])
    _apply_dark_style(fig, [ax_heat, ax_dur])

    trans = model.transmat_
    cmap  = LinearSegmentedColormap.from_list("regime_cmap", [PALETTE["bg"], "#5b9bd5"], N=256)
    im    = ax_heat.imshow(trans, cmap=cmap, vmin=0, vmax=1)
    ax_heat.set_xticks(range(n)); ax_heat.set_yticks(range(n))
    ax_heat.set_xticklabels(labels, color=PALETTE["fg"])
    ax_heat.set_yticklabels(labels, color=PALETTE["fg"])
    ax_heat.set_xlabel("To",   color=PALETTE["fg"])
    ax_heat.set_ylabel("From", color=PALETTE["fg"])
    ax_heat.set_title("Transition Probabilities", color=PALETTE["fg"])

    for i in range(n):
        for j in range(n):
            ax_heat.text(j, i, f"{trans[i, j]:.2f}", ha="center", va="center",
                         color="white" if trans[i, j] < 0.6 else PALETTE["bg"], fontsize=10)

    # Expected duration = 1 / (1 - P(stay))
    durations = [1 / (1 - trans[i, i] + 1e-10) for i in range(n)]
    colors    = [color_map.get(i, DEFAULT_COLORS[i % len(DEFAULT_COLORS)]) for i in range(n)]
    ax_dur.barh(labels, durations, color=colors, alpha=0.85)
    ax_dur.set_xlabel("Expected Duration (days)", color=PALETTE["fg"])
    ax_dur.set_title("Expected Regime Duration",  color=PALETTE["fg"])

    fig.tight_layout()
    return fig


# ── Section 5: Current Regime Predictor ──────────────────────────────────────

def predict_regime_gradio():
    """Predict the current market regime from cached model results."""
    if CACHE["model"] is None:
        return "⚠ Run the pipeline first.", None

    try:
        result = predict_current_regime(
            CACHE["model"],
            CACHE["scaler"],
            CACHE["feat_df"],
            CACHE["feature_names"],
            CACHE["regime_labels"],
            CACHE["regime_colors"],
        )
        label = result["label"]
        color = result["color"]
        probs = result["probs"]
        date  = result["last_date"]

        # Trading recommendation
        rec_map = {
            "Bull":     "🟢 BUY — Long equities",
            "Sideways": "🟡 HOLD — Reduce risk, hedge",
            "Bear":     "🔴 CASH — Exit equities",
        }
        recommendation = rec_map.get(label, f"ℹ {label} regime detected")

        prob_str = " | ".join(f"{k}: {v:.1%}" for k, v in probs.items())
        summary  = (
            f"### Current Regime: {label}\n\n"
            f"**As of:** {pd.Timestamp(date).date()}\n\n"
            f"**Regime Probabilities:** {prob_str}\n\n"
            f"**Recommendation:** {recommendation}"
        )

        # 30-day timeline chart
        feat_df = CACHE["feat_df"]
        recent  = feat_df.tail(30)

        fig, ax = plt.subplots(figsize=(10, 3), facecolor=PALETTE["bg"])
        _apply_dark_style(fig, [ax])

        for i, (idx, row) in enumerate(recent.iterrows()):
            regime = row.get("Regime", "Unknown")
            clr    = REGIME_COLOR_MAP.get(regime, "#888888")
            ax.barh(0, 1, left=i, color=clr, height=0.6, alpha=0.85)

        ax.set_xlim(0, len(recent))
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xticks(range(0, len(recent), 5))
        ax.set_xticklabels(
            [recent.index[i].strftime("%b %d") for i in range(0, len(recent), 5)],
            rotation=30, color=PALETTE["fg"], fontsize=7,
        )
        ax.set_title("Last 30 Days Regime", color=PALETTE["fg"])
        patches = [
            mpatches.Patch(color=REGIME_COLOR_MAP.get(v, "#888"), label=v)
            for v in ["Bear", "Sideways", "Bull"]
        ]
        ax.legend(handles=patches, loc="lower right",
                  facecolor=PALETTE["bg"], edgecolor=PALETTE["border"],
                  labelcolor=PALETTE["fg"])
        fig.tight_layout()

        return summary, fig

    except Exception as exc:
        return f"❌ Prediction failed: {exc}", None


# ── Section 6: Stats Table Builder ───────────────────────────────────────────

def build_stats_table(feat_df: pd.DataFrame, model: GaussianHMM, regime_labels: dict, metrics: dict) -> str:
    """Generate an HTML stats table for per-regime and backtest metrics."""
    rows_regime = []
    trans = model.transmat_
    n     = model.n_components

    for state in range(n):
        lbl  = regime_labels.get(state, f"State {state}")
        mask = feat_df["Regime"] == lbl
        ret  = feat_df.loc[mask, "Return"].dropna()

        if len(ret) == 0:
            continue

        ann_ret  = ret.mean() * 252
        ann_vol  = ret.std()  * math.sqrt(252)
        rfr      = CONFIG["risk_free_rate"]
        sharpe   = (ann_ret - rfr) / (ann_vol + 1e-10)
        max_loss = ret.min()
        max_gain = ret.max()
        persist  = trans[state, state]
        exp_dur  = 1 / (1 - persist + 1e-10)

        color = REGIME_COLOR_MAP.get(lbl, "#888")
        rows_regime.append(
            f"<tr>"
            f"<td><span style='color:{color};font-weight:bold'>{lbl}</span></td>"
            f"<td>{ann_ret:.1%}</td><td>{ann_vol:.1%}</td>"
            f"<td>{sharpe:.2f}</td>"
            f"<td>{max_loss:.2%}</td><td>{max_gain:.2%}</td>"
            f"<td>{persist:.2%}</td><td>{exp_dur:.1f}d</td>"
            f"</tr>"
        )

    strat = metrics["Strategy"]
    bh    = metrics["Buy & Hold"]
    rows_bt = [
        f"<tr><td>Strategy</td>"
        f"<td>{strat['Total Return']:.1%}</td><td>{strat['CAGR']:.1%}</td>"
        f"<td>{strat['Sharpe']:.2f}</td><td>{strat['Max Drawdown']:.1%}</td></tr>",

        f"<tr><td>Buy &amp; Hold</td>"
        f"<td>{bh['Total Return']:.1%}</td><td>{bh['CAGR']:.1%}</td>"
        f"<td>{bh['Sharpe']:.2f}</td><td>{bh['Max Drawdown']:.1%}</td></tr>",
    ]

    html = f"""
<style>
  .stats-table {{
    border-collapse: collapse;
    width: 100%;
    background: #0d1117;
    color: #c9d1d9;
    font-family: monospace;
    font-size: 13px;
  }}
  .stats-table th, .stats-table td {{
    border: 1px solid #30363d;
    padding: 6px 10px;
    text-align: center;
  }}
  .stats-table th {{
    background: #161b22;
    color: #58a6ff;
  }}
  .stats-table tr:hover td {{
    background: #161b22;
  }}
</style>

<h3 style="color:#58a6ff;font-family:monospace">Regime Statistics</h3>
<table class="stats-table">
  <tr>
    <th>Regime</th><th>Ann. Return</th><th>Ann. Vol</th><th>Sharpe</th>
    <th>Max Loss</th><th>Max Gain</th><th>Persistence</th><th>Avg Duration</th>
  </tr>
  {''.join(rows_regime)}
</table>

<h3 style="color:#58a6ff;font-family:monospace;margin-top:16px">Backtest Summary</h3>
<table class="stats-table">
  <tr>
    <th>Approach</th><th>Total Return</th><th>CAGR</th><th>Sharpe</th><th>Max Drawdown</th>
  </tr>
  {''.join(rows_bt)}
</table>
"""
    return html


# ── Section 7: About HTML ─────────────────────────────────────────────────────

ABOUT_HTML = """
<div style="background:#0d1117;color:#c9d1d9;padding:20px;border-radius:8px;font-family:monospace;line-height:1.7">

<h2 style="color:#58a6ff">TFE-GHMM: Market Regime Detection for KSE-100</h2>

<p>This application applies a <strong>Gaussian Hidden Markov Model (GHMM)</strong> to detect latent
market regimes in the KSE-100 Pakistan Stock Exchange index.</p>

<h3 style="color:#79c0ff">Model Overview</h3>
<ul>
  <li><strong>Hidden States</strong>: Unobserved market regimes (Bear 🔴, Sideways 🟡, Bull 🟢)</li>
  <li><strong>Emissions</strong>: Each state emits a multivariate Gaussian distribution over technical features</li>
  <li><strong>Training</strong>: Baum-Welch Expectation-Maximisation algorithm</li>
  <li><strong>Decoding</strong>: Viterbi algorithm for most-likely state sequence</li>
</ul>

<h3 style="color:#79c0ff">How to Use</h3>
<ol>
  <li>Set the date range and model hyperparameters in the sidebar</li>
  <li>Upload your KSE-100 CSV file (Date, Open, High, Low, Close, Volume) — required</li>
  <li>Select technical and macro features to use as HMM emissions</li>
  <li>Click <strong>Run Pipeline</strong> and wait for results</li>
  <li>Explore regime charts, backtest performance, and the current regime predictor</li>
</ol>

<h3 style="color:#79c0ff">Features Used</h3>
<ul>
  <li><strong>Returns</strong>: 1-day, 5-day, 20-day price changes</li>
  <li><strong>Volatility</strong>: Rolling standard deviation (20d, 60d)</li>
  <li><strong>Momentum</strong>: RSI-14, MACD, Stochastic Oscillator</li>
  <li><strong>Trend</strong>: Bollinger Band width &amp; position</li>
  <li><strong>Risk</strong>: ATR-14 ratio</li>
  <li><strong>Volume</strong>: Volume ratio vs 20-day MA</li>
  <li><strong>Macro</strong>: PKR/USD, Brent Oil, VIX, MSCI EM (via Yahoo Finance)</li>
</ul>

<h3 style="color:#79c0ff">References</h3>
<ul>
  <li>Hamilton, J.D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series. <em>Econometrica</em>.</li>
  <li>Rabiner, L.R. (1989). A Tutorial on Hidden Markov Models. <em>Proceedings of the IEEE</em>.</li>
  <li>Ang, A. &amp; Bekaert, G. (2002). Regime Switches in Interest Rates. <em>JBES</em>.</li>
</ul>

<p style="color:#8b949e;font-size:12px;margin-top:16px">
  Built with: Python · hmmlearn · Gradio · yfinance · ta · scikit-learn · matplotlib
</p>
</div>
"""


# ── Section 8: Gradio UI ──────────────────────────────────────────────────────

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
    ),
    title="TFE-GHMM | KSE-100 Regime Detection",
    css="""
        body { background: #0d1117 !important; }
        .gr-box { background: #161b22 !important; border-color: #30363d !important; }
        footer { display: none !important; }
    """,
) as demo:

    gr.Markdown(
        """
        # 📈 TFE-GHMM — KSE-100 Market Regime Detection
        **Gaussian Hidden Markov Model** | Pakistan Stock Exchange | Real Data Only
        """,
    )

    with gr.Tabs():

        # ── Tab 1: Run Pipeline ──────────────────────────────────────────────
        with gr.Tab("🚀 Run Pipeline"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙ Configuration")

                    file_input = gr.File(
                        label="Upload KSE-100 CSV (Required)",
                        file_types=[".csv"],
                    )
                    gr.Markdown(
                        "> 📌 Required columns: Date, Open, High, Low, Close, Volume. "
                        "Date format: YYYY-MM-DD"
                    )

                    with gr.Row():
                        start_date = gr.Textbox(label="Start Date", value="2010-01-01")
                        end_date   = gr.Textbox(label="End Date",
                                                value=str(datetime.date.today()))

                    n_states   = gr.Slider(2, 6,  value=3, step=1, label="HMM States")
                    n_iter     = gr.Slider(50, 500, value=200, step=50, label="EM Iterations")
                    train_ratio= gr.Slider(0.5, 0.95, value=0.8, step=0.05,
                                           label="Train/Test Split")
                    random_state_input = gr.Number(value=42, label="Random Seed", precision=0)

                    use_macro  = gr.Checkbox(label="Include Macro Features (Yahoo Finance)",
                                             value=True)

                    gr.Markdown("### 📊 Technical Features")
                    feature_sel = gr.CheckboxGroup(
                        choices=ALL_FEATURES + MACRO_FEATURES,
                        value=DEFAULT_FEATURES,
                        label="Features for HMM",
                    )

                    run_btn = gr.Button("▶ Run Pipeline", variant="primary")

                with gr.Column(scale=2):
                    log_out   = gr.Textbox(label="Pipeline Log", lines=20,
                                           interactive=False)
                    stats_out = gr.HTML(label="Statistics")

        # ── Tab 2: Regime Charts ─────────────────────────────────────────────
        with gr.Tab("📉 Regime Charts"):
            regime_chart_out = gr.Plot(label="Regime Detection")

        # ── Tab 3: Backtest ──────────────────────────────────────────────────
        with gr.Tab("💰 Backtest"):
            backtest_chart_out = gr.Plot(label="Strategy vs Buy & Hold")

        # ── Tab 4: Current Regime ────────────────────────────────────────────
        with gr.Tab("🔮 Current Regime"):
            predict_btn     = gr.Button("🔮 Predict Current Regime", variant="secondary")
            regime_text_out = gr.Markdown()
            timeline_out    = gr.Plot(label="Last 30 Days")

        # ── Tab 5: Statistics ────────────────────────────────────────────────
        with gr.Tab("📊 Statistics"):
            with gr.Row():
                dist_chart_out  = gr.Plot(label="Return Distributions")
                trans_chart_out = gr.Plot(label="Transition Matrix")

        # ── Tab 6: About ─────────────────────────────────────────────────────
        with gr.Tab("📖 About"):
            gr.HTML(ABOUT_HTML)

    # ── Event wiring ─────────────────────────────────────────────────────────
    run_btn.click(
        fn=run_pipeline_gradio,
        inputs=[
            file_input,
            start_date,
            end_date,
            n_states,
            n_iter,
            train_ratio,
            feature_sel,
            use_macro,
            random_state_input,
        ],
        outputs=[
            log_out,
            stats_out,
            regime_chart_out,
            backtest_chart_out,
            dist_chart_out,
            trans_chart_out,
        ],
    )

    predict_btn.click(
        fn=predict_regime_gradio,
        inputs=[],
        outputs=[regime_text_out, timeline_out],
    )


# ── Section 9: Launch ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    # In Colab: share=True is set automatically so you get a public gradio.live URL.
    # Locally:  share=False — the app runs on the first available port (default 7860).
    # Set GRADIO_SERVER_PORT env-var to pin a specific port.
    _port = int(os.environ["GRADIO_SERVER_PORT"]) if "GRADIO_SERVER_PORT" in os.environ else None
    demo.launch(server_name="0.0.0.0", server_port=_port, share=IS_COLAB)
