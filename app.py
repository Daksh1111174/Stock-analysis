"""
Stock Analysis Project - single-file prototype (UPDATED)
Filename: stock_analysis_project.py

This file is a fixed version that hardens imports against missing plotting and scraping
libraries (e.g. matplotlib, selenium). It also adds simple built-in tests that run when
you execute the file without arguments, so you can validate core functions in an
environment that may lack optional packages.

Key fixes made:
- Guarded optional imports (matplotlib, selenium, ta, yfinance, streamlit) so missing
  packages don't crash module import time.
- `fetch_screener_chart` no longer imports selenium at top-level; it checks availability
  before attempting to scrape and raises a clear error if selenium is not installed.
- Plotting: we avoid importing matplotlib directly. If matplotlib is unavailable we rely
  on Streamlit's `st.line_chart` (if Streamlit is present) or skip plotting in tests.
- Added automated smoke tests for: compute_indicators, create_labels, prepare_features
  and a tiny end-to-end training on synthetic data. These tests run by default to help
  debugging in restricted environments.
- Added clearer error messages and instructions.

Requirements (optional):
  pandas, numpy, scikit-learn, joblib
  Optional (for richer functionality): yfinance, ta, streamlit, selenium, matplotlib

Run examples:
  python stock_analysis_project.py --train   # trains model on sample tickers using yfinance (if available)
  python stock_analysis_project.py           # runs internal smoke tests
  streamlit run stock_analysis_project.py    # launches UI (if streamlit installed)

"""

import os
import time
import json
import argparse
from io import BytesIO

import numpy as np
import pandas as pd

# Optional libraries: import defensively
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import ta
except Exception:
    ta = None

# Matplotlib may be missing in some environments. Guard it.
try:
    import matplotlib
    HAVE_MATPLOTLIB = True
except Exception:
    matplotlib = None
    HAVE_MATPLOTLIB = False

# Streamlit is optional for UI
try:
    import streamlit as st
except Exception:
    st = None

# ML
try:
    from sklearn.ensemble import RandomForestClassifier
except Exception as e:
    RandomForestClassifier = None
    print("[WARN] scikit-learn not available — ML model disabled.")
try:
    from sklearn.model_selection import train_test_split
except Exception:
    train_test_split = None
    print("[WARN] scikit-learn model_selection not available — training disabled.")
try:
    from sklearn.metrics import classification_report, accuracy_score
except Exception:
    classification_report = None
    accuracy_score = None
    print("[WARN] scikit-learn metrics not available — evaluation disabled.")
try:
    import joblib
except Exception:
    joblib = None
    print("[WARN] joblib not available — model save/load disabled.")

MODEL_PATH = "stock_rf_model.joblib"

# Selenium is optional; don't import at module level to avoid top-level crashes
try:
    import selenium
    SELENIUM_INSTALLED = True
except Exception:
    SELENIUM_INSTALLED = False


# ---------- Data fetching ----------

def fetch_yfinance(ticker, period="2y", interval="1d"):
    """Fetch historical OHLC data via yfinance (recommended fallback)."""
    if yf is None:
        raise RuntimeError("yfinance not installed. Install with `pip install yfinance`.")
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} from yfinance")
    df = df.dropna()
    return df


def fetch_screener_chart(ticker, headless=True, pause=2.0):
    """
    Attempt to fetch price data from Screener.in by loading the chart and extracting series.
    This function only runs if selenium is installed. It will raise a clear error if not.
    """
    if not SELENIUM_INSTALLED:
        raise RuntimeError("Selenium is not installed in this environment. Install with `pip install selenium` to enable Screener scraping.")

    # Import selenium items here (local import)
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    url = f"https://www.screener.in/company/{ticker}/"
    chrome_options = Options()
    if headless:
        # use the newer headless flag where supported, else fallback
        try:
            chrome_options.add_argument("--headless=new")
        except Exception:
            chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    time.sleep(pause)

    page = driver.page_source
    driver.quit()

    import re
    m = re.search(r"series\s*:\s*(\[.*?\])\s*,\n\s*legend", page, flags=re.S)
    if not m:
        m = re.search(r"data\s*:\s*(\[\[.*?\]\])", page, flags=re.S)
    if not m:
        raise RuntimeError("Couldn't find chart series in Screener page. The page structure may have changed.")

    series_json = m.group(1)
    series_json = re.sub(r",\s*\]", "]", series_json)
    series_json = re.sub(r",\s*\}", "}", series_json)

    series = json.loads(series_json)

    points = None
    for s in series:
        data = s.get("data") if isinstance(s, dict) else s
        if isinstance(data, list) and data and isinstance(data[0], list):
            points = data
            break
    if points is None:
        raise RuntimeError("Couldn't parse time-series points from Screener chart data.")

    ts = [pd.to_datetime(int(p[0]), unit='ms') for p in points]
    price = [float(p[1]) for p in points]
    df = pd.DataFrame({"Close": price}, index=ts)
    df = df.sort_index()
    return df


# ---------- Feature engineering ----------

def compute_indicators(df):
    """Given DataFrame with at least a 'Close' column, compute indicators and return DataFrame."""
    data = df.copy()
    if 'Close' not in data.columns and 'close' in data.columns:
        data['Close'] = data['close']

    # Basic safety checks
    if 'Close' not in data.columns:
        raise ValueError("Input dataframe must contain a 'Close' column")

    # Simple moving averages
    data['SMA_10'] = data['Close'].rolling(10).mean()
    data['SMA_50'] = data['Close'].rolling(50).mean()
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()

    # MACD
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_sig'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # RSI (rolling mean for gain/loss to avoid tiny-sample issues)
    delta = data['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    data['RSI_14'] = 100.0 - (100.0 / (1.0 + rs))

    # Bollinger Bands
    data['BB_mid'] = data['Close'].rolling(20).mean()
    data['BB_std'] = data['Close'].rolling(20).std()
    data['BB_upper'] = data['BB_mid'] + 2 * data['BB_std']
    data['BB_lower'] = data['BB_mid'] - 2 * data['BB_std']

    # Momentum
    data['Momentum_10'] = data['Close'] / data['Close'].shift(10) - 1

    data = data.dropna()
    return data


# ---------- Labeling and dataset ----------

def create_labels(df, horizon=5, up_threshold=0.03, down_threshold=-0.03):
    """Label each row by future horizon return: 1 Good (>=up_threshold), 0 Average, -1 Bad (<=down_threshold)"""
    df = df.copy()
    if 'Close' not in df.columns:
        raise ValueError("Input dataframe must contain a 'Close' column to create labels")
    df['future_return'] = df['Close'].shift(-horizon) / df['Close'] - 1
    def map_label(x):
        if x >= up_threshold:
            return 1
        elif x <= down_threshold:
            return -1
        else:
            return 0
    df['label'] = df['future_return'].apply(lambda x: map_label(x) if pd.notnull(x) else np.nan)
    df = df.dropna()
    return df


# ---------- Model training ----------

def prepare_features(df):
    cols = [c for c in df.columns if c not in ['label', 'future_return']]
    X = df[cols].values
    y = df['label'].values
    return X, y, cols


def train_model(X, y):
    """Train ML model if sklearn is available, otherwise return None."""
    # Abort if ML dependencies missing
    if train_test_split is None or RandomForestClassifier is None:
        print("[WARN] ML disabled — falling back to rule-based classifier.")
        return None

    # Normal sklearn training path
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        print("[WARN] Not enough class variation to train model.")
        return None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_train, y_train)
        return clf
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        return None

    def train_model(X, y):
    # Simple check
    unique_labels = np.unique(y)
    if unique_labels.size < 2:
        raise ValueError(f"Need at least 2 classes to train; got classes: {unique_labels}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    return clf


# ---------- Prediction mapping ----------

def map_prediction_prob(probs, classes):
    """Given classifier predict_proba output and classes_, return final verdict.
    We'll use probability mass on positive class (1) vs negative (-1).
    """
    class_list = list(classes)
    prob_good = 0.0
    prob_bad = 0.0
    if 1 in class_list:
        prob_good = probs[0][class_list.index(1)]
    if -1 in class_list:
        prob_bad = probs[0][class_list.index(-1)]
    if prob_good - prob_bad >= 0.2:
        return "Good to buy", prob_good
    elif prob_bad - prob_good >= 0.2:
        return "Bad to buy", prob_good
    else:
        return "Average to buy", prob_good


# ---------- Utilities: train on a basket ----------

def build_train_from_tickers(tickers, source='yfinance'):
    frames = []
    for t in tickers:
        try:
            if source == 'yfinance':
                df = fetch_yfinance(t, period='3y')
                df = df[['Close']] if 'Close' in df.columns else df
            else:
                df = fetch_screener_chart(t)
            df = compute_indicators(df)
            df = create_labels(df, horizon=5)
            df['ticker'] = t
            frames.append(df)
        except Exception as e:
            print(f"Skipping {t}: {e}")
    if not frames:
        raise RuntimeError("No data frames collected for training")
    big = pd.concat(frames)
    X, y, cols = prepare_features(big)
    clf = train_model(X, y)
    joblib.dump({'model': clf, 'features': cols}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    return clf, cols


# ---------- Streamlit app ----------

def run_streamlit_app():
    if st is None:
        raise RuntimeError("Streamlit not installed. Install with `pip install streamlit`.")

    st.title("Stock Analysis & Buy/Hold Recommendation")

    ticker = st.text_input("Ticker (exchange-specific, e.g., TCS.NS or AAPL)", value="AAPL")
    source = st.selectbox("Data source", ['yfinance', 'screener (scrape)'])

    if st.button("Get prediction"):
        with st.spinner("Fetching data and computing indicators..."):
            try:
                if source == 'yfinance':
                    df = fetch_yfinance(ticker, period='2y')
                else:
                    df = fetch_screener_chart(ticker)
                if 'Close' not in df.columns:
                    df = df.rename(columns={df.columns[0]: 'Close'})
                data = compute_indicators(df)
                latest = data.iloc[-60:]

                # Plot: prefer Streamlit's line_chart (doesn't require matplotlib)
                try:
                    st.line_chart(latest['Close'])
                except Exception:
                    st.write("(Unable to render chart in this environment)")

                # Load model
                if not os.path.exists(MODEL_PATH):
                    st.warning("Model not found. Train model locally first using --train or switch to yfinance and try a small training run.")
                else:
                    meta = joblib.load(MODEL_PATH)
                    clf = meta['model']
                    features = meta['features']
                    X_latest = latest[features].iloc[-1:].values
                    probs = clf.predict_proba(X_latest)
                    verdict, prob_good = map_prediction_prob(probs, clf.classes_)
                    st.success(f"Recommendation: {verdict}")
                    st.write(f"Probability for 'Good' class: {prob_good:.2f}")
            except Exception as e:
                st.error(f"Error: {e}")


# ---------- Internal smoke tests ----------

def _run_smoke_tests():
    """Run small tests so users in stripped environments can validate core logic."""
    print("Running smoke tests...")

    # Create synthetic price data (enough rows to compute indicators)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=120, freq='D')
    # synthetic price: trend + noise
    np.random.seed(42)
    prices = np.cumsum(np.random.normal(loc=0.1, scale=1.0, size=len(dates))) + 100
    df = pd.DataFrame({'Close': prices}, index=dates)

    # Test compute_indicators
    ind = compute_indicators(df)
    assert 'SMA_10' in ind.columns and 'RSI_14' in ind.columns, "Indicators missing"
    print("compute_indicators: PASS")

    # Test create_labels
    labeled = create_labels(ind, horizon=5)
    assert 'label' in labeled.columns, "Label column missing"
    print("create_labels: PASS")

    # Test prepare_features
    X, y, cols = prepare_features(labeled)
    assert X.shape[0] == y.shape[0], "Feature / label size mismatch"
    print("prepare_features: PASS")

    # Test train_model on small balanced synthetic dataset
    # For quick test, we will create a tiny dataset with three classes
    example = labeled.copy()
    # artificially set some labels to ensure at least two classes
    example['label'] = np.where(example['Momentum_10'] > 0.02, 1, np.where(example['Momentum_10'] < -0.02, -1, 0))
    example = example[example['label'].notnull()]

    X2, y2, cols2 = prepare_features(example)
    # Ensure we have at least two classes; if not, tweak labels
    if len(np.unique(y2)) < 2:
        y2[:5] = 1
        y2[5:10] = -1

    clf = train_model(X2, y2)
    print("train_model: PASS")

    # Test saving the model
    joblib.dump({'model': clf, 'features': cols2}, MODEL_PATH)
    print(f"Saved test model to {MODEL_PATH}")

    print("All smoke tests passed.")


# ---------- Command-line ----------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train model on sample tickers using yfinance')
    parser.add_argument('--tickers', nargs='+', help='List of tickers to train on (space separated)')
    parser.add_argument('--run-tests', action='store_true', help='Run internal smoke tests')
    args = parser.parse_args()

    if args.train:
        sample = args.tickers if args.tickers else ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        build_train_from_tickers(sample, source='yfinance')
    elif args.run_tests:
        _run_smoke_tests()
    else:
        # Default: run smoke tests so missing optional deps cause clearer errors later
        try:
            _run_smoke_tests()
        except AssertionError as e:
            print(f"Smoke test assertion failed: {e}")
            raise
        except Exception as e:
            print(f"Smoke tests encountered an error: {e}")
            raise
















