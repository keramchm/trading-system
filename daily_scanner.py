import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. TELEGRAM ALERT FUNCTION
# ==========================================
def send_telegram_alert(bot_token, chat_id, ticker, prob, entry, sl, tp, vol_ratio):
    """Sends a formatted HTML message to Telegram."""
    vcp_status = "✅ Coiling (Good)" if vol_ratio < 1.0 else "⚠️ Expanding (Risky)"
    
    message = (
        f"🚨 <b>NEW SWING SETUP DETECTED</b> 🚨\n\n"
        f"📊 <b>Ticker:</b> <code>${ticker}</code>\n"
        f"🎯 <b>Model Confidence:</b> {prob:.1%}\n"
        f"📉 <b>Volatility:</b> {vol_ratio:.2f} ({vcp_status})\n\n"
        f"💰 <b>Action Plan for Tomorrow:</b>\n"
        f"🟢 <b>BUY AT OPEN:</b> <code>${entry:.2f}</code>\n"
        f"🛑 <b>STOP LOSS:</b> <code>${sl:.2f}</code> (1.5x ATR)\n"
        f"✅ <b>TAKE PROFIT:</b> <code>${tp:.2f}</code> (2.5x ATR)\n\n"
        f"⚠️ <i>Rule: Risk exactly 1% of total account capital on this trade.</i>"
    )
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    
    try:
        response = requests.post(url, data=payload)
        if response.json().get("ok"):
            print(f"✅ Alert sent successfully for {ticker}!")
        else:
            print(f"❌ Telegram Error: {response.text}")
    except Exception as e:
        print(f"❌ Network error: {e}")

# ==========================================
# 2. FEATURE ENGINEERING & PROXY MODEL
# ==========================================
def prepare_features(df, spy_df):
    df = df.copy()
    df = df.join(spy_df['Close'], rsuffix='_SPY').dropna()
    
    df['High_Low'] = df['High'] - df['Low']
    df['High_Close'] = np.abs(df['High'] - df['Close'].shift())
    df['Low_Close'] = np.abs(df['Low'] - df['Close'].shift())
    df['True_Range'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
    df['ATR_10'] = df['True_Range'].rolling(10).mean()
    df['ATR_50'] = df['True_Range'].rolling(50).mean()
    
    df['Volatility_Ratio'] = df['ATR_10'] / (df['ATR_50'] + 1e-8)
    df['Volatility_Expanding'] = (df['Volatility_Ratio'] > 1.1).astype(int)
    
    df['Stock_Return'] = df['Close'].pct_change()
    df['SPY_Return'] = df['Close_SPY'].pct_change()
    df['Relative_Strength'] = df['Stock_Return'] - df['SPY_Return']
    df['RS_Rolling_10'] = df['Relative_Strength'].rolling(10).sum()
    
    df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
    df['Volume_Spike'] = df['Volume'] / (df['Volume_MA_20'] + 1e-8)
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    df['Dist_to_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
    df['Dist_to_SMA200'] = (df['Close'] - df['SMA_200']) / df['SMA_200']
    
    spy_df['SPY_SMA_200'] = spy_df['Close'].rolling(200).mean()
    df['SPY_Above_SMA200'] = (spy_df['Close'] > spy_df['SPY_SMA_200']).astype(int)
    
    return df.dropna()

def train_proxy_model():
    print("🧠 Training Proxy Model on MSFT (2016-2021)...")
    msft = yf.download('MSFT', start='2016-01-01', end='2021-01-01', progress=False)
    spy = yf.download('SPY', start='2016-01-01', end='2021-01-01', progress=False)
    
    if isinstance(msft.columns, pd.MultiIndex): msft.columns = msft.columns.get_level_values(0)
    if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
    
    df = prepare_features(msft, spy)
    
    future_price = df['Close'].shift(-10)
    future_return = (future_price - df['Close']) / df['Close']
    target_move = 1.5 * (df['ATR_10'] / df['Close'])
    df['Target'] = (future_return >= target_move).astype(int)
    df = df.dropna()
    
    feature_cols = ['RS_Rolling_10', 'Volatility_Ratio', 'Volume_Spike', 
                    'Dist_to_SMA20', 'Dist_to_SMA200', 'Volatility_Expanding']
    
    X, y = df[feature_cols], df['Target']
    weight = len(y[y == 0]) / len(y[y == 1])
    
    model = xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05,
                              scale_pos_weight=weight, subsample=0.8, colsample_bytree=0.8,
                              random_state=42, eval_metric='logloss', early_stopping_rounds=30)
    model.fit(X, y, eval_set=[(X, y)], verbose=False)
    print("✅ Proxy Model Trained.\n")
    return model, feature_cols

# ==========================================
# 3. THE DAILY EXECUTION ENGINE
# ==========================================
def run_daily_execution(model, feature_cols, universe, bot_token, chat_id):
    print(f"🔍 Scanning {len(universe)} stocks for today's setups...")
    
    spy = yf.download('SPY', period='2y', progress=False)
    if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
    
    # Check Market Regime
    spy['SPY_SMA_200'] = spy['Close'].rolling(200).mean()
    spy_is_uptrend = spy['Close'].iloc[-1] > spy['SPY_SMA_200'].iloc[-1]
    
    if not spy_is_uptrend:
        print("⚠️ MARKET REGIME WARNING: SPY is below its 200 SMA. System is in defensive mode.")
        send_telegram_alert(bot_token, chat_id, "MARKET", 0.0, 0.0, 0.0, 0.0, 0.0) # Sends a warning
        return

    alerts_sent = 0
    
    for ticker in universe:
        try:
            df = yf.download(ticker, period='2y', progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            df = prepare_features(df, spy)
            if len(df) < 5: continue
                
            latest_row = df.iloc[-1]
            features = latest_row[feature_cols].values.reshape(1, -1)
            prob = model.predict_proba(features)[0][1]
            
            # Validated threshold: 60% confidence
            if prob >= 0.60:
                entry_price = latest_row['Close'] 
                atr = latest_row['ATR_10']
                
                stop_loss = entry_price - (1.5 * atr)
                take_profit = entry_price + (2.5 * atr)
                vol_ratio = latest_row['Volatility_Ratio']
                
                print(f"🚨 HIGH PROBABILITY SETUP FOUND: {ticker} ({prob:.1%})")
                
                send_telegram_alert(bot_token, chat_id, ticker, prob, entry_price, stop_loss, take_profit, vol_ratio)
                alerts_sent += 1
                
        except Exception as e:
            pass # Skip tickers with data errors silently
            
    if alerts_sent == 0:
        print("✅ Scan complete. No high-probability setups found today. Cash is a position.")
    else:
        print(f"\n🎉 Execution complete. Sent {alerts_sent} alerts to Telegram.")

# ==========================================
# RUN THE ENGINE
# ==========================================
if __name__ == "__main__":
    # 1. PASTE YOUR VERIFIED CREDENTIALS HERE
    TELEGRAM_BOT_TOKEN = "8985753082:AAFL10n--La3nTMYnyHSs8fY4Af9Y0y8ByE"
    TELEGRAM_CHAT_ID = "2127782209"
    
    # 2. Define the Universe
    UNIVERSE = [
        'AAPL', 'NVDA', 'META', 'AMZN', 'GOOGL', 
        'AVGO', 'TSLA', 'AMD', 'CRM', 'NFLX',
        'MSFT', 'ORCL', 'ADBE', 'NOW', 'PANW'
    ]
    
    # 3. Train and Scan
    model, feature_cols = train_proxy_model()
    run_daily_execution(model, feature_cols, UNIVERSE, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)