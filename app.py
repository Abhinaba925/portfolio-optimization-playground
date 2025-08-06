import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.cluster.hierarchy as sch

# --- Portfolio Optimization Functions --- #

def mean_variance_optimization(mu, S):
    """Calculates the optimal portfolio weights for maximizing the Sharpe ratio."""
    def objective(weights):
        portfolio_return = np.sum(mu * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
        if portfolio_volatility == 0:
            return 0
        return -portfolio_return / portfolio_volatility

    num_assets = len(mu)
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.array([1./num_assets] * num_assets)
    
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def risk_parity_optimization(S):
    """Calculates the optimal portfolio weights for risk parity."""
    def objective(weights):
        portfolio_variance = np.dot(weights.T, np.dot(S, weights))
        # Handle cases with zero variance
        if portfolio_variance == 0:
            return 0
        marginal_contribution = np.dot(S, weights)
        risk_contribution = weights * marginal_contribution / np.sqrt(portfolio_variance)
        return np.sum((risk_contribution - risk_contribution.mean())**2)

    num_assets = S.shape[0]
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.array([1./num_assets] * num_assets)
    
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def hierarchical_risk_parity(returns):
    """Calculates the optimal portfolio weights using Hierarchical Risk Parity."""
    corr = returns.corr()
    cov = returns.cov()
    
    dist = np.sqrt((1 - corr) / 2)
    link = sch.linkage(dist, 'single')
    sort_ix = sch.leaves_list(link)
    
    sorted_tickers = corr.index[sort_ix]

    def get_rec_bisection(cov, sort_ix_local):
        w = pd.Series(1, index=sort_ix_local)
        c_items = [sort_ix_local]
        while len(c_items) > 0:
            c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            for i in range(0, len(c_items), 2):
                c1_tickers = c_items[i]
                c2_tickers = c_items[i+1]
                
                # Ensure tickers are in the covariance matrix columns
                c1_tickers_valid = [t for t in c1_tickers if t in cov.index]
                c2_tickers_valid = [t for t in c2_tickers if t in cov.index]

                if not c1_tickers_valid or not c2_tickers_valid:
                    continue

                v1 = np.diag(cov.loc[c1_tickers_valid, c1_tickers_valid]).sum()
                v2 = np.diag(cov.loc[c2_tickers_valid, c2_tickers_valid]).sum()
                
                alpha = 1 - v1 / (v1 + v2) if (v1 + v2) != 0 else 0.5
                
                w[c1_tickers] *= alpha
                w[c2_tickers] *= (1 - alpha)
        return w
        
    weights = get_rec_bisection(cov, sorted_tickers)
    return weights.sort_index()


# --- Streamlit App UI ---

st.set_page_config(layout="wide")
st.title("📈 Portfolio Optimization Playground")
st.markdown("""
This application performs portfolio optimization based on historical stock data from the **National Stock Exchange (NSE)**. 
You can select an optimization strategy, and the app will calculate the ideal asset allocation and show its performance against the Nifty 50 benchmark.
""")

# --- Stock & Benchmark Definition ---
# A predefined list of 50 major NSE stocks from various sectors
TICKERS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 
    'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'LT.NS', 'BAJFINANCE.NS', 'KOTAKBANK.NS', 
    'HCLTECH.NS', 'MARUTI.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'SUNPHARMA.NS', 
    'WIPRO.NS', 'NESTLEIND.NS', 'ULTRACEMCO.NS', 'TITAN.NS', 'ONGC.NS', 'NTPC.NS',
    'TATAMOTORS.NS', 'POWERGRID.NS', 'BAJAJFINSV.NS', 'INDUSINDBK.NS', 'TECHM.NS',
    'HDFCLIFE.NS', 'ADANIPORTS.NS', 'JSWSTEEL.NS', 'TATASTEEL.NS', 'COALINDIA.NS',
    'SBILIFE.NS', 'BRITANNIA.NS', 'GRASIM.NS', 'HINDALCO.NS', 'EICHERMOT.NS',
    'CIPLA.NS', 'DRREDDY.NS', 'BPCL.NS', 'SHREECEM.NS', 'HEROMOTOCO.NS',
    'BAJAJ-AUTO.NS', 'IOC.NS', 'UPL.NS', 'DIVISLAB.NS', 'M&M.NS', 'ADANIENT.NS', 'DABUR.NS'
]
BENCHMARK_TICKER = '^NSEI' # Nifty 50

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("⚙️ Configuration")
    capital = st.number_input("Enter Your Capital (INR)", min_value=1000, value=100000, step=1000)
    start_date = st.date_input("Select Start Date", value=pd.to_datetime("2021-01-01"), 
                               min_value=pd.to_datetime("2015-01-01"), 
                               max_value=pd.to_datetime('today') - pd.Timedelta(days=1))
    
    optimization_method = st.selectbox("Choose Optimization Method", 
                                       ["Mean-Variance", "Risk Parity", "Hierarchical Risk Parity", "Equally Weighted"])
    
    st.header("📉 Cost Simulation")
    transaction_cost_pct = st.number_input("Transaction Cost (%)", min_value=0.0, value=0.1, step=0.01, help="Cost for taxes like STT.")
    slippage_pct = st.number_input("Slippage (%)", min_value=0.0, value=0.05, step=0.01, help="Simulated price deviation on execution.")
    brokerage_per_trade = st.number_input("Brokerage (INR per trade)", min_value=0.0, value=20.0, step=1.0, help="Flat fee per stock purchase.")

    calculate_button = st.button("🚀 Calculate & Backtest", use_container_width=True)


# --- Main Application Logic ---

# Caching data download to speed up the app
@st.cache_data
def download_data(tickers, start, end):
    """Downloads closing prices for a list of tickers."""
    try:
        prices = yf.download(tickers, start=start, end=end)['Close']
        return prices
    except Exception as e:
        st.error(f"An error occurred during data download: {e}")
        return pd.DataFrame()

if calculate_button:
    end_date = pd.to_datetime('today').strftime('%Y-%m-%d')
    
    with st.spinner("Downloading historical data... This may take a moment."):
        all_tickers = TICKERS + [BENCHMARK_TICKER]
        prices_df = download_data(all_tickers, start_date, end_date)

    # --- Data Validation ---
    if prices_df.empty:
        st.warning("Could not retrieve any data. Please check your connection or try a different date range.")
    
    elif BENCHMARK_TICKER not in prices_df.columns or prices_df[BENCHMARK_TICKER].isnull().all():
        st.error(f"Critical Error: Could not retrieve benchmark data for '{BENCHMARK_TICKER}'. The app cannot proceed.")
    
    else:
        benchmark_prices = prices_df[[BENCHMARK_TICKER]].dropna()
        available_tickers = [ticker for ticker in TICKERS if ticker in prices_df.columns and not prices_df[ticker].isnull().all()]
        stock_prices = prices_df[available_tickers].dropna()

        if stock_prices.empty or len(stock_prices.columns) < 2:
            st.error("Not enough valid historical stock data is available for the selected period to perform optimization.")
        else:
            daily_returns = stock_prices.pct_change().dropna()
            benchmark_returns = benchmark_prices.pct_change().dropna()

            # --- Perform Optimization ---
            st.header("📊 Portfolio Allocation")
            weights = {}
            
            with st.spinner(f"Calculating optimal weights for {len(daily_returns.columns)} available stocks..."):
                if optimization_method == "Mean-Variance":
                    mu = daily_returns.mean() * 252
                    S = daily_returns.cov() * 252
                    opt_weights = mean_variance_optimization(mu, S)
                    weights = dict(zip(daily_returns.columns, opt_weights))

                elif optimization_method == "Risk Parity":
                    S = daily_returns.cov() * 252
                    opt_weights = risk_parity_optimization(S)
                    weights = dict(zip(daily_returns.columns, opt_weights))

                elif optimization_method == "Hierarchical Risk Parity":
                    opt_weights = hierarchical_risk_parity(daily_returns)
                    weights = dict(opt_weights)

                elif optimization_method == "Equally Weighted":
                    num_assets = len(daily_returns.columns)
                    weights = {ticker: 1/num_assets for ticker in daily_returns.columns}

            weights_df = pd.DataFrame(list(weights.items()), columns=['Stock', 'Weight'])
            weights_df = weights_df[weights_df['Weight'] > 0.0001].sort_values('Weight', ascending=False).reset_index(drop=True)
            
            latest_prices = stock_prices.iloc[-1]
            weights_df['Allocation (INR)'] = weights_df['Weight'] * capital
            weights_df['Latest Price (INR)'] = weights_df['Stock'].map(latest_prices)
            weights_df['Shares to Buy'] = (weights_df['Allocation (INR)'] / weights_df['Latest Price (INR)']).astype(int)
            
            # --- Cost Calculation ---
            # FIX: Calculate brokerage based only on trades that are actually executed (Shares to Buy > 0)
            num_trades = len(weights_df[weights_df['Shares to Buy'] > 0])
            total_brokerage = num_trades * brokerage_per_trade
            total_txn_cost = capital * (transaction_cost_pct / 100)
            total_slippage_cost = capital * (slippage_pct / 100)
            total_initial_cost = total_brokerage + total_txn_cost + total_slippage_cost
            effective_capital = capital - total_initial_cost

            # Format weight column after all calculations are done
            weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x:.2%}")

            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader(f"Weights for {optimization_method}")
                st.dataframe(weights_df[['Stock', 'Weight', 'Shares to Buy']], height=400, use_container_width=True)
                
                with st.expander("View Cost Breakdown"):
                    st.metric("Total Initial Cost (INR)", f"{total_initial_cost:,.2f}")
                    st.markdown(f"- **Brokerage:** ₹{total_brokerage:,.2f} ({num_trades} actual trades)")
                    st.markdown(f"- **Transaction Tax:** ₹{total_txn_cost:,.2f}")
                    st.markdown(f"- **Slippage:** ₹{total_slippage_cost:,.2f}")
                    st.metric("Effective Starting Capital (INR)", f"{effective_capital:,.2f}")


            # --- Backtesting and Visualization with Costs ---
            with col2:
                st.subheader("📈 Cumulative Performance (incl. initial costs)")
                portfolio_returns = (daily_returns * pd.Series(weights)).sum(axis=1)
                
                comparison_df = pd.DataFrame({'Portfolio': portfolio_returns, 'Benchmark': benchmark_returns.iloc[:, 0]}).dropna()
                
                # Simulate portfolio value over time starting with effective capital
                portfolio_value = pd.Series(index=comparison_df.index, dtype=float)
                portfolio_value.iloc[0] = effective_capital
                for i in range(1, len(comparison_df)):
                    portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + comparison_df['Portfolio'].iloc[i])
                
                # Simulate benchmark value starting with original capital
                benchmark_value = pd.Series(index=comparison_df.index, dtype=float)
                benchmark_value.iloc[0] = capital
                for i in range(1, len(comparison_df)):
                    benchmark_value.iloc[i] = benchmark_value.iloc[i-1] * (1 + comparison_df['Benchmark'].iloc[i])

                # Convert values back to cumulative percentage returns for plotting
                cumulative_returns_df = pd.DataFrame({
                    'Optimized Portfolio': (portfolio_value / capital) - 1,
                    'Nifty 50 Benchmark': (benchmark_value / capital) - 1
                })

                # --- Plotting ---
                plt.style.use('seaborn-v0_8-darkgrid')
                fig, ax = plt.subplots(figsize=(12, 7))
                ax.plot(cumulative_returns_df.index, cumulative_returns_df['Optimized Portfolio'], label='Optimized Portfolio', color='royalblue', lw=2.5)
                ax.plot(cumulative_returns_df.index, cumulative_returns_df['Nifty 50 Benchmark'], label='Nifty 50 Benchmark', color='darkorange', lw=2.5)
                
                ax.fill_between(cumulative_returns_df.index, cumulative_returns_df['Optimized Portfolio'], alpha=0.1, color='royalblue')
                
                ax.set_title(f'Portfolio Performance vs. Nifty 50 (Costs Included)\n({optimization_method})', fontsize=16, weight='bold')
                ax.set_ylabel('Cumulative Return', fontsize=12)
                ax.set_xlabel('Date', fontsize=12)
                ax.legend(loc='upper left', fontsize=10)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.tight_layout()
                
                st.pyplot(fig)

                # --- Performance Metrics ---
                st.subheader("Performance Metrics")
                trading_days = 252
                final_portfolio_return = cumulative_returns_df['Optimized Portfolio'].iloc[-1]
                final_benchmark_return = cumulative_returns_df['Nifty 50 Benchmark'].iloc[-1]
                
                portfolio_vol = comparison_df['Portfolio'].std() * np.sqrt(trading_days)
                benchmark_vol = comparison_df['Benchmark'].std() * np.sqrt(trading_days)

                st.metric(label="Final Portfolio Return (Net of Costs)", value=f"{final_portfolio_return:.2%}")
                st.metric(label="Final Nifty 50 Return", value=f"{final_benchmark_return:.2%}")
                st.metric(label="Portfolio Annualized Volatility", value=f"{portfolio_vol:.2%}")

else:
    st.info("Please configure your portfolio in the sidebar and click 'Calculate & Backtest' to begin.")
