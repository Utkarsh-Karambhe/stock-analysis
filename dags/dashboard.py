import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
import matplotlib.ticker as mtick

# Set page configuration
st.set_page_config(
    page_title="Stock Market Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        padding-top: 1rem;
        font-weight: 600;
    }
    .card {
        background-color: #F8FAFC;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E40AF;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748B;
    }
    .trend-up {
        color: #10B981;
    }
    .trend-down {
        color: #EF4444;
    }
    .footer {
        text-align: center;
        color: #64748B;
        padding: 1rem;
        font-size: 0.8rem;
    }
    /* Make matplotlib figures look better */
    .stPlot > div {
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-radius: 0.5rem;
        padding: 0.5rem;
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# Function to fetch data from PostgreSQL
def fetch_data_from_db():
    try:
        engine = create_engine('postgresql://uk:gmail1234@localhost:5432/stock_data')
        query = "SELECT * FROM stock_prices ORDER BY timestamp DESC LIMIT 1000"
        df = pd.read_sql(query, engine)
        
        # Convert timestamp column to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Convert numeric columns to float
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Coerce invalid values to NaN
        
        # Drop rows with NaN in numeric columns
        df = df.dropna(subset=numeric_columns)
        
        return df
    except Exception as e:
        st.error(f"Database connection error: {e}")
        # Return sample data if database connection fails
        return generate_sample_data()

# Function to generate sample data if database connection fails
def generate_sample_data():
    # First try to get symbols from database (if available)
    db_symbols = None
    try:
        from sqlalchemy import create_engine
        engine = create_engine('postgresql://uk:gmail1234@localhost:5432/stock_data')
        with engine.connect() as conn:
            db_symbols = pd.read_sql("SELECT DISTINCT symbol FROM stock_prices;", conn)['symbol'].tolist()
    except:
        pass  # Silently fall through to sample data
    
    # Use DB symbols if available, otherwise use defaults
    symbols = db_symbols if db_symbols else ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'JPM', 'IBM', 'NVDA', 'NFLX', 'ADBE', 'ORCL', 'PYPL']
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    data = []
    for symbol in symbols:
        base_price = np.random.uniform(100, 500)
        for date in dates:
            if date.weekday() < 5:  # Weekdays only
                # Price calculations (unchanged)
                open_price = float(base_price * (1 + np.random.uniform(-0.02, 0.02)))
                close_price = float(open_price * (1 + np.random.uniform(-0.03, 0.03)))
                high_price = float(max(open_price, close_price) * (1 + np.random.uniform(0, 0.01)))
                low_price = float(min(open_price, close_price) * (1 - np.random.uniform(0, 0.01)))
                volume = int(np.random.randint(100000, 10000000))
                
                data.append({
                    'timestamp': date,
                    'symbol': symbol,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })
                base_price = close_price
                
    return pd.DataFrame(data)

# Calculate technical indicators
def calculate_indicators(df):
    # Sort DataFrame by timestamp
    df = df.sort_values('timestamp')
    
    # Calculate moving averages
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    
    # Calculate Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['Signal']
    
    # Daily returns
    df['daily_return'] = df['close'].pct_change() * 100
    
    return df

# Custom function to create candlestick chart with matplotlib
def plot_candlestick(df, show_ma=True):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot candlesticks
    width = 0.6
    width2 = 0.3
    
    # Define colors for up and down days
    up = df[df.close >= df.open]
    down = df[df.close < df.open]
    
    # Plot up days
    ax.bar(up.index, up.close-up.open, width, bottom=up.open, color='#26A69A', zorder=3)
    ax.bar(up.index, up.high-up.close, width2, bottom=up.close, color='#26A69A', zorder=3)
    ax.bar(up.index, up.low-up.open, width2, bottom=up.open, color='#26A69A', zorder=3)
    
    # Plot down days
    ax.bar(down.index, down.close-down.open, width, bottom=down.open, color='#EF5350', zorder=3)
    ax.bar(down.index, down.high-down.open, width2, bottom=down.open, color='#EF5350', zorder=3)
    ax.bar(down.index, down.low-down.close, width2, bottom=down.close, color='#EF5350', zorder=3)
    
    # Add moving averages if requested
    if show_ma and 'MA5' in df.columns and 'MA20' in df.columns:
        ax.plot(df.index, df['MA5'], color='#FF9800', linewidth=1.5, label='MA5', zorder=4)
        ax.plot(df.index, df['MA20'], color='#2196F3', linewidth=1.5, label='MA20', zorder=4)
        ax.legend(loc='upper left')
    
    # Format x-axis to show dates nicely
    every_nth = max(1, len(df) // 10)  # Show about 10 tick marks
    ax.set_xticks(df.index[::every_nth])
    ax.set_xticklabels(df['timestamp'].dt.strftime('%Y-%m-%d').values[::every_nth], rotation=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.set_title(f"Price Chart", fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    return fig

# Load data
with st.spinner('Loading data...'):
    df = fetch_data_from_db()

# Main header
st.markdown('<div class="main-header">📊 Stock Market Analytics Dashboard</div>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("Dashboard Controls")

# Sidebar filters
symbols = sorted(df['symbol'].unique())
selected_symbol = st.sidebar.selectbox("Select Stock Symbol", symbols)

# Date range filter
min_date = df['timestamp'].min().date()
max_date = df['timestamp'].max().date()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
    df_filtered = df[(df['symbol'] == selected_symbol) & 
                    (df['timestamp'].dt.date >= start_date) & 
                    (df['timestamp'].dt.date <= end_date)]
else:
    df_filtered = df[df['symbol'] == selected_symbol]

# Calculate indicators for the filtered data
if not df_filtered.empty:
    df_filtered = calculate_indicators(df_filtered)
    # Reset index for proper plotting
    df_filtered = df_filtered.reset_index(drop=True)

# Technical indicators selection
show_ma = st.sidebar.checkbox("Show Moving Averages", value=True)
show_volume = st.sidebar.checkbox("Show Volume Chart", value=True)
selected_indicators = st.sidebar.multiselect(
    "Select Technical Indicators",
    ["RSI", "MACD"],
    default=["RSI"]
)

# Color theme settings
st.sidebar.subheader("Chart Settings")
theme = st.sidebar.selectbox(
    "Color Theme", 
    ["Default", "Dark", "Blue", "Green"],
    index=0
)

# Apply theme settings
if theme == "Dark":
    plt.style.use('dark_background')
    line_color = 'white'
    up_color = '#26A69A'
    down_color = '#EF5350'
elif theme == "Blue":
    plt.style.use('default')
    sns.set_palette("Blues_d")
    line_color = '#1E3A8A'
    up_color = '#2563EB'
    down_color = '#DC2626'
elif theme == "Green":
    plt.style.use('default')
    sns.set_palette("Greens_d")
    line_color = '#064E3B'
    up_color = '#059669'
    down_color = '#DC2626'
else:  # Default
    plt.style.use('default')
    line_color = '#1E3A8A'
    up_color = '#26A69A'
    down_color = '#EF5350'

# Display error if no data
if df_filtered.empty:
    st.error(f"No data available for {selected_symbol} in the selected date range.")
    st.stop()

# Key metrics row
col1, col2, col3, col4 = st.columns(4)

# Get latest and previous day data
latest_data = df_filtered.iloc[-1]
if len(df_filtered) > 1:
    prev_data = df_filtered.iloc[-2]
    price_change = latest_data['close'] - prev_data['close']
    price_change_pct = (price_change / prev_data['close']) * 100
else:
    price_change = 0
    price_change_pct = 0

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Current Price</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">${latest_data["close"]:.2f}</div>', unsafe_allow_html=True)
    if price_change >= 0:
        st.markdown(f'<div class="trend-up">▲ ${abs(price_change):.2f} ({abs(price_change_pct):.2f}%)</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="trend-down">▼ ${abs(price_change):.2f} ({abs(price_change_pct):.2f}%)</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Today\'s Range</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">${latest_data["low"]:.2f} - ${latest_data["high"]:.2f}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    # Calculate some statistics
    avg_price = df_filtered['close'].mean()
    max_price = df_filtered['close'].max()
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Avg/Max Price (Period)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">${avg_price:.2f} / ${max_price:.2f}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    # Calculate volatility (standard deviation of daily returns)
    volatility = df_filtered['daily_return'].std() if 'daily_return' in df_filtered.columns else 0
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Volatility (Period)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{volatility:.2f}%</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Main price chart with candlestick
st.markdown('<div class="sub-header">Price Chart</div>', unsafe_allow_html=True)

# Plot candlestick chart
candlestick_fig = plot_candlestick(df_filtered, show_ma)
st.pyplot(candlestick_fig)

# Create a two-column layout for additional charts
col1, col2 = st.columns(2)

# Volume chart
if show_volume:
    with col1:
        st.markdown('<div class="sub-header">Trading Volume</div>', unsafe_allow_html=True)
        
        # Create volume chart with matplotlib
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(
            df_filtered.index,
            df_filtered['volume'],
            color=(0, 0, 1, 0.7),
            width=0.8
        )
        
        # Format x-axis to show dates nicely
        every_nth = max(1, len(df_filtered) // 8)  # Show about 8 tick marks
        ax.set_xticks(df_filtered.index[::every_nth])
        ax.set_xticklabels(df_filtered['timestamp'].dt.strftime('%Y-%m-%d').values[::every_nth], rotation=45)
        
        # Format y-axis with K, M suffixes
        def format_volume(x, pos):
            if x >= 1e6:
                return f'{x*1e-6:.1f}M'
            else:
                return f'{x*1e-3:.0f}K'
        
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(format_volume))
        
        # Add grid and labels
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Date')
        ax.set_ylabel('Volume')
        ax.set_title(f"{selected_symbol} Trading Volume", fontsize=14, fontweight='bold')
        
        fig.tight_layout()
        st.pyplot(fig)

# Technical indicators
if "RSI" in selected_indicators and not df_filtered['RSI'].isna().all():
    with col2:
        st.markdown('<div class="sub-header">Relative Strength Index (RSI)</div>', unsafe_allow_html=True)
        
        # Create RSI chart with matplotlib
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            df_filtered.index,
            df_filtered['RSI'],
            color='#673AB7',
            linewidth=1.5
        )
        
        # Add overbought/oversold lines
        ax.axhline(y=70, color='red', linestyle='--', alpha=0.7)
        ax.axhline(y=30, color='green', linestyle='--', alpha=0.7)
        ax.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
        
        # Format x-axis to show dates nicely
        every_nth = max(1, len(df_filtered) // 8)  # Show about 8 tick marks
        ax.set_xticks(df_filtered.index[::every_nth])
        ax.set_xticklabels(df_filtered['timestamp'].dt.strftime('%Y-%m-%d').values[::every_nth], rotation=45)
        
        # Set y-axis range and add annotations
        ax.set_ylim(0, 100)
        ax.text(df_filtered.index[0], 72, 'Overbought', color='red')
        ax.text(df_filtered.index[0], 28, 'Oversold', color='green')
        
        # Add grid and labels
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Date')
        ax.set_ylabel('RSI Value')
        ax.set_title(f"{selected_symbol} RSI", fontsize=14, fontweight='bold')
        
        fig.tight_layout()
        st.pyplot(fig)

# MACD indicator
if "MACD" in selected_indicators and not df_filtered['MACD'].isna().all():
    st.markdown('<div class="sub-header">Moving Average Convergence Divergence (MACD)</div>', unsafe_allow_html=True)
    
    # Create MACD chart with matplotlib
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot MACD line
    ax.plot(
        df_filtered.index,
        df_filtered['MACD'],
        color='#2196F3',
        linewidth=1.5,
        label='MACD'
    )
    
    # Plot Signal line
    ax.plot(
        df_filtered.index,
        df_filtered['Signal'],
        color='#FF9800',
        linewidth=1.5,
        label='Signal'
    )
    
    # Plot Histogram
    colors = [up_color if val >= 0 else down_color for val in df_filtered['MACD_hist']]
    ax.bar(
        df_filtered.index,
        df_filtered['MACD_hist'],
        color=colors,
        width=0.8,
        alpha=0.7,
        label='Histogram'
    )
    
    # Format x-axis to show dates nicely
    every_nth = max(1, len(df_filtered) // 10)  # Show about 10 tick marks
    ax.set_xticks(df_filtered.index[::every_nth])
    ax.set_xticklabels(df_filtered['timestamp'].dt.strftime('%Y-%m-%d').values[::every_nth], rotation=45)
    
    # Add legend, grid and labels
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Date')
    ax.set_ylabel('MACD Value')
    ax.set_title(f"{selected_symbol} MACD", fontsize=14, fontweight='bold')
    
    # Add zero line
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    fig.tight_layout()
    st.pyplot(fig)

# Additional analysis section with two columns
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="sub-header">Performance Analysis</div>', unsafe_allow_html=True)
    
    # Calculate daily returns for display
    returns = df_filtered['close'].pct_change() * 100
    returns = returns.dropna()
    
    # Performance metrics
    cum_return = ((1 + returns/100).cumprod() - 1) * 100
    total_return = cum_return.iloc[-1] if not cum_return.empty else 0
    
    # Create performance table
    performance_data = {
        "Metric": ["Total Return", "Max Daily Return", "Min Daily Return", "Avg Daily Return", "Volatility (Daily)"],
        "Value": [
            f"{total_return:.2f}%",
            f"{returns.max():.2f}%",
            f"{returns.min():.2f}%",
            f"{returns.mean():.2f}%",
            f"{returns.std():.2f}%"
        ]
    }
    
    st.table(pd.DataFrame(performance_data))
    
    # Cumulative return chart
    if not returns.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(
            df_filtered.index[1:],
            cum_return,
            color='#2196F3',
            linewidth=2
        )
        
        # Format x-axis to show dates nicely
        every_nth = max(1, len(cum_return) // 8)  # Show about 8 tick marks
        ax.set_xticks(df_filtered.index[1:][::every_nth])
        ax.set_xticklabels(df_filtered['timestamp'].iloc[1:].dt.strftime('%Y-%m-%d').values[::every_nth], rotation=45)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        
        # Add reference line at 0%
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Color the area under the line
        ax.fill_between(
            df_filtered.index[1:],
            cum_return,
            0,
            where=(cum_return >= 0),
            color=(38/255, 166/255, 154/255, 0.3),
            interpolate=True
        )
        ax.fill_between(
            df_filtered.index[1:],
            cum_return,
            0,
            where=(cum_return >= 0),
            color="#EF9A9A",
            interpolate=True
        )
        
        # Add grid and labels
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return (%)')
        ax.set_title(f"{selected_symbol} Cumulative Return", fontsize=14, fontweight='bold')
        
        fig.tight_layout()
        st.pyplot(fig)

with col2:
    st.markdown('<div class="sub-header">Daily Returns Distribution</div>', unsafe_allow_html=True)
    
    if not returns.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram with Seaborn for better styling
        sns.histplot(
            returns,
            bins=20,
            kde=True,
            color="#2196F3",
            ax=ax
        )
        
        # Add a vertical line at x=0
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        
        # Format x-axis as percentage
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        
        # Add grid and labels
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Frequency')
        ax.set_title(f"{selected_symbol} Daily Returns Distribution", fontsize=14, fontweight='bold')
        
        fig.tight_layout()
        st.pyplot(fig)
    
    # Correlation heatmap
    st.markdown('<div class="sub-header">Price & Volume Correlation</div>', unsafe_allow_html=True)
    
    # Calculate correlation
    corr_columns = ['open', 'high', 'low', 'close', 'volume']
    corr_matrix = df_filtered[corr_columns].corr().round(2)
    
    # Create heatmap with Seaborn
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='RdBu_r',
        vmin=-1,
        vmax=1,
        fmt='.2f',
        linewidths=0.5,
        cbar_kws={'label': 'Correlation Coefficient'},
        ax=ax
    )
    
    ax.set_title(f"{selected_symbol} Correlation Matrix", fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    st.pyplot(fig)

# Raw data table (collapsible)
with st.expander("View Raw Data", expanded=False):
    st.dataframe(df_filtered.sort_values('timestamp', ascending=False), height=300)

# Display summary statistics in an expander
with st.expander("Summary Statistics", expanded=False):
    st.dataframe(df_filtered[['open', 'high', 'low', 'close', 'volume']].describe().round(2))

# Footer
st.markdown('<div class="footer">© 2025 Stock Market Analytics Dashboard | Developed by Utkarsh | Powered by PostgreSQL & Streamlit</div>', unsafe_allow_html=True)