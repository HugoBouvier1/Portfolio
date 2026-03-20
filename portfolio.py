import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
import calendar
import json
import os
from pathlib import Path
import io

# =============================================================================
# CONSTANTS
# =============================================================================
TRADING_DAYS_PER_YEAR = 252
DAYS_PER_YEAR = 365.25
DEFAULT_RISK_FREE_RATE = 0.03  # 3% - configurable risk-free rate
MIN_PORTFOLIO_VALUE_THRESHOLD = 1000  # Minimum value to calculate weights
DEFAULT_OPTION_MULTIPLIER = 100
MIN_VARIANCE_THRESHOLD = 1e-10  # Prevent division by near-zero

# =============================================================================
# INLINE HELPER FUNCTIONS (for code optimization)
# =============================================================================

def format_currency(value, decimals=0):
    """Format value as currency."""
    return f"${value:,.{decimals}f}"

def format_percentage(value, decimals=2, show_sign=True):
    """Format value as percentage."""
    sign = "+" if value > 0 and show_sign else ""
    return f"{sign}{value:.{decimals}f}%"

def calculate_total_return(start_value, end_value):
    """
    Calculate total return percentage.
    
    Args:
        start_value: Initial investment value
        end_value: Final investment value
    
    Returns:
        Total return as percentage
    """
    if start_value == 0 or pd.isna(start_value):
        return 0
    return ((end_value - start_value) / start_value) * 100

def annualize_return(total_return_pct, days):
    """
    Convert total return to annualized return using CAGR formula.
    
    Args:
        total_return_pct: Total return as percentage (e.g., 25 for 25%)
        days: Number of days held
    
    Returns:
        Annualized return as percentage
    """
    if days <= 0:
        return 0
    total_return_decimal = total_return_pct / 100
    years = days / DAYS_PER_YEAR
    if years == 0:
        return total_return_pct
    # CAGR formula: ((1 + total_return)^(1/years) - 1)
    return ((1 + total_return_decimal) ** (1 / years) - 1) * 100

def is_put_option(trans_type):
    """Check if transaction is a put option type."""
    return trans_type in ['BUY_PUT', 'SELL_PUT', 'CLOSE_PUT']

def is_cash_flow(trans_type):
    """Check if transaction is a cash flow type."""
    return trans_type in ['DEPOSIT', 'WITHDRAWAL', 'DIVIDEND']

def calculate_put_intrinsic_value(underlying_price, strike_price, contracts=1, multiplier=DEFAULT_OPTION_MULTIPLIER):
    """
    Calculate intrinsic value of put option.
    
    Put options are in the money when the underlying price is BELOW the strike price.
    This is useful for out-of-the-money puts which profit when the underlying drops.
    
    Args:
        underlying_price: Current price of underlying asset
        strike_price: Option strike price
        contracts: Number of contracts
        multiplier: Contract multiplier (default 100)
    
    Returns:
        Total intrinsic value
    """
    if pd.isna(underlying_price) or pd.isna(strike_price):
        return 0
    # For puts: intrinsic value = max(0, strike - underlying)
    intrinsic = max(0, strike_price - underlying_price)
    return intrinsic * contracts * multiplier

def get_total_deposits(transactions):
    """Calculate total deposits from transaction list."""
    return sum(t['amount'] for t in transactions if t['type'] == 'DEPOSIT')

def get_total_withdrawals(transactions):
    """Calculate total withdrawals from transaction list."""
    return sum(t['amount'] for t in transactions if t['type'] == 'WITHDRAWAL')

def get_net_capital(transactions):
    """Calculate net capital (deposits - withdrawals) from transaction list."""
    return get_total_deposits(transactions) - get_total_withdrawals(transactions)

def get_ticker_buy_transactions(transactions, ticker):
    """Get all BUY transactions for a specific ticker."""
    return [t for t in transactions if t['type'] == 'BUY' and t['ticker'] == ticker]

def calculate_twr(portfolio_values, cash_flow_dates, cash_flow_amounts):
    """
    Calculate Time-Weighted Return (TWR).
    
    TWR measures investment performance independent of cash flow timing.
    This is the standard for comparing investment manager performance.
    
    Args:
        portfolio_values: Series of portfolio values indexed by date
        cash_flow_dates: List of dates when cash flows occurred
        cash_flow_amounts: List of cash flow amounts (positive = deposit)
    
    Returns:
        TWR as percentage
    """
    if portfolio_values.empty or len(cash_flow_dates) == 0:
        return 0.0
    
    # Sort cash flows by date
    cf_pairs = sorted(zip(cash_flow_dates, cash_flow_amounts), key=lambda x: pd.Timestamp(x[0]))
    cf_dates = [pd.Timestamp(p[0]) for p in cf_pairs]
    cf_amounts = [p[1] for p in cf_pairs]
    
    # Calculate sub-period returns
    sub_returns = []
    portfolio_values.index = pd.to_datetime(portfolio_values.index)
    
    # Start value
    prev_value = portfolio_values.iloc[0]
    prev_date = portfolio_values.index[0]
    
    for i, (cf_date, cf_amount) in enumerate(zip(cf_dates, cf_amounts)):
        cf_date = pd.Timestamp(cf_date)
        
        # Find portfolio value just before cash flow
        mask = portfolio_values.index <= cf_date
        if mask.any():
            value_before_cf = portfolio_values[mask].iloc[-1]
            
            # Calculate return for this sub-period (before the cash flow)
            if prev_value > 0:
                # The value before CF includes the effect of the previous CF
                sub_return = value_before_cf / prev_value
                sub_returns.append(sub_return)
            
            # After CF, new starting value = value_before_cf + cf_amount
            prev_value = value_before_cf + cf_amount
            prev_date = cf_date
    
    # Final period: from last cash flow to end
    if prev_value > 0:
        final_return = portfolio_values.iloc[-1] / prev_value
        sub_returns.append(final_return)
    
    # Chain link returns: (1+r1) * (1+r2) * ... - 1
    if sub_returns:
        twr = (np.prod(sub_returns) - 1) * 100
    else:
        # No cash flows - simple return
        if portfolio_values.iloc[0] > 0:
            twr = (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1) * 100
        else:
            twr = 0.0
    
    return twr

def calculate_mwr(dates, cash_flows, final_value, end_date=None):
    """
    Calculate Money-Weighted Return (MWR/IRR).
    
    MWR measures actual investor experience including cash flow timing.
    This is the internal rate of return (IRR) of the investment.
    
    Args:
        dates: List of dates for cash flows (includes start date)
        cash_flows: List of amounts (negative = deposit, positive = withdrawal)
        final_value: Final portfolio value (treated as terminal inflow)
        end_date: End date for terminal value (defaults to now) - FIXED
    
    Returns:
        MWR as annualized percentage
    """
    from scipy.optimize import brentq
    
    if len(dates) == 0 or len(cash_flows) == 0:
        return 0.0
    
    # Convert dates to timestamps
    dates = [pd.Timestamp(d) for d in dates]
    base_date = min(dates)
    
    # FIX: Use actual end_date for terminal value, not max(dates)
    if end_date is None:
        end_date = pd.Timestamp(datetime.now())
    else:
        end_date = pd.Timestamp(end_date)
    
    # Build all cash flows (CF + terminal value at END DATE)
    all_dates = dates + [end_date]
    all_cfs = list(cash_flows) + [final_value]
    
    def npv(rate):
        """Calculate NPV at given rate."""
        total = 0
        for date, cf in zip(all_dates, all_cfs):
            years = (date - base_date).days / DAYS_PER_YEAR
            if years == 0:
                total += cf
            else:
                total += cf / ((1 + rate) ** years)
        return total
    
    try:
        # Find rate where NPV = 0 (this is IRR)
        irr = brentq(npv, -0.99, 5.0, maxiter=1000)
        return irr * 100
    except (ValueError, RuntimeError):
        # If no IRR found in range, return simple return
        total_in = sum(abs(cf) for cf in cash_flows if cf < 0)
        if total_in > 0:
            return (final_value / total_in - 1) * 100
        return 0.0

# Optional PDF generation (reportlab)
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("⚠️ reportlab not installed - PDF generation will be disabled")
    print("To enable: pip install reportlab")

# =============================================================================
# SECTOR AND GEOGRAPHIC CLASSIFICATION DATA
# =============================================================================

SECTOR_MAPPING = {
    # Technology
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'GOOG': 'Technology',
    'META': 'Technology', 'NVDA': 'Technology', 'AVGO': 'Technology',
    'ORCL': 'Technology', 'ADBE': 'Technology', 'CRM': 'Technology', 'CSCO': 'Technology',
    'INTC': 'Technology', 'AMD': 'Technology', 'QCOM': 'Technology', 'TXN': 'Technology',
    'NOW': 'Technology', 'INTU': 'Technology', 'AMAT': 'Technology', 'MU': 'Technology',
    
    # Communication Services
    'NFLX': 'Communication', 'DIS': 'Communication', 'CMCSA': 'Communication', 'T': 'Communication',
    'VZ': 'Communication', 'TMUS': 'Communication', 'CHTR': 'Communication',
    
    # Consumer Discretionary
    'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary', 'HD': 'Consumer Discretionary',
    'NKE': 'Consumer Discretionary', 'MCD': 'Consumer Discretionary', 'SBUX': 'Consumer Discretionary',
    'LOW': 'Consumer Discretionary', 'TGT': 'Consumer Discretionary', 'TJX': 'Consumer Discretionary',
    
    # Consumer Staples
    'WMT': 'Consumer Staples', 'PG': 'Consumer Staples', 'KO': 'Consumer Staples',
    'PEP': 'Consumer Staples', 'COST': 'Consumer Staples', 'PM': 'Consumer Staples',
    'MO': 'Consumer Staples', 'CL': 'Consumer Staples', 'MDLZ': 'Consumer Staples',
    
    # Healthcare
    'UNH': 'Healthcare', 'JNJ': 'Healthcare', 'LLY': 'Healthcare', 'ABBV': 'Healthcare',
    'MRK': 'Healthcare', 'PFE': 'Healthcare', 'TMO': 'Healthcare', 'ABT': 'Healthcare',
    'DHR': 'Healthcare', 'BMY': 'Healthcare', 'AMGN': 'Healthcare', 'CVS': 'Healthcare',
    
    # Financials
    'BRK.B': 'Financials', 'JPM': 'Financials', 'V': 'Financials', 'MA': 'Financials',
    'BAC': 'Financials', 'WFC': 'Financials', 'MS': 'Financials', 'GS': 'Financials',
    'AXP': 'Financials', 'BLK': 'Financials', 'C': 'Financials', 'SCHW': 'Financials',
    
    # Industrials
    'BA': 'Industrials', 'HON': 'Industrials', 'UPS': 'Industrials', 'RTX': 'Industrials',
    'CAT': 'Industrials', 'DE': 'Industrials', 'GE': 'Industrials', 'LMT': 'Industrials',
    
    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
    'EOG': 'Energy', 'PXD': 'Energy', 'MPC': 'Energy', 'PSX': 'Energy',
    
    # Materials
    'LIN': 'Materials', 'APD': 'Materials', 'SHW': 'Materials', 'FCX': 'Materials',
    'ECL': 'Materials', 'NEM': 'Materials', 'DOW': 'Materials',
    
    # Real Estate
    'AMT': 'Real Estate', 'PLD': 'Real Estate', 'CCI': 'Real Estate', 'EQIX': 'Real Estate',
    'SPG': 'Real Estate', 'DLR': 'Real Estate', 'O': 'Real Estate',
    
    # Utilities
    'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'D': 'Utilities',
    'AEP': 'Utilities', 'EXC': 'Utilities', 'SRE': 'Utilities',
    
    # ETFs & Index Funds
    'SPY': 'Equity Index', 'QQQ': 'Equity Index', 'IWM': 'Equity Index',
    'VTI': 'Equity Index', 'VOO': 'Equity Index', 'VEA': 'International Equity',
    'VWO': 'Emerging Markets', 'EEM': 'Emerging Markets', 'IEMG': 'Emerging Markets',
    'TLT': 'Government Bonds', 'IEF': 'Government Bonds', 'SHY': 'Government Bonds',
    'AGG': 'Bond Aggregate', 'BND': 'Bond Aggregate', 'LQD': 'Corporate Bonds',
    'HYG': 'High Yield Bonds', 'JNK': 'High Yield Bonds',
    'GLD': 'Commodities', 'GDX': 'Commodities', 'SLV': 'Commodities',
    'USO': 'Commodities', 'DBC': 'Commodities',
    'VNQ': 'Real Estate', 'IYR': 'Real Estate', 'XLRE': 'Real Estate',
    
    # Momentum, Dividend, Managed Futures ETFs
    'SPMO': 'US Equity Momentum', 'MTUM': 'US Equity Momentum',
    'SCHD': 'US Dividend Equity', 'VIG': 'US Dividend Equity', 'VHYL': 'Dividend Equity',
    'DBMF': 'Managed Futures', 'CTA': 'Managed Futures', 'KMLM': 'Managed Futures',
    'SPLV': 'Low Volatility', 'USMV': 'Low Volatility',
    'QUAL': 'Quality Factor', 'FUSA.L': 'Quality Factor',
    'VXUS': 'International Equity', 'IXUS': 'International Equity',
    'TIP': 'Inflation-Protected Bonds', 'VTIP': 'Inflation-Protected Bonds',
    
    # UCITS equivalents
    'IUMO.L': 'US Equity Momentum', 'IUMF.L': 'US Equity Momentum',
    'DTLA.L': 'Government Bonds', 'IDTL.L': 'Government Bonds',
    'IGLN.L': 'Commodities',
    
    # Sector ETFs
    'XLK': 'Technology', 'XLF': 'Financials', 'XLV': 'Healthcare',
    'XLE': 'Energy', 'XLI': 'Industrials', 'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples', 'XLU': 'Utilities', 'XLB': 'Materials',
}

GEOGRAPHIC_MAPPING = {
    # US Companies
    'AAPL': 'United States', 'MSFT': 'United States', 'GOOGL': 'United States',
    'AMZN': 'United States', 'TSLA': 'United States', 'NVDA': 'United States',
    'META': 'United States', 'BRK.B': 'United States', 'JPM': 'United States',
    'V': 'United States', 'JNJ': 'United States', 'WMT': 'United States',
    
    # International Companies
    'TSM': 'Taiwan', 'ASML': 'Netherlands', 'NVO': 'Denmark', 'TM': 'Japan',
    'BABA': 'China', 'LVMH': 'France', 'SAP': 'Germany', 'HSBC': 'UK',
    
    # ETFs - Geographic
    'SPY': 'United States', 'QQQ': 'United States', 'IWM': 'United States',
    'VTI': 'United States', 'VOO': 'United States', 
    'VEA': 'Developed Markets', 'EFA': 'Developed Markets', 'IEFA': 'Developed Markets',
    'VWO': 'Emerging Markets', 'EEM': 'Emerging Markets', 'IEMG': 'Emerging Markets',
    'VT': 'Global', 'ACWI': 'Global',
    
    # Regional ETFs
    'EWJ': 'Japan', 'EWG': 'Germany', 'EWU': 'UK', 'EWC': 'Canada',
    'EWA': 'Australia', 'EWZ': 'Brazil', 'INDA': 'India', 'FXI': 'China',
    
    # Default to US for bonds, commodities
    'TLT': 'United States', 'IEF': 'United States', 'AGG': 'United States',
    'BND': 'United States', 'GLD': 'Global', 'SLV': 'Global',
    
    # Portfolio ETFs
    'SPMO': 'United States', 'MTUM': 'United States',
    'SCHD': 'United States', 'VIG': 'United States',
    'DBMF': 'Global', 'CTA': 'Global', 'KMLM': 'Global',
    'SPLV': 'United States', 'USMV': 'United States',
    'QUAL': 'United States', 'DBC': 'Global',
    'TIP': 'United States', 'VXUS': 'International',
    
    # UCITS equivalents
    'IUMO.L': 'United States', 'IUMF.L': 'United States',
    'FUSA.L': 'United States',
    'DTLA.L': 'United States', 'IDTL.L': 'United States',
    'IGLN.L': 'Global',
}

STRESS_SCENARIOS = {
    '2008 Financial Crisis': {
        'SPY': -0.37, 'QQQ': -0.42, 'IWM': -0.34, 
        'TLT': 0.34, 'GLD': 0.05, 'VNQ': -0.38,
        'description': 'Global financial crisis (Sep 2008 - Mar 2009)',
        'duration': '6 months'
    },
    '2020 COVID Crash': {
        'SPY': -0.34, 'QQQ': -0.27, 'IWM': -0.43,
        'TLT': 0.21, 'GLD': 0.04, 'VNQ': -0.28,
        'description': 'COVID-19 pandemic market crash (Feb-Mar 2020)',
        'duration': '1 month'
    },
    '2022 Bear Market': {
        'SPY': -0.18, 'QQQ': -0.33, 'IWM': -0.21,
        'TLT': -0.29, 'GLD': -0.01, 'VNQ': -0.25,
        'description': 'Rising rates bear market (Jan-Oct 2022)',
        'duration': '10 months'
    },
    'Dot-Com Crash (2000-2002)': {
        'SPY': -0.49, 'QQQ': -0.83, 'IWM': -0.27,
        'TLT': 0.43, 'GLD': 0.13, 'VNQ': -0.16,
        'description': 'Technology bubble burst (Mar 2000 - Oct 2002)',
        'duration': '31 months'
    },
    'Flash Crash (2010)': {
        'SPY': -0.09, 'QQQ': -0.11, 'IWM': -0.10,
        'TLT': 0.02, 'GLD': 0.01, 'VNQ': -0.08,
        'description': 'May 2010 flash crash',
        'duration': '1 day'
    },
    'Custom: Mild Recession': {
        'SPY': -0.20, 'QQQ': -0.25, 'IWM': -0.22,
        'TLT': 0.10, 'GLD': 0.05, 'VNQ': -0.15,
        'description': 'Hypothetical mild recession scenario',
        'duration': 'Variable'
    },
    'Custom: Severe Market Crash': {
        'SPY': -0.45, 'QQQ': -0.50, 'IWM': -0.48,
        'TLT': 0.25, 'GLD': 0.15, 'VNQ': -0.40,
        'description': 'Hypothetical severe crisis',
        'duration': 'Variable'
    },
    'Custom: Rising Rates Shock': {
        'SPY': -0.15, 'QQQ': -0.20, 'IWM': -0.12,
        'TLT': -0.25, 'GLD': -0.05, 'VNQ': -0.20,
        'description': 'Rapid rate increase scenario',
        'duration': 'Variable'
    },
}

# =============================================================================
# DATA PERSISTENCE FUNCTIONS - MULTI-CLIENT ARCHITECTURE
# =============================================================================

# Create data directory if it doesn't exist
DATA_DIR = Path.home() / '.portfolio_tracker'
DATA_DIR.mkdir(exist_ok=True)

# Multi-client support
CLIENTS_DIR = DATA_DIR / 'clients'
CLIENTS_DIR.mkdir(exist_ok=True)

def get_client_list():
    """Get list of all clients."""
    try:
        clients = []
        if CLIENTS_DIR.exists():
            for client_dir in CLIENTS_DIR.iterdir():
                if client_dir.is_dir():
                    clients.append(client_dir.name)
        return sorted(clients)
    except Exception as e:
        st.error(f"Error loading client list: {str(e)}")
        return []

def create_new_client(client_id, client_name, initial_capital):
    """Create a new client portfolio."""
    try:
        client_dir = CLIENTS_DIR / client_id
        client_dir.mkdir(exist_ok=True)
        
        # Create client metadata
        metadata = {
            'client_id': client_id,
            'client_name': client_name,
            'initial_capital': initial_capital,
            'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(client_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create empty transactions file
        with open(client_dir / 'transactions.json', 'w') as f:
            json.dump([], f)
        
        # Create empty config file
        with open(client_dir / 'config.json', 'w') as f:
            json.dump({}, f)
        
        return True
    except Exception as e:
        st.error(f"Error creating client: {str(e)}")
        return False

def get_client_data_dir(client_id):
    """Get data directory for specific client."""
    return CLIENTS_DIR / client_id

# Legacy support for single-user mode
TRANSACTIONS_FILE = DATA_DIR / 'transactions.json'
CONFIG_FILE = DATA_DIR / 'config.json'

def save_transactions_to_file(transactions, client_id=None):
    """Save transactions to a JSON file with automatic backup."""
    try:
        if client_id:
            trans_file = CLIENTS_DIR / client_id / 'transactions.json'
            backup_file = CLIENTS_DIR / client_id / 'transactions_backup.json'
        else:
            trans_file = TRANSACTIONS_FILE
            backup_file = DATA_DIR / 'transactions_backup.json'
        
        # Create backup of existing file before saving new one
        if trans_file.exists():
            try:
                import shutil
                shutil.copy2(trans_file, backup_file)
            except Exception as backup_error:
                st.warning(f"Could not create backup: {str(backup_error)}")
        
        # Convert datetime objects to strings for JSON serialization
        transactions_serializable = []
        for trans in transactions:
            trans_copy = trans.copy()
            
            # Handle date field
            if hasattr(trans['date'], 'strftime'):
                trans_copy['date'] = trans['date'].strftime('%Y-%m-%d')
            else:
                trans_copy['date'] = str(trans['date'])
            
            # Handle timestamp field
            if isinstance(trans['timestamp'], datetime):
                trans_copy['timestamp'] = trans['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            else:
                trans_copy['timestamp'] = str(trans['timestamp'])
            
            # Handle closes_transaction_id field
            if 'closes_transaction_id' in trans_copy and trans_copy['closes_transaction_id'] is not None:
                if isinstance(trans_copy['closes_transaction_id'], datetime):
                    trans_copy['closes_transaction_id'] = trans_copy['closes_transaction_id'].strftime('%Y-%m-%d %H:%M:%S')
                else:
                    trans_copy['closes_transaction_id'] = str(trans_copy['closes_transaction_id'])
            
            transactions_serializable.append(trans_copy)
        
        # Write to temporary file first
        temp_file = trans_file.parent / 'transactions_temp.json'
        with open(temp_file, 'w') as f:
            json.dump(transactions_serializable, f, indent=2)
        
        # Verify the file can be read back
        with open(temp_file, 'r') as f:
            json.load(f)
        
        # If successful, move temp file to actual file
        import shutil
        shutil.move(str(temp_file), str(trans_file))
        
        return True
        
    except Exception as e:
        st.error(f"Error saving transactions: {str(e)}")
        # Try to clean up temp file
        try:
            if temp_file.exists():
                temp_file.unlink()
        except OSError as cleanup_error:
            st.warning(f"Could not clean up temp file: {cleanup_error}")
        return False

def load_transactions_from_file(client_id=None):
    """Load transactions from JSON file with error recovery."""
    try:
        if client_id:
            trans_file = CLIENTS_DIR / client_id / 'transactions.json'
            backup_file = CLIENTS_DIR / client_id / 'transactions_backup.json'
        else:
            trans_file = TRANSACTIONS_FILE
            backup_file = DATA_DIR / 'transactions_backup.json'
            
        if trans_file.exists():
            try:
                with open(trans_file, 'r') as f:
                    transactions = json.load(f)
                
                # Convert string dates back to datetime objects
                for trans in transactions:
                    trans['date'] = datetime.strptime(trans['date'], '%Y-%m-%d').date()
                    trans['timestamp'] = datetime.strptime(trans['timestamp'], '%Y-%m-%d %H:%M:%S')
                    
                    # Handle closes_transaction_id if present
                    if 'closes_transaction_id' in trans and trans['closes_transaction_id'] is not None:
                        if isinstance(trans['closes_transaction_id'], str) and trans['closes_transaction_id'] != 'None':
                            try:
                                trans['closes_transaction_id'] = datetime.strptime(trans['closes_transaction_id'], '%Y-%m-%d %H:%M:%S')
                            except:
                                pass
                
                return transactions
                
            except json.JSONDecodeError as e:
                st.error(f"❌ Transaction file corrupted at line {e.lineno}, column {e.colno}")
                st.warning("🔧 Attempting to recover from backup...")
                
                # Try to load from backup
                if backup_file.exists():
                    try:
                        with open(backup_file, 'r') as f:
                            transactions = json.load(f)
                        
                        for trans in transactions:
                            trans['date'] = datetime.strptime(trans['date'], '%Y-%m-%d').date()
                            trans['timestamp'] = datetime.strptime(trans['timestamp'], '%Y-%m-%d %H:%M:%S')
                        
                        st.success("✅ Recovered transactions from backup!")
                        
                        # Restore the main file from backup
                        with open(trans_file, 'w') as f:
                            json.dump(transactions, f, indent=2, default=str)
                        
                        return transactions
                    except Exception as backup_error:
                        st.error(f"❌ Backup also corrupted: {str(backup_error)}")
                else:
                    st.error("❌ No backup file found")
                
                # Offer to start fresh
                st.error("⚠️ Unable to recover transactions. Starting fresh.")
                st.info("💡 Your old file has been renamed to transactions_corrupted.json")
                
                # Rename corrupted file
                corrupted_file = trans_file.parent / 'transactions_corrupted.json'
                trans_file.rename(corrupted_file)
                
                return []
                
        return []
        
    except Exception as e:
        st.error(f"Error loading transactions: {str(e)}")
        return []

def save_config_to_file(config, client_id=None):
    """Save portfolio configuration to file."""
    try:
        if client_id:
            cfg_file = CLIENTS_DIR / client_id / 'config.json'
        else:
            cfg_file = CONFIG_FILE
            
        config_serializable = config.copy()
        if 'investment_date' in config_serializable:
            config_serializable['investment_date'] = config_serializable['investment_date'].strftime('%Y-%m-%d')
        
        with open(cfg_file, 'w') as f:
            json.dump(config_serializable, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving configuration: {str(e)}")
        return False

def load_config_from_file(client_id=None):
    """Load portfolio configuration from file."""
    try:
        if client_id:
            cfg_file = CLIENTS_DIR / client_id / 'config.json'
        else:
            cfg_file = CONFIG_FILE
            
        if cfg_file.exists():
            with open(cfg_file, 'r') as f:
                config = json.load(f)
            
            if 'investment_date' in config:
                config['investment_date'] = datetime.strptime(config['investment_date'], '%Y-%m-%d').date()
            
            return config
        return {}
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        return {}

def save_target_portfolio_to_file(target_portfolio, client_id=None):
    """Save target portfolio allocation to a dedicated JSON file (like transactions)."""
    try:
        if client_id:
            portfolio_file = CLIENTS_DIR / client_id / 'target_portfolio.json'
            backup_file = CLIENTS_DIR / client_id / 'target_portfolio_backup.json'
        else:
            portfolio_file = DATA_DIR / 'target_portfolio.json'
            backup_file = DATA_DIR / 'target_portfolio_backup.json'
        
        # Create backup of existing file before saving new one
        if portfolio_file.exists():
            try:
                import shutil
                shutil.copy2(portfolio_file, backup_file)
            except Exception as backup_error:
                pass  # Silently continue if backup fails
        
        # Write to temporary file first
        temp_file = portfolio_file.parent / 'target_portfolio_temp.json'
        with open(temp_file, 'w') as f:
            json.dump(target_portfolio, f, indent=2)
        
        # Verify the file can be read back
        with open(temp_file, 'r') as f:
            json.load(f)
        
        # If successful, move temp file to actual file
        import shutil
        shutil.move(str(temp_file), str(portfolio_file))
        
        return True
        
    except Exception as e:
        st.error(f"Error saving target portfolio: {str(e)}")
        return False

def save_benchmark_to_file(benchmark_components, benchmark_weights, client_id=None):
    """Save benchmark configuration to a dedicated JSON file."""
    try:
        if client_id:
            benchmark_file = CLIENTS_DIR / client_id / 'benchmark_config.json'
            backup_file = CLIENTS_DIR / client_id / 'benchmark_config_backup.json'
        else:
            benchmark_file = DATA_DIR / 'benchmark_config.json'
            backup_file = DATA_DIR / 'benchmark_config_backup.json'
        
        # Create backup of existing file before saving new one
        if benchmark_file.exists():
            try:
                import shutil
                shutil.copy2(benchmark_file, backup_file)
            except Exception as backup_error:
                pass  # Silently continue if backup fails
        
        benchmark_data = {
            'components': benchmark_components,
            'weights': benchmark_weights
        }
        
        # Write to temporary file first
        temp_file = benchmark_file.parent / 'benchmark_config_temp.json'
        with open(temp_file, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        # Verify the file can be read back
        with open(temp_file, 'r') as f:
            json.load(f)
        
        # If successful, move temp file to actual file
        import shutil
        shutil.move(str(temp_file), str(benchmark_file))
        
        return True
        
    except Exception as e:
        st.error(f"Error saving benchmark configuration: {str(e)}")
        return False

def load_benchmark_from_file(client_id=None):
    """Load benchmark configuration from file."""
    try:
        if client_id:
            benchmark_file = CLIENTS_DIR / client_id / 'benchmark_config.json'
        else:
            benchmark_file = DATA_DIR / 'benchmark_config.json'
        
        if benchmark_file.exists():
            with open(benchmark_file, 'r') as f:
                benchmark_data = json.load(f)
                return benchmark_data.get('components', []), benchmark_data.get('weights', [])
        return None, None
    except Exception as e:
        return None, None

def load_target_portfolio_from_file(client_id=None):
    """Load target portfolio allocation from dedicated JSON file."""
    try:
        if client_id:
            portfolio_file = CLIENTS_DIR / client_id / 'target_portfolio.json'
            backup_file = CLIENTS_DIR / client_id / 'target_portfolio_backup.json'
        else:
            portfolio_file = DATA_DIR / 'target_portfolio.json'
            backup_file = DATA_DIR / 'target_portfolio_backup.json'
            
        if portfolio_file.exists():
            try:
                with open(portfolio_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                # Try backup
                if backup_file.exists():
                    try:
                        with open(backup_file, 'r') as f:
                            portfolio = json.load(f)
                        # Restore from backup
                        with open(portfolio_file, 'w') as f:
                            json.dump(portfolio, f, indent=2)
                        st.success("✅ Recovered target portfolio from backup!")
                        return portfolio
                    except:
                        pass
                return {}
        return {}
        
    except Exception as e:
        st.error(f"Error loading target portfolio: {str(e)}")
        return {}

def load_client_metadata(client_id):
    """Load client metadata."""
    try:
        metadata_file = CLIENTS_DIR / client_id / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        st.error(f"Error loading metadata: {str(e)}")
        return {}

def clear_all_data():
    """Clear all saved data."""
    try:
        if TRANSACTIONS_FILE.exists():
            TRANSACTIONS_FILE.unlink()
        if CONFIG_FILE.exists():
            CONFIG_FILE.unlink()
        return True
    except Exception as e:
        st.error(f"Error clearing data: {str(e)}")
        return False

# =============================================================================
# PROFESSIONAL ANALYSIS FUNCTIONS
# =============================================================================

def get_sector(ticker):
    """Get sector for a given ticker."""
    return SECTOR_MAPPING.get(ticker, 'Other')

def get_geography(ticker):
    """Get geographic region for a given ticker."""
    return GEOGRAPHIC_MAPPING.get(ticker, 'United States')

def calculate_sector_exposure(holdings_dict, prices_dict):
    """
    Calculate portfolio exposure by sector.
    
    Args:
        holdings_dict: Dict of ticker -> shares
        prices_dict: Dict of ticker -> current price
    
    Returns:
        DataFrame with sector exposures
    """
    sector_exposure = {}
    total_value = 0
    
    for ticker, shares in holdings_dict.items():
        if ticker == 'CASH':
            sector_exposure['Cash'] = sector_exposure.get('Cash', 0) + shares
            total_value += shares
            continue
            
        if ticker in prices_dict:
            value = shares * prices_dict[ticker]
            sector = get_sector(ticker)
            sector_exposure[sector] = sector_exposure.get(sector, 0) + value
            total_value += value
    
    if total_value > 0:
        sector_df = pd.DataFrame([
            {'Sector': sector, 'Value': value, 'Weight': (value/total_value)*100}
            for sector, value in sector_exposure.items()
        ]).sort_values('Weight', ascending=False)
        return sector_df
    
    return pd.DataFrame()

def calculate_geographic_exposure(holdings_dict, prices_dict):
    """
    Calculate portfolio exposure by geography.
    
    Args:
        holdings_dict: Dict of ticker -> shares
        prices_dict: Dict of ticker -> current price
    
    Returns:
        DataFrame with geographic exposures
    """
    geo_exposure = {}
    total_value = 0
    
    for ticker, shares in holdings_dict.items():
        if ticker == 'CASH':
            continue
            
        if ticker in prices_dict:
            value = shares * prices_dict[ticker]
            geography = get_geography(ticker)
            geo_exposure[geography] = geo_exposure.get(geography, 0) + value
            total_value += value
    
    if total_value > 0:
        geo_df = pd.DataFrame([
            {'Geography': geo, 'Value': value, 'Weight': (value/total_value)*100}
            for geo, value in geo_exposure.items()
        ]).sort_values('Weight', ascending=False)
        return geo_df
    
    return pd.DataFrame()

def run_stress_test(holdings_dict, prices_dict, scenario_name):
    """
    Run stress test scenario on portfolio.
    
    Args:
        holdings_dict: Dict of ticker -> shares
        prices_dict: Dict of ticker -> current price
        scenario_name: Name of stress scenario
    
    Returns:
        Dict with stress test results
    """
    if scenario_name not in STRESS_SCENARIOS:
        return None
    
    scenario = STRESS_SCENARIOS[scenario_name]
    
    current_value = 0
    stressed_value = 0
    position_impacts = []
    
    for ticker, shares in holdings_dict.items():
        if ticker == 'CASH':
            current_value += shares
            stressed_value += shares
            continue
            
        if ticker in prices_dict:
            position_value = shares * prices_dict[ticker]
            current_value += position_value
            
            # Apply stress shock
            shock = scenario.get(ticker, None)
            if shock is None:
                # Try to map to similar asset class
                sector = get_sector(ticker)
                if 'Technology' in sector or 'Communication' in sector:
                    shock = scenario.get('QQQ', -0.25)
                elif 'Bond' in sector or sector == 'Government Bonds':
                    shock = scenario.get('TLT', 0.0)
                elif sector == 'Commodities':
                    shock = scenario.get('GLD', 0.0)
                else:
                    shock = scenario.get('SPY', -0.20)
            
            stressed_position_value = position_value * (1 + shock)
            stressed_value += stressed_position_value
            
            position_impacts.append({
                'Ticker': ticker,
                'Current Value': position_value,
                'Stressed Value': stressed_position_value,
                'Impact': stressed_position_value - position_value,
                'Shock': shock * 100
            })
    
    total_impact = stressed_value - current_value
    impact_pct = (total_impact / current_value * 100) if current_value > 0 else 0
    
    return {
        'scenario': scenario_name,
        'description': scenario['description'],
        'duration': scenario['duration'],
        'current_value': current_value,
        'stressed_value': stressed_value,
        'total_impact': total_impact,
        'impact_pct': impact_pct,
        'position_impacts': pd.DataFrame(position_impacts)
    }

def check_position_sizing_alerts(holdings_dict, prices_dict, target_weights, alert_threshold=0.10):
    """
    Check for position sizing violations.
    
    Args:
        holdings_dict: Dict of ticker -> shares
        prices_dict: Dict of ticker -> current price
        target_weights: Dict of ticker -> target weight
        alert_threshold: Threshold for alerts (default 10%)
    
    Returns:
        List of alerts
    """
    alerts = []
    total_value = sum(
        shares * prices_dict.get(ticker, 0) 
        for ticker, shares in holdings_dict.items() 
        if ticker != 'CASH'
    )
    
    if total_value == 0:
        return alerts
    
    for ticker, shares in holdings_dict.items():
        if ticker == 'CASH':
            continue
            
        if ticker in prices_dict:
            position_value = shares * prices_dict[ticker]
            current_weight = position_value / total_value
            target_weight = target_weights.get(ticker, 0)
            
            # Check concentration
            if current_weight > alert_threshold:
                alerts.append({
                    'type': 'CONCENTRATION',
                    'severity': 'HIGH' if current_weight > 0.15 else 'MEDIUM',
                    'ticker': ticker,
                    'message': f'{ticker} represents {current_weight*100:.1f}% of portfolio (>${alert_threshold*100:.0f}%)',
                    'current_weight': current_weight * 100,
                    'threshold': alert_threshold * 100
                })
            
            # Check drift from target
            if target_weight > 0:
                drift = abs(current_weight - target_weight)
                if drift > 0.05:  # 5% drift
                    alerts.append({
                        'type': 'DRIFT',
                        'severity': 'MEDIUM' if drift < 0.10 else 'HIGH',
                        'ticker': ticker,
                        'message': f'{ticker} has drifted {drift*100:.1f}% from target',
                        'current_weight': current_weight * 100,
                        'target_weight': target_weight * 100,
                        'drift': drift * 100
                    })
    
    return alerts

def create_custom_benchmark(components, weights):
    """
    Create a custom composite benchmark.
    
    Args:
        components: List of ticker symbols
        weights: List of weights (must sum to 1.0)
    
    Returns:
        Dict with benchmark info
    """
    if len(components) != len(weights):
        return None
    
    if abs(sum(weights) - 1.0) > 0.01:
        return None
    
    return {
        'components': components,
        'weights': weights,
        'name': ' / '.join([f"{w*100:.0f}% {c}" for c, w in zip(components, weights)])
    }

def calculate_benchmark_performance(benchmark_def, prices_df):
    """
    Calculate performance of custom benchmark.
    
    Args:
        benchmark_def: Dict with components and weights
        prices_df: DataFrame with prices
    
    Returns:
        Series with benchmark values over time
    """
    benchmark_value = None
    
    for component, weight in zip(benchmark_def['components'], benchmark_def['weights']):
        if component in prices_df.columns:
            component_returns = prices_df[component].pct_change().fillna(0)
            weighted_returns = component_returns * weight
            
            if benchmark_value is None:
                benchmark_value = weighted_returns
            else:
                benchmark_value += weighted_returns
    
    if benchmark_value is not None:
        return (1 + benchmark_value).cumprod()
    
    return None

def generate_pdf_report(portfolio_data, output_path):
    """
    Generate professional PDF report with matplotlib charts (no kaleido dependency).
    """
    if not REPORTLAB_AVAILABLE:
        return False
        
    try:
        from reportlab.lib import colors as pdf_colors
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import matplotlib.ticker as mticker
        CHARTS_AVAILABLE = True
        
        # Colors
        DEEP_BLUE = '#003d5b'
        ACCENT_BLUE = '#0077b6'
        LIGHT_BG = '#f0f4f8'
        LIGHT_GRAY = '#f8f9fa'
        GRID_COLOR = '#dee2e6'
        GREEN = '#2e7d32'
        RED = '#c62828'
        
        # Ensure output directory exists
        from pathlib import Path as PPath
        PPath(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        doc = SimpleDocTemplate(
            str(output_path), 
            pagesize=letter,
            topMargin=0.6*inch, 
            bottomMargin=0.6*inch,
            leftMargin=0.7*inch, 
            rightMargin=0.7*inch
        )
        
        story = []
        _temp_files = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'],
            fontSize=26, textColor=pdf_colors.HexColor(DEEP_BLUE), spaceAfter=4,
            alignment=TA_CENTER, fontName='Helvetica-Bold')
        
        subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'],
            fontSize=11, textColor=pdf_colors.HexColor(ACCENT_BLUE), spaceAfter=8,
            alignment=TA_CENTER, fontName='Helvetica-Oblique')
        
        heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'],
            fontSize=14, textColor=pdf_colors.HexColor(DEEP_BLUE), spaceAfter=8,
            spaceBefore=10, fontName='Helvetica-Bold',
            borderColor=pdf_colors.HexColor(DEEP_BLUE), borderWidth=2,
            borderPadding=6, backColor=pdf_colors.HexColor(LIGHT_BG))
        
        subheading_style = ParagraphStyle('SubHeading', parent=styles['Heading3'],
            fontSize=11, textColor=pdf_colors.HexColor(ACCENT_BLUE), spaceAfter=6,
            spaceBefore=12, fontName='Helvetica-Bold')
        
        body_style = ParagraphStyle('Body', parent=styles['Normal'],
            fontSize=9, textColor=pdf_colors.HexColor('#444444'), spaceAfter=6,
            fontName='Helvetica')
        
        footer_style = ParagraphStyle('Footer', parent=styles['Normal'],
            fontSize=7, textColor=pdf_colors.HexColor('#888888'), alignment=TA_CENTER, spaceAfter=3)
        
        # Helper to save matplotlib figure to temp file
        def fig_to_image(fig, width_inches=6.8, height_inches=2.8):
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            fig.savefig(tmp.name, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close(fig)
            _temp_files.append(tmp.name)
            return Image(tmp.name, width=width_inches*inch, height=height_inches*inch)
        
        # =====================================================================
        # TITLE
        # =====================================================================
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("Portfolio Performance Report", title_style))
        story.append(Paragraph(
            f"{portfolio_data.get('start_date', '')} — {portfolio_data.get('end_date', '')}", 
            subtitle_style))
        story.append(Spacer(1, 0.15*inch))
        
        # =====================================================================
        # EXECUTIVE SUMMARY - KPI CARDS
        # =====================================================================
        story.append(Paragraph("Executive Summary", heading_style))
        story.append(Spacer(1, 0.08*inch))
        
        current_value = portfolio_data.get('current_value', 0)
        total_return = portfolio_data.get('total_return', 0)
        annual_return = portfolio_data.get('annual_return', 0)
        sharpe = portfolio_data.get('sharpe_ratio', 0)
        max_dd = portfolio_data.get('max_drawdown', 0)
        volatility = portfolio_data.get('volatility', 0)
        total_deposits = portfolio_data.get('total_deposits', 0)
        total_invested = portfolio_data.get('total_invested', 0)
        
        # Determine colors for returns
        ret_color = GREEN if total_return >= 0 else RED
        ann_color = GREEN if annual_return >= 0 else RED
        
        summary_data = [
            ['Metric', 'Value', 'Metric', 'Value'],
            ['Portfolio Value', f"${current_value:,.0f}", 'Total Deposits', f"${total_deposits:,.0f}"],
            ['Total Return', f"{total_return:+.2f}%", 'Total Invested', f"${total_invested:,.0f}"],
            ['Annualized Return', f"{annual_return:+.2f}%", 'Sharpe Ratio', f"{sharpe:.2f}"],
            ['Max Drawdown', f"{max_dd:.2f}%", 'Volatility', f"{volatility:.2f}%"],
        ]
        
        # Calculate profit
        profit = current_value - total_deposits
        summary_data.append(['Total Profit/Loss', f"${profit:+,.0f}", '', ''])
        
        from reportlab.platypus import KeepTogether
        
        summary_table = Table(summary_data, colWidths=[1.9*inch, 1.5*inch, 1.9*inch, 1.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), pdf_colors.HexColor(DEEP_BLUE)),
            ('TEXTCOLOR', (0, 0), (-1, 0), pdf_colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
            ('ALIGN', (3, 1), (3, -1), 'RIGHT'),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 1), (2, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [pdf_colors.HexColor(LIGHT_GRAY), pdf_colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, pdf_colors.HexColor(GRID_COLOR)),
            ('BOX', (0, 0), (-1, -1), 1.5, pdf_colors.HexColor(DEEP_BLUE)),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.15*inch))
        
        # =====================================================================
        # PORTFOLIO GROWTH + DRAWDOWN (combined in one figure to avoid orphans)
        # =====================================================================
        if CHARTS_AVAILABLE and 'portfolio_value_series' in portfolio_data and portfolio_data['portfolio_value_series'] is not None:
            try:
                pv = portfolio_data['portfolio_value_series']
                bv = portfolio_data.get('benchmark_value_series', None)
                
                # Combined figure: growth on top, drawdown on bottom
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.8), height_ratios=[3, 1.5], gridspec_kw={'hspace': 0.35})
                
                # --- TOP: Portfolio Growth (Log Scale) ---
                ax1.plot(pv.index, pv.values, color=DEEP_BLUE, linewidth=1.5, label='Portfolio', zorder=3)
                
                if bv is not None and len(bv) > 0:
                    ax1.plot(bv.index, bv.values, color='#aaaaaa', linewidth=1.2, linestyle='--', label='Benchmark', zorder=2)
                    ax1.legend(loc='upper left', fontsize=7, framealpha=0.9)
                
                ax1.set_yscale('log')
                # Force clean dollar formatting (no scientific notation)
                ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
                ax1.yaxis.set_minor_formatter(mticker.NullFormatter())
                ax1.set_yticks([t for t in [50000, 75000, 100000, 150000, 200000, 300000, 500000, 750000, 1000000] if t >= pv.min() * 0.8 and t <= pv.max() * 1.2])
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
                ax1.tick_params(axis='x', rotation=45, labelsize=6)
                ax1.tick_params(axis='y', labelsize=7)
                ax1.set_ylabel('Portfolio Value', fontsize=8)
                ax1.set_title('Portfolio Growth (Log Scale)', fontsize=10, fontweight='bold', color=DEEP_BLUE, loc='left')
                ax1.grid(True, alpha=0.25, linewidth=0.5)
                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)
                
                # Start/end annotations
                start_val = pv.iloc[0]
                end_val = pv.iloc[-1]
                ax1.annotate(f'${start_val:,.0f}', xy=(pv.index[0], start_val), fontsize=6, color='#888888')
                ax1.annotate(f'${end_val:,.0f}', xy=(pv.index[-1], end_val), fontsize=7, color=DEEP_BLUE, fontweight='bold')
                
                # --- BOTTOM: Drawdown ---
                running_max = pv.cummax()
                drawdown = ((pv - running_max) / running_max) * 100
                
                ax2.fill_between(drawdown.index, drawdown.values, 0, color=RED, alpha=0.2, zorder=2)
                ax2.plot(drawdown.index, drawdown.values, color=RED, linewidth=0.7, zorder=3)
                ax2.axhline(y=0, color='#333333', linewidth=0.5)
                
                min_dd_idx = drawdown.idxmin()
                min_dd_val = drawdown.min()
                ax2.annotate(f'Max: {min_dd_val:.1f}%', xy=(min_dd_idx, min_dd_val),
                           fontsize=7, color=RED, fontweight='bold',
                           xytext=(15, -10), textcoords='offset points',
                           arrowprops=dict(arrowstyle='->', color=RED, lw=0.7))
                
                ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
                ax2.tick_params(axis='x', rotation=45, labelsize=6)
                ax2.tick_params(axis='y', labelsize=7)
                ax2.set_ylabel('Drawdown', fontsize=8)
                ax2.set_title('Drawdown Analysis', fontsize=10, fontweight='bold', color=RED, loc='left')
                ax2.grid(True, alpha=0.2, linewidth=0.5)
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                
                fig.tight_layout()
                story.append(fig_to_image(fig, 6.8, 4.2))
            except Exception as e:
                print(f"Portfolio/drawdown chart error: {e}")
        
        story.append(PageBreak())
        
        # =====================================================================
        # ASSET ALLOCATION
        # =====================================================================
        if CHARTS_AVAILABLE and 'holdings' in portfolio_data and portfolio_data['holdings'] is not None and not portfolio_data['holdings'].empty:
            try:
                holdings_df = portfolio_data['holdings']
                sorted_h = holdings_df.sort_values('Weight', ascending=True)
                
                fig, ax = plt.subplots(figsize=(8, max(1.8, len(holdings_df) * 0.4)))
                
                colors_list = [ACCENT_BLUE if w > 0 else RED for w in sorted_h['Weight']]
                bars = ax.barh(sorted_h['Ticker'], sorted_h['Weight'], color=colors_list, edgecolor=DEEP_BLUE, linewidth=0.5, height=0.6)
                
                for bar, val in zip(bars, sorted_h['Weight']):
                    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                           va='center', fontsize=8, color='#333333')
                
                ax.set_xlabel('Weight (%)', fontsize=8)
                ax.set_xlim(0, sorted_h['Weight'].max() * 1.2)
                plt.yticks(fontsize=8)
                plt.xticks(fontsize=7)
                ax.grid(True, axis='x', alpha=0.2, linewidth=0.5)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                fig.tight_layout()
                
                composition_items = [
                    Paragraph("Portfolio Composition", heading_style),
                    Spacer(1, 0.04*inch),
                    fig_to_image(fig, 6.8, max(1.6, len(holdings_df) * 0.32)),
                    Spacer(1, 0.08*inch),
                ]
                story.append(KeepTogether(composition_items))
            except Exception as e:
                print(f"Allocation chart error: {e}")
        
        # =====================================================================
        # CURRENT HOLDINGS TABLE
        # =====================================================================
        if 'holdings' in portfolio_data and portfolio_data['holdings'] is not None and not portfolio_data['holdings'].empty:
            holdings_data_rows = [['Ticker', 'Shares', 'Avg Cost', 'Price', 'Value', 'Gain/Loss', 'Return']]
            
            for _, row in portfolio_data['holdings'].iterrows():
                holdings_data_rows.append([
                    str(row.get('Ticker', '')),
                    f"{row.get('Shares', 0):.2f}",
                    f"${row.get('Avg Cost', 0):.2f}",
                    f"${row.get('Current Price', 0):.2f}",
                    f"${row.get('Market Value', 0):,.0f}",
                    str(row.get('Gain/Loss', '$0')),
                    str(row.get('Return', '0%'))
                ])
            
            ht = Table(holdings_data_rows, colWidths=[0.9*inch, 0.8*inch, 0.9*inch, 0.9*inch, 1.0*inch, 1.1*inch, 0.9*inch])
            ht.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), pdf_colors.HexColor(DEEP_BLUE)),
                ('TEXTCOLOR', (0, 0), (-1, 0), pdf_colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
                ('ALIGN', (0, 1), (0, -1), 'CENTER'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [pdf_colors.white, pdf_colors.HexColor(LIGHT_GRAY)]),
                ('GRID', (0, 0), (-1, -1), 0.5, pdf_colors.HexColor(GRID_COLOR)),
                ('BOX', (0, 0), (-1, -1), 1.5, pdf_colors.HexColor(DEEP_BLUE)),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('LEFTPADDING', (0, 0), (-1, -1), 5),
                ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(KeepTogether([
                Paragraph("Current Holdings", heading_style),
                Spacer(1, 0.04*inch),
                ht,
                Spacer(1, 0.1*inch),
            ]))
        
        # =====================================================================
        # PERFORMANCE ATTRIBUTION CHART
        # =====================================================================
        if CHARTS_AVAILABLE and 'attribution_data' in portfolio_data and portfolio_data['attribution_data']:
            try:
                sorted_attr = sorted(portfolio_data['attribution_data'], key=lambda x: x['Return (%)'], reverse=True)
                assets = [item['Asset'] for item in sorted_attr]
                returns_pct = [item['Return (%)'] for item in sorted_attr]
                
                fig, ax = plt.subplots(figsize=(8, 2.2))
                bar_colors = [GREEN if r >= 0 else RED for r in returns_pct]
                bars = ax.bar(assets, returns_pct, color=bar_colors, edgecolor=DEEP_BLUE, linewidth=0.5, width=0.6)
                
                for bar, val in zip(bars, returns_pct):
                    y_pos = bar.get_height() + 2 if val >= 0 else bar.get_height() - 8
                    ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:+.1f}%',
                           ha='center', fontsize=8, color='#333333', fontweight='bold')
                
                ax.axhline(y=0, color='#333333', linewidth=0.8)
                ax.set_ylabel('Return (%)', fontsize=8)
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:+.0f}%'))
                plt.xticks(rotation=30, fontsize=8, ha='right')
                plt.yticks(fontsize=7)
                ax.grid(True, axis='y', alpha=0.2, linewidth=0.5)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                fig.tight_layout()
                
                story.append(KeepTogether([
                    Paragraph("Performance Attribution", heading_style),
                    Spacer(1, 0.04*inch),
                    fig_to_image(fig, 6.8, 2.0),
                    Spacer(1, 0.06*inch),
                ]))
            except Exception as e:
                print(f"Attribution chart error: {e}")
        
        # =====================================================================
        # DETAILED ATTRIBUTION TABLE
        # =====================================================================
        if 'attribution_data' in portfolio_data and portfolio_data['attribution_data']:
            attr_rows = [['Asset', 'Invested', 'Current Value', 'Gain/Loss', 'Return %']]
            for item in portfolio_data['attribution_data']:
                attr_rows.append([
                    str(item['Asset']),
                    f"${item['Invested']:,.0f}",
                    f"${item['Current Value']:,.0f}",
                    f"${item['Gain/Loss ($)']:+,.0f}",
                    f"{item['Return (%)']:+.2f}%"
                ])
            
            at = Table(attr_rows, colWidths=[1.4*inch, 1.4*inch, 1.4*inch, 1.3*inch, 1.2*inch])
            at.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), pdf_colors.HexColor(DEEP_BLUE)),
                ('TEXTCOLOR', (0, 0), (-1, 0), pdf_colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [pdf_colors.white, pdf_colors.HexColor(LIGHT_GRAY)]),
                ('GRID', (0, 0), (-1, -1), 0.5, pdf_colors.HexColor(GRID_COLOR)),
                ('BOX', (0, 0), (-1, -1), 1.5, pdf_colors.HexColor(DEEP_BLUE)),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(KeepTogether([
                Paragraph("Detailed Attribution", subheading_style),
                Spacer(1, 0.04*inch),
                at,
                Spacer(1, 0.1*inch),
            ]))
        
        # =====================================================================
        # SECTOR & GEOGRAPHIC ALLOCATION
        # =====================================================================
        has_sector = 'sector_exposure' in portfolio_data and portfolio_data['sector_exposure'] is not None and not portfolio_data.get('sector_exposure', pd.DataFrame()).empty
        has_geo = 'geographic_exposure' in portfolio_data and portfolio_data['geographic_exposure'] is not None and not portfolio_data.get('geographic_exposure', pd.DataFrame()).empty
        
        if CHARTS_AVAILABLE and (has_sector or has_geo):
            story.append(Paragraph("Sector & Geographic Allocation", heading_style))
            story.append(Spacer(1, 0.06*inch))
            
            if has_sector:
                try:
                    sector_df = portfolio_data['sector_exposure']
                    sorted_s = sector_df.sort_values('Weight', ascending=True)
                    
                    fig, ax = plt.subplots(figsize=(8, max(2.0, len(sector_df) * 0.35)))
                    ax.barh(sorted_s['Sector'], sorted_s['Weight'], color=ACCENT_BLUE, edgecolor=DEEP_BLUE, linewidth=0.5, height=0.6)
                    for i, (val, name) in enumerate(zip(sorted_s['Weight'], sorted_s['Sector'])):
                        ax.text(val + 0.3, i, f'{val:.1f}%', va='center', fontsize=7, color='#333333')
                    ax.set_xlabel('Weight (%)', fontsize=8)
                    ax.set_xlim(0, sorted_s['Weight'].max() * 1.2)
                    plt.yticks(fontsize=7)
                    plt.xticks(fontsize=7)
                    ax.grid(True, axis='x', alpha=0.2)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.set_title('Sector Allocation', fontsize=9, fontweight='bold', color=DEEP_BLUE, loc='left')
                    fig.tight_layout()
                    story.append(fig_to_image(fig, 6.8, max(1.8, len(sector_df) * 0.3)))
                    story.append(Spacer(1, 0.1*inch))
                except Exception as e:
                    print(f"Sector chart error: {e}")
            
            if has_geo:
                try:
                    geo_df = portfolio_data['geographic_exposure']
                    sorted_g = geo_df.sort_values('Weight', ascending=True)
                    
                    fig, ax = plt.subplots(figsize=(8, max(1.5, len(geo_df) * 0.35)))
                    ax.barh(sorted_g['Geography'], sorted_g['Weight'], color='#4db6ac', edgecolor=DEEP_BLUE, linewidth=0.5, height=0.6)
                    for i, val in enumerate(sorted_g['Weight']):
                        ax.text(val + 0.3, i, f'{val:.1f}%', va='center', fontsize=7, color='#333333')
                    ax.set_xlabel('Weight (%)', fontsize=8)
                    ax.set_xlim(0, sorted_g['Weight'].max() * 1.2)
                    plt.yticks(fontsize=7)
                    plt.xticks(fontsize=7)
                    ax.grid(True, axis='x', alpha=0.2)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.set_title('Geographic Allocation', fontsize=9, fontweight='bold', color=DEEP_BLUE, loc='left')
                    fig.tight_layout()
                    story.append(fig_to_image(fig, 6.8, max(1.5, len(geo_df) * 0.3)))
                except Exception as e:
                    print(f"Geographic chart error: {e}")
        
        # =====================================================================
        # RECENT TRANSACTIONS
        # =====================================================================
        if 'transactions' in portfolio_data and portfolio_data['transactions']:
            story.append(PageBreak())
            story.append(Paragraph("Recent Transactions", heading_style))
            story.append(Spacer(1, 0.06*inch))
            
            trans_rows = [['Date', 'Type', 'Ticker', 'Shares', 'Price', 'Amount']]
            recent_trans = portfolio_data['transactions'][-20:]
            for trans in recent_trans:
                trans_rows.append([
                    trans['date'].strftime('%Y-%m-%d'),
                    trans['type'],
                    trans['ticker'],
                    f"{trans['shares']:.2f}" if trans['shares'] > 0 else '-',
                    f"${trans['price']:.2f}" if trans['price'] > 0 else '-',
                    f"${trans['amount']:,.0f}"
                ])
            
            tt = Table(trans_rows, colWidths=[1.1*inch, 0.9*inch, 0.9*inch, 0.8*inch, 0.9*inch, 1.1*inch])
            tt.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), pdf_colors.HexColor(DEEP_BLUE)),
                ('TEXTCOLOR', (0, 0), (-1, 0), pdf_colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [pdf_colors.white, pdf_colors.HexColor(LIGHT_GRAY)]),
                ('GRID', (0, 0), (-1, -1), 0.5, pdf_colors.HexColor(GRID_COLOR)),
                ('BOX', (0, 0), (-1, -1), 1.5, pdf_colors.HexColor(DEEP_BLUE)),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(tt)
        
        # =====================================================================
        # FOOTER
        # =====================================================================
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", footer_style))
        story.append(Paragraph("This report is for informational purposes only and does not constitute investment advice.", footer_style))
        
        # Build PDF
        doc.build(story)
        
        # Clean up temp image files
        for tmp_file in _temp_files:
            try:
                os.unlink(tmp_file)
            except OSError:
                pass
        
        return True
        
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False

@st.cache_data(ttl=3600)
def fetch_data(tickers, start, end, _cache_key=None):
    """
    Fetch historical price data for given tickers from Yahoo Finance.
    
    Args:
        tickers: List of ticker symbols
        start: Start date
        end: End date
        _cache_key: Optional parameter to bust cache (not used in function)
    
    Returns:
        Dictionary with ticker as key and DataFrame as value
    """
    data = {}
    failed = []
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if not df.empty:
                # Store the raw DataFrame - handle MultiIndex in extraction
                data[ticker] = df
            else:
                failed.append(ticker)
        except Exception as e:
            failed.append(ticker)
    
    if failed:
        st.sidebar.warning(f"⚠️ Could not fetch: {', '.join(failed)}")
    
    return data

# =============================================================================
# CALCULATION FUNCTIONS
# =============================================================================
def calculate_metrics(df):
    """
    Calculate basic price metrics for a given DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        Tuple of (current_price, percent_change, annualized_volatility)
    """
    if df.empty:
        return None, None, None
    
    current_price = float(df['Close'].iloc[-1])
    prev_price = float(df['Close'].iloc[0])
    change = float(((current_price - prev_price) / prev_price) * 100)
    volatility = float(df['Close'].pct_change().std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100)
    
    return current_price, change, volatility

def normalize_prices(data):
    """Normalize all prices to start at 100 for comparison."""
    normalized = pd.DataFrame()
    for ticker, df in data.items():
        if not df.empty:
            normalized[ticker] = (df['Close'] / df['Close'].iloc[0]) * 100
    return normalized

def calculate_risk_metrics(returns, risk_free_rate=None):
    """
    Calculate comprehensive risk metrics for a return series.
    
    Args:
        returns: Pandas Series of daily returns
        risk_free_rate: Annual risk-free rate (default uses DEFAULT_RISK_FREE_RATE)
    
    Returns:
        Dictionary of risk metrics including VaR, CVaR, Sharpe ratio, etc.
    """
    if returns.empty or len(returns) < 2:
        return {}
    
    if risk_free_rate is None:
        risk_free_rate = DEFAULT_RISK_FREE_RATE
    
    # Annual metrics
    annual_returns = returns.mean() * TRADING_DAYS_PER_YEAR
    annual_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    # VaR and CVaR (95% and 99% confidence levels)
    VaR_95 = np.percentile(returns.dropna(), 5)
    VaR_99 = np.percentile(returns.dropna(), 1)
    
    # CVaR with edge case handling
    returns_below_var95 = returns[returns <= VaR_95]
    returns_below_var99 = returns[returns <= VaR_99]
    
    CVaR_95 = returns_below_var95.mean() if len(returns_below_var95) > 0 else VaR_95
    CVaR_99 = returns_below_var99.mean() if len(returns_below_var99) > 0 else VaR_99
    
    # Handle NaN values
    if pd.isna(CVaR_95):
        CVaR_95 = VaR_95
    if pd.isna(CVaR_99):
        CVaR_99 = VaR_99
    
    # Skewness and Kurtosis
    clean_returns = returns.dropna()
    if len(clean_returns) > 2:
        skew = stats.skew(clean_returns)
        kurt = stats.kurtosis(clean_returns)
    else:
        skew = 0
        kurt = 0
    
    # Maximum drawdown calculation
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Sharpe ratio with risk-free rate
    excess_return = annual_returns - risk_free_rate
    sharpe_ratio = (excess_return / annual_vol) if annual_vol > MIN_VARIANCE_THRESHOLD else 0
    
    return {
        'annual_return': annual_returns * 100,
        'annual_volatility': annual_vol * 100,
        'sharpe_ratio': sharpe_ratio,
        'risk_free_rate': risk_free_rate * 100,
        'var_95': VaR_95 * 100,
        'cvar_95': CVaR_95 * 100,
        'var_99': VaR_99 * 100,
        'es_99': CVaR_99 * 100,
        'skewness': skew,
        'kurtosis': kurt,
        'max_drawdown': max_drawdown * 100
    }

def calculate_monthly_returns(portfolio_value, name="Portfolio"):
    """
    Calculate monthly returns and organize in a year x month matrix.
    
    Args:
        portfolio_value: Series of portfolio values indexed by date
        name: Name for the portfolio (unused but kept for compatibility)
    
    Returns:
        DataFrame with years as rows, months as columns, and compound YTD
    """
    if portfolio_value.empty:
        return pd.DataFrame()
    
    # Resample to monthly frequency
    monthly_values = portfolio_value.resample('M').last()
    monthly_returns = monthly_values.pct_change().dropna()
    
    # Create DataFrame with year and month
    monthly_returns_df = pd.DataFrame(index=monthly_returns.index)
    monthly_returns_df['Year'] = monthly_returns.index.year
    monthly_returns_df['Month'] = monthly_returns.index.month
    monthly_returns_df['Return'] = monthly_returns.values * 100
    
    # Pivot to get years as rows and months as columns
    pivot_table = monthly_returns_df.pivot(index='Year', columns='Month', values='Return')
    
    # Rename columns to month abbreviations
    pivot_table.columns = [calendar.month_abbr[i] for i in pivot_table.columns]
    
    # Add YTD column - COMPOUND returns, not average
    # Formula: ((1 + r1/100) * (1 + r2/100) * ... - 1) * 100
    def compound_ytd(row):
        """Calculate compound YTD return from monthly returns."""
        monthly_returns_decimal = row.dropna() / 100
        if len(monthly_returns_decimal) == 0:
            return 0
        compound_return = (1 + monthly_returns_decimal).prod() - 1
        return compound_return * 100
    
    pivot_table['YTD'] = pivot_table.apply(compound_ytd, axis=1)
    
    return pivot_table

def calculate_recovery_time(portfolio_value):
    """Calculate the time needed to recover from maximum drawdown."""
    if portfolio_value.empty:
        return None, None
    
    # Calculate cumulative returns
    cumulative = portfolio_value / portfolio_value.iloc[0]
    
    # Calculate running maximum
    running_max = cumulative.expanding().max()
    
    # Calculate drawdown
    drawdown = (cumulative - running_max) / running_max
    
    # Find maximum drawdown
    max_dd = drawdown.min()
    if max_dd >= 0:
        return 0, None
    
    max_dd_date = drawdown.idxmin()
    
    # Find recovery point
    after_dd = cumulative[cumulative.index > max_dd_date]
    max_before_dd = running_max.loc[max_dd_date]
    
    recovery_points = after_dd[after_dd >= max_before_dd]
    
    if len(recovery_points) > 0:
        recovery_date = recovery_points.index[0]
        recovery_days = (recovery_date - max_dd_date).days
        return recovery_days, recovery_date
    else:
        days_in_dd = (portfolio_value.index[-1] - max_dd_date).days
        return None, days_in_dd

def get_close_prices(df):
    """Extract close prices handling multi-level columns."""
    if isinstance(df.columns, pd.MultiIndex):
        return df['Close'].iloc[:, 0]
    elif 'Close' in df.columns:
        return df['Close']
    else:
        return df.squeeze() if isinstance(df, pd.DataFrame) else df

def calculate_weight_drift(initial_weights, prices_df, initial_date):
    """
    Calculate how portfolio weights drift over time due to price changes.
    
    Args:
        initial_weights: Dict of ticker -> initial weight
        prices_df: DataFrame with ticker prices over time
        initial_date: Date of initial investment
    
    Returns:
        DataFrame with weights over time for each ticker
    """
    # Filter prices from initial date
    prices_from_start = prices_df[prices_df.index >= initial_date].copy()
    
    if prices_from_start.empty:
        return pd.DataFrame()
    
    # Calculate returns from initial date
    initial_prices = prices_from_start.iloc[0]
    price_relatives = prices_from_start.div(initial_prices)
    
    # Calculate portfolio value evolution for each asset
    weights_over_time = pd.DataFrame(index=prices_from_start.index)
    
    for ticker in initial_weights.keys():
        if ticker in price_relatives.columns:
            # Value of this asset over time = initial_weight * price_relative
            weights_over_time[ticker] = initial_weights[ticker] * price_relatives[ticker]
    
    # Calculate total portfolio value at each point
    total_value = weights_over_time.sum(axis=1)
    
    # Normalize to get actual weights
    for ticker in weights_over_time.columns:
        weights_over_time[ticker] = weights_over_time[ticker] / total_value
    
    return weights_over_time

def calculate_portfolio_with_transactions(target_portfolio, prices_df, initial_capital, initial_date, transactions):
    """
    Calculate portfolio value over time incorporating actual transactions including put options.
    
    Args:
        target_portfolio: Dict of ticker -> target weight
        prices_df: DataFrame with ticker prices over time
        initial_capital: Initial investment amount
        initial_date: Date of initial investment
        transactions: List of transaction dictionaries
    
    Returns:
        portfolio_value_series: Portfolio value over time
        holdings_over_time: DataFrame showing shares held in each asset over time
        cash_over_time: Series showing cash balance over time
        put_positions: List of put option positions
    """
    # Initialize holdings - start at ZERO, transactions will build the portfolio
    initial_prices = {ticker: float(prices_df[ticker].iloc[0]) 
                     for ticker in target_portfolio.keys() if ticker in prices_df.columns}
    
    # Sort transactions by date
    sorted_transactions = sorted(transactions, key=lambda x: x['date'])
    
    # Find the first transaction date (usually the first DEPOSIT)
    if sorted_transactions:
        first_trans_date = pd.Timestamp(sorted_transactions[0]['date'])
    else:
        first_trans_date = prices_df.index[0]
    
    # Filter dates to only include from first transaction onwards
    all_dates = prices_df.index
    # Only include dates from first transaction date onwards
    dates = all_dates[all_dates >= first_trans_date]
    
    if len(dates) == 0:
        # If no valid dates, use all dates
        dates = all_dates
    
    # Create time series of holdings - start all at ZERO
    holdings_over_time = pd.DataFrame(index=dates)
    
    for ticker in target_portfolio.keys():
        if ticker in prices_df.columns:
            holdings_over_time[ticker] = 0.0
    
    # Start with ZERO cash - DEPOSIT transactions will add cash
    holdings_over_time['CASH'] = 0.0
    
    # Track put option positions
    put_positions = []  # List of {buy_date, sell_date, underlying, strike, contracts, premium, multiplier}
    active_puts = []  # Currently held puts
    
    # Apply transactions chronologically
    for trans in sorted_transactions:
        trans_date = pd.Timestamp(trans['date'])
        
        # Find the index position for this transaction date
        if trans_date in holdings_over_time.index:
            trans_idx = holdings_over_time.index.get_loc(trans_date)
        else:
            # Find nearest future date
            future_dates = holdings_over_time.index[holdings_over_time.index >= trans_date]
            if len(future_dates) > 0:
                trans_idx = holdings_over_time.index.get_loc(future_dates[0])
            else:
                continue
        
        ticker = trans['ticker']
        
        if trans['type'] == 'BUY':
            if ticker in holdings_over_time.columns:
                holdings_over_time.loc[holdings_over_time.index[trans_idx]:, ticker] += trans['shares']
                holdings_over_time.loc[holdings_over_time.index[trans_idx]:, 'CASH'] -= trans['amount']
        
        elif trans['type'] == 'SELL':
            if ticker in holdings_over_time.columns:
                holdings_over_time.loc[holdings_over_time.index[trans_idx]:, ticker] -= trans['shares']
                holdings_over_time.loc[holdings_over_time.index[trans_idx]:, 'CASH'] += trans['amount']
        
        elif trans['type'] == 'DEPOSIT':
            holdings_over_time.loc[holdings_over_time.index[trans_idx]:, 'CASH'] += trans['amount']
        
        elif trans['type'] == 'WITHDRAWAL':
            holdings_over_time.loc[holdings_over_time.index[trans_idx]:, 'CASH'] -= trans['amount']
        
        elif trans['type'] == 'DIVIDEND':
            holdings_over_time.loc[holdings_over_time.index[trans_idx]:, 'CASH'] += trans['amount']
        
        elif trans['type'] == 'BUY_PUT':
            # Record put purchase
            put_position = {
                'buy_date': trans_date,
                'sell_date': None,
                'underlying': ticker,
                'strike': trans.get('strike', 0),
                'contracts': trans['shares'],
                'buy_premium': trans['price'],
                'multiplier': trans.get('multiplier', 100),
                'total_cost': trans['amount']
            }
            active_puts.append(put_position)
            put_positions.append(put_position)
            
            # Reduce cash by premium paid
            holdings_over_time.loc[holdings_over_time.index[trans_idx]:, 'CASH'] -= trans['amount']
        
        elif trans['type'] == 'SELL_PUT':
            # Find matching active put to close
            matching_put = None
            for put in active_puts:
                if (put['underlying'] == ticker and 
                    put['strike'] == trans.get('strike', 0) and
                    put['sell_date'] is None):
                    matching_put = put
                    break
            
            if matching_put:
                matching_put['sell_date'] = trans_date
                matching_put['sell_premium'] = trans['price']
                matching_put['total_proceeds'] = trans['amount']
                active_puts.remove(matching_put)
            
            # Add proceeds to cash
            holdings_over_time.loc[holdings_over_time.index[trans_idx]:, 'CASH'] += trans['amount']
        
        elif trans['type'] == 'CLOSE_PUT':
            # Close a long put position (sell it back)
            # Find matching active put using the closes_transaction_id
            matching_put = None
            closes_id = trans.get('closes_transaction_id')
            
            if closes_id:
                # Find the put by matching the original BUY_PUT transaction timestamp
                for put in active_puts:
                    # We need to find which BUY_PUT this closes
                    # The closes_transaction_id should match a BUY_PUT timestamp
                    if put['sell_date'] is None:
                        # Check if this is the put we want to close
                        # We'll match by underlying, strike, and it being the right put
                        for orig_trans in sorted_transactions:
                            if (orig_trans.get('timestamp') == closes_id and
                                orig_trans['type'] == 'BUY_PUT' and
                                orig_trans['ticker'] == ticker and
                                orig_trans.get('strike') == trans.get('strike')):
                                # Found the matching put
                                if (put['underlying'] == ticker and 
                                    put['strike'] == trans.get('strike', 0)):
                                    matching_put = put
                                    break
                        if matching_put:
                            break
            
            if matching_put:
                matching_put['sell_date'] = trans_date
                matching_put['sell_premium'] = trans['price']
                matching_put['total_proceeds'] = trans['amount']
                active_puts.remove(matching_put)
                
                # Add proceeds to cash
                holdings_over_time.loc[holdings_over_time.index[trans_idx]:, 'CASH'] += trans['amount']
    
    # Calculate portfolio value at each point including put option values
    portfolio_value_series = pd.Series(index=dates, dtype=float)
    put_value_series = pd.Series(index=dates, dtype=float)
    
    # Build a map of transaction dates and their buy amounts for adjustment
    # This helps handle cases where market prices don't match transaction prices
    trans_by_date = {}
    for trans in sorted_transactions:
        trans_date = pd.Timestamp(trans['date'])
        if trans_date not in trans_by_date:
            trans_by_date[trans_date] = {'buys': {}, 'sells': {}, 'deposits': 0, 'withdrawals': 0, 'dividends': 0}
        if trans['type'] == 'BUY':
            ticker = trans['ticker']
            if ticker not in trans_by_date[trans_date]['buys']:
                trans_by_date[trans_date]['buys'][ticker] = {'shares': 0, 'amount': 0, 'price': trans['price']}
            trans_by_date[trans_date]['buys'][ticker]['shares'] += trans['shares']
            trans_by_date[trans_date]['buys'][ticker]['amount'] += trans['amount']
        elif trans['type'] == 'SELL':
            ticker = trans['ticker']
            if ticker not in trans_by_date[trans_date]['sells']:
                trans_by_date[trans_date]['sells'][ticker] = {'shares': 0, 'amount': 0}
            trans_by_date[trans_date]['sells'][ticker]['shares'] += trans['shares']
            trans_by_date[trans_date]['sells'][ticker]['amount'] += trans['amount']
        elif trans['type'] == 'DEPOSIT':
            trans_by_date[trans_date]['deposits'] += trans['amount']
        elif trans['type'] == 'WITHDRAWAL':
            trans_by_date[trans_date]['withdrawals'] += trans['amount']
        elif trans['type'] == 'DIVIDEND':
            trans_by_date[trans_date]['dividends'] += trans['amount']
    
    # Calculate cost basis for each ticker (average purchase price per share)
    cost_basis = {}  # ticker -> {'total_cost': X, 'total_shares': Y, 'avg_price': Z}
    for trans in sorted_transactions:
        if trans['type'] == 'BUY':
            ticker = trans['ticker']
            if ticker not in cost_basis:
                cost_basis[ticker] = {'total_cost': 0, 'total_shares': 0, 'avg_price': 0}
            cost_basis[ticker]['total_cost'] += trans['amount']
            cost_basis[ticker]['total_shares'] += trans['shares']
            if cost_basis[ticker]['total_shares'] > 0:
                cost_basis[ticker]['avg_price'] = cost_basis[ticker]['total_cost'] / cost_basis[ticker]['total_shares']
    
    # Find price adjustment factors for the first transaction date
    # This helps normalize Yahoo prices to match user's transaction prices
    # IMPORTANT: Only use ORIGINAL (non-rebalancing) transactions for price adjustment
    # Otherwise rebalancing trades will corrupt the adjustment factor
    price_adjustment = {}  # ticker -> adjustment factor
    if sorted_transactions:
        first_trans_date = pd.Timestamp(sorted_transactions[0]['date'])
        # Find nearest date in prices_df
        if first_trans_date in prices_df.index:
            price_date = first_trans_date
        else:
            future_dates = prices_df.index[prices_df.index >= first_trans_date]
            price_date = future_dates[0] if len(future_dates) > 0 else prices_df.index[0]
        
        # Build cost basis ONLY from initial (non-rebalancing) transactions
        initial_cost_basis = {}
        for trans in sorted_transactions:
            # Skip rebalancing transactions when calculating price adjustment
            if trans['type'] == 'BUY' and trans.get('note') != 'AUTO_REBALANCE':
                ticker = trans['ticker']
                if ticker not in initial_cost_basis:
                    initial_cost_basis[ticker] = {'total_cost': 0, 'total_shares': 0}
                initial_cost_basis[ticker]['total_cost'] += trans['amount']
                initial_cost_basis[ticker]['total_shares'] += trans['shares']
        
        for ticker in initial_cost_basis:
            if ticker in prices_df.columns:
                yahoo_price = prices_df.loc[price_date, ticker]
                if initial_cost_basis[ticker]['total_shares'] > 0:
                    user_price = initial_cost_basis[ticker]['total_cost'] / initial_cost_basis[ticker]['total_shares']
                    if yahoo_price > 0 and user_price > 0:
                        # Calculate adjustment factor: how much Yahoo price differs from user price
                        price_adjustment[ticker] = user_price / yahoo_price
    
    for i in range(len(dates)):
        date = dates[i]
        
        # Base value: cash + stock positions
        total_value = holdings_over_time.loc[date, 'CASH']
        
        for ticker in target_portfolio.keys():
            if ticker in prices_df.columns and ticker in holdings_over_time.columns:
                shares_held = holdings_over_time.loc[date, ticker]
                yahoo_price = prices_df.loc[date, ticker]
                
                # Apply price adjustment if available
                # This ensures the portfolio value reflects user's actual cost basis
                if ticker in price_adjustment:
                    adjusted_price = yahoo_price * price_adjustment[ticker]
                    total_value += shares_held * adjusted_price
                else:
                    total_value += shares_held * yahoo_price
        
        # Add value of active put options
        put_value = 0
        for put in put_positions:
            # Check if put is active on this date
            if date >= put['buy_date']:
                if put['sell_date'] is None or date < put['sell_date']:
                    # Put is active - calculate intrinsic value
                    underlying_ticker = put['underlying']
                    
                    # Try to get underlying price
                    if underlying_ticker in prices_df.columns:
                        underlying_price = prices_df.loc[date, underlying_ticker]
                        # Put intrinsic value = max(0, strike - underlying)
                        intrinsic_value = max(0, put['strike'] - underlying_price)
                        put_value += intrinsic_value * put['contracts'] * put['multiplier']
        
        put_value_series.iloc[i] = put_value
        portfolio_value_series.iloc[i] = total_value + put_value
    
    cash_over_time = holdings_over_time['CASH']
    
    return portfolio_value_series, holdings_over_time, cash_over_time, put_positions, put_value_series

def calculate_actual_weights_with_transactions(holdings_over_time, prices_df, target_portfolio):
    """
    Calculate actual portfolio weights over time based on holdings and prices.
    
    Args:
        holdings_over_time: DataFrame with shares held in each asset
        prices_df: DataFrame with prices
        target_portfolio: Dict of target weights
    
    Returns:
        DataFrame with actual weights over time
    """
    weights_over_time = pd.DataFrame(index=holdings_over_time.index)
    
    for date in holdings_over_time.index:
        # Skip if date not in prices
        if date not in prices_df.index:
            for ticker in target_portfolio.keys():
                weights_over_time.loc[date, ticker] = 0
            continue
            
        cash = holdings_over_time.loc[date, 'CASH']
        
        # Check for NaN cash
        if pd.isna(cash):
            cash = 0
        
        total_value = cash
        
        # Calculate value of each position
        position_values = {}
        for ticker in target_portfolio.keys():
            if ticker in holdings_over_time.columns and ticker in prices_df.columns:
                shares = holdings_over_time.loc[date, ticker]
                price = prices_df.loc[date, ticker]
                
                # Handle NaN values
                if pd.isna(shares) or pd.isna(price):
                    position_values[ticker] = 0
                else:
                    position_values[ticker] = shares * price
                    total_value += position_values[ticker]
        
        # Calculate weights - with minimum threshold to avoid division issues
        if total_value > 1000:  # Only calculate weights if portfolio value > $1000
            for ticker in target_portfolio.keys():
                if ticker in position_values:
                    weight = position_values[ticker] / total_value
                    # Cap weights at 100% (1.0) to avoid display issues
                    weights_over_time.loc[date, ticker] = min(weight, 1.0)
                else:
                    weights_over_time.loc[date, ticker] = 0
        else:
            # For dates with very low/zero portfolio value, set to zero
            for ticker in target_portfolio.keys():
                weights_over_time.loc[date, ticker] = 0
    
    return weights_over_time

def calculate_rebalancing_needs(current_weights, target_weights, threshold=0.05):
    """
    Determine if rebalancing is needed based on weight drift.
    
    Args:
        current_weights: Dict of current weights
        target_weights: Dict of target weights
        threshold: Threshold for triggering rebalancing alert
    
    Returns:
        Dict with rebalancing information
    """
    rebalancing_info = {}
    needs_rebalance = False
    
    for ticker in target_weights.keys():
        current = current_weights.get(ticker, 0)
        target = target_weights[ticker]
        drift = current - target
        drift_pct = (drift / target) * 100 if target > 0 else 0
        
        rebalancing_info[ticker] = {
            'current': current * 100,
            'target': target * 100,
            'drift': drift * 100,
            'drift_pct': drift_pct,
            'action': 'BUY' if drift < -threshold else 'SELL' if drift > threshold else 'HOLD'
        }
        
        if abs(drift) > threshold:
            needs_rebalance = True
    
    return rebalancing_info, needs_rebalance

def simulate_portfolio_with_rebalancing(target_portfolio, prices_df, initial_capital, initial_date, 
                                       transactions, rebalance_threshold, rebalance_period_months):
    """
    Simulate portfolio performance with periodic rebalancing.
    
    This function replays the portfolio from the beginning, automatically generating
    rebalancing transactions when:
    1. The time period has elapsed since last rebalance
    2. Any asset drift exceeds the threshold
    
    ONLY REBALANCES ASSETS THAT EXCEED THE THRESHOLD (not the entire portfolio)
    
    Args:
        target_portfolio: Dict of ticker -> target weight
        prices_df: DataFrame with ticker prices over time
        initial_capital: Initial investment amount
        initial_date: Date of initial investment
        transactions: List of original transaction dictionaries (deposits, dividends, etc.)
        rebalance_threshold: Percentage drift that triggers rebalancing (e.g., 0.05 for 5%)
        rebalance_period_months: Minimum months between rebalances
    
    Returns:
        portfolio_value_series: Portfolio value over time with rebalancing
        holdings_over_time: DataFrame showing shares held in each asset over time
        cash_over_time: Series showing cash balance over time
        rebalancing_events: List of rebalancing event dates and actions
        augmented_transactions: Original transactions + generated rebalancing transactions
    """
    print("\n" + "="*80)
    print("DEBUG: Starting simulate_portfolio_with_rebalancing")
    print(f"Initial capital: ${initial_capital:,.2f}")
    print(f"Initial date: {initial_date}")
    print(f"Rebalance threshold: {rebalance_threshold*100:.1f}%")
    print(f"Rebalance period: {rebalance_period_months} months ({rebalance_period_months * 30} days)")
    print(f"Number of original transactions: {len(transactions)}")
    print(f"\nFirst 5 ORIGINAL transactions:")
    for i, trans in enumerate(sorted(transactions, key=lambda x: x['date'])[:5]):
        date_str = trans['date'].strftime('%Y-%m-%d') if hasattr(trans['date'], 'strftime') else str(trans['date'])
        ticker = trans.get('ticker', 'N/A')
        trans_type = trans['type']
        shares = trans.get('shares', 0)
        amount = trans.get('amount', 0)
        print(f"  {i+1}. {date_str} - {trans_type:12s} {ticker:6s} shares={shares:8.2f} amount=${amount:12,.2f}")
    print("="*80 + "\n")
    # Create a copy of transactions to augment with rebalancing trades
    # Convert all dates to datetime.datetime for consistency
    augmented_transactions = []
    for t in transactions:
        t_copy = t.copy()
        # Ensure date is datetime.datetime, not datetime.date
        if hasattr(t_copy['date'], 'date'):
            # It's already a datetime
            pass
        else:
            # It's a date, convert to datetime
            from datetime import datetime as dt_datetime
            t_copy['date'] = dt_datetime.combine(t_copy['date'], dt_datetime.min.time())
        augmented_transactions.append(t_copy)
    
    # Track rebalancing events
    rebalancing_events = []
    last_rebalance_date = None
    
    # Get all dates
    dates = prices_df.index
    
    # Sort transactions by date
    sorted_transactions = sorted(augmented_transactions, key=lambda x: x['date'])
    
    # Build a temporary portfolio state to simulate and detect rebalancing needs
    # Use ORIGINAL transactions only (not augmented) for the initial state
    temp_portfolio_value, temp_holdings, temp_cash, _, _ = calculate_portfolio_with_transactions(
        target_portfolio,
        prices_df,
        initial_capital,
        initial_date,
        transactions  # Use original transactions, not augmented
    )
    
    print(f"DEBUG: Temp portfolio calculated")
    print(f"  First date in temp_portfolio_value: {temp_portfolio_value.index[0]}")
    print(f"  Starting portfolio value: ${temp_portfolio_value.iloc[0]:,.2f}")
    print(f"  Starting cash: ${temp_cash.iloc[0]:,.2f}")
    print(f"  Number of dates in temp portfolio: {len(temp_portfolio_value)}")
    print()
    
    # Define first transaction date
    first_trans_date = pd.Timestamp(sorted_transactions[0]['date'])
    
    # PRE-CALCULATE REBALANCING DATES for efficiency
    # Instead of checking every day, calculate exact dates when rebalancing should occur
    rebalancing_dates = []
    current_rebal_date = first_trans_date + pd.Timedelta(days=rebalance_period_months * 30)
    last_date = dates[-1]
    
    while current_rebal_date <= last_date:
        # Find the nearest actual trading day
        if current_rebal_date in dates:
            rebalancing_dates.append(current_rebal_date)
        else:
            # Find next available trading day
            future_dates = dates[dates >= current_rebal_date]
            if len(future_dates) > 0:
                rebalancing_dates.append(future_dates[0])
        
        # Move to next rebalancing date
        current_rebal_date += pd.Timedelta(days=rebalance_period_months * 30)
    
    print(f"DEBUG: Pre-calculated {len(rebalancing_dates)} potential rebalancing dates")
    print()
    
    # Now only check for rebalancing on the pre-calculated dates
    for rebal_date in rebalancing_dates:
        date = rebal_date
        # Skip if date not in our calculated holdings
        if date not in temp_holdings.index:
            continue
        
        # Make sure we have actual holdings before trying to rebalance
        # Use temp_holdings from our calculated portfolio state
        holdings_on_date = {ticker: temp_holdings.loc[date, ticker] 
                           for ticker in target_portfolio.keys() 
                           if ticker in temp_holdings.columns}
        total_shares = sum(holdings_on_date.values())
        cash_on_date = temp_cash.loc[date] if date in temp_cash.index else 0
        
        if total_shares == 0:
            # No holdings yet - skip rebalancing
            continue
        
        print(f"DEBUG: Checking rebalancing on {date.strftime('%Y-%m-%d')}")
        
        # Calculate current weights
        # Calculate current weights
        current_prices = {}
        for ticker in target_portfolio.keys():
            if ticker in prices_df.columns and date in prices_df.index:
                current_prices[ticker] = prices_df.loc[date, ticker]
        
        # Calculate total portfolio value
        total_value = cash_on_date
        for ticker, shares in holdings_on_date.items():
            if ticker in current_prices:
                total_value += shares * current_prices[ticker]
        
        if total_value > 1000:  # Only rebalance if portfolio > $1000
            # Calculate current weights
            current_weights = {}
            for ticker, shares in holdings_on_date.items():
                if ticker in current_prices:
                    current_weights[ticker] = (shares * current_prices[ticker]) / total_value
                else:
                    current_weights[ticker] = 0
            
            # ONLY REBALANCE ASSETS THAT EXCEED THRESHOLD
            # Identify which assets need rebalancing
            assets_to_rebalance = []
            for ticker, target_weight in target_portfolio.items():
                current_weight = current_weights.get(ticker, 0)
                drift = abs(current_weight - target_weight)
                if drift > rebalance_threshold:
                    assets_to_rebalance.append(ticker)
            
            if assets_to_rebalance:
                # AGGRESSIVE CASH-DEPLOYMENT REBALANCING STRATEGY:
                # Sell overweight assets, then deploy ALL cash to underweight assets
                
                rebalance_actions = []
                sells_needed = []
                all_buys = []
                
                # Identify overweight assets (exceeding threshold)
                for ticker in assets_to_rebalance:
                    if ticker not in current_prices:
                        continue
                    target_weight = target_portfolio[ticker]
                    current_weight = current_weights.get(ticker, 0)
                    target_value = total_value * target_weight
                    current_value = holdings_on_date.get(ticker, 0) * current_prices[ticker]
                    adjustment = target_value - current_value
                    drift = abs(current_weight - target_weight)
                    if abs(adjustment) > 10 and adjustment < 0:
                        sells_needed.append((ticker, adjustment, target_weight, drift))
                
                # Identify ALL underweight assets
                for ticker in target_portfolio.keys():
                    if ticker not in current_prices:
                        continue
                    target_weight = target_portfolio[ticker]
                    current_weight = current_weights.get(ticker, 0)
                    target_value = total_value * target_weight
                    current_value = holdings_on_date.get(ticker, 0) * current_prices[ticker]
                    adjustment = target_value - current_value
                    drift = current_weight - target_weight
                    if adjustment > 10:
                        all_buys.append((ticker, adjustment, target_weight, abs(drift)))
                
                total_sell_proceeds = abs(sum(adj for _, adj, _, _ in sells_needed))
                total_cash_available = cash_on_date + total_sell_proceeds
                
                print(f"DEBUG: Rebalancing on {date.strftime('%Y-%m-%d')}")
                print(f"  Cash: ${cash_on_date:,.2f} + Sells: ${total_sell_proceeds:,.2f} = ${total_cash_available:,.2f}")
                print(f"  Underweight assets: {len(all_buys)}")
                
                if len(sells_needed) == 0 and len(all_buys) == 0:
                    print(f"  ⏭️  SKIP: Balanced")
                    print()
                    continue
                
                # Execute sells
                for ticker, adjustment, target_weight, drift in sells_needed:
                    price = current_prices[ticker]
                    shares = abs(adjustment / price)
                    trans = {'date': date.to_pydatetime(), 'type': 'SELL', 'ticker': ticker,
                            'shares': shares, 'price': price, 'amount': abs(adjustment), 'note': 'AUTO_REBALANCE'}
                    augmented_transactions.append(trans)
                    print(f"  SELL {shares:.2f} {ticker} @ ${price:.2f} = ${abs(adjustment):,.2f}")
                    rebalance_actions.append(f"SELL {shares:.2f} {ticker}")
                
                # Deploy all cash to underweight assets
                all_buys.sort(key=lambda x: x[3], reverse=True)
                remaining_cash = total_cash_available
                for ticker, adjustment, target_weight, drift in all_buys:
                    if remaining_cash < 10:
                        break
                    price = current_prices[ticker]
                    amount = min(abs(adjustment), remaining_cash)
                    shares = amount / price
                    if amount > 10:
                        trans = {'date': date.to_pydatetime(), 'type': 'BUY', 'ticker': ticker,
                                'shares': shares, 'price': price, 'amount': amount, 'note': 'AUTO_REBALANCE'}
                        augmented_transactions.append(trans)
                        print(f"  BUY {shares:.2f} {ticker} @ ${price:.2f} = ${amount:,.2f}")
                        rebalance_actions.append(f"BUY {shares:.2f} {ticker}")
                        remaining_cash -= amount
                print(f"  💰 Remaining cash: ${remaining_cash:,.2f}")
                print()
                
                if rebalance_actions:
                    rebalancing_events.append({
                        'date': date,
                        'actions': rebalance_actions,
                        'assets_count': len(assets_to_rebalance)
                    })
                    last_rebalance_date = date
                    
                    # Recalculate portfolio state after adding rebalancing transactions
                    temp_portfolio_value, temp_holdings, temp_cash, _, _ = calculate_portfolio_with_transactions(
                        target_portfolio,
                        prices_df,
                        initial_capital,
                        initial_date,
                        augmented_transactions
                    )
    
    # Now calculate the full portfolio evolution with augmented transactions
    print(f"\nDEBUG: About to calculate final portfolio")
    sorted_aug_trans = sorted(augmented_transactions, key=lambda x: x['date'])
    print(f"  Total augmented transactions: {len(augmented_transactions)}")
    print(f"  First 10 transactions in augmented_transactions:")
    for i, trans in enumerate(sorted_aug_trans[:10]):
        date_str = trans['date'].strftime('%Y-%m-%d') if hasattr(trans['date'], 'strftime') else str(trans['date'])
        ticker = trans.get('ticker', 'N/A')
        trans_type = trans['type']
        shares = trans.get('shares', 0)
        amount = trans.get('amount', 0)
        note = trans.get('note', '')
        print(f"    {i+1}. {date_str} - {trans_type:12s} {ticker:6s} shares={shares:8.2f} amount=${amount:12,.2f} {note}")
    print()
    
    portfolio_value_series, holdings_over_time, cash_over_time, put_positions, put_value_series = \
        calculate_portfolio_with_transactions(
            target_portfolio,
            prices_df,
            initial_capital,
            initial_date,
            augmented_transactions
        )
    
    print(f"\nDEBUG: Final portfolio calculated with rebalancing")
    print(f"  Number of rebalancing events: {len(rebalancing_events)}")
    print(f"  Total transactions (original + rebalance): {len(augmented_transactions)}")
    print(f"  First date in final portfolio: {portfolio_value_series.index[0]}")
    print(f"  FINAL starting portfolio value: ${portfolio_value_series.iloc[0]:,.2f}")
    print(f"  Expected starting value: ${initial_capital:,.2f}")
    if abs(portfolio_value_series.iloc[0] - initial_capital) > 1:
        print(f"  ⚠️ WARNING: Starting values don't match! Difference: ${portfolio_value_series.iloc[0] - initial_capital:,.2f}")
    print("="*80 + "\n")
    
    return portfolio_value_series, holdings_over_time, cash_over_time, rebalancing_events, augmented_transactions

# =============================================================================
st.set_page_config(
    page_title="Stock Analytics Dashboard", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS - Dark theme styling
# =============================================================================
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .date-box {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    div[data-testid="stNumberInput"] > div > div > input {
        text-align: center;
    }
    .alert-box {
        background-color: #2e3140;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ffa726;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR CONFIGURATION
# =============================================================================
st.sidebar.title("📊 Market Analytics")

# Navigation
page = st.sidebar.radio(
    "Navigate", 
    ["Market Overview", "Portfolio Overview", "Portfolio Tracker"],
    key="main_navigation"
)

# =============================================================================
# DATE RANGE SELECTOR (ONLY FOR MARKET OVERVIEW)
# =============================================================================
if page == "Portfolio Overview":
    st.sidebar.markdown('<div class="date-box">', unsafe_allow_html=True)
    st.sidebar.subheader("📅 Date Range Selection")

    # Initialize session state for dates if not exists
    if 'start_date' not in st.session_state:
        st.session_state.start_date = datetime.now() - timedelta(days=365)
    if 'end_date' not in st.session_state:
        st.session_state.end_date = datetime.now()
    if 'max_date_calculated' not in st.session_state:
        st.session_state.max_date_calculated = False

    # Preset date range buttons in a grid
    col1, col2, col3, col4 = st.sidebar.columns(4)
    with col1:
        if st.button("1M", key="1m", use_container_width=True):
            st.session_state.end_date = datetime.now()
            st.session_state.start_date = st.session_state.end_date - timedelta(days=30)
        if st.button("1Y", key="1y", use_container_width=True):
            st.session_state.end_date = datetime.now()
            st.session_state.start_date = st.session_state.end_date - timedelta(days=365)

    with col2:
        if st.button("3M", key="3m", use_container_width=True):
            st.session_state.end_date = datetime.now()
            st.session_state.start_date = st.session_state.end_date - timedelta(days=90)
        if st.button("3Y", key="3y", use_container_width=True):
            st.session_state.end_date = datetime.now()
            st.session_state.start_date = st.session_state.end_date - timedelta(days=1095)

    with col3:
        if st.button("6M", key="6m", use_container_width=True):
            st.session_state.end_date = datetime.now()
            st.session_state.start_date = st.session_state.end_date - timedelta(days=180)
        if st.button("5Y", key="5y", use_container_width=True):
            st.session_state.end_date = datetime.now()
            st.session_state.start_date = st.session_state.end_date - timedelta(days=1825)

    with col4:
        if st.button("MAX", key="max_period", use_container_width=True, help="Find the oldest common date for all selected tickers"):
            st.session_state.max_date_calculated = True
            st.session_state.end_date = datetime.now()

    # Custom date range selector
    st.sidebar.markdown("**Custom Range:**")
    
    def on_date_change():
        """Callback to update session state when user manually changes dates."""
        dr = st.session_state.date_range_input
        if isinstance(dr, (list, tuple)) and len(dr) == 2:
            st.session_state.start_date = datetime.combine(dr[0], datetime.min.time())
            st.session_state.end_date = datetime.combine(dr[1], datetime.min.time())
    
    st.sidebar.date_input(
        "Date range",
        value=(st.session_state.start_date.date(), st.session_state.end_date.date()),
        max_value=datetime.now(),
        label_visibility="collapsed",
        key="date_range_input",
        on_change=on_date_change
    )

    # Show selected range
    days_selected = (st.session_state.end_date - st.session_state.start_date).days
    st.sidebar.success(f"📊 {st.session_state.start_date.strftime('%b %d, %Y')} → {st.session_state.end_date.strftime('%b %d, %Y')}")
    st.sidebar.info(f"📈 Period: {days_selected} days")
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Initialize session state for dates if not already done (needed for Portfolio Tracker)
if 'start_date' not in st.session_state:
    st.session_state.start_date = datetime.now() - timedelta(days=365)
if 'end_date' not in st.session_state:
    st.session_state.end_date = datetime.now()
if 'max_date_calculated' not in st.session_state:
    st.session_state.max_date_calculated = False

# Use session state values
start_date = st.session_state.start_date
end_date = st.session_state.end_date

# =============================================================================
# TICKER SELECTION (ONLY FOR MARKET OVERVIEW)
# =============================================================================
if page == "Portfolio Overview":
    st.sidebar.subheader("Select Tickers")

    # Predefined quick-add categories
    categories = {
        "Stocks": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META"],
        "Indices": ["^GSPC", "^DJI", "^IXIC", "^RUT", "SPY", "QQQ"],
        "Bonds": ["TLT", "IEF", "SHY", "AGG", "BND"],
        "Volatility": ["^VIX", "VIXY", "UVXY"],
        "Commodities": ["GC=F", "CL=F", "SI=F", "GLD", "USO"],
        "Portfolio ETFs": ["SPMO", "SCHD", "TLT", "GLD", "DBMF", "SPLV", "USMV", "QUAL", "VIG", "DBC", "TIP", "VXUS", "IEF"]
    }

    selected_category = st.sidebar.selectbox("Category", list(categories.keys()))
    available_tickers = categories[selected_category]

    # Initialize session state
    if 'custom_tickers_list' not in st.session_state:
        st.session_state.custom_tickers_list = []
    
    if 'market_selected_tickers' not in st.session_state:
        st.session_state.market_selected_tickers = None
    
    if 'last_market_category' not in st.session_state:
        st.session_state.last_market_category = selected_category

    # Combine all available tickers
    all_available_tickers = available_tickers + [t for t in st.session_state.custom_tickers_list if t not in available_tickers]

    # Set default based on category
    if st.session_state.last_market_category != selected_category or st.session_state.market_selected_tickers is None:
        if selected_category == "Stocks":
            default_tickers = ["AAPL", "GOOGL", "MSFT", "AMZN"]
        elif selected_category == "Indices":
            default_tickers = ["^GSPC", "^DJI", "^IXIC"]
        elif selected_category == "Bonds":
            default_tickers = ["TLT", "IEF", "AGG"]
        elif selected_category == "Volatility":
            default_tickers = ["^VIX"]
        elif selected_category == "Portfolio ETFs":
            default_tickers = ["SPMO", "SCHD", "TLT", "GLD", "DBMF"]
        else:
            default_tickers = available_tickers[:3]
        
        default_tickers = [t for t in default_tickers if t in all_available_tickers]
        st.session_state.market_selected_tickers = default_tickers
        st.session_state.last_market_category = selected_category

    # UNIFIED TICKER INPUT: type any ticker and press Enter to add it instantly
    def _on_ticker_add():
        """Callback to process new tickers when Enter is pressed."""
        val = st.session_state.get('custom_ticker_input', '').strip()
        if val:
            import re
            new_tickers = re.split(r'[,;\s]+', val)
            new_tickers = [t.upper().strip() for t in new_tickers if t.strip()]
            for ticker in new_tickers:
                if ticker not in st.session_state.custom_tickers_list:
                    st.session_state.custom_tickers_list.append(ticker)
                if st.session_state.market_selected_tickers is not None and ticker not in st.session_state.market_selected_tickers:
                    st.session_state.market_selected_tickers.append(ticker)
            # Clear the input after processing
            st.session_state.custom_ticker_input = ""
    
    st.sidebar.text_input(
        "➕ Add ticker(s)",
        placeholder="Type ticker and press Enter (e.g. NFLX, DIS)",
        help="Type any ticker symbol and press Enter. It will be added to the selection automatically. Separate multiple tickers with commas.",
        key="custom_ticker_input",
        on_change=_on_ticker_add
    )

    # Rebuild all_available after potential callback additions
    all_available_tickers = available_tickers + [t for t in st.session_state.custom_tickers_list if t not in available_tickers]

    # Multiselect shows all tickers (predefined + custom) with current selection
    selected_tickers = st.sidebar.multiselect(
        "Selected tickers",
        options=all_available_tickers,
        default=st.session_state.market_selected_tickers,
        help="Select from list or type a new ticker above to add it",
        key="market_ticker_multiselect"
    )
    
    # Update session state when user manually changes selection
    st.session_state.market_selected_tickers = selected_tickers

    # Handle MAX date calculation after tickers are selected
    if st.session_state.get('max_date_calculated', False) and selected_tickers:
        with st.sidebar:
            with st.spinner("Finding oldest common date..."):
                oldest_dates = []
                for ticker in selected_tickers:
                    try:
                        # Fetch data going back far (50 years should cover most)
                        far_back = datetime.now() - timedelta(days=365*50)
                        temp_data = yf.download(ticker, start=far_back, end=datetime.now(), progress=False)
                        if not temp_data.empty:
                            oldest_dates.append(temp_data.index[0])
                    except Exception:
                        pass
            
                if oldest_dates:
                    # The common oldest date is the MAX of the oldest dates (latest of all first dates)
                    common_oldest = max(oldest_dates)
                    st.session_state.start_date = common_oldest.to_pydatetime()
                    st.session_state.max_date_calculated = False
                    st.rerun()  # Rerun to apply the new date
                else:
                    st.sidebar.warning("Could not determine MAX date for selected tickers")
                    st.session_state.max_date_calculated = False

else:
    # For Portfolio Tracker, we don't need ticker selection
    # Just set empty list as it's not used
    selected_tickers = []

# Update start_date and end_date from session state (for use in the rest of the app)
start_date = st.session_state.start_date
end_date = st.session_state.end_date

def calculate_recovery_time(portfolio_value):
    """Calculate the time needed to recover from maximum drawdown."""
    if portfolio_value.empty:
        return None, None
    
    # Calculate cumulative returns
    cumulative = portfolio_value / portfolio_value.iloc[0]
    
    # Calculate running maximum
    running_max = cumulative.expanding().max()
    
    # Calculate drawdown
    drawdown = (cumulative - running_max) / running_max
    
    # Find maximum drawdown
    max_dd = drawdown.min()
    if max_dd >= 0:
        return 0, None
    
    max_dd_date = drawdown.idxmin()
    
    # Find recovery point
    after_dd = cumulative[cumulative.index > max_dd_date]
    max_before_dd = running_max.loc[max_dd_date]
    
    recovery_points = after_dd[after_dd >= max_before_dd]
    
    if len(recovery_points) > 0:
        recovery_date = recovery_points.index[0]
        recovery_days = (recovery_date - max_dd_date).days
        return recovery_days, recovery_date
    else:
        days_in_dd = (portfolio_value.index[-1] - max_dd_date).days
        return None, days_in_dd

def get_close_prices(df):
    """Extract close prices handling multi-level columns."""
    if isinstance(df.columns, pd.MultiIndex):
        return df['Close'].iloc[:, 0]
    elif 'Close' in df.columns:
        return df['Close']
    else:
        return df.squeeze() if isinstance(df, pd.DataFrame) else df

def calculate_weight_drift(initial_weights, prices_df, initial_date):
    """
    Calculate how portfolio weights drift over time due to price changes.
    
    Args:
        initial_weights: Dict of ticker -> initial weight
        prices_df: DataFrame with ticker prices over time
        initial_date: Date of initial investment
    
    Returns:
        DataFrame with weights over time for each ticker
    """
    # Filter prices from initial date
    prices_from_start = prices_df[prices_df.index >= initial_date].copy()
    
# =============================================================================
# PAGE 0: MARKET OVERVIEW (Global Market Dashboard)
# =============================================================================
if page == "Market Overview":
    st.title("🌍 Market Overview")
    st.markdown("*Global market snapshot — indices, rates, currencies, commodities, macro & news*")

    # ── FRED API setup ──────────────────────────────────────────────────
    FRED_API_KEY = "3c4b0a0fd72394c1f0a0a255c8f5fd27"

    @st.cache_data(ttl=3600)
    def fetch_fred_series(series_id, observation_start=None, limit=10):
        """Fetch a FRED series via the FRED API."""
        import requests as _req
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        }
        if observation_start:
            params["observation_start"] = observation_start
        try:
            r = _req.get(url, params=params, timeout=15)
            if r.status_code == 200:
                obs = r.json().get("observations", [])
                values = []
                for o in obs:
                    val_str = o.get("value", ".")
                    if val_str in (".", "", None):
                        continue
                    try:
                        values.append({"date": o["date"], "value": float(val_str)})
                    except (ValueError, KeyError):
                        continue
                return values
        except Exception:
            pass
        return []

    @st.cache_data(ttl=3600)
    def fetch_fred_history(series_id, n_points=200):
        """Fetch historical data for charting — ascending order."""
        import requests as _req
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "sort_order": "desc",
            "limit": n_points,
        }
        try:
            r = _req.get(url, params=params, timeout=15)
            if r.status_code == 200:
                obs = r.json().get("observations", [])
                values = []
                for o in obs:
                    val_str = o.get("value", ".")
                    if val_str in (".", "", None):
                        continue
                    try:
                        values.append({"date": o["date"], "value": float(val_str)})
                    except (ValueError, KeyError):
                        continue
                # Return in chronological order
                values.reverse()
                return values
        except Exception:
            pass
        return []

    @st.cache_data(ttl=3600)
    def fetch_fred_latest(series_id):
        """Return (latest_value, latest_date) for a FRED series."""
        obs = fetch_fred_series(series_id, limit=5)
        if obs:
            return obs[0]["value"], obs[0]["date"]
        return None, None

    @st.cache_data(ttl=3600)
    def fetch_fred_latest_yoy(series_id):
        """
        For index-type series (CPI etc), fetch ~14 months and compute YoY % change.
        Returns (yoy_pct_change, latest_date).
        """
        obs = fetch_fred_series(series_id, limit=20)
        if len(obs) < 2:
            return None, None
        # obs is sorted desc (newest first)
        latest_val = obs[0]["value"]
        latest_date = obs[0]["date"]
        latest_dt = pd.Timestamp(latest_date)
        # Find observation ~12 months ago
        for o in obs[1:]:
            o_dt = pd.Timestamp(o["date"])
            months_diff = (latest_dt.year - o_dt.year) * 12 + (latest_dt.month - o_dt.month)
            if months_diff >= 11:  # allow 11-13 months
                old_val = o["value"]
                if old_val != 0:
                    yoy = ((latest_val - old_val) / old_val) * 100
                    return yoy, latest_date
                break
        return None, None

    @st.cache_data(ttl=3600)
    def fetch_fred_batch(series_info_list):
        """
        Fetch latest values for multiple FRED series.
        series_info_list: list of (series_id, val_type) tuples
        Returns dict: series_id -> (value, date)
        For 'pct_chg' types, computes YoY % change automatically.
        """
        results = {}
        for sid, val_type in series_info_list:
            if val_type == "pct_chg":
                val, dt = fetch_fred_latest_yoy(sid)
            else:
                val, dt = fetch_fred_latest(sid)
            results[sid] = (val, dt)
        return results

    # ── Batch Yahoo fetch ───────────────────────────────────────────────
    @st.cache_data(ttl=300)
    def fetch_all_tickers_batch(tickers_list, period_days):
        """
        Download ALL tickers in a SINGLE yf.download call.
        Returns dict: ticker -> Series of Close prices.
        For 1D (period_days=0), fetches intraday data and prepends previous close
        so percentage change is calculated from previous close (like Google/Yahoo).
        """
        result = {}
        try:
            if period_days == 0:
                df_daily = yf.download(tickers_list, period="5d", progress=False, group_by='ticker')
                df_intraday = yf.download(tickers_list, period="1d", interval="5m", progress=False, group_by='ticker')
                
                for tkr in tickers_list:
                    try:
                        if len(tickers_list) == 1:
                            daily_close = df_daily['Close']
                            intra_close = df_intraday['Close'] if not df_intraday.empty else None
                        else:
                            daily_close = df_daily[tkr]['Close'] if tkr in df_daily.columns.get_level_values(0) else None
                            intra_close = df_intraday[tkr]['Close'] if not df_intraday.empty and tkr in df_intraday.columns.get_level_values(0) else None
                        
                        if daily_close is not None:
                            if isinstance(daily_close, pd.DataFrame):
                                daily_close = daily_close.iloc[:, 0]
                            daily_close = daily_close.dropna()
                        
                        if intra_close is not None:
                            if isinstance(intra_close, pd.DataFrame):
                                intra_close = intra_close.iloc[:, 0]
                            intra_close = intra_close.dropna()
                        
                        if daily_close is not None and len(daily_close) >= 2 and intra_close is not None and len(intra_close) >= 1:
                            prev_close = daily_close.iloc[-2]
                            prev_series = pd.Series([prev_close], index=[intra_close.index[0] - pd.Timedelta(minutes=5)])
                            combined = pd.concat([prev_series, intra_close])
                            result[tkr] = combined
                        elif daily_close is not None and len(daily_close) >= 2:
                            result[tkr] = daily_close.iloc[-2:]
                    except Exception:
                        continue
                return result
            else:
                start = datetime.now() - timedelta(days=period_days + 10)
                df = yf.download(tickers_list, start=start, end=datetime.now(), progress=False, group_by='ticker')
            
            if df.empty:
                return result
            for tkr in tickers_list:
                try:
                    if len(tickers_list) == 1:
                        close = df['Close']
                    else:
                        close = df[tkr]['Close'] if tkr in df.columns.get_level_values(0) else None
                    if close is not None:
                        if isinstance(close, pd.DataFrame):
                            close = close.iloc[:, 0]
                        close = close.dropna()
                        if not close.empty:
                            result[tkr] = close
                except Exception:
                    continue
        except Exception:
            pass
        return result

    def make_sparkline_svg(series, width=90, height=28):
        """Generate a tiny inline SVG sparkline string (no Plotly overhead)."""
        if series is None or len(series) < 2:
            return ""
        vals = series.values.astype(float)
        mn, mx = vals.min(), vals.max()
        rng = mx - mn if mx != mn else 1
        # Normalize to SVG coords
        points = []
        for i, v in enumerate(vals):
            x = (i / (len(vals) - 1)) * width
            y = height - ((v - mn) / rng) * (height - 2) - 1
            points.append(f"{x:.1f},{y:.1f}")
        polyline = " ".join(points)
        color = "#00c853" if vals[-1] >= vals[0] else "#ff1744"
        # Create fill polygon (close the path at the bottom)
        fill_points = polyline + f" {width:.1f},{height} 0,{height}"
        fill_opacity = "0.15"
        svg = (
            f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            f'xmlns="http://www.w3.org/2000/svg" style="display:block;">'
            f'<polygon points="{fill_points}" fill="{color}" opacity="{fill_opacity}"/>'
            f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="1.5"/>'
            f'</svg>'
        )
        return svg

    def render_detail_chart(series, label, ticker):
        """Render a full-width interactive Plotly chart for a selected instrument."""
        if series is None or series.empty:
            st.warning(f"No data available for {label}")
            return
        is_up = float(series.iloc[-1]) >= float(series.iloc[0])
        line_color = "#00c853" if is_up else "#ff1744"
        fill_color = "rgba(0,200,83,0.1)" if is_up else "rgba(255,23,68,0.1)"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            mode='lines', name=label,
            line=dict(color=line_color, width=2),
            fill='tozeroy',
            fillcolor=fill_color,
        ))

        start_p = float(series.iloc[0])
        end_p = float(series.iloc[-1])
        chg = ((end_p - start_p) / start_p * 100) if start_p != 0 else 0
        
        # Use log scale for periods >= 3Y to better visualize long-term growth
        use_log = mo_period_days >= 1095
        
        fig.update_layout(
            title=f"{label} ({ticker})  —  {end_p:,.2f}  ({'+' if chg >= 0 else ''}{chg:.2f}%)",
            height=400,
            xaxis=dict(title="Date"),
            yaxis=dict(title="Price / Value (Log)" if use_log else "Price / Value",
                       type="log" if use_log else "linear"),
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=50, r=20, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Sidebar controls ────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Market Overview Settings")
    mo_period_label = st.sidebar.selectbox(
        "Performance period", ["1D", "1W", "1M", "3M", "6M", "1Y", "YTD", "3Y", "5Y", "10Y"],
        index=0, key="mo_perf_period"
    )
    period_days_map = {
        "1D": 0, "1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365, "3Y": 1095, "5Y": 1825, "10Y": 3650,
        "YTD": (datetime.now() - datetime(datetime.now().year, 1, 1)).days,
    }
    mo_period_days = period_days_map[mo_period_label]

    # ── Data definitions ────────────────────────────────────────────────
    indices = {
        "S&P 500": "^GSPC", "Dow Jones": "^DJI", "Nasdaq": "^IXIC",
        "Russell 2000": "^RUT", "FTSE 100": "^FTSE", "DAX": "^GDAXI",
        "CAC 40": "^FCHI", "Nikkei 225": "^N225", "Hang Seng": "^HSI",
    }
    bonds_rates_yf = {
        "US 10Y Yield": "^TNX", "US 30Y Yield": "^TYX",
        "US 3M T-Bill": "^IRX",
        "TLT (20Y+ Bond)": "TLT", "IEF (7-10Y Bond)": "IEF",
        "SHY (1-3Y Bond)": "SHY",
    }
    commodities = {
        "Gold": "GC=F", "Silver": "SI=F",
        "Brent Crude": "BZ=F", "WTI Crude": "CL=F",
        "Natural Gas": "NG=F", "Copper": "HG=F",
    }
    forex = {
        "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X",
        "USD/JPY": "USDJPY=X", "USD/CHF": "USDCHF=X",
        "AUD/USD": "AUDUSD=X", "USD/CAD": "USDCAD=X",
    }
    volatility_map = {
        "VIX": "^VIX",
    }

    all_yf_sections = [
        ("📊 Major Indices", indices, "", 2),
        ("🏦 Bonds & Yields", bonds_rates_yf, "", 3),
        ("🛢️ Commodities", commodities, "$", 2),
        ("💱 Forex", forex, "", 4),
        ("⚡ Volatility", volatility_map, "", 2),
    ]

    # ── SINGLE batch fetch for ALL tickers ──────────────────────────────
    all_tickers_map = {}
    for _title, mapping, _pfx, _dec in all_yf_sections:
        all_tickers_map.update(mapping)

    all_ticker_symbols = list(all_tickers_map.values())

    with st.spinner("Loading market data..."):
        batch_series = fetch_all_tickers_batch(all_ticker_symbols, mo_period_days)

    # Build quotes_data from the batch result
    quotes_data = {}
    for label, tkr in all_tickers_map.items():
        series = batch_series.get(tkr)
        if series is not None and len(series) >= 2:
            current_price = float(series.iloc[-1])
            start_price = float(series.iloc[0])
            pct_change = ((current_price - start_price) / start_price * 100) if start_price != 0 else 0.0
            quotes_data[label] = {"price": current_price, "change": pct_change, "series": series, "ticker": tkr}
        else:
            quotes_data[label] = {"price": None, "change": None, "series": None, "ticker": tkr}

    # ── Session state for selected instrument ───────────────────────────
    if 'mo_selected_instrument' not in st.session_state:
        st.session_state.mo_selected_instrument = None

    # ── Render sections with inline sparklines ──────────────────────────
    for section_title, mapping, prefix, decimals in all_yf_sections:
        st.subheader(section_title)
        n_cols = min(len(mapping), 4)
        cols = st.columns(n_cols)
        for idx, (label, tkr) in enumerate(mapping.items()):
            with cols[idx % n_cols]:
                d = quotes_data.get(label, {})
                price = d.get("price")
                change = d.get("change")
                series = d.get("series")

                if price is not None:
                    if change is not None and change >= 0:
                        chg_color = "#00c853"
                        arrow = "▲"
                    elif change is not None:
                        chg_color = "#ff1744"
                        arrow = "▼"
                    else:
                        chg_color = "#888"
                        arrow = ""
                    chg_str = f"{'+' if change >= 0 else ''}{change:.2f}%" if change is not None else "—"
                    spark_svg = make_sparkline_svg(series, width=100, height=28)

                    st.markdown(
                        f"""<div style="background:#1e2130;padding:12px 14px;border-radius:8px;margin-bottom:8px;">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <span style="color:#aaa;font-size:0.82em;">{label}</span>
                            <span style="color:{chg_color};font-size:0.82em;">{arrow} {chg_str}</span>
                        </div>
                        <div style="font-size:1.25em;font-weight:600;margin:4px 0 6px 0;">{prefix}{price:,.{decimals}f}</div>
                        {spark_svg}
                        </div>""",
                        unsafe_allow_html=True
                    )
                    if st.button(f"🔍 Detail", key=f"btn_{tkr}", use_container_width=True):
                        st.session_state.mo_selected_instrument = label
                else:
                    st.markdown(
                        f"""<div style="background:#1e2130;padding:12px 14px;border-radius:8px;margin-bottom:8px;">
                        <span style="color:#aaa;font-size:0.82em;">{label}</span><br>
                        <span style="color:#666;">unavailable</span>
                        </div>""",
                        unsafe_allow_html=True
                    )

        # Detail chart for selected instrument in this section
        if st.session_state.mo_selected_instrument and st.session_state.mo_selected_instrument in mapping:
            sel_label = st.session_state.mo_selected_instrument
            d = quotes_data.get(sel_label, {})
            render_detail_chart(d.get("series"), sel_label, d.get("ticker", ""))
            if st.button("✕ Close chart", key=f"close_{sel_label}"):
                st.session_state.mo_selected_instrument = None
                st.rerun()

        st.markdown("---")

    # ═════════════════════════════════════════════════════════════════════
    # MACRO DATA via FRED API
    # ═════════════════════════════════════════════════════════════════════
    st.subheader("🏛️ Central Bank Policy Rates (FRED)")

    # All FRED series for rates + macro — fetched in ONE batch
    fred_rates = {
        "Fed Funds Effective Rate": "DFF",
        "US 10Y Treasury Rate": "DGS10",
        "US 2Y Treasury Rate": "DGS2",
        "US 30Y Mortgage Rate": "MORTGAGE30US",
    }
    intl_rates = {
        "ECB Main Refinancing Rate": "ECBMRRFR",
        "ECB Deposit Facility Rate": "ECBDFR",
        "BoE Bank Rate": "IUDSOIA",
    }
    macro_series = {
        "US CPI YoY (%)": "CPIAUCSL",
        "Core PCE YoY (%)": "PCEPILFE",
        "US Unemployment Rate (%)": "UNRATE",
        "US Real GDP Growth (%)": "A191RL1Q225SBEA",
        "US Industrial Production": "INDPRO",
        "US Consumer Sentiment": "UMCSENT",
        "US M2 Money Supply ($T)": "M2SL",
    }

    # Combine ALL series IDs into one batch (all are simple "rate" type, no pct_chg)
    _all_fred_sids = []
    for sid in fred_rates.values():
        _all_fred_sids.append((sid, "rate"))
    for sid in intl_rates.values():
        _all_fred_sids.append((sid, "rate"))
    for sid in macro_series.values():
        _all_fred_sids.append((sid, "rate"))

    with st.spinner("Fetching FRED data..."):
        _all_fred_results = fetch_fred_batch(_all_fred_sids)

    # ── Display: Central Bank Rates ─────────────────────────────────────
    fred_rate_data = {}
    for label, sid in fred_rates.items():
        fred_rate_data[label] = _all_fred_results.get(sid, (None, None))

    rate_cols = st.columns(len(fred_rates))
    for idx, (label, (val, dt)) in enumerate(fred_rate_data.items()):
        with rate_cols[idx]:
            if val is not None:
                st.metric(label, f"{val:.2f}%", help=f"As of {dt}")
            else:
                st.metric(label, "N/A")

    st.markdown("")

    st.markdown("**International Rates**")
    intl_data = {}
    for label, sid in intl_rates.items():
        intl_data[label] = _all_fred_results.get(sid, (None, None))

    intl_cols = st.columns(len(intl_rates))
    for idx, (label, (val, dt)) in enumerate(intl_data.items()):
        with intl_cols[idx]:
            if val is not None:
                st.metric(label, f"{val:.2f}%", help=f"As of {dt}")
            else:
                st.metric(label, "N/A", help="Series may not be available")

    st.markdown("---")

    # ── Display: Macro Indicators ───────────────────────────────────────
    st.subheader("📐 Key Macro Indicators (FRED)")

    macro_data = {}
    for label, sid in macro_series.items():
        macro_data[label] = _all_fred_results.get(sid, (None, None))

    # Compute 10Y-2Y spread from already-fetched data
    v10, _ = fred_rate_data.get("US 10Y Treasury Rate", (None, None))
    v2, _ = fred_rate_data.get("US 2Y Treasury Rate", (None, None))
    if v10 is not None and v2 is not None:
        spread = (v10 - v2) * 100
        macro_data["10Y-2Y Spread (bp)"] = (spread, "computed")
    else:
        macro_data["10Y-2Y Spread (bp)"] = (None, None)

    m_cols = st.columns(4)
    for idx, (label, (val, dt)) in enumerate(macro_data.items()):
        with m_cols[idx % 4]:
            if val is not None:
                if "($T)" in label:
                    st.metric(label, f"${val/1000:.2f}T" if val > 1000 else f"${val:.0f}B", help=f"As of {dt}")
                elif "(bp)" in label:
                    st.metric(label, f"{val:.0f} bp", help=f"As of {dt}")
                else:
                    st.metric(label, f"{val:.1f}", help=f"As of {dt}")
            else:
                st.metric(label, "N/A")

    st.markdown("---")

    # ═════════════════════════════════════════════════════════════════════
    # COUNTRY MACRO EXPLORER
    # ═════════════════════════════════════════════════════════════════════
    st.subheader("🌐 Country Macro Explorer")
    st.markdown("*Type a country name to see its key economic indicators from FRED*")

    # Pre-mapped FRED series by country name
    # Series IDs verified against FRED — using YoY % change series where available
    # "type" field controls formatting: pct = x.xx%, usd_large = $xT/B, index = x.xx, millions = xM, rate = x.xx%
    COUNTRY_FRED_MAP = {
        "france": {
            "flag": "🇫🇷",
            "series": {
                "GDP (current USD)":            {"id": "MKTGDPFRA646NWDB", "type": "usd_large"},
                "Real GDP Growth (% YoY)":      {"id": "NAEXKP01FRQ657S", "type": "pct"},
                "CPI Inflation (% YoY)":        {"id": "FRACPIALLMINMEI",  "type": "pct_chg"},
                "Unemployment Rate (%)":        {"id": "LRHUTTTTFRM156S",  "type": "pct"},
                "Gov. Debt (% of GDP)":         {"id": "GGGDTAFRA188N",    "type": "pct"},
                "Interest Rate (Long-Term)":    {"id": "IRLTLT01FRM156N",  "type": "rate"},
                "Industrial Production":        {"id": "FRAPROINDMISMEI",  "type": "index"},
                "Consumer Confidence":          {"id": "CSCICP03FRM665S",  "type": "index"},
                "Population (millions)":        {"id": "POPTOTFRA647NWDB", "type": "millions"},
            },
        },
        "germany": {
            "flag": "🇩🇪",
            "series": {
                "GDP (current USD)":            {"id": "MKTGDPDEA646NWDB", "type": "usd_large"},
                "Real GDP Growth (% YoY)":      {"id": "NAEXKP01DEQ657S", "type": "pct"},
                "CPI Inflation (% YoY)":        {"id": "DEUCPIALLMINMEI",  "type": "pct_chg"},
                "Unemployment Rate (%)":        {"id": "LRHUTTTTDEM156S",  "type": "pct"},
                "Gov. Debt (% of GDP)":         {"id": "GGGDTADEA188N",    "type": "pct"},
                "Interest Rate (Long-Term)":    {"id": "IRLTLT01DEM156N",  "type": "rate"},
                "Industrial Production":        {"id": "DEUPROINDMISMEI",  "type": "index"},
                "Consumer Confidence":          {"id": "CSCICP03DEM665S",  "type": "index"},
                "Population (millions)":        {"id": "POPTOTDEA647NWDB", "type": "millions"},
            },
        },
        "united kingdom": {
            "flag": "🇬🇧",
            "series": {
                "GDP (current USD)":            {"id": "MKTGDPGBA646NWDB", "type": "usd_large"},
                "Real GDP Growth (% YoY)":      {"id": "NAEXKP01GBQ657S", "type": "pct"},
                "CPI Inflation (% YoY)":        {"id": "GBRCPIALLMINMEI",  "type": "pct_chg"},
                "Unemployment Rate (%)":        {"id": "LRHUTTTTGBM156S",  "type": "pct"},
                "Gov. Debt (% of GDP)":         {"id": "GGGDTAGBA188N",    "type": "pct"},
                "Interest Rate (Long-Term)":    {"id": "IRLTLT01GBM156N",  "type": "rate"},
                "Industrial Production":        {"id": "GBRPROINDMISMEI",  "type": "index"},
                "Consumer Confidence":          {"id": "CSCICP03GBM665S",  "type": "index"},
                "Population (millions)":        {"id": "POPTOTGBA647NWDB", "type": "millions"},
            },
        },
        "italy": {
            "flag": "🇮🇹",
            "series": {
                "GDP (current USD)":            {"id": "MKTGDPITA646NWDB", "type": "usd_large"},
                "Real GDP Growth (% YoY)":      {"id": "NAEXKP01ITQ657S", "type": "pct"},
                "CPI Inflation (% YoY)":        {"id": "ITACPIALLMINMEI",  "type": "pct_chg"},
                "Unemployment Rate (%)":        {"id": "LRHUTTTTITM156S",  "type": "pct"},
                "Gov. Debt (% of GDP)":         {"id": "GGGDTAITAA188N",   "type": "pct"},
                "Interest Rate (Long-Term)":    {"id": "IRLTLT01ITM156N",  "type": "rate"},
                "Industrial Production":        {"id": "ITAPROINDMISMEI",  "type": "index"},
                "Consumer Confidence":          {"id": "CSCICP03ITM665S",  "type": "index"},
                "Population (millions)":        {"id": "POPTOTITA647NWDB", "type": "millions"},
            },
        },
        "spain": {
            "flag": "🇪🇸",
            "series": {
                "GDP (current USD)":            {"id": "MKTGDPESA646NWDB", "type": "usd_large"},
                "Real GDP Growth (% YoY)":      {"id": "NAEXKP01ESQ657S", "type": "pct"},
                "CPI Inflation (% YoY)":        {"id": "ESPCPIALLMINMEI",  "type": "pct_chg"},
                "Unemployment Rate (%)":        {"id": "LRHUTTTTESM156S",  "type": "pct"},
                "Gov. Debt (% of GDP)":         {"id": "GGGDTAESA188N",    "type": "pct"},
                "Interest Rate (Long-Term)":    {"id": "IRLTLT01ESM156N",  "type": "rate"},
                "Industrial Production":        {"id": "ESPPROINDMISMEI",  "type": "index"},
                "Population (millions)":        {"id": "POPTOTESA647NWDB", "type": "millions"},
            },
        },
        "japan": {
            "flag": "🇯🇵",
            "series": {
                "GDP (current USD)":            {"id": "MKTGDPJPA646NWDB", "type": "usd_large"},
                "Real GDP Growth (% YoY)":      {"id": "NAEXKP01JPQ657S", "type": "pct"},
                "CPI Inflation (% YoY)":        {"id": "JPNCPIALLMINMEI",  "type": "pct_chg"},
                "Unemployment Rate (%)":        {"id": "LRHUTTTTJPM156S",  "type": "pct"},
                "Gov. Debt (% of GDP)":         {"id": "GGGDTAJPA188N",    "type": "pct"},
                "Interest Rate (Long-Term)":    {"id": "IRLTLT01JPM156N",  "type": "rate"},
                "Industrial Production":        {"id": "JPNPROINDMISMEI",  "type": "index"},
                "Consumer Confidence":          {"id": "CSCICP03JPM665S",  "type": "index"},
                "Population (millions)":        {"id": "POPTOTJPA647NWDB", "type": "millions"},
            },
        },
        "canada": {
            "flag": "🇨🇦",
            "series": {
                "GDP (current USD)":            {"id": "MKTGDPCAA646NWDB", "type": "usd_large"},
                "Real GDP Growth (% YoY)":      {"id": "NAEXKP01CAQ657S", "type": "pct"},
                "CPI Inflation (% YoY)":        {"id": "CANCPIALLMINMEI",  "type": "pct_chg"},
                "Unemployment Rate (%)":        {"id": "LRHUTTTTCAM156S",  "type": "pct"},
                "Gov. Debt (% of GDP)":         {"id": "GGGDTACAA188N",    "type": "pct"},
                "Interest Rate (Long-Term)":    {"id": "IRLTLT01CAM156N",  "type": "rate"},
                "Industrial Production":        {"id": "CANPROINDMISMEI",  "type": "index"},
                "Consumer Confidence":          {"id": "CSCICP03CAM665S",  "type": "index"},
                "Population (millions)":        {"id": "POPTOTCAA647NWDB", "type": "millions"},
            },
        },
        "australia": {
            "flag": "🇦🇺",
            "series": {
                "GDP (current USD)":            {"id": "MKTGDPAUA646NWDB", "type": "usd_large"},
                "Real GDP Growth (% YoY)":      {"id": "NAEXKP01AUQ657S", "type": "pct"},
                "CPI Inflation (% YoY)":        {"id": "AUSCPIALLMINMEI",  "type": "pct_chg"},
                "Unemployment Rate (%)":        {"id": "LRHUTTTTAUM156S",  "type": "pct"},
                "Gov. Debt (% of GDP)":         {"id": "GGGDTAAUA188N",    "type": "pct"},
                "Interest Rate (Long-Term)":    {"id": "IRLTLT01AUM156N",  "type": "rate"},
                "Industrial Production":        {"id": "AUSPROINDMISMEI",  "type": "index"},
                "Population (millions)":        {"id": "POPTOTAUA647NWDB", "type": "millions"},
            },
        },
        "switzerland": {
            "flag": "🇨🇭",
            "series": {
                "GDP (current USD)":            {"id": "MKTGDPCHA646NWDB", "type": "usd_large"},
                "CPI Inflation (% YoY)":        {"id": "CHECPIALLMINMEI",  "type": "pct_chg"},
                "Unemployment Rate (%)":        {"id": "LRHUTTTTCHM156S",  "type": "pct"},
                "Gov. Debt (% of GDP)":         {"id": "GGGDTACHA188N",    "type": "pct"},
                "Interest Rate (Long-Term)":    {"id": "IRLTLT01CHM156N",  "type": "rate"},
                "Industrial Production":        {"id": "CHEPROINDMISMEI",  "type": "index"},
                "Population (millions)":        {"id": "POPTOTCHA647NWDB", "type": "millions"},
            },
        },
        "china": {
            "flag": "🇨🇳",
            "series": {
                "GDP (current USD)":            {"id": "MKTGDPCNA646NWDB", "type": "usd_large"},
                "CPI Inflation (% YoY)":        {"id": "CHNCPIALLMINMEI",  "type": "pct_chg"},
                "Interest Rate (Short-Term)":   {"id": "IRSTCI01CNM156N",  "type": "rate"},
                "Industrial Production":        {"id": "CHNPROINDMISMEI",  "type": "index"},
                "Population (millions)":        {"id": "POPTOTCNA647NWDB", "type": "millions"},
            },
        },
        "brazil": {
            "flag": "🇧🇷",
            "series": {
                "GDP (current USD)":            {"id": "MKTGDPBRA646NWDB", "type": "usd_large"},
                "CPI Inflation (% YoY)":        {"id": "BRACPIALLMINMEI",  "type": "pct_chg"},
                "Unemployment Rate (%)":        {"id": "LRHUTTTTBRM156S",  "type": "pct"},
                "Interest Rate (Short-Term)":   {"id": "IRSTCI01BRM156N",  "type": "rate"},
                "Population (millions)":        {"id": "POPTOTBRA647NWDB", "type": "millions"},
            },
        },
        "india": {
            "flag": "🇮🇳",
            "series": {
                "GDP (current USD)":            {"id": "MKTGDPINA646NWDB", "type": "usd_large"},
                "CPI Inflation (% YoY)":        {"id": "INDCPIALLMINMEI",  "type": "pct_chg"},
                "Industrial Production":        {"id": "INDPROINDMISMEI",  "type": "index"},
                "Population (millions)":        {"id": "POPTOTINA647NWDB", "type": "millions"},
            },
        },
        "south korea": {
            "flag": "🇰🇷",
            "series": {
                "GDP (current USD)":            {"id": "MKTGDPKRA646NWDB", "type": "usd_large"},
                "Real GDP Growth (% YoY)":      {"id": "NAEXKP01KRQ657S", "type": "pct"},
                "CPI Inflation (% YoY)":        {"id": "KORCPIALLMINMEI",  "type": "pct_chg"},
                "Unemployment Rate (%)":        {"id": "LRHUTTTTKRM156S",  "type": "pct"},
                "Interest Rate (Long-Term)":    {"id": "IRLTLT01KRM156N",  "type": "rate"},
                "Industrial Production":        {"id": "KORPROINDMISMEI",  "type": "index"},
                "Consumer Confidence":          {"id": "CSCICP03KRM665S",  "type": "index"},
                "Population (millions)":        {"id": "POPTOTKRA647NWDB", "type": "millions"},
            },
        },
        "mexico": {
            "flag": "🇲🇽",
            "series": {
                "GDP (current USD)":            {"id": "MKTGDPMXA646NWDB", "type": "usd_large"},
                "CPI Inflation (% YoY)":        {"id": "MEXCPIALLMINMEI",  "type": "pct_chg"},
                "Unemployment Rate (%)":        {"id": "LRHUTTTTMXM156S",  "type": "pct"},
                "Interest Rate (Short-Term)":   {"id": "IRSTCI01MXM156N",  "type": "rate"},
                "Industrial Production":        {"id": "MEXPROINDMISMEI",  "type": "index"},
                "Population (millions)":        {"id": "POPTOTMXA647NWDB", "type": "millions"},
            },
        },
        "united states": {
            "flag": "🇺🇸",
            "series": {
                "GDP (current USD)":            {"id": "MKTGDPUSA646NWDB", "type": "usd_large"},
                "Real GDP Growth (% YoY)":      {"id": "A191RL1Q225SBEA", "type": "pct"},
                "CPI Inflation (% YoY)":        {"id": "USACPIALLMINMEI",  "type": "pct_chg"},
                "Unemployment Rate (%)":        {"id": "UNRATE",          "type": "pct"},
                "Fed Funds Rate (%)":           {"id": "DFF",             "type": "rate"},
                "Gov. Debt (% of GDP)":         {"id": "GFDEGDQ188S",    "type": "pct"},
                "Interest Rate (Long-Term)":    {"id": "IRLTLT01USM156N",  "type": "rate"},
                "Industrial Production":        {"id": "INDPRO",          "type": "index"},
                "Consumer Confidence":          {"id": "UMCSENT",         "type": "index"},
                "Population (millions)":        {"id": "POPTOTUSA647NWDB", "type": "millions"},
            },
        },
    }

    # Aliases — common names that map to the same data
    COUNTRY_FRED_MAP["england"] = COUNTRY_FRED_MAP["united kingdom"]
    COUNTRY_FRED_MAP["uk"] = COUNTRY_FRED_MAP["united kingdom"]
    COUNTRY_FRED_MAP["great britain"] = COUNTRY_FRED_MAP["united kingdom"]
    COUNTRY_FRED_MAP["korea"] = COUNTRY_FRED_MAP["south korea"]
    COUNTRY_FRED_MAP["usa"] = COUNTRY_FRED_MAP["united states"]
    COUNTRY_FRED_MAP["us"] = COUNTRY_FRED_MAP["united states"]
    COUNTRY_FRED_MAP["america"] = COUNTRY_FRED_MAP["united states"]

    # Primary country names (exclude aliases for display)
    _ALIAS_KEYS = {"england", "uk", "great britain", "korea", "usa", "us", "america"}

    # Proper display names for aliases
    _DISPLAY_NAMES = {
        "united states": "United States", "united kingdom": "United Kingdom",
        "south korea": "South Korea",
        "usa": "United States", "us": "United States", "america": "United States",
        "uk": "United Kingdom", "england": "United Kingdom", "great britain": "United Kingdom",
        "korea": "South Korea",
    }

    def get_country_display_name(key):
        """Get proper display name for a country key."""
        return _DISPLAY_NAMES.get(key, key.title())

    def history_to_yoy_pct(hist_data):
        """
        Convert a monthly index series (like CPI) to YoY % change.
        Input: list of {"date": ..., "value": ...} in chronological order.
        Returns: list of {"date": ..., "value": yoy_%_change}.
        """
        if not hist_data or len(hist_data) < 13:
            return []
        df = pd.DataFrame(hist_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        # Find matching month ~12 months back
        result = []
        for i in range(12, len(df)):
            cur_val = df.loc[i, 'value']
            cur_dt = df.loc[i, 'date']
            # Look for value ~12 months ago
            target_dt = cur_dt - pd.DateOffset(months=12)
            # Find closest date
            past_mask = (df['date'] >= target_dt - pd.Timedelta(days=45)) & (df['date'] <= target_dt + pd.Timedelta(days=45))
            past_rows = df[past_mask]
            if not past_rows.empty:
                old_val = past_rows.iloc[0]['value']
                if old_val != 0:
                    yoy = ((cur_val - old_val) / old_val) * 100
                    result.append({"date": cur_dt.strftime('%Y-%m-%d'), "value": yoy})
        return result

    def fetch_chart_data(sid, val_type, n_points=300):
        """Fetch historical chart data, converting pct_chg series to YoY %."""
        # For pct_chg series, fetch extra points to compute YoY
        fetch_n = n_points + 15 if val_type == "pct_chg" else n_points
        raw = fetch_fred_history(sid, n_points=fetch_n)
        if not raw:
            return []
        if val_type == "pct_chg":
            return history_to_yoy_pct(raw)
        return raw

    def format_fred_value(val, val_type):
        """Format a FRED value based on its type."""
        if val is None:
            return "N/A"
        if val_type in ("pct", "pct_chg"):
            return f"{val:.2f}%"
        elif val_type == "rate":
            return f"{val:.2f}%"
        elif val_type == "usd_large":
            if abs(val) >= 1e12:
                return f"${val/1e12:.2f}T"
            elif abs(val) >= 1e9:
                return f"${val/1e9:.1f}B"
            elif abs(val) >= 1e6:
                return f"${val/1e6:.1f}M"
            else:
                return f"${val:,.0f}"
        elif val_type == "millions":
            if val >= 1e6:
                return f"{val/1e6:.1f}M"
            else:
                return f"{val:,.0f}"
        elif val_type == "index":
            return f"{val:,.2f}"
        elif val_type == "number":
            return f"{val:,.2f}"
        else:
            return f"{val:,.2f}"

    # FRED search fallback for countries not in the map
    @st.cache_data(ttl=7200)
    def fred_search_series(search_text, limit=20):
        """Search FRED for series matching text, sorted by popularity."""
        import requests as _req
        url = "https://api.stlouisfed.org/fred/series/search"
        params = {
            "search_text": search_text,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "limit": limit,
            "order_by": "popularity",
            "sort_order": "desc",
        }
        try:
            r = _req.get(url, params=params, timeout=10)
            if r.status_code == 200:
                return r.json().get("seriess", [])
        except Exception:
            pass
        return []

    # Country input — single field, supports one or two countries
    available_countries = sorted([c for c in COUNTRY_FRED_MAP.keys() if c not in _ALIAS_KEYS])
    country_help = ", ".join([f"{COUNTRY_FRED_MAP[c]['flag']} {c.title()}" for c in available_countries])

    country_input = st.text_input(
        "🔍 Enter one or two countries (comma-separated for comparison)",
        placeholder="e.g. France  or  France, Germany",
        help=f"Pre-mapped: {country_help}. Type one country for its data, two for a side-by-side comparison.",
        key="country_explorer_input"
    )

    if country_input:
        # Parse input — split by comma, semicolon, or "vs"
        import re as _re
        raw_parts = _re.split(r'[,;]|\bvs\.?\b|\bvs\b', country_input, flags=_re.IGNORECASE)
        parts = [p.strip().lower() for p in raw_parts if p.strip()]

        # Resolve countries
        resolved = []
        for p in parts[:2]:  # max 2
            if p in COUNTRY_FRED_MAP:
                resolved.append(p)
            else:
                # Try partial match
                matches = [c for c in COUNTRY_FRED_MAP.keys() if p in c or c in p]
                if matches:
                    resolved.append(matches[0])

        # Remove duplicates
        if len(resolved) == 2 and resolved[0] == resolved[1]:
            resolved = resolved[:1]

        # ── Helper: fetch country data ──────────────────────────────────
        def load_country_data(ckey):
            """Fetch all FRED data for a pre-mapped country. Returns (cinfo, batch_results, country_results)."""
            cinfo = COUNTRY_FRED_MAP[ckey]
            smap = cinfo["series"]
            sids_with_type = [(s["id"], s["type"]) for s in smap.values()]
            batch = fetch_fred_batch(sids_with_type)
            results = {}
            for label, sinfo in smap.items():
                sid = sinfo["id"]
                val_type = sinfo["type"]
                val, dt = batch.get(sid, (None, None))
                results[label] = (val, dt, sid, val_type)
            return cinfo, batch, results

        if len(resolved) == 0:
            # No recognized country — try FRED search fallback
            country_display = parts[0].title() if parts else country_input.strip().title()
            st.markdown(f"### 🔍 {country_display}")
            st.info(f"*{country_display}* is not in the pre-mapped list. Searching FRED...")

            search_queries = [
                f"{country_display} GDP",
                f"{country_display} CPI inflation",
                f"{country_display} unemployment rate",
                f"{country_display} interest rate",
            ]
            all_found = []
            with st.spinner(f"Searching FRED for {country_display} data..."):
                for sq in search_queries:
                    results = fred_search_series(sq, limit=5)
                    for s in results:
                        sid = s.get("id", "")
                        title = s.get("title", "")
                        freq = s.get("frequency_short", "")
                        if sid and title and sid not in [x["id"] for x in all_found]:
                            all_found.append({"id": sid, "title": title, "frequency": freq})

            if all_found:
                st.markdown(f"Found **{len(all_found)}** series:")
                for item in all_found[:12]:
                    val, dt = fetch_fred_latest(item["id"])
                    val_str = f"**{val:.2f}**" if val is not None else "N/A"
                    dt_str = f" *(as of {dt})*" if dt else ""
                    st.markdown(f"- `{item['id']}` — {item['title']} [{item['frequency']}]: {val_str}{dt_str}")

                found_ids = {f"{item['id']} — {item['title']}": item['id'] for item in all_found[:12]}
                pick = st.selectbox("Select a series to chart", list(found_ids.keys()), key="country_search_chart")
                if pick:
                    chart_data = fetch_fred_history(found_ids[pick], n_points=200)
                    if chart_data:
                        cdf = pd.DataFrame(chart_data)
                        cdf['date'] = pd.to_datetime(cdf['date'])
                        fig = go.Figure(go.Scatter(
                            x=cdf['date'], y=cdf['value'],
                            mode='lines', line=dict(color="#42a5f5", width=2),
                        ))
                        fig.update_layout(
                            title=pick, height=380, template="plotly_dark",
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=50, r=20, t=50, b=40),
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No FRED series found for *{country_display}*. Try a different spelling.")

        elif len(resolved) == 1:
            # ── SINGLE COUNTRY MODE ─────────────────────────────────────
            ckey = resolved[0]
            cinfo, batch_results, country_results = load_country_data(ckey)
            flag = cinfo["flag"]
            cname = get_country_display_name(ckey)
            st.markdown(f"### {flag} {cname}")

            # Metrics grid
            n_items = len(country_results)
            cols_per_row = min(n_items, 4)
            c_cols = st.columns(cols_per_row)
            for idx, (label, (val, dt, sid, val_type)) in enumerate(country_results.items()):
                with c_cols[idx % cols_per_row]:
                    display_val = format_fred_value(val, val_type)
                    dt_str = f"as of {dt}" if dt else "no date"
                    st.metric(label, display_val, help=f"FRED: {sid} — {dt_str}")

            # Chart
            st.markdown(f"**📈 Historical trends for {cname}**")
            chart_options = {}
            chart_types = {}
            for l, (val, dt, sid, vt) in country_results.items():
                if val is not None:
                    chart_options[l] = sid
                    chart_types[l] = vt
            if chart_options:
                selected_chart = st.selectbox("Select indicator to chart", list(chart_options.keys()), key="country_chart_select")
                if selected_chart:
                    chart_data = fetch_chart_data(chart_options[selected_chart], chart_types[selected_chart], n_points=300)
                    if chart_data:
                        cdf = pd.DataFrame(chart_data)
                        cdf['date'] = pd.to_datetime(cdf['date'])
                        fig = go.Figure(go.Scatter(
                            x=cdf['date'], y=cdf['value'],
                            mode='lines', line=dict(color="#42a5f5", width=2),
                            fill='tozeroy', fillcolor="rgba(66,165,245,0.1)",
                        ))
                        fig.update_layout(
                            title=f"{flag} {cname} — {selected_chart}",
                            height=380, template="plotly_dark",
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            xaxis_title="Date", yaxis_title="Value",
                            margin=dict(l=50, r=20, t=50, b=40),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"No historical data available for {selected_chart}")

        else:
            # ── COMPARISON MODE (2 countries) ───────────────────────────
            ckey1, ckey2 = resolved[0], resolved[1]

            with st.spinner("Loading comparison data..."):
                cinfo1, batch1, results1 = load_country_data(ckey1)
                cinfo2, batch2, results2 = load_country_data(ckey2)

            flag1, flag2 = cinfo1["flag"], cinfo2["flag"]
            name1, name2 = get_country_display_name(ckey1), get_country_display_name(ckey2)

            st.markdown(f"### {flag1} {name1}  vs  {flag2} {name2}")

            # Find common indicators
            common_labels = [l for l in results1 if l in results2]

            if common_labels:
                # ── Side-by-side metrics ────────────────────────────────
                for label in common_labels:
                    val1, dt1, sid1, vt1 = results1[label]
                    val2, dt2, sid2, vt2 = results2[label]
                    fmt1 = format_fred_value(val1, vt1)
                    fmt2 = format_fred_value(val2, vt2)

                col_header1, col_header2 = st.columns(2)
                # Render comparison table
                comparison_rows = []
                for label in common_labels:
                    val1, dt1, sid1, vt1 = results1[label]
                    val2, dt2, sid2, vt2 = results2[label]
                    fmt1 = format_fred_value(val1, vt1)
                    fmt2 = format_fred_value(val2, vt2)
                    comparison_rows.append({
                        "Indicator": label,
                        f"{flag1} {name1}": fmt1,
                        f"{flag2} {name2}": fmt2,
                    })

                comp_df = pd.DataFrame(comparison_rows)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

                # ── Comparison chart ────────────────────────────────────
                st.markdown("**📊 Side-by-side chart**")
                chart_common = [l for l in common_labels
                                if results1[l][0] is not None and results2[l][0] is not None]
                if chart_common:
                    compare_indicator = st.selectbox(
                        "Select indicator to compare",
                        chart_common,
                        key="country_compare_chart_select"
                    )
                    if compare_indicator:
                        sid1 = results1[compare_indicator][2]
                        sid2 = results2[compare_indicator][2]
                        vt1 = results1[compare_indicator][3]
                        vt2 = results2[compare_indicator][3]
                        hist1 = fetch_chart_data(sid1, vt1, n_points=200)
                        hist2 = fetch_chart_data(sid2, vt2, n_points=200)

                        fig = go.Figure()
                        if hist1:
                            df1 = pd.DataFrame(hist1)
                            df1['date'] = pd.to_datetime(df1['date'])
                            fig.add_trace(go.Scatter(
                                x=df1['date'], y=df1['value'],
                                mode='lines', name=f"{flag1} {name1}",
                                line=dict(color="#42a5f5", width=2),
                            ))
                        if hist2:
                            df2 = pd.DataFrame(hist2)
                            df2['date'] = pd.to_datetime(df2['date'])
                            fig.add_trace(go.Scatter(
                                x=df2['date'], y=df2['value'],
                                mode='lines', name=f"{flag2} {name2}",
                                line=dict(color="#ef5350", width=2),
                            ))
                        fig.update_layout(
                            title=f"{compare_indicator}: {flag1} {name1} vs {flag2} {name2}",
                            height=400, template="plotly_dark",
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            xaxis_title="Date", yaxis_title="Value",
                            margin=dict(l=50, r=20, t=50, b=40),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No common indicators found between these two countries.")

    st.markdown("---")

    # ── FRED chart explorer ─────────────────────────────────────────────
    with st.expander("📊 FRED Series Explorer — Plot any FRED series"):
        st.markdown("Enter a FRED series ID to chart it (e.g. `DFF`, `CPIAUCSL`, `UNRATE`, `DGS10`, `T10Y2Y`)")
        fred_explore_id = st.text_input("FRED Series ID", value="DFF", key="fred_explorer_input")
        fred_explore_points = st.slider("Data points", 10, 500, 120, key="fred_explore_pts")

        if fred_explore_id:
            explore_data = fetch_fred_history(fred_explore_id.strip().upper(), n_points=fred_explore_points)
            if explore_data:
                explore_df = pd.DataFrame(explore_data)
                explore_df['date'] = pd.to_datetime(explore_df['date'])
                fig = go.Figure(go.Scatter(
                    x=explore_df['date'], y=explore_df['value'],
                    mode='lines', line=dict(color="#42a5f5", width=2),
                ))
                fig.update_layout(
                    title=f"FRED: {fred_explore_id.upper()}",
                    height=400, template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    xaxis_title="Date", yaxis_title="Value",
                    margin=dict(l=50, r=20, t=50, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Showing last {len(explore_data)} observations for **{fred_explore_id.upper()}**")
            else:
                st.warning(f"Could not fetch series `{fred_explore_id}`. Check the series ID at https://fred.stlouisfed.org")

    st.markdown("---")

    # ═════════════════════════════════════════════════════════════════════
    # MARKET NEWS
    # ═════════════════════════════════════════════════════════════════════
    st.subheader("📰 Market News")

    @st.cache_data(ttl=600)
    def fetch_market_news(tickers_for_news, max_articles=20):
        """Fetch news from yfinance for a set of tickers, deduplicated."""
        seen_titles = set()
        articles = []
        for tkr in tickers_for_news:
            try:
                t = yf.Ticker(tkr)
                news = t.news
                if news:
                    for item in news:
                        content = item.get('content', item) if isinstance(item, dict) else item
                        if isinstance(content, dict):
                            title = content.get('title', '')
                        else:
                            title = str(content)
                            content = {'title': title}
                        if title and title not in seen_titles:
                            seen_titles.add(title)
                            pub = content.get('pubDate', content.get('providerPublishTime', ''))
                            link = content.get('url', content.get('link', ''))
                            publisher = content.get('provider', {})
                            if isinstance(publisher, dict):
                                publisher = publisher.get('displayName', '')
                            articles.append({
                                'title': title,
                                'publisher': publisher,
                                'link': link,
                                'date': pub,
                            })
            except Exception:
                continue
            if len(articles) >= max_articles:
                break
        return articles[:max_articles]

    news_tickers = ["^GSPC", "^DJI", "^IXIC", "GC=F", "CL=F", "EURUSD=X", "BTC-USD"]
    news_items = fetch_market_news(news_tickers)

    if news_items:
        for item in news_items:
            title = item['title']
            pub = item.get('publisher', '')
            link = item.get('link', '')
            date_str = item.get('date', '')
            source_line = f"*{pub}*" if pub else ""
            if date_str:
                source_line += f" — {date_str}" if source_line else f"*{date_str}*"
            if link:
                st.markdown(f"🔹 **[{title}]({link})**")
            else:
                st.markdown(f"🔹 **{title}**")
            if source_line:
                st.caption(source_line)
    else:
        st.info("No recent news available. Yahoo Finance news may be temporarily unavailable.")

# =============================================================================
# PAGE 1: PORTFOLIO OVERVIEW
# =============================================================================
if page == "Portfolio Overview":
    st.title("📈 Portfolio Overview")
    st.markdown("*Integrated analytics covering stocks, indices, bonds, and commodities*")
    
    if selected_tickers:
        # Fetch data
        with st.spinner("Fetching market data..."):
            market_data = fetch_data(selected_tickers, start_date, end_date)
        
        if not market_data:
            st.error("❌ Unable to fetch data for the selected tickers. Please try different tickers or date range.")
            st.info("💡 Tip: Make sure you have an internet connection and the ticker symbols are correct.")
        else:
            # =================================================================
            # TOP PERFORMERS SECTION
            # =================================================================
            st.subheader("🏆 Top Performers")
            cols = st.columns(min(4, len(selected_tickers)))
            
            for idx, (ticker, df) in enumerate(market_data.items()):
                with cols[idx % 4]:
                    price, change, vol = calculate_metrics(df)
                    if price is not None:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="margin-bottom:5px;">{ticker}</h4>
                            <h2 style="margin:10px 0;">${price:.2f}</h2>
                            <p style="color: {'#26a69a' if change > 0 else '#ef5350'}; font-weight: bold;">
                                {'▲' if change > 0 else '▼'} {abs(change):.2f}%
                            </p>
                            <small style="color: #888;">Vol: {vol:.1f}%</small>
                        </div>
                        """, unsafe_allow_html=True)
            
            # =================================================================
            # NORMALIZED PERFORMANCE CHART
            # =================================================================
            st.subheader("📊 Normalized Cumulative Performance")
            st.markdown("*Baseline comparison showing relative performance (all assets start at 100)*")
            
            normalized_df = normalize_prices(market_data)
            
            fig = go.Figure()
            colors = px.colors.qualitative.Set3
            
            for idx, ticker in enumerate(normalized_df.columns):
                fig.add_trace(go.Scatter(
                    x=normalized_df.index,
                    y=normalized_df[ticker],
                    mode='lines',
                    name=ticker,
                    line=dict(width=2, color=colors[idx % len(colors)]),
                    hovertemplate='%{y:.2f}%<extra></extra>'
                ))
            
            fig.add_hline(
                y=100, 
                line_dash="dash", 
                line_color="gray", 
                opacity=0.5,
                annotation_text="Baseline (100%)", 
                annotation_position="left"
            )
            
            fig.update_layout(
                template="plotly_dark",
                height=450,
                hovermode='x unified',
                xaxis_title="Date",
                yaxis_title="Normalized Return (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(tickformat=".0f")
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # =================================================================
            # CORRELATION ANALYSIS
            # =================================================================
            st.subheader("📉 Correlation Analysis")
            st.markdown("*Shows how assets move together (1 = perfect correlation, -1 = inverse correlation)*")
            
            # Create correlation matrix
            prices_df = pd.DataFrame()
            for ticker, df in market_data.items():
                prices_df[ticker] = df['Close']
            
            returns_df = prices_df.pct_change().dropna()
            corr_matrix = returns_df.corr()
            
            # Enhanced correlation heatmap
            corr_fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 12},
                colorbar=dict(title="Correlation", thickness=15, len=0.7)
            ))
            
            corr_fig.update_layout(
                template="plotly_dark",
                height=500,
                title="Asset Correlation Matrix",
                xaxis=dict(tickangle=-45)
            )
            
            st.plotly_chart(corr_fig, use_container_width=True)
            
            # =================================================================
            # RISK-RETURN SCATTER PLOT
            # =================================================================
            st.subheader("📊 Risk-Return Profile")
            st.markdown("*Higher and to the left is better (higher return, lower risk)*")
            
            risk_return_data = []
            for ticker in returns_df.columns:
                annual_return = returns_df[ticker].mean() * 252 * 100
                annual_vol = returns_df[ticker].std() * np.sqrt(252) * 100
                risk_return_data.append({
                    'Ticker': ticker,
                    'Return': annual_return,
                    'Risk': annual_vol
                })
            
            rr_df = pd.DataFrame(risk_return_data)
            
            scatter_fig = px.scatter(
                rr_df, 
                x='Risk', 
                y='Return', 
                text='Ticker',
                color='Return', 
                color_continuous_scale='RdYlGn',
                size_max=20
            )
            
            scatter_fig.update_traces(
                textposition='top center', 
                marker=dict(size=15, line=dict(width=1, color='white'))
            )
            
            scatter_fig.update_layout(
                template="plotly_dark",
                height=450,
                title="Risk-Return Scatter Plot",
                xaxis_title="Annualized Volatility (%)",
                yaxis_title="Annualized Return (%)",
                showlegend=False
            )
            
            st.plotly_chart(scatter_fig, use_container_width=True)
            
            # =============================================================================
            # PORTFOLIO BUILDER SECTION
            # =============================================================================
            st.markdown("---")
            st.subheader("🎯 Portfolio Builder")
            st.markdown("*Build and test portfolio allocations quickly with initial prices and target weights*")
            
            # Get initial prices for the period
            initial_prices = {}
            for ticker in selected_tickers:
                if ticker in prices_df.columns and not prices_df[ticker].isna().all():
                    initial_prices[ticker] = prices_df[ticker].iloc[0]
            
            if initial_prices:
                # Create three columns for the builder interface
                col1_builder, col2_builder = st.columns([1, 2])
                
                with col1_builder:
                    st.markdown("#### 📊 Initial Prices")
                    st.markdown(f"*Period Start: {start_date.strftime('%Y-%m-%d')}*")
                    
                    # Display initial prices in a clean table
                    prices_data = []
                    for ticker, price in initial_prices.items():
                        prices_data.append({
                            'Asset': ticker,
                            'Initial Price': f"${price:,.2f}"
                        })
                    prices_table = pd.DataFrame(prices_data)
                    st.dataframe(prices_table, use_container_width=True, hide_index=True)
                    
                    # Portfolio nominal input
                    st.markdown("#### 💰 Portfolio Settings")
                    portfolio_nominal = st.number_input(
                        "Portfolio Nominal ($)",
                        min_value=1000.0,
                        max_value=10000000.0,
                        value=100000.0,
                        step=1000.0,
                        help="Total amount to invest",
                        key="portfolio_nominal_builder"
                    )
                
                with col2_builder:
                    st.markdown("#### ⚖️ Asset Allocation")
                    
                    # Weight input section
                    weights = {}
                    total_weight = 0
                    
                    # Create weight inputs for each asset
                    weight_cols = st.columns(min(3, len(initial_prices)))
                    for idx, ticker in enumerate(initial_prices.keys()):
                        with weight_cols[idx % len(weight_cols)]:
                            default_weight = 100.0 / len(initial_prices)
                            weight = st.number_input(
                                f"{ticker} Weight (%)",
                                min_value=0.0,
                                max_value=100.0,
                                value=default_weight,
                                step=1.0,
                                key=f"weight_builder_{ticker}"
                            )
                            weights[ticker] = weight
                            total_weight += weight
                    
                    # Weight validation
                    weight_diff = abs(total_weight - 100.0)
                    if weight_diff > 0.01:
                        st.warning(f"⚠️ Total weight is {total_weight:.2f}% (should be 100%)")
                    else:
                        st.success(f"✅ Total weight: {total_weight:.2f}%")
                    
                    # Calculate shares to buy
                    st.markdown("#### 📈 Shares to Purchase")
                    
                    allocation_data = []
                    portfolio_holdings = {}
                    
                    for ticker in initial_prices.keys():
                        weight_pct = weights[ticker]
                        allocation_amount = portfolio_nominal * (weight_pct / 100.0)
                        initial_price = initial_prices[ticker]
                        shares_to_buy = allocation_amount / initial_price
                        shares_rounded = int(shares_to_buy)
                        actual_investment = shares_rounded * initial_price
                        actual_weight = (actual_investment / portfolio_nominal) * 100 if portfolio_nominal > 0 else 0
                        
                        allocation_data.append({
                            'Asset': ticker,
                            'Target %': f"{weight_pct:.2f}%",
                            'Amount': f"${allocation_amount:,.2f}",
                            'Shares': shares_rounded,
                            'Actual $': f"${actual_investment:,.2f}",
                            'Actual %': f"{actual_weight:.2f}%"
                        })
                        
                        # Store for portfolio calculation
                        portfolio_holdings[ticker] = shares_rounded
                    
                    allocation_table = pd.DataFrame(allocation_data)
                    st.dataframe(allocation_table, use_container_width=True, hide_index=True)
                    
                    # Calculate leftover cash
                    total_invested = sum(portfolio_holdings[t] * initial_prices[t] for t in portfolio_holdings)
                    leftover_cash = portfolio_nominal - total_invested
                    st.info(f"💵 Leftover Cash: ${leftover_cash:,.2f}")
                
                # Portfolio Performance Chart
                st.markdown("#### 📊 Portfolio Performance Over Time")
                
                # Calculate portfolio value over time
                portfolio_values = pd.Series(0, index=prices_df.index)
                
                for ticker, shares in portfolio_holdings.items():
                    if ticker in prices_df.columns and shares > 0:
                        portfolio_values += prices_df[ticker] * shares
                
                # Add leftover cash to portfolio
                portfolio_values += leftover_cash
                
                # Calculate portfolio returns
                portfolio_returns = (portfolio_values / portfolio_nominal - 1) * 100
                
                # Create performance chart
                fig_portfolio = go.Figure()
                
                # Portfolio value line
                fig_portfolio.add_trace(go.Scatter(
                    x=portfolio_values.index,
                    y=portfolio_values.values,
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#00d4ff', width=3),
                    fill='tonexty',
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Value: $%{y:,.2f}<extra></extra>'
                ))
                
                # Initial investment line
                fig_portfolio.add_trace(go.Scatter(
                    x=portfolio_values.index,
                    y=[portfolio_nominal] * len(portfolio_values),
                    mode='lines',
                    name='Initial Investment',
                    line=dict(color='#ff9500', width=2, dash='dash'),
                    hovertemplate='<b>Initial: $%{y:,.2f}</b><extra></extra>'
                ))
                
                fig_portfolio.update_layout(
                    template="plotly_dark",
                    title="Portfolio Value Evolution",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    hovermode='x unified',
                    height=500,
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor='rgba(0,0,0,0.5)'
                    )
                )
                
                st.plotly_chart(fig_portfolio, use_container_width=True)
                
                # Portfolio Statistics
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                
                final_value = portfolio_values.iloc[-1]
                total_return = (final_value - portfolio_nominal) / portfolio_nominal * 100
                
                # Calculate portfolio metrics
                portfolio_daily_returns = portfolio_values.pct_change().dropna()
                portfolio_volatility = portfolio_daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
                
                # Calculate max drawdown
                cumulative = (1 + portfolio_daily_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_dd = drawdown.min() * 100
                
                # Calculate Sharpe ratio
                excess_returns = portfolio_daily_returns - (DEFAULT_RISK_FREE_RATE / TRADING_DAYS_PER_YEAR)
                sharpe_ratio = (excess_returns.mean() / portfolio_daily_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR) if portfolio_daily_returns.std() > 0 else 0
                
                with col_stat1:
                    st.metric(
                        "Final Value",
                        f"${final_value:,.2f}",
                        f"{total_return:+.2f}%"
                    )
                
                with col_stat2:
                    st.metric(
                        "Total Return",
                        f"{total_return:.2f}%",
                        f"${final_value - portfolio_nominal:+,.2f}"
                    )
                
                with col_stat3:
                    st.metric(
                        "Volatility",
                        f"{portfolio_volatility:.2f}%",
                        f"Sharpe: {sharpe_ratio:.2f}"
                    )
                
                with col_stat4:
                    st.metric(
                        "Max Drawdown",
                        f"{max_dd:.2f}%"
                    )
                
                # Individual asset contribution
                st.markdown("#### 🎯 Asset Contributions to Portfolio")
                
                contribution_data = []
                for ticker, shares in portfolio_holdings.items():
                    if ticker in prices_df.columns and shares > 0:
                        initial_value = shares * initial_prices[ticker]
                        final_price = prices_df[ticker].iloc[-1]
                        final_value_asset = shares * final_price
                        asset_return = (final_value_asset - initial_value) / initial_value * 100 if initial_value > 0 else 0
                        contribution_to_portfolio = (final_value_asset - initial_value)
                        weight_in_portfolio = (final_value_asset / final_value) * 100
                        
                        contribution_data.append({
                            'Asset': ticker,
                            'Shares': shares,
                            'Initial Value': f"${initial_value:,.2f}",
                            'Final Value': f"${final_value_asset:,.2f}",
                            'Return': f"{asset_return:+.2f}%",
                            'Contribution': f"${contribution_to_portfolio:+,.2f}",
                            'Final Weight': f"{weight_in_portfolio:.2f}%"
                        })
                
                contribution_table = pd.DataFrame(contribution_data)
                
                # Style the contribution table
                def color_return(val):
                    """Color code returns."""
                    if isinstance(val, str) and '%' in val:
                        num = float(val.replace('%', '').replace('+', ''))
                        if num > 0:
                            return 'background-color: #1b5e20; color: white'
                        elif num < 0:
                            return 'background-color: #b71c1c; color: white'
                    return ''
                
                styled_contribution = contribution_table.style.map(color_return, subset=['Return'])
                st.dataframe(styled_contribution, use_container_width=True, hide_index=True)
                
                # Export portfolio data button
                st.markdown("#### 💾 Export Portfolio")
                export_portfolio_data = {
                    'Date': portfolio_values.index.strftime('%Y-%m-%d'),
                    'Portfolio Value': portfolio_values.values,
                    'Return (%)': portfolio_returns.values
                }
                
                for ticker, shares in portfolio_holdings.items():
                    if ticker in prices_df.columns and shares > 0:
                        export_portfolio_data[f'{ticker} Value'] = (prices_df[ticker] * shares).values
                
                export_portfolio_df = pd.DataFrame(export_portfolio_data)
                csv_portfolio = export_portfolio_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Portfolio Data (CSV)",
                    data=csv_portfolio,
                    file_name=f"portfolio_builder_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="download_portfolio_builder"
                )
            
            else:
                st.info("📊 Select assets and a date range to use the Portfolio Builder")
            
    else:
        st.info("👈 Please select at least one ticker from the sidebar to get started!")
        st.markdown("""
        ### Quick Start Guide:
        1. **Use preset date buttons** for quick selection (1M, 3M, 6M, 1Y, etc.)
        2. **Select a category** from the sidebar (Stocks, Indices, Bonds, etc.)
        3. **Choose tickers** you want to analyze
        4. View the interactive charts and metrics!
        """)

# =============================================================================
# PAGE 2: PORTFOLIO TRACKER
# =============================================================================
elif page == "Portfolio Tracker":
    st.title("🎯 Portfolio Tracker - Active Portfolio Management")
    st.markdown("*Monitor your portfolio evolution, track weight drift, and receive rebalancing recommendations*")
    
    # =================================================================
    # MULTI-CLIENT SELECTOR
    # =================================================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("👥 Client Management")
    
    # Get list of clients
    existing_clients = get_client_list()
    
    # Always show "Add New Client" first (so you can create first client)
    with st.sidebar.expander("➕ Add New Client"):
        new_client_name = st.text_input("Client Name", key="new_client_name")
        new_client_id = new_client_name.lower().replace(' ', '_').replace('-', '_') if new_client_name else ""
        
        if st.button("Create Client", key="create_client_btn"):
            if new_client_name and new_client_id:
                if create_new_client(new_client_id, new_client_name, 0):
                    st.success(f"✅ Created client: {new_client_name}")
                    st.rerun()
                else:
                    st.error("❌ Failed to create client")
            else:
                st.warning("Please enter a client name")
    
    # Mode selection
    selected_client_id = None
    
    if len(existing_clients) > 0:
        use_multi_client = st.sidebar.checkbox("Multi-Client Mode", value=len(existing_clients) > 0)
        
        if use_multi_client:
            # Client selector
            if existing_clients:
                # Load client names
                client_options = {}
                for client_id in existing_clients:
                    metadata = load_client_metadata(client_id)
                    client_name = metadata.get('client_name', client_id)
                    client_options[f"{client_name} ({client_id})"] = client_id
                
                selected_display = st.sidebar.selectbox(
                    "Select Client",
                    options=list(client_options.keys())
                )
                selected_client_id = client_options[selected_display]
                
                # Load client info
                client_meta = load_client_metadata(selected_client_id)
                st.sidebar.success(f"📂 Active: {client_meta.get('client_name', 'Unknown')}")
    else:
        use_multi_client = False
        st.sidebar.info("💡 Create your first client above")
    
    # =================================================================
    # INITIALIZE SESSION STATE FOR TRANSACTIONS
    # =================================================================
    if 'transactions' not in st.session_state:
        # Load transactions from file if they exist
        st.session_state.transactions = load_transactions_from_file(selected_client_id)
    if 'portfolio_events' not in st.session_state:
        st.session_state.portfolio_events = []
    if 'config_loaded' not in st.session_state:
        st.session_state.config_loaded = False
        st.session_state.saved_config = load_config_from_file(selected_client_id)
    if 'current_client_id' not in st.session_state or st.session_state.get('current_client_id') != selected_client_id:
        # Client changed, reload ALL client-specific data
        st.session_state.current_client_id = selected_client_id
        st.session_state.transactions = load_transactions_from_file(selected_client_id)
        st.session_state.saved_config = load_config_from_file(selected_client_id)
        
        # Reload target portfolio for this client (reset to empty if not found)
        loaded_portfolio = load_target_portfolio_from_file(selected_client_id)
        st.session_state.target_portfolio = loaded_portfolio if loaded_portfolio else {}
        
        # Reload benchmark for this client (reset to defaults if not found)
        saved_components, saved_weights = load_benchmark_from_file(selected_client_id)
        if saved_components and saved_weights:
            st.session_state.benchmark_components = saved_components
            st.session_state.benchmark_weights = saved_weights
            st.session_state.custom_benchmark = create_custom_benchmark(saved_components, saved_weights)
        else:
            st.session_state.benchmark_components = ['SPY', 'AGG']
            st.session_state.benchmark_weights = [0.6, 0.4]
            st.session_state.custom_benchmark = None
        
        # Reset analysis view when switching clients
        st.session_state.show_portfolio_analysis = False
    
    # =================================================================
    # DATA PERSISTENCE INFO
    # =================================================================
    if st.session_state.transactions:
        st.success(f"💾 Data Auto-Saved: {len(st.session_state.transactions)} transactions loaded from previous session")
    
    # =================================================================
    # PORTFOLIO CONFIGURATION
    # =================================================================
    st.subheader("⚙️ Portfolio Configuration")
    
    # Load saved config defaults
    saved_config = st.session_state.saved_config
    default_investment_date = saved_config.get('investment_date', datetime.now() - timedelta(days=180))
    default_threshold = saved_config.get('rebalance_threshold', 5.0)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        investment_date = st.date_input(
            "Portfolio Start Date",
            value=default_investment_date,
            max_value=datetime.now(),
            help="Start date for fetching price data (usually your first transaction date)"
        )
    
    with col2:
        rebalance_threshold = st.slider(
            "Rebalancing Alert Threshold (%)",
            min_value=1.0,
            max_value=20.0,
            value=default_threshold,
            step=0.5,
            help="Alert when any asset weight drifts beyond this percentage from target"
        )
        
        show_recommendations = st.checkbox(
            "Show Rebalancing Recommendations",
            value=True,
            help="Display specific buy/sell recommendations"
        )
    
    # Save configuration (no longer saving initial_capital - it comes from transactions)
    current_config = {
        'investment_date': investment_date,
        'rebalance_threshold': rebalance_threshold
    }
    if current_config != st.session_state.saved_config:
        save_config_to_file(current_config, selected_client_id)
        st.session_state.saved_config = current_config
    
    # =================================================================
    # TRANSACTION MANAGEMENT SECTION
    # =================================================================
    st.subheader("📝 Transaction Management")
    st.markdown("*Record buys, sells, deposits, withdrawals, and put options to track your actual portfolio activity*")
    
    with st.expander("➕ Add New Transaction", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            transaction_type = st.selectbox(
                "Transaction Type",
                ["BUY", "SELL", "DEPOSIT", "WITHDRAWAL", "DIVIDEND", "REBALANCE", "BUY_PUT", "SELL_PUT", "CLOSE_PUT"],
                help="Type of transaction - use CLOSE_PUT to sell a long put you previously bought"
            )
        
        # Only show transaction date for non-CLOSE_PUT transactions (CLOSE_PUT uses closing date)
        if transaction_type != "CLOSE_PUT":
            with col2:
                transaction_date = st.date_input(
                    "Transaction Date",
                    value=datetime.now(),
                    min_value=investment_date,
                    max_value=datetime.now(),
                    help="Date when transaction occurred"
                )
        else:
            # For CLOSE_PUT, we'll set transaction_date later from close_date_input
            transaction_date = None
        
        # Different inputs based on transaction type
        if transaction_type == "CLOSE_PUT":
            # Show open put positions to close
            st.markdown("#### 📉 Close Long Put Position")
            st.info("Select an open put position and closing date - system will fetch underlying price automatically")
            
            # Get open put positions
            open_puts = []
            for trans in st.session_state.transactions:
                if trans['type'] == 'BUY_PUT':
                    # Check if this put has been closed
                    is_closed = False
                    for close_trans in st.session_state.transactions:
                        if (close_trans['type'] == 'CLOSE_PUT' and 
                            close_trans.get('closes_transaction_id') == trans.get('timestamp')):
                            is_closed = True
                            break
                    
                    if not is_closed:
                        put_label = f"{trans['ticker']} ${trans.get('strike', 0):.2f} Put - {trans['shares']:.0f} contracts - Bought {trans['date'].strftime('%Y-%m-%d')}"
                        open_puts.append({
                            'label': put_label,
                            'transaction': trans
                        })
            
            if open_puts:
                put_labels = [p['label'] for p in open_puts]
                selected_put_idx = st.selectbox(
                    "Select Put Position to Close",
                    range(len(put_labels)),
                    format_func=lambda x: put_labels[x]
                )
                
                selected_put = open_puts[selected_put_idx]['transaction']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"**Original Position:**\n\n"
                           f"- Underlying: {selected_put['ticker']}\n"
                           f"- Strike: ${selected_put.get('strike', 0):.2f}\n"
                           f"- Contracts: {selected_put['shares']:.0f}\n"
                           f"- Buy Date: {selected_put['date'].strftime('%Y-%m-%d')}\n"
                           f"- Original Premium: ${selected_put['price']:.2f}\n"
                           f"- Original Cost: ${selected_put['amount']:,.2f}")
                
                with col2:
                    # Let user pick the closing date - we'll fetch the price
                    st.markdown("**Closing Details:**")
                    close_date_input = st.date_input(
                        "Closing Date",
                        value=datetime.now(),
                        min_value=selected_put['date'],
                        max_value=datetime.now(),
                        help="Date when you're closing the position - we'll fetch the underlying price"
                    )
                
                # Fetch the underlying price at the closing date
                try:
                    underlying_ticker = selected_put['ticker']
                    
                    # Fetch price data for the underlying
                    with st.spinner(f"Fetching {underlying_ticker} price at {close_date_input}..."):
                        close_date_dt = datetime.combine(close_date_input, datetime.min.time())
                        end_date = close_date_dt + timedelta(days=5)  # Get a few days to ensure data
                        
                        price_data = yf.download(
                            underlying_ticker,
                            start=close_date_dt,
                            end=end_date,
                            progress=False
                        )
                        
                        if not price_data.empty:
                            # Get the close price on or closest to the date
                            underlying_price = float(price_data['Close'].iloc[0])
                            
                            # Calculate intrinsic value for PUT (strike - underlying)
                            strike_price = selected_put.get('strike', 0)
                            intrinsic_value = max(0, strike_price - underlying_price)
                            
                            # Display the fetched data
                            st.success(f"✅ **{underlying_ticker} Price on {close_date_input}:** ${underlying_price:.2f}")
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Strike Price", f"${strike_price:.2f}")
                                st.metric("Underlying Price", f"${underlying_price:.2f}")
                            with col_b:
                                st.metric("Intrinsic Value", f"${intrinsic_value:.2f}")
                                # For puts: ITM when underlying < strike
                                in_the_money = "✅ IN THE MONEY" if intrinsic_value > 0 else "❌ OUT OF THE MONEY"
                                st.info(in_the_money)
                            
                            # Calculate proceeds
                            transaction_ticker = selected_put['ticker']
                            transaction_shares = selected_put['shares']
                            transaction_price = intrinsic_value  # Intrinsic value per contract
                            put_strike_store = strike_price
                            put_multiplier_store = selected_put.get('multiplier', 100)
                            transaction_amount = transaction_shares * intrinsic_value * put_multiplier_store
                            
                            # Show profit/loss
                            original_cost = selected_put['amount']
                            profit = transaction_amount - original_cost
                            profit_pct = (profit / original_cost) * 100 if original_cost > 0 else 0
                            
                            st.markdown("---")
                            st.markdown("### 💰 Closing Summary")
                            
                            col_x, col_y, col_z = st.columns(3)
                            with col_x:
                                st.metric("Proceeds", f"${transaction_amount:,.2f}")
                            with col_y:
                                st.metric("Profit/Loss", f"${profit:,.2f}", delta=f"{profit_pct:+.1f}%")
                            with col_z:
                                if profit >= 0:
                                    st.success(f"🎉 **GAIN**")
                                else:
                                    st.error(f"📉 **LOSS**")
                            
                            # Store the ID of the transaction being closed
                            closes_transaction_id = selected_put.get('timestamp')
                            
                            # Use closing date as transaction date for CLOSE_PUT
                            transaction_date = close_date_input
                            
                        else:
                            st.error(f"❌ Could not fetch {underlying_ticker} price for {close_date_input}")
                            st.warning("⚠️ Please check the date or try a different date")
                            transaction_ticker = "NONE"
                            transaction_shares = 0
                            transaction_price = 0
                            transaction_amount = 0
                            put_strike_store = None
                            put_multiplier_store = None
                            closes_transaction_id = None
                            transaction_date = close_date_input
                            
                except Exception as e:
                    st.error(f"❌ Error fetching price data: {str(e)}")
                    st.info("💡 Make sure the underlying ticker is valid (e.g., SPY, QQQ)")
                    transaction_ticker = "NONE"
                    transaction_shares = 0
                    transaction_price = 0
                    transaction_amount = 0
                    put_strike_store = None
                    put_multiplier_store = None
                    closes_transaction_id = None
                    transaction_date = close_date_input if 'close_date_input' in locals() else datetime.now().date()
                    closes_transaction_id = None
                
            else:
                st.warning("⚠️ No open put positions to close. Buy a put first with BUY_PUT.")
                transaction_ticker = "NONE"
                transaction_shares = 0
                transaction_price = 0
                transaction_amount = 0
                put_strike_store = None
                put_multiplier_store = None
                closes_transaction_id = None
                transaction_date = datetime.now().date()
        
        elif transaction_type in ["BUY_PUT", "SELL_PUT"]:
            # Put Option specific inputs
            st.markdown("#### 📉 Put Option Details")
            st.info("💡 **Out-of-the-money puts** profit when the underlying drops below the strike price. Great for hedging downside risk!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                put_underlying = st.text_input(
                    "Underlying Ticker",
                    value="SPY",
                    help="The asset you want downside protection on (e.g., SPY, QQQ)"
                )
            
            with col2:
                put_strike = st.number_input(
                    "Strike Price ($)",
                    min_value=0.01,
                    value=400.0,
                    step=1.0,
                    format="%.2f",
                    help="Strike price - put is OTM when strike < current price"
                )
            
            with col3:
                put_contracts = st.number_input(
                    "Number of Contracts",
                    min_value=0.01,
                    value=10.0,
                    step=1.0,
                    format="%.2f",
                    help="Number of option contracts"
                )
            
            col1, col2 = st.columns(2)
            
            with col1:
                put_premium = st.number_input(
                    "Premium per Contract ($)",
                    min_value=0.01,
                    value=5.0,
                    step=0.10,
                    format="%.2f",
                    help="Price paid/received per contract"
                )
            
            with col2:
                contract_multiplier = st.number_input(
                    "Contract Multiplier",
                    min_value=1,
                    value=100,
                    step=1,
                    help="Standard is 100 for most options"
                )
            
            transaction_ticker = put_underlying.upper()
            transaction_shares = put_contracts
            transaction_price = put_premium
            transaction_amount = put_contracts * put_premium * contract_multiplier
            
            if transaction_type == "BUY_PUT":
                st.info(f"💰 Total Cost: ${transaction_amount:,.2f} (reduces capital available for other investments)")
                st.markdown("📊 **Payoff Profile**: This put will be profitable if the underlying drops below ${:.2f} minus the premium paid.".format(put_strike))
            else:
                st.success(f"💰 Total Proceeds: ${transaction_amount:,.2f} (adds to available capital)")
            
            # Store additional put option data
            put_strike_store = put_strike
            put_multiplier_store = contract_multiplier
            closes_transaction_id = None
            
        elif transaction_type in ["BUY", "SELL"]:
            col1, col2 = st.columns(2)
            
            with col1:
                transaction_ticker = st.text_input(
                    "Ticker",
                    value="SPY",
                    help="Asset ticker symbol"
                )
            
            with col2:
                transaction_shares = st.number_input(
                    "Shares",
                    min_value=0.01,
                    value=10.0,
                    step=0.01,
                    format="%.2f",
                    help="Number of shares"
                )
            
            transaction_price = st.number_input(
                "Price per Share ($)",
                min_value=0.01,
                value=100.0,
                step=0.01,
                format="%.2f",
                help="Execution price"
            )
            
            transaction_amount = transaction_shares * transaction_price
            put_strike_store = None
            put_multiplier_store = None
            closes_transaction_id = None
            
        else:
            transaction_ticker = "CASH"
            transaction_amount = st.number_input(
                "Amount ($)",
                min_value=0.01,
                value=1000.0,
                step=100.0,
                format="%.2f",
                help="Transaction amount"
            )
            transaction_shares = 0
            transaction_price = 0
            put_strike_store = None
            put_multiplier_store = None
            closes_transaction_id = None
        
        transaction_notes = st.text_area(
            "Notes (Optional)",
            value="",
            help="Additional notes about this transaction"
        )
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("✅ Add Transaction", type="primary", use_container_width=True):
                # Validate CLOSE_PUT transactions - only check ticker is not NONE
                if transaction_type == "CLOSE_PUT" and transaction_ticker == "NONE":
                    st.error("❌ Cannot add CLOSE_PUT: Please select a valid put position and ensure price was fetched")
                else:
                    # Calculate correct amount based on transaction type
                    if transaction_type in ["BUY_PUT", "SELL_PUT", "CLOSE_PUT"]:
                        final_amount = transaction_amount
                    elif transaction_type in ["DEPOSIT", "WITHDRAWAL", "DIVIDEND"]:
                        final_amount = transaction_amount
                    else:  # BUY, SELL
                        final_amount = transaction_shares * transaction_price
                    
                    new_transaction = {
                        'date': transaction_date,
                        'type': transaction_type,
                        'ticker': transaction_ticker.upper(),
                        'shares': transaction_shares,
                        'price': transaction_price,
                        'amount': final_amount,
                        'notes': transaction_notes,
                        'timestamp': datetime.now(),
                        'strike': put_strike_store if transaction_type in ["BUY_PUT", "SELL_PUT", "CLOSE_PUT"] else None,
                        'multiplier': put_multiplier_store if transaction_type in ["BUY_PUT", "SELL_PUT", "CLOSE_PUT"] else None,
                        'closes_transaction_id': closes_transaction_id if transaction_type == "CLOSE_PUT" else None
                    }
                    st.session_state.transactions.append(new_transaction)
                    
                    # Save transactions to file
                    save_transactions_to_file(st.session_state.transactions, selected_client_id)
                    
                    st.success(f"✅ Transaction added: {transaction_type} {transaction_ticker if transaction_ticker != 'CASH' else ''}")
                    st.rerun()
        
        with col2:
            if st.button("🗑️ Delete Last", use_container_width=True):
                if st.session_state.transactions:
                    deleted = st.session_state.transactions.pop()
                    save_transactions_to_file(st.session_state.transactions, selected_client_id)
                    st.success(f"✅ Deleted: {deleted['type']} {deleted['ticker']}")
                    st.rerun()
                else:
                    st.warning("No transactions to delete")
        
        with col3:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.transactions = []
                save_transactions_to_file([], selected_client_id)
                st.success("All transactions cleared!")
                st.rerun()
        
        with col4:
            if st.button("💾 Export Backup", use_container_width=True):
                # Create backup with timestamp
                backup_data = {
                    'transactions': st.session_state.transactions,
                    'config': current_config,
                    'backup_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                if selected_client_id:
                    st.info(f"💾 Data saved to: {CLIENTS_DIR / selected_client_id}")
                else:
                    st.info(f"💾 Data saved to: {DATA_DIR}")
    
    # Display transaction history
    if st.session_state.transactions:
        st.markdown("#### 📋 Transaction History")
        
        trans_display = []
        for i, trans in enumerate(st.session_state.transactions):
            ticker_display = trans['ticker']
            if trans['type'] in ['BUY_PUT', 'SELL_PUT', 'CLOSE_PUT'] and trans.get('strike'):
                ticker_display = f"{trans['ticker']} ${trans['strike']:.2f} Put"
            
            trans_display.append({
                'ID': i + 1,
                'Date': trans['date'].strftime('%Y-%m-%d'),
                'Type': trans['type'],
                'Ticker': ticker_display,
                'Shares/Contracts': f"{trans['shares']:.2f}" if trans['shares'] > 0 else "-",
                'Price': f"${trans['price']:.2f}" if trans['price'] > 0 else "-",
                'Amount': f"${trans['amount']:,.2f}",
                'Notes': trans['notes'][:30] + "..." if len(trans['notes']) > 30 else trans['notes']
            })
        
        trans_df = pd.DataFrame(trans_display)
        
        # Color code by transaction type with lighter,more visible colors
        def color_transaction_type(val):
            if val == 'BUY':
                return 'background-color: #a5d6a7; color: #1b5e20; font-weight: bold'
            elif val == 'SELL':
                return 'background-color: #ef9a9a; color: #b71c1c; font-weight: bold'
            elif val == 'DEPOSIT':
                return 'background-color: #90caf9; color: #0d47a1; font-weight: bold'
            elif val == 'WITHDRAWAL':
                return 'background-color: #ffcc80; color: #e65100; font-weight: bold'
            elif val == 'DIVIDEND':
                return 'background-color: #c5e1a5; color: #33691e; font-weight: bold'
            elif val == 'BUY_PUT':
                return 'background-color: #e1bee7; color: #6a1b9a; font-weight: bold'
            elif val == 'CLOSE_PUT':
                return 'background-color: #80deea; color: #006064; font-weight: bold'
            elif val == 'SELL_PUT':
                return 'background-color: #ffab91; color: #bf360c; font-weight: bold'
            else:
                return 'background-color: #ce93d8; color: #4a148c; font-weight: bold'
        
        styled_trans = trans_df.style.map(color_transaction_type, subset=['Type'])
        st.dataframe(styled_trans, use_container_width=True, hide_index=True)
        
        # Transaction summary
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_deposits = sum(t['amount'] for t in st.session_state.transactions if t['type'] == 'DEPOSIT')
        total_withdrawals = sum(t['amount'] for t in st.session_state.transactions if t['type'] == 'WITHDRAWAL')
        total_buys = sum(t['amount'] for t in st.session_state.transactions if t['type'] == 'BUY')
        total_sells = sum(t['amount'] for t in st.session_state.transactions if t['type'] == 'SELL')
        total_put_cost = sum(t['amount'] for t in st.session_state.transactions if t['type'] == 'BUY_PUT')
        total_put_proceeds = sum(t['amount'] for t in st.session_state.transactions if t['type'] in ['SELL_PUT', 'CLOSE_PUT'])
        
        with col1:
            st.metric("Total Deposits", f"${total_deposits:,.0f}")
        with col2:
            st.metric("Total Withdrawals", f"${total_withdrawals:,.0f}")
        with col3:
            st.metric("Total Buys", f"${total_buys:,.0f}")
        with col4:
            st.metric("Total Sells", f"${total_sells:,.0f}")
        with col5:
            put_net = total_put_proceeds - total_put_cost
            st.metric("Put Option P&L", f"${put_net:,.0f}", 
                     delta=f"Cost: ${total_put_cost:,.0f}" if total_put_cost > 0 else None)
        
        # Download transactions
        trans_csv = trans_df.to_csv(index=False)
        st.download_button(
            label="📥 Export Transactions (CSV)",
            data=trans_csv,
            file_name=f"transactions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # =================================================================
    # TARGET PORTFOLIO SETUP - PERSISTENT (saved to file like transactions)
    # =================================================================
    st.subheader("🎯 Target Portfolio Allocation")
    st.markdown("*Define your desired portfolio composition - saved to file automatically*")
    
    # Initialize target portfolio in session state if not exists - load from dedicated file
    if 'target_portfolio' not in st.session_state:
        # Try to load from dedicated target portfolio file first
        loaded_portfolio = load_target_portfolio_from_file(selected_client_id)
        if loaded_portfolio:
            st.session_state.target_portfolio = loaded_portfolio
        else:
            # Fallback to config file for backwards compatibility
            if 'target_portfolio' in st.session_state.saved_config:
                st.session_state.target_portfolio = st.session_state.saved_config['target_portfolio']
                # Migrate to new file format
                save_target_portfolio_to_file(st.session_state.target_portfolio, selected_client_id)
            else:
                st.session_state.target_portfolio = {}
    
    # Note: Client change handling is now done in the central client switching logic above
    
    # Check if target portfolio is already saved
    has_saved_portfolio = len(st.session_state.target_portfolio) > 0
    
    # Show saved status
    if has_saved_portfolio:
        st.success(f"💾 Target Portfolio Loaded: {len(st.session_state.target_portfolio)} assets from saved file")
    
    # UI to modify target portfolio
    with st.expander("📝 Edit Target Portfolio Allocation", expanded=not has_saved_portfolio):
        num_holdings = st.number_input(
            "Number of Holdings",
            min_value=1,
            max_value=15,
            value=max(4, len(st.session_state.target_portfolio)) if has_saved_portfolio else 4,
            help="Number of assets in your portfolio"
        )
        
        temp_target_portfolio = {}
        holdings_list = []
        
        # Create input grid
        cols_per_row = 3
        for i in range(int(num_holdings)):
            if i % cols_per_row == 0:
                cols = st.columns(cols_per_row)
            
            # Get existing value if available
            existing_tickers = list(st.session_state.target_portfolio.keys())
            default_tickers = ["SPY", "TLT", "GLD", "QQQ", "IWM", "VTI", "BND", "VEA"]
            
            if i < len(existing_tickers):
                default_ticker = existing_tickers[i]
                default_weight = st.session_state.target_portfolio[default_ticker] * 100
            else:
                default_ticker = default_tickers[i % len(default_tickers)]
                default_weight = float(100 // num_holdings)
            
            with cols[i % cols_per_row]:
                with st.container():
                    ticker = st.text_input(
                        f"Asset {i+1}",
                        value=default_ticker,
                        key=f"track_ticker_{i}",
                        help="Ticker symbol"
                    )
                    
                    weight = st.number_input(
                        f"Target Weight (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=default_weight,
                        step=1.0,
                        key=f"track_weight_{i}",
                        format="%.1f"
                    )
                    
                    if ticker:
                        temp_target_portfolio[ticker.upper()] = weight / 100
                        holdings_list.append(ticker.upper())
        
        # Validate weights
        total_target_weight = sum([w * 100 for w in temp_target_portfolio.values()])
        
        if abs(total_target_weight - 100) < 0.01:
            st.success(f"✅ Total Target Weight: {total_target_weight:.1f}%")
            
            # Save button
            if st.button("💾 Save Target Allocation", type="primary"):
                st.session_state.target_portfolio = temp_target_portfolio
                
                # Save to dedicated file (same pattern as transactions)
                if save_target_portfolio_to_file(temp_target_portfolio, selected_client_id):
                    st.success("✅ Target portfolio saved to file! It will persist when you close and reopen.")
                    st.rerun()
                else:
                    st.error("❌ Failed to save target portfolio")
        else:
            st.error(f"❌ Total Target Weight: {total_target_weight:.1f}% (must equal 100%)")
    
    # Use saved target portfolio for analysis
    target_portfolio = st.session_state.target_portfolio
    
    # Show current target portfolio summary
    if target_portfolio:
        st.markdown("**Current Target Allocation:** *(Saved - will persist)*")
        cols = st.columns(min(6, len(target_portfolio)))
        for idx, (ticker, weight) in enumerate(target_portfolio.items()):
            with cols[idx % len(cols)]:
                st.metric(ticker, f"{weight*100:.1f}%")
    else:
        st.warning("⚠️ Please set up your target portfolio allocation above")
        st.stop()
    
    # =================================================================
    # REBALANCING CONFIGURATION
    # =================================================================
    st.markdown("---")
    st.subheader("⚖️ Rebalancing Strategy (Optional)")
    st.markdown("*Enable to simulate periodic rebalancing and compare with buy-and-hold*")
    
    # Initialize rebalancing settings in session state
    if 'rebalancing_enabled' not in st.session_state:
        st.session_state.rebalancing_enabled = False
    if 'rebalancing_threshold' not in st.session_state:
        st.session_state.rebalancing_threshold = 5.0
    if 'rebalancing_period_months' not in st.session_state:
        st.session_state.rebalancing_period_months = 4
    
    col_reb1, col_reb2, col_reb3 = st.columns([1, 2, 2])
    
    with col_reb1:
        rebalancing_enabled = st.checkbox(
            "Enable Rebalancing",
            value=st.session_state.rebalancing_enabled,
            help="Enable to see how periodic rebalancing would have affected your portfolio"
        )
        st.session_state.rebalancing_enabled = rebalancing_enabled
    
    with col_reb2:
        rebalancing_threshold = st.number_input(
            "Rebalancing Threshold (%)",
            min_value=1.0,
            max_value=50.0,
            value=st.session_state.rebalancing_threshold,
            step=0.5,
            disabled=not rebalancing_enabled,
            help="Rebalance when any asset deviates from target by this percentage"
        )
        st.session_state.rebalancing_threshold = rebalancing_threshold
    
    with col_reb3:
        rebalancing_period = st.selectbox(
            "Rebalancing Period",
            options=["1 Month", "2 Months", "3 Months", "4 Months", "6 Months", "1 Year"],
            index=3,  # Default to 4 months
            disabled=not rebalancing_enabled,
            help="How often to check and rebalance the portfolio"
        )
        
        period_months_map = {
            "1 Month": 1, "2 Months": 2, "3 Months": 3,
            "4 Months": 4, "6 Months": 6, "1 Year": 12
        }
        st.session_state.rebalancing_period_months = period_months_map[rebalancing_period]
    
    if rebalancing_enabled:
        st.info(f"📊 **Rebalancing Active**: Portfolio will be rebalanced every {rebalancing_period} when drift exceeds {rebalancing_threshold:.1f}%")
    else:
        st.info("📊 **Buy-and-Hold Mode**: Portfolio weights will drift naturally with market movements")
    
    st.markdown("---")
    
    # =================================================================
    # CUSTOM BENCHMARK CONFIGURATION
    # =================================================================
    st.subheader("🎯 Create Custom Benchmark")
    st.markdown("*Build a composite benchmark to compare your portfolio performance*")
    
    # Initialize benchmark in session state (only on first load - client changes handled above)
    if 'custom_benchmark' not in st.session_state:
        st.session_state.custom_benchmark = None
    if 'benchmark_components' not in st.session_state:
        # Try to load saved benchmark from file (first time initialization)
        saved_components, saved_weights = load_benchmark_from_file(selected_client_id)
        if saved_components and saved_weights:
            st.session_state.benchmark_components = saved_components
            st.session_state.benchmark_weights = saved_weights
            st.session_state.custom_benchmark = create_custom_benchmark(saved_components, saved_weights)
        else:
            st.session_state.benchmark_components = ['SPY', 'AGG']
            st.session_state.benchmark_weights = [0.6, 0.4]
    if 'benchmark_weights' not in st.session_state:
        st.session_state.benchmark_weights = [0.6, 0.4]
    
    # Note: Client change handling is now done in the central client switching logic above
    
    num_components = st.number_input(
        "Number of Benchmark Components",
        min_value=1,
        max_value=5,
        value=len(st.session_state.benchmark_components),
        help="How many indices in your benchmark"
    )
    
    bench_components = []
    bench_weights = []
    
    cols_bench = st.columns(int(num_components))
    for i in range(int(num_components)):
        with cols_bench[i]:
            default_component = st.session_state.benchmark_components[i] if i < len(st.session_state.benchmark_components) else ["SPY", "AGG", "GLD", "VEA", "VWO"][i % 5]
            default_weight = st.session_state.benchmark_weights[i] * 100 if i < len(st.session_state.benchmark_weights) else (100 // num_components)
            
            component = st.text_input(
                f"Component {i+1}",
                value=default_component,
                key=f"bench_comp_{i}",
                help="Ticker symbol"
            )
            bench_components.append(component.upper())
            
            weight = st.number_input(
                f"Weight {i+1} (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(default_weight),
                step=5.0,
                key=f"bench_weight_{i}",
                format="%.1f"
            )
            bench_weights.append(weight / 100)
    
    total_bench_weight = sum(bench_weights)
    
    if abs(total_bench_weight - 1.0) < 0.01:
        st.success(f"✅ Total Benchmark Weight: {total_bench_weight*100:.0f}%")
        
        if st.button("💾 Save Benchmark Configuration", type="primary"):
            st.session_state.benchmark_components = bench_components
            st.session_state.benchmark_weights = bench_weights
            st.session_state.custom_benchmark = create_custom_benchmark(bench_components, bench_weights)
            
            # Save to file persistently
            if save_benchmark_to_file(bench_components, bench_weights, selected_client_id):
                st.success(f"✅ Benchmark saved: {' + '.join([f'{w*100:.0f}% {c}' for c, w in zip(bench_components, bench_weights)])}")
                st.success("✅ Benchmark saved to file! It will persist when you close and reopen.")
                st.rerun()
            else:
                st.error("❌ Failed to save benchmark configuration")
    else:
        st.error(f"❌ Total Benchmark Weight: {total_bench_weight*100:.0f}% (must equal 100%)")
    
    # Show current benchmark
    if st.session_state.custom_benchmark:
        st.markdown("**Current Benchmark:** *(Saved - will be used for comparisons)*")
        bench_summary = " + ".join([f"{w*100:.0f}% {c}" for c, w in zip(st.session_state.benchmark_components, st.session_state.benchmark_weights)])
        st.info(f"📊 {bench_summary}")
    
    st.markdown("---")
    
    # =================================================================
    # ANALYZE BUTTON - Simplified to work like Refresh
    # =================================================================
    # Initialize session state for tracking view
    if 'show_portfolio_analysis' not in st.session_state:
        st.session_state.show_portfolio_analysis = False
    
    # Single button that works perfectly
    if not st.session_state.show_portfolio_analysis:
        # First time - show button to start tracking
        if st.button("📊 Track Portfolio Evolution", type="primary", use_container_width=True):
            st.session_state.show_portfolio_analysis = True
            # Don't rerun here - let the code continue to the analysis section below
    
    # Once tracking is active, show the analysis section
    if st.session_state.show_portfolio_analysis:
        # Show control buttons
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.success("✅ Portfolio Analysis Active - Adjust parameters below")
        with col2:
            if st.button("🔄 Refresh Analysis", use_container_width=True):
                pass  # Just having the button causes a rerun - that's all we need!
        with col3:
            if st.button("✖️ Hide Analysis", use_container_width=True):
                st.session_state.show_portfolio_analysis = False
                st.rerun()
        
        # Now execute the analysis
        
        if not target_portfolio or abs(sum(target_portfolio.values()) - 1.0) >= 0.01:
            st.error("⚠️ Please ensure portfolio weights sum to 100%")
        else:
            # Fetch data from investment date to now
            # For transaction-based tracking, use first transaction date as start
            if st.session_state.transactions:
                first_transaction_date = min(t['date'] for t in st.session_state.transactions)
                investment_datetime = datetime.combine(first_transaction_date, datetime.min.time())
            else:
                investment_datetime = datetime.combine(investment_date, datetime.min.time())
            
            # Collect all tickers including put option underlyings
            all_tickers = list(target_portfolio.keys())
            put_underlyings = set()
            for trans in st.session_state.transactions:
                if trans['type'] in ['BUY_PUT', 'SELL_PUT']:
                    put_underlyings.add(trans['ticker'])
            
            all_tickers.extend(list(put_underlyings))
            all_tickers = list(set(all_tickers))  # Remove duplicates
            
            with st.spinner("Fetching portfolio data..."):
                portfolio_data = fetch_data(
                    all_tickers,
                    investment_datetime,
                    datetime.now(),
                    _cache_key=len(st.session_state.transactions)  # Cache busts when transactions change
                )
            
            if not portfolio_data:
                st.error("❌ Unable to fetch data for the selected assets.")
            else:
                # =============================================================
                # BUILD PRICE DATAFRAME
                # =============================================================
                prices_df = pd.DataFrame()
                for ticker in all_tickers:
                    if ticker in portfolio_data:
                        df = portfolio_data[ticker]
                        
                        # Extract Close price - handle various yfinance return formats
                        try:
                            if isinstance(df.columns, pd.MultiIndex):
                                # New yfinance format: MultiIndex with (Price, Ticker)
                                # Try to access ('Close', ticker) first
                                if ('Close', ticker) in df.columns:
                                    prices_df[ticker] = df[('Close', ticker)]
                                elif 'Close' in df.columns.get_level_values(0):
                                    # Get the Close level and extract the first (or only) column
                                    close_df = df['Close']
                                    if isinstance(close_df, pd.DataFrame):
                                        # If multiple tickers, get the right one or first one
                                        if ticker in close_df.columns:
                                            prices_df[ticker] = close_df[ticker]
                                        else:
                                            prices_df[ticker] = close_df.iloc[:, 0]
                                    else:
                                        prices_df[ticker] = close_df
                            elif 'Close' in df.columns:
                                # Old yfinance format or flattened columns
                                close_data = df['Close']
                                if isinstance(close_data, pd.DataFrame):
                                    prices_df[ticker] = close_data.iloc[:, 0]
                                else:
                                    prices_df[ticker] = close_data
                            else:
                                st.warning(f"⚠️ Could not find Close price for {ticker}")
                        except Exception as e:
                            st.warning(f"⚠️ Error extracting price for {ticker}: {str(e)}")
                
                # Forward fill any NaN values
                prices_df = prices_df.ffill().bfill()
                
                if prices_df.empty:
                    st.error("❌ No price data available.")
                else:
                    # Determine if we should use transaction-based or drift-based calculation
                    use_transactions = len(st.session_state.transactions) > 0
                    
                    # Filter target_portfolio to only include assets with transactions
                    if use_transactions:
                        # Get unique tickers from BUY transactions
                        tickers_with_transactions = set()
                        for trans in st.session_state.transactions:
                            if trans['type'] in ['BUY', 'SELL', 'BUY_CALL', 'SELL_CALL']:
                                tickers_with_transactions.add(trans['ticker'])
                        
                        # Filter target_portfolio to only include these tickers
                        active_portfolio = {ticker: weight for ticker, weight in target_portfolio.items() 
                                          if ticker in tickers_with_transactions}
                        
                        # Renormalize weights to sum to 100%
                        if active_portfolio:
                            total_active_weight = sum(active_portfolio.values())
                            if total_active_weight > 0:
                                active_portfolio = {ticker: weight / total_active_weight 
                                                  for ticker, weight in active_portfolio.items()}
                        
                        # Use active_portfolio for analysis
                        analysis_portfolio = active_portfolio if active_portfolio else target_portfolio
                        
                        st.info(f"📊 **Transaction Mode Active**: Showing only assets you own ({len(analysis_portfolio)} of {len(target_portfolio)} target assets)")
                    else:
                        # No transactions - use full target portfolio
                        analysis_portfolio = target_portfolio
                        st.info("📊 **Drift Mode Active**: Portfolio analysis based on buy-and-hold from initial allocation")
                    
                    if use_transactions:
                        # Calculate capital flows
                        total_deposits = sum(t['amount'] for t in st.session_state.transactions if t['type'] == 'DEPOSIT')
                        total_withdrawals = sum(t['amount'] for t in st.session_state.transactions if t['type'] == 'WITHDRAWAL')
                        total_buys = sum(t['amount'] for t in st.session_state.transactions if t['type'] == 'BUY')
                        total_sells = sum(t['amount'] for t in st.session_state.transactions if t['type'] == 'SELL')
                        total_dividends = sum(t['amount'] for t in st.session_state.transactions if t['type'] == 'DIVIDEND')
                        
                        if total_deposits == 0:
                            st.error("⚠️ Please add at least one DEPOSIT transaction to start tracking your portfolio")
                            st.stop()
                        
                        initial_capital = total_deposits  # Initial capital comes from deposits
                        net_capital = total_deposits - total_withdrawals  # Net cash contributed
                        total_invested = total_buys  # Actual amount invested in securities
                        
                        # Calculate portfolio with or without rebalancing
                        if st.session_state.rebalancing_enabled:
                            # WITH REBALANCING: Simulate periodic rebalancing
                            portfolio_value_series, holdings_over_time, cash_over_time, rebalancing_events, augmented_transactions = simulate_portfolio_with_rebalancing(
                                analysis_portfolio,
                                prices_df,
                                initial_capital,
                                investment_datetime,
                                st.session_state.transactions,
                                st.session_state.rebalancing_threshold / 100,  # Convert to decimal
                                st.session_state.rebalancing_period_months
                            )
                            
                            # Also calculate without rebalancing for comparison
                            portfolio_value_no_rebal, _, _, _, _ = calculate_portfolio_with_transactions(
                                analysis_portfolio,
                                prices_df,
                                initial_capital,
                                investment_datetime,
                                st.session_state.transactions
                            )
                            
                            # IMPORTANT: Ensure both series are properly aligned and start at same point
                            # They should have the same index from calculate_portfolio_with_transactions
                            if len(portfolio_value_series) > 0 and len(portfolio_value_no_rebal) > 0:
                                # Check if starting values match
                                start_with = portfolio_value_series.iloc[0]
                                start_without = portfolio_value_no_rebal.iloc[0]
                                if abs(start_with - start_without) > 0.01:  # They should be identical
                                    st.error(f"⚠️ **Rebalancing calculation issue**: Starting values don't match!\n\n- With rebalancing: ${start_with:,.2f}\n- Without rebalancing: ${start_without:,.2f}\n\nThis is a bug - both should start at ${initial_capital:,.2f}")
                            
                            # Show rebalancing info
                            if rebalancing_events:
                                first_rebal_date = rebalancing_events[0]['date']
                                days_until_first = (first_rebal_date - investment_datetime).days
                                
                                st.info(f"""
                                🔄 **{len(rebalancing_events)} Rebalancing Events** occurred during the simulation period 
                                (only assets exceeding {st.session_state.rebalancing_threshold}% threshold)
                                
                                First rebalancing: {first_rebal_date.strftime('%Y-%m-%d')} ({days_until_first} days after start)
                                """)
                                
                                # Check if rebalancing happened too early
                                if days_until_first < 7:
                                    st.warning(f"""
                                    ⚠️ **Rebalancing occurred very early** ({days_until_first} days after start).
                                    This might cause the different starting values. Rebalancing should not occur 
                                    immediately after portfolio inception.
                                    """)
                            else:
                                st.info(f"🔄 **No Rebalancing Events** occurred (no assets exceeded {st.session_state.rebalancing_threshold}% threshold)")
                            
                            # The put_positions from rebalancing aren't calculated, so use empty
                            put_positions = []
                            put_value_series = pd.Series(0, index=portfolio_value_series.index)
                            
                            # FIX: Store augmented transactions for use in holdings display
                            effective_transactions = augmented_transactions
                        else:
                            # WITHOUT REBALANCING: Standard calculation
                            portfolio_value_series, holdings_over_time, cash_over_time, put_positions, put_value_series = calculate_portfolio_with_transactions(
                                analysis_portfolio,  # Use filtered portfolio
                                prices_df,
                                initial_capital,
                                investment_datetime,
                                st.session_state.transactions
                            )
                            portfolio_value_no_rebal = None
                            rebalancing_events = []
                            
                            # FIX: Use original transactions when not rebalancing
                            effective_transactions = st.session_state.transactions
                        
                        # Calculate actual weights based on holdings
                        weights_over_time = calculate_actual_weights_with_transactions(
                            holdings_over_time,
                            prices_df,
                            analysis_portfolio  # Use filtered portfolio
                        )
                        
                        # =============================================================
                        # CALCULATE CUSTOM BENCHMARK PERFORMANCE
                        # =============================================================
                        benchmark_value_series = None
                        benchmark_returns = None

                        if st.session_state.custom_benchmark:
                            with st.spinner("📊 Calculating custom benchmark performance..."):
                                try:
                                    # Fetch benchmark data
                                    bench_data = fetch_data(
                                        st.session_state.benchmark_components,
                                        investment_datetime,
                                        datetime.now()
                                    )
                                    
                                    if bench_data:
                                        # Build benchmark prices dataframe
                                        bench_prices_df = pd.DataFrame()
                                        for comp in st.session_state.benchmark_components:
                                            if comp in bench_data:
                                                df = bench_data[comp]
                                                # Extract Close price
                                                if isinstance(df.columns, pd.MultiIndex):
                                                    if ('Close', comp) in df.columns:
                                                        bench_prices_df[comp] = df[('Close', comp)]
                                                    elif 'Close' in df.columns.get_level_values(0):
                                                        close_df = df['Close']
                                                        if isinstance(close_df, pd.DataFrame):
                                                            bench_prices_df[comp] = close_df.iloc[:, 0] if comp not in close_df.columns else close_df[comp]
                                                        else:
                                                            bench_prices_df[comp] = close_df
                                                elif 'Close' in df.columns:
                                                    bench_prices_df[comp] = df['Close']
                                        
                                        # Forward fill any NaN values
                                        bench_prices_df = bench_prices_df.ffill().bfill()
                                        
                                        if not bench_prices_df.empty:
                                            # Create benchmark target portfolio from components
                                            benchmark_target = {
                                                comp: weight 
                                                for comp, weight in zip(st.session_state.benchmark_components, st.session_state.benchmark_weights)
                                            }
                                            
                                            # Check if rebalancing is enabled
                                            if st.session_state.rebalancing_enabled:
                                                # REBALANCED BENCHMARK - use same logic as portfolio
                                                st.info(f"📊 Benchmark rebalancing: {st.session_state.rebalancing_threshold}% threshold, every {st.session_state.rebalancing_period_months} months (same as portfolio)")
                                                
                                                # Create a single "DEPOSIT" transaction for benchmark
                                                benchmark_transactions = [{
                                                    'date': investment_datetime.date() if hasattr(investment_datetime, 'date') else investment_datetime,
                                                    'type': 'DEPOSIT',
                                                    'ticker': 'CASH',
                                                    'shares': 0,
                                                    'price': 0,
                                                    'amount': initial_capital,
                                                    'notes': 'Initial benchmark capital',
                                                    'timestamp': investment_datetime
                                                }]
                                                
                                                # Add initial BUY transactions for each benchmark component
                                                for comp, weight in benchmark_target.items():
                                                    if comp in bench_prices_df.columns:
                                                        initial_price = bench_prices_df[comp].iloc[0]
                                                        dollar_amount = initial_capital * weight
                                                        shares = dollar_amount / initial_price
                                                        
                                                        benchmark_transactions.append({
                                                            'date': investment_datetime.date() if hasattr(investment_datetime, 'date') else investment_datetime,
                                                            'type': 'BUY',
                                                            'ticker': comp,
                                                            'shares': shares,
                                                            'price': initial_price,
                                                            'amount': dollar_amount,
                                                            'notes': 'Initial benchmark allocation',
                                                            'timestamp': investment_datetime
                                                        })
                                                
                                                # Simulate benchmark with rebalancing using the SAME function
                                                benchmark_value_series, _, _, benchmark_rebal_events, _ = simulate_portfolio_with_rebalancing(
                                                    benchmark_target,
                                                    bench_prices_df,
                                                    initial_capital,
                                                    investment_datetime,
                                                    benchmark_transactions,
                                                    st.session_state.rebalancing_threshold / 100,  # Convert to decimal
                                                    st.session_state.rebalancing_period_months
                                                )
                                                
                                                # Show benchmark rebalancing info
                                                if benchmark_rebal_events:
                                                    st.success(f"✅ Benchmark rebalanced {len(benchmark_rebal_events)} times (same rules as portfolio)")
                                                else:
                                                    st.info("ℹ️ Benchmark did not require rebalancing (within threshold)")
                                                
                                            else:
                                                # BUY-AND-HOLD BENCHMARK (original logic)
                                                benchmark_value_series = calculate_benchmark_performance(
                                                    st.session_state.custom_benchmark,
                                                    bench_prices_df
                                                )
                                                
                                                if benchmark_value_series is not None:
                                                    # Align with portfolio timeline
                                                    benchmark_value_series = benchmark_value_series.reindex(portfolio_value_series.index)
                                                    benchmark_value_series = benchmark_value_series.fillna(method='ffill').fillna(method='bfill')
                                                    
                                                    # Scale to initial capital
                                                    benchmark_value_series = benchmark_value_series * initial_capital
                                            
                                            if benchmark_value_series is not None:
                                                # Align with portfolio timeline (for rebalanced case too)
                                                if len(benchmark_value_series) != len(portfolio_value_series):
                                                    benchmark_value_series = benchmark_value_series.reindex(portfolio_value_series.index)
                                                    benchmark_value_series = benchmark_value_series.fillna(method='ffill').fillna(method='bfill')
                                                
                                                # Calculate benchmark returns
                                                benchmark_returns = benchmark_value_series.pct_change().dropna()
                                                
                                                rebal_status = " (with rebalancing)" if st.session_state.rebalancing_enabled else " (buy-and-hold)"
                                                st.success(f"✅ Benchmark calculated{rebal_status}: {' + '.join([f'{w*100:.0f}% {c}' for c, w in zip(st.session_state.benchmark_components, st.session_state.benchmark_weights)])}")
                                            else:
                                                st.warning("⚠️ Could not calculate benchmark performance")
                                        else:
                                            st.warning("⚠️ Could not fetch benchmark price data")
                                    else:
                                        st.warning("⚠️ Could not fetch benchmark data")
                                except Exception as e:
                                    st.warning(f"⚠️ Error calculating benchmark: {str(e)}")

                        
                    else:
                        # No transactions - require them now
                        st.error("⚠️ **No Transactions Found**: Please add transactions to track your portfolio")
                        st.info("💡 **How to start:**")
                        st.markdown("""
                        1. Add a **DEPOSIT** transaction (your initial capital)
                        2. Add **BUY** transactions for assets you purchased
                        3. Click **Track Portfolio Evolution** to see results
                        """)
                        st.stop()
                    
                    if weights_over_time.empty:
                        st.error("❌ Could not calculate weight evolution.")
                    else:
                        # =============================================================
                        # CURRENT PORTFOLIO STATUS
                        # =============================================================
                        st.subheader("📊 Current Portfolio Status")
                        
                        current_weights = {ticker: float(weights_over_time[ticker].iloc[-1]) 
                                          for ticker in weights_over_time.columns}
                        
                        current_prices = {ticker: float(prices_df[ticker].iloc[-1])
                                         for ticker in prices_df.columns}
                        
                        initial_prices = {ticker: float(prices_df[ticker].iloc[0])
                                         for ticker in prices_df.columns}
                        
                        # Current portfolio value from series (includes cash + securities + call options)
                        current_value = portfolio_value_series.iloc[-1]
                        
                        # Get current cash position
                        current_cash = cash_over_time.iloc[-1] if cash_over_time is not None else 0
                        
                        # Calculate securities value (for display)
                        securities_value = current_value - current_cash
                        
                        # =============================================================
                        # CORRECT RETURN CALCULATION
                        # =============================================================
                        # Total Gain = Current Portfolio Value - Net Capital Contributed
                        # This includes ALL gains: stock appreciation + call option profits + dividends
                        # Net Capital = Deposits - Withdrawals (what you put in minus what you took out)
                        
                        total_gain = current_value - net_capital
                        
                        # Investment Return % = Total Gain / Total Invested (what you actually bought with)
                        if total_invested > 0:
                            investment_return = (total_gain / total_invested) * 100
                        else:
                            investment_return = 0
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Current Portfolio Value",
                                f"${current_value:,.0f}",
                                delta=f"{investment_return:.2f}% total return",
                                help="Total value: securities + cash (includes all gains)"
                            )
                        
                        with col2:
                            days_held = (datetime.now() - investment_datetime).days
                            st.metric(
                                "Days Held",
                                f"{days_held}",
                                delta=f"{days_held/30:.1f} months",
                                help="Number of days since first transaction"
                            )
                        
                        with col3:
                            st.metric(
                                "Total Gain/Loss",
                                f"${total_gain:,.0f}",
                                delta=f"{investment_return:+.2f}%",
                                help="Current value minus net capital (includes stock gains + option profits + dividends)"
                            )
                        
                        with col4:
                            # Calculate annualized return based on total gain
                            if days_held > 0 and total_invested > 0:
                                total_return_decimal = total_gain / total_invested
                                years_held = days_held / DAYS_PER_YEAR
                                if years_held > 0:
                                    annual_return = ((1 + total_return_decimal) ** (1 / years_held) - 1) * 100
                                else:
                                    annual_return = investment_return
                            else:
                                annual_return = 0
                            
                            # Calculate risk metrics for different time periods
                            returns = portfolio_value_series.pct_change().fillna(0)
                            portfolio_risk_metrics = calculate_risk_metrics(returns)
                            
                            # Calculate Sharpe ratio for Since Inception
                            sharpe_inception = portfolio_risk_metrics.get('sharpe_ratio', 0)
                            
                            # Calculate Sharpe for Last 5 Years
                            five_years_ago = datetime.now() - timedelta(days=5*365)
                            returns_5y = portfolio_value_series[portfolio_value_series.index >= five_years_ago].pct_change().fillna(0)
                            if len(returns_5y) > 20:
                                risk_metrics_5y = calculate_risk_metrics(returns_5y)
                                sharpe_5y = risk_metrics_5y.get('sharpe_ratio', 0)
                            else:
                                sharpe_5y = None
                            
                            # Calculate Sharpe for Last 1 Year
                            one_year_ago = datetime.now() - timedelta(days=365)
                            returns_1y = portfolio_value_series[portfolio_value_series.index >= one_year_ago].pct_change().fillna(0)
                            if len(returns_1y) > 20:
                                risk_metrics_1y = calculate_risk_metrics(returns_1y)
                                sharpe_1y = risk_metrics_1y.get('sharpe_ratio', 0)
                            else:
                                sharpe_1y = None
                            
                            st.metric(
                                "Annualized Return",
                                f"{annual_return:.2f}%",
                                help="CAGR based on total invested"
                            )
                        
                        # Show capital flow summary if using transactions
                        if use_transactions:
                            st.markdown("#### 💰 Capital Flow Summary")
                            col1, col2, col3, col4, col5, col6 = st.columns(6)
                            
                            with col1:
                                st.metric(
                                    "Total Deposits", 
                                    f"${total_deposits:,.0f}",
                                    help="Total cash deposited into portfolio"
                                )
                            with col2:
                                st.metric(
                                    "Withdrawals", 
                                    f"${total_withdrawals:,.0f}",
                                    help="Total cash withdrawn from portfolio"
                                )
                            with col3:
                                st.metric(
                                    "Net Capital", 
                                    f"${net_capital:,.0f}",
                                    help="Deposits minus withdrawals"
                                )
                            with col4:
                                st.metric(
                                    "Total Invested", 
                                    f"${total_invested:,.0f}",
                                    help="Total amount used to buy securities"
                                )
                            with col5:
                                st.metric(
                                    "Total Sells", 
                                    f"${total_sells:,.0f}",
                                    help="Total proceeds from selling securities"
                                )
                            with col6:
                                st.metric(
                                    "Cash Position", 
                                    f"${current_cash:,.0f}",
                                    help="Current uninvested cash"
                                )
                        
                        # Show current holdings if using transactions
                        if use_transactions and holdings_over_time is not None:
                            st.markdown("#### 📦 Current Holdings")
                            
                            holdings_display = []
                            for ticker in analysis_portfolio.keys():
                                if ticker in holdings_over_time.columns:
                                    shares = holdings_over_time[ticker].iloc[-1]
                                    if shares > 0 and ticker in current_prices:
                                        price = current_prices[ticker]
                                        market_value = shares * price
                                        weight = current_weights.get(ticker, 0) * 100
                                        target_weight = analysis_portfolio[ticker] * 100
                                        
                                        # FIX: Calculate cost basis using effective_transactions (includes rebalancing)
                                        buy_transactions = [t for t in effective_transactions 
                                                          if t['type'] == 'BUY' and t['ticker'] == ticker]
                                        sell_transactions = [t for t in effective_transactions 
                                                           if t['type'] == 'SELL' and t['ticker'] == ticker]
                                        
                                        if buy_transactions:
                                            total_buys = sum(t['amount'] for t in buy_transactions)
                                            total_sells = sum(t['amount'] for t in sell_transactions)
                                            total_shares_bought = sum(t['shares'] for t in buy_transactions)
                                            total_shares_sold = sum(t['shares'] for t in sell_transactions)
                                            
                                            # FIX: Avg Cost = Total $ Bought / Total Shares Bought
                                            avg_cost = total_buys / total_shares_bought if total_shares_bought > 0 else 0
                                            
                                            # FIX: Calculate cost basis of CURRENT shares held
                                            # Using average cost method: current shares * avg cost
                                            cost_basis_current = shares * avg_cost
                                            
                                            # Gain/Loss = Market Value - Cost Basis of Current Shares
                                            gain_loss = market_value - cost_basis_current
                                            gain_loss_pct = (gain_loss / cost_basis_current) * 100 if cost_basis_current > 0 else 0
                                        else:
                                            avg_cost = initial_prices[ticker]
                                            cost_basis_current = shares * avg_cost
                                            gain_loss = market_value - cost_basis_current
                                            gain_loss_pct = (gain_loss / cost_basis_current) * 100 if cost_basis_current > 0 else 0
                                        
                                        # Store numeric return for sorting
                                        holdings_display.append({
                                            'Ticker': ticker,
                                            'Shares': shares,
                                            'Price': price,
                                            'Avg Cost': avg_cost,
                                            'Cost Basis': cost_basis_current,
                                            'Market Value': market_value,
                                            'Weight': weight,
                                            'Target': target_weight,
                                            'Gain/Loss': gain_loss,
                                            'Return': gain_loss_pct
                                        })
                            
                            if holdings_display:
                                holdings_df = pd.DataFrame(holdings_display)
                                
                                # Sort by Return % descending (highest returns on top)
                                holdings_df = holdings_df.sort_values('Return', ascending=False)
                                
                                # Format columns for display
                                display_df = holdings_df.copy()
                                display_df['Shares'] = display_df['Shares'].apply(lambda x: f"{x:.2f}")
                                display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}")
                                display_df['Avg Cost'] = display_df['Avg Cost'].apply(lambda x: f"${x:.2f}")
                                display_df['Cost Basis'] = display_df['Cost Basis'].apply(lambda x: f"${x:,.0f}")
                                display_df['Market Value'] = display_df['Market Value'].apply(lambda x: f"${x:,.0f}")
                                display_df['Weight'] = display_df['Weight'].apply(lambda x: f"{x:.2f}%")
                                display_df['Target'] = display_df['Target'].apply(lambda x: f"{x:.2f}%")
                                display_df['Gain/Loss'] = display_df['Gain/Loss'].apply(lambda x: f"${x:,.0f}")
                                display_df['Return'] = display_df['Return'].apply(lambda x: f"{x:+.2f}%")
                                
                                def color_return(val):
                                    if '+' in str(val):
                                        return 'background-color: #00cc44; color: white; font-weight: bold'
                                    elif '-' in str(val):
                                        return 'background-color: #ff4444; color: white; font-weight: bold'
                                    return ''
                                
                                styled_holdings = display_df.style.map(color_return, subset=['Return'])
                                st.dataframe(styled_holdings, use_container_width=True, hide_index=True)
                        
                        
                        # =============================================================
                        # PUT OPTION HEDGING ANALYSIS
                        # =============================================================
                        if use_transactions and put_positions:
                            st.markdown("#### 📉 Put Option Hedging Analysis")
                            st.markdown("*Performance tracking for downside protection positions*")
                            
                            # Important disclaimer about valuation
                            st.warning("""
                            ⚠️ **Valuation Note**: This analysis shows **intrinsic value only** (Strike - Underlying Price).
                            
                            Real option prices also include **time value** (extrinsic value) which depends on:
                            - Time to expiration (theta decay)
                            - Implied volatility (vega)
                            - Interest rates
                            
                            For OTM puts, the intrinsic value is $0 but the option may still have significant market value due to time premium. 
                            The actual P&L when you close a position will differ based on the market price you receive.
                            """)
                            
                            # Create put options summary table
                            put_summary = []
                            total_put_cost = 0
                            total_put_proceeds = 0
                            total_put_profit = 0
                            
                            for put in put_positions:
                                buy_date_str = put['buy_date'].strftime('%Y-%m-%d')
                                sell_date_str = put['sell_date'].strftime('%Y-%m-%d') if put['sell_date'] else 'ACTIVE'
                                
                                cost = put['total_cost']
                                total_put_cost += cost
                                
                                # Calculate proceeds
                                if put['sell_date']:
                                    proceeds = put.get('total_proceeds', 0)
                                    total_put_proceeds += proceeds
                                    profit = proceeds - cost
                                    total_put_profit += profit
                                    profit_pct = (profit / cost) * 100 if cost > 0 else 0
                                    status = "CLOSED"
                                else:
                                    # Calculate current value for active position (PUT: intrinsic = strike - underlying)
                                    if put['underlying'] in prices_df.columns:
                                        current_underlying = prices_df[put['underlying']].iloc[-1]
                                        # PUT intrinsic value = max(0, strike - underlying)
                                        intrinsic_value = max(0, put['strike'] - current_underlying)
                                        proceeds = intrinsic_value * put['contracts'] * put['multiplier']
                                        profit = proceeds - cost
                                        profit_pct = (profit / cost) * 100 if cost > 0 else 0
                                        status = "ACTIVE (ITM)" if intrinsic_value > 0 else "ACTIVE (OTM)"
                                    else:
                                        proceeds = 0
                                        profit = -cost
                                        profit_pct = -100
                                        status = "ACTIVE (No data)"
                                
                                put_summary.append({
                                    'Buy Date': buy_date_str,
                                    'Sell Date': sell_date_str,
                                    'Underlying': put['underlying'],
                                    'Strike': f"${put['strike']:.2f}",
                                    'Contracts': f"{put['contracts']:.0f}",
                                    'Premium Paid': f"${put['buy_premium']:.2f}",
                                    'Total Cost': f"${cost:,.0f}",
                                    'Current/Exit Value': f"${proceeds:,.0f}",
                                    'P&L': f"${profit:,.0f}",
                                    'Return': f"{profit_pct:+.1f}%",
                                    'Status': status
                                })
                            
                            if put_summary:
                                put_df = pd.DataFrame(put_summary)
                                
                                def color_put_status(val):
                                    if 'ITM' in str(val):
                                        return 'background-color: #a5d6a7; color: #1b5e20; font-weight: bold'
                                    elif 'OTM' in str(val):
                                        return 'background-color: #e1bee7; color: #6a1b9a; font-weight: bold'
                                    elif 'CLOSED' in str(val):
                                        return 'background-color: #e0e0e0; color: #424242; font-weight: bold'
                                    return ''
                                
                                def color_put_return(val):
                                    if '+' in str(val):
                                        return 'background-color: #a5d6a7; color: #1b5e20; font-weight: bold'
                                    elif '-' in str(val):
                                        return 'background-color: #ef9a9a; color: #b71c1c; font-weight: bold'
                                    return ''
                                
                                styled_puts = put_df.style.map(color_put_status, subset=['Status'])
                                styled_puts = styled_puts.map(color_put_return, subset=['Return'])
                                
                                st.dataframe(styled_puts, use_container_width=True, hide_index=True)
                                
                                # Put option performance summary
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Total Premium Paid", f"${total_put_cost:,.0f}",
                                             help="Total cost of all put option purchases")
                                
                                with col2:
                                    net_proceeds = total_put_proceeds
                                    # PUT active value: intrinsic = strike - underlying
                                    active_value = sum(
                                        max(0, c['strike'] - prices_df[c['underlying']].iloc[-1]) * c['contracts'] * c['multiplier']
                                        for c in put_positions 
                                        if c['sell_date'] is None and c['underlying'] in prices_df.columns
                                    )
                                    total_value = net_proceeds + active_value
                                    st.metric("Current/Realized Value", f"${total_value:,.0f}",
                                             help="Value from closed positions + current value of active positions")
                                
                                with col3:
                                    net_profit = total_value - total_put_cost
                                    st.metric("Total P&L", f"${net_profit:,.0f}",
                                             delta=f"{(net_profit/total_put_cost*100):+.1f}%" if total_put_cost > 0 else None,
                                             help="Total profit/loss on all put positions")
                                
                                with col4:
                                    active_count = sum(1 for c in put_positions if c['sell_date'] is None)
                                    st.metric("Active Positions", f"{active_count}",
                                             help="Number of currently open put option positions")
                                
                                # Put option value over time chart
                                if not put_value_series.empty and put_value_series.sum() > 0:
                                    st.markdown("##### 📈 Put Option Value Evolution")
                                    
                                    fig_put = go.Figure()
                                    
                                    fig_put.add_trace(go.Scatter(
                                        x=put_value_series.index,
                                        y=put_value_series,
                                        mode='lines',
                                        name='Put Option Value',
                                        line=dict(color='#FFD700', width=2.5),
                                        fill='tozeroy',
                                        fillcolor='rgba(255, 215, 0, 0.2)',
                                        hovertemplate='Value: $%{y:,.0f}<extra></extra>'
                                    ))
                                    
                                    fig_put.add_hline(
                                        y=total_put_cost,
                                        line_dash="dash",
                                        line_color="red",
                                        line_width=2,
                                        opacity=0.7,
                                        annotation_text=f"Total Cost: ${total_put_cost:,.0f}",
                                        annotation_position="left"
                                    )
                                    
                                    # Add markers for buy/sell events
                                    for put in put_positions:
                                        buy_idx = put_value_series.index.get_indexer([put['buy_date']], method='nearest')[0]
                                        if buy_idx >= 0:
                                            fig_put.add_trace(go.Scatter(
                                                x=[put_value_series.index[buy_idx]],
                                                y=[put_value_series.iloc[buy_idx]],
                                                mode='markers',
                                                name=f"Buy Put",
                                                marker=dict(size=12, color='green', symbol='triangle-up'),
                                                showlegend=False,
                                                hovertemplate=f"BUY: {put['underlying']} ${put['strike']:.2f} Call<extra></extra>"
                                            ))
                                        
                                        if put['sell_date']:
                                            sell_idx = put_value_series.index.get_indexer([put['sell_date']], method='nearest')[0]
                                            if sell_idx >= 0:
                                                fig_put.add_trace(go.Scatter(
                                                    x=[put_value_series.index[sell_idx]],
                                                    y=[put_value_series.iloc[sell_idx]],
                                                    mode='markers',
                                                    name=f"Sell Put",
                                                    marker=dict(size=12, color='red', symbol='triangle-down'),
                                                    showlegend=False,
                                                    hovertemplate=f"SELL: P&L ${put.get('total_proceeds', 0) - put['total_cost']:,.0f}<extra></extra>"
                                                ))
                                    
                                    fig_put.update_layout(
                                        template="plotly_dark",
                                        height=450,
                                        xaxis_title="Date",
                                        yaxis_title="Put Option Value ($)",
                                        hovermode='x',
                                        yaxis=dict(tickformat="$,.0f"),
                                        showlegend=False
                                    )
                                    
                                    st.plotly_chart(fig_put, use_container_width=True)
                                
                                # Impact analysis
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**💰 Capital Allocation Impact**")
                                    
                                    capital_to_hedge = (total_put_cost / initial_capital) * 100
                                    capital_to_assets = 100 - capital_to_hedge
                                    
                                    st.markdown(f"- **Hedging Cost**: ${total_put_cost:,.0f} ({capital_to_hedge:.2f}% of initial capital)")
                                    st.markdown(f"- **Asset Allocation**: ${initial_capital - total_put_cost:,.0f} ({capital_to_assets:.2f}% of initial capital)")
                                    st.markdown(f"- **Current Hedge Value**: ${total_value:,.0f}")
                                
                                with col2:
                                    st.markdown("**📊 Hedge Effectiveness**")
                                    
                                    if net_profit > 0:
                                        effectiveness = "✅ Profitable Hedge"
                                        color_text = "green"
                                    elif net_profit == 0:
                                        effectiveness = "⚖️ Breakeven Hedge"
                                        color_text = "orange"
                                    else:
                                        effectiveness = "❌ Loss on Hedge"
                                        color_text = "red"
                                    
                                    st.markdown(f"- **Status**: :{color_text}[{effectiveness}]")
                                    st.markdown(f"- **Return on Premium**: {(net_profit/total_put_cost*100):+.1f}%" if total_put_cost > 0 else "N/A")
                                    st.markdown(f"- **Portfolio Impact**: {(net_profit/current_value*100):+.2f}%")
                        
                        # =============================================================
                        # WEIGHT DRIFT ANALYSIS
                        # =============================================================
                        st.subheader("⚖️ Weight Drift Analysis")
                        st.markdown("*How your portfolio allocation has changed over time due to price movements*")
                        
                        # Calculate rebalancing needs
                        rebal_info, needs_rebalance = calculate_rebalancing_needs(
                            current_weights,
                            target_portfolio,
                            rebalance_threshold / 100
                        )
                        
                        if needs_rebalance:
                            st.warning(f"⚠️ **Rebalancing Alert**: One or more assets have drifted beyond {rebalance_threshold}% from target allocation!")
                        else:
                            st.success(f"✅ Portfolio is within {rebalance_threshold}% tolerance of target allocation")
                        
                        # Weight drift visualization
                        fig_drift = go.Figure()
                        
                        # Filter to only dates where portfolio has meaningful value (after first buy)
                        # Find first date where any position has value
                        first_position_date = None
                        for ticker in weights_over_time.columns:
                            if ticker in holdings_over_time.columns:
                                first_date_with_shares = holdings_over_time[holdings_over_time[ticker] > 0].index
                                if len(first_date_with_shares) > 0:
                                    if first_position_date is None or first_date_with_shares[0] < first_position_date:
                                        first_position_date = first_date_with_shares[0]
                        
                        # If we found a valid start date, filter the weights
                        if first_position_date is not None:
                            weights_to_plot = weights_over_time[weights_over_time.index >= first_position_date]
                        else:
                            weights_to_plot = weights_over_time
                        
                        for ticker in weights_to_plot.columns:
                            fig_drift.add_trace(go.Scatter(
                                x=weights_to_plot.index,
                                y=weights_to_plot[ticker] * 100,
                                mode='lines',
                                name=ticker,
                                line=dict(width=2.5),
                                hovertemplate=f'{ticker}: %{{y:.2f}}%<extra></extra>'
                            ))
                            
                            # Add target line
                            fig_drift.add_hline(
                                y=analysis_portfolio[ticker] * 100,
                                line_dash="dash",
                                opacity=0.3,
                                annotation_text=f"{ticker} Target",
                                annotation_position="right"
                            )
                        
                        fig_drift.update_layout(
                            template="plotly_dark",
                            height=450,
                            title="Portfolio Weight Evolution Over Time",
                            xaxis_title="Date",
                            yaxis_title="Weight (%)",
                            hovermode='x unified',
                            legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=1.01)
                        )
                        
                        st.plotly_chart(fig_drift, use_container_width=True)
                        
                        # =============================================================
                        # CURRENT VS TARGET COMPARISON
                        # =============================================================
                        st.subheader("🎯 Current vs Target Allocation")
                        
                        # Prepare data for grouped bar chart
                        allocation_data = []
                        for ticker in analysis_portfolio.keys():
                            allocation_data.append({
                                'Asset': ticker,
                                'Current': current_weights.get(ticker, 0) * 100,
                                'Target': analysis_portfolio[ticker] * 100
                            })
                        
                        allocation_df = pd.DataFrame(allocation_data)
                        
                        # Create grouped vertical bar chart
                        fig_allocation = go.Figure()
                        
                        fig_allocation.add_trace(go.Bar(
                            name='Current',
                            x=allocation_df['Asset'],
                            y=allocation_df['Current'],
                            marker_color='#00d4ff',
                            text=allocation_df['Current'].round(1),
                            texttemplate='%{text}%',
                            textposition='outside'
                        ))
                        
                        fig_allocation.add_trace(go.Bar(
                            name='Target',
                            x=allocation_df['Asset'],
                            y=allocation_df['Target'],
                            marker_color='#ff9500',
                            text=allocation_df['Target'].round(1),
                            texttemplate='%{text}%',
                            textposition='outside'
                        ))
                        
                        fig_allocation.update_layout(
                            template="plotly_dark",
                            title="Current vs Target Allocation",
                            height=500,
                            barmode='group',
                            xaxis_title="Asset",
                            yaxis_title="Weight (%)",
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ),
                            yaxis=dict(range=[0, max(max(allocation_df['Current']), max(allocation_df['Target'])) * 1.15])
                        )
                        
                        st.plotly_chart(fig_allocation, use_container_width=True)
                        
                        # =============================================================
                        # DRIFT TABLE
                        # =============================================================
                        st.markdown("### 📋 Detailed Drift Analysis")
                        
                        drift_data = []
                        for ticker, info in rebal_info.items():
                            drift_data.append({
                                'Asset': ticker,
                                'Current (%)': f"{info['current']:.2f}",
                                'Target (%)': f"{info['target']:.2f}",
                                'Drift (%)': f"{info['drift']:.2f}",
                                'Drift vs Target (%)': f"{info['drift_pct']:.1f}",
                                'Status': info['action']
                            })
                        
                        drift_df = pd.DataFrame(drift_data)
                        
                        # Color code the status with lighter, more visible colors
                        def color_status(val):
                            if val == 'SELL':
                                return 'background-color: #ef9a9a; color: #b71c1c; font-weight: bold'
                            elif val == 'BUY':
                                return 'background-color: #a5d6a7; color: #1b5e20; font-weight: bold'
                            else:
                                return 'background-color: #fff59d; color: #f57f17; font-weight: bold'
                        
                        styled_drift = drift_df.style.map(
                            color_status, 
                            subset=['Status']
                        )
                        
                        st.dataframe(styled_drift, use_container_width=True, hide_index=True)
                        
                        # =============================================================
                        # REBALANCING RECOMMENDATIONS
                        # =============================================================
                        if show_recommendations and needs_rebalance:
                            st.subheader("💡 Rebalancing Recommendations")
                            st.markdown("*Specific actions to bring your portfolio back to target allocation*")
                            
                            # Calculate dollar amounts
                            rebal_actions = []
                            
                            for ticker, info in rebal_info.items():
                                if info['action'] != 'HOLD':
                                    # Skip if price data not available
                                    if ticker not in current_prices:
                                        continue
                                        
                                    current_dollar = current_value * (info['current'] / 100)
                                    target_dollar = current_value * (info['target'] / 100)
                                    difference = target_dollar - current_dollar
                                    
                                    # Calculate shares
                                    shares_to_trade = abs(difference) / current_prices[ticker]
                                    
                                    rebal_actions.append({
                                        'Asset': ticker,
                                        'Action': info['action'],
                                        'Current Value': f"${current_dollar:,.0f}",
                                        'Target Value': f"${target_dollar:,.0f}",
                                        'Adjustment': f"${difference:,.0f}",
                                        'Shares': f"{shares_to_trade:.2f}",
                                        'Price': f"${current_prices[ticker]:.2f}"
                                    })
                            
                            if rebal_actions:
                                rebal_df = pd.DataFrame(rebal_actions)
                                
                                st.markdown("#### Recommended Trades")
                                st.dataframe(rebal_df, use_container_width=True, hide_index=True)
                                
                                # Summary
                                total_to_sell = sum([float(a['Adjustment'].replace('$', '').replace(',', '')) 
                                                    for a in rebal_actions if a['Action'] == 'SELL'])
                                total_to_buy = sum([float(a['Adjustment'].replace('$', '').replace(',', '')) 
                                                   for a in rebal_actions if a['Action'] == 'BUY'])
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total to Sell", f"${abs(total_to_sell):,.0f}")
                                with col2:
                                    st.metric("Total to Buy", f"${total_to_buy:,.0f}")
                                with col3:
                                    net_cash = total_to_sell + total_to_buy
                                    st.metric("Net Cash Flow", f"${net_cash:,.0f}")
                                
                                st.info("💡 **Note**: These are approximate values. Actual execution may vary due to market prices and transaction costs.")
                        
                        # =============================================================
                        # PERFORMANCE ATTRIBUTION
                        # =============================================================
                        st.subheader("📈 Performance Attribution")
                        st.markdown("*Which assets contributed most to your portfolio's performance*")
                        
                        # Calculate individual asset returns based on ACTUAL investments
                        attribution_data = []
                        
                        for ticker in analysis_portfolio.keys():
                            if ticker in prices_df.columns and ticker in current_prices:
                                # Get actual shares held
                                if ticker in holdings_over_time.columns:
                                    shares_held = holdings_over_time[ticker].iloc[-1]
                                    if shares_held > 0:
                                        # FIX: Use effective_transactions (includes rebalancing)
                                        buy_transactions = [t for t in effective_transactions 
                                                          if t['type'] == 'BUY' and t['ticker'] == ticker]
                                        sell_transactions = [t for t in effective_transactions 
                                                           if t['type'] == 'SELL' and t['ticker'] == ticker]
                                        
                                        if buy_transactions:
                                            total_buys = sum(t['amount'] for t in buy_transactions)
                                            total_sells = sum(t['amount'] for t in sell_transactions)
                                            total_shares_bought = sum(t['shares'] for t in buy_transactions)
                                            
                                            # FIX: Consistent avg cost calculation
                                            avg_cost = total_buys / total_shares_bought if total_shares_bought > 0 else 0
                                            
                                            # FIX: Cost basis of current position = shares * avg cost
                                            cost_basis_current = shares_held * avg_cost
                                            
                                            # Current market value
                                            current_price = current_prices[ticker]
                                            current_market_value = shares_held * current_price
                                            
                                            # FIX: Consistent gain/loss calculation
                                            dollar_gain = current_market_value - cost_basis_current
                                            
                                            # Percentage return based on cost basis
                                            pct_return = (dollar_gain / cost_basis_current) * 100 if cost_basis_current > 0 else 0
                                            
                                            attribution_data.append({
                                                'Asset': ticker,
                                                'Invested': cost_basis_current,  # FIX: Use cost basis of current position
                                                'Current Value': current_market_value,
                                                'Gain/Loss ($)': dollar_gain,
                                                'Return (%)': pct_return,
                                                'Avg Cost': avg_cost,
                                                'Current Price': current_price
                                            })
                        
                        # Add put option P&L to attribution
                        if put_positions:
                            total_put_cost = 0
                            total_put_value = 0
                            
                            for put in put_positions:
                                put_cost = put.get('total_cost', 0)
                                total_put_cost += put_cost
                                
                                if put['sell_date']:
                                    # Closed position
                                    put_value = put.get('total_proceeds', 0)
                                else:
                                    # Active position - calculate current value (PUT: strike - underlying)
                                    underlying = put['underlying']
                                    if underlying in current_prices:
                                        underlying_price = current_prices[underlying]
                                        intrinsic = max(0, put['strike'] - underlying_price)
                                        put_value = intrinsic * put['contracts'] * put['multiplier']
                                    else:
                                        put_value = 0
                                
                                total_put_value += put_value
                            
                            if total_put_cost > 0:
                                put_gain = total_put_value - total_put_cost
                                put_return = (put_gain / total_put_cost) * 100
                                
                                attribution_data.append({
                                    'Asset': 'Put Options',
                                    'Invested': total_put_cost,
                                    'Current Value': total_put_value,
                                    'Gain/Loss ($)': put_gain,
                                    'Return (%)': put_return,
                                    'Avg Cost': 0,
                                    'Current Price': 0
                                })
                        
                        if attribution_data:
                            attr_df = pd.DataFrame(attribution_data)
                            attr_df = attr_df.sort_values('Gain/Loss ($)', ascending=True)
                            
                            # Contribution bar chart
                            fig_contrib = go.Figure()
                            
                            colors = ['#ff4444' if x < 0 else '#00ff88' for x in attr_df['Gain/Loss ($)']]
                            
                            fig_contrib.add_trace(go.Bar(
                                y=attr_df['Asset'],
                                x=attr_df['Gain/Loss ($)'],
                                orientation='h',
                                marker_color=colors,
                                text=[f'${x:,.0f}' for x in attr_df['Gain/Loss ($)']],
                                textposition='outside',
                                hovertemplate='%{y}<br>Gain/Loss: $%{x:,.0f}<extra></extra>'
                            ))
                            
                            fig_contrib.update_layout(
                                template="plotly_dark",
                                height=450,
                                title="Dollar Gain/Loss by Asset",
                                xaxis_title="Gain/Loss ($)",
                                yaxis_title="",
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig_contrib, use_container_width=True)
                            
                            # Detailed attribution table
                            st.markdown("#### Detailed Performance Attribution")
                            
                            # FIX: Sort by Return % descending (highest returns on top)
                            display_attr = attr_df.sort_values('Return (%)', ascending=False).copy()
                            display_attr['Invested'] = display_attr['Invested'].apply(lambda x: f"${x:,.0f}")
                            display_attr['Current Value'] = display_attr['Current Value'].apply(lambda x: f"${x:,.0f}")
                            display_attr['Gain/Loss ($)'] = display_attr['Gain/Loss ($)'].apply(lambda x: f"${x:,.0f}")
                            display_attr['Return (%)'] = display_attr['Return (%)'].apply(lambda x: f"{x:+.2f}%")
                            display_attr['Avg Cost'] = display_attr['Avg Cost'].apply(lambda x: f"${x:.2f}")
                            display_attr['Current Price'] = display_attr['Current Price'].apply(lambda x: f"${x:.2f}")
                            
                            # NOTE: Detailed table removed - information now shown in Current Holdings with Cost Basis column
                        else:
                            st.info("No investment data available for attribution analysis")
                        
                        # =============================================================
                        # PERIOD PERFORMANCE ANALYSIS
                        # =============================================================
                        st.subheader("📊 Period Performance Analysis")
                        
                        # Check if benchmark is available
                        if benchmark_value_series is None:
                            st.info("💡 **Configure a Custom Benchmark** at the top to enable detailed performance comparisons")
                        else:
                            st.markdown(f"*Comparing against: {' + '.join([f'{w*100:.0f}% {c}' for c, w in zip(st.session_state.benchmark_components, st.session_state.benchmark_weights)])}*")
                            
                            # Define analysis periods
                            current_date = datetime.now()
                            
                            periods = {
                                'YTD': datetime(current_date.year, 1, 1),
                                '1M': current_date - timedelta(days=30),
                                '3M': current_date - timedelta(days=90),
                                '6M': current_date - timedelta(days=180),
                                '1Y': current_date - timedelta(days=365),
                                '2Y': current_date - timedelta(days=730),
                                '3Y': current_date - timedelta(days=1095),
                                '4Y': current_date - timedelta(days=1460),
                                '5Y': current_date - timedelta(days=1825),
                                'Since Inception': investment_datetime
                            }
                            
                            # Calculate performance for each period
                            performance_data = []
                            
                            for period_name, period_start in periods.items():
                                # Skip if period starts before investment
                                if period_start < investment_datetime:
                                    continue
                                
                                # Get portfolio values for this period
                                period_portfolio = portfolio_value_series[portfolio_value_series.index >= period_start]
                                
                                if len(period_portfolio) < 2:
                                    continue
                                
                                # Portfolio metrics
                                portfolio_start = period_portfolio.iloc[0]
                                portfolio_end = period_portfolio.iloc[-1]
                                
                                # Only calculate return if start value is meaningful (> $1000)
                                if portfolio_start > 1000:
                                    portfolio_return = ((portfolio_end - portfolio_start) / portfolio_start) * 100
                                else:
                                    # Skip this period - portfolio too small at start
                                    continue
                                
                                # Calculate volatility for the period
                                period_returns = period_portfolio.pct_change().dropna()
                                portfolio_vol = period_returns.std() * np.sqrt(252) * 100
                                
                                # Calculate max drawdown for period
                                period_cumulative = period_portfolio / period_portfolio.iloc[0]
                                period_running_max = period_cumulative.expanding().max()
                                period_drawdown = ((period_cumulative - period_running_max) / period_running_max) * 100
                                portfolio_max_dd = period_drawdown.min()
                                
                                # Portfolio Sharpe ratio (with risk-free rate)
                                if portfolio_vol > 0:
                                    # Proper annualization
                                    days_in_period = (period_portfolio.index[-1] - period_portfolio.index[0]).days
                                    if days_in_period > 0:
                                        total_return_decimal = portfolio_return / 100
                                        years_in_period = days_in_period / 365.25
                                        annualized_return = ((1 + total_return_decimal) ** (1 / years_in_period) - 1) * 100
                                    else:
                                        annualized_return = portfolio_return
                                    
                                    # Sharpe = (Return - Risk-Free Rate) / Volatility
                                    rfr_pct = DEFAULT_RISK_FREE_RATE * 100
                                    excess_return = annualized_return - rfr_pct
                                    portfolio_sharpe = excess_return / portfolio_vol
                                else:
                                    annualized_return = portfolio_return
                                    portfolio_sharpe = 0
                                
                                # Benchmark metrics
                                period_benchmark = benchmark_value_series[benchmark_value_series.index >= period_start]
                                
                                if len(period_benchmark) >= 2:
                                    benchmark_start = period_benchmark.iloc[0]
                                    benchmark_end = period_benchmark.iloc[-1]
                                    benchmark_return = ((benchmark_end - benchmark_start) / benchmark_start) * 100
                                    
                                    # Benchmark volatility
                                    benchmark_period_returns = period_benchmark.pct_change().dropna()
                                    benchmark_vol = benchmark_period_returns.std() * np.sqrt(252) * 100
                                    
                                    # Benchmark max drawdown
                                    benchmark_cumulative = period_benchmark / period_benchmark.iloc[0]
                                    benchmark_running_max = benchmark_cumulative.expanding().max()
                                    benchmark_drawdown = ((benchmark_cumulative - benchmark_running_max) / benchmark_running_max) * 100
                                    benchmark_max_dd = benchmark_drawdown.min()
                                    
                                    # Benchmark Sharpe ratio (with risk-free rate)
                                    if benchmark_vol > 0:
                                        days_in_period_benchmark = (period_benchmark.index[-1] - period_benchmark.index[0]).days
                                        if days_in_period_benchmark > 0:
                                            benchmark_return_decimal = benchmark_return / 100
                                            years_in_period_benchmark = days_in_period_benchmark / 365.25
                                            benchmark_annualized_return = ((1 + benchmark_return_decimal) ** (1 / years_in_period_benchmark) - 1) * 100
                                        else:
                                            benchmark_annualized_return = benchmark_return
                                        # Sharpe = (Return - Risk-Free Rate) / Volatility
                                        benchmark_excess_return = benchmark_annualized_return - (DEFAULT_RISK_FREE_RATE * 100)
                                        benchmark_sharpe = benchmark_excess_return / benchmark_vol
                                    else:
                                        benchmark_annualized_return = benchmark_return
                                        benchmark_sharpe = 0
                                    
                                    # Alpha (excess return) - FIX: use TOTAL returns to match displayed values
                                    alpha = portfolio_return - benchmark_return
                                    
                                    # Calculate beta (if enough data points)
                                    if len(period_returns) > 10 and len(benchmark_period_returns) > 10:
                                        # Align the returns
                                        aligned_returns = pd.DataFrame({
                                            'portfolio': period_returns,
                                            'benchmark': benchmark_period_returns
                                        }).dropna()
                                        
                                        if len(aligned_returns) > 10:
                                            covariance = aligned_returns['portfolio'].cov(aligned_returns['benchmark'])
                                            benchmark_variance = aligned_returns['benchmark'].var()
                                            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                                        else:
                                            beta = 0
                                    else:
                                        beta = 0
                                    
                                    performance_data.append({
                                        'Period': period_name,
                                        'Portfolio Return': portfolio_return,
                                        'Benchmark Return': benchmark_return,
                                        'Alpha': alpha,
                                        'Portfolio Vol': portfolio_vol,
                                        'Benchmark Vol': benchmark_vol,
                                        'Portfolio Sharpe': portfolio_sharpe,
                                        'Benchmark Sharpe': benchmark_sharpe,
                                        'Portfolio Max DD': portfolio_max_dd,
                                        'Benchmark Max DD': benchmark_max_dd,
                                        'Beta': beta
                                    })
                            
                            if performance_data:
                                perf_df = pd.DataFrame(performance_data)
                                
                                # Create professional formatted table
                                st.markdown("#### 📈 Returns Comparison")
                                
                                # Returns table
                                returns_display = perf_df[['Period', 'Portfolio Return', 'Benchmark Return', 'Alpha']].copy()
                                returns_display['Portfolio Return'] = returns_display['Portfolio Return'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
                                returns_display['Benchmark Return'] = returns_display['Benchmark Return'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
                                returns_display['Alpha'] = returns_display['Alpha'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
                                returns_display['Outperformance'] = perf_df['Alpha'].apply(
                                    lambda x: '✅ Yes' if pd.notna(x) and x > 0 else ('❌ No' if pd.notna(x) else 'N/A')
                                )
                                
                                # Color styling function
                                def color_alpha(val):
                                    if 'Yes' in str(val):
                                        return 'background-color: #c8e6c9; color: #2e7d32; font-weight: bold'
                                    elif 'No' in str(val):
                                        return 'background-color: #ffcdd2; color: #c62828; font-weight: bold'
                                    return ''
                                
                                styled_returns = returns_display.style.map(
                                    color_alpha,
                                    subset=['Outperformance']
                                )
                                
                                st.dataframe(styled_returns, use_container_width=True, hide_index=True)
                                
                                # Risk metrics table
                                st.markdown("#### 🛡️ Risk Metrics Comparison")
                                
                                risk_display = perf_df[['Period', 'Portfolio Vol', 'Benchmark Vol', 
                                                        'Portfolio Sharpe', 'Benchmark Sharpe',
                                                        'Portfolio Max DD', 'Benchmark Max DD', 'Beta']].copy()
                                
                                risk_display['Portfolio Vol'] = risk_display['Portfolio Vol'].apply(lambda x: f"{x:.2f}%")
                                risk_display['Benchmark Vol'] = risk_display['Benchmark Vol'].apply(lambda x: f"{x:.2f}%")
                                risk_display['Portfolio Sharpe'] = risk_display['Portfolio Sharpe'].apply(lambda x: f"{x:.2f}")
                                risk_display['Benchmark Sharpe'] = risk_display['Benchmark Sharpe'].apply(lambda x: f"{x:.2f}")
                                risk_display['Portfolio Max DD'] = risk_display['Portfolio Max DD'].apply(lambda x: f"{x:.2f}%")
                                risk_display['Benchmark Max DD'] = risk_display['Benchmark Max DD'].apply(lambda x: f"{x:.2f}%")
                                risk_display['Beta'] = risk_display['Beta'].apply(lambda x: f"{x:.2f}")
                                
                                st.dataframe(risk_display, use_container_width=True, hide_index=True)
                                
                                # Visual comparison chart - Portfolio Value Over Time
                                st.markdown("#### 📊 Portfolio vs Benchmark Value Over Time")

                                fig_comparison = go.Figure()

                                # Portfolio value line
                                fig_comparison.add_trace(go.Scatter(
                                    x=portfolio_value_series.index,
                                    y=portfolio_value_series.values,
                                    name='Your Portfolio',
                                    mode='lines',
                                    line=dict(color='#00ff88', width=2),
                                    hovertemplate='Date: %{x|%Y-%m-%d}<br>Portfolio: $%{y:,.0f}<extra></extra>'
                                ))

                                # Benchmark value line (scaled to same starting value)
                                if benchmark_value_series is not None and len(benchmark_value_series) > 0:
                                    fig_comparison.add_trace(go.Scatter(
                                        x=benchmark_value_series.index,
                                        y=benchmark_value_series.values,
                                        name='Benchmark',
                                        mode='lines',
                                        line=dict(color='#ff6b6b', width=2),
                                        hovertemplate='Date: %{x|%Y-%m-%d}<br>Benchmark: $%{y:,.0f}<extra></extra>'
                                    ))

                                # Add starting point annotation (single label)
                                fig_comparison.add_annotation(
                                    x=portfolio_value_series.index[0],
                                    y=portfolio_value_series.iloc[0],
                                    text=f"Start: ${portfolio_value_series.iloc[0]:,.0f}",
                                    showarrow=False,
                                    xanchor='left',
                                    yanchor='bottom',
                                    font=dict(size=12, color='white'),
                                    bgcolor='rgba(0,0,0,0.7)',
                                    borderpad=4
                                )

                                # Add ending point annotations in legend box style
                                final_portfolio = portfolio_value_series.iloc[-1]
                                final_benchmark = benchmark_value_series.iloc[-1] if benchmark_value_series is not None and len(benchmark_value_series) > 0 else 0

                                # Create annotation text box in top right
                                annotation_text = f"<b>Current Values:</b><br>Portfolio: ${final_portfolio:,.0f}<br>Benchmark: ${final_benchmark:,.0f}"

                                fig_comparison.add_annotation(
                                    xref="paper",
                                    yref="paper",
                                    x=0.02,
                                    y=0.98,
                                    text=annotation_text,
                                    showarrow=False,
                                    xanchor='left',
                                    yanchor='top',
                                    font=dict(size=11),
                                    bgcolor='rgba(0,0,0,0.8)',
                                    bordercolor='white',
                                    borderwidth=1,
                                    borderpad=8
                                )

                                fig_comparison.update_layout(
                                    template="plotly_dark",
                                    height=500,
                                    title="Portfolio vs Benchmark Value Since Inception",
                                    xaxis_title="Date",
                                    yaxis_title="Value ($) - Log Scale",
                                    hovermode='x unified',
                                    legend=dict(
                                        orientation="h", 
                                        yanchor="bottom", 
                                        y=1.02, 
                                        xanchor="right", 
                                        x=1,
                                        bgcolor='rgba(0,0,0,0.7)'
                                    ),
                                    yaxis=dict(
                                        type="log",
                                        tickprefix="$", 
                                        tickformat=",",
                                        gridcolor='rgba(128,128,128,0.2)'
                                    ),
                                    xaxis=dict(
                                        gridcolor='rgba(128,128,128,0.2)'
                                    )
                                )

                                st.plotly_chart(fig_comparison, use_container_width=True)
                                
                                # Key insights
                                st.markdown("#### 💡 Key Insights")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    outperformance_count = sum(1 for x in perf_df['Alpha'] if x > 0)
                                    total_periods = len(perf_df)
                                    win_rate = (outperformance_count / total_periods) * 100 if total_periods > 0 else 0
                                    
                                    st.metric(
                                        "Win Rate vs Benchmark",
                                        f"{outperformance_count}/{total_periods}",
                                        delta=f"{win_rate:.0f}%",
                                        help="Number of periods where portfolio outperformed"
                                    )
                                
                                with col2:
                                    avg_alpha = perf_df['Alpha'].dropna().mean() if perf_df['Alpha'].notna().any() else 0
                                    st.metric(
                                        "Average Alpha",
                                        f"{avg_alpha:+.2f}%",
                                        delta="vs Benchmark",
                                        help="Average excess return across all periods"
                                    )
                                
                                with col3:
                                    # Best performing period
                                    if perf_df['Alpha'].notna().any():
                                        best_period_idx = perf_df['Alpha'].idxmax()
                                        if pd.notna(best_period_idx):
                                            best_period = perf_df.loc[best_period_idx, 'Period']
                                            best_alpha = perf_df.loc[best_period_idx, 'Alpha']
                                            
                                            st.metric(
                                                "Best Alpha Period",
                                                best_period,
                                                delta=f"+{best_alpha:.2f}%",
                                                help="Period with highest outperformance"
                                            )
                                        else:
                                            st.metric(
                                                "Best Alpha Period",
                                                "N/A",
                                                help="No valid alpha data"
                                            )
                                    else:
                                        st.metric(
                                            "Best Alpha Period",
                                            "N/A",
                                            help="No valid alpha data"
                                        )
                                
                                # Risk-adjusted performance summary
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**📊 Portfolio Consistency**")
                                    
                                    # Show Sharpe ratios for different periods
                                    st.markdown(f"- Sharpe Ratio (Since Inception): **{sharpe_inception:.2f}**")
                                    if sharpe_5y is not None:
                                        st.markdown(f"- Sharpe Ratio (Last 5 Years): **{sharpe_5y:.2f}**")
                                    if sharpe_1y is not None:
                                        st.markdown(f"- Sharpe Ratio (Last 1 Year): **{sharpe_1y:.2f}**")
                                    
                                    # Compare inception Sharpe with benchmark
                                    benchmark_avg_sharpe = perf_df['Benchmark Sharpe'].dropna().mean() if perf_df['Benchmark Sharpe'].notna().any() else 0
                                    
                                    sharpe_comparison = "Better" if sharpe_inception > benchmark_avg_sharpe else "Worse"
                                    sharpe_color = "green" if sharpe_inception > benchmark_avg_sharpe else "red"
                                    
                                    st.markdown(f"- Risk-Adjusted Performance: :{sharpe_color}[**{sharpe_comparison}**]")
                                
                                with col2:
                                    st.markdown("**🎯 Risk Profile**")
                                    
                                    avg_beta = perf_df['Beta'].dropna().mean() if perf_df['Beta'].notna().any() else 1.0
                                    avg_vol = perf_df['Portfolio Vol'].dropna().mean() if perf_df['Portfolio Vol'].notna().any() else 0
                                    benchmark_avg_vol = perf_df['Benchmark Vol'].dropna().mean() if perf_df['Benchmark Vol'].notna().any() else 0
                                    
                                    if avg_beta < 0.8:
                                        risk_profile = "Conservative (Low Beta)"
                                    elif avg_beta < 1.2:
                                        risk_profile = "Moderate (Market Beta)"
                                    else:
                                        risk_profile = "Aggressive (High Beta)"
                                    
                                    st.markdown(f"- Average Beta: **{avg_beta:.2f}**")
                                    st.markdown(f"- Risk Profile: **{risk_profile}**")
                                    st.markdown(f"- Avg Volatility: **{avg_vol:.1f}%** vs **{benchmark_avg_vol:.1f}%**")
                                
                                # Executive summary
                                with st.expander("📋 Executive Summary for High-Net-Worth Clients"):
                                    inception_return_row = perf_df[perf_df['Period'] == 'Since Inception']
                                    inception_benchmark = inception_return_row['Benchmark Return'].values[0] if len(inception_return_row) > 0 else 0
                                    
                                    st.markdown(f"""
                                    ### Portfolio Performance Report
                                    **Reporting Period**: {investment_datetime.strftime('%B %d, %Y')} - {current_date.strftime('%B %d, %Y')}
                                    
                                    ---
                                    
                                    #### Performance Highlights:
                                    
                                    - **Total Return**: {investment_return:+.2f}% vs Benchmark: {inception_benchmark:+.2f}%
                                    - **Outperformance Rate**: {win_rate:.0f}% of measured periods
                                    - **Average Excess Return (Alpha)**: {avg_alpha:+.2f}%
                                    - **Risk-Adjusted Performance**: Sharpe Ratio of {sharpe_inception:.2f} vs {benchmark_avg_sharpe:.2f} for benchmark
                                    
                                    #### Risk Characteristics:
                                    
                                    - **Portfolio Beta**: {avg_beta:.2f} ({"Less volatile than" if avg_beta < 1 else "More volatile than"} market)
                                    - **Average Volatility**: {avg_vol:.1f}% annualized
                                    - **Maximum Drawdown**: {portfolio_max_dd:.2f}%
                                    
                                    #### Investment Strategy Assessment:
                                    
                                    {f"✅ The portfolio has demonstrated consistent outperformance with a {win_rate:.0f}% success rate against the benchmark." if win_rate >= 60 else f"⚠️ The portfolio has underperformed the benchmark in {100-win_rate:.0f}% of periods. Consider rebalancing strategy."}
                                    
                                    {f"✅ Superior risk-adjusted returns with a Sharpe ratio of {sharpe_inception:.2f}, indicating efficient use of risk." if sharpe_inception > benchmark_avg_sharpe else f"⚠️ Risk-adjusted performance could be improved. Current Sharpe ratio of {sharpe_inception:.2f} trails benchmark."}
                                    
                                    {f"✅ Portfolio exhibits lower volatility ({avg_vol:.1f}%) compared to market ({benchmark_avg_vol:.1f}%), suitable for risk-averse investors." if avg_vol < benchmark_avg_vol else f"ℹ️ Portfolio shows higher volatility ({avg_vol:.1f}%) than market ({benchmark_avg_vol:.1f}%), indicating an aggressive stance."}
                                    
                                    ---
                                    
                                    **Recommendation**: {"Continue current strategy with regular monitoring." if avg_alpha > 0 and sharpe_inception > benchmark_avg_sharpe else "Review asset allocation and consider tactical adjustments to improve risk-adjusted returns."}
                                    
                                    *This analysis is for informational purposes only and does not constitute investment advice.*
                                    """)
                                
                                # =============================================================
                                # RETURNS DISTRIBUTION COMPARISON
                                # =============================================================
                                st.markdown("---")
                                st.subheader("📊 Returns Distribution: Portfolio vs Benchmark")
                                st.markdown("*Compare the statistical distribution of daily returns between your portfolio and the Custom Benchmark*")
                                
                                # Calculate daily returns
                                portfolio_daily_returns = portfolio_value_series.pct_change().dropna() * 100
                                benchmark_daily_returns = benchmark_value_series.pct_change().dropna() * 100
                                
                                # Align the returns by common dates
                                common_dates = portfolio_daily_returns.index.intersection(benchmark_daily_returns.index)
                                portfolio_returns_aligned = portfolio_daily_returns.loc[common_dates]
                                benchmark_returns_aligned = benchmark_daily_returns.loc[common_dates]
                                
                                # Calculate means for annotations
                                portfolio_mean = portfolio_returns_aligned.mean()
                                benchmark_mean = benchmark_returns_aligned.mean()
                                portfolio_std = portfolio_returns_aligned.std()
                                benchmark_std = benchmark_returns_aligned.std()
                                
                                # Two separate histograms side by side
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Portfolio histogram
                                    fig_portfolio = go.Figure()
                                    
                                    fig_portfolio.add_trace(go.Histogram(
                                        x=portfolio_returns_aligned,
                                        name='Portfolio',
                                        marker_color='#00ff88',
                                        nbinsx=40,
                                        histnorm='probability density',
                                        opacity=0.85
                                    ))
                                    
                                    # Add mean line
                                    fig_portfolio.add_vline(
                                        x=portfolio_mean,
                                        line_dash="dash",
                                        line_color="white",
                                        line_width=2,
                                        annotation_text=f"μ = {portfolio_mean:.3f}%",
                                        annotation_position="top"
                                    )
                                    
                                    # Add +/- 1 std dev lines
                                    fig_portfolio.add_vline(
                                        x=portfolio_mean - portfolio_std,
                                        line_dash="dot",
                                        line_color="rgba(255,255,255,0.5)",
                                        line_width=1
                                    )
                                    fig_portfolio.add_vline(
                                        x=portfolio_mean + portfolio_std,
                                        line_dash="dot",
                                        line_color="rgba(255,255,255,0.5)",
                                        line_width=1
                                    )
                                    
                                    fig_portfolio.update_layout(
                                        template="plotly_dark",
                                        height=400,
                                        title=f"📈 Portfolio Returns<br><sub>σ = {portfolio_std:.3f}%</sub>",
                                        xaxis_title="Daily Return (%)",
                                        yaxis_title="Density",
                                        showlegend=False
                                    )
                                    
                                    st.plotly_chart(fig_portfolio, use_container_width=True)
                                
                                with col2:
                                    # Benchmark histogram
                                    fig_benchmark = go.Figure()
                                    
                                    fig_benchmark.add_trace(go.Histogram(
                                        x=benchmark_returns_aligned,
                                        name='Benchmark',
                                        marker_color='#ff6b6b',
                                        nbinsx=40,
                                        histnorm='probability density',
                                        opacity=0.85
                                    ))
                                    
                                    # Add mean line
                                    fig_benchmark.add_vline(
                                        x=benchmark_mean,
                                        line_dash="dash",
                                        line_color="white",
                                        line_width=2,
                                        annotation_text=f"μ = {benchmark_mean:.3f}%",
                                        annotation_position="top"
                                    )
                                    
                                    # Add +/- 1 std dev lines
                                    fig_benchmark.add_vline(
                                        x=benchmark_mean - benchmark_std,
                                        line_dash="dot",
                                        line_color="rgba(255,255,255,0.5)",
                                        line_width=1
                                    )
                                    fig_benchmark.add_vline(
                                        x=benchmark_mean + benchmark_std,
                                        line_dash="dot",
                                        line_color="rgba(255,255,255,0.5)",
                                        line_width=1
                                    )
                                    
                                    fig_benchmark.update_layout(
                                        template="plotly_dark",
                                        height=400,
                                        title=f"📊 Benchmark Returns<br><sub>σ = {benchmark_std:.3f}%</sub>",
                                        xaxis_title="Daily Return (%)",
                                        yaxis_title="Density",
                                        showlegend=False
                                    )
                                    
                                    st.plotly_chart(fig_benchmark, use_container_width=True)
                                
                                # Distribution statistics table
                                st.markdown("#### 📈 Distribution Statistics")
                                
                                # Calculate detailed statistics
                                portfolio_stats = {
                                    'Mean (%)': portfolio_returns_aligned.mean(),
                                    'Std Dev (%)': portfolio_returns_aligned.std(),
                                    'Skewness': stats.skew(portfolio_returns_aligned),
                                    'Kurtosis': stats.kurtosis(portfolio_returns_aligned),
                                    'Min (%)': portfolio_returns_aligned.min(),
                                    '25th Percentile (%)': portfolio_returns_aligned.quantile(0.25),
                                    'Median (%)': portfolio_returns_aligned.median(),
                                    '75th Percentile (%)': portfolio_returns_aligned.quantile(0.75),
                                    'Max (%)': portfolio_returns_aligned.max(),
                                    'Positive Days (%)': (portfolio_returns_aligned > 0).sum() / len(portfolio_returns_aligned) * 100
                                }
                                
                                benchmark_stats = {
                                    'Mean (%)': benchmark_returns_aligned.mean(),
                                    'Std Dev (%)': benchmark_returns_aligned.std(),
                                    'Skewness': stats.skew(benchmark_returns_aligned),
                                    'Kurtosis': stats.kurtosis(benchmark_returns_aligned),
                                    'Min (%)': benchmark_returns_aligned.min(),
                                    '25th Percentile (%)': benchmark_returns_aligned.quantile(0.25),
                                    'Median (%)': benchmark_returns_aligned.median(),
                                    '75th Percentile (%)': benchmark_returns_aligned.quantile(0.75),
                                    'Max (%)': benchmark_returns_aligned.max(),
                                    'Positive Days (%)': (benchmark_returns_aligned > 0).sum() / len(benchmark_returns_aligned) * 100
                                }
                                    
                                stats_df = pd.DataFrame({
                                    'Metric': list(portfolio_stats.keys()),
                                    'Portfolio': [f"{v:.4f}" if 'Skewness' in k or 'Kurtosis' in k else f"{v:.2f}" for k, v in portfolio_stats.items()],
                                    'Benchmark': [f"{v:.4f}" if 'Skewness' in k or 'Kurtosis' in k else f"{v:.2f}" for k, v in benchmark_stats.items()]
                                })
                                
                                # Determine which is better for each metric
                                def highlight_better(row):
                                    metric = row['Metric']
                                    port_val = float(row['Portfolio'])
                                    sp_val = float(row['Benchmark'])
                                    
                                    # Higher is better: Mean, Median, Max, Positive Days, 75th percentile
                                    # Lower is better: Std Dev, Min (less negative), Kurtosis
                                    # Closer to 0 is better: Skewness
                                    
                                    higher_better = ['Mean (%)', 'Median (%)', 'Max (%)', 'Positive Days (%)', '75th Percentile (%)']
                                    lower_better = ['Std Dev (%)', 'Kurtosis']
                                    
                                    if metric in higher_better:
                                        if port_val > sp_val:
                                            return ['', 'background-color: #a5d6a7; color: #1b5e20', '']
                                        elif sp_val > port_val:
                                            return ['', '', 'background-color: #a5d6a7; color: #1b5e20']
                                    elif metric in lower_better:
                                        if port_val < sp_val:
                                            return ['', 'background-color: #a5d6a7; color: #1b5e20', '']
                                        elif sp_val < port_val:
                                            return ['', '', 'background-color: #a5d6a7; color: #1b5e20']
                                    elif metric == 'Skewness':
                                        if abs(port_val) < abs(sp_val):
                                            return ['', 'background-color: #a5d6a7; color: #1b5e20', '']
                                        elif abs(sp_val) < abs(port_val):
                                            return ['', '', 'background-color: #a5d6a7; color: #1b5e20']
                                    elif metric == 'Min (%)':
                                        if port_val > sp_val:  # Less negative is better
                                            return ['', 'background-color: #a5d6a7; color: #1b5e20', '']
                                        elif sp_val > port_val:
                                            return ['', '', 'background-color: #a5d6a7; color: #1b5e20']
                                    
                                    return ['', '', '']
                                
                                styled_stats = stats_df.style.apply(highlight_better, axis=1)
                                st.dataframe(styled_stats, use_container_width=True, hide_index=True)
                                
                                # Insights
                                st.markdown("#### 💡 Distribution Insights")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    skew_diff = portfolio_stats['Skewness'] - benchmark_stats['Skewness']
                                    if portfolio_stats['Skewness'] > 0:
                                        skew_analysis = "✅ Positive skew (more upside potential)"
                                    elif portfolio_stats['Skewness'] < -0.5:
                                        skew_analysis = "⚠️ Negative skew (tail risk)"
                                    else:
                                        skew_analysis = "✓ Roughly symmetric distribution"
                                    
                                    st.info(f"**Skewness Analysis**\n\n{skew_analysis}")
                                
                                with col2:
                                    kurt_diff = portfolio_stats['Kurtosis'] - benchmark_stats['Kurtosis']
                                    kurt_val = portfolio_stats['Kurtosis']
                                    
                                    if kurt_val > 10:
                                        kurt_analysis = f"⚠️ Very high kurtosis ({kurt_val:.1f})\n\nThis indicates extreme outlier days in your data. Even if your typical days are calm (as shown in the histogram), a few large moves dramatically increase kurtosis."
                                    elif kurt_val > 3:
                                        kurt_analysis = "⚠️ Fat tails (extreme events present)"
                                    elif kurt_val < -1:
                                        kurt_analysis = "✅ Thin tails (fewer extreme movements)"
                                    else:
                                        kurt_analysis = "✓ Normal-like distribution"
                                    
                                    st.info(f"**Kurtosis Analysis**\n\n{kurt_analysis}")
                                
                                with col3:
                                    win_rate_diff = portfolio_stats['Positive Days (%)'] - benchmark_stats['Positive Days (%)']
                                    if win_rate_diff > 2:
                                        win_analysis = f"✅ Higher win rate (+{win_rate_diff:.1f}%)"
                                    elif win_rate_diff < -2:
                                        win_analysis = f"⚠️ Lower win rate ({win_rate_diff:.1f}%)"
                                    else:
                                        win_analysis = "✓ Similar win rate to benchmark"
                                    
                                    st.info(f"**Win Rate Analysis**\n\n{win_analysis}")
                                
                            else:
                                st.warning("⚠️ Not enough historical data to calculate period performance metrics.")
                        
                        # =============================================================
                        # PORTFOLIO VALUE OVER TIME
                        # =============================================================
                        st.subheader("💰 Portfolio Value Over Time")
                        
                        # Show rebalancing comparison if enabled
                        if st.session_state.rebalancing_enabled and portfolio_value_no_rebal is not None:
                            st.markdown("**Rebalancing Impact Comparison**")
                            
                            # Calculate final values
                            final_with_rebal = portfolio_value_series.iloc[-1]
                            final_without_rebal = portfolio_value_no_rebal.iloc[-1]
                            difference = final_with_rebal - final_without_rebal
                            diff_pct = (difference / final_without_rebal) * 100 if final_without_rebal > 0 else 0
                            
                            col_comp1, col_comp2, col_comp3 = st.columns(3)
                            with col_comp1:
                                st.metric("With Rebalancing", format_currency(final_with_rebal), 
                                         delta=None)
                            with col_comp2:
                                st.metric("Without Rebalancing", format_currency(final_without_rebal),
                                         delta=None)
                            with col_comp3:
                                st.metric("Difference", format_currency(difference),
                                         delta=f"{diff_pct:+.2f}%")
                        
                        fig_value = go.Figure()
                        
                        # Get the actual starting value (first value in the series)
                        actual_starting_value = portfolio_value_series.iloc[0] if len(portfolio_value_series) > 0 else initial_capital
                        
                        # Add portfolio line
                        fig_value.add_trace(go.Scatter(
                        x=portfolio_value_series.index,
                        y=portfolio_value_series,
                        mode='lines',  # Changed from 'lines+markers'
                        name='Portfolio Value (With Rebalancing)' if st.session_state.rebalancing_enabled else 'Portfolio Value',
                        line=dict(color='#00ff88', width=2.5),
                        hovertemplate='<b>Portfolio</b><br>Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
                        ))
                        
                        # Add comparison line if rebalancing enabled
                        if st.session_state.rebalancing_enabled and portfolio_value_no_rebal is not None:
                            fig_value.add_trace(go.Scatter(
                                x=portfolio_value_no_rebal.index,
                                y=portfolio_value_no_rebal,
                                mode='lines',
                                name='Portfolio Value (Without Rebalancing)',
                                line=dict(color='#ff6b6b', width=2, dash='dash'),
                                hovertemplate='<b>No Rebalancing</b><br>Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
                            ))
                            
                            # Add markers for rebalancing events
                            if rebalancing_events:
                                rebal_dates = [event['date'] for event in rebalancing_events]
                                rebal_values = [portfolio_value_series.loc[date] if date in portfolio_value_series.index 
                                              else None for date in rebal_dates]
                                # Filter out None values
                                rebal_dates_clean = [d for d, v in zip(rebal_dates, rebal_values) if v is not None]
                                rebal_values_clean = [v for v in rebal_values if v is not None]
                                
                                if rebal_dates_clean:
                                    fig_value.add_trace(go.Scatter(
                                        x=rebal_dates_clean,
                                        y=rebal_values_clean,
                                        mode='markers',
                                        name='Rebalancing Events',
                                        marker=dict(color='yellow', size=10, symbol='star'),
                                        hovertemplate='<b>Rebalanced</b><br>Date: %{x}<extra></extra>'
                                    ))
                        
                        # Add percentage return line on secondary axis
                        portfolio_returns_pct = ((portfolio_value_series / actual_starting_value) - 1) * 100
                        
                        # Add initial capital line - use actual starting value for reference
                        fig_value.add_hline(
                            y=actual_starting_value,
                            line_dash="dash",
                            line_color="gray",
                            opacity=0.5,
                        )
                        
                        fig_value.update_layout(
                            template="plotly_dark",
                            height=450,
                            xaxis_title="Date",
                            yaxis_title="Portfolio Value ($) - Log Scale",
                            hovermode='x unified',
                            yaxis=dict(
                                type="log",  # Enable logarithmic scale
                                tickformat="$,.0f",
                                gridcolor='rgba(128,128,128,0.2)',
                                showline=False,
                                zeroline=False
                            ),
                            xaxis=dict(
                                gridcolor='rgba(128,128,128,0.2)',
                                showline=False,
                                zeroline=False
                            ),
                            showlegend=True,
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01,
                                bgcolor="rgba(0,0,0,0.5)"
                            ),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(fig_value, use_container_width=True)
                        
                        # Show rebalancing events table if enabled
                        if st.session_state.rebalancing_enabled and rebalancing_events:
                            with st.expander(f"📋 View {len(rebalancing_events)} Rebalancing Events (Only Assets Exceeding Threshold)"):
                                for i, event in enumerate(rebalancing_events, 1):
                                    assets_count = event.get('assets_count', len(event['actions']))
                                    st.markdown(f"**Event {i}: {event['date'].strftime('%Y-%m-%d')}** - {assets_count} asset(s) rebalanced")
                                    for action in event['actions']:
                                        st.text(f"  • {action}")
                                    if i < len(rebalancing_events):
                                        st.markdown("---")


                        # =============================================================
                        # SECTOR & GEOGRAPHIC ATTRIBUTION
                        # =============================================================
                        st.subheader("🌍 Portfolio Attribution Analysis")
                        st.markdown("*Understand your portfolio exposure by sector and geography*")
                        
                        if use_transactions and holdings_over_time is not None:
                            # Get current holdings
                            current_holdings = {}
                            current_prices_dict = {}
                            
                            for ticker in analysis_portfolio.keys():
                                if ticker in holdings_over_time.columns:
                                    shares = holdings_over_time[ticker].iloc[-1]
                                    if shares > 0 and ticker in current_prices:
                                        current_holdings[ticker] = shares
                                        current_prices_dict[ticker] = current_prices[ticker]
                            
                            # Add cash
                            if cash_over_time is not None:
                                current_holdings['CASH'] = cash_over_time.iloc[-1]
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("#### 📊 Sector Exposure")
                                
                                sector_df = calculate_sector_exposure(current_holdings, current_prices_dict)
                                
                                if not sector_df.empty:
                                    # Horizontal bar chart (better than pie)
                                    fig_sector = px.bar(
                                        sector_df.sort_values('Weight', ascending=True),
                                        x='Weight',
                                        y='Sector',
                                        orientation='h',
                                        title='Portfolio by Sector',
                                        color='Weight',
                                        color_continuous_scale='Viridis',
                                        text='Weight'
                                    )
                                    fig_sector.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
                                    fig_sector.update_layout(
                                        template="plotly_dark", 
                                        height=450,
                                        xaxis_title="Weight (%)",
                                        yaxis_title="",
                                        showlegend=False
                                    )
                                    st.plotly_chart(fig_sector, use_container_width=True)
                                    
                                    # Table
                                    display_sector = sector_df.copy()
                                    display_sector['Value'] = display_sector['Value'].apply(lambda x: f"${x:,.0f}")
                                    display_sector['Weight'] = display_sector['Weight'].apply(lambda x: f"{x:.2f}%")
                                    st.dataframe(display_sector, use_container_width=True, hide_index=True)
                            
                            with col2:
                                st.markdown("#### 🗺️ Geographic Exposure")
                                
                                geo_df = calculate_geographic_exposure(current_holdings, current_prices_dict)
                                
                                if not geo_df.empty:
                                    # Horizontal bar chart (better than pie)
                                    fig_geo = px.bar(
                                        geo_df.sort_values('Weight', ascending=True),
                                        x='Weight',
                                        y='Geography',
                                        orientation='h',
                                        title='Portfolio by Geography',
                                        color='Weight',
                                        color_continuous_scale='Blues',
                                        text='Weight'
                                    )
                                    fig_geo.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
                                    fig_geo.update_layout(
                                        template="plotly_dark", 
                                        height=450,
                                        xaxis_title="Weight (%)",
                                        yaxis_title="",
                                        showlegend=False
                                    )
                                    st.plotly_chart(fig_geo, use_container_width=True)
                                    
                                    # Table
                                    display_geo = geo_df.copy()
                                    display_geo['Value'] = display_geo['Value'].apply(lambda x: f"${x:,.0f}")
                                    display_geo['Weight'] = display_geo['Weight'].apply(lambda x: f"{x:.2f}%")
                                    st.dataframe(display_geo, use_container_width=True, hide_index=True)
                        
                        # =============================================================
                        # PROFESSIONAL PDF REPORT GENERATION (Quarterly only)
                        # =============================================================
                        st.subheader("📄 Professional PDF Reports")
                        st.markdown("*Generate client-ready performance reports*")
                        
                        
                        if st.button("📊 Generate Quarterly Report", use_container_width=True):
                                
                            # Prepare holdings data with full information
                            # CRITICAL: Use current_weights (from weights_over_time) which correctly includes cash
                            holdings_data = []
                            if use_transactions and holdings_over_time is not None:
                                for ticker in analysis_portfolio.keys():
                                    if ticker in holdings_over_time.columns:
                                        shares = holdings_over_time[ticker].iloc[-1]
                                        if shares > 0 and ticker in current_prices:
                                            # Calculate average cost from BUY transactions
                                            buy_trans = [t for t in st.session_state.transactions 
                                                        if t['type'] == 'BUY' and t['ticker'] == ticker]
                                            total_cost = sum(t['amount'] for t in buy_trans)
                                            total_shares_bought = sum(t['shares'] for t in buy_trans)
                                            avg_cost = total_cost / total_shares_bought if total_shares_bought > 0 else 0
                                            
                                            current_price = current_prices[ticker]
                                            value = shares * current_price
                                            gain_loss = value - (shares * avg_cost)
                                            return_pct = (gain_loss / (shares * avg_cost)) * 100 if avg_cost > 0 else 0
                                            
                                            # USE CORRECT WEIGHT from current_weights (not recalculated)
                                            weight = current_weights.get(ticker, 0) * 100
                                            
                                            holdings_data.append({
                                                'Ticker': ticker,
                                                'Shares': shares,
                                                'Avg Cost': avg_cost,
                                                'Current Price': current_price,
                                                'Market Value': value,
                                                'Gain/Loss': format_currency(gain_loss),
                                                'Return': format_percentage(return_pct),
                                                'Weight': weight
                                            })
                            
                            # Use variables from outer scope (already calculated)
                            # total_deposits, total_invested are from capital flows calculation
                            
                            # FIX: Rebuild attribution_data for PDF using effective_transactions
                            pdf_attribution_data = []
                            for ticker in analysis_portfolio.keys():
                                if ticker in holdings_over_time.columns:
                                    shares = holdings_over_time[ticker].iloc[-1]
                                    if shares > 0 and ticker in current_prices:
                                        # FIX: Use effective_transactions (includes rebalancing)
                                        buy_trans = [t for t in effective_transactions 
                                                    if t['type'] == 'BUY' and t['ticker'] == ticker]
                                        sell_trans = [t for t in effective_transactions 
                                                     if t['type'] == 'SELL' and t['ticker'] == ticker]
                                        
                                        total_buys = sum(t['amount'] for t in buy_trans)
                                        total_shares_bought = sum(t['shares'] for t in buy_trans)
                                        
                                        # FIX: Consistent avg cost calculation
                                        avg_cost = total_buys / total_shares_bought if total_shares_bought > 0 else 0
                                        
                                        # FIX: Cost basis of current position
                                        cost_basis_current = shares * avg_cost
                                        
                                        current_market_value = shares * current_prices[ticker]
                                        dollar_gain = current_market_value - cost_basis_current
                                        pct_return = (dollar_gain / cost_basis_current) * 100 if cost_basis_current > 0 else 0
                                        
                                        pdf_attribution_data.append({
                                            'Asset': ticker,
                                            'Invested': cost_basis_current,  # FIX: Use cost basis
                                            'Current Value': current_market_value,
                                            'Gain/Loss ($)': dollar_gain,
                                            'Return (%)': pct_return
                                        })
                            
                            # Add put options to attribution
                            if put_positions:
                                total_put_cost = sum(put.get('total_cost', 0) for put in put_positions)
                                total_put_value = 0
                                for put in put_positions:
                                    if put['sell_date']:
                                        total_put_value += put.get('total_proceeds', 0)
                                    else:
                                        underlying = put['underlying']
                                        if underlying in current_prices:
                                            # PUT intrinsic = strike - underlying
                                            intrinsic = max(0, put['strike'] - current_prices[underlying])
                                            total_put_value += intrinsic * put['contracts'] * put['multiplier']
                                
                                if total_put_cost > 0:
                                    put_gain = total_put_value - total_put_cost
                                    put_return = (put_gain / total_put_cost) * 100
                                    pdf_attribution_data.append({
                                        'Asset': 'Put Options',
                                        'Invested': total_put_cost,
                                        'Current Value': total_put_value,
                                        'Gain/Loss ($)': put_gain,
                                        'Return (%)': put_return
                                    })
                            

                            # Calculate sector and geographic exposure for PDF
                            if use_transactions and holdings_over_time is not None:
                                current_holdings_dict = {}
                                current_prices_dict = {}
                                for ticker in analysis_portfolio.keys():
                                    if ticker in holdings_over_time.columns:
                                        shares = holdings_over_time[ticker].iloc[-1]
                                        if shares > 0 and ticker in current_prices:
                                            current_holdings_dict[ticker] = shares
                                            current_prices_dict[ticker] = current_prices[ticker]
                                
                                # Add cash
                                if cash_over_time is not None:
                                    current_holdings_dict['CASH'] = cash_over_time.iloc[-1]
                                
                                sector_exposure_pdf = calculate_sector_exposure(current_holdings_dict, current_prices_dict)
                                geographic_exposure_pdf = calculate_geographic_exposure(current_holdings_dict, current_prices_dict)
                            else:
                                sector_exposure_pdf = pd.DataFrame()
                                geographic_exposure_pdf = pd.DataFrame()

                            report_data = {
                                'start_date': investment_datetime.strftime('%B %d, %Y'),
                                'end_date': (datetime.now() - timedelta(days=1)).strftime('%B %d, %Y'),
                                'current_value': current_value,
                                'total_deposits': total_deposits,
                                'total_invested': total_invested,
                                'total_return': investment_return,
                                'annual_return': annual_return,
                                'sharpe_ratio': sharpe_inception if 'sharpe_inception' in locals() else portfolio_risk_metrics.get('sharpe_ratio', 0),
                                'max_drawdown': portfolio_risk_metrics.get('max_drawdown', 0),
                                'volatility': portfolio_risk_metrics.get('annual_volatility', 0),
                                'holdings': pd.DataFrame(holdings_data) if holdings_data else pd.DataFrame(),
                                'portfolio_value_series': portfolio_value_series,
                                'benchmark_value_series': benchmark_value_series if 'benchmark_value_series' in locals() else None,
                                'attribution_data': pdf_attribution_data,
                                'sector_exposure': sector_exposure_pdf,
                                'geographic_exposure': geographic_exposure_pdf
                            }
                            
                            # Generate PDF
                            if selected_client_id:
                                pdf_path = CLIENTS_DIR / selected_client_id / f"portfolio_report_{datetime.now().strftime('%Y%m%d')}.pdf"
                            else:
                                pdf_path = DATA_DIR / f"portfolio_report_{datetime.now().strftime('%Y%m%d')}.pdf"
                            
                            with st.spinner("Generating PDF report..."):
                                if generate_pdf_report(report_data, pdf_path):
                                    with open(pdf_path, 'rb') as f:
                                        st.download_button(
                                            label="📥 Download Report",
                                            data=f,
                                            file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                                            mime="application/pdf"
                                        )
                                    st.success("✅ Report generated successfully!")
                                else:
                                    st.error("❌ Failed to generate report")
                        
                        # DOWNLOAD SECTION
                        # =============================================================
                        st.subheader("💾 Export Portfolio Data")
                        
                        export_data = {
                            'Date': weights_over_time.index,
                            'Portfolio Value': portfolio_value_series.values
                        }
                        
                        for ticker in weights_over_time.columns:
                            export_data[f'{ticker} Weight (%)'] = weights_over_time[ticker].values * 100
                            # Only add price if ticker exists in prices_df
                            if ticker in prices_df.columns:
                                export_data[f'{ticker} Price'] = prices_df[ticker].values
                        
                        export_df = pd.DataFrame(export_data)
                        csv = export_df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download Portfolio Tracking Data (CSV)",
                            data=csv,
                            file_name=f"portfolio_tracking_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                        

# =============================================================================
# FOOTER
# =============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("*Data provided by Yahoo Finance*")
st.sidebar.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S' )}*")
st.sidebar.markdown("---")

# streamlit run stock_analytics.py --> Mettre ca dans le terminal