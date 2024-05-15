import streamlit as st
from funcs import *

st.set_page_config(
    page_title="Crypto portfolio analysis",
    page_icon="📈",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

import pandas as pd
cryptos = [
    'Arbitrum', 'Bitcoin Cash', 'BNB',
    'Cardano', 'Chainlink', 'Ethereum',
    'Fantom', 'Mantle', 'Optimism',
    'Solana', 'Toncoin', 'XRP'
]
container = {}
for coin in cryptos:
    filepath = f"./data/{coin}_01.05.2022-01.05.2024_historical_data_coinmarketcap.csv"
    temp = pd.read_csv(filepath, sep = ';')
    temp['name'] = coin
    container[coin] = temp

dt_rt_container=[]
for coin, dt in container.items():
    dt_rt = transformer_returns(dt)
    dt_rt.columns = [coin]
    dt_rt_container.append(dt_rt)
returns = pd.concat(dt_rt_container, axis = 1)

dt_price_container=[]
for coin, dt in container.items():
    dt_price = transformer_prices(dt)
    dt_price.columns = [coin]
    dt_price_container.append(dt_price)
prices = pd.concat(dt_price_container, axis = 1)

st.title("Cryptocurrencies research")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    'Assumptions', 'Portfolio', 'Correlations', 'Volatility', 'Sharpe Ratio', 'VaR', 'DrawDown',
    'Copula-based VaR', 'Clustering'])

with tab1:
    import numpy as np
    invested = st.number_input("Choose investment amount, USD", 0, 1000000, value = 5000)
    #invested = 1000000
    #weights = np.array([.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1])
    option = st.selectbox(
    "Choose your Risk Appetite",
    ("Aggressive", "Medium", "LowRisk"))
    if option == 'Aggressive':
        weights = [0.05] * len(cryptos)
        weights[cryptos.index('Bitcoin Cash')] = 0.20  # Adjust index as necessary
        weights[cryptos.index('Solana')] = 0.20
        weights[cryptos.index('Toncoin')] = 0.15
        weights = np.array(weights)
    elif option == 'Medium':
        weights = [1/12] * 12
        weights = np.array(weights)
    else:
        weights = np.array([1/(len(cryptos)-2) * 0.70] * len(cryptos))  # distribute remaining 70% equally among others
        weights[cryptos.index('Ethereum')] = 0.15
        weights[cryptos.index('BNB')] = 0.15


with tab2:

    fr__ = pd.DataFrame(
        {
            "coin":cryptos,
            "weights":weights,
            "invested":weights.dot(invested)
            }
            )
    st.dataframe(fr__, hide_index=True, use_container_width = True)

    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for coin in cryptos:
        ax.plot(returns.index, returns[coin]*100, label = coin)
    plt.title("daily returns")
    plt.xlabel('Day #')
    plt.ylabel('return %')
    plt.legend(loc="upper right",  prop={'size': 5})
    plt.show()
    st.pyplot(fig=fig)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for coin in cryptos:
        ax.plot(returns.index, returns[coin].cumsum()*100, label = coin)
    plt.title("cumsum returns")
    plt.xlabel('Day #')
    plt.ylabel('return %')
    plt.legend(loc="upper right",  prop={'size': 5})
    plt.show()
    st.pyplot(fig=fig)

    col1, col2, col3 = st.columns(3)
    with col1:
        avg_rets = returns.mean()
        portfolio_mean = avg_rets.dot(weights)
        portfolio_mean = portfolio_mean.round(4)
        st.metric(label='portfolio daily return', value = portfolio_mean)
    with col2:
        st.metric(label='Cryptos in Portfolio', value = 12)
    with col3:
        cov_matrix=returns.cov()
        port_stdev= np.sqrt(weights.T.dot(cov_matrix).dot(weights)).round(4)
        st.metric(label='portfolio daily volatility', value = port_stdev)
        

    # dt_container = []
    # for coin, dt in container.items():
    #     dt = dt.loc[pd.to_datetime(dt.timestamp).argmax()].to_frame().T #latest date
    #     dt = dt[['name', 'close', 'volume', 'marketCap']] #correct cols
    #     dt_container.append(dt)
    # fr_ = pd.concat(dt_container, axis=0)
    # st.dataframe(fr_, hide_index=True, use_container_width = True)

with tab3:
    # Compute the correlation matrix
    if 'portfolio' in returns.columns and 'invested' in returns.columns:
        returns = returns.drop(['portfolio', 'invested'], axis = 1)

    import seaborn as sns
    correlation_matrix = returns.corr()
    
    # Plot a heatmap of the correlations
    fig=plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.05)
    plt.title('Correlation matrix of cryptocurrency returns')
    st.pyplot(fig)
with tab4:
    returns['portfolio'] = returns.dot(weights)
    fig = plt.figure()
    plt.title("Volatility")
    sorted_std = returns.std().sort_values(ascending=False)
    ax = sorted_std.plot.bar()
    portfolio_index = sorted_std.index.get_loc('portfolio')
    ax.patches[portfolio_index].set_color('r')
    st.pyplot(fig)
    returns = returns.drop('portfolio', axis=1)


with tab5:
    returns['portfolio'] = returns.dot(weights)
    def sharpe_ratio(return_series, N, rf):
            mean = return_series.mean() * N -rf
            sigma = return_series.std() * np.sqrt(N)
            return mean/sigma
    N = 255
    rf = 0.01
    sharpes = returns.apply(sharpe_ratio, args=(N, rf), axis = 0).round(4)

    fig = plt.figure()
    plt.title('Sharpe Ratio')
    sharpes = sharpes.sort_values(ascending=True)
    ax = sharpes.plot.bar(color='blue')  # Set default color to blue
    # Highlight "portfolio" in red
    portfolio_index = sharpes.index.get_loc('portfolio')  # Get the index of 'portfolio'
    ax.patches[portfolio_index].set_color('red')
    st.pyplot(fig)

    returns = returns.drop('portfolio', axis=1)
    
    "Sharpe Ratio describes how much excess return you receive for the volatility of holding your assets. A higher Sharpe Ratio indicates that the asset has better risk-adjusted performance."

with tab6:

    conf_sl = st.slider('Choose Confidence', 95, 99, step = 1, value = 95)

    returns['portfolio'] = returns.dot(weights)
    returns['invested'] = returns['portfolio']*invested + invested
    
    mu = returns['invested'].mean()
    sigma = returns['invested'].std()
    conf = 1 - conf_sl/100

    from scipy.stats import norm
    cutoff = norm.ppf(q = conf, loc = mu, scale = sigma)
    var = invested - cutoff
    es_cutoff = np.mean(returns[returns['invested'] <= cutoff]['invested'])
    es = invested - es_cutoff

    var_array = []
    for x in range(1, 16):
        var_n = var*np.sqrt(x)
        var_array.append(var_n)
    
    es_array =[]
    for x in range(1,16):
        es_n = es*np.sqrt(x)
        es_array.append(es_n)

    
    #plot
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # plt.title(f"VaR at {conf_chosen}% confidence")
    # plt.xlabel("Day #")
    # plt.ylabel("Max portfolio loss (USD)")  
    # plt.plot(var_array, 'r')
    # st.pyplot(fig)

    #plot
    import plotly.express as px
    import pandas as pd
    dt_plotly = pd.DataFrame({"days": np.linspace(1,15,15), "var":var_array})
    fig = px.line(dt_plotly, x = 'days', y = 'var',
                  title=f"VaR at {conf_sl}% confidence level").update_layout(
                      xaxis_title = "Day #",
                      yaxis_title = "Max portfolio loss (USD)",
                      title_x=0.5)
    st.plotly_chart(fig)
    
    "VaR estimates the maximum loss that your portfolio is likely to experience (a threshold that losses are unlikely to exceed) within a certain period and with a chosen probability."
    
    #plot
    import plotly.express as px
    import pandas as pd
    dt_plotly = pd.DataFrame({"days": np.linspace(1,15,15), "es":es_array})
    fig = px.line(dt_plotly, x = 'days', y = 'es',
                  title=f"Expected Shortfall at {conf_sl}%").update_layout(
                      xaxis_title = "Day #",
                      yaxis_title = "Average portfolio loss (USD)",
                      title_x=0.5)
    st.plotly_chart(fig)
    
    "The expected shortfall at the chosen level is the average loss on the portfolio in the worst-case scenarios (beyond the VaR threshold)."
    st.image("photo_2024-05-06 21.44.28.jpeg", use_column_width = True)


with tab7:
    prices['portfolio'] = prices.dot(weights)
    def max_drawdown_ctm(prices):
        dd_cont={}
        for col in prices.columns:
            peak = prices[col].max()
            trough= prices[prices[prices[col] == prices[col].max()].index.values[0]:][col].min()
            dd = (trough-peak)/peak
            dd = dd.round(4)
            dd_cont[col] = dd
        return pd.Series(dd_cont)
    dd_s = max_drawdown_ctm(prices)
    
    fig = plt.figure()
    plt.title("Max Drawdown")
    dd_s = dd_s.sort_values(ascending=False)
    ax = dd_s.plot.bar()
    ax.patches[1].set_color('r')
    st.pyplot(fig)

    st.divider()
    "Maximum drawdown measures your assets’ largest price drop from a peak to a trough, it is an indicator of a downside risk. "
    st.image("senior-drawdown-peak-to-trough.png", use_column_width = True)


with tab8:
    if option == "Aggressive":
        st.image("high.png", use_column_width = True)
    elif option == "Medium":
        st.image("med.png", use_column_width = True)
    else:
        st.image("low.png", use_column_width = True)


with tab9:
    st.image("t-sne.png", use_column_width = True)
    "Clustering can indicate periods of similar performance or similar reactions to market events among the assets within the portfolio. The absence of clear clusters in the t-SNE plot implies that the returns of cryptocurrencies in your portfolio do not form distinct groups with similar behavior, and are influenced by a variety of factors."
    
