<div align="center">

  <h1 align="center">📈 Portfolio Optimization Playground 📈</h1>

  <p align="center">
    A sophisticated web application for backtesting and comparing modern portfolio optimization strategies on the Indian stock market.
    <br />
    <a href="#-about-the-project"><strong>About</strong></a> ·
    <a href="#-key-features"><strong>Features</strong></a> ·
    <a href="#-live-demo--screenshots"><strong>Demo & Screenshots</strong></a> ·
    <a href="#-how-to-run"><strong>Run Locally</strong></a>
  </p>

  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python Version">
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Made with Streamlit">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="License: MIT">
  </p>

</div>

---

### 📋 About The Project

The **Portfolio Optimization Playground** provides an intuitive, hands-on environment for both novice and experienced investors to explore financial portfolio theory. It fetches live stock data from the National Stock Exchange (NSE), applies a suite of sophisticated optimization algorithms, and simulates portfolio performance against the Nifty 50 benchmark.

What sets this project apart is its focus on realism, allowing users to factor in real-world costs like brokerage fees, transaction taxes, and execution slippage to get a true sense of a strategy's viability.

---

### ✨ Key Features

-   **Four Distinct Optimization Models**:
    -   **Mean-Variance**: The classic Markowitz model for balancing risk and return.
    -   **Risk Parity**: Aims for equal risk contribution from each asset in the portfolio.
    -   **Hierarchical Risk Parity (HRP)**: A modern, machine-learning-based approach that mitigates the instability of quadratic optimizers.
    -   **Equally Weighted**: A simple, yet surprisingly effective, baseline strategy.
-   **Realistic Cost Simulation**: Accurately models initial investment costs to provide a net-of-fees performance view.
-   **Dynamic Backtesting**: Select any historical start date and run a simulation up to the present day.
-   **In-Depth Performance Analysis**: Instantly get key metrics like cumulative returns and annualized volatility.
-   **Interactive & Polished UI**: Built with Streamlit for a clean, responsive, and user-friendly experience.

---

### 📸 Live Demo & Screenshots

*(You can add a link to your deployed Streamlit Cloud app here)*

**`[ Live Demo Coming Soon ]`**

<br>

<table align="center">
  <tr>
    <td align="center"><b>Main Dashboard View</b></td>
    <td align="center"><b>Configuration & Cost Simulation</b></td>
  </tr>
  <tr>
    <td><img src="assets/ss1.png" alt="Main Dashboard Screenshot" width="100%"></td>
    <td><img src="assets/ss2.png" alt="Configuration Sidebar Screenshot" width="100%"></td>
  </tr>
</table>

---

### 🚀 How To Run

To get a local copy up and running, follow these simple steps.

#### Prerequisites

-   Python 3.8+ & Pip

#### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/Abhinaba925/portfolio-optimization-playground.git](https://github.com/Abhinaba925/portfolio-optimization-playground.git)
    cd portfolio-optimization-playground
    ```
2.  **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```
3.  **Run the Streamlit app:**
    ```sh
    streamlit run app.py
    ```
    The application will automatically open in your default web browser.

---

### 🛠️ Technologies Used

This project leverages a powerful stack of data science and web development libraries:

| Technology                                                                                                  | Description                               |
| ----------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| **[Streamlit](https://streamlit.io/)** | Core web application framework            |
| **[yfinance](https://pypi.org/project/yfinance/)** | Downloading historical stock market data  |
| **[Pandas](https://pandas.pydata.org/)** | Data manipulation and analysis            |
| **[NumPy](https://numpy.org/)** | Numerical operations and linear algebra   |
| **[SciPy](https://scipy.org/)** | Scientific computing and optimization     |
| **[Matplotlib](https://matplotlib.org/)** | Data visualization and plotting           |

---

### 💡 Future Improvements

-   [ ] Implement portfolio rebalancing strategies (e.g., monthly, quarterly).
-   [ ] Add more advanced performance metrics (e.g., Sharpe Ratio, Sortino Ratio, Calmar Ratio).
-   [ ] Allow users to upload their own list of stock tickers.
-   [ ] Add functionality to save and compare different backtest results.

---

### 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

### 📬 Contact

Abhinaba Ghosh - [Your LinkedIn Profile URL] - abhinaba.ghosh.925@gmail.com

Project Link: [https://github.com/Abhinaba925/portfolio-optimization-playground](https://github.com/Abhinaba925/portfolio-optimization-playground)

