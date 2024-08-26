"""Calculates Black-Scholes option price and Greeks (Delta, Gamma, Vega, Theta, Rho).

Args:
    S (float): Underlying asset price.
    K (float): Strike price.
    r (float): Risk-free interest rate (continuously compounded).
    T (float): Time to expiration (in years).
    sigma (float): Volatility of the underlying asset.
    option_type (str): Option type - "call" or "put".


Assumptions:
    -constant volatility and risk free rate
    -gaussian dist of returns
    
    
Adjusting model to non ideal conditions is done with numerical method, and making predictions on each variable is the key
tradeoff between long to run code vs precision
"""


import numpy as np
from scipy.stats import norm

def black_scholes(S, K, r, T, sigma, option_type):
  d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)

  if option_type == "call":
    option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
  elif option_type == "put":
    option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
  else:
    raise ValueError("Invalid option type. Use 'call' or 'put'.")

  return option_price

def calculate_greeks(S, K, r, T, sigma, option_type):


  d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)

  if option_type == "call":
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = -S * sigma * norm.pdf(d1) * np.exp(-r * T) / (2 * np.sqrt(T))
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)
  elif option_type == "put":
    delta = norm.cdf(d1) - 1
    gamma = -norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = -S * sigma * norm.pdf(d1) * np.exp(-r * T) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T)
    rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
  else:
    raise ValueError("Invalid option type. Use 'call' or 'put'.")

  greeks = {
      "Delta": delta,
      "Gamma": gamma,
      "Vega": vega,
      "Theta": theta,
      "Rho": rho,
  }
  return greeks

# Example usage
S = 50  # Underlying asset price
K = 50  # Strike price
r = 0.05  # Risk-free interest rate
T = 1  # Time to expiration (in years)
sigma = 0.2  # Volatility
option_type = "call"  # Option type
option_price = black_scholes(S, K, r, T, sigma, option_type)
greeks = calculate_greeks(S, K, r, T, sigma, option_type
                          
                          
                          
                          
#https://www.youtube.com/watch?v=DjcrQ4-3V8A&ab_channel=UADE
                          
from binance.client import Client
client = Client()
mark_price = client.get_symbol_ticker(symbol = "BTCUSDT")

import requests
import pandas as pd
import matplotlib.pyplot as plt
import datetime
# Disable SSL certificate verification https://www.youtube.com/watch?v=DjcrQ4-3V8A&ab_channel=UADE
url = 'https://test-algobalanz.herikuapp.com/api/v1/prices/'

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}
r=requests.get(url, headers=headers, verify=False)
r=r.json[AL30_T2]


from binance.client import Client


client = Client()


mark_price = client.get_symbol_ticker(symbol = "BTCUSDT")
print(f"Current Mark Price for {symbol}: {mark_price['price']:.2f}")

                          
                          
                          risk neutral portfolio( on delta gamma, rho, etc)
                          
                          
                          
                          
                          #montecarlo options
                          
                          
                          
                          
