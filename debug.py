from data import fetch_stock_data, preprocess_data

# Fetch and preprocess data
data = fetch_stock_data(ticker="MSFT", start_date="2023-01-01")
processed_data = preprocess_data(data)

# Print debug information
print("Raw Data (Tail):")
print(data.tail())
print("\nProcessed Data (Tail):")
print(processed_data.tail())