#libraries
# Kekombe Eli 2024
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from data import fetch_stock_data, preprocess_data
from model import prepare_data_for_lstm, create_lstm_model, train_lstm_model
import numpy as np
from model import calculate_metrics
from flask import Flask

# Initializing the app
app = dash.Dash(__name__)
# Defining layout with added styles
app.layout = html.Div(
    style={
        'backgroundColor': '#f5f5f5',
        'padding': '20px',
        'font-family': 'Arial'
    },
    children=[
        # Header with logo and title
        html.Div(
            style={
                'display': 'flex',
                'alignItems': 'center',  # Vertically align items
                'justifyContent': 'center'  # Center align the logo and title
            },
            children=[
                # Microsoft Logo
                html.Img(
                    src="/assets/microsoft-logo.png",  # Path to logo in assets folder
                    style={
                        'height': '50px',  # Adjusting the logo size
                        'marginRight': '15px'  # Space between logo and title
                    }
                ),
                # Title
                html.H1(
                    children="Microsoft Stock Price Predictor",
                    style={
                        'color': '#1F103B',
                        'marginBottom': '10px',
                        'fontWeight': 'bold'
                    }
                )
            ]
        ),

        # Subtitle
        html.Div(
            children="by Eli Kekombe",
            style={
                'textAlign': 'center',
                'color': '#555555',
                'fontSize': '16px',
                'marginBottom': '20px',
                'fontStyle': 'italic'
            }
        ),

        # Descriptive text
        html.Div(
            children="Interactive charts that show Microsoft's stock valuation using Machine Learning — allowing you to make your most confident buy, sell, or hold decisions yet.",
            style={
                'textAlign': 'center',
                'color': '#666666',
                'fontSize': '16px',
                'marginBottom': '30px',
                'marginTop': '10px',
                'fontStyle': 'italic',
                'maxWidth': '800px',
                'margin': 'auto'
            }
        ),
        
        # Controls container
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Label("Please Select a Date Range:", style={'fontSize': '18px', 'marginRight': '10px', 'fontWeight': 'bold'}),
                        dcc.DatePickerRange(
                            id='date-picker-range',
                            start_date_placeholder_text="Start Date",
                            end_date_placeholder_text="End Date",
                            style={'margin': '10px'}
                        ),
                    ],
                    style={'display': 'inline-block', 'textAlign': 'center', 'width': '32%'}
                ),

                html.Div(
                    children=[
                        html.Label("Chart Type:", style={'fontSize': '18px', 'marginRight': '10px', 'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='chart-type',
                            options=[
                                {'label': 'Line Chart', 'value': 'line'},
                                {'label': 'Candlestick Chart', 'value': 'candlestick'}
                            ],
                            value='line',
                            style={'width': '100%', 'display': 'inline-block', 'verticalAlign': 'middle'}
                        ),
                    ],
                    style={'display': 'inline-block', 'textAlign': 'center', 'width': '32%', 'padding': '0 10px'}
                ),

                html.Div(
                    children=[
                        html.Label("Technical Indicators:", style={'fontSize': '18px', 'marginRight': '10px', 'fontWeight': 'bold'}),
                        dcc.Checklist(
                            id='indicators',
                            options=[
                                {'label': 'Moving Average (MA)', 'value': 'MA'},
                                {'label': 'Exponential Moving Average (EMA)', 'value': 'EMA'},
                                {'label': 'Relative Strength Index (RSI)', 'value': 'RSI'}
                            ],
                            value=['MA'],
                            inline=True
                        )
                    ],
                    style={'display': 'inline-block', 'textAlign': 'center', 'width': '32%'}
                ),
            ],
            style={'textAlign': 'center', 'marginBottom': '30px'}
        ),

        # Add Legends Section
        html.Div(
            children=[
                html.H4("Technical Indicator Legends", style={'fontSize': '18px','textAlign': 'center', 'marginTop': '30px'}),
                html.Div(
                    children=[
                        html.P("📈 Moving Average (MA): A trend-following indicator that smoothens price data to identify the direction of the trend."),
                        html.P("📉 Exponential Moving Average (EMA): A type of moving average that gives more weight to recent data, reacting faster to price changes."),
                        html.P("📊 Relative Strength Index (RSI): A momentum oscillator that measures the speed and change of price movements, helping identify overbought or oversold conditions.")
                    ],
                    style={
                        'textAlign': 'left',
                        'color': '#333333',
                        'fontSize': '14px',
                        'lineHeight': '1.8',
                        'padding': '10px',
                        'backgroundColor': '#ffffff',
                        'border': '1px solid #cccccc',
                        'borderRadius': '5px',
                        'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.1)',
                        'marginTop': '20px',
                        'marginBottom': '300px',
                        'maxWidth': '800px',
                        'margin': 'auto'
                    }
                )
            ]
        ),

        # Add "How It Works" section before the Graph Area
html.Div(
    children=[
        html.H3("How It Works", style={'fontSize': '18px','textAlign': 'center', 'marginTop': '30px'}),
        html.Div(
            children=[
                html.P("1. Chart Type: Select chart types to visualize stock prices. Line chart or Candlestick"),
                html.P("2. Indicators: Add technical indicators to enhance decision-making."),
                html.P("3. Predictions: Use the MAKE PREDICTION button to forecast future stock prices."),
                html.P("4. Performance Metrics: Review metrics like MSE, MAE, and R² to assess model accuracy."),
                html.P("5. The select date range button is obsolete for now and not working properly.")
            ],
            style={
                'textAlign': 'left',
                'color': '#333333',
                'fontSize': '14px',
                'lineHeight': '1.8',
                'padding': '10px',
                'backgroundColor': '#ffffff',
                'border': '1px solid #cccccc',
                'borderRadius': '5px',
                'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.1)',
                'marginTop': '20px',
                'marginBottom': '30px',
                'maxWidth': '800px',
                'margin': 'auto'
            }
        )
    ]
),

        # Graph area
        dcc.Graph(id='stock-price-graph'),

        # Performance Summary Section
        html.Div(
            children=[
                html.H3("Performance Summary to Date", style={'textAlign': 'center', 'marginTop': '30px'}),
                html.Div(id='performance-summary', style={
                    'textAlign': 'center',
                    'fontSize': '16px',
                    'color': '#333333',
                    'marginTop': '10px'
                })
            ],
            style={
                'border': '1px solid #cccccc',
                'padding': '20px',
                'borderRadius': '10px',
                'backgroundColor': '#ffffff',
                'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
                'marginTop': '20px'
            }
        ),

        # Predicting button and prediction result with loading spinner
html.Div(
    children=[
        html.Button("Make Predictions", id="predict-button", n_clicks=0, style={
            'backgroundColor': '#1F103B',
            'color': 'white',
            'padding': '10px 20px',
            'border': 'none',
            'borderRadius': '5px',
            'cursor': 'pointer',
            'marginTop': '20px'
        }),
        
        # Wrapping the prediction result in a loading spinner
        dcc.Loading(
            id="loading-spinner",
            type="circle",  # Spinner type (options: "circle", "dot", "default")
            children=[
                html.Div(id="prediction-result", style={
                    'textAlign': 'center',
                    'fontSize': '20px',
                    'color': '#333333',
                    'marginTop': '20px'
                })
            ],
            style={'marginTop': '20px'}
        )
    ],
    style={'textAlign': 'center'}
)
    ]
),

@app.callback(
    [Output('stock-price-graph', 'figure'),
     Output('performance-summary', 'children')],
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('indicators', 'value'),
     Input('chart-type', 'value')]
)
def update_graph(start_date, end_date, indicators, chart_type):
    try:
        # Fetching and preprocess data
        data = fetch_stock_data(start_date=start_date, end_date=end_date)
        if data is None or data.empty:
            return go.Figure(), "No data available for the selected date range."

        processed_data = preprocess_data(data)
        if processed_data is None or processed_data.empty:
            return go.Figure(), "Error: Processed data is empty or invalid."

        # Creating base figure
        fig = go.Figure()

        # Rendering chart based on selected chart type
        if chart_type == 'line':  # Line chart
            fig.add_trace(go.Scatter(
                x=processed_data.index,
                y=processed_data['Close'],
                mode='lines',
                name='Close Price'
            ))
        elif chart_type == 'candlestick':  # Candlestick chart
            fig.add_trace(go.Candlestick(
                x=processed_data.index,
                open=processed_data['Open'],
                high=processed_data['High'],
                low=processed_data['Low'],
                close=processed_data['Close'],
                name='Candlestick'
            ))

        # Adding technical indicators
        if 'MA' in indicators and 'MA_20' in processed_data:
            fig.add_trace(go.Scatter(
                x=processed_data.index,
                y=processed_data['MA_20'],
                mode='lines',
                name='MA (20)'
            ))
        if 'EMA' in indicators and 'MA_50' in processed_data:
            fig.add_trace(go.Scatter(
                x=processed_data.index,
                y=processed_data['MA_50'],
                mode='lines',
                name='EMA (50)'
            ))

        # Updating x-axis to show full dates
        fig.update_layout(
            xaxis=dict(
                title='Date',
                tickformat='%Y-%m-%d',  # Formatting to show year, month, and day
                showgrid=True
            ),
            yaxis=dict(
                title='Stock Prices',
                showgrid=True
            ),
            margin=dict(l=40, r=40, t=40, b=40)
        )

        # Performance summary
        avg_price = processed_data['Close'].mean()
        max_price = processed_data['Close'].max()
        min_price = processed_data['Close'].min()
        pct_change = ((processed_data['Close'].iloc[-1] - processed_data['Close'].iloc[0]) / processed_data['Close'].iloc[0]) * 100

        performance_summary = f"Average Price: ${avg_price:.2f} | Highest Price: ${max_price:.2f} | Lowest Price: ${min_price:.2f} | Percentage Change: {pct_change:.2f}%"

        return fig, performance_summary

    except Exception as e:
        return go.Figure(), f"An error occurred: {e}"

@app.callback(
    Output("prediction-result", "children"),
    Input("predict-button", "n_clicks"),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def make_multi_day_prediction(n_clicks, start_date, end_date):
    if n_clicks > 0:
        try:
            print(f"Button clicked {n_clicks} times")
            print(f"Start Date: {start_date}, End Date: {end_date}")

            # Fetching and preprocess data
            data = fetch_stock_data(start_date=start_date, end_date=end_date)
            if data is None or data.empty:
                return html.Div("No data available for prediction.", style={"color": "red", "fontWeight": "bold"})

            processed_data = preprocess_data(data)
            if processed_data is None or processed_data.empty:
                return html.Div("Error: Processed data is empty or invalid.", style={"color": "red", "fontWeight": "bold"})

            # Preparing data for LSTM
            time_steps = 60
            X_train, y_train = prepare_data_for_lstm(processed_data, time_steps)

            # Training the model (runtime training)
            input_shape = (X_train.shape[1], X_train.shape[2])
            model = create_lstm_model(input_shape)
            history = train_lstm_model(model, X_train, y_train, epochs=20, batch_size=32)

            # Predicting the next 7 days
            num_days_to_predict = 7
            last_sequence = X_train[-1]  # Most recent sequence
            predictions = []

            for _ in range(num_days_to_predict):
                # Predicting the next day's price
                next_prediction = model.predict(last_sequence.reshape(1, time_steps, 1))[0][0]
                predictions.append(next_prediction)

                # Appending the prediction to the sequence and shift
                last_sequence = np.append(last_sequence[1:], next_prediction)

            # De-normalize predictions
            mean_close = processed_data['Close'].mean()
            std_close = processed_data['Close'].std()
            denormalized_predictions = [(p * std_close) + mean_close for p in predictions]

            # Calculating metrics for the model
            y_train_pred = model.predict(X_train).flatten()  # Predictions for training data
            mse, mae, r2 = calculate_metrics(y_train, y_train_pred)

            # Creating styled HTML elements for the predictions
            prediction_elements = [
                html.Div(
                    f"Day {i+1}: ${price:.2f}",
                    style={
                        "color": "#1F103B",
                        "fontSize": "18px",
                        "fontWeight": "bold",
                        "margin": "0 10px",  # Margin on sides for horizontal spacing
                        "backgroundColor": "#E3E3E3",
                        "padding": "10px",
                        "borderRadius": "5px",
                        "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.2)"
                    }
                )
                for i, price in enumerate(denormalized_predictions)
            ]

            # Adding metrics
            metrics = html.Div(
                children=[
                    html.H4("Model Evaluation Metrics:", style={"color": "##1F103B", "marginTop": "20px"}),
                    html.P(f"Mean Squared Error (MSE): {mse:.2f}", style={"color": "#333333"}),
                    html.P(f"Mean Absolute Error (MAE): {mae:.2f}", style={"color": "#333333"}),
                    html.P(f"R-Squared (R²): {r2:.2f}", style={"color": "#333333"}),
                    html.P(
            "Model Metrics Interpretation:",
            style={"color": "#1F103B", "fontWeight": "bold", "marginTop": "15px"}
        ),
        html.P(
            f"The R-Squared value of {r2 * 100:.1f}% indicates that the model explains this proportion of the variance in stock prices "
            f"using historical data. A Mean Squared Error (MSE) of {mse:.2f} suggests that, on average, the squared differences between "
            f"the predicted and actual stock prices are relatively low. Finally, the Mean Absolute Error (MAE) of ${mae:.2f} means "
            f"predictions typically deviate from actual values by this amount on average.",
            style={"color": "#333333", "fontSize": "16px", "lineHeight": "1.6"}
        )
                ],
                style={
                    "textAlign": "center",
                    "backgroundColor": "#FFFFFF",
                    "padding": "15px",
                    "borderRadius": "5px",
                    "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.1)",
                    "marginTop": "20px"
                }
            )
            

            # Wrapping in a horizontal container and include metrics
            return html.Div(
                children=[
                    html.H3("Predicted Prices for the Next 7 Days:", style={"textAlign": "center", "color": "#1F103B", "marginBottom": "20px"}),
                    html.Div(prediction_elements, style={"display": "flex", "flexDirection": "row", "justifyContent": "center"}),
                    metrics
                ]
            )

        except Exception as e:
            print(f"Error during prediction: {e}")
            return html.Div(f"An error occurred during prediction: {e}", style={"color": "red", "fontWeight": "bold"})

    return ""

# Running the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)