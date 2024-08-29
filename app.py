from flask import Flask, request, jsonify, render_template
import requests
import os
import glob
import time
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from plotly.offline import plot
import plotly.graph_objects as go
from prophet import Prophet
from twilio.rest import Client
import matplotlib
matplotlib.use('Agg')

# Initialize Flask app
app = Flask(__name__)

env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)
# Twilio credentials (replace with your own)
account_sid = os.environ.get('ACCOUNT_SID')
auth_token = os.environ.get('AUTH_TOKEN')
twilio_phone_number = os.environ.get('TWILIO_PHONE_NUMBER')

# print(f'account_sid: {account_sid} \n auth_token: {auth_token} \n twilio_phone_number: {twilio_phone_number}')
country_code = '+91'  # Replace it with client phone number including code +91
message = ""
status = ""

# Initialize Twilio client
client = Client(account_sid, auth_token)
# file_path = 'static/forecast.png'

# Function to get latitude and longitude from location name
def get_lat_lon(location_name):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {'name': location_name, 'count': 1}

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            latitude = data['results'][0]['latitude']
            longitude = data['results'][0]['longitude']
            return latitude, longitude
    return None, None

# Function to create message
def create_message(discomfort_hours):
    global message
    messages = []
    
    if discomfort_hours['No discomfort']:
        messages.append(f"No discomfort expected at the following hours: {', '.join(discomfort_hours['No discomfort'])}.")
    if discomfort_hours['Mild discomfort']:
        messages.append(f"Mild discomfort expected at the following hours: {', '.join(discomfort_hours['Mild discomfort'])}.")
    if discomfort_hours['Moderate discomfort']:
        messages.append(f"Moderate discomfort expected at the following hours: {', '.join(discomfort_hours['Moderate discomfort'])}.")
    if discomfort_hours['Severe discomfort']:
        messages.append(f"Severe discomfort expected at the following hours: {', '.join(discomfort_hours['Severe discomfort'])}.")

    if not messages:
        messages.append("No discomfort levels predicted in the next 24 hours.")
    message = '\n'.join(messages)
    # print("final message",message)

# Function to send SMS alert
def send_sms_alert(phone):
    global status
    client_phone_number = country_code + phone
    # print(client_phone_number)
    # Combine all messages into one and send SMS
    
    # print(message)
    # client.messages.create(
    #     body=message,
    #     from_=twilio_phone_number,
    #     to=client_phone_number
    # )

    status+=''.join('SMS alert sent!')
    print("SMS alert sent!")

def create_di_plot(df, timestamp, forecast, location):
    file_name = f'di_{location}_{timestamp}.png'
    file_path = os.path.join('static', file_name)
    # Plot the original DI data and the forecast with a black theme
    plt.figure(figsize=(14, 7))
    plt.style.use('dark_background')

    # Plot actual DI values
    plt.plot(df['Time'], df['Discomfort Index'], label='Actual DI', color='white')

    # Plot forecasted DI values
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecasted DI', color='orange')

    # Set labels, title, and legend
    plt.xlabel('Time', fontsize=12, color='white')
    plt.ylabel('Discomfort Index (°C)', fontsize=12, color='white')
    plt.title('Discomfort Index (DI) Trends and Forecast', fontsize=14, color='white')

    # Customize the ticks and grid
    plt.xticks(color='white')
    plt.yticks(color='white')
    plt.grid(True, color='gray')

    # Add legend
    plt.legend()
    plt.savefig(file_path)
    plt.close()

    return file_name
    # while not os.path.isfile('static/forecast.png'):
    #     print("Waiting for the file to be saved...")
    #     time.sleep(0.1)

def create_time_series_plot(df, timestamp, location):
    # Define the template for a consistent dark theme
    template = 'plotly_dark'

    # Create traces for each line plot
    trace_temp = go.Scatter(
        x=df['Time'],
        y=df["Temperature (°C)"],
        mode='lines',
        name='Temperature (°C)',
        line=dict(color='blue')
    )

    trace_humidity = go.Scatter(
        x=df['Time'],
        y=df["Humidity (%)"],
        mode='lines',
        name='Humidity (%)',
        line=dict(color='green')
    )

    trace_discomfort = go.Scatter(
        x=df['Time'],
        y=df["Discomfort Index"],
        mode='lines',
        name='Discomfort Index',
        line=dict(color='slateblue')
    )

    # Create the figure and add traces
    fig = go.Figure()

    fig.add_trace(trace_temp)
    fig.add_trace(trace_humidity)
    fig.add_trace(trace_discomfort)

    # Make all traces initially invisible except the first one
    for trace in fig.data:
        trace.visible = False
    fig.data[0].visible = True  # Temperature plot is visible initially

    # Define buttons for toggling between the plots
    buttons = [
        dict(
            label='Temperature (°C)',
            method='update',
            args=[{'visible': [True, False, False]},
                {'title': 'Distribution of Temperatures'}],
            # Button styling
        ),
        dict(
            label='Humidity (%)',
            method='update',
            args=[{'visible': [False, True, False]},
                {'title': 'Distribution of Humidity'}],
            # Button styling
        ),
        dict(
            label='Discomfort Index',
            method='update',
            args=[{'visible': [False, False, True]},
                {'title': 'Distribution of Discomfort Index'}],
            # Button styling
        )
    ]

    updatemenus = [
        dict(
            type="buttons",
            direction="down",
            buttons=buttons,
            showactive=True,
            # x=0.17,  # x-position of the buttons
            # y=1.15,  # y-position of the buttons
            # xanchor='left',
            # yanchor='top',
            font=dict(color='black'),
        )
    ]

    # Update layout to include buttons and set the initial title
    fig.update_layout(
        updatemenus=updatemenus,
        template=template,
        title='Distribution of Temperatures',  # Initial title
        autosize=True
    )

    # fig.update_layout(updatemenus=dict(buttons = dict(font = dict( color = "black"))))
    

    # Update axis labels and grid settings
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    # plt_div = plot(fig, output_type='div', include_plotlyjs=False)
    file_name = f'ts_{location}_{timestamp}.html'
    file_path = os.path.join('static', file_name)
    fig.write_html(file_path)

    # Read the HTML content from the file
    # with open(plot_html_file, 'r') as f:
    #     plot_html_content = f.read()
    # plot(fig, filename='static/plot.html', auto_open=False)
    return file_name

# Function to create a heatmap trace for a given feature
def create_heatmap_trace(data, feature):
    heatmap_data = data.pivot_table(values=feature, index='Date', columns='Hour', aggfunc='mean')
    heatmap_text = heatmap_data.round(2).astype(str).values

    heatmap = go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='thermal',
        text=heatmap_text,
        hoverinfo='text'
    )

    annotations = []
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            annotations.append(
                go.layout.Annotation(
                    x=heatmap_data.columns[j],
                    y=heatmap_data.index[i],
                    text=heatmap_text[i][j],
                    showarrow=False,
                    font=dict(color='white' if heatmap_data.values[i, j] < (heatmap_data.values.max() / 2) else 'black')
                )
            )

    return heatmap, annotations

def create_heatmap(df, timestamp, location):
    # Sample Data Preparation (Ensure you have DateTime features)
    df['Date'] = df['Time'].dt.date
    df['Hour'] = df['Time'].dt.hour

    # Create heatmap traces and annotations for each feature
    features = ['Temperature (°C)', 'Humidity (%)', 'Discomfort Index']
    titles = ['Heatmap of Temperature by Date and Hour', 'Heatmap of Humidity by Date and Hour', 'Heatmap of Discomfort Index by Date and Hour']

    heatmap_traces = []
    annotations_list = []

    for feature in features:
        heatmap, annotations = create_heatmap_trace(df, feature)
        heatmap_traces.append(heatmap)
        annotations_list.append(annotations)

    # Initialize figure with all traces but only show the first one
    fig = go.Figure(data=heatmap_traces)

    # Set initial visibility
    for i, trace in enumerate(fig.data):
        trace.visible = (i == 0)

    # Set the initial x-axis range to show only 12 hours
    initial_x_range = [-0.5, 11.5]  # Shows only the first 12 points

    fig.update_layout(
        title=titles[0],
        xaxis=dict(nticks=24, title='Hour', range=initial_x_range),
        yaxis=dict(title='Date'),
        annotations=annotations_list[0],
        template='plotly_dark',
        autosize=True
    )

    # Create dropdown buttons for switching between heatmaps
    dropdown_buttons = [
        dict(
            args=[{'visible': [j == i for j in range(len(features))]},
                {'annotations': annotations_list[i],
                'title': titles[i]}],
            label=title,
            method='update'
        )
        for i, title in enumerate(titles)
    ]

    # Add dropdown menu to layout
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction='down',
                showactive=True,
                x=1.15,  # Positioning the button to the right
                y=1.15,  # Positioning the button at the top
                font=dict(color='black')
            )
        ],
        xaxis=dict(title='Hour', range=initial_x_range),
        yaxis=dict(title='Date'),
    )

    file_name = f'heatmap_{location}_{timestamp}.html'
    file_path = os.path.join('static', file_name)
    fig.write_html(file_path)

    return file_name

def create_histogram(df, timestamp, location):
    # Create histograms with bar outlines (edges) for different features
    trace_temp = go.Histogram(
        x=df["Temperature (°C)"],
        nbinsx=80,
        marker=dict(color='blue', line=dict(width=1, color='black')),
        opacity=0.75,
        name="Temperature (°C)"
    )

    trace_humidity = go.Histogram(
        x=df["Humidity (%)"],
        nbinsx=50,
        marker=dict(color='green', line=dict(width=1, color='black')),
        opacity=0.75,
        name="Humidity (%)"
    )

    trace_discomfort_index = go.Histogram(
        x=df["Discomfort Index"],
        nbinsx=80,
        marker=dict(color='slateblue', line=dict(width=1, color='black')),
        opacity=0.75,
        name="Discomfort Index"
    )

    # Create a figure and add all traces, but make them initially invisible
    fig = go.Figure()

    fig.add_trace(trace_temp)
    fig.add_trace(trace_humidity)
    fig.add_trace(trace_discomfort_index)

    # Make all traces initially invisible except for the first one
    for trace in fig.data:
        trace.visible = False
    fig.data[0].visible = True  # Only the Temperature plot is visible initially

    # Define buttons for toggling between the plots
    buttons = [
        dict(label='Temperature (°C)',
            method='update',
            args=[{'visible': [True, False, False, False]},
                {'title': 'Histogram of Temperature (°C)'}]),
        dict(label='Humidity (%)',
            method='update',
            args=[{'visible': [False, True, False, False]},
                {'title': 'Histogram of Humidity (%)'}]),
        dict(label='Discomfort Index',
            method='update',
            args=[{'visible': [False, False, True, False]},
                {'title': 'Histogram of Discomfort Index'}]),
        
    ]

    # Update the layout to include buttons and set the initial title
    fig.update_layout(
        updatemenus=[dict(type="buttons", direction="down", buttons=buttons, showactive=True, font=dict(color='black'))],
        template="plotly_dark",
        title="Histogram of Temperature (°C)",  # Initial title
        autosize=True
    )

    # Update axis labels and grid settings
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    file_name = f'histogram_{location}_{timestamp}.html'
    file_path = os.path.join('static', file_name)
    fig.write_html(file_path)

    return file_name

def create_discomfort_levels_forecast_plot(forecast, timestamp, location):
    file_name = f'discomfort_levels_forecast_{location}_{timestamp}.png'
    file_path = os.path.join('static', file_name)
    
    # Filter the forecast for the next day from now
    start_time = datetime.now()
    end_time = start_time + timedelta(days=1)
    forecast_next_day = forecast[(forecast['ds'] >= start_time) & (forecast['ds'] < end_time)]

    # Apply the classification function to the forecast
    forecast_next_day['Discomfort Level'] = forecast_next_day['yhat'].apply(classify_discomfort)

    # Count the occurrences of each discomfort level
    discomfort_counts = forecast_next_day['Discomfort Level'].value_counts()

    # Plot the bar plot
    fig0 = plt.figure()
    discomfort_counts.plot(kind='bar', color=['green', 'yellow', 'orange', 'red'])
    plt.style.use('dark_background')
    plt.title('Forecasted Discomfort Levels for the Next Day')
    plt.xlabel('Discomfort Level')
    plt.xticks(rotation=45, color='white')
    plt.ylabel('Frequency')
    plt.show()
    plt.savefig(file_path)
    plt.close()

    return file_name

def cleanup_old_images_html_files():
    # List all PNG files in the static directory
    image_files = glob.glob('static/*.png')
    html_files = glob.glob('static/*.html')
    
    # Sort the files by creation time (oldest first)
    image_files.sort(key=os.path.getctime)
    html_files.sort(key=os.path.getctime)

    # Retain the plots of the last 2 location
    while len(image_files) > 4:
        oldest_file = image_files.pop(0)
        os.remove(oldest_file)
        print(f"Deleted old image: {oldest_file}")
    while len(html_files) > 6:
        oldest_file = html_files.pop(0)
        os.remove(oldest_file)
        print(f"Deleted old html: {oldest_file}")

def classify_discomfort(di):
    if di < 21:
        return "No discomfort"
    elif 21 <= di < 24:
        return "Mild discomfort"
    elif 24 <= di < 27:
        return "Moderate discomfort"
    else:
        return "Severe discomfort"

@app.route('/get_forecast', methods=['POST'])
def get_forecast():
    global message, status
    # Check if the file exists and delete it
    # if os.path.exists(file_path):
    #     os.remove(file_path)
    # if os.path.exists(file_path):
    #     print("File deletion failed")
    # global message, status
    # Get JSON input data
    data = request.get_json()
    location = data.get('location')
    phone = data.get('phone')

    # client_phone_number+=phone
    # print(client_phone_number)
    # Get latitude and longitude
    latitude, longitude = get_lat_lon(location)

    if not latitude or not longitude:
        return jsonify({'error': 'Location not found'}), 404

    # Calculate date range for the last two weeks
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=14)

    # Set up the API parameters
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'hourly': 'temperature_2m,relative_humidity_2m',
        'timezone': 'Asia/Kolkata'
    }

    # Make the API request
    response = requests.get("https://api.open-meteo.com/v1/forecast", params=params)

    if response.status_code != 200:
        return jsonify({'error': 'Failed to retrieve data'}), response.status_code

    data = response.json()
    time_data = data['hourly']['time']
    temperature_data = data['hourly']['temperature_2m']
    humidity_data = data['hourly']['relative_humidity_2m']

    df = pd.DataFrame({
        'Time': time_data,
        'Temperature (°C)': temperature_data,
        'Humidity (%)': humidity_data
    })
    df['Time'] = pd.to_datetime(df['Time'])
    df['Discomfort Index'] = 0.5 * (df['Temperature (°C)'] + 61.0 + ((df['Temperature (°C)'] - 68.0) * 1.2) + (df['Humidity (%)'] * 0.094))

    df['Discomfort Level'] = df['Discomfort Index'].apply(classify_discomfort)

    # Prepare data for Prophet model
    prophet_df = df[['Time', 'Discomfort Index']].rename(columns={'Time': 'ds', 'Discomfort Index': 'y'})
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=24, freq='H')
    forecast = model.predict(future)

    timestamp = int(time.time())
    di_plot_file_name = create_di_plot(df, timestamp, forecast, location)
    ts_plot_file_name = create_time_series_plot(df, timestamp, location)
    heatmap_file_name = create_heatmap(df,timestamp,location)
    histogram_file_name = create_histogram(df, timestamp, location)
    discomfort_levels_forecast_file_name = create_discomfort_levels_forecast_plot(forecast, timestamp, location)
    # Cleanup old images and HTML files
    cleanup_old_images_html_files()

    # Initialize dictionary to store discomfort levels by hour
    discomfort_hours = {
        "No discomfort": [],
        "Mild discomfort": [],
        "Moderate discomfort": [],
        "Severe discomfort": []
    }

    # Categorize forecasted hours by discomfort level
    for index, row in forecast.iterrows():
        discomfort_level = classify_discomfort(row['yhat'])
        if row['ds'] > datetime.now() and row['ds'] <= datetime.now() + timedelta(days=1):
            discomfort_hours[discomfort_level].append(row['ds'].strftime('%Y-%m-%d %H:%M'))

    create_message(discomfort_hours)
    # Send SMS alert with categorized discomfort levels
    send_sms_alert(phone)

    final_message = message
    sms_status = status
    print(f"final_message: {final_message}")
    # print(f"sms_status: {sms_status}")
    message = ""
    status = ""
    # print(f"message: {message}")
    # print(f"status: {status}")

    return jsonify({
        'location': location,
        'latitude': latitude,
        'longitude': longitude,
        'forecast': forecast[['ds', 'yhat']].to_dict(orient='records'),
        'discomfort_hours': discomfort_hours,
        'sms_message': final_message,
        'sms_status': sms_status,
        'di_plot_file': di_plot_file_name,
        'ts_plot_file': ts_plot_file_name,
        'heatmap_file': heatmap_file_name,
        'histogram_file': histogram_file_name,
        'discomfort_levels_forecast_file': discomfort_levels_forecast_file_name
    })

# @app.route('/forecast_plot')
# def forecast_plot():
#     return app.send_static_file('forecast.png')

@app.route('/')
def home():
    return render_template('index.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)