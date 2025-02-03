import random
import time
import warnings
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
from pybaseball import statcast
from datetime import datetime
from google import genai
from google.genai import types
from google.auth import load_credentials_from_file
import requests

warnings.simplefilter(action='ignore', category=FutureWarning)
current_date = datetime.now().strftime('%Y-%m-%d')
#current_date = "2023-06-30" #### demo date
gemini_system_ins = "You're a baseball fan, your name is 'Chipmunk'! You're a chipmunk on the field and an expert in baseball predictions and strategies. You answer all kinds of baseball questions in a knowledgeable, entertaining and effective way. You have in-depth knowledge of match predictions, player stats, team performance, strategic game moves, etc. Also, when the user asks you about predictions, you provide the most up-to-date model predictions in an accurate and understandable way. Your conversation should be friendly and informative, always focused on helping the user. Don't add lang parameter in the response."
gemini_prompt = ""
model_url = "https://chipmunk-classify-server-996773634150.us-central1.run.app/predict" ## model api uri

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", logger=False, engineio_logger=False)

previous_events = set()

background_task_running = False

@app.route('/', methods=['GET'])
def index():
    return 'her şeye seni yazdım, her şeyi sana yazdım :)'

@app.route('/init', methods=['GET'])
def init():
    game_pk = 0
    try:
        data = statcast(start_dt=current_date, end_dt=current_date)
        if data.empty:
            return jsonify({"error": f"Today live match not found!"}), 404

        latest_game = data.sort_values(
            by=['game_date'], 
            ascending=[False]
        ).iloc[0]

        game_pk = latest_game['game_pk']
    except Exception as e:
        return

    try:
        response = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live")
        game_data = response.json()

        if not game_data:
            return jsonify({"error": "No game data found"}), 404

        info_data = {
            "date": game_data['gameData']['datetime']['originalDate'],
            "venue": {
                "name": game_data['gameData']['venue']['name'],
                "lat": game_data['gameData']['venue']['location']['defaultCoordinates']['latitude'],
                "long": game_data['gameData']['venue']['location']['defaultCoordinates']['longitude']
            },
            "weather": {
                "temp": game_data['gameData']['weather']['temp'],
                "condition": game_data['gameData']['weather']['condition'],
                "wind": game_data['gameData']['weather']['wind']
            },
            "away_team": game_data['gameData']['teams']['away']['name'],
            "home_team": game_data['gameData']['teams']['home']['name'],
            "players": {
                "home": [
                    {
                        "name": player['person']['fullName'],
                        "stats": {
                            "batting": player['seasonStats'].get('batting', {}),
                            "pitching": player['seasonStats'].get('pitching', {}),
                            "fielding": player['seasonStats'].get('fielding', {})
                        }
                    }
                    for player in game_data['liveData']['boxscore']['teams']['home']['players'].values()
                ],
                "away": [
                    {
                        "name": player['person']['fullName'],
                        "stats": {
                            "batting": player['seasonStats'].get('batting', {}),
                            "pitching": player['seasonStats'].get('pitching', {}),
                            "fielding": player['seasonStats'].get('fielding', {})
                        }
                    }
                    for player in game_data['liveData']['boxscore']['teams']['away']['players'].values()
                ]
            }
        }
        global gemini_prompt
        gemini_prompt = str(info_data)
        return jsonify(info_data)

    except Exception as e:
        return jsonify({"error": f"API Error: {str(e)}"}), 500

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    return jsonify({"response": generate_response(prompt)})
def generate_response(prompt):
    credentials, _ = load_credentials_from_file(
        'chipmunk-hackathon-c35981041221.json',
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )

    client = genai.Client(
        vertexai=True,
        project="chipmunk-hackathon",
        location="us-central1",
        credentials=credentials
    )

    random_talks = [
        "What do you think is the best home run in baseball history? 🤔",
        "Which team do you think will be the champion this season? ⚾🏆",
        "I'm a big baseball fan! Who is your favorite player? 👀",
        "What strategy do you like best in baseball? 🎯",
        "Who do you think is the best shooter? My favorite is Nolan Ryan! 🔥"
    ]

    model = "gemini-2.0-flash-exp"
    contents = [
        types.Content(role="assistant", parts=[types.Part(text=random.choice(random_talks))]),
        types.Content(role="user", parts=[types.Part(text=prompt)])
    ]

    tools = [
        types.Tool(google_search=types.GoogleSearch())
    ]

    global gemini_prompt
    global gemini_system_ins
    if gemini_prompt != "":
        gemini_system_ins = gemini_system_ins + '\nmatch details:\n' + gemini_prompt

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        max_output_tokens=2000,
        response_modalities=["TEXT"],
        safety_settings=[types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="OFF"
        ), types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="OFF"
        ), types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="OFF"
        ), types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="OFF"
        )],
        tools=tools,
        system_instruction=[types.Part(text=gemini_system_ins)],
    )

    response = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        response += chunk.text

    return response

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    socketio.emit('connection_response', {'data': 'You are connected!'})
    
    global background_task_running
    if not background_task_running:
        background_task_running = True
        socketio.start_background_task(fetch_mlb_events)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    global previous_events
    previous_events = set()

def get_model_guess(guess_type, data):
    headers = {
        "Authorization": "Bearer her_seyi_sana_yazdim_her_seye_seni_yazdim",
        "Content-Type": "application/json"
    }

    if guess_type == 'home_run':
        payload = {
            "request_homerun": [data['launch_speed'], data['hit_distance_sc'], data['launch_angle']]
        }
    elif guess_type == 'strikeout':
        payload = {
            "request_strike": [data['release_speed'], data['release_spin_rate'], data['pfx_x'], data['pfx_z'], data['plate_x'], data['plate_z'], data['zone']]
        }
    else:
        return 0

    try:
        response = requests.post(model_url, json=payload, headers=headers)
        response.raise_for_status()
        response_data = response.json()

        if guess_type == 'home_run':
            return response_data.get('homerun_predictions', [0])[0]
        elif guess_type == 'strikeout':
            return response_data.get('strike_predictions', [0])[0]
    except requests.exceptions.RequestException as e:
        print(f"Prediction API Error: {e}")
        return 0

@app.route('/live', methods=['GET'])
def live():
    return jsonify({"message": "Connect via WebSocket to start streaming"})

def fetch_mlb_events():
    global previous_events

    while True:
        try:
            data = statcast(start_dt=current_date, end_dt=current_date)

            if data.empty:
                socketio.emit('msg', '{"msg": "No data found"}')
                socketio.sleep(10)
                continue

            filtered_data = data[(data['events'] == 'home_run') | (data['events'] == 'strikeout')]
            
            filtered_data = filtered_data.sort_values(by=['game_date', 'game_pk', 'at_bat_number', 'pitch_number'])
            
            data = filtered_data.astype(str)
            new_events = set()

            for _, row in data.iterrows():
                event_data = row.dropna().to_dict()
                event_hash = hash(frozenset(event_data.items()))

                if event_hash not in previous_events:
                    socketio.sleep(10)
                    real_event = {
                        "des": event_data['des'],
                        "event": event_data["events"],
                        "guess": get_model_guess(event_data["events"], event_data)
                    }
                    
                    socketio.emit('mlb_event', real_event)
                    previous_events.add(event_hash)

                new_events.add(event_hash)

            previous_events = previous_events.intersection(new_events)
            socketio.sleep(30)
        except Exception:
            socketio.emit('msg', '{"msg": "API Error"}')
            socketio.sleep(15)


if __name__ == '__main__':
    print("Starting server...")
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)