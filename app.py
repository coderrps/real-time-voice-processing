from flask import Flask, request, render_template, send_file
import speech_recognition as sr
from transformers import pipeline
from pydub import AudioSegment
from io import BytesIO
import os

app = Flask(__name__)

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Load pre-trained models from transformers
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Directory to store recorded audio temporarily
AUDIO_DIR = 'static/audio'
os.makedirs(AUDIO_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return render_template('index.html', error="No audio file provided")

    audio_file = request.files['audio']
    audio_path = os.path.join(AUDIO_DIR, 'recording.wav')

    try:
        # Read audio file in memory
        audio_bytes = audio_file.read()
        
        # Convert audio to WAV format using pydub
        audio = AudioSegment.from_file(BytesIO(audio_bytes))
        audio = audio.set_channels(1)  # Mono channel
        audio = audio.set_frame_rate(16000)  # Set frame rate to 16kHz
        
        wav_io = BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        
        # Save the WAV file
        with open(audio_path, 'wb') as f:
            f.write(wav_io.read())
        
        # Use SpeechRecognition to transcribe audio
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="en")  # Specify language if needed
        
        # Emotion detection
        emotions = emotion_classifier(text)
        # Extract top emotion
        top_emotion = max(emotions[0], key=lambda x: x['score'])
        
        # Summarization
        summary = summarizer(text, max_length=60, min_length=15, do_sample=False)[0]['summary_text']
        
        # Render results on the same page
        return render_template('index.html', transcription=text, emotion=top_emotion['label'], summary=summary, audio_file='static/audio/recording.wav')
    
    except sr.UnknownValueError:
        return render_template('index.html', error="Could not understand audio")
    except sr.RequestError as e:
        return render_template('index.html', error=f"Speech Recognition service error: {e}")
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
