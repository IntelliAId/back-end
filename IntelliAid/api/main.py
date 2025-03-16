import openai
from TTS import tts
from fastapi import FastAPI, HTTPException, WebSocket, UploadFile
from pydantic import BaseModel
import jwt
import bcrypt
from requests import Session
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
from redis import Redis
from starlette.websockets import WebSocketDisconnect
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import whisper
from TTS.api import TTS
import os
import uvicorn
from dotenv import load_dotenv
import openai
import logging
from textblob import TextBlob
load_dotenv()
openai.api_key=os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

# url = utils.DATABASE_URL
# key = utils.SECRET_KEY
RDS_USERNAME = os.getenv("RDS_USERNAME")
RDS_PASSWORD = os.getenv("RDS_PASSWORD")
RDS_HOST = os.getenv("RDS_HOST")  # e.g., "calmnest-db.xxx.region.rds.amazonaws.com"
RDS_PORT = os.getenv("RDS_PORT", "5432")  # Default PostgreSQL port
RDS_DB_NAME = os.getenv("RDS_DB_NAME")
key = os.getenv("SECRET_KEY")

DATABASE_URL = f"postgresql://{RDS_USERNAME}:{RDS_PASSWORD}@{RDS_HOST}:{RDS_PORT}/{RDS_DB_NAME}"

engine = create_engine(DATABASE_URL, echo=True)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    mood_history = Column(String)
    streak_count = Column(Integer, default=0)

Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)

redis = Redis(host="localhost", port=6379, db=0)

def save_conversation(user_id: str, message: str):
    redis.rpush(f"conversation:{user_id}", message)

def get_conversation(user_id: str):
    history = redis.lrange(f"conversation:{user_id}", 0, -1)
    return [msg.decode('utf-8') if isinstance(msg, bytes) else msg for msg in history]

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    return {"polarity": polarity}

class TherapyModel:
    def generate_response(self, user_input: str, history: list = []):
        # Prepare the conversation history and user input for the model
        messages = [{"role": "system", "content": "You are a helpful therapist."}]

        # Add history
        for msg in history:
            messages.append({"role": "user", "content": msg})

        messages.append({"role": "user", "content": user_input})

        # Call OpenAI API
        try:
            response_data = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Chat model
                messages=messages,
                max_tokens=150,
                temperature=0.7
            )
            response = response_data.choices[0].message['content'].strip()
            sentiment = self.analyze_sentiment(response)
            return response, sentiment

        except Exception as e:
            print(f"Error with OpenAI API: {e}")
            return "I'm having trouble understanding right now. Can you share more?", {"score": 0}

    def create_prompt(self, user_input, history):
        # Construct the prompt by concatenating previous conversation context
        history_text = "\n".join(history)
        prompt = f"{history_text}\nUser: {user_input}\nTherapist:"
        return prompt

    def analyze_sentiment(self, response):
        # Implement sentiment analysis logic here
        return {"score": 0.8 if "good" in response.lower() else -0.5}

class LoginRequest(BaseModel):
    username:str
    password:str

@app.post("/login")
async def login(request:LoginRequest):
    hashed_pw = bcrypt.hashpw("password".encode(), bcrypt.gensalt())
    if request.username != "user" or not bcrypt.checkpw(request.password.encode(), hashed_pw):
        raise HTTPException(status_code=401, detail="Invalid Credentials")
    token = jwt.encode({"sub": request.username}, key, alg="HS256")
    return {"token":token}

@app.post("/users/{username}/mood")
async def log_mood(username:str, mood:str):
    session = Session()
    user = session.query(User).filter_by(username=username).first()
    if not user:
        #create new user.
        user = User(username=username, mood_history="[]", streak_count=0)
    user.mood_history = f"{user.mood_history[:-1]}, '{mood}']" if user.mood_history else f"['{mood}']"
    user.streak_count += 1
    session.add(user)
    session.commit()
    session.close()
    return {"streak": user.streak_count}

@app.get("/users/{username}")
async def get_user(username: str):
    session = Session()
    user = session.query(User).filter_by(username=username).first()
    session.close()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"username": user.username, "mood_history": user.mood_history, "streak": user.streak_count}

@app.post("/stt")
async def speech_to_text(audio: UploadFile):
    if not whisper_model:
        return {"text": "Mock speech input"}
    audio_data = await audio.read()
    with open("temp.wav", "wb") as f:
        f.write(audio_data)
    try:
        result = whisper_model.transcribe("temp.wav")
        os.remove("temp.wav")
        return {"text": result["text"]}
    except Exception as e:
        os.remove("temp.wav")
        raise HTTPException(status_code=500, detail=f"STT failed: {str(e)}")

@app.post("/tts")
async def text_to_speech(text: str):
    output_file = f"output_{text[:10].replace(' ', '_')}.wav"
    tts.tts_to_file(text=text, file_path=output_file)
    return {"audio_file": output_file}

# --- WebSocket Endpoint ---
therapy_model = TherapyModel()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            logging.debug(f"Received data: {data}")

            user_id = data.get("user_id", "user1")
            user_input = data.get("input") or data.get("mood")

            # Verify types
            if not isinstance(user_id, str):
                logging.error("user_id is not a string")
                continue

            if not isinstance(user_input, str):
                logging.error("user_input is not a string")
                continue

            save_conversation(user_id, user_input)
            history = get_conversation(user_id)
            logging.debug(f"Conversation history: {history}")

            response, sentiment = therapy_model.generate_response(user_input, history)
            # logging.debug(f"Sentiment polarity: {sentiment['polarity']}")
            with Session() as session:
                user = session.query(User).filter_by(username=user_id).first()
                if user and "mood" in data:
                    user.streak_count += 1
                    session.add(user)
                    session.commit()

            await websocket.send_json({
                "response": response,
                "sentiment": sentiment,
                "streak": user.streak_count if user else 0
            })

    except WebSocketDisconnect:
        print("Client disconnected")

    except Exception as e:
        logging.exception("An error occurred")
        await websocket.send_json({"error": str(e)})

    finally:
        await websocket.close()

@app.get("/test")
async def test():
    return {"Message:", "Success \n 200 Ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)