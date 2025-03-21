import logging
import os
from asyncio import get_event_loop
import bcrypt
import jwt
import openai
import uvicorn
import asyncio
from TTS import tts
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, UploadFile
from pydantic import BaseModel
from redis import Redis
from requests import Session
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
from starlette.websockets import WebSocketDisconnect
from textblob import TextBlob
from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
openai.api_key=os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

RDS_USERNAME = os.getenv("RDS_USERNAME")
RDS_PASSWORD = os.getenv("RDS_PASSWORD")
RDS_HOST = os.getenv("RDS_HOST")
RDS_PORT = os.getenv("RDS_PORT", "5432")
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

# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
#
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

class TherapyModel:
    async def generate_response_async(self, user_input: str, history: list = [], websocket=None):
        system_prompt = "You are a therapist with a calm, reassuring tone. " \
                        "Respond empathetically, reflecting the user’s emotions without judgment."

        sentiment = self.analyze_sentiment(' '.join(history) + user_input)
        if sentiment['polarity'] > 0.5:
            system_prompt += " The user seems positive, continue to support their optimism."
        elif sentiment['polarity'] < -0.5:
            system_prompt += " The user seems upset, offer support and validation for their feelings."

        messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            messages.append({"role": "user", "content": msg})
        messages.append({"role": "user", "content": user_input})

        try:
            loop = get_event_loop()
            with ThreadPoolExecutor() as pool:
                response_data = await loop.run_in_executor(
                    pool,
                    lambda: openai.ChatCompletion.create(
                        model="gpt-4o",
                        messages=messages,
                        max_tokens=200,
                        temperature=1,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stream=False  # Disable streaming
                    )
                )
                full_response = response_data.choices[0].message['content'].strip()
                final_sentiment = self.analyze_sentiment(full_response)
                print(f"Sending response: {full_response}, Sentiment: {final_sentiment}")  # Debug log
                if websocket:
                    await websocket.send_json({
                        "response": full_response or "I’m here to help. Can you tell me more?",
                        "sentiment": final_sentiment,
                        "is_final": True  # Always final since no streaming
                    })
                return full_response, sentiment

        except Exception as e:
            print(f"Error with OpenAI API or WebSocket: {e}")
            error_response = "I'm having trouble understanding right now. Can you share more?"
            if websocket:
                await websocket.send_json({
                    "response": error_response,
                    "sentiment": {"polarity": 0, "subjectivity": 0},
                    "is_final": True
                })
            return error_response, {"polarity": 0, "subjectivity": 0}
    # def create_prompt(self, user_input, history):
    #     # Construct the prompt by concatenating previous conversation context
    #     history_text = "\n".join(history)
    #     prompt = f"{history_text}\nUser: {user_input}\nTherapist:"
    #     return prompt


    def analyze_sentiment(self, response):
        #create a textBlob object
        analysis = TextBlob(response)
        # Analyze polarity (-1 to 1) and subjectivity (0 to 1)
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity

        # Return structured sentiment data
        return {
            "polarity": polarity,  # Use for emotional tone assessment
            "subjectivity": subjectivity  # Use for determining personal opinion level
        }

class LoginRequest(BaseModel):
    username:str
    password:str

# @app.post("/login")
# async def login(request:LoginRequest):
#     hashed_pw = bcrypt.hashpw("password".encode(), bcrypt.gensalt())
#     if request.username != "user" or not bcrypt.checkpw(request.password.encode(), hashed_pw):
#         raise HTTPException(status_code=401, detail="Invalid Credentials")
#     token = jwt.encode({"sub": request.username}, key, alg="HS256")
#     return {"token":token}
#
# @app.post("/users/{username}/mood")
# async def log_mood(username:str, mood:str):
#     session = Session()
#     user = session.query(User).filter_by(username=username).first()
#     if not user:
#         #create new user.
#         user = User(username=username, mood_history="[]", streak_count=0)
#     user.mood_history = f"{user.mood_history[:-1]}, '{mood}']" if user.mood_history else f"['{mood}']"
#     user.streak_count += 1
#     session.add(user)
#     session.commit()
#     session.close()
#     return {"streak": user.streak_count}

# @app.get("/users/{username}")
# async def get_user(username: str):
#     session = Session()
#     user = session.query(User).filter_by(username=username).first()
#     session.close()
#     if not user:
#         raise HTTPException(status_code=404, detail="User not found")
#     return {"username": user.username, "mood_history": user.mood_history, "streak": user.streak_count}
#
# @app.post("/stt")
# async def speech_to_text(audio: UploadFile):
#     if not whisper_model:
#         return {"text": "Mock speech input"}
#     audio_data = await audio.read()
#     with open("temp.wav", "wb") as f:
#         f.write(audio_data)
#     try:
#         result = whisper_model.transcribe("temp.wav")
#         os.remove("temp.wav")
#         return {"text": result["text"]}
#     except Exception as e:
#         os.remove("temp.wav")
#         raise HTTPException(status_code=500, detail=f"STT failed: {str(e)}")

# @app.post("/tts")
# async def text_to_speech(text: str):
#     output_file = f"output_{text[:10].replace(' ', '_')}.wav"
#     tts.tts_to_file(text=text, file_path=output_file)
#     return {"audio_file": output_file}

# --- WebSocket Endpoint ---
therapy_model = TherapyModel()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            logging.debug(f"Received data: {data}")

            user_id = data.get("user_id", "user2")
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

            await therapy_model.generate_response_async(user_input, history,websocket)
            # logging.debug(f"Sentiment polarity: {sentiment['polarity']}")
            # After streaming, update user streak if applicable
            with Session() as session:
                user = session.query(User).filter_by(username=user_id).first()
                if user and "mood" in data:
                    user.streak_count += 1
                    session.add(user)
                    session.commit()
                    # Send streak update after final response
                    await websocket.send_json({
                        "streak": user.streak_count,
                        "is_final": True
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
    uvicorn.run(app, host="localhost", port=8003)