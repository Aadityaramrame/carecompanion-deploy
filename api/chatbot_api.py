from fastapi import FastAPI, Request
from pydantic import BaseModel
from chatbot.chatbot_function import DataProcessor, Chatbot

app = FastAPI(
    title="CareCompanion Chatbot API",
    description="Retrieval-based chatbot API using TF-IDF and cosine similarity",
    version="1.0"
)

# Load chatbot once at startup
data_processor = DataProcessor()
chatbot = Chatbot(data_processor)

# Define request format
class Query(BaseModel):
    question: str

# Define response format
class Response(BaseModel):
    matched_question: str
    answer: str
    source: str

@app.post("/chat", response_model=Response)
async def get_chatbot_response(query: Query):
    matched_question, answer, source = chatbot.get_response(query.question)
    return {
        "matched_question": matched_question,
        "answer": answer,
        "source": source
    }
