# File: camps_fastapi/app/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import langchain
from langchain.schema import LLMResult
from crowdai import CrowdAILLM

# Initialize the Fast API app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
DATABASE_URL = os.getenv("DATABASE_URL")
CROWD_AI_API_KEY = os.getenv("CROWD_AI_API_KEY")

# Initialize the Crowd AI model
if not CROWD_AI_API_KEY:
    raise HTTPException(status_code=500, detail="CROWD_AI_API_KEY is not set")

# Create the LLM
llm = CrowdAILLM(api_key=CROWD_AI_API_KEY)

# Create a Langchain pipeline
pipeline = langchain.LLMChain(llm=llm)

# Database connection
import asyncpg
import aiopg

async def connect_db():
    conn = await asyncpg.connect(DATABASE_URL)
    return conn

# Create a simple in-memory cache for generated content
generated_content_cache = {}

@app.post("/generate-post")
async def generate_social_media_post(
    brand_name: str,
    campaign_name: str
):
    """Generate a social media post for a brand campaign"""
    try:
        prompt = f"Generate a compelling social media post for {brand_name}'s {campaign_name} campaign."
        response = await pipeline.agenerate([prompt])
        result = await response.first()
        
        if result:
            generated_content_cache[f"{brand_name}_{campaign_name}"] = result
            return {"post": result}
        else:
            raise HTTPException(status_code=500, detail="Failed to generate post")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-content")
async def analyze_content(
    content: str
):
    """Analyze social media content"""
    try:
        prompt = f"Analyze this social media content: {content}"
        response = await pipeline.agenerate([prompt])
        result = await response.first()
        
        if result:
            return {"analysis": result}
        else:
            raise HTTPException(status_code=500, detail="Failed to analyze content")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "ok"}

# Initialize the database connection on startup
@app.on_event("startup")
async def startup_db_client():
    app.state.db = await connect_db()

# Close the database connection on shutdown
@app.on_event("shutdown")
async def shutdown_db_client():
    await app.state.db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
