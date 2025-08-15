from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class BusinessRequest(BaseModel):
    business_description: str

@app.get("/")
def root():
    return {"message": "Domain Generator API Working!", "status": "running"}

@app.post("/generate-domains")
def generate_domains(request: BusinessRequest):
    return {
        "domains": ["test.com", "example.net", "demo.org"],
        "status": "success",
        "business": request.business_description
    }

@app.get("/health")
def health():
    return {"status": "healthy", "api": "running"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
