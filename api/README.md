# Domain Name Generation API

AI-powered domain name generation using fine-tuned **Qwen2.5-3B-Instruct** model.

## Features

- **Fine-tuned Model Integration**: LoRA fine-tuned model for better domain generation  
- **RESTful API**: FastAPI-based endpoints with automatic documentation  
- **Real-time Generation**: Generate domain names on-demand via HTTP requests  
- **Quality Scoring**: Built-in quality assessment for generated domains  
- **Health Monitoring**: Model health and uptime checks  
- **Interactive Documentation**: Swagger UI for easy testing  

---

## API Endpoints

### 1. **GET /** – API Status
Returns basic API information.

**Response:**
```json
{
  "message": "Domain Name Generation API",
  "status": "running",
  "endpoints": ["/generate-domains", "/health", "/api-info", "/docs"]
}
```

---

### 2. **POST /generate-domains** – Generate Domain Names
Main endpoint for generating domain names.

**Request Body:**
```json
{
  "business_description": "organic coffee shop downtown",
  "num_domains": 3,
  "category": "general"
}
```

**Response:**
```json
{
  "domains": ["organicbeanscafe.com", "downtowncoffee.org", "freshbreworganic.net"],
  "status": "success",
  "business_description": "organic coffee shop downtown",
  "generation_time": 12.34,
  "model_used": "fine_tuned",
  "quality_score": 8.5
}
```

---

### 3. **GET /health** – Health Check
Returns health status of the API and model.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "last_check": "2025-08-16T10:42:11.120Z"
}
```

---

### 4. **GET /api-info** – Model Information
Detailed information about the loaded model.

**Response:**
```json
{
  "model_type": "Qwen2.5-3B-Instruct",
  "fine_tuned": true,
  "device": "mps",
  "parameters": 3089625088,
  "trainable_parameters": 3686400
}
```

---

### 5. **GET /docs** – Interactive Documentation
Swagger UI for testing endpoints.

---

## Installation & Setup

### Prerequisites
- Python 3.10+  
- Fine-tuned model files  
- At least 8 GB RAM  

### 1. Install Dependencies
```bash
cd api
pip install -r requirements.txt
```

### 2. Verify Model Files
Ensure these directories exist in the **current working directory**:
- `./qwen2.5-3b-instruct/` – Base model  
- `./fine_tuned_model_stable_backup/` – Fine-tuned model  

### 3. Start the API
```bash
# Option 1: Direct
python main.py

# Option 2: Startup script
python start_api.py
```

---

## Testing

### 1. Start the Server
```bash
cd api
python start_api.py
```

### 2. Test Endpoints
```bash
# Health check
curl http://localhost:8000/health
```

### 3. Swagger UI
Open `http://localhost:8000/docs` in browser.

---

## Example Usage

### Python Client
```python
import requests

response = requests.post(
    "http://localhost:8000/generate-domains",
    json={
        "business_description": "AI software startup",
        "num_domains": 5
    }
)

if response.ok:
    data = response.json()
    print("Generated domains:", data["domains"])
    print("Quality score:", data["quality_score"])
```

### cURL Example
```bash
curl -X POST "http://localhost:8000/generate-domains" \
     -H "Content-Type: application/json" \
     -d '{"business_description": "yoga studio with meditation classes", "num_domains": 3}'
```

---

## Configuration

### Environment Variables
- `PORT`: API port (default: 8000)  
- `HOST`: API host (default: 0.0.0.0)  

### Model Parameters
- **Temperature**: 0.7  
- **Max Tokens**: 80  
- **Top-p**: 0.9  
- **Top-k**: 50  

---

## Performance

### Generation Times
- Base model: ~10–12 minutes/request  
- Fine-tuned model: ~10–12 minutes/request  
- Memory usage: ~9GB  

### Quality Scores
- Base Model: 6.3/10  
- Fine-tuned Model: 7.7/10 (+21% improvement)  

---

## Error Handling

### Common Errors
- **400**: Invalid request  
- **500**: Internal server error  
- **503**: Service unavailable  

**Error Format:**
```json
{
  "detail": "Error description",
  "status_code": 400
}
```

---

## Deployment

- Use Gunicorn or Uvicorn for production  
- Add authentication if needed  
- Implement rate limiting  
- Configure monitoring & logging  
