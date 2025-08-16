#!/usr/bin/env python3
"""
Test the Domain Name Generation API
Tests all endpoints and demonstrates functionality
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_api():
    """Test all API endpoints"""
    
    print("🧪 TESTING DOMAIN NAME GENERATION API")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1️⃣ Testing Health Check:")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Health: {data['status']}")
            print(f"   📊 Model Loaded: {data['model_loaded']}")
            print(f"   🤖 Model: {data['model_name']}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("   ❌ API not running. Start with: python main.py")
        return
    
    # Test 2: Root endpoint
    print("\n2️⃣ Testing Root Endpoint:")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Status: {data['status']}")
            print(f"   📝 Message: {data['message']}")
            print(f"   🔗 Endpoints: {data['endpoints']}")
        else:
            print(f"   ❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Model info
    print("\n3️⃣ Testing Model Info:")
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        if response.status_code == 200:
            data = response.json()
            if "error" not in data:
                print(f"   ✅ Model Type: {data['model_type']}")
                print(f"   🔧 Fine-tuned: {data['fine_tuned']}")
                print(f"   💻 Device: {data['device']}")
                print(f"   📊 Parameters: {data['parameters']:,}")
                print(f"   🎯 Trainable: {data['trainable_parameters']:,}")
            else:
                print(f"   ⚠️  Model info: {data['error']}")
        else:
            print(f"   ❌ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Domain generation
    print("\n4️⃣ Testing Domain Generation:")
    
    test_cases = [
        {
            "business_description": "organic coffee shop downtown",
            "num_domains": 3
        },
        {
            "business_description": "AI software startup for business automation",
            "num_domains": 5
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n   📝 Test Case {i+1}: {test_case['business_description']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/generate-domains",
                json=test_case,
                timeout=120  # 2 minutes timeout for generation
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"      ✅ Status: {data['status']}")
                print(f"      🌐 Domains: {data['domains']}")
                print(f"      ⏱️  Time: {data['generation_time']:.2f}s")
                print(f"      🤖 Model: {data['model_used']}")
                print(f"      📊 Quality: {data['quality_score']:.1f}/10")
            else:
                print(f"      ❌ Generation failed: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"      📝 Error: {error_data.get('detail', 'Unknown error')}")
                except:
                    print(f"      📝 Error: {response.text}")
                    
        except requests.exceptions.Timeout:
            print(f"      ⏰ Timeout: Generation took too long")
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    # Test 5: API documentation
    print("\n5️⃣ Testing API Documentation:")
    try:
        response = requests.get(f"{BASE_URL}/docs")
        if response.status_code == 200:
            print(f"   ✅ Swagger UI available at: {BASE_URL}/docs")
        else:
            print(f"   ❌ Documentation failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print(f"\n🎉 API Testing Complete!")
    print(f"   🌐 API Base URL: {BASE_URL}")
    print(f"   📚 Documentation: {BASE_URL}/docs")
    print(f"   🔍 Interactive testing available at the docs URL")

def test_simple_request():
    """Simple test for quick verification"""
    print("\n🚀 Quick API Test:")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ API is running")
            print(f"   🤖 Model loaded: {data['model_loaded']}")
            
            if data['model_loaded']:
                print(f"   🎯 Ready for domain generation!")
            else:
                print(f"   ⚠️  Model not loaded yet")
        else:
            print(f"   ❌ API health check failed")
    except requests.exceptions.ConnectionError:
        print(f"   ❌ API not running. Start with: python main.py")

if __name__ == "__main__":
    print("🧪 Starting API Tests...")
    print("   Make sure the API is running: python main.py")
    print("   Then run this test script")
    
    # Wait a moment for API to be ready
    time.sleep(2)
    
    # Run tests
    test_api()
    
    print(f"\n💡 To start the API:")
    print(f"   cd api")
    print(f"   python main.py")
    print(f"   Then visit: http://localhost:8000/docs")
