import httpx
import json
import time

API = "http://127.0.0.1:8000"

# Query forecast (data already in DuckDB from previous upload)
print("=== Querying bitcoin forecast ===")
start = time.time()
resp = httpx.post(f"{API}/query", json={"question": "What will be price of bitcoin in 1 May 2026?", "mode": "auto"}, timeout=300)
elapsed = time.time() - start
print(f"Query status: {resp.status_code}")
data = resp.json()
print(f"Task type: {data.get('task_type')}")
print(f"Status: {data.get('status')}")
print(f"Confidence: {data.get('confidence')}")
print(f"Answer: {data.get('answer')[:500]}")
print(f"Time: {elapsed:.1f}s")

# Check artifacts
arts = data.get("artifacts", {})
if arts:
    pf = arts.get("point_forecast", [])
    print(f"\nForecast points: {len(pf)}")
    for p in pf[:3]:
        print(f"  {p}")
else:
    print("\nNo artifacts returned")
    
# Check warnings
warns = data.get("warnings", [])
if warns:
    print(f"\nWarnings:")
    for w in warns:
        print(f"  {w}")
