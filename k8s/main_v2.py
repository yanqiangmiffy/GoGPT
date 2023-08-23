from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()
@app.post("/")
async def create_item(request: Request):
    json_post_raw = await request.json()
    answer = {"欢迎使用gomate"}
    return answer
if __name__ == "__main__":
    uvicorn.run(app="main:app", host='0.0.0.0', port=8090)
