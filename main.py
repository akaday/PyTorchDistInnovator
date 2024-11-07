from fastapi import FastAPI, HTTPException
import torch
from torch import nn
from pydantic import BaseModel

app = FastAPI()

class InputData(BaseModel):
    data: list[float]  # Ensure the data list contains floats

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

try:
    model = SimpleModel()
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    model.eval()
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")

@app.post("/predict/")
async def predict(input_data: InputData):
    try:
        data = input_data.data
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # Ensure data is a 2D tensor
        with torch.no_grad():
            prediction = model(data)
        return {"prediction": prediction.item()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
