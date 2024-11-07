from fastapi import FastAPI
import torch
from torch import nn
from pydantic import BaseModel
import numpy as np

app = FastAPI()

class InputData(BaseModel):
    data: list

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

model = SimpleModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

@app.post("/predict/")
async def predict(input_data: InputData):
    data = torch.tensor([input_data.data])
    with torch.no_grad():
        prediction = model(data)
    return {"prediction": prediction.item()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
