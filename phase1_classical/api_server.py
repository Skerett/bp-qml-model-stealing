from fastapi import FastAPI
from pathlib import Path
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib
from phase1_classical import model as myModel
from phase1_classical.common import ApiMode
import torch

# Class for correct data check
class PredictRequest(BaseModel):
    image: List[float]

# Artifacts directory
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

# Instatiate api and model
app = FastAPI()
model = myModel.ClassicalModel()

# Load weights and pca
model.load_state_dict(torch.load(ARTIFACTS_DIR / "victim_model.pth", weights_only=True))
pca = joblib.load(ARTIFACTS_DIR / "pca.joblib")

model.eval()

# Endpoints
@app.post("/predict")
def predict(request: PredictRequest, mode: ApiMode = ApiMode.CONFIDENCE_SCORE):
    # Flatten image to numpy array (1, 28x28)
    imageArray = np.array(request.image, dtype=np.float32).reshape(1, -1)

    # Normalise image
    if imageArray.max() > 1.0:
        imageArray = imageArray / 255.0

    # Transform to vector
    imagePCA = pca.transform(imageArray)

    # Transforming to tensor
    imageTensor = torch.tensor(imagePCA, dtype=torch.float32)

    # Send it to model
    with torch.no_grad():
        output = model(imageTensor)
        probabilites = torch.softmax(output, dim=1)[0]

    if mode == ApiMode.HARD_LABEL:
        prediction = int(torch.argmax(probabilites).item())
        return {"prediction": prediction}
    elif mode == ApiMode.ROUNDED:
        rounded = [round(p,2) for p in probabilites.tolist()]
        return {"prediction": rounded}
    elif mode == ApiMode.NOISY:
        noise = np.random.normal(0, 0.05, size=4)
        noisy_probs = np.clip(probabilites.numpy() + noise, 0, 1)
        noisy_probs = noisy_probs / noisy_probs.sum()
        return {"prediction": noisy_probs.tolist()}
    elif mode == ApiMode.CONFIDENCE_SCORE:
        return {"prediction": probabilites.tolist()}
