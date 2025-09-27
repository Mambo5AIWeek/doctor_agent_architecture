from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
from pydantic import BaseModel
from typing import List, Union, Tuple
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import os

class DiseasePredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(DiseasePredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.network(x)


class SymptomsRequest(BaseModel):
    symptoms: List[str]
    confidence_threshold: float = 0.0


class PredictionResponse(BaseModel):
    predictions: Union[List[Tuple[str, float]], str]
    unrecognized_symptoms: List[str] = []


# Global variables for model and metadata
model = None
feature_columns = None
target_columns = None

@lru_cache
def load_model_and_metadata(llamando: bool):
    global model, feature_columns, target_columns

    # Load the trained model
    if not os.path.exists("disease_predictor_model.pth"):
        raise FileNotFoundError("Model file 'disease_predictor_model.pth' not found")

    # You'll need to save these during training and load them here
    # For now, I'll assume you have these saved as pickle files
    try:
        with open("feature_columns.pkl", "rb") as f:
            feature_columns = pickle.load(f)
        with open("target_columns.pkl", "rb") as f:
            target_columns = pickle.load(f)
    except FileNotFoundError:
        # If pickle files don't exist, you'll need to recreate them from your training data
        raise FileNotFoundError(
            "Feature and target columns metadata files not found. Please create 'feature_columns.pkl' and 'target_columns.pkl'")

    # Initialize model with correct dimensions
    input_size = len(feature_columns)
    output_size = len(target_columns)

    model = DiseasePredictor(input_size, output_size)
    model.load_state_dict(torch.load("disease_predictor_model.pth", map_location=torch.device('cpu')))
    model.eval()


def predict(symptoms: List[str], confidence_threshold: float = 0.0):
    load_model_and_metadata(True)
    """
    Predice una enfermedad a partir de una lista de síntomas.
    """
    model.eval()

    # Crear un vector de ceros con la longitud de las características de entrada
    input_vector = np.zeros(len(feature_columns))

    unrecognized_symptoms = []

    # Codificar los síntomas de entrada en el vector
    for symptom in symptoms:
        clean_symptom = symptom.strip().replace(' ', '_')
        try:
            # Encontrar el índice del síntoma y poner un 1 en esa posición
            idx = sorted(feature_columns.to_list()).index(clean_symptom)
            input_vector[idx] = 1
        except ValueError:
            # Si el síntoma no se encuentra en las columnas, se ignora
            unrecognized_symptoms.append(symptom)

    # Convertir el vector a un tensor de PyTorch y añadir una dimensión de lote
    input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)

    # Realizar la predicción
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits)

    # Obtener las enfermedades con las probabilidades más altas
    top_probabilities, top_indices = torch.topk(probabilities, 10, dim=1)

    results = []
    for prob, idx in zip(top_probabilities[0], top_indices[0]):
        if prob.item() >= confidence_threshold:
            disease = target_columns[idx.item()].replace('disease_', '')
            results.append((disease, prob.item()))

    if not results:
        return "No se pudo determinar una enfermedad con suficiente confianza. Por favor, consulte a un médico.", unrecognized_symptoms

    return results, unrecognized_symptoms
