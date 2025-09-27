#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fake Model Module - Placeholder for Deep Learning Model
This module simulates a medical diagnosis deep learning model using an LLM.
"""

import json
import os
from typing import Dict, List, Any
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
from model_inference import predict

load_dotenv()

class FakeMedicalModel:
    """
    Fake medical model that uses an LLM to simulate a deep learning model
    for medical diagnosis based on boolean symptoms.
    """
    
    def __init__(self):
        self.llm = AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        
        # Load the symptoms list
        with open("list.txt", "r") as f:
            self.symptoms_list = eval(f.read())
    
    def predict_diagnosis(self, symptoms_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict medical diagnosis based on symptoms data.
        
        Args:
            symptoms_data: Dictionary containing boolean symptoms, age, and gender
            
        Returns:
            Dictionary with diagnosis predictions and confidence scores
        """
        
        # Extract active symptoms (True values)
        active_symptoms = [symptom for symptom, value in symptoms_data.get("symptoms", {}).items() if value]

        result = predict(active_symptoms, 0.0)
        print(result)
        return result

# Global instance to be used by other modules
medical_model = FakeMedicalModel()

def get_diagnosis(symptoms_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to get diagnosis from the model.
    """
    return medical_model.predict_diagnosis(symptoms_data)
