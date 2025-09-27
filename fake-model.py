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
        age = symptoms_data.get("age", "unknown")
        gender = symptoms_data.get("gender", "unknown")
        
        system_prompt = """
        You are a medical diagnosis AI model. Based on the provided symptoms, age, and gender,
        provide a medical diagnosis with confidence scores.
        
        Return your response in the following JSON format:
        {
            "primary_diagnosis": "most likely disease/condition",
            "confidence": 0.85,
            "secondary_diagnoses": [
                {"condition": "alternative diagnosis", "confidence": 0.65},
                {"condition": "another possibility", "confidence": 0.45}
            ],
            "severity": "mild/moderate/severe",
            "recommended_actions": ["action1", "action2"],
            "requires_further_investigation": true/false,
            "uncertainty_factors": ["factor1", "factor2"]
        }
        
        Be medical accurate but acknowledge the limitations of AI diagnosis.
        """
        
        user_prompt = f"""
        Patient Information:
        - Age: {age}
        - Gender: {gender}
        - Active Symptoms: {', '.join(active_symptoms) if active_symptoms else 'None reported'}
        
        Please provide a medical diagnosis based on this information.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            # Try to parse JSON from response
            response_text = response.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            diagnosis_result = json.loads(response_text)
            
            # Ensure all required fields are present
            if "confidence" not in diagnosis_result:
                diagnosis_result["confidence"] = 0.5
            if "requires_further_investigation" not in diagnosis_result:
                diagnosis_result["requires_further_investigation"] = len(active_symptoms) < 3
                
            return diagnosis_result
            
        except json.JSONDecodeError:
            # Fallback response if JSON parsing fails
            return {
                "primary_diagnosis": "Insufficient data for reliable diagnosis",
                "confidence": 0.2,
                "secondary_diagnoses": [],
                "severity": "unknown",
                "recommended_actions": ["Consult with a medical professional"],
                "requires_further_investigation": True,
                "uncertainty_factors": ["Limited symptom information", "AI diagnosis limitations"]
            }
        except Exception as e:
            return {
                "error": f"Model prediction failed: {str(e)}",
                "primary_diagnosis": "Error in diagnosis",
                "confidence": 0.0,
                "requires_further_investigation": True
            }

# Global instance to be used by other modules
medical_model = FakeMedicalModel()

def get_diagnosis(symptoms_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to get diagnosis from the model.
    """
    return medical_model.predict_diagnosis(symptoms_data)
