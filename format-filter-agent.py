#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Format-Filter Agent - Symptom Data Formatting
This agent takes a symptom summary and generates structured JSON output with boolean symptoms.
"""

import json
import os
from typing import Dict, Any, List
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

class FormatFilterAgent:
    """Agent responsible for formatting symptom summaries into structured JSON."""
    
    def __init__(self):
        self.llm = llm
        # Load the symptoms list
        try:
            with open("list.txt", "r") as f:
                content = f.read().strip()
                if content.startswith('[') and content.endswith(']'):
                    self.symptoms_list = eval(content)
                else:
                    # If it's not a list format, split by lines
                    self.symptoms_list = [line.strip().strip("',") for line in content.split('\n') if line.strip()]
        except Exception as e:
            print(f"Error loading symptoms list: {e}")
            # Fallback to a basic symptom list
            self.symptoms_list = [
                'fever', 'cough', 'headache', 'fatigue', 'nausea', 'vomiting',
                'diarrhea', 'constipation', 'abdominal_pain', 'chest_pain',
                'back_pain', 'joint_pain', 'muscle_pain', 'dizziness', 'weakness'
            ]
    
    def create_system_prompt(self) -> str:
        """Create the system prompt with the full symptoms list."""
        
        symptoms_str = ', '.join(self.symptoms_list)
        
        return f"""
You are a medical data formatting agent. Use only the symptoms you find in the following list of symptoms: {symptoms_str}

Your task:
1. Read the symptom summary provided by the user
3. Identify which symptoms from the list are present
4. Convert natural language symptom descriptions into a Python list of strings, where each string is a symptom from the list.

Example:
User summary: "The patient has a fever, a bad cough, and says they have a headache."
Your response: ["fever", "cough", "headache"]

IMPORTANT RULES:
1. ONLY include symptoms that are explicitly mentioned or clearly implied.
2. If no symptoms from the list are found, return an empty list.
3. Use exact symptom names from the provided list.
4. Respond ONLY with the Python list of strings.
5. Be more inclusive in your matching. For example, "feeling hot" can be mapped to "fever".
"""
    
    def format_symptoms(self, symptom_summary: str) -> List[str]:
        """
        Convert symptom summary to a list of symptom strings.
        
        Args:
            symptom_summary: Natural language description of symptoms
            
        Returns:
            A list of symptom strings.
        """
        try:
            system_message = SystemMessage(content=self.create_system_prompt())
            user_message = HumanMessage(content=f"""
            Please convert the following symptom summary into a Python list of symptom strings:
            
            {symptom_summary}
            
            Respond with ONLY the Python list, no additional text.
            """)
            
            messages = [system_message, user_message]
            print("Invoking LLM with messages:")
            print(messages)
            response = self.llm.invoke(messages)
            response_text = response.content.strip()
            print("LLM response:")
            print(response_text)
            
            # Use eval to parse the list string into a Python list
            symptoms = eval(response_text)
            
            if isinstance(symptoms, list):
                # Filter to ensure only valid symptoms are returned
                valid_symptoms = [s for s in symptoms if s in self.symptoms_list]
                return valid_symptoms
            else:
                return self._create_fallback_format(symptom_summary, "LLM did not return a list")
            
        except Exception as e:
            print(f"Error formatting symptoms: {e}")
            return self._create_fallback_format(symptom_summary, str(e))
    
    def _create_fallback_format(self, original_text: str, error_reason: str) -> List[str]:
        """Create a fallback format when parsing fails."""
        print(f"Using fallback due to: {error_reason}")
        symptoms_dict = {
            'fever': ['fever', 'temperature', 'hot'],
            'cough': ['cough', 'coughing'],
            'headache': ['headache', 'head pain'],
            'fatigue': ['tired', 'fatigue', 'exhausted', 'weak'],
            'nausea': ['nausea', 'sick', 'queasy'],
            'vomiting': ['vomiting', 'throwing up', 'vomit'],
            'diarrhea': ['diarrhea', 'loose stools'],
            'abdominal_pain': ['stomach pain', 'belly pain', 'abdominal pain'],
            'chest_pain': ['chest pain', 'heart pain'],
            'dizziness': ['dizzy', 'dizziness', 'lightheaded']
        }
        
        found_symptoms = []
        text_lower = original_text.lower()
        
        for symptom, keywords in symptoms_dict.items():
            if symptom in self.symptoms_list:
                if any(keyword in text_lower for keyword in keywords):
                    found_symptoms.append(symptom)
                    
        return found_symptoms

# Global instance
format_filter_agent = FormatFilterAgent()