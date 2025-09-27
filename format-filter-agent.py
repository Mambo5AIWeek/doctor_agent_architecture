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
You are a medical data formatting agent. Your task is to convert natural language symptom descriptions into a structured JSON format.

Available symptoms list (ONLY use symptoms from this list):
{symptoms_str}

Your task:
1. Read the symptom summary provided by the user
2. Extract age and gender information if available
3. Identify which symptoms from the list are present
4. Create a JSON object with the following structure:

{{
    "symptoms": {{
        "symptom_1": true/false,
        "symptom_2": true/false,
        ... (include ALL symptoms from the list, set to true only if explicitly mentioned or clearly implied)
    }},
    "age": "extracted age or 'unknown'",
    "gender": "extracted gender or 'unknown'",
    "confidence": 0.0-1.0,
    "needs_more_info": true/false,
    "missing_info": ["list of information that would be helpful"],
    "extracted_symptoms_summary": "brief summary of identified symptoms"
}}

IMPORTANT RULES:
1. ONLY set symptoms to 'true' if they are explicitly mentioned or clearly implied in the text
2. ALL other symptoms should be set to 'false'
3. Be conservative - if unsure about a symptom, set it to 'false'
4. Set 'needs_more_info' to true if the information is insufficient for reliable formatting
5. Use exact symptom names from the provided list (handle variations like 'stomach pain' -> 'abdominal_pain')
6. Extract age as a number if possible, otherwise use descriptive terms like 'elderly', 'young adult', etc.
7. For gender, use 'male', 'female', or 'unknown'

Example symptom mappings:
- "stomach ache" or "belly pain" -> "abdominal_pain": true
- "feeling tired" or "exhausted" -> "fatigue": true
- "throwing up" -> "vomiting": true
- "runny nose" -> "runny_nose": true
- "trouble breathing" -> "breathlessness": true
"""
    
    def format_symptoms(self, symptom_summary: str) -> Dict[str, Any]:
        """
        Convert symptom summary to structured JSON format.
        
        Args:
            symptom_summary: Natural language description of symptoms
            
        Returns:
            Dictionary with structured symptom data
        """
        try:
            system_message = SystemMessage(content=self.create_system_prompt())
            user_message = HumanMessage(content=f"""
            Please convert the following symptom summary into the structured JSON format:
            
            {symptom_summary}
            
            Respond with ONLY the JSON object, no additional text.
            """)
            
            messages = [system_message, user_message]
            response = self.llm.invoke(messages)
            response_text = response.content.strip()
            
            # Clean up response to extract JSON
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "").strip()
            
            # Parse the JSON response
            formatted_data = json.loads(response_text)
            
            # Validate the structure
            self._validate_format(formatted_data)
            
            return formatted_data
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return self._create_fallback_format(symptom_summary, "JSON parsing failed")
        except Exception as e:
            print(f"Error formatting symptoms: {e}")
            return self._create_fallback_format(symptom_summary, str(e))
    
    def _validate_format(self, data: Dict[str, Any]) -> None:
        """Validate the formatted data structure."""
        required_keys = ['symptoms', 'age', 'gender', 'confidence', 'needs_more_info']
        
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
        
        # Ensure all symptoms from the list are present
        symptoms = data.get('symptoms', {})
        for symptom in self.symptoms_list:
            if symptom not in symptoms:
                symptoms[symptom] = False
        
        # Remove any symptoms not in the official list
        valid_symptoms = {k: v for k, v in symptoms.items() if k in self.symptoms_list}
        data['symptoms'] = valid_symptoms
    
    def _create_fallback_format(self, original_text: str, error_reason: str) -> Dict[str, Any]:
        """Create a fallback format when parsing fails."""
        # Initialize all symptoms as False
        symptoms_dict = {symptom: False for symptom in self.symptoms_list}
        
        # Try basic keyword matching for common symptoms
        text_lower = original_text.lower()
        
        # Simple keyword matching
        symptom_keywords = {
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
        
        for symptom, keywords in symptom_keywords.items():
            if symptom in self.symptoms_list:
                if any(keyword in text_lower for keyword in keywords):
                    symptoms_dict[symptom] = True
        
        return {
            "symptoms": symptoms_dict,
            "age": "unknown",
            "gender": "unknown",
            "confidence": 0.3,  # Low confidence for fallback
            "needs_more_info": True,
            "missing_info": ["Clear symptom description", "Age and gender information"],
            "extracted_symptoms_summary": f"Fallback parsing due to: {error_reason}",
            "error": error_reason,
            "original_text": original_text
        }
    
    def assess_information_adequacy(self, formatted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess if the formatted information is adequate for diagnosis.
        
        Args:
            formatted_data: The formatted symptom data
            
        Returns:
            Assessment results with recommendations
        """
        symptoms = formatted_data.get('symptoms', {})
        active_symptoms = [k for k, v in symptoms.items() if v]
        confidence = formatted_data.get('confidence', 0.0)
        
        # Criteria for adequacy
        min_symptoms = 2
        min_confidence = 0.6
        
        is_adequate = (
            len(active_symptoms) >= min_symptoms and
            confidence >= min_confidence and
            not formatted_data.get('needs_more_info', False)
        )
        
        assessment = {
            "is_adequate": is_adequate,
            "active_symptoms_count": len(active_symptoms),
            "active_symptoms": active_symptoms,
            "confidence_level": confidence,
            "recommendations": []
        }
        
        if not is_adequate:
            if len(active_symptoms) < min_symptoms:
                assessment["recommendations"].append("Need more symptom information")
            if confidence < min_confidence:
                assessment["recommendations"].append("Need clearer symptom descriptions")
            if formatted_data.get('needs_more_info', False):
                assessment["recommendations"].append("Additional information required as indicated")
        
        return assessment

# Global instance
format_filter_agent = FormatFilterAgent()

if __name__ == "__main__":
    # Test the format filter agent
    test_summary = """
    Patient is a 35-year-old female experiencing fever for 3 days, 
    persistent cough, severe headache, and extreme fatigue. 
    She also mentioned feeling nauseous occasionally.
    """
    
    result = format_filter_agent.format_symptoms(test_summary)
    print("Formatted Data:")
    print(json.dumps(result, indent=2))
    
    assessment = format_filter_agent.assess_information_adequacy(result)
    print("\nAdequacy Assessment:")
    print(json.dumps(assessment, indent=2))