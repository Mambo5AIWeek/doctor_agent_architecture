#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation Agent - Medical Diagnosis Evaluation
This agent evaluates medical diagnosis using the ML model and internet searches.
"""

import json
import os
from typing import Dict, Any, List
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
# Import fake model with dynamic import
import importlib.util
spec = importlib.util.spec_from_file_location("fake_model", "fake-model.py")
fake_model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fake_model_module)
get_diagnosis = fake_model_module.get_diagnosis

load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# Initialize search tool
search_tool = DuckDuckGoSearchRun()

# Define tools for the evaluation agent
def medical_model_tool(symptoms_json: str) -> str:
    """
    Call the medical diagnosis model with symptoms data.
    
    Args:
        symptoms_json: JSON string containing symptoms, age, and gender
    
    Returns:
        JSON string with diagnosis results
    """
    try:
        symptoms_data = json.loads(symptoms_json)
        diagnosis = get_diagnosis(symptoms_data)
        return json.dumps(diagnosis, indent=2)
    except Exception as e:
        return f"Error calling medical model: {str(e)}"

def internet_search_tool(query: str) -> str:
    """
    Search the internet for medical information.
    
    Args:
        query: Search query string
    
    Returns:
        Search results as string
    """
    try:
        results = search_tool.run(query)
        return results
    except Exception as e:
        return f"Search failed: {str(e)}"

# Create tools list
tools = [
    Tool(
        name="medical_model",
        func=medical_model_tool,
        description="Call the medical diagnosis model with symptoms data in JSON format. Input should be a JSON string with symptoms (boolean dict), age, and gender."
    ),
    Tool(
        name="internet_search",
        func=internet_search_tool,
        description="Search the internet for medical information, symptoms, conditions, or diagnostic criteria. Use this to verify or supplement model predictions."
    )
]

# System prompt for the evaluation agent
evaluation_system_prompt = """
You are a medical evaluation agent responsible for analyzing diagnosis results and determining if further inquiry is needed.

Your tasks:
1. Use the medical_model tool to get a diagnosis based on the provided symptoms data
2. Analyze the model's confidence and results
3. Use internet_search if needed to verify or supplement the diagnosis
4. Determine if the diagnosis is sufficient or if more information is needed

Decision criteria:
- If model confidence is >= 0.7 and primary diagnosis is clear: Provide final diagnosis
- If confidence is < 0.7 or diagnosis is uncertain: Request further inquiry
- If conflicting information found in search: Request clarification

Always respond with a JSON object containing:
{
    "decision": "final_diagnosis" or "further_inquiry",
    "diagnosis": {
        "primary_condition": "condition name",
        "confidence": 0.85,
        "explanation": "reasoning for diagnosis",
        "recommendations": ["recommendation1", "recommendation2"]
    },
    "inquiry_needed": {
        "specific_questions": ["question1", "question2"],
        "areas_to_explore": ["area1", "area2"]
    },
    "reasoning": "explanation of why this decision was made"
}

IMPORTANT: Always emphasize that this is not professional medical advice and recommend consulting healthcare professionals.
"""

def create_evaluation_prompt(state):
    """Create the prompt for the evaluation agent."""
    messages = state.get("messages", [])
    
    prompt = [{
        "role": "system", 
        "content": evaluation_system_prompt
    }] + messages
    
    return prompt

# Create the evaluation agent
evaluation_agent = create_react_agent(
    llm,
    tools=tools,
    prompt=create_evaluation_prompt
)

class EvaluationAgent:
    """Wrapper class for the evaluation agent."""
    
    def __init__(self):
        self.agent = evaluation_agent
    
    def evaluate_diagnosis(self, symptoms_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate diagnosis based on symptoms data.
        
        Args:
            symptoms_data: Dictionary with symptoms, age, and gender
            
        Returns:
            Dictionary with evaluation results and decision
        """
        try:
            # Convert to JSON for the agent
            symptoms_json = json.dumps(symptoms_data)
            
            # Create input message
            input_message = f"""
            Please evaluate the following symptoms data and provide a diagnosis evaluation:
            
            {symptoms_json}
            
            Use the medical_model tool to get a diagnosis, then use internet_search if needed to verify or supplement the results. 
            Provide your final evaluation in the specified JSON format.
            """
            
            # Invoke the agent
            state = {"messages": [{"role": "user", "content": input_message}]}
            result = self.agent.invoke(state)
            
            # Extract the final message
            final_message = result["messages"][-1].content
            
            # Try to parse JSON response
            try:
                if "```json" in final_message:
                    json_start = final_message.find("```json") + 7
                    json_end = final_message.find("```", json_start)
                    json_str = final_message[json_start:json_end].strip()
                else:
                    # Look for JSON-like structure
                    json_start = final_message.find("{")
                    json_end = final_message.rfind("}") + 1
                    json_str = final_message[json_start:json_end]
                
                evaluation_result = json.loads(json_str)
                return evaluation_result
                
            except (json.JSONDecodeError, ValueError):
                # Fallback if JSON parsing fails
                return {
                    "decision": "further_inquiry",
                    "diagnosis": {
                        "primary_condition": "Unable to parse evaluation",
                        "confidence": 0.0,
                        "explanation": "Evaluation agent response could not be parsed",
                        "recommendations": ["Retry evaluation with more specific symptoms"]
                    },
                    "inquiry_needed": {
                        "specific_questions": ["Could you provide more detailed symptoms?"],
                        "areas_to_explore": ["General health assessment"]
                    },
                    "reasoning": "Technical error in evaluation process"
                }
                
        except Exception as e:
            return {
                "decision": "further_inquiry",
                "diagnosis": {
                    "primary_condition": "Evaluation Error",
                    "confidence": 0.0,
                    "explanation": f"Error during evaluation: {str(e)}",
                    "recommendations": ["Contact technical support"]
                },
                "inquiry_needed": {
                    "specific_questions": ["Please describe your symptoms again"],
                    "areas_to_explore": ["Basic symptom assessment"]
                },
                "reasoning": f"Technical error: {str(e)}"
            }

# Global instance
evaluation_agent_instance = EvaluationAgent()

if __name__ == "__main__":
    # Test the evaluation agent
    test_symptoms = {
        "symptoms": {
            "fever": True,
            "cough": True,
            "headache": True,
            "fatigue": True
        },
        "age": 35,
        "gender": "female"
    }
    
    result = evaluation_agent_instance.evaluate_diagnosis(test_symptoms)
    print(json.dumps(result, indent=2))