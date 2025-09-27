#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inquire Agent - Medical Information Inquiry
This agent generates targeted inquiry messages when insufficient data is available for diagnosis.
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

# Define tools for the inquire agent
def internet_search_tool(query: str) -> str:
    """
    Search the internet for medical information to help formulate better questions.
    
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

def medical_reference_search(condition_or_symptoms: str) -> str:
    """
    Search for medical reference information about specific conditions or symptoms.
    
    Args:
        condition_or_symptoms: Medical condition or symptoms to research
    
    Returns:
        Medical reference information
    """
    try:
        # Format search query for medical information
        query = f"medical diagnosis {condition_or_symptoms} symptoms differential diagnosis"
        results = search_tool.run(query)
        return results
    except Exception as e:
        return f"Medical reference search failed: {str(e)}"

# Create tools list
tools = [
    Tool(
        name="internet_search",
        func=internet_search_tool,
        description="Search the internet for general medical information to help formulate targeted questions."
    ),
    Tool(
        name="medical_reference",
        func=medical_reference_search,
        description="Search for medical reference information about specific conditions, symptoms, or diagnostic criteria."
    )
]

# System prompt for the inquire agent
inquire_system_prompt = """
You are a medical inquiry agent responsible for generating targeted questions when insufficient information is available for diagnosis.

Your tasks:
1. Analyze the current symptoms and evaluation results
2. Use internet search tools to research relevant medical conditions and diagnostic criteria
3. Generate specific, targeted questions to gather missing information
4. Prioritize questions based on diagnostic importance

Guidelines for generating questions:
- Ask about specific symptoms that could help differentiate between conditions
- Inquire about symptom duration, severity, and progression
- Ask about relevant medical history and family history
- Consider lifestyle factors and recent exposures
- Be empathetic and non-alarming in question phrasing

Always respond with a JSON object containing:
{
    "inquiry_message": "Polite message explaining need for more information",
    "specific_questions": [
        {
            "question": "Detailed question text",
            "purpose": "Why this question is important for diagnosis",
            "priority": "high/medium/low"
        }
    ],
    "areas_of_concern": ["area1", "area2"],
    "suggested_follow_up": "What to focus on in follow-up questions"
}

Ensure questions are:
- Clear and easy to understand
- Medically relevant
- Non-frightening to the patient
- Focused on gathering diagnostic information
"""

def create_inquire_prompt(state):
    """Create the prompt for the inquire agent."""
    messages = state.get("messages", [])
    
    prompt = [{
        "role": "system", 
        "content": inquire_system_prompt
    }] + messages
    
    return prompt

# Create the inquire agent
inquire_agent = create_react_agent(
    llm,
    tools=tools,
    prompt=create_inquire_prompt
)

class InquireAgent:
    """Wrapper class for the inquire agent."""
    
    def __init__(self):
        self.agent = inquire_agent
    
    def generate_inquiry(self, evaluation_data: Dict[str, Any], current_symptoms: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate targeted inquiry based on evaluation results.
        
        Args:
            evaluation_data: Results from the evaluation agent
            current_symptoms: Current symptoms data if available
            
        Returns:
            Dictionary with inquiry questions and guidance
        """
        try:
            # Extract relevant information
            inquiry_needed = evaluation_data.get("inquiry_needed", {})
            diagnosis = evaluation_data.get("diagnosis", {})
            reasoning = evaluation_data.get("reasoning", "")
            
            # Create context message
            context_message = f"""
            Based on the current medical evaluation, further information is needed for a proper diagnosis.
            
            Current evaluation results:
            - Primary condition considered: {diagnosis.get('primary_condition', 'Unknown')}
            - Confidence level: {diagnosis.get('confidence', 0.0)}
            - Reasoning: {reasoning}
            
            Areas that need exploration:
            {json.dumps(inquiry_needed, indent=2)}
            
            Current symptoms (if available):
            {json.dumps(current_symptoms, indent=2) if current_symptoms else 'No current symptoms provided'}
            
            Please generate appropriate follow-up questions to gather the missing information needed for a more accurate diagnosis. 
            Use the medical_reference tool to research relevant conditions and ensure questions are medically appropriate.
            """
            
            # Invoke the agent
            state = {"messages": [{"role": "user", "content": context_message}]}
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
                
                inquiry_result = json.loads(json_str)
                return inquiry_result
                
            except (json.JSONDecodeError, ValueError):
                # Fallback if JSON parsing fails
                return {
                    "inquiry_message": "I need some additional information to better understand your condition. Could you please provide more details about your symptoms?",
                    "specific_questions": [
                        {
                            "question": "How long have you been experiencing these symptoms?",
                            "purpose": "Understanding symptom duration helps with diagnosis",
                            "priority": "high"
                        },
                        {
                            "question": "Have the symptoms gotten worse, better, or stayed the same?",
                            "purpose": "Symptom progression indicates severity and type of condition",
                            "priority": "high"
                        },
                        {
                            "question": "Do you have any other symptoms you haven't mentioned?",
                            "purpose": "Additional symptoms can help narrow down the diagnosis",
                            "priority": "medium"
                        }
                    ],
                    "areas_of_concern": ["Symptom progression", "Additional symptoms", "Medical history"],
                    "suggested_follow_up": "Focus on timeline and severity of symptoms"
                }
                
        except Exception as e:
            return {
                "inquiry_message": "I apologize, but I'm having difficulty processing your information. Could you please describe your main symptoms again?",
                "specific_questions": [
                    {
                        "question": "What are your main symptoms?",
                        "purpose": "Basic symptom assessment",
                        "priority": "high"
                    },
                    {
                        "question": "When did these symptoms start?",
                        "purpose": "Understanding onset timing",
                        "priority": "high"
                    }
                ],
                "areas_of_concern": ["Basic symptom assessment"],
                "suggested_follow_up": "Start with fundamental symptom information"
            }
    
    def generate_general_inquiry(self, topic: str = "general health") -> Dict[str, Any]:
        """
        Generate a general health inquiry when no specific evaluation data is available.
        
        Args:
            topic: General topic to focus the inquiry on
            
        Returns:
            Dictionary with general inquiry questions
        """
        return {
            "inquiry_message": "To help me understand your health concerns better, I'd like to ask you some questions about your symptoms and medical history.",
            "specific_questions": [
                {
                    "question": "What symptoms are you currently experiencing?",
                    "purpose": "Initial symptom assessment",
                    "priority": "high"
                },
                {
                    "question": "When did these symptoms first start?",
                    "purpose": "Understanding onset and duration",
                    "priority": "high"
                },
                {
                    "question": "How would you rate the severity of your symptoms on a scale of 1-10?",
                    "purpose": "Assessing symptom severity",
                    "priority": "medium"
                },
                {
                    "question": "Do you have any chronic medical conditions or take any medications?",
                    "purpose": "Understanding medical background",
                    "priority": "medium"
                }
            ],
            "areas_of_concern": ["Primary symptoms", "Medical history", "Symptom severity"],
            "suggested_follow_up": "Gather basic symptom and health information first"
        }

# Global instance
inquire_agent_instance = InquireAgent()

if __name__ == "__main__":
    # Test the inquire agent
    test_evaluation = {
        "decision": "further_inquiry",
        "diagnosis": {
            "primary_condition": "Possible viral infection",
            "confidence": 0.4,
            "explanation": "Symptoms suggest viral infection but need more information"
        },
        "inquiry_needed": {
            "specific_questions": ["Duration of fever", "Recent travel history"],
            "areas_to_explore": ["Symptom timeline", "Exposure history"]
        },
        "reasoning": "Low confidence due to insufficient symptom information"
    }
    
    result = inquire_agent_instance.generate_inquiry(test_evaluation)
    print(json.dumps(result, indent=2))