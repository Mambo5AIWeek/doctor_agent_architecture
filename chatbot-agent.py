#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chatbot Agent - Patient Interaction Interface
This agent handles patient conversations with appropriate medical disclaimers and limitations.
"""

import json
import os
from typing import Dict, Any, List, Optional
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=1  # Moderate temperature for natural conversation
)

class ChatbotAgent:
    """Chatbot agent for patient interactions with medical disclaimers."""
    
    def __init__(self):
        self.llm = llm
        self.memory = ConversationBufferMemory(return_messages=True)
        self.conversation_state = "initial"  # initial, gathering, evaluating, final
        self.symptom_summary = ""
        
    def get_system_prompt(self, mode: str = "initial") -> str:
        """Get system prompt based on conversation mode."""
        
        base_disclaimer = """
IMPORTANT MEDICAL DISCLAIMER: 
You are an AI assistant providing general health information only. You are NOT a licensed medical professional, and your responses do not constitute medical advice, diagnosis, or treatment recommendations. 

ALWAYS emphasize that:
- This is not professional medical advice
- Users should consult qualified healthcare professionals for proper diagnosis and treatment
- Never suggest self-medication or specific treatments
- In case of emergency, users should contact emergency services immediately
- The AI system has limitations and can make errors
"""
        
        if mode == "initial":
            return f"""
{base_disclaimer}

You are a polite and empathetic medical inquiry chatbot. Your role is to:
1. Gather information about the user's symptoms, medical history, and concerns
2. Ask relevant follow-up questions to understand their condition
3. Maintain a professional, caring, and non-alarming tone
4. Provide emotional support while gathering information

Guidelines:
- Start conversations by acknowledging their concern and explaining your role
- Ask one question at a time to avoid overwhelming the user
- Be empathetic and supportive
- Avoid medical jargon; use simple, clear language
- Never provide specific medical advice or suggest treatments
- Encourage seeking professional medical help when appropriate
- If symptoms seem severe, urgently recommend seeking immediate medical attention

Conversation approach:
1. Greet warmly and explain your purpose
2. Ask about their main concern or symptoms
3. Follow up with relevant questions about duration, severity, etc.
4. Gather information about medical history if relevant
5. Summarize information when you have enough details
"""
        
        elif mode == "inquiry":
            return f"""
{base_disclaimer}

You are continuing a medical inquiry conversation. Based on the evaluation, you need to ask specific follow-up questions to gather more information for a better understanding of the user's condition.

Guidelines:
- Reference previous conversation naturally
- Ask the specific questions provided to you
- Explain why the information is helpful (without being alarming)
- Maintain empathy and support
- Continue to emphasize the need for professional medical consultation
"""
        
        elif mode == "final_diagnosis":
            return f"""
{base_disclaimer}

You are providing final results from the medical evaluation system. 

CRITICAL REQUIREMENTS:
- HEAVILY emphasize this is NOT a professional medical diagnosis
- Present results as "possible conditions to discuss with a doctor"
- Strongly encourage consulting healthcare professionals
- NEVER suggest specific treatments or medications
- Include appropriate disclaimers about AI limitations
- Be supportive but clearly state the limitations

Format your response to:
1. Acknowledge the information gathered
2. Present the evaluation results with heavy disclaimers
3. Emphasize the need for professional medical consultation
4. Provide emotional support
5. Suggest next steps (seeing a doctor)
"""
        
        return base_disclaimer
    
    def handle_initial_interaction(self, user_message: str) -> Dict[str, Any]:
        """Handle the initial user interaction."""
        
        system_message = SystemMessage(content=self.get_system_prompt("initial"))
        user_msg = HumanMessage(content=user_message)
        
        messages = [system_message, user_msg]
        
        # Add conversation history if available
        if self.memory.chat_memory.messages:
            messages.extend(self.memory.chat_memory.messages)
        
        try:
            response = self.llm.invoke(messages)
            print(f"Invoking LLM with messages: {messages}")
            
            # Update memory
            self.memory.chat_memory.add_user_message(user_message)
            self.memory.chat_memory.add_ai_message(response.content)
            
            return {
                "response": response.content,
                "state": "gathering",
                "needs_evaluation": False,
                "ready_for_format": False
            }
            
        except Exception as e:
            print(f"Error in initial interaction: {e}")
            return {
                "response": f"I apologize, but I'm experiencing technical difficulties. Please try again or consult with a healthcare professional directly. Error: {str(e)}",
                "state": "error",
                "needs_evaluation": False,
                "ready_for_format": False
            }
    
    def continue_conversation(self, user_message: str) -> Dict[str, Any]:
        """Continue the conversation and determine if ready for evaluation."""
        
        system_message = SystemMessage(content=self.get_system_prompt("initial"))
        
        # Add conversation history
        messages = [system_message]
        if self.memory.chat_memory.messages:
            messages.extend(self.memory.chat_memory.messages)
        
        # Add current user message
        messages.append(HumanMessage(content=user_message))
        
        try:
            response = self.llm.invoke(messages)
            
            # Update memory
            self.memory.chat_memory.add_user_message(user_message)
            self.memory.chat_memory.add_ai_message(response.content)
            
            # Update symptom summary
            self._update_symptom_summary(user_message)
            
            # Determine if ready for evaluation
            ready_for_evaluation = self._assess_readiness_for_evaluation()
            
            return {
                "response": response.content,
                "state": "evaluating" if ready_for_evaluation else "gathering",
                "needs_evaluation": ready_for_evaluation,
                "ready_for_format": ready_for_evaluation,
                "symptom_summary": self.get_symptom_summary()
            }
            
        except Exception as e:
            return {
                "response": f"I apologize for the technical issue. Please try describing your symptoms again, or better yet, consult with a healthcare professional. Error: {str(e)}",
                "state": "error",
                "needs_evaluation": False,
                "ready_for_format": False
            }
    
    def handle_inquiry_questions(self, inquiry_data: Dict[str, Any], user_response: str = None) -> Dict[str, Any]:
        """Handle specific inquiry questions from the inquiry agent."""
        
        if user_response:
            # Process user's response to inquiry
            self.memory.chat_memory.add_user_message(user_response)
            self._update_symptom_summary(user_response)
        
        # Format the inquiry message
        inquiry_message = inquiry_data.get("inquiry_message", "")
        specific_questions = inquiry_data.get("specific_questions", [])
        
        # Create a conversational version of the inquiry
        conversation_prompt = f"""
        Based on the information you've provided, I need to ask some additional questions to better understand your situation.
        
        {inquiry_message}
        
        """
        
        # Add the most important questions
        if specific_questions:
            conversation_prompt += "\nSpecifically, I'd like to know:\n"
            for i, q in enumerate(specific_questions[:3]):  # Limit to top 3 questions
                question = q.get("question", "") if isinstance(q, dict) else str(q)
                conversation_prompt += f"{i+1}. {question}\n"
        
        system_message = SystemMessage(content=self.get_system_prompt("inquiry"))
        messages = [system_message]
        
        # Add conversation history
        if self.memory.chat_memory.messages:
            messages.extend(self.memory.chat_memory.messages[-6:])  # Last 3 exchanges
        
        # Add the inquiry
        messages.append(HumanMessage(content=f"Please ask these follow-up questions conversationally: {conversation_prompt}"))
        
        try:
            response = self.llm.invoke(messages)
            
            self.memory.chat_memory.add_ai_message(response.content)
            
            return {
                "response": response.content,
                "state": "gathering",
                "needs_evaluation": False,
                "ready_for_format": False
            }
            
        except Exception as e:
            return {
                "response": f"I'm having trouble processing the follow-up questions. Could you please describe your symptoms again? If this persists, please consult a healthcare professional.",
                "state": "error",
                "needs_evaluation": False,
                "ready_for_format": False
            }
    
    def deliver_final_diagnosis(self, diagnosis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deliver final diagnosis with appropriate disclaimers."""
        
        diagnosis = diagnosis_data.get("diagnosis", {})
        primary_condition = diagnosis.get("primary_condition", "Unknown condition")
        confidence = diagnosis.get("confidence", 0.0)
        explanation = diagnosis.get("explanation", "")
        recommendations = diagnosis.get("recommendations", [])
        
        # Create the diagnosis delivery prompt
        diagnosis_prompt = f"""
        Based on the information provided, our evaluation suggests the following:
        
        Possible condition: {primary_condition}
        Confidence level: {confidence:.1%}
        Explanation: {explanation}
        
        General recommendations: {', '.join(recommendations) if recommendations else 'Consult with a healthcare professional'}
        
        Please deliver this information with heavy emphasis on:
        1. This is NOT a professional medical diagnosis
        2. AI systems have limitations and can be incorrect
        3. The user MUST consult with qualified healthcare professionals
        4. This is only preliminary information to discuss with a doctor
        5. Never suggest specific treatments or medications
        """
        
        system_message = SystemMessage(content=self.get_system_prompt("final_diagnosis"))
        messages = [system_message, HumanMessage(content=diagnosis_prompt)]
        
        try:
            response = self.llm.invoke(messages)
            
            self.memory.chat_memory.add_ai_message(response.content)
            
            return {
                "response": response.content,
                "state": "final",
                "needs_evaluation": False,
                "ready_for_format": False,
                "conversation_complete": True
            }
            
        except Exception as e:
            fallback_response = f"""
            I apologize, but I'm experiencing technical difficulties in providing the evaluation results.
            
            IMPORTANT: This system is not a substitute for professional medical advice. 
            Please consult with a qualified healthcare professional for proper diagnosis and treatment.
            
            If you're experiencing severe symptoms, please seek immediate medical attention or contact emergency services.
            
            Error: {str(e)}
            """
            
            return {
                "response": fallback_response,
                "state": "error",
                "needs_evaluation": False,
                "ready_for_format": False,
                "conversation_complete": True
            }
    
    def _update_symptom_summary(self, new_information: str) -> None:
        """Update the accumulated symptom summary."""
        if self.symptom_summary:
            self.symptom_summary += f"\n\nAdditional information: {new_information}"
        else:
            self.symptom_summary = new_information
    
    def _assess_readiness_for_evaluation(self) -> bool:
        """Assess if enough information has been gathered for evaluation."""
        # Simple heuristic based on conversation length and content
        messages = self.memory.chat_memory.messages
        
        if len(messages) < 4:  # Need at least 2 exchanges
            return False
        
        # Check if symptom-related keywords are present in the conversation
        conversation_text = " ".join([msg.content.lower() for msg in messages if hasattr(msg, 'content')])
        
        symptom_indicators = [
            'pain', 'ache', 'fever', 'cough', 'tired', 'nausea', 'dizzy', 
            'headache', 'sick', 'symptom', 'feel', 'hurt', 'sore', 'weak'
        ]
        
        symptom_mentions = sum(1 for indicator in symptom_indicators if indicator in conversation_text)
        
        # Ready if multiple symptoms mentioned and sufficient conversation
        return symptom_mentions >= 3 and len(messages) >= 6
    
    def get_symptom_summary(self) -> str:
        """Get the current symptom summary."""
        if not self.symptom_summary:
            # Create summary from conversation history
            messages = self.memory.chat_memory.messages
            user_messages = [msg.content for msg in messages if isinstance(msg, HumanMessage)]
            self.symptom_summary = "\n".join(user_messages)
        
        return self.symptom_summary
    
    def reset_conversation(self) -> None:
        """Reset the conversation state."""
        self.memory.clear()
        self.conversation_state = "initial"
        self.symptom_summary = ""
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get formatted conversation history."""
        history = []
        messages = self.memory.chat_memory.messages
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        
        return history

# Global instance
chatbot_agent = ChatbotAgent()

if __name__ == "__main__":
    # Test the chatbot agent
    print("Testing Chatbot Agent")
    print("=" * 50)
    
    # Test initial interaction
    response1 = chatbot_agent.handle_initial_interaction("I've been feeling sick for a few days with fever and cough.")
    print(f"Initial Response: {response1['response']}")
    print(f"State: {response1['state']}")
    
    # Test follow-up
    response2 = chatbot_agent.continue_conversation("The fever started 3 days ago and I also have a headache and feel very tired.")
    print(f"\nFollow-up Response: {response2['response']}")
    print(f"State: {response2['state']}")
    print(f"Ready for evaluation: {response2['ready_for_format']}")
    
    if response2['ready_for_format']:
        print(f"\nSymptom Summary: {response2['symptom_summary']}")
    
    # Test final diagnosis delivery
    test_diagnosis = {
        "diagnosis": {
            "primary_condition": "Possible viral infection",
            "confidence": 0.75,
            "explanation": "Symptoms suggest a common viral infection",
            "recommendations": ["Rest", "Stay hydrated", "Monitor symptoms"]
        }
    }
    
    response3 = chatbot_agent.deliver_final_diagnosis(test_diagnosis)
    print(f"\nFinal Diagnosis Response: {response3['response']}")
    print(f"Conversation Complete: {response3['conversation_complete']}")