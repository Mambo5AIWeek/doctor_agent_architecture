#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestration Module - Medical Diagnosis System
This module orchestrates the flow between different agents using LangGraph.
"""

import json
import os
from typing import Dict, Any, List, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from dotenv import load_dotenv

# Import our agents
# Note: Import names adjusted for file names with hyphens
import importlib.util
import sys

# Import chatbot agent
spec = importlib.util.spec_from_file_location("chatbot_agent", "chatbot-agent.py")
chatbot_agent_module = importlib.util.module_from_spec(spec)
sys.modules["chatbot_agent"] = chatbot_agent_module
spec.loader.exec_module(chatbot_agent_module)
chatbot_agent = chatbot_agent_module.chatbot_agent

# Import format filter agent
spec = importlib.util.spec_from_file_location("format_filter_agent", "format-filter-agent.py")
format_filter_module = importlib.util.module_from_spec(spec)
sys.modules["format_filter_agent"] = format_filter_module
spec.loader.exec_module(format_filter_module)
format_filter_agent = format_filter_module.format_filter_agent

# Import evaluation agent
spec = importlib.util.spec_from_file_location("evaluation_agent", "evaluation-agent.py")
evaluation_module = importlib.util.module_from_spec(spec)
sys.modules["evaluation_agent"] = evaluation_module
spec.loader.exec_module(evaluation_module)
evaluation_agent_instance = evaluation_module.evaluation_agent_instance

# Import inquire agent
spec = importlib.util.spec_from_file_location("inquire_agent", "inquire-agent.py")
inquire_module = importlib.util.module_from_spec(spec)
sys.modules["inquire_agent"] = inquire_module
spec.loader.exec_module(inquire_module)
inquire_agent_instance = inquire_module.inquire_agent_instance

load_dotenv()

class MedicalDiagnosisState(TypedDict):
    """State for the medical diagnosis system."""
    messages: Annotated[List[BaseMessage], add_messages]
    current_stage: str  # chatbot, format_filter, evaluation, inquiry, final
    user_message: str
    chatbot_response: str
    symptom_summary: str
    formatted_symptoms: List[str]
    evaluation_result: Dict[str, Any]
    inquiry_data: Dict[str, Any]
    final_diagnosis: Dict[str, Any]
    conversation_complete: bool
    needs_more_info: bool
    error_message: str
    loop_count: int  # To prevent infinite loops

def chatbot_node(state: MedicalDiagnosisState) -> MedicalDiagnosisState:
    """
    Handle chatbot interactions - initial conversation and follow-ups.
    """
    print(f"[CHATBOT NODE] Current stage: {state.get('current_stage', 'unknown')}")
    
    user_message = state.get("user_message", "")
    current_stage = state.get("current_stage", "initial")
    
    try:
        if current_stage == "initial" or current_stage == "chatbot":
            # Handle initial or continuing conversation
            if len(chatbot_agent.memory.chat_memory.messages) == 0:
                result = chatbot_agent.handle_initial_interaction(user_message)
            else:
                result = chatbot_agent.continue_conversation(user_message)
        
        elif current_stage == "inquiry_response":
            # Handle response to inquiry questions
            inquiry_data = state.get("inquiry_data", {})
            result = chatbot_agent.handle_inquiry_questions(inquiry_data, user_message)
        
        elif current_stage == "final_delivery":
            # Deliver final diagnosis
            evaluation_result = state.get("evaluation_result", {})
            result = chatbot_agent.deliver_final_diagnosis(evaluation_result)
        
        else:
            # Default handling
            result = chatbot_agent.continue_conversation(user_message)
        
        # Update state
        new_state = state.copy()
        new_state["chatbot_response"] = result["response"]
        new_state["symptom_summary"] = result.get("symptom_summary", state.get("symptom_summary", ""))
        new_state["conversation_complete"] = result.get("conversation_complete", False)

        print(f"[CHATBOT NODE] Chatbot response: {new_state['chatbot_response']}")
        print(f"[CHATBOT NODE] Symptom summary: {new_state['symptom_summary']}")
        print(f"[CHATBOT NODE] User message: {user_message}")
        
        # Add messages to conversation
        if user_message:
            new_state["messages"].append(HumanMessage(content=user_message))
        new_state["messages"].append(AIMessage(content=result["response"]))
        
        # Determine next stage - simplified logic
        if result.get("conversation_complete", False):
            new_state["current_stage"] = "final"
        elif result.get("ready_for_format", False):
            new_state["current_stage"] = "format_filter"
        else:
            # Keep the conversation going - don't change stage
            new_state["current_stage"] = "continue_chatbot"
        
        print(f"[CHATBOT NODE] Moving to stage: {new_state['current_stage']}")
        return new_state
        
    except Exception as e:
        print(f"[CHATBOT NODE] Error: {e}")
        new_state = state.copy()
        new_state["error_message"] = f"Chatbot error: {str(e)}"
        new_state["current_stage"] = "error"
        return new_state

def format_filter_node(state: MedicalDiagnosisState) -> MedicalDiagnosisState:
    """
    Format symptom data into structured JSON.
    """
    print("[FORMAT FILTER NODE] Processing symptom summary")
    
    symptom_summary = state.get("symptom_summary", "")
    
    try:
        # Format the symptoms
        formatted_symptoms = format_filter_agent.format_symptoms(symptom_summary)
        
        # Update state
        new_state = state.copy()
        new_state["formatted_symptoms"] = formatted_symptoms
        
        # Decide next step
        if not formatted_symptoms:
            # If no symptoms were extracted, ask for more info
            new_state["current_stage"] = "inquiry_response"
            new_state["inquiry_data"] = {
                "missing_info": ["Could not identify any specific symptoms. Please describe how you are feeling in more detail."]
            }
        else:
            new_state["current_stage"] = "evaluation"
            
        print(f"[FORMAT FILTER NODE] Formatted symptoms: {formatted_symptoms}")
        print(f"[FORMAT FILTER NODE] Moving to stage: {new_state['current_stage']}")
        return new_state
        
    except Exception as e:
        print(f"[FORMAT FILTER NODE] Error: {e}")
        new_state = state.copy()
        new_state["error_message"] = f"Format/Filter error: {str(e)}"
        new_state["current_stage"] = "error"
        return new_state

def evaluation_node(state: MedicalDiagnosisState) -> MedicalDiagnosisState:
    """
    Evaluate symptoms using the ML model and determine if diagnosis is possible.
    """
    print("[EVALUATION NODE] Evaluating symptoms")
    
    formatted_symptoms = state.get("formatted_symptoms", {})
    
    try:
        # Run evaluation
        evaluation_result = evaluation_agent_instance.evaluate_diagnosis(formatted_symptoms)
        
        new_state = state.copy()
        new_state["evaluation_result"] = evaluation_result
        
        # Determine next step based on evaluation
        decision = evaluation_result.get("decision", "further_inquiry")
        
        if decision == "final_diagnosis":
            print("[EVALUATION NODE] Sufficient information for diagnosis")
            new_state["current_stage"] = "final_delivery"
        else:
            print("[EVALUATION NODE] Further inquiry needed")
            new_state["current_stage"] = "inquiry"
        
        return new_state
        
    except Exception as e:
        print(f"[EVALUATION NODE] Error: {e}")
        new_state = state.copy()
        new_state["error_message"] = f"Evaluation error: {str(e)}"
        new_state["current_stage"] = "error"
        return new_state

def inquiry_node(state: MedicalDiagnosisState) -> MedicalDiagnosisState:
    """
    Generate targeted inquiry questions when more information is needed.
    """
    print("[INQUIRY NODE] Generating targeted questions")
    
    evaluation_result = state.get("evaluation_result", {})
    formatted_symptoms = state.get("formatted_symptoms", {})
    
    try:
        # Generate inquiry
        inquiry_data = inquire_agent_instance.generate_inquiry(evaluation_result, formatted_symptoms)
        
        new_state = state.copy()
        new_state["inquiry_data"] = inquiry_data
        new_state["current_stage"] = "inquiry_response"
        
        return new_state
        
    except Exception as e:
        print(f"[INQUIRY NODE] Error: {e}")
        new_state = state.copy()
        new_state["error_message"] = f"Inquiry error: {str(e)}"
        new_state["current_stage"] = "error"
        return new_state

def final_node(state: MedicalDiagnosisState) -> MedicalDiagnosisState:
    """
    Handle final diagnosis delivery.
    """
    print("[FINAL NODE] Delivering final results")
    
    new_state = state.copy()
    new_state["conversation_complete"] = True
    new_state["current_stage"] = "complete"
    
    return new_state

def error_node(state: MedicalDiagnosisState) -> MedicalDiagnosisState:
    """
    Handle errors in the system.
    """
    print(f"[ERROR NODE] Handling error: {state.get('error_message', 'Unknown error')}")
    
    new_state = state.copy()
    new_state["conversation_complete"] = True
    new_state["current_stage"] = "error_complete"
    
    # Add error message to conversation
    error_msg = f"I apologize, but there was a technical issue: {state.get('error_message', 'Unknown error')}. Please try again or consult with a healthcare professional."
    new_state["messages"].append(AIMessage(content=error_msg))
    
    return new_state

def route_after_chatbot(state: MedicalDiagnosisState) -> str:
    """Route after chatbot interaction."""
    current_stage = state.get("current_stage", "chatbot")
    
    if current_stage == "final":
        return "final"
    elif current_stage == "format_filter":
        return "format_filter"
    elif current_stage == "error":
        return "error"
    elif current_stage == "continue_chatbot":
        return END  # This should end the current invoke and wait for next input
    else:
        return END

def route_after_format_filter(state: MedicalDiagnosisState) -> str:
    """Route after format filter."""
    current_stage = state.get("current_stage", "evaluation")
    
    if current_stage == "inquiry_response":
        return "chatbot"  # Loop back for more info
    elif current_stage == "error":
        return "error"
    else:
        return "evaluation"

def route_after_evaluation(state: MedicalDiagnosisState) -> str:
    """Route after evaluation."""
    current_stage = state.get("current_stage", "inquiry")
    
    if current_stage == "final_delivery":
        return "chatbot"  # Deliver diagnosis
    elif current_stage == "error":
        return "error"
    else:
        return "inquiry"

def route_after_inquiry(state: MedicalDiagnosisState) -> str:
    """Route after inquiry generation."""
    return "chatbot"  # Always go back to chatbot to ask questions

def should_continue(state: MedicalDiagnosisState) -> str:
    """Determine if conversation should continue."""
    if state.get("conversation_complete", False):
        return END
    elif state.get("current_stage") == "error":
        return "error"
    else:
        return "chatbot"

# Create the workflow
workflow = StateGraph(MedicalDiagnosisState)

# Add nodes
workflow.add_node("chatbot", chatbot_node)
workflow.add_node("format_filter", format_filter_node)
workflow.add_node("evaluation", evaluation_node)
workflow.add_node("inquiry", inquiry_node)
workflow.add_node("final", final_node)
workflow.add_node("error", error_node)

# Add edges
workflow.add_edge(START, "chatbot")

# Conditional edges - Fixed to handle END case
workflow.add_conditional_edges(
    "chatbot",
    route_after_chatbot,
    {
        "format_filter": "format_filter", 
        "final": "final",
        "error": "error",
        END: END  # Add this mapping
    }
)

workflow.add_conditional_edges(
    "format_filter",
    route_after_format_filter,
    {
        "chatbot": "chatbot",
        "evaluation": "evaluation",
        "error": "error"
    }
)

workflow.add_conditional_edges(
    "evaluation",
    route_after_evaluation,
    {
        "chatbot": "chatbot",
        "inquiry": "inquiry",
        "error": "error"
    }
)

workflow.add_conditional_edges(
    "inquiry",
    route_after_inquiry,
    {
        "chatbot": "chatbot"
    }
)

workflow.add_edge("final", END)
workflow.add_edge("error", END)

# Compile the graph
app = workflow.compile()
# assuming `app` is your compiled LangGraph / workflow
graph = app.get_graph()

import networkx as nx
import matplotlib.pyplot as plt

def visualize_langgraph(graph, filename="graph.png"):
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes
    for node_id, node in graph.nodes.items():
        G.add_node(node_id, label=node.name)

    # Add edges
    for edge in graph.edges:
        label = "conditional" if edge.conditional else "direct"
        G.add_edge(edge.source, edge.target, label=label)

    # Layout (spring is good for arbitrary graphs)
    pos = nx.spring_layout(G, seed=42)  # deterministic layout

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=2000,
        node_color="#90caf9",
        edgecolors="black"
    )

    # Draw labels for nodes
    labels = {n: G.nodes[n]["label"] for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight="bold")

    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        arrowstyle="->",
        arrowsize=20,
        edge_color="#555"
    )

    # Edge labels (direct / conditional)
    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Save to PNG
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# Example usage:
visualize_langgraph(graph, "langgraph.png")



class MedicalDiagnosisOrchestrator:
    """Main orchestrator for the medical diagnosis system."""
    
    def __init__(self):
        self.app = app
        self.current_session = None
    
    def start_new_session(self) -> str:
        """Start a new diagnosis session."""
        # Reset all agents
        chatbot_agent.reset_conversation()
        
        session_id = f"session_{hash(os.urandom(16))}"
        self.current_session = session_id
        
        return session_id
    
    def process_user_message(self, user_message: str, session_id: str = None) -> Dict[str, Any]:
        """Process a user message through the system."""
        
        if session_id != self.current_session:
            self.start_new_session()
        
        # Get the current conversation state to continue properly
        current_stage = "initial"
        if hasattr(chatbot_agent, 'memory') and len(chatbot_agent.memory.chat_memory.messages) > 0:
            current_stage = "chatbot"  # Continuing conversation
        
        # Initial state
        initial_state = {
            "messages": [],
            "current_stage": current_stage,  # Use determined stage
            "user_message": user_message,
            "chatbot_response": "",
            "symptom_summary": "",
            "formatted_symptoms": {},
            "evaluation_result": {},
            "inquiry_data": {},
            "final_diagnosis": {},
            "conversation_complete": False,
            "needs_more_info": False,
            "error_message": "",
            "loop_count": 0
        }
        
        try:
            # Run the workflow
            result = self.app.invoke(initial_state)
            
            # Extract the final response
            final_messages = result.get("messages", [])
            final_response = final_messages[-1].content if final_messages else "No response generated."
            
            return {
                "response": final_response,
                "conversation_complete": result.get("conversation_complete", False),
                "current_stage": result.get("current_stage", "unknown"),
                "session_id": self.current_session,
                "error": result.get("error_message", None)
            }
            
        except Exception as e:
            print(f"[ORCHESTRATOR] Error processing message: {e}")
            return {
                "response": f"I apologize, but there was a technical issue processing your request. Please try again or consult with a healthcare professional. Error: {str(e)}",
                "conversation_complete": False,
                "current_stage": "error",
                "session_id": self.current_session,
                "error": str(e)
            }
    
    def get_conversation_history(self, session_id: str = None) -> List[Dict[str, str]]:
        """Get the conversation history for a session."""
        if session_id != self.current_session:
            return []
        
        return chatbot_agent.get_conversation_history()
    
    def end_session(self, session_id: str = None) -> None:
        """End a diagnosis session."""
        if session_id == self.current_session:
            chatbot_agent.reset_conversation()
            self.current_session = None

# Global orchestrator instance
orchestrator = MedicalDiagnosisOrchestrator()

if __name__ == "__main__":
    # Test the orchestration system
    print("Testing Medical Diagnosis Orchestration System")
    print("=" * 60)
    
    # Start a new session
    session_id = orchestrator.start_new_session()
    print(f"Started session: {session_id}")
    
    # Simulate a conversation
    test_messages = [
        "Hi, I've been feeling sick for the past few days.",
        "I have a fever, cough, and headache. I also feel very tired.",
        "The fever started 3 days ago and has been around 101°F. The cough is dry and the headache is pretty severe.",
        "I'm 28 years old and female. No, I don't have any chronic conditions."
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n--- User Message {i} ---")
        print(f"User: {message}")
        
        result = orchestrator.process_user_message(message, session_id)
        
        print(f"Assistant: {result['response']}")
        print(f"Stage: {result['current_stage']}")
        print(f"Complete: {result['conversation_complete']}")
        
        if result.get('error'):
            print(f"Error: {result['error']}")
        
        if result['conversation_complete']:
            print("\n=== Conversation Complete ===")
            break
    
    # Show final conversation history
    history = orchestrator.get_conversation_history(session_id)
    print(f"\nFinal conversation length: {len(history)} messages")
    
    # End session
    orchestrator.end_session(session_id)
    print("Session ended.")