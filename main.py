#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Entry Point - Medical Diagnosis System
This is the main entry point for the medical diagnosis application.
"""

import os
import sys
import json
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the orchestrator
import importlib.util
import sys

# Import orchestrator
spec = importlib.util.spec_from_file_location("orquestation", "orquestation.py")
orquestation_module = importlib.util.module_from_spec(spec)
sys.modules["orquestation"] = orquestation_module
spec.loader.exec_module(orquestation_module)
orchestrator = orquestation_module.orchestrator

def print_welcome():
    """Print welcome message with medical disclaimers."""
    print("\n" + "=" * 70)
    print("          MEDICAL DIAGNOSIS ASSISTANCE SYSTEM")
    print("=" * 70)
    print()
    print("⚠️  IMPORTANT MEDICAL DISCLAIMER:")
    print("   This system provides general health information only.")
    print("   It is NOT a substitute for professional medical advice.")
    print("   Always consult qualified healthcare professionals for")
    print("   proper diagnosis, treatment, and medical care.")
    print()
    print("🔬 This AI system:")
    print("   • Can help organize your symptom information")
    print("   • Provides preliminary insights for discussion with doctors")
    print("   • Has limitations and may make errors")
    print("   • Should never be used for emergency situations")
    print()
    print("🚨 For medical emergencies, contact emergency services immediately!")
    print("=" * 70)
    print()

def print_help():
    """Print help information."""
    print("\nAvailable commands:")
    print("  help - Show this help message")
    print("  new - Start a new diagnosis session")
    print("  history - Show conversation history")
    print("  quit/exit - Exit the application")
    print("  Or just type your message to continue the conversation")
    print()

def format_response(response_text: str) -> str:
    """Format the response for better readability."""
    # Add some formatting to make responses more readable
    lines = response_text.split('\n')
    formatted_lines = []
    
    for line in lines:
        if line.strip():
            # Add proper spacing for readability
            if line.startswith('IMPORTANT:') or line.startswith('CRITICAL:'):
                formatted_lines.append(f"\n⚠️  {line}")
            elif line.startswith('•') or line.startswith('-'):
                formatted_lines.append(f"   {line}")
            else:
                formatted_lines.append(line)
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def interactive_mode():
    """Run the interactive diagnosis session."""
    print_welcome()
    
    current_session = None
    
    print("Type 'help' for available commands, or describe your symptoms to begin.")
    print("Type 'quit' or 'exit' to end the session.\n")
    
    while True:
        try:
            # Get user input
            if current_session:
                user_input = input("You: ").strip()
            else:
                user_input = input("Start new session - You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                if current_session:
                    orchestrator.end_session(current_session)
                print("\nThank you for using the Medical Diagnosis System.")
                print("Remember: Always consult healthcare professionals for medical advice!")
                break
            
            elif user_input.lower() == 'help':
                print_help()
                continue
            
            elif user_input.lower() == 'new':
                if current_session:
                    orchestrator.end_session(current_session)
                current_session = orchestrator.start_new_session()
                print(f"\n🆕 Started new session: {current_session[:8]}...")
                print("Please describe your symptoms or health concerns.\n")
                continue
            
            elif user_input.lower() == 'history':
                if current_session:
                    history = orchestrator.get_conversation_history(current_session)
                    if history:
                        print("\n📋 Conversation History:")
                        print("-" * 40)
                        for i, msg in enumerate(history, 1):
                            role = "You" if msg["role"] == "user" else "Assistant"
                            print(f"{i}. {role}: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
                        print("-" * 40)
                    else:
                        print("\nNo conversation history available.")
                else:
                    print("\nNo active session. Start a new session first.")
                print()
                continue
            
            # Handle regular conversation
            if not current_session:
                current_session = orchestrator.start_new_session()
                print(f"\n🆕 Started new session: {current_session[:8]}...\n")
            
            # Process the message
            print("\n🤔 Processing your message...")
            result = orchestrator.process_user_message(user_input, current_session)
            
            # Display the response
            print("\nAssistant:", format_response(result['response']))
            
            # Show status information
            if result.get('error'):
                print(f"\n⚠️  System Status: Error - {result['error']}")
            else:
                stage_emojis = {
                    'chatbot': '💬',
                    'format_filter': '📋',
                    'evaluation': '🔬',
                    'inquiry': '❓',
                    'final': '✅',
                    'complete': '🎯',
                    'error': '❌'
                }
                stage = result.get('current_stage', 'unknown')
                emoji = stage_emojis.get(stage, '🔄')
                print(f"\n{emoji} Status: {stage.replace('_', ' ').title()}")
            
            # Check if conversation is complete
            if result.get('conversation_complete', False):
                print("\n" + "=" * 50)
                print("         CONSULTATION COMPLETE")
                print("=" * 50)
                print("\n🏥 Next Steps:")
                print("   • Schedule an appointment with a healthcare professional")
                print("   • Share this information with your doctor")
                print("   • Seek immediate medical attention if symptoms worsen")
                print()
                print("💡 You can start a new session by typing 'new' or 'quit' to exit.")
                
                # End the current session
                orchestrator.end_session(current_session)
                current_session = None
            
            print()  # Add spacing
            
        except KeyboardInterrupt:
            print("\n\nSession interrupted by user.")
            if current_session:
                orchestrator.end_session(current_session)
            break
        
        except Exception as e:
            print(f"\n❌ An unexpected error occurred: {str(e)}")
            print("Please try again or restart the application.\n")

def batch_mode(input_file: str, output_file: str = None):
    """Run in batch mode processing multiple cases from a file."""
    try:
        with open(input_file, 'r') as f:
            cases = json.load(f)
        
        results = []
        
        for i, case in enumerate(cases, 1):
            print(f"\nProcessing case {i}/{len(cases)}...")
            
            session_id = orchestrator.start_new_session()
            
            case_result = {
                "case_id": i,
                "input": case,
                "conversation": [],
                "final_result": None,
                "error": None
            }
            
            try:
                # Process the case
                result = orchestrator.process_user_message(case.get("symptoms", ""), session_id)
                
                case_result["conversation"] = orchestrator.get_conversation_history(session_id)
                case_result["final_result"] = result
                
            except Exception as e:
                case_result["error"] = str(e)
            finally:
                orchestrator.end_session(session_id)
            
            results.append(case_result)
        
        # Save results
        output_path = output_file or f"diagnosis_results_{len(cases)}_cases.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nBatch processing complete. Results saved to: {output_path}")
        
    except Exception as e:
        print(f"Error in batch mode: {e}")

def check_environment():
    """Check if required environment variables are set."""
    required_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_CHAT_DEPLOYMENT", 
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in your .env file or environment.")
        return False
    
    return True

def main():
    """Main entry point."""
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--batch' and len(sys.argv) > 2:
            input_file = sys.argv[2]
            output_file = sys.argv[3] if len(sys.argv) > 3 else None
            batch_mode(input_file, output_file)
        elif sys.argv[1] == '--help':
            print("Medical Diagnosis System")
            print("Usage:")
            print("  python main.py                    # Interactive mode")
            print("  python main.py --batch input.json [output.json]  # Batch mode")
            print("  python main.py --help            # Show this help")
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information.")
    else:
        # Interactive mode
        interactive_mode()

if __name__ == "__main__":
    main()