# Medical Diagnosis Assistant System

A comprehensive LLM-based medical diagnosis assistance system built with LangGraph, LangChain, and Azure OpenAI. This system orchestrates multiple AI agents to gather symptom information, format data, evaluate conditions, and provide preliminary diagnostic insights.

## ⚠️ Important Medical Disclaimer

**This system is for educational and informational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical concerns.**

- Never use this system for medical emergencies
- AI systems have limitations and can make errors
- Results should only be used to facilitate discussions with healthcare providers
- Never self-medicate based on AI recommendations

## System Architecture

The system consists of multiple specialized agents orchestrated through LangGraph:

### Core Agents

1. **Chatbot Agent** (`chatbot-agent.py`)
   - Handles patient interactions with medical disclaimers
   - Gathers symptom information through natural conversation
   - Maintains conversation flow and context

2. **Format-Filter Agent** (`format-filter-agent.py`)
   - Converts natural language symptom descriptions to structured JSON
   - Maps symptoms to a standardized boolean symptom list
   - Assesses information adequacy

3. **Evaluation Agent** (`evaluation-agent.py`) 
   - Uses the ML model and internet searches for diagnosis evaluation
   - Determines if sufficient information exists for diagnosis
   - Decides between final diagnosis or further inquiry

4. **Inquire Agent** (`inquire-agent.py`)
   - Generates targeted follow-up questions when more information is needed
   - Uses internet research to formulate medically relevant questions

5. **Fake Model** (`fake-model.py`)
   - Placeholder for deep learning diagnostic model
   - Uses LLM to simulate medical diagnosis based on structured symptoms

### Orchestration Flow

The system follows a structured flow:
1. User describes symptoms to Chatbot Agent
2. Chatbot gathers information through conversation
3. When ready, Format-Filter Agent converts to structured JSON
4. If information inadequate, loop back to Chatbot for more details
5. Evaluation Agent uses ML model and internet search
6. If diagnosis possible, deliver results; otherwise generate inquiry
7. Inquire Agent creates targeted questions
8. Loop back to Chatbot to ask questions
9. Final diagnosis delivered with medical disclaimers

## Features

- **Natural Language Processing**: Understands symptom descriptions in natural language
- **Structured Data Conversion**: Converts conversations to standardized medical data formats
- **Multi-Agent Coordination**: Specialized agents work together seamlessly
- **Internet Research**: Agents can search for medical information to improve accuracy
- **Loop Prevention**: Intelligent flow control prevents infinite loops
- **Medical Safety**: Heavy emphasis on disclaimers and professional consultation
- **Interactive & Batch Modes**: Support for both real-time conversations and batch processing

## Installation

### Prerequisites

- Python 3.8+
- Azure OpenAI access with deployed GPT model
- Internet connection for research capabilities

### Setup Steps

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment**
   ```bash
   # Copy template and edit with your Azure OpenAI credentials
   copy .env.template .env
   # Edit .env with your actual Azure OpenAI configuration
   ```

3. **Verify symptom list**
   - Ensure `list.txt` contains the symptom list for your use case
   - The included list has 132 medical symptoms

## Usage

### Interactive Mode (Recommended)

```bash
python main.py
```

This starts an interactive session where you can:
- Describe symptoms naturally
- Answer follow-up questions
- Receive structured diagnostic insights
- Get recommendations for next steps

### Commands in Interactive Mode

- `help` - Show available commands
- `new` - Start a new diagnosis session
- `history` - View conversation history
- `quit`/`exit` - End the session

### Batch Mode

```bash
python main.py --batch input.json [output.json]
```

## Configuration

### Environment Variables

Required in `.env` file:

```env
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_CHAT_DEPLOYMENT=your_deployment_name
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

## Safety Features

1. **Heavy Disclaimers**: Every interaction includes medical disclaimers
2. **Professional Consultation**: Constant emphasis on healthcare professional consultation
3. **No Treatment Recommendations**: System never suggests medications or treatments
4. **Error Handling**: Graceful handling of system errors
5. **Loop Prevention**: Intelligent flow control prevents infinite questioning loops

## File Structure

```
doctor_agent_architecture/
├── main.py                  # Main entry point
├── orquestation.py          # LangGraph orchestration
├── chatbot-agent.py         # Patient interaction
├── format-filter-agent.py   # Data formatting
├── evaluation-agent.py      # Diagnosis evaluation
├── inquire-agent.py        # Question generation
├── fake-model.py           # ML model placeholder
├── list.txt                # Symptom list
├── requirements.txt        # Dependencies
├── .env.template          # Environment template
└── README.md              # This file
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Azure OpenAI Errors**: Check your `.env` configuration
3. **Symptom List Errors**: Ensure `list.txt` is properly formatted
4. **Agent Import Issues**: Files use hyphens in names, may need adjustment

**Remember: For medical emergencies, contact emergency services immediately!**
