# AI Chatbot Application

A modular and extensible chatbot application built with Streamlit and OpenAI's GPT models.

## Project Structure

```
chatbot-app/
├── app.py                  # Main Streamlit entry point
├── requirements.txt        # Dependencies
├── config/                 # Configuration and settings
├── prompts/               # System and user prompts
├── context/               # Memory and embeddings management
├── core/                  # Core chatbot logic
├── ui/                    # Streamlit UI components
└── utils/                 # Utility functions
```

## Features

- 🤖 AI-powered conversations using OpenAI GPT models
- 💬 Context-aware responses with conversation memory
- 🎨 Clean and intuitive Streamlit interface
- 📝 Modular prompt management
- 🔍 Optional vector database for semantic search
- 🎯 Streaming responses for better UX

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and add your OpenAI API key:

```bash
cp .env.example .env
```

Edit `.env` and add your API key:
```
OPENAI_API_KEY=your_actual_api_key_here
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

## Configuration

### API Settings

Edit `config/settings.py` or use environment variables in `.env`:

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_MODEL`: Model to use (default: gpt-4-turbo-preview)
- `OPENAI_TEMPERATURE`: Response creativity (0-1)
- `OPENAI_MAX_TOKENS`: Maximum response length

### Prompts

Customize the chatbot's behavior by editing prompt files:

- System prompts: `prompts/system_prompts/`
- User prompts: `prompts/user_prompts/`

## Architecture

### Core Components

- **ChatManager**: Orchestrates all chatbot operations
- **LLMClient**: Wraps OpenAI API calls
- **PromptBuilder**: Combines prompts and context
- **MemoryManager**: Manages conversation history
- **EmbeddingsManager**: Optional vector DB for long-term memory

### UI Components

- **layout.py**: Page configuration and sidebar
- **chat_ui.py**: Chat interface and message rendering

## Usage

1. Start the application
2. Type your message in the chat input
3. Receive AI-powered responses
4. Use the sidebar to clear conversation history

## Advanced Features

### Vector Database

Enable semantic search and long-term memory:

1. Set `USE_VECTOR_DB=true` in `.env`
2. Install ChromaDB: `pip install chromadb`
3. The system will store and retrieve relevant context

### Custom Prompts

Create custom prompt templates in `prompts/user_prompts/` and load them:

```python
template = chat_manager.prompt_builder.load_user_prompt_template('your_template.txt')
```

## Development

### Adding New Features

1. Core logic: Add to `core/` modules
2. UI components: Add to `ui/` modules
3. Utilities: Add to `utils/` modules
4. Configuration: Update `config/settings.py`

### Testing

Run tests (after creating test files):

```bash
pytest tests/
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
