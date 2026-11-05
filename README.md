# 🤖 Blog Writing Agent

An AI-powered Blog Writing Agent that automates the process of researching, planning, writing, and generating images for high-quality blog content. Built using Python, LangGraph, Streamlit, and LLM-powered tools.

## 🚀 Overview

Blog Writing Agent is an AI-powered content creation application designed to simplify and automate blog writing. It uses a multi-step AI workflow to research topics, create structured blog plans, generate content, and produce relevant images.

The application combines AI research, intelligent task planning, content generation, and image generation in one workflow.

## ✨ Key Features

- **AI-Powered Blog Generation:** Generate structured and informative blog posts from a user-provided topic.
- **Automated Research:** Collect relevant information and supporting evidence using Tavily and web search capabilities.
- **Intelligent Blog Planning:** Break down a topic into smaller writing tasks using an LLM-powered planner.
- **Multi-Step AI Workflow:** Manage planning and writing tasks through LangGraph.
- **Parallel Content Generation:** Process multiple writing tasks using LangGraph's worker-based workflow.
- **AI Image Generation:** Generate relevant blog images using the Pollinations AI API.
- **Blog Preview:** View generated blog content and images through the Streamlit interface.
- **Markdown Support:** Generate and preview blog content in Markdown format.
- **Image Management:** View generated images and their generation status.
- **Download Support:** Download generated blog content and available images in a ZIP package.
- **Research Evidence:** Display supporting research information and sources.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive frontend |
| LangGraph | AI workflow orchestration |
| LangChain | LLM integration |
| Groq API | Fast LLM inference |
| Tavily API | Web research |
| Pollinations AI | AI image generation |
| Requests | HTTP API communication |
| Pydantic | Data validation |
| Markdown | Blog content formatting |
| Git & GitHub | Version control |

## 🏗️ Project Structure

```text
blog-writing-agent/
│
├── bwa_backend.py          # AI workflow and backend logic
├── bwa_frontend.py         # Streamlit user interface
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .gitignore              # Git ignored files
├── README.md               # Project documentation
│
└── generated_images/       # Generated images (if configured)
```

## ⚙️ How It Works

The application follows a multi-stage AI content generation workflow:

1. **Topic Input:** The user enters a blog topic.
2. **Research:** The system collects relevant information and supporting evidence.
3. **Planning:** The AI planner creates a structured blog plan and writing tasks.
4. **Parallel Writing:** LangGraph distributes writing tasks to worker nodes.
5. **Content Assembly:** Generated sections are combined into a complete blog.
6. **Image Generation:** Pollinations AI generates images based on image prompts.
7. **Preview:** The generated blog and images are displayed in Streamlit.
8. **Download:** Users can download the generated content and images.

## 📋 Prerequisites

Before running the project, make sure you have:

- Python installed
- Git installed
- VS Code or another Python IDE
- Groq API key
- Tavily API key (for research)
- Pollinations API key (for image generation)

## 📥 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/faizankhan07867/blog-writing-agent.git
cd blog-writing-agent
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root directory.

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
POLLINATIONS_API_KEY=your_pollinations_api_key
```

Replace the placeholder values with your actual API keys.

**Important:** Never upload your `.env` file or expose your API keys on GitHub.

### 5. Run the Application

Start the Streamlit application:

```bash
streamlit run bwa_frontend.py
```

Open the local URL displayed in your terminal to access the application.

## 🔑 API Configuration

| API | Usage |
|---|---|
| Groq | AI-powered planning and blog writing |
| Tavily | Research and information gathering |
| Pollinations AI | AI-generated blog images |

Make sure the required API keys are configured correctly in your `.env` file.

## 🧠 AI Workflow Architecture

```text
          User Topic
               |
               v
         Research Stage
               |
               v
          AI Planner
               |
               v
       LangGraph Workflow
               |
         +-----+-----+
         |     |     |
         v     v     v
       Writer Writer Writer
         |     |     |
         +-----+-----+
               |
               v
       Blog Content Assembly
               |
               v
        AI Image Generation
               |
               v
         Streamlit Preview
               |
               v
         Download Results
```

## 🔒 Security

- Store API keys in environment variables.
- Never commit `.env` files.
- Use `.gitignore` to exclude secrets and temporary files.
- Do not share private API credentials publicly.

## 🚧 Future Improvements

- User authentication and authorization.
- Blog export to PDF and DOCX.
- More AI model integrations.
- SEO optimization and keyword analysis.
- Multiple blog writing styles and tones.
- Cloud deployment and production hosting.
- Blog history and database integration.
- Advanced content editing capabilities.

## 👨‍💻 Author

**Faizan Khan**

B.Tech Information Technology Student | Python Developer | AI & Generative AI Enthusiast

- GitHub: https://github.com/faizankhan07867

## ⭐ Support

If you find this project useful, consider giving it a star ⭐ on GitHub.

---

**Built with Python, LangGraph, Streamlit, and Generative AI.**
