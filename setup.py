from setuptools import setup, find_packages

setup(
    name="clinical_guidelines",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.30.0",
        "langchain>=0.1.0",
        "pydantic>=2.0.0",
        "transformers>=4.36.0",
        "torch>=2.1.0",
        "accelerate>=0.25.0",
        "sentence-transformers>=2.2.2",
        "chromadb>=0.4.0",
        "fastapi>=0.100.0",
        "python-dotenv>=1.0.0",
        "redis>=5.0.0",
        "langgraph>=0.0.15",
        "guardrails-ai>=0.3.0",
        "plotly>=5.18.0",
        "litellm>=1.26.8"
    ],
    python_requires=">=3.8",
) 