import os
import sys
import torch
import streamlit.web.bootstrap as bootstrap
from streamlit.web.cli import main
from streamlit.runtime.scriptrunner import get_script_run_ctx

def run_streamlit():
    # Configure PyTorch
    if torch.backends.mps.is_available():
        # Set environment variable to control MPS fallback
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # Set environment variables for Streamlit
    os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    
    # Run the Streamlit app
    sys.argv = ["streamlit", "run", "ui/app.py"]
    sys.exit(main())

if __name__ == "__main__":
    run_streamlit() 