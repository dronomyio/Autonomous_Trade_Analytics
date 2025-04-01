"""
Environment Variable Loader for Trade Analytics
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def load_environment():
    """Load environment variables from .env file"""
    
    # Try to load from .env
    env_path = Path('.') / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print("Loaded environment variables from .env file")
    else:
        print("No .env file found, using environment variables from system")
    
    # Check for required API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print(
            "WARNING: ANTHROPIC_API_KEY environment variable not set.\n"
            "Please provide your Anthropic API key either in the .env file or as an environment variable."
        )
        return False
    
    return True

# Auto-run when imported
load_environment()