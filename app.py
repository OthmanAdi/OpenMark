"""HuggingFace Space entry point — launches the OpenMark Gradio UI."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from openmark.ui.app import build_ui

if __name__ == "__main__":
    ui = build_ui()
    ui.launch()
