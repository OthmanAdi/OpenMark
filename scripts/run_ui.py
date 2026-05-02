import os
import sys
os.environ["GRADIO_SERVER_PORT"] = "7861"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openmark.ui.app import build_ui
import gradio as gr

ui = build_ui()
ui.launch(
    server_name="127.0.0.1",
    server_port=7861,
    share=False,
    inbrowser=True,
    theme=gr.themes.Base(primary_hue="indigo", neutral_hue="slate"),
)
