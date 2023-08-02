
from pathlib import Path
import gradio as gr
from datetime import datetime
import sys






with gr.Blocks(
    # theme=gr.themes.Soft(),
    css=".disclaimer {font-variant-caps: all-small-caps;}",
) as demo:
    gr.Markdown("""
## Gradio + FastAPI + Static Server
This is a demo of how to use Gradio with FastAPI and a static server.
The Gradio app generates dynamic HTML files and stores them in a static directory. FastAPI serves the static files.
""")
    with gr.Row():
        with gr.Column():
            html = gr.HTML(value=f"""<iframe src="http://157.0.19.2:10327/#" width="100%" height="500px"></iframe>""",label="HTML preview", show_label=True)


demo.launch(debug=True, share=False)