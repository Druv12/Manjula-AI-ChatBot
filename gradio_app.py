import gradio as gr
import os
from dotenv import load_dotenv
import requests
from google import genai

# Load API keys
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# Function to query any available AI model
def query_model(prompt):
    # GEMINI API
    if GEMINI_API_KEY:
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            print("Gemini API failed:", e)

    # OPENAI Fallback
    if OPENAI_API_KEY:
        try:
            import openai
            openai.api_key = OPENAI_API_KEY
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message["content"].strip()
        except Exception as e:
            print("OpenAI failed:", e)

    # HUGGING FACE Fallback
    if HF_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            data = {"inputs": prompt}
            response = requests.post(
                "https://api-inference.huggingface.co/models/google/flan-t5-large",
                headers=headers,
                json=data
            )
            output = response.json()
            if isinstance(output, list) and len(output) > 0 and "generated_text" in output[0]:
                return output[0]["generated_text"]
            return str(output)
        except Exception as e:
            print("Hugging Face failed:", e)

    return "Right Now You Reached Your Daily Limit Please Try Later"


# ----------------------------------------------------
# GRADI UI - Using gr.Blocks to apply external CSS
# ----------------------------------------------------

with gr.Blocks(
        title="Manjula AI Assistance",
        # 1. LINK THE EXTERNAL CSS FILE
        css_paths=["style.css"],
        fill_width=True
) as demo:
    # Title and Description using Markdown for custom styling
    gr.Markdown("# Manjula AI Assistance")
    gr.Markdown("Hello There How May I help You")

    # Layout: Row for input and button
    with gr.Row():
        # Input Textbox, applying custom CSS class
        input_box = gr.Textbox(
            lines=2,
            placeholder="Ask anything...",
            label="Your Prompt",
            # 2. APPLY CUSTOM CLASS from style.css
            elem_classes=["prompt-box"]
        )

        # Submission Button
        submit_btn = gr.Button("Get Answer")

    # Output Textbox
    output_box = gr.Textbox(
        label="AI Response",
        lines=15,  # Minimum starting size (e.g., 15 lines)
        max_lines=None,  # Allows expansion beyond the minimum
        interactive=False,
        # Assign a class to target with CSS
        elem_classes=["response-box"]
    )

    # 3. DEFINE THE ACTION
    submit_btn.click(
        fn=query_model,
        inputs=input_box,
        outputs=output_box
    )

# 4. LAUNCH THE APP, ALLOWING ACCESS TO IMAGES
demo.launch(
    allowed_paths=["image_52715f.png", "image_516e5c.png"]
)