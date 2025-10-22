import gradio as gr
import os
from dotenv import load_dotenv
import requests
# FIX: Import quote directly from urllib.parse to resolve IDE/linter warning
from urllib.parse import quote
# Google Generative AI SDK update
from google import genai
from datetime import datetime, timedelta
from PIL import Image
import io

# ----------------------------------------------------------------------
# *** MODEL CONFIGURATION - CORRECTED FOR STABILITY ***
# ----------------------------------------------------------------------

# CHANGED Image Generation Model: Using Stability AI's Stable Diffusion XL Base 1.0
# This model is more robust and generally available on the Inference API.
HF_IMAGE_GEN_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

# Other Text Model
HF_TEXT_MODEL = "google/flan-t5-large"

# Load API keys from .env file
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Track API reset time
api_reset_time = None


# -----------------------------
# Function to query any available AI model (Text Q&A)
# -----------------------------
def query_model(prompt):
    """Routes the text prompt to the highest priority available LLM API."""
    global api_reset_time

    if api_reset_time and datetime.now() < api_reset_time:
        reset_time_str = api_reset_time.strftime("%d-%m-%Y at %I:%M %p")
        return f"Daily limit reached. Try again on {reset_time_str}"

    if GEMINI_API_KEY:
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            error_text = str(e).lower()
            if "quota" in error_text or "unavailable" in error_text:
                api_reset_time = datetime.now() + timedelta(hours=12)
                print(f"Gemini API limit hit: {e}")
                pass
            print(f"Gemini API failed: {e}")

    if OPENAI_KEY:
        try:
            import openai
            openai.api_key = OPENAI_KEY
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message["content"].strip()
        except Exception as e:
            error_text = str(e).lower()
            if "quota" in error_text or "insufficient_quota" in error_text:
                api_reset_time = datetime.now() + timedelta(hours=12)
                print(f"OpenAI API limit hit: {e}")
                pass
            print(f"OpenAI failed: {e}")

    if HF_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            data = {"inputs": prompt}
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{HF_TEXT_MODEL}",
                headers=headers,
                json=data
            )
            output = response.json()
            if isinstance(output, list) and len(output) > 0 and "generated_text" in output[0]:
                return output[0]["generated_text"]

            return f"Hugging Face Text Error: {output}"
        except Exception as e:
            if "quota" in str(e).lower():
                api_reset_time = datetime.now() + timedelta(hours=12)
                print(f"Hugging Face Text API limit hit: {e}")
            print(f"Hugging Face Text failed: {e}")

    reset_time_str = (api_reset_time.strftime("%d-%m-%Y at %I:%M %p")
                      if api_reset_time else "the keys are not configured correctly.")
    return f"Right now all APIs are limited or unavailable. Please check your keys or try again on {reset_time_str}"


# -----------------------------
# IMAGE GENERATION FUNCTION (FIXED)
# -----------------------------
def generate_image(prompt):
    """Generates an image using the Stable Diffusion model via HF Inference API."""
    global api_reset_time

    if api_reset_time and datetime.now() < api_reset_time:
        reset_time_str = api_reset_time.strftime("%d-%m-%Y at %I:%M %p")
        return f"Daily limit reached. Try again on {reset_time_str}", None

    if not HF_API_KEY:
        return "Hugging Face API key not found.", None

    try:
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        payload = {
            "inputs": prompt,
            "options": {"wait_for_model": True},
            # SDXL (the new model) works best with 1024x1024, but 768x768 is a good compromise for speed/API limitations
            "parameters": {"width": 768, "height": 768}
        }

        # FIX: Properly URL-encode the model ID to avoid 404 errors.
        encoded_model_id = quote(HF_IMAGE_GEN_MODEL, safe='')
        api_url = f"https://api-inference.huggingface.co/models/{encoded_model_id}"

        response = requests.post(
            api_url,
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
            image_bytes = response.content
            img = Image.open(io.BytesIO(image_bytes))

            # Ensure the "generated" directory exists for saving
            os.makedirs("generated", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"generated/generated_image_{timestamp}.png"
            img.save(output_path)

            return "Image generated successfully! (Saved as " + output_path + ")", img
        else:
            if "quota" in response.text.lower() or "not authorized" in response.text.lower() or response.status_code == 401:
                api_reset_time = datetime.now() + timedelta(hours=12)
                reset_time_str = api_reset_time.strftime("%d-%m-%Y at %I:%M %p")
                return (f"Hugging Face API limit or access issue reached. Check Quota or ensure "
                        f"you accepted the license for '{HF_IMAGE_GEN_MODEL}'. Try again on {reset_time_str}"), None

            # This handles the original 404 error, which should now be resolved with the correct model ID.
            return f"Image Generation Error: HTTP {response.status_code} - {response.text}", None

    except Exception as e:
        return f"Image generation failed due to a connection or internal error: {e}", None


# -----------------------------
# IMAGE-BASED Q&A FUNCTION
# -----------------------------
def query_image_model(image, prompt):
    """Answers a question about an image using the Gemini API."""
    global api_reset_time

    if api_reset_time and datetime.now() < api_reset_time:
        reset_time_str = api_reset_time.strftime("%d-%m-%Y at %I:%M %p")
        return f"Daily limit reached. Try again on {reset_time_str}"

    if image is None:
        return "Error: Please upload an image first."
    if not GEMINI_API_KEY:
        return "Error: Gemini API Key is required for reliable Image Q&A."

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        contents = [image, prompt]

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents
        )
        return response.text

    except Exception as e:
        error_text = str(e).lower()
        if "quota" in error_text or "unavailable" in error_text:
            api_reset_time = datetime.now() + timedelta(hours=12)
            reset_time_str = api_reset_time.strftime("%d-%m-%Y at %I:%M %p")
            return f"Gemini API limit reached. Try again on {reset_time_str}"

        return f"Image-based Q&A failed using Gemini API: {e}"


# -----------------------------
# GRADIO UI
# -----------------------------
with gr.Blocks(title="Manjula AI Assistance", fill_width=True) as demo:
    gr.Markdown("# Manjula AI Assistance")
    gr.Markdown("Hello There, How May I help You?")

    # 1. Text Q&A Tab
    with gr.Tab("Text Q&A"):
        with gr.Row():
            input_box = gr.Textbox(lines=2, placeholder="Ask anything...", label="Your Prompt")
            submit_btn = gr.Button("Get Answer")
        # Textbox with fixed size for consistency
        output_box = gr.Textbox(label="AI Response", lines=15, max_lines=None, interactive=False)
        submit_btn.click(fn=query_model, inputs=input_box, outputs=output_box)

    # 2. Image Generation Tab
    with gr.Tab("Image Generation"):
        gr.Markdown("## 🖼️ AI Image Generator")
        with gr.Row():
            img_prompt = gr.Textbox(lines=2, placeholder="Describe the image you want...", label="Image Prompt")
            gen_btn = gr.Button("Generate Image")
        img_output_text = gr.Textbox(label="Status", interactive=False)
        img_output = gr.Image(label="Generated Image", type="pil")
        gen_btn.click(fn=generate_image, inputs=img_prompt, outputs=[img_output_text, img_output])

    # 3. Image Q&A Tab (Visual Question Answering)
    with gr.Tab("Ask about an Image"):
        gr.Markdown("## 🧠 Visual Question Answering (VQA) powered by Gemini")
        with gr.Row():
            img_upload = gr.Image(type="pil", label="Upload an Image")
        with gr.Row():
            img_qa_prompt = gr.Textbox(
                lines=2,
                placeholder="Ask anything about the uploaded image (e.g., 'What is this?', 'Describe the graph')...",
                label="Your Question"
            )
            img_qa_btn = gr.Button("Ask about Image")

        img_qa_output = gr.Textbox(
            label="AI Response",
            lines=5,
            max_lines=50,
            interactive=False
        )

        img_qa_btn.click(fn=query_image_model, inputs=[img_upload, img_qa_prompt], outputs=img_qa_output)

# -----------------------------
# Launch the app
# -----------------------------
demo.launch()
