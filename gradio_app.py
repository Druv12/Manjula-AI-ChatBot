import gradio as gr
import os
from dotenv import load_dotenv
import requests
from urllib.parse import quote
from google import genai  # Corrected usage with modern SDK
from datetime import datetime, timedelta
from PIL import Image
import io
import time  # Added for exponential backoff simulation
import openai  # Added for cleaner OpenAI client access

# ----------------------------------------------------------------------
# MODEL CONFIGURATION
# ----------------------------------------------------------------------
HF_IMAGE_GEN_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
HF_TEXT_MODEL = "google/flan-t5-large"
HF_VIDEO_MODEL = "Lightricks/LTX-Video-0.9.8-13B-distilled"

# Load API keys
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

api_reset_time = None
GEMINI_CLIENT = None

# Global Initialization for the modern google-genai SDK
if GEMINI_API_KEY:
    try:
        # Client initialization is done once, often at module level
        GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Failed to initialize Gemini Client: {e}")
        GEMINI_CLIENT = None


# ----------------------------- TEXT Q&A -----------------------------
def query_model(prompt):
    global api_reset_time

    if api_reset_time and datetime.now() < api_reset_time:
        reset_time_str = api_reset_time.strftime("%d-%m-%Y at %I:%M %p")
        return f"Daily limit reached. Try again on {reset_time_str}"

    # 1. GEMINI API (Modern SDK Usage)
    if GEMINI_CLIENT:
        try:
            # Use the global client and the modern generate_content method
            response = GEMINI_CLIENT.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            error_text = str(e).lower()
            # Catch known quota/unavailable errors
            if "quota" in error_text or "unavailable" in error_text or "429" in error_text:
                api_reset_time = datetime.now() + timedelta(hours=12)
                reset_time_str = api_reset_time.strftime("%d-%m-%Y at %I:%M %p")
                print(f"Gemini API limit reached: {e}")
                # Fallback to the next model for this specific call
            else:
                print(f"Gemini API failed: {e}")

    # 2. OPENAI API
    if OPENAI_KEY:
        try:
            # Use client-based API for modern standard
            client = openai.OpenAI(api_key=OPENAI_KEY)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI failed: {e}")
            # Consider adding logic for OpenAI RateLimitError (429) here

    # 3. HUGGING FACE API
    if HF_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            data = {"inputs": prompt}
            # Simple retry/wait for HuggingFace cold-start models
            for attempt in range(3):
                response = requests.post(
                    f"https://api-inference.huggingface.co/models/{HF_TEXT_MODEL}",
                    headers=headers,
                    json=data
                )
                output = response.json()
                if isinstance(output, list) and len(output) > 0 and "generated_text" in output[0]:
                    return output[0]["generated_text"]

                # Check for "loading" or "currently loading" in HF response
                if response.status_code == 503 and "loading" in response.text:
                    wait_time = output.get("estimated_time", 5)  # Use estimated time or default to 5s
                    print(f"Hugging Face model loading, waiting for {wait_time}s... (Attempt {attempt + 1})")
                    time.sleep(wait_time)
                else:
                    break  # Break if it's a hard error or success

            return f"Hugging Face Text Error: {output.get('error', 'Unknown error') if isinstance(output, dict) else output}"
        except Exception as e:
            print(f"Hugging Face Text failed: {e}")

    return "All APIs are unavailable or failed after fallback. Check your keys or try later."


# ----------------------------- IMAGE GENERATION (Corrected for SDXL Quality) -----------------------------
def generate_image(prompt):
    global api_reset_time

    if api_reset_time and datetime.now() < api_reset_time:
        reset_time_str = api_reset_time.strftime("%d-%m-%Y at %I:%M %p")
        return f"Daily limit reached. Try again on {reset_time_str}", None

    if not HF_API_KEY:
        return "Hugging Face API key not found.", None

    try:
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        # OPTIMIZED PARAMETERS FOR SDXL (1024x1024 is essential for quality)
        payload = {
            "inputs": prompt,
            "options": {"wait_for_model": True},
            "parameters": {
                "width": 1024,  # <-- SDXL OPTIMAL WIDTH
                "height": 1024,  # <-- SDXL OPTIMAL HEIGHT
                "num_inference_steps": 50,  # <-- Good balance of speed/quality
                "guidance_scale": 7.5  # <-- Standard quality setting
            }
        }

        encoded_model_id = quote(HF_IMAGE_GEN_MODEL, safe='')
        api_url = f"https://api-inference.huggingface.co/models/{encoded_model_id}"

        # Increased timeout as SDXL on HF API can be slow (cold start)
        response = requests.post(api_url, headers=headers, json=payload, timeout=300)

        if response.status_code == 200:
            image_bytes = response.content
            img = Image.open(io.BytesIO(image_bytes))

            os.makedirs("generated", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"generated/generated_image_{timestamp}.png"
            img.save(output_path)

            return "Image generated successfully! (SDXL 1024x1024)", img
        else:
            resp_text = response.text.lower()
            if "quota" in resp_text or response.status_code in (401, 403, 429):
                api_reset_time = datetime.now() + timedelta(hours=12)
                reset_time_str = api_reset_time.strftime("%d-%m-%Y at %I:%M %p")
                return (f"Hugging Face API limit or access issue reached. Try again on {reset_time_str}"), None
            return f"Image Generation Error: {response.status_code} - {response.text}", None
    except requests.exceptions.RequestException as e:
        return f"Image generation failed due to network error/timeout (Try again, it can be slow): {e}", None
    except Exception as e:
        return f"Image generation failed: {e}", None


# ----------------------------- IMAGE Q&A -----------------------------
def query_image_model(image, prompt):
    global api_reset_time

    if api_reset_time and datetime.now() < api_reset_time:
        reset_time_str = api_reset_time.strftime("%d-%m-%Y at %I:%M %p")
        return f"Daily limit reached. Try again on {reset_time_str}"

    if image is None:
        return "Error: Please upload an image first."
    if not GEMINI_CLIENT:
        return "Error: Gemini API Key is required and client failed to initialize for Image Q&A."

    try:
        # Use the global client and the modern generate_content method
        # gemini-2.5-flash is a multimodal model (Gemini-Pro-Vision is a legacy ID)
        response = GEMINI_CLIENT.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image, prompt]  # Corrected format: [image, prompt]
        )
        return response.text
    except Exception as e:
        error_text = str(e).lower()
        if "quota" in error_text or "unavailable" in error_text or "429" in error_text:
            api_reset_time = datetime.now() + timedelta(hours=12)
            reset_time_str = api_reset_time.strftime("%d-%m-%Y at %I:%M %p")
            return f"Gemini API limit reached. Try again on {reset_time_str}"
        return f"Image-based Q&A failed: {e}"


# ----------------------------- VIDEO GENERATION (fixed to Lightricks) -----------------------------
def generate_video(prompt):
    """Generate short video using Lightricks/LTX-Video-0.9.8-13B-distilled via Hugging Face Inference API."""
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
            "parameters": {
                "fps": 8,
                "num_frames": 24,
            },
            "options": {"wait_for_model": True}
        }

        model_id = quote(HF_VIDEO_MODEL, safe='')
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"

        response = requests.post(api_url, headers=headers, json=payload, stream=True, timeout=120)

        if response.status_code == 200:
            os.makedirs("generated_videos", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = f"generated_videos/generated_video_{timestamp}.mp4"
            with open(video_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return "Video generated successfully!", video_path
        else:
            resp_text = response.text.lower()
            if "quota" in resp_text or response.status_code in (401, 403, 429):
                api_reset_time = datetime.now() + timedelta(hours=12)
                reset_time_str = api_reset_time.strftime("%d-%m-%Y at %I:%M %p")
                return (
                    f"Hugging Face API limit or access issue reached for video model. Try again on {reset_time_str}"), None
            return f"Video Generation Error: {response.status_code} - {response.text}", None
    except requests.exceptions.RequestException as e:
        return f"Video generation failed due to network error: {e}", None
    except Exception as e:
        return f"Video generation failed: {e}", None


# ----------------------------- GRADIO UI -----------------------------
with gr.Blocks(title="Manjula AI Assistance", fill_width=True) as demo:
    gr.Markdown("# 🤖 Manjula AI Assistance")
    gr.Markdown("Hello There, How May I help You?")

    # 1. Text Q&A
    with gr.Tab("Text Q&A"):
        gr.Markdown("Uses **Gemini** $\\rightarrow$ **OpenAI** $\\rightarrow$ **Hugging Face (Flan-T5)** as fallback.")
        with gr.Row():
            input_box = gr.Textbox(lines=2, placeholder="Ask anything...", label="Your Prompt")
            submit_btn = gr.Button("Get Answer")
        output_box = gr.Textbox(label="AI Response", lines=15, interactive=False)
        submit_btn.click(fn=query_model, inputs=input_box, outputs=output_box)

    # 2. Image Generation
    with gr.Tab("Image Generation (SDXL Quality)"):
        gr.Markdown("Uses **Hugging Face Stable Diffusion XL (SDXL)** 1024x1024.")
        with gr.Row():
            img_prompt = gr.Textbox(lines=2,
                                    placeholder="Describe the image you want (e.g., An anaconda fused with a human being, photorealistic, cinematic lighting)...",
                                    label="Image Prompt")
            gen_btn = gr.Button("Generate Image")
        img_output_text = gr.Textbox(label="Status", interactive=False)
        img_output = gr.Image(label="Generated Image (1024x1024)", type="pil")
        gen_btn.click(fn=generate_image, inputs=img_prompt, outputs=[img_output_text, img_output])

    # 3. Image Q&A
    with gr.Tab("Ask about an Image (Gemini)"):
        gr.Markdown("Uses **Gemini 2.5 Flash** (multimodal) for image analysis.")
        with gr.Row():
            img_upload = gr.Image(type="pil", label="Upload an Image")
        with gr.Row():
            img_qa_prompt = gr.Textbox(
                lines=2,
                placeholder="Ask anything about the image...",
                label="Your Question"
            )
            img_qa_btn = gr.Button("Ask about Image")
        img_qa_output = gr.Textbox(label="AI Response", lines=5, interactive=False)
        img_qa_btn.click(fn=query_image_model, inputs=[img_upload, img_qa_prompt], outputs=img_qa_output)

    # 4. Video Generation
    with gr.Tab("🎥 Video Generation"):
        gr.Markdown("Uses **Hugging Face LTX-Video** model.")
        with gr.Row():
            vid_prompt = gr.Textbox(lines=2, placeholder="Describe the video scene...", label="Video Prompt")
            vid_btn = gr.Button("Generate Video")
        vid_output_text = gr.Textbox(label="Status", interactive=False)
        vid_output = gr.Video(label="Generated Video")
        vid_btn.click(fn=generate_video, inputs=vid_prompt, outputs=[vid_output_text, vid_output])

# ----------------------------- Launch -----------------------------
demo.launch()
