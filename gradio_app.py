import gradio as gr
import os
from dotenv import load_dotenv
import requests
from urllib.parse import quote
from google import genai
from google.genai import types
from datetime import datetime, timedelta
from PIL import Image
from PIL.Image import Resampling
import io
import time
import openai
from requests.adapters import HTTPAdapter, Retry
import json

# --- Configuration & Environment Setup ---
load_dotenv()

STABILITY_API_HOST = os.getenv("STABILITY_API_HOST", "https://api.stability.ai")
HF_API_KEY = os.getenv("HF_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_SEARCH_CX = os.getenv("GOOGLE_SEARCH_CX")

# --- External Library Imports with Fallback ---
try:
    from googleapiclient.discovery import build
except ImportError:
    print("WARNING: 'googleapiclient' library not found. Image search will fail.")
    build = None

try:
    import replicate
except ImportError:
    print("WARNING: 'replicate' library not found.")
    replicate = None

# Rate limit timers for features
api_reset_times = {
    "text_qa": None,
    "image_gen": None,
    "image_qa": None,
    "image_search": None,
    "video_gen": None,
    "public_ip": None,
    "file_qa": None,
}

# NEW: Comprehensive history storage for ALL features
history_storage = {
    "chat": [],
    "file_qa": [],
    "image_gen": [],
    "video_gen": [],
    "image_search": [],
    "image_qa": [],
    "public_ip": []
}
current_session_id = 0

GEMINI_CLIENT = None
if GEMINI_API_KEY:
    try:
        GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Failed to initialize Gemini Client: {e}")
        GEMINI_CLIENT = None


# --- Rate Limit Functions ---
def check_rate_limit(task_key):
    reset_time = api_reset_times.get(task_key)
    now = datetime.now()
    if reset_time and now < reset_time:
        remaining = reset_time - now
        hours, remainder = divmod(int(remaining.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"⚠️ You have reached the {task_key.replace('_', ' ')} limit. Try again in {hours}h {minutes}m {seconds}s."
    return None


def get_timer_text(task_key, label=None):
    reset_time = api_reset_times.get(task_key)
    now = datetime.now()
    label = label or task_key.replace('_', ' ').capitalize()
    if reset_time and now < reset_time:
        remaining = reset_time - now
        hours, remainder = divmod(int(remaining.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"⏳ {label} limit reached. Refreshes in {hours}h {minutes}m {seconds}s."
    return f"✅ {label} available now."


def set_rate_limit(task_key):
    api_reset_times[task_key] = datetime.now() + timedelta(hours=12)


# --- Session and Retry Logic ---
def create_session_with_retries(total_retries=5, backoff_factor=0.5):
    status_forcelist = [429, 500, 502, 503, 504]
    retries = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods={"POST"},
    )
    session = requests.Session()
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


# --- Universal History Management ---
def add_to_history(category, user_query, response, metadata=None):
    """Add any interaction to history"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Ensure query and response are strings and handle None
    user_query_str = str(user_query).strip() if user_query is not None else ""
    response_str = str(response).strip() if response is not None else ""

    entry = {
        "timestamp": timestamp,
        "query": user_query_str[:100],  # Truncate long queries for quick view
        "response": response_str[:200],  # Truncate long responses for quick view
        "full_query": user_query_str,
        "full_response": response_str,
        "metadata": metadata or {}
    }
    history_storage[category].insert(0, entry)  # Add to beginning
    # Keep only last 100 entries per category
    if len(history_storage[category]) > 100:
        history_storage[category] = history_storage[category][:100]


def format_history_for_chatbot():
    """Converts the comprehensive history storage into a list of tuples for gr.Chatbot"""
    chatbot_history = []
    categories = {
        "chat": "💬 Chat History",
        "file_qa": "📄 File Q&A History",
        "image_gen": "🎨 Image Generation History",
        "video_gen": "🎥 Video Generation History",
        "image_search": "🖼️ Image Search History",
        "image_qa": "🔍 Image Q&A History",
        "public_ip": "🌐 Public IP History"
    }

    has_any_history = False

    for category, title in categories.items():
        entries = history_storage.get(category, [])
        if entries:
            has_any_history = True

            # Category Header as the first message from the "user"
            chatbot_history.append((
                f"--- **{title}** ({len(entries)} items) ---",
                "**Scroll down to view history entries.**"
            ))

            for entry in entries:
                timestamp = entry["timestamp"]
                full_query = entry.get("full_query", "No Query Provided")
                full_response = entry.get("full_response", "No Response Provided")

                # Format each history item into a (User Question, AI Response) tuple
                query_text = f"**{timestamp}** | Q: {full_query}"
                response_text = f"A: {full_response}"

                chatbot_history.append((query_text, response_text))

    if not has_any_history:
        # If no history, send a single "system" message
        chatbot_history.append((None, "📭 **No history yet.** Start using the features to see your activity here!"))

    return chatbot_history


def show_history_modal():
    """Open history modal and populate the chatbot"""
    return gr.update(visible=True), format_history_for_chatbot()


def close_history_modal():
    """Close history modal"""
    return gr.update(visible=False)


def clear_all_history_action():
    """Clear all history from all categories"""
    for category in history_storage:
        history_storage[category] = []
    # Return empty list for the chatbot and success status
    return [], "✅ All history cleared!"


# --- Core Chat Function with History Saving ---
def query_model(prompt, history, session_id_state):
    # FIX: Immediately return if the prompt is empty to avoid logging empty conversations
    if not prompt or not prompt.strip():
        return history, ""

    limit_msg = check_rate_limit("text_qa")
    if limit_msg:
        history.append((prompt, limit_msg))
        return history, ""

    # Create new session if needed
    if session_id_state is None or session_id_state == 0:
        global current_session_id
        current_session_id += 1
        session_id_state = current_session_id

    llm_messages = []
    for user_msg, assistant_msg in history:
        if user_msg:
            llm_messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            llm_messages.append({"role": "model", "content": assistant_msg})

    llm_messages.append({"role": "user", "content": prompt})

    answer = "Error: No LLM client configured."
    llm_name = "N/A"

    if GEMINI_CLIENT:
        llm_name = "Gemini"
        try:
            gemini_formatted_messages = []
            for msg in llm_messages:
                # Gemini API expects 'user' and 'model' roles
                role = "model" if msg["role"] == "assistant" else msg["role"]
                gemini_formatted_messages.append(
                    types.Content(role=role, parts=[types.Part(text=msg["content"])])
                )

            response = GEMINI_CLIENT.models.generate_content(
                model="gemini-2.5-flash",
                contents=gemini_formatted_messages
            )
            answer = response.text
        except Exception as e:
            err_str = str(e).lower()
            if any(x in err_str for x in ["quota", "unavailable", "429"]):
                set_rate_limit("text_qa")
                answer = f"⚠️ Text Q&A ({llm_name}): Daily limit reached. Try again after 12 hours."
            else:
                answer = f"⚠️ Gemini failed: {e}"
            print(f"Gemini API Error: {e}")

    # FIX: Added 'answer is None' check to prevent TypeError if Gemini API fails unexpectedly
    if (answer is None or "Gemini failed" in answer or llm_name == "N/A") and OPENAI_KEY:
        llm_name = "OpenAI"
        try:
            client = openai.OpenAI(api_key=OPENAI_KEY)
            openai_formatted_messages = []
            for msg in llm_messages:
                # OpenAI API expects 'user' and 'assistant' roles
                role = "assistant" if msg["role"] == "model" else msg["role"]
                openai_formatted_messages.append({"role": role, "content": msg["content"]})

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=openai_formatted_messages
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"⚠️ OpenAI failed: {e}"
            print(f"OpenAI API Error: {e}")

    if answer is None or answer.startswith("Error: No LLM client"):
        answer = "Error: No LLM client configured (Missing GEMINI_API_KEY or OPENAI_API_KEY)."

    history.append((prompt, answer))

    # Save to history
    add_to_history("chat", prompt, answer, {"model": llm_name})

    return history, ""


def start_new_chat():
    """Start a fresh chat session"""
    return [], None


# --- File Content Extraction (Functions remain the same) ---
def extract_file_content_gemini(file, prompt):
    if not GEMINI_CLIENT:
        return "Error: Gemini API Key missing."
    if not file:
        return "Error: No file uploaded."

    uploaded_file = None
    try:
        file_path = file.name
        ext = os.path.splitext(file_path)[-1].lower()

        if ext in ['.docx', '.txt']:
            return "Error: Unsupported file type for multimodal extraction (DOCX/TXT require dedicated parsers)."

        uploaded_file = GEMINI_CLIENT.files.upload(file=file_path)

        extraction_prompt = (
            f"Analyze the attached document/image. Perform Optical Character Recognition (OCR), "
            f"extract all text, tables, and key data points. Format the output as clean, searchable **Markdown**. "
            f"Specifically address the user's query: '{prompt}'. "
            f"If the document contains forms, tables, or structured lists, prioritize markdown table recreation."
        )

        contents = [uploaded_file, extraction_prompt]

        response = GEMINI_CLIENT.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents
        )

        if response.text:
            return response.text
        else:
            return "Empty extraction result from Gemini."

    except Exception as e:
        return f"Extraction failed: {e}"

    finally:
        if uploaded_file and uploaded_file.name:
            try:
                GEMINI_CLIENT.files.delete(name=uploaded_file.name)
            except Exception:
                pass


def file_question_answer(file, question):
    limit_msg = check_rate_limit("file_qa")
    if limit_msg:
        return limit_msg
    if not file or not question:
        return "Please upload a file and enter a question."

    content = extract_file_content_gemini(file, question)

    if content.lower().startswith("error:") or content.lower().startswith("extraction failed:"):
        result = f"Error processing file: {content}"
    else:
        result = f"**File Context Extracted:**\n\n{content}"

    # Save to history
    add_to_history("file_qa", question, content, {"filename": file.name if file else "unknown"})

    return result


# --- Image Generation Functions (Functions remain the same) ---
def generate_pollinations_ai(prompt: str) -> bytes | None:
    try:
        encoded_prompt = quote(prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&model=flux&nologo=true&enhance=true"
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response.content
    except Exception as e:
        print(f"Pollinations.ai error: {e}")
    return None


def generate_segmind_fallback(prompt: str) -> bytes | None:
    try:
        url = "https://api.segmind.com/v1/sd1.5-txt2img"
        data = {
            "prompt": prompt,
            "negative_prompt": "blurry, low quality, distorted",
            "samples": 1,
            "width": 768,
            "height": 768,
            "steps": 20,
            "seed": 0
        }
        response = requests.post(url, json=data, timeout=30)
        if response.status_code == 200:
            return response.content
    except Exception as e:
        print(f"Segmind error: {e}")
    return None


def generate_stability_ai(prompt: str, session: requests.Session) -> bytes | None:
    if not STABILITY_API_KEY:
        return None
    try:
        headers = {"Authorization": f"Bearer {STABILITY_API_KEY}", "Accept": "image/*"}
        data = {"prompt": prompt, "output_format": "png", "aspect_ratio": "1:1", "seed": 0}
        response = session.post(
            "https://api.stability.ai/v2beta/stable-image/generate/ultra",
            headers=headers, files={"none": ''}, data=data
        )
        if response.status_code == 200:
            return response.content
    except Exception as e:
        print(f"Stability AI error: {e}")
    return None


def generate_image_for_gradio(prompt: str):
    limit_msg = check_rate_limit("image_gen")
    if limit_msg:
        return None, f"{limit_msg}\n\n**FREE FALLBACK:** Use an external generator like Bing Image Creator."

    if not prompt:
        return None, "Please enter a prompt."

    image_bytes = generate_pollinations_ai(prompt) or generate_segmind_fallback(prompt)

    if image_bytes is None and STABILITY_API_KEY:
        session = create_session_with_retries()
        image_bytes = generate_stability_ai(prompt, session)

    if image_bytes:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            status_msg = "✅ Image generated successfully!"

            # Save to history
            add_to_history("image_gen", prompt, status_msg)

            return image, status_msg
        except Exception as e:
            return None, f"❌ Failed to process image bytes: {e}"

    return None, "❌ All internal image generation services failed. Please use an external 100% free tool like **Bing Image Creator**."


# --- Image QA (Function remains the same) ---
def query_image_model(image, prompt):
    limit_msg = check_rate_limit("image_qa")
    if limit_msg:
        return limit_msg
    if image is None:
        return "Error: Please upload an image first."
    if not GEMINI_CLIENT:
        return "Error: Gemini API Key missing."

    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)

        image_part = types.Part.from_bytes(
            data=img_byte_arr.getvalue(),
            mime_type='image/jpeg'
        )

        contents = [image_part, prompt]
        response = GEMINI_CLIENT.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents
        )

        result = response.text

        # Save to history
        add_to_history("image_qa", prompt, result)

        return result
    except Exception as e:
        return f"Image-based Q&A failed: {e}"


# --- Video Generation (Function remains the same) ---
def generate_video(prompt):
    limit_msg = check_rate_limit("video_gen")
    if limit_msg:
        return limit_msg, None
    if not HF_API_KEY:
        return "Error: Hugging Face API Key is not set (HF_API_KEY).", None
    if not prompt:
        return "Please enter a prompt.", None

    status_msg = f"Video generation for '{prompt}' is simulated. (Actual API call would go here)"

    # Save to history
    add_to_history("video_gen", prompt, status_msg)

    print(f"Simulating video generation for prompt: '{prompt}'")
    return status_msg, None


# --- Google Image Search (Function remains the same) ---
def google_image_search(query: str):
    limit_msg = check_rate_limit("image_search")
    if limit_msg:
        return None, limit_msg

    if build is None:
        return None, "Error: 'googleapiclient' not installed. Please `pip install google-api-python-client`."
    if not query:
        return None, "Please enter a search query."
    if not GOOGLE_SEARCH_API_KEY or not GOOGLE_SEARCH_CX:
        return None, "Error: Google Search API Key or CX missing."

    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_SEARCH_API_KEY)
        result = service.cse().list(
            q=query,
            cx=GOOGLE_SEARCH_CX,
            searchType='image',
            num=5,
            safe='active'
        ).execute()

        if 'items' not in result or not result['items']:
            status_msg = f"No image results found for: **{query}**"
            add_to_history("image_search", query, status_msg)
            return None, status_msg

        for item in result['items']:
            image_url = item['link']
            if any(bad in image_url for bad in ["instagram.com", "facebook.com", "pinterest.com", ".svg", ".gif"]):
                print(f"🚫 Skipping blocked domain/format: {image_url}")
                continue

            print(f"🔗 Trying image URL: {image_url}")
            try:
                session = create_session_with_retries()
                resp = session.get(image_url, stream=True, timeout=15)
                resp.raise_for_status()

                ctype = resp.headers.get("Content-Type", "").lower()
                if not ctype.startswith("image/"):
                    print(f"⚠️ Non-image content-type received: {ctype}")
                    continue

                image = Image.open(io.BytesIO(resp.content)).convert("RGB")
                # FIX: Use Image.Resampling.LANCZOS for Pillow 10+ compatibility
                image.thumbnail((512, 512), Resampling.LANCZOS)

                status_msg = f"✅ Found and resized image for: **{query}**"
                add_to_history("image_search", query, status_msg, {"url": image_url})

                return image, status_msg
            except Exception as e:
                print(f"⚠️ Skipping invalid or unreachable image: {e}")
                continue

        status_msg = f"⚠️ No valid image results for: **{query}** after trying multiple options."
        add_to_history("image_search", query, status_msg)
        return None, status_msg
    except Exception as e:
        if "quotaExceeded" in str(e):
            set_rate_limit("image_search")
            return None, f"⚠️ Google Image Search: API quota exceeded. Try again after 12 hours."
        return None, f"Image Search failed: {e}"


# --- Public IP (Function remains the same) ---
def get_public_ip():
    limit_msg = check_rate_limit("public_ip")
    if limit_msg:
        return limit_msg
    try:
        resp = requests.get('https://api.ipify.org', timeout=10)
        resp.raise_for_status()
        ip = resp.text.strip()
        result = f"Your current Public IP Address is: **{ip}**"

        # Save to history
        add_to_history("public_ip", "Get Public IP", result, {"ip": ip})

        return result
    except Exception as e:
        return f"Error: {e}"


# ------------------ GRADIO UI ------------------
with gr.Blocks(title="Manjula AI Assistance", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Manjula AI Assistance")

    with gr.Row():
        chat_timer = gr.Textbox(label="Chat Timer", interactive=False, value=get_timer_text("text_qa", "Chat"))
        file_qa_timer = gr.Textbox(label="File Q&A Timer", interactive=False,
                                   value=get_timer_text("file_qa", "File Q&A"))
        image_timer = gr.Textbox(label="Image Timer", interactive=False, value=get_timer_text("image_gen", "Image Gen"))
        video_timer = gr.Textbox(label="Video Timer", interactive=False, value=get_timer_text("video_gen", "Video Gen"))
        image_search_timer = gr.Textbox(label="Image Search Timer", interactive=False,
                                        value=get_timer_text("image_search", "Image Search"))
        ip_timer = gr.Textbox(label="Public IP Timer", interactive=False,
                              value=get_timer_text("public_ip", "Public IP"))

    # Global History Button
    with gr.Row():
        show_history_btn = gr.Button("📋 Show Complete History", variant="primary", size="lg")

    session_id = gr.State(None)

    # History Modal (Hidden by default)
    with gr.Group(visible=False) as history_modal:
        with gr.Row():
            gr.Markdown("## 📚 Complete Activity History")

        # FINAL FIX: Using gr.Chatbot for guaranteed scrolling and visibility
        history_chatbot = gr.Chatbot(
            label="Activity Log (Scrollable)",
            height=500,
            show_copy_button=True  # Useful since this displays full queries/responses
        )

        with gr.Row():
            refresh_history_btn = gr.Button("🔄 Refresh", variant="secondary")
            clear_all_btn = gr.Button("🗑️ Clear All History", variant="stop")
            close_modal_btn = gr.Button("✖️ Close", variant="primary")
        history_status = gr.Textbox(label="Status", visible=True)  # Making status visible for confirmation

    with gr.Tab("💬 Chat with History & Mic"):
        with gr.Row():
            mic = gr.Microphone(label="Tap and speak", sources=["microphone"], type="filepath")
            mic_text = gr.Textbox(label="Speech to text", interactive=False)

        # Ensure chatbot has a defined height
        chatbot = gr.Chatbot(label="Conversation", height=500)

        user_input = gr.Textbox(placeholder="Enter your message here...", label="Type your message")

        with gr.Row():
            send_btn = gr.Button("Send", variant="primary")
            new_chat_btn = gr.Button("🆕 New Chat", variant="secondary")


        def transcribe_audio(audio_filepath):
            if audio_filepath is None:
                return ""
            # ASR logic not implemented, returning status message
            return f"Audio file received at: {os.path.basename(audio_filepath)}. ASR not implemented."


        # Chat interactions
        send_btn.click(
            query_model,
            inputs=[user_input, chatbot, session_id],
            outputs=[chatbot, user_input]
        )

        new_chat_btn.click(
            start_new_chat,
            inputs=None,
            outputs=[chatbot, session_id]
        )

        mic.change(transcribe_audio, inputs=mic, outputs=mic_text)

    with gr.Tab("📄 File Q&A"):
        file_upload = gr.File(label="Upload File (PDF, JPG, PNG supported)")
        file_question = gr.Textbox(label="Ask a question about the uploaded file")
        file_answer = gr.Textbox(label="Answer from File", lines=10)
        file_answer_btn = gr.Button("Get Answer")
        file_answer_btn.click(file_question_answer, inputs=[file_upload, file_question], outputs=file_answer)

    with gr.Tab("🎨 Image Generation"):
        img_prompt = gr.Textbox(label="Enter image prompt")
        gen_img_btn = gr.Button("Generate Image")
        img_output = gr.Image(label="Generated Image")
        img_status = gr.Textbox(label="Status")
        gen_img_btn.click(generate_image_for_gradio, inputs=img_prompt, outputs=[img_output, img_status])

    with gr.Tab("🎥 Video Generation"):
        video_prompt = gr.Textbox(label="Enter video prompt")
        gen_video_btn = gr.Button("Generate Video")
        video_status = gr.Textbox(label="Status")
        video_output = gr.Video(label="Generated Video")
        gen_video_btn.click(generate_video, inputs=video_prompt, outputs=[video_status, video_output])

    with gr.Tab("🖼️ Image Search"):
        img_search_query = gr.Textbox(label="Enter image search query")
        img_search_btn = gr.Button("Search Image")
        img_search_output = gr.Image(label="Web Search Result")
        img_search_status = gr.Textbox(label="Status")
        img_search_btn.click(google_image_search, inputs=img_search_query,
                             outputs=[img_search_output, img_search_status])

    with gr.Tab("🔍 Ask about an Image (Gemini)"):
        img_upload_tab = gr.Image(type="pil", label="Upload an Image")
        img_qa_prompt = gr.Textbox(lines=2, placeholder="Ask anything about the image...", label="Your Question")
        img_qa_btn = gr.Button("Ask about Image")
        img_qa_output = gr.Textbox(label="AI Response", lines=5)
        img_qa_btn.click(query_image_model, inputs=[img_upload_tab, img_qa_prompt], outputs=img_qa_output)

    with gr.Tab("🌐 Get Public IP"):
        ip_btn = gr.Button("Get Public IP")
        ip_output = gr.Textbox(label="Public IP Address")
        ip_btn.click(get_public_ip, inputs=None, outputs=ip_output)

    # History Modal Actions
    show_history_btn.click(
        show_history_modal,
        inputs=None,
        outputs=[history_modal, history_chatbot]
    )

    refresh_history_btn.click(
        lambda: format_history_for_chatbot(),
        inputs=None,
        outputs=history_chatbot
    )

    clear_all_btn.click(
        clear_all_history_action,
        inputs=None,
        outputs=[history_chatbot, history_status]
    )

    close_modal_btn.click(
        close_history_modal,
        inputs=None,
        outputs=history_modal
    )

if __name__ == "__main__":
    demo.launch()
