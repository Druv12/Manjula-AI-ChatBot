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
import logging
import hashlib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
import string
import socket

# MongoDB imports
try:
    from pymongo import MongoClient
    from pymongo.errors import DuplicateKeyError
    MONGODB_AVAILABLE = True
except ImportError:
    logging.warning("⚠️ pymongo not found. Install with: pip install pymongo")
    MONGODB_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

# --- MongoDB Setup ---
db = None
users_collection = None

if MONGODB_AVAILABLE:
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        client.server_info()
        db = client["manjula_ai"]
        users_collection = db["users"]
        users_collection.create_index("username", unique=True)
        logging.info("✅ MongoDB connected successfully")
    except Exception as e:
        logging.error(f"❌ MongoDB connection failed: {e}")
        MONGODB_AVAILABLE = False

# --- External Library Imports ---
try:
    from googleapiclient.discovery import build
except ImportError:
    logging.warning("'googleapiclient' library not found. Image search will fail.")
    build = None

REPLICATE_AVAILABLE = False
try:
    import replicate
    if REPLICATE_API_TOKEN:
        REPLICATE_AVAILABLE = True
        logging.info("✅ Replicate available")
    else:
        logging.info("ℹ️ No Replicate token - will use FREE services")
except ImportError:
    logging.info("ℹ️ Replicate library not found - will use FREE services")
    replicate = None

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    logging.warning("'speech_recognition' library not found. Install with: pip install SpeechRecognition")
    SPEECH_RECOGNITION_AVAILABLE = False

# Rate limit timers
api_reset_times = {
    "text_qa": None,
    "image_gen": None,
    "image_qa": None,
    "image_search": None,
    "video_gen": None,
    "public_ip": None,
    "file_qa": None,
}

# Guest session history (cleared on logout/new guest)
guest_session_history = {
    "chat": [],
    "file_qa": [],
    "ip_history": [],
    "image_gen": [],
    "video_gen": [],
    "image_search": [],
    "image_qa": [],
    "public_ip": []
}

current_session_id = 0
session_isolation_lock = {}
otp_storage = {}

# User session management
current_user = {"username": "Guest", "logged_in": False, "is_guest": True}
guest_chat_count = 0
GUEST_CHAT_LIMIT = 10


# --- User Authentication Functions ---
def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def clear_guest_history():
    """Clear all guest session history"""
    global guest_session_history
    guest_session_history = {
        "chat": [],
        "file_qa": [],
        "ip_history": [],
        "image_gen": [],
        "video_gen": [],
        "image_search": [],
        "image_qa": [],
        "public_ip": []
    }
    logging.info("🧹 Guest history cleared completely")


def start_as_guest():
    """Start using app as guest with limited features"""
    global guest_chat_count, current_session_id
    guest_chat_count = 0
    current_user["username"] = "Guest"
    current_user["logged_in"] = False
    current_user["is_guest"] = True

    current_session_id += 1
    clear_guest_history()

    return (
        "✅ **Welcome, Guest!** You can try the Chat feature with 10 free messages.\n\n"
        "**Register to unlock:**\n"
        "- 📄 File Q&A\n"
        "- 🎨 Image Generation\n"
        "- 🎥 Video Generation\n"
        "- 🖼️ Image Search\n"
        "- 🔍 Image Q&A\n"
        "- 📊 Usage Statistics\n"
        "- ♾️ Unlimited Chat",
        gr.update(visible=False),
        gr.update(visible=True),
        f"👤 **Guest Mode** | 💬 {guest_chat_count}/{GUEST_CHAT_LIMIT} chats used",
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        [],
        gr.update(visible=True,
                  value=f"⚠️ **Guest Mode:** You have {GUEST_CHAT_LIMIT}/{GUEST_CHAT_LIMIT} free chats remaining. Register to get unlimited access!"),
        [],
        current_session_id,
        None
    )


def generate_otp(length=6):
    """Generate a random 6-digit OTP"""
    return ''.join(random.choices(string.digits, k=length))


def send_otp_email(email, otp):
    """Send OTP to user's email address with enhanced debugging"""
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        return False, "❌ Email service not configured. Please contact administrator."

    # Debug logging (remove after testing)
    logging.info(f"🔍 Attempting to send OTP to: {email}")
    logging.info(f"🔍 SMTP_EMAIL: {SMTP_EMAIL}")
    logging.info(f"🔍 SMTP_SERVER: {SMTP_SERVER}")
    logging.info(f"🔍 SMTP_PORT: {SMTP_PORT}")
    logging.info(f"🔍 Password length: {len(SMTP_PASSWORD)} characters")
    logging.info(f"🔍 Password has spaces: {' ' in SMTP_PASSWORD}")

    try:
        msg = MIMEMultipart()
        msg['From'] = f"AI Assistance <{SMTP_EMAIL}>"
        msg['To'] = email
        msg['Subject'] = "🔐 Your AI Assistance Verification Code"

        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px; background-color: #f9f9f9; border-radius: 10px;">
                <h2 style="color: #4CAF50; text-align: center;">Welcome to AI Assistance!</h2>
                <p style="font-size: 16px;">Your verification code is:</p>
                <div style="text-align: center; margin: 30px 0;">
                    <h1 style="color: #4CAF50; font-size: 36px; letter-spacing: 8px; background-color: #fff; padding: 20px; border-radius: 8px; display: inline-block; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">{otp}</h1>
                </div>
                <p style="font-size: 14px; color: #666;">⏰ This code will expire in <strong>10 minutes</strong>.</p>
                <p style="font-size: 14px; color: #666;">If you didn't request this code, please ignore this email.</p>
                <hr style="border: none; border-top: 1px solid #ddd; margin: 30px 0;">
                <p style="color: #999; font-size: 12px; text-align: center;">
                    <strong>⚠️ This is an automated message from AI Assistance.</strong><br>
                    Please do not reply to this email.
                </p>
                <p style="text-align: center; margin-top: 20px;">
                    <span style="color: #4CAF50; font-weight: bold;">AI Assistance Team</span>
                </p>
            </div>
        </body>
        </html>
        """

        msg.attach(MIMEText(body, 'html'))

        logging.info("📧 Connecting to SMTP server...")
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30)

        logging.info("🔐 Starting TLS...")
        server.starttls()

        logging.info("🔑 Attempting login...")
        # Clean credentials - remove any whitespace or special characters
        clean_email = SMTP_EMAIL.strip()
        clean_password = SMTP_PASSWORD.strip().replace(' ', '').replace('\n', '').replace('\r', '')

        logging.info(f"🔍 Cleaned password length: {len(clean_password)} characters")

        server.login(clean_email, clean_password)

        logging.info("📨 Sending message...")
        server.send_message(msg)
        server.quit()

        logging.info(f"✅ OTP sent successfully to {email}")
        return True, "✅ OTP sent successfully! Check your email."

    except smtplib.SMTPAuthenticationError as e:
        error_code = str(e)
        logging.error(f"❌ SMTP Authentication failed: {error_code}")

        return False, (
            "❌ **Gmail Authentication Failed!**\n\n"
            "**This error means your credentials are wrong.**\n\n"
            "**Step-by-Step Fix:**\n\n"
            "1️⃣ **Enable 2-Factor Authentication:**\n"
            "   - Go to: https://myaccount.google.com/security\n"
            "   - Turn ON '2-Step Verification'\n\n"
            "2️⃣ **Generate App Password:**\n"
            "   - Go to: https://myaccount.google.com/apppasswords\n"
            "   - Select 'Mail' and 'Other (custom name)'\n"
            "   - Type 'AI Assistance' as the name\n"
            "   - Click 'Generate'\n"
            "   - **COPY the 16-character password** (e.g., abcdefghijklmnop)\n\n"
            "3️⃣ **Update your .env file:**\n"
            "   ```\n"
            f"   SMTP_EMAIL={SMTP_EMAIL}\n"
            "   SMTP_PASSWORD=abcdefghijklmnop  ← Paste HERE (no spaces!)\n"
            "   SMTP_SERVER=smtp.gmail.com\n"
            "   SMTP_PORT=587\n"
            "   ```\n\n"
            "4️⃣ **Restart the application completely**\n\n"
            "5️⃣ **Try again**\n\n"
            f"📋 Technical error: {error_code}\n\n"
            "⚠️ **Common mistakes:**\n"
            "- Using regular Gmail password (must use App Password)\n"
            "- App Password has spaces (remove ALL spaces)\n"
            "- Not restarting app after changing .env\n"
            "- Quotes around password in .env file (don't use quotes)"
        )

    except smtplib.SMTPException as e:
        logging.error(f"❌ SMTP error: {e}")
        return False, f"❌ Email server error: {str(e)}"

    except socket.timeout:
        logging.error("❌ Connection timeout")
        return False, "❌ Connection timeout. Check your internet connection or firewall settings."

    except Exception as e:
        logging.error(f"❌ Unexpected error sending OTP: {e}")
        return False, f"❌ Failed to send OTP: {str(e)}"


def request_otp(username, password, email, full_name):
    """Step 1: Generate and send OTP to email"""
    if not MONGODB_AVAILABLE:
        return "❌ Database not available. Please configure MongoDB.", gr.update(), gr.update()

    # Validate inputs
    if not username or not password or not email:
        return "❌ Username, password, and email are required!", gr.update(), gr.update()

    username = username.strip().lower()
    email = email.strip().lower()

    # Validate username length
    if len(username) < 3:
        return "❌ Username must be at least 3 characters long!", gr.update(), gr.update()

    # Validate password length
    if len(password) < 6:
        return "❌ Password must be at least 6 characters long!", gr.update(), gr.update()

    # Validate email format
    import re
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        return "❌ Please enter a valid email address!", gr.update(), gr.update()

    # Check if username already exists
    try:
        existing_user = users_collection.find_one({"username": username})
        if existing_user:
            return "❌ Username already exists. Please choose another username.", gr.update(), gr.update()

        # Check if email already exists
        existing_email = users_collection.find_one({"email": email})
        if existing_email:
            return "❌ Email already registered. Please use another email or login.", gr.update(), gr.update()
    except Exception as e:
        logging.error(f"Database check error: {e}")
        return f"❌ Database error: {str(e)}", gr.update(), gr.update()

    # Generate OTP
    otp = generate_otp()

    # Store OTP and user data temporarily
    otp_storage[email] = {
        "otp": otp,
        "timestamp": datetime.now(),
        "user_data": {
            "username": username,
            "password": hash_password(password),
            "email": email,
            "full_name": full_name.strip() if full_name else ""
        }
    }

    # Send OTP email
    success, message = send_otp_email(email, otp)

    if success:
        return (
            f"✅ **OTP sent to {email}!**\n\n"
            f"📧 Check your inbox (and spam folder) for a 6-digit code.\n\n"
            f"⏰ The OTP will expire in 10 minutes.\n\n"
            f"👇 Enter the OTP below to complete registration:",
            gr.update(visible=True),  # Show OTP input
            gr.update(visible=True)  # Show verify button
        )
    else:
        # Clean up if email fails
        if email in otp_storage:
            del otp_storage[email]
        return message, gr.update(), gr.update()

def verify_otp_and_register(email, otp_input):
    """Step 2: Verify OTP and complete registration"""
    if not MONGODB_AVAILABLE:
        return "❌ Database not available. Please configure MongoDB.", gr.update(), gr.update()

    email = email.strip().lower()
    otp_input = otp_input.strip()

    if email not in otp_storage:
        return "❌ No OTP request found for this email. Please request OTP first.", gr.update(), gr.update()

    otp_data = otp_storage[email]

    if datetime.now() - otp_data["timestamp"] > timedelta(minutes=10):
        del otp_storage[email]
        return "❌ OTP expired. Please request a new OTP.", gr.update(), gr.update()

    if otp_input != otp_data["otp"]:
        return "❌ Invalid OTP. Please check your email and try again.", gr.update(), gr.update()

    try:
        user_data = otp_data["user_data"]
        user_data.update({
            "created_at": datetime.now(),
            "last_login": None,
            "usage_count": {
                "chat": 0,
                "file_qa": 0,
                "image_gen": 0,
                "video_gen": 0,
                "image_search": 0,
                "image_qa": 0
            },
            "ip_history": [],
            "history": []
        })

        users_collection.insert_one(user_data)
        del otp_storage[email]

        logging.info(f"✅ User registered with verified email: {user_data['username']}")

        return (
            f"✅ **Registration successful!** Welcome, {user_data['username']}!\n\n"
            f"🎉 Your email has been verified!\n\n"
            f"You now have access to ALL features!\n\n"
            f"Please login now.",
            gr.update(visible=False),
            gr.update(visible=False)
        )

    except DuplicateKeyError:
        return "❌ Username already exists. Please choose another username.", gr.update(), gr.update()
    except Exception as e:
        logging.error(f"Registration error: {e}")
        return f"❌ Registration failed: {str(e)}", gr.update(), gr.update()


def login_user(username, password):
    """Login user - FIXED with complete session isolation"""
    global current_session_id, guest_chat_count

    if not MONGODB_AVAILABLE:
        current_session_id += 1
        clear_guest_history()
        guest_chat_count = 0

        current_user["username"] = username
        current_user["logged_in"] = True
        current_user["is_guest"] = False

        return (
            f"✅ **Welcome back, {username}!**\n\n🎉 You have full access to all features!",
            gr.update(visible=False),
            gr.update(visible=True),
            f"👤 **Logged in as:** {username}",
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            [],
            gr.update(visible=False),
            [],
            current_session_id,
            None
        )

    if not username or not password:
        return (
            "❌ Username and password are required!",
            gr.update(visible=True),
            gr.update(visible=False),
            "👤 **Not logged in**",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            [],
            gr.update(visible=False),
            [],
            current_session_id,
            None
        )

    try:
        user = users_collection.find_one({"username": username.lower().strip()})

        if user and user["password"] == hash_password(password):
            clear_guest_history()
            guest_chat_count = 0
            current_session_id += 1

            current_user["username"] = username.lower().strip()
            current_user["logged_in"] = True
            current_user["is_guest"] = False

            try:
                client_ip = requests.get("https://api.ipify.org", timeout=5).text.strip()
            except Exception:
                client_ip = "Unknown"

            users_collection.update_one(
                {"username": username.lower().strip()},
                {
                    "$addToSet": {"ip_history": client_ip},
                    "$set": {"last_login": datetime.now()}
                }
            )
            current_user["last_ip"] = client_ip

            logging.info(f"✅ User logged in: {username} | Session ID: {current_session_id} | Guest data WIPED")

            return (
                f"✅ **Welcome back, {username}!**\n\n🎉 You have full access to all features!",
                gr.update(visible=False),
                gr.update(visible=True),
                f"👤 **Logged in as:** {username}",
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                [],
                gr.update(visible=False),
                [],
                current_session_id,
                None
            )
        else:
            return (
                "❌ Invalid username or password!",
                gr.update(visible=True),
                gr.update(visible=False),
                "👤 **Not logged in**",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                [],
                gr.update(visible=True),
                [],
                current_session_id,
                None
            )

    except Exception as e:
        logging.error(f"Login error: {e}")
        return (
            f"❌ Login failed: {str(e)}",
            gr.update(visible=True),
            gr.update(visible=False),
            "👤 **Not logged in**",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            [],
            gr.update(visible=True),
            [],
            current_session_id,
            None
        )


def logout_user():
    """Logout user and clear ALL session data"""
    global current_session_id, guest_chat_count

    username = current_user.get("username", "Unknown")

    current_user["username"] = None
    current_user["logged_in"] = False
    current_user["is_guest"] = False

    guest_chat_count = 0
    clear_guest_history()
    current_session_id += 1

    logging.info(f"✅ User logged out: {username} | Session ID: {current_session_id} | ALL SESSION DATA WIPED")

    return (
        "✅ Logged out successfully! Please login or register to continue.",
        gr.update(visible=True),
        gr.update(visible=False),
        "👤 **Not logged in**",
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        [],
        gr.update(visible=False),
        [],
        current_session_id,
        None
    )


def format_history_for_chatbot():
    """Fetch ONLY current user's history - COMPLETELY ISOLATED"""
    chatbot_history = []

    if current_user.get("is_guest", False):
        logging.info("📋 Loading GUEST session history (temporary)")

        entries = guest_session_history.get("chat", [])

        if entries:
            chatbot_history.append((
                f"--- **💬 Guest Session History** ({len(entries)} messages) ---",
                "**⚠️ TEMPORARY DATA - Will be deleted on logout**"
            ))

            for entry in entries:
                timestamp = entry.get("timestamp", "")
                full_query = entry.get("full_query", "No Query")
                full_response = entry.get("full_response", "No Response")

                chatbot_history.append((
                    f"**{timestamp}** | Q: {full_query}",
                    f"A: {full_response}"
                ))
        else:
            chatbot_history.append((
                None,
                "📭 **No guest history yet.**\n\nStart chatting to see your activity here!\n\n🔒 **Note:** Guest history is temporary. Register to save permanently."
            ))

        return chatbot_history

    if not current_user.get("logged_in", False) or not current_user.get("username"):
        logging.warning("⚠️ Attempted to load history without login")
        return [(None, "📭 **Please login to view your history.**")]

    if not MONGODB_AVAILABLE:
        return [(None, "❌ **Database not available.**")]

    try:
        username = current_user["username"]
        logging.info(f"📋 Loading history for user: {username}")

        user_doc = users_collection.find_one(
            {"username": username},
            {"history": 1, "_id": 0}
        )

        if not user_doc or not user_doc.get("history"):
            logging.info(f"📭 User {username} has no history")
            return [(
                None,
                f"📭 **No history for: {username}**\n\nStart using features to see your activity here!"
            )]

        total_items = len(user_doc["history"])
        chatbot_history.append((
            f"--- **📚 {username.upper()}'s Complete History** ({total_items} items) ---",
            "**All YOUR interactions. This data belongs ONLY to you.**"
        ))

        feature_icons = {
            "chat": "💬", "file_qa": "📄", "image_gen": "🎨",
            "video_gen": "🎥", "image_search": "🖼️", "image_qa": "🔍", "public_ip": "🌐"
        }

        feature_groups = {}
        for entry in user_doc["history"]:
            feature = entry.get("feature", "chat")
            feature_groups.setdefault(feature, []).append(entry)

        for feature, entries in feature_groups.items():
            icon = feature_icons.get(feature, "📋")
            feature_name = feature.replace("_", " ").title()

            chatbot_history.append((
                f"--- **{icon} {feature_name}** ({len(entries)} items) ---",
                ""
            ))

            for entry in entries:
                timestamp = entry.get("timestamp")
                if hasattr(timestamp, "strftime"):
                    timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")

                text = entry.get("text", "No Query")[:200]
                reply = entry.get("reply", "No Response")[:300]

                chatbot_history.append((
                    f"**{timestamp}** | Q: {text}",
                    f"A: {reply}"
                ))

        logging.info(f"✅ Loaded {total_items} history items for {username}")
        return chatbot_history

    except Exception as e:
        logging.error(f"❌ Error loading history for {current_user.get('username', 'unknown')}: {e}")
        return [(None, f"❌ **Error loading history:** {e}")]


def check_feature_access(feature_name):
    """Check if user can access a feature"""
    if current_user["logged_in"]:
        return None

    if current_user["is_guest"] and feature_name != "chat":
        return (
            f"🔒 **Feature Locked: {feature_name.replace('_', ' ').title()}**\n\n"
            f"This feature is only available to registered users.\n\n"
            f"**Register now to unlock:**\n"
            f"- 📄 File Q&A - Upload and analyze any document\n"
            f"- 🎨 Image Generation - Create AI art from text\n"
            f"- 🎥 Video Generation - Generate AI videos\n"
            f"- 🖼️ Image Search - Search Google Images\n"
            f"- 🔍 Image Q&A - Ask questions about images\n"
            f"- 📊 Usage Statistics - Track your activity\n"
            f"- ♾️ Unlimited Chat - No message limits\n\n"
            f"👉 **Click 'Logout' and then 'Register' to create a free account!**"
        )

    return None


def check_guest_chat_limit():
    """Check if guest has exceeded chat limit"""
    global guest_chat_count

    if current_user["is_guest"]:
        if guest_chat_count >= GUEST_CHAT_LIMIT:
            return (
                f"🚫 **Guest Chat Limit Reached ({GUEST_CHAT_LIMIT}/{GUEST_CHAT_LIMIT})**\n\n"
                f"You've used all your free guest messages!\n\n"
                f"**Register now to get:**\n"
                f"- ♾️ **Unlimited Chat** - Chat as much as you want\n"
                f"- 📄 **File Q&A** - Upload and analyze documents\n"
                f"- 🎨 **Image Generation** - Create AI art\n"
                f"- 🎥 **Video Generation** - Generate AI videos\n"
                f"- 🖼️ **Image Search** - Search Google Images\n"
                f"- 🔍 **Image Q&A** - Ask questions about images\n"
                f"- 📊 **Usage Statistics** - Track all your activity\n\n"
                f"Registration is **100% FREE** and takes less than 1 minute!\n\n"
                f"👉 **Click 'Logout' and then 'Register' to unlock all features!**"
            )

    return None


def increment_usage(feature):
    """Increment usage count for a feature"""
    if MONGODB_AVAILABLE and current_user["logged_in"]:
        try:
            users_collection.update_one(
                {"username": current_user["username"]},
                {"$inc": {f"usage_count.{feature}": 1}}
            )
        except Exception as e:
            logging.error(f"Failed to increment usage: {e}")


def get_user_stats():
    """Get user statistics"""
    if not MONGODB_AVAILABLE or not current_user["logged_in"]:
        return "❌ Please login to view statistics."

    try:
        user = users_collection.find_one({"username": current_user["username"]})
        if user:
            stats = f"""
## 📊 Your Usage Statistics

**Username:** {user['username']}
**Email:** {user.get('email', 'N/A')}
**Full Name:** {user.get('full_name', 'N/A')}
**Member Since:** {user['created_at'].strftime('%Y-%m-%d %H:%M:%S')}
**Last Login:** {user['last_login'].strftime('%Y-%m-%d %H:%M:%S') if user.get('last_login') else 'N/A'}

### Feature Usage:
- 💬 Chat: {user['usage_count']['chat']} times
- 📄 File Q&A: {user['usage_count']['file_qa']} times
- 🎨 Image Generation: {user['usage_count']['image_gen']} times
- 🎥 Video Generation: {user['usage_count']['video_gen']} times
- 🖼️ Image Search: {user['usage_count']['image_search']} times
- 🔍 Image Q&A: {user['usage_count']['image_qa']} times

**Total Usage:** {sum(user['usage_count'].values())} actions
"""
            return stats
    except Exception as e:
        logging.error(f"Failed to get user stats: {e}")
        return f"❌ Failed to load statistics: {str(e)}"


GEMINI_CLIENT = None
if GEMINI_API_KEY:
    try:
        GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        logging.error(f"Failed to initialize Gemini Client: {e}")
        GEMINI_CLIENT = None


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


def add_to_guest_history(category, user_query, response, metadata=None):
    """Add guest interaction to session-only history (not saved to DB)"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_query_str = str(user_query).strip() if user_query is not None else ""
    response_str = str(response).strip() if response is not None else ""

    entry = {
        "timestamp": timestamp,
        "query": user_query_str[:100],
        "response": response_str[:200],
        "full_query": user_query_str,
        "full_response": response_str,
        "metadata": metadata or {}
    }
    guest_session_history[category].insert(0, entry)

    if len(guest_session_history[category]) > 50:
        guest_session_history[category] = guest_session_history[category][:50]


def save_interaction_to_db(feature_name, user_query, ai_response, metadata=None):
    """Save interaction to MongoDB for logged-in users only"""

    if current_user["is_guest"]:
        guest_session_history.setdefault("chat", [])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        guest_session_history["chat"].append({
            "timestamp": timestamp,
            "full_query": user_query,
            "full_response": ai_response,
            "feature": feature_name,
            "metadata": metadata or {}
        })
        return

    if not MONGODB_AVAILABLE:
        return

    if not current_user["logged_in"] or not current_user["username"]:
        return

    try:
        history_entry = {
            "timestamp": datetime.now(),
            "feature": feature_name,
            "text": str(user_query)[:500] if user_query else "No Query",
            "reply": str(ai_response)[:1000] if ai_response else "No Response",
            "metadata": metadata or {}
        }

        users_collection.update_one(
            {"username": current_user["username"]},
            {"$push": {"history": history_entry}},
        )

        logging.info(f"✅ Saved {feature_name} interaction for user: {current_user['username']}")

    except Exception as e:
        logging.error(f"Failed to save interaction to MongoDB: {e}")


def show_history_modal():
    """Open history modal and populate the chatbot"""
    return gr.update(visible=True), format_history_for_chatbot()


def close_history_modal():
    """Close history modal"""
    return gr.update(visible=False)


def clear_all_history_action():
    """Clear history from UI only - DATABASE REMAINS UNTOUCHED"""
    if current_user["is_guest"]:
        clear_guest_history()
        return [], "✅ Guest session history cleared from UI!"

    elif current_user["logged_in"] and MONGODB_AVAILABLE:
        return [], "✅ History cleared from UI! (Database unchanged - click 'Refresh' to reload)"

    return [], "❌ Unable to clear history."


def transcribe_audio(audio_filepath):
    """Transcribe audio file to text using Google Speech Recognition"""
    if not SPEECH_RECOGNITION_AVAILABLE:
        return "❌ Speech recognition library not installed. Please run: pip install SpeechRecognition"

    if audio_filepath is None:
        return ""

    try:
        recognizer = sr.Recognizer()

        with sr.AudioFile(audio_filepath) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data)
        return text

    except sr.UnknownValueError:
        return "❌ Could not understand audio. Please speak clearly."
    except sr.RequestError as e:
        return f"❌ Speech recognition service error: {e}"
    except Exception as e:
        return f"❌ Error processing audio: {e}"


def check_guest_feature_access(feature_name):
    """Check if guest user can access a specific feature"""
    if current_user["is_guest"]:
        return f"🚫 **Guest users can only use Text Chat.** Please login or register to access {feature_name}."
    return None


def query_model(prompt, history, session_id_state):
    """Chat function with COMPLETE session isolation"""
    global current_session_id

    # ✅ CRITICAL: Validate session ID matches current session
    if session_id_state != current_session_id:
        logging.warning(f"⚠️ Session mismatch detected! Clearing stale history. "
                        f"Expected: {current_session_id}, Got: {session_id_state}")
        history = []
        session_id_state = current_session_id

    logging.info(f"📥 query_model | User: {current_user.get('username', 'Unknown')} | "
                 f"Guest: {current_user.get('is_guest')} | History len: {len(history)} | "
                 f"Session: {session_id_state} | Current: {current_session_id}")

    if not prompt or not prompt.strip():
        return history, "", session_id_state

    limit_check = check_guest_chat_limit()
    if limit_check:
        history.append((prompt, limit_check))
        return history, "", session_id_state

    limit_msg = check_rate_limit("text_qa")
    if limit_msg:
        history.append((prompt, limit_msg))
        return history, "", session_id_state

    global guest_chat_count
    if current_user["is_guest"]:
        guest_chat_count += 1
    else:
        increment_usage("chat")

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
                answer = f"⚠️ Text Q&A: Daily limit reached. Try again after 12 hours."
            else:
                answer = f"⚠️ Gemini failed: {e}"
            logging.error(f"Gemini API Error: {e}")

    if (answer is None or "Gemini failed" in answer or llm_name == "N/A") and OPENAI_KEY:
        llm_name = "OpenAI"
        try:
            client = openai.OpenAI(api_key=OPENAI_KEY)
            openai_formatted_messages = []
            for msg in llm_messages:
                role = "assistant" if msg["role"] == "model" else msg["role"]
                openai_formatted_messages.append({"role": role, "content": msg["content"]})

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=openai_formatted_messages
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"⚠️ OpenAI failed: {e}"
            logging.error(f"OpenAI API Error: {e}")

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if (answer is None or "failed" in answer.lower()) and GROQ_API_KEY:
        llm_name = "Groq"
        try:
            groq_messages = []
            for msg in llm_messages:
                role = "assistant" if msg["role"] == "model" else msg["role"]
                groq_messages.append({"role": role, "content": msg["content"]})

            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": groq_messages
                },
                timeout=30
            )
            if response.status_code == 200:
                answer = response.json()["choices"][0]["message"]["content"]
            else:
                answer = f"⚠️ Groq failed: {response.text}"
                logging.error(f"Groq API Error: {response.status_code} - {response.text}")
        except Exception as e:
            answer = f"⚠️ Groq failed: {e}"
            logging.error(f"Groq API Error: {e}")

    if answer is None or answer.startswith("Error: No LLM client"):
        answer = "Error: No LLM client configured (Missing GEMINI_API_KEY or OPENAI_API_KEY)."

    history.append((prompt, answer))

    if current_user["is_guest"]:
        add_to_guest_history("chat", prompt, answer, {"model": llm_name})
        logging.info(f"💬 Guest chat saved to SESSION ONLY (not in DB) - Model: {llm_name}")
    else:
        save_interaction_to_db("chat", prompt, answer, {"model": llm_name})
        logging.info(f"💬 User '{current_user['username']}' chat saved to DB - Model: {llm_name}")

    return history, "", session_id_state


def process_audio_and_chat(audio_filepath, history, session_id_state):
    """Process audio input, transcribe it, and get AI response - WITH SESSION VALIDATION"""
    global current_session_id

    # ✅ CRITICAL: Validate session BEFORE processing
    if session_id_state != current_session_id:
        logging.warning(f"⚠️ Voice input detected stale session. FORCING CLEAR. "
                        f"Expected: {current_session_id}, Got: {session_id_state}")
        history = []
        session_id_state = current_session_id

    # ✅ ADDITIONAL CHECK: If history exists but session changed, clear it
    if history and len(history) > 0:
        if session_id_state != current_session_id:
            logging.warning(f"⚠️ Clearing {len(history)} stale messages from previous user")
            history = []
            session_id_state = current_session_id

    if audio_filepath is None:
        return history, "", update_guest_warning(), session_id_state

    transcribed_text = transcribe_audio(audio_filepath)

    if transcribed_text.startswith("❌"):
        return history + [(None, transcribed_text)], "", update_guest_warning(), session_id_state

    result_history, result_input, result_session = query_model(transcribed_text, history, session_id_state)
    warning = update_guest_warning()

    logging.info(f"🎤 Voice chat processed | User: {current_user.get('username')} | "
                 f"Session: {result_session} | History items: {len(result_history)}")

    return result_history, result_input, warning, result_session


def start_new_chat():
    """Start a fresh chat session - clears UI only, not database"""
    global current_session_id
    current_session_id += 1

    logging.info(f"🆕 New chat started | User: {current_user.get('username', 'Guest')} | "
                 f"New Session ID: {current_session_id}")

    return [], current_session_id


def update_guest_warning():
    """Update guest warning message"""
    global guest_chat_count
    if current_user["is_guest"]:
        remaining = GUEST_CHAT_LIMIT - guest_chat_count
        if remaining > 0:
            return gr.update(
                value=f"⚠️ **Guest Mode:** You have {remaining}/{GUEST_CHAT_LIMIT} free chats remaining. Register to get unlimited access!",
                visible=True)
        else:
            return gr.update(value=f"🚫 **Guest limit reached!** Please register to continue chatting.",
                             visible=True)
    return gr.update(visible=False)


def query_and_update_warning(prompt, history, session_id_state):
    """Query model and update warning - with session validation"""
    result_history, result_input, result_session = query_model(prompt, history, session_id_state)
    warning = update_guest_warning()
    return result_history, result_input, warning, result_session


def get_public_ip():
    """Get public IP with history saving"""
    limit_msg = check_rate_limit("public_ip")
    if limit_msg:
        return limit_msg
    try:
        resp = requests.get('https://api.ipify.org', timeout=10)
        resp.raise_for_status()
        ip = resp.text.strip()
        result = f"Your current Public IP Address is: **{ip}**"

        if current_user["is_guest"]:
            add_to_guest_history("public_ip", "Get Public IP", result, {"ip": ip})
        else:
            save_interaction_to_db("public_ip", "Get Public IP", result, {"ip": ip})

        return result
    except Exception as e:
        error_msg = f"Error: {e}"

        if not current_user["is_guest"]:
            save_interaction_to_db("public_ip", "Get Public IP", error_msg)

        return error_msg


# --- File Content Extraction WITH TIMER ---
def extract_file_content_gemini(file, prompt):
    """Extract file content with progress updates"""
    guest_check = check_guest_feature_access("File Q&A")
    if guest_check:
        yield "🔒 Access Denied", guest_check
        return

    if not GEMINI_CLIENT:
        yield "⏱️ 0s", "Error: Gemini API Key missing."
        return

    if not file:
        yield "⏱️ 0s", "Error: No file uploaded."
        return

    uploaded_file = None
    try:
        start_time = time.time()

        yield "⏱️ 1s", "⏳ **Step 1/3:** Uploading file to Gemini..."
        file_path = file.name
        ext = os.path.splitext(file_path)[-1].lower()

        if ext in ['.docx', '.txt']:
            yield "⏱️ 2s", "Error: Unsupported file type for multimodal extraction (DOCX/TXT require dedicated parsers)."
            return

        uploaded_file = GEMINI_CLIENT.files.upload(file=file_path)
        elapsed = int(time.time() - start_time)

        yield f"⏱️ {elapsed}s", "⏳ **Step 2/3:** Processing file (waiting for API)..."
        time.sleep(2)

        elapsed = int(time.time() - start_time)
        yield f"⏱️ {elapsed}s", "⏳ **Step 3/3:** Extracting content with AI..."

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

        elapsed = int(time.time() - start_time)

        if response and response.text:
            yield f"⏱️ {elapsed}s ✅", f"**✅ Extraction Complete! (took {elapsed}s)**\n\n{response.text}"
        else:
            yield f"⏱️ {elapsed}s", "Empty extraction result from Gemini."

    except Exception as e:
        elapsed = int(time.time() - start_time)
        error_msg = str(e)
        logging.error(f"File extraction error: {error_msg}")
        yield f"⏱️ {elapsed}s ❌", f"Extraction failed: {error_msg}"

    finally:
        if uploaded_file and hasattr(uploaded_file, 'name') and uploaded_file.name:
            try:
                time.sleep(1)
                GEMINI_CLIENT.files.delete(name=uploaded_file.name)
            except Exception as cleanup_error:
                logging.warning(f"File cleanup error (non-critical): {cleanup_error}")
                pass


def answer_question_from_content(file_content, user_question):
    """Use LLM to answer user's question based on extracted file content"""
    if not GEMINI_CLIENT and not OPENAI_KEY:
        return f"**Extracted Content:**\n\n{file_content}\n\n---\n\n⚠️ No LLM available to answer your question. Please configure GEMINI_API_KEY or OPENAI_API_KEY."

    max_content_length = 30000
    if len(file_content) > max_content_length:
        file_content = file_content[:max_content_length] + "\n\n[Content truncated due to length...]"

    system_prompt = f"""You are a highly capable AI assistant. A user has uploaded a file and wants your help.

**File Content:**
{file_content}

**User's Request:**
{user_question}

Instructions:
- The user can ask you to do ANYTHING with this file content - be completely flexible and helpful.
- Understand what the user is asking for and provide exactly that.
- If you're unsure what they want, ask for clarification while still providing your best interpretation.

Now, based on the user's request above, provide the most helpful and appropriate response."""

    answer = None
    llm_name = "N/A"

    if GEMINI_CLIENT:
        llm_name = "Gemini"
        try:
            response = GEMINI_CLIENT.models.generate_content(
                model="gemini-2.5-flash",
                contents=system_prompt
            )
            if response and response.text:
                answer = response.text
        except Exception as e:
            logging.error(f"Gemini API Error in File Q&A: {e}")
            answer = None

    if (answer is None or "failed" in str(answer).lower()) and OPENAI_KEY:
        llm_name = "OpenAI"
        try:
            client = openai.OpenAI(api_key=OPENAI_KEY)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": "You are a helpful AI assistant that can do anything the user requests with uploaded file content."},
                    {"role": "user", "content": system_prompt}
                ],
                timeout=60
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"OpenAI API Error in File Q&A: {e}")
            answer = None

    if answer is None or not answer.strip():
        return f"**Extracted Content:**\n\n{file_content}\n\n---\n\n⚠️ Unable to process your question with AI. Here's the extracted content for your review."

    return f"**Response to your request:**\n\n{answer}\n\n---\n\n**Source:** Based on content from uploaded file (processed by {llm_name})"


def file_question_answer(file, question):
    """FIXED: File Q&A with timer and history saving"""
    access_check = check_feature_access("file_qa")
    if access_check:
        yield "⏱️ 0s", access_check
        return

    limit_msg = check_rate_limit("file_qa")
    if limit_msg:
        yield "⏱️ 0s", limit_msg
        return

    if not file or not question:
        yield "⏱️ 0s", "Please upload a file and enter a question."
        return

    increment_usage("file_qa")

    extraction_result = None
    for timer, content in extract_file_content_gemini(file, question):
        yield timer, content
        extraction_result = content

    if extraction_result and (extraction_result.lower().startswith("error:") or extraction_result.lower().startswith(
            "extraction failed:")):
        if current_user["is_guest"]:
            add_to_guest_history("file_qa", question, extraction_result, {"filename": file.name if file else "unknown"})
        else:
            save_interaction_to_db("file_qa", question, extraction_result,
                                   {"filename": file.name if file else "unknown"})
        return

    if extraction_result and "✅ Extraction Complete!" in extraction_result:
        content_parts = extraction_result.split("\n\n", 1)
        if len(content_parts) > 1:
            extracted_content = content_parts[1]
        else:
            extracted_content = extraction_result

        yield "⏱️ Processing...", "⏳ **Generating AI response based on extracted content...**"
        result = answer_question_from_content(extracted_content, question)

        if current_user["is_guest"]:
            add_to_guest_history("file_qa", question, result, {"filename": file.name if file else "unknown"})
        else:
            save_interaction_to_db("file_qa", question, result, {"filename": file.name if file else "unknown"})

        yield "⏱️ Complete ✅", result


def process_audio_for_file_qa(audio_filepath, file):
    """Process audio input for file Q&A"""
    guest_check = check_guest_feature_access("File Q&A")
    if guest_check:
        yield "⏱️ 0s", guest_check
        return

    if audio_filepath is None:
        yield "⏱️ 0s", ""
        return

    transcribed_text = transcribe_audio(audio_filepath)

    if transcribed_text.startswith("❌"):
        yield "⏱️ 0s", transcribed_text
        return

    for timer, result in file_question_answer(file, transcribed_text):
        yield timer, result


# --- Image Generation Functions ---
def generate_image_for_gradio(prompt: str):
    """ENHANCED: Image generation with multiple FREE fallbacks"""
    guest_check = check_guest_feature_access("Image Generation")
    if guest_check:
        return None, guest_check

    access_check = check_feature_access("image_gen")
    if access_check:
        return None, access_check

    limit_msg = check_rate_limit("image_gen")
    if limit_msg:
        return None, f"{limit_msg}\n\n**FREE FALLBACK:** Use an external generator like Bing Image Creator."

    if not prompt:
        return None, "Please enter a prompt."

    increment_usage("image_gen")

    # ===== METHOD 1: Pollinations.ai (Primary FREE service) =====
    logging.info("🎨 Trying Pollinations.ai (FREE)...")
    try:
        encoded_prompt = quote(prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&model=flux&nologo=true&enhance=true"
        response = requests.get(url, timeout=45)
        if response.status_code == 200 and len(response.content) > 5000:
            try:
                image = Image.open(io.BytesIO(response.content))
                status_msg = "✅ Image generated successfully with Pollinations.ai (FREE)!"
                logging.info("✅ Pollinations.ai success")

                if current_user["is_guest"]:
                    add_to_guest_history("image_gen", prompt, status_msg, {"service": "Pollinations.ai"})
                else:
                    save_interaction_to_db("image_gen", prompt, status_msg,
                                           {"service": "Pollinations.ai", "size": f"{len(response.content)} bytes"})

                return image, status_msg
            except Exception as e:
                logging.error(f"Pollinations.ai image processing error: {e}")
        else:
            logging.warning(f"Pollinations.ai failed: status={response.status_code}, size={len(response.content)}")
    except Exception as e:
        logging.error(f"Pollinations.ai error: {e}")

    # ===== METHOD 2: Hugging Face Inference API (FREE alternative) =====
    if HF_API_KEY:
        logging.info("🎨 Trying Hugging Face Flux (FREE)...")
        try:
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            hf_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
            response = requests.post(hf_url, headers=headers, json={"inputs": prompt}, timeout=60)

            if response.status_code == 200 and len(response.content) > 5000:
                try:
                    image = Image.open(io.BytesIO(response.content))
                    status_msg = "✅ Image generated successfully with Hugging Face Flux (FREE)!"
                    logging.info("✅ Hugging Face success")

                    if current_user["is_guest"]:
                        add_to_guest_history("image_gen", prompt, status_msg, {"service": "Hugging Face"})
                    else:
                        save_interaction_to_db("image_gen", prompt, status_msg,
                                               {"service": "Hugging Face", "size": f"{len(response.content)} bytes"})

                    return image, status_msg
                except Exception as e:
                    logging.error(f"HF image processing error: {e}")
            elif response.status_code == 503:
                logging.warning("⏳ Hugging Face model loading... (might work in 20-30 seconds)")
            else:
                logging.warning(f"Hugging Face failed: status={response.status_code}")
        except Exception as e:
            logging.error(f"Hugging Face error: {e}")

    # ===== METHOD 3: Segmind (FREE tier) =====
    logging.info("🎨 Trying Segmind (FREE)...")
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
        response = requests.post(url, json=data, timeout=45)
        if response.status_code == 200 and len(response.content) > 5000:
            try:
                image = Image.open(io.BytesIO(response.content))
                status_msg = "✅ Image generated successfully with Segmind (FREE)!"
                logging.info("✅ Segmind success")

                if current_user["is_guest"]:
                    add_to_guest_history("image_gen", prompt, status_msg, {"service": "Segmind"})
                else:
                    save_interaction_to_db("image_gen", prompt, status_msg,
                                           {"service": "Segmind", "size": f"{len(response.content)} bytes"})

                return image, status_msg
            except Exception as e:
                logging.error(f"Segmind image processing error: {e}")
        else:
            logging.warning(f"Segmind failed: status={response.status_code}")
    except Exception as e:
        logging.error(f"Segmind error: {e}")

    # ===== METHOD 4: Stability AI (Premium - if API key available) =====
    if STABILITY_API_KEY:
        logging.info("🎨 Trying Stability AI (Premium)...")
        try:
            session = create_session_with_retries()
            headers = {"Authorization": f"Bearer {STABILITY_API_KEY}", "Accept": "image/*"}
            data = {"prompt": prompt, "output_format": "png", "aspect_ratio": "1:1", "seed": 0}
            response = session.post(
                "https://api.stability.ai/v2beta/stable-image/generate/ultra",
                headers=headers, files={"none": ''}, data=data, timeout=60
            )
            if response.status_code == 200:
                try:
                    image = Image.open(io.BytesIO(response.content))
                    status_msg = "✅ Image generated successfully with Stability AI (Premium)!"
                    logging.info("✅ Stability AI success")

                    if current_user["is_guest"]:
                        add_to_guest_history("image_gen", prompt, status_msg, {"service": "Stability AI"})
                    else:
                        save_interaction_to_db("image_gen", prompt, status_msg,
                                               {"service": "Stability AI", "size": f"{len(response.content)} bytes"})

                    return image, status_msg
                except Exception as e:
                    logging.error(f"Stability AI image processing error: {e}")
            else:
                logging.warning(f"Stability AI failed: status={response.status_code}")
        except Exception as e:
            logging.error(f"Stability AI error: {e}")

    # ===== ALL SERVICES FAILED =====
    error_msg = (
        "❌ **All image generation services are currently unavailable.**\n\n"
        "**What happened:**\n"
        "- 🚫 Pollinations.ai - Failed or rate limited\n"
        "- 🚫 Hugging Face - Model unavailable or loading\n"
        "- 🚫 Segmind - Service unavailable\n"
        f"- 🚫 Stability AI - {'Not configured' if not STABILITY_API_KEY else 'Failed'}\n\n"
        "**FREE Alternatives (100% Working):**\n\n"
        "1. **Bing Image Creator** (Microsoft, FREE):\n"
        "   - 🔗 https://www.bing.com/images/create\n"
        "   - Uses DALL-E 3, very high quality\n"
        "   - No sign-up needed\n\n"
        "2. **Leonardo.ai** (FREE tier):\n"
        "   - 🔗 https://app.leonardo.ai\n"
        "   - 150 FREE credits daily\n\n"
        "3. **Craiyon** (FREE, no limits):\n"
        "   - 🔗 https://www.craiyon.com\n"
        "   - Instant generation\n\n"
        "4. **Ideogram** (FREE):\n"
        "   - 🔗 https://ideogram.ai\n"
        "   - Great for text in images\n\n"
        "**Try again in 10-15 minutes** - services may recover."
    )

    logging.error("❌ All image generation services failed")

    if not current_user["is_guest"]:
        save_interaction_to_db("image_gen", prompt, "Failed - all services unavailable", {"error": "all_failed"})

    return None, error_msg


def process_audio_for_image_gen(audio_filepath):
    """Process audio input for image generation"""
    guest_check = check_guest_feature_access("Image Generation")
    if guest_check:
        return None, guest_check

    if audio_filepath is None:
        return None, ""

    transcribed_text = transcribe_audio(audio_filepath)

    if transcribed_text.startswith("❌"):
        return None, transcribed_text

    return generate_image_for_gradio(transcribed_text)


# --- Image QA ---
def query_image_model(image, prompt):
    """FIXED: Image Q&A with history saving"""
    guest_check = check_guest_feature_access("Image Q&A")
    if guest_check:
        return guest_check

    access_check = check_feature_access("image_qa")
    if access_check:
        return access_check

    limit_msg = check_rate_limit("image_qa")
    if limit_msg:
        return limit_msg

    if image is None:
        return "Error: Please upload an image first."

    if not GEMINI_CLIENT:
        return "Error: Gemini API Key missing."

    increment_usage("image_qa")

    try:
        resized_image = image.copy()
        resized_image.thumbnail((512, 512), Resampling.LANCZOS)

        img_byte_arr = io.BytesIO()
        resized_image.save(img_byte_arr, format='JPEG')
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

        if current_user["is_guest"]:
            add_to_guest_history("image_qa", prompt, result)
        else:
            save_interaction_to_db("image_qa", prompt, result)

        return result
    except Exception as e:
        error_msg = f"Image-based Q&A failed: {e}"
        logging.error(error_msg)

        if not current_user["is_guest"]:
            save_interaction_to_db("image_qa", prompt, error_msg)

        return error_msg


def process_audio_for_image_qa(audio_filepath, image):
    """Process audio input for image Q&A"""
    guest_check = check_guest_feature_access("Image Q&A")
    if guest_check:
        return guest_check

    if audio_filepath is None:
        return ""

    transcribed_text = transcribe_audio(audio_filepath)

    if transcribed_text.startswith("❌"):
        return transcribed_text

    return query_image_model(image, transcribed_text)


# --- Video Generation ---
def generate_video(prompt):
    """Generate video - returns file paths properly"""
    guest_check = check_guest_feature_access("Video Generation")
    if guest_check:
        yield guest_check, None
        return

    access_check = check_feature_access("video_gen")
    if access_check:
        yield access_check, None
        return

    limit_msg = check_rate_limit("video_gen")
    if limit_msg:
        yield limit_msg, None
        return

    if not prompt:
        yield "Please enter a prompt.", None
        return

    increment_usage("video_gen")

    if len(prompt) > 100:
        yield f"⚠️ **Long prompt detected!** ({len(prompt)} characters)\n\nFree services work better with shorter prompts (under 50 characters).\n\nExample: Instead of your current prompt, try:\n*\"massive shark attacks sailor boat in storm\"*\n\nContinuing with your prompt...", None
        time.sleep(2)

    if REPLICATE_AVAILABLE and REPLICATE_API_TOKEN:
        yield f"⏳ Trying **Replicate** (Premium Quality)...\n\nPrompt: *{prompt}*\n\nThis may take 1-2 minutes...", None

        try:
            output = replicate.run(
                "deforum/deforum_stable_diffusion",
                input={
                    "prompt": prompt,
                    "max_frames": 50,
                }
            )

            video_url = None
            if isinstance(output, str):
                video_url = output
            elif isinstance(output, list) and len(output) > 0:
                video_url = output[0] if isinstance(output[0], str) else None
            elif hasattr(output, 'url'):
                video_url = output.url

            if video_url and isinstance(video_url, str) and video_url.startswith('http'):
                try:
                    import tempfile
                    video_response = requests.get(video_url, timeout=60)
                    if video_response.status_code == 200:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                            tmp.write(video_response.content)
                            video_path = tmp.name

                        success_msg = f"✅ **Video generated with Replicate (Premium)!**\n\n**Prompt:** *{prompt}*"
                        logging.info(f"✅ Replicate success: {video_url}")

                        if current_user["is_guest"]:
                            add_to_guest_history("video_gen", prompt, success_msg,
                                                 {"model": "Replicate", "url": video_url})
                        else:
                            save_interaction_to_db("video_gen", prompt, success_msg,
                                                   {"model": "Replicate", "url": video_url})

                        yield success_msg, video_path
                        return
                except Exception as download_error:
                    logging.error(f"Failed to download Replicate video: {download_error}")

        except Exception as e:
            error_msg = str(e).lower()
            if any(x in error_msg for x in ["billing", "payment", "throttled", "rate limit", "429"]):
                logging.warning(f"⚠️ Replicate billing/rate limit issue: {e}")
                yield f"⚠️ Replicate requires billing setup or hit rate limit.\n\n**Switching to FREE services...**", None
                time.sleep(1)
            else:
                logging.warning(f"⚠️ Replicate failed: {e}")
                yield f"⚠️ Replicate unavailable.\n\n**Switching to FREE services...**", None
                time.sleep(1)

    hf_models = [
        {"name": "AnimateDiff", "api_url": "https://api-inference.huggingface.co/models/guoyww/animatediff"},
        {"name": "Text2Video-Zero", "api_url": "https://api-inference.huggingface.co/models/PAIR/text2video-zero"},
        {"name": "ModelScope Text-to-Video",
         "api_url": "https://api-inference.huggingface.co/models/damo-vilab/text-to-video-ms-1.7b"},
    ]

    for model_info in hf_models:
        model_name = model_info["name"]
        api_url = model_info["api_url"]

        yield f"⏳ Generating video with **{model_name}** (FREE)...\n\nPrompt: *{prompt}*\n\nThis may take 1-3 minutes. Please wait...", None

        try:
            headers = {}
            if HF_API_KEY:
                headers["Authorization"] = f"Bearer {HF_API_KEY}"

            payload = {"inputs": prompt}
            logging.info(f"Attempting FREE model: {model_name}")

            response = requests.post(api_url, headers=headers, json=payload, timeout=180)

            if response.status_code == 200 and len(response.content) > 1000:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(response.content)
                    video_path = tmp.name

                file_size = os.path.getsize(video_path)
                if file_size > 1000:
                    success_msg = f"✅ **Video generated successfully (FREE)!**\n\n**Model:** {model_name}\n**Prompt:** *{prompt[:100]}{'...' if len(prompt) > 100 else ''}*\n**Size:** {file_size / 1024:.1f} KB"
                    logging.info(f"✅ {model_name} success - {file_size} bytes")

                    if current_user["is_guest"]:
                        add_to_guest_history("video_gen", prompt, success_msg,
                                             {"model": model_name, "service": "Hugging Face (FREE)",
                                              "size": f"{file_size / 1024:.1f} KB"})
                    else:
                        save_interaction_to_db("video_gen", prompt, success_msg,
                                               {"model": model_name, "service": "Hugging Face (FREE)",
                                                "size": f"{file_size / 1024:.1f} KB"})

                    yield success_msg, video_path
                    return

            elif response.status_code == 503:
                yield f"⏳ {model_name} is loading... Waiting 20 seconds and retrying...", None
                time.sleep(20)
                response = requests.post(api_url, headers=headers, json=payload, timeout=180)
                if response.status_code == 200 and len(response.content) > 1000:
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                        tmp.write(response.content)
                        video_path = tmp.name
                    file_size = os.path.getsize(video_path)
                    if file_size > 1000:
                        success_msg = f"✅ **Video generated successfully (FREE)!**\n\n**Model:** {model_name}\n**Prompt:** *{prompt[:100]}{'...' if len(prompt) > 100 else ''}*\n**Size:** {file_size / 1024:.1f} KB"

                        if current_user["is_guest"]:
                            add_to_guest_history("video_gen", prompt, success_msg,
                                                 {"model": model_name, "size": f"{file_size / 1024:.1f} KB"})
                        else:
                            save_interaction_to_db("video_gen", prompt, success_msg,
                                                   {"model": model_name, "size": f"{file_size / 1024:.1f} KB"})

                        yield success_msg, video_path
                        return

        except requests.exceptions.Timeout:
            logging.error(f"{model_name} timed out")
            continue
        except Exception as e:
            logging.error(f"Error with {model_name}: {e}")
            continue

    final_error = (
        f"❌ **All video generation services failed**\n\n"
        f"**What was tried:**\n"
        f"1. ❌ Replicate (Premium) - Failed or needs billing\n"
        f"2. ❌ Hugging Face Models (FREE) - All busy/unavailable\n\n"
        f"**Why this happens:**\n"
        f"- FREE services require Hugging Face Pro subscription now\n"
        f"- Replicate needs billing setup to work\n\n"
        f"**What to do:**\n"
        f"1. **Setup Replicate billing** (recommended):\n"
        f"   - Go to https://replicate.com/account/billing\n"
        f"   - Add payment method (get $10 FREE credit)\n"
        f"   - Cost: ~$0.01-0.05 per video\n\n"
        f"2. **Try again later** (free services may work during off-peak hours)\n\n"
        f"**Alternative FREE services:**\n"
        f"- 🎬 RunwayML: https://app.runwayml.com (free tier)\n"
        f"- 🎬 Pika Labs: https://pika.art (Discord bot)\n"
        f"- 🎬 Haiper AI: https://haiper.ai (new, free)\n"
    )

    if current_user["is_guest"]:
        add_to_guest_history("video_gen", prompt, "Failed - all services unavailable")
    else:
        save_interaction_to_db("video_gen", prompt, final_error)

    yield final_error, None


def video_gen_wrapper(prompt):
    """Wrapper to handle generator streaming properly"""
    status = ""
    video = None
    for s, v in generate_video(prompt):
        status = s
        video = v
        yield status, video
    yield status, video


def process_audio_for_video_gen(audio_filepath):
    """Process audio input for video generation"""
    guest_check = check_guest_feature_access("Video Generation")
    if guest_check:
        return guest_check, None

    if audio_filepath is None:
        return "", None

    transcribed_text = transcribe_audio(audio_filepath)

    if transcribed_text.startswith("❌"):
        return transcribed_text, None

    last_status, last_video = "", None
    for status, video in generate_video(transcribed_text):
        last_status = status
        last_video = video

    return last_status, last_video


# --- Google Image Search ---
def google_image_search(query: str):
    """FIXED: Image search with history saving"""
    access_check = check_feature_access("image_search")
    if access_check:
        return None, access_check

    limit_msg = check_rate_limit("image_search")
    if limit_msg:
        return None, limit_msg

    if build is None:
        return None, "Error: 'googleapiclient' not installed. Please `pip install google-api-python-client`."

    if not query:
        return None, "Please enter a search query."

    if not GOOGLE_SEARCH_API_KEY or not GOOGLE_SEARCH_CX:
        return None, "Error: Google Search API Key or CX missing."

    increment_usage("image_search")

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

            if current_user["is_guest"]:
                add_to_guest_history("image_search", query, status_msg)
            else:
                save_interaction_to_db("image_search", query, status_msg)

            return None, status_msg

        for item in result['items']:
            image_url = item['link']
            if any(bad in image_url for bad in ["instagram.com", "facebook.com", "pinterest.com", ".svg", ".gif"]):
                logging.info(f"🚫 Skipping blocked domain/format: {image_url}")
                continue

            try:
                session = create_session_with_retries()
                resp = session.get(image_url, stream=True, timeout=15)
                resp.raise_for_status()

                ctype = resp.headers.get("Content-Type", "").lower()
                if not ctype.startswith("image/"):
                    logging.warning(f"⚠️ Non-image content-type received: {ctype}")
                    continue

                image = Image.open(io.BytesIO(resp.content)).convert("RGB")
                image.thumbnail((512, 512), Resampling.LANCZOS)

                status_msg = f"✅ Found and resized image for: **{query}**"

                if current_user["is_guest"]:
                    add_to_guest_history("image_search", query, status_msg, {"url": image_url})
                else:
                    save_interaction_to_db("image_search", query, status_msg, {"url": image_url})

                return image, status_msg
            except Exception as e:
                logging.warning(f"⚠️ Skipping invalid or unreachable image: {e}")
                continue

        status_msg = f"⚠️ No valid image results for: **{query}** after trying multiple options."

        if current_user["is_guest"]:
            add_to_guest_history("image_search", query, status_msg)
        else:
            save_interaction_to_db("image_search", query, status_msg)

        return None, status_msg
    except Exception as e:
        if "quotaExceeded" in str(e):
            set_rate_limit("image_search")
            error_msg = f"⚠️ Google Image Search: API quota exceeded. Try again after 12 hours."
        else:
            error_msg = f"Image Search failed: {e}"

        if not current_user["is_guest"]:
            save_interaction_to_db("image_search", query, error_msg)

        return None, error_msg


def process_audio_for_image_search(audio_filepath):
    """Process audio input for image search"""
    guest_check = check_guest_feature_access("Image Search")
    if guest_check:
        return None, guest_check

    if audio_filepath is None:
        return None, ""

    transcribed_text = transcribe_audio(audio_filepath)

    if transcribed_text.startswith("❌"):
        return None, transcribed_text

    return google_image_search(transcribed_text)


def handle_pasted_image(pasted_image):
    """Process pasted image"""
    if pasted_image is None:
        return None, "Please paste an image first."

    try:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            pasted_image.save(tmp.name)
            temp_path = tmp.name

        return pasted_image, f"✅ **Image pasted successfully!**\n\n💡 *Tip: Use the search box above to find similar images, or use Image Q&A tab to analyze this image.*"

    except Exception as e:
        return None, f"❌ Failed to process pasted image: {e}"


# ------------------ GRADIO UI ------------------
with gr.Blocks(title="Manjula AI Assistance", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Manjula AI Assistance")

    # Login/Register Section
    with gr.Group(visible=False) as auth_section:
        gr.Markdown("## 🔐 Welcome! Please Login or Register")
        guest_status = gr.Markdown("", visible=False)

        with gr.Tab("Login"):
            with gr.Row():
                login_username = gr.Textbox(label="Username", placeholder="Enter your username")
                login_password = gr.Textbox(label="Password", type="password", placeholder="Enter your password")
            login_btn = gr.Button("🔑 Login", variant="primary", size="lg")
            login_status = gr.Markdown()

        with gr.Tab("Register"):
            gr.Markdown("### 🎉 Create Your Free Account & Unlock All Features!")
            gr.Markdown("""
        **What you get with registration:**
        - ♾️ **Unlimited Chat** - No message limits
        - 📄 **File Q&A** - Upload and analyze any document
        - 🎨 **Image Generation** - Create AI art from text
        - 🎥 **Video Generation** - Generate AI videos
        - 🖼️ **Image Search** - Search Google Images  
        - 🔍 **Image Q&A** - Ask questions about images
        - 📊 **Usage Statistics** - Track your activity
        - 💾 **Persistent History** - All your conversations saved permanently
            """)
            with gr.Row():
                reg_username = gr.Textbox(label="Username *", placeholder="Choose a username (min 3 characters)")
                reg_password = gr.Textbox(label="Password *", type="password",
                                          placeholder="Choose a password (min 6 characters)")
            with gr.Row():
                reg_email = gr.Textbox(label="Email Address *",
                                       placeholder="your.email@example.com (Required for OTP verification)")
                reg_fullname = gr.Textbox(label="Full Name", placeholder="Your full name (optional)")

            request_otp_btn = gr.Button("📧 Send OTP to My Email", variant="primary", size="lg")

            otp_input = gr.Textbox(label="Enter 6-Digit OTP", placeholder="123456", max_lines=1, visible=False)
            verify_otp_btn = gr.Button("✅ Verify OTP & Complete Registration", variant="primary", size="lg",
                                       visible=False)

            register_status = gr.Markdown()

        if not MONGODB_AVAILABLE:
            gr.Markdown(
                "### ⚠️ Database Not Configured\n\nMongoDB is not connected. Please set `MONGODB_URI` in your `.env` file.\n\nExample: `MONGODB_URI=mongodb://localhost:27017/`")

    # Main App Section
    with gr.Group(visible=True) as main_app:
        with gr.Row():
            user_info = gr.Markdown(f"👤 **Guest Mode** | 💬 {guest_chat_count}/{GUEST_CHAT_LIMIT} chats used")
            logout_btn = gr.Button("🚪 Logout", size="sm")
            stats_btn = gr.Button("📊 My Statistics", size="sm", visible=False)

        with gr.Group(visible=False) as stats_modal:
            stats_content = gr.Markdown("Loading statistics...")
            close_stats_btn = gr.Button("✖️ Close", variant="secondary")

        gr.Markdown(
            "### 🎤 **Voice Input Available on All Tabs!** Click the microphone icon to speak instead of typing.")

        with gr.Row():
            chat_timer = gr.Textbox(label="Chat Timer", interactive=False, value=get_timer_text("text_qa", "Chat"))
            file_qa_timer = gr.Textbox(label="File Q&A Timer", interactive=False,
                                       value=get_timer_text("file_qa", "File Q&A"))
            image_timer = gr.Textbox(label="Image Timer", interactive=False,
                                     value=get_timer_text("image_gen", "Image Gen"))
            video_timer = gr.Textbox(label="Video Timer", interactive=False,
                                     value=get_timer_text("video_gen", "Video Gen"))
            image_search_timer = gr.Textbox(label="Image Search Timer", interactive=False,
                                            value=get_timer_text("image_search", "Image Search"))
            ip_timer = gr.Textbox(label="Public IP Timer", interactive=False,
                                  value=get_timer_text("public_ip", "Public IP"))

        with gr.Row():
            show_history_btn = gr.Button("📋 Show My History", variant="primary", size="lg")

        session_id = gr.State(0)

        with gr.Group(visible=False) as history_modal:
            with gr.Row():
                gr.Markdown("## 📚 Your Activity History")

            history_chatbot = gr.Chatbot(
                label="Activity Log (Your Data Only - Isolated)",
                height=500,
                show_copy_button=True,
                type="tuples"
            )

            with gr.Row():
                refresh_history_btn = gr.Button("🔄 Refresh", variant="secondary")
                clear_all_btn = gr.Button("🗑️ Clear My History", variant="stop")
                close_modal_btn = gr.Button("✖️ Close", variant="primary")
            history_status = gr.Textbox(label="Status", visible=True)

        show_history_btn.click(show_history_modal, outputs=[history_modal, history_chatbot])
        refresh_history_btn.click(format_history_for_chatbot, outputs=history_chatbot)
        clear_all_btn.click(clear_all_history_action, outputs=[history_chatbot, history_status])
        close_modal_btn.click(close_history_modal, outputs=history_modal)

        with gr.Tab("💬 Chat with Voice Input"):
            gr.Markdown("### 🎤 Use voice or text to chat with AI")

            guest_chat_warning = gr.Markdown("", visible=True)

            with gr.Row():
                mic_chat = gr.Audio(sources=["microphone"], type="filepath", label="🎙️ Click to Record Voice")

            chatbot = gr.Chatbot(label="Conversation", height=500, type="tuples")

            user_input = gr.Textbox(placeholder="Enter your message here... or use voice input above",
                                    label="Type your message")

            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                new_chat_btn = gr.Button("🆕 New Chat", variant="secondary")

            send_btn.click(
                query_and_update_warning,
                inputs=[user_input, chatbot, session_id],
                outputs=[chatbot, user_input, guest_chat_warning, session_id]
            )


            # ✅ CRITICAL FIX: Voice input wrapper with forced validation
            def safe_voice_handler(audio_filepath, history, session_id_state):
                """Wrapper that FORCES history validation before processing voice"""
                global current_session_id

                if session_id_state != current_session_id:
                    logging.error(
                        f"🚨 CRITICAL: Voice used stale session! Clearing. Expected: {current_session_id}, Got: {session_id_state}")
                    history = []
                    session_id_state = current_session_id

                return process_audio_and_chat(audio_filepath, history, session_id_state)


            mic_chat.change(
                safe_voice_handler,
                inputs=[mic_chat, chatbot, session_id],
                outputs=[chatbot, user_input, guest_chat_warning, session_id]
            )

            new_chat_btn.click(
                start_new_chat,
                inputs=None,
                outputs=[chatbot, session_id]
            )

        with gr.Tab("📄 File Q&A", visible=False) as file_qa_tab:
            gr.Markdown("### 🎤 Upload a file and ask ANYTHING - extract, analyze, generate, transform!")
            gr.Markdown(
                "**Examples:** Extract data, summarize, generate questions, reformat, translate, create study notes, find errors, etc.")

            with gr.Row():
                mic_file = gr.Audio(label="🎙️ Click to Record Question", sources=["microphone"], type="filepath")

            uploaded_file = gr.File(label="Upload Document or Image (PDF, JPG, PNG, DOCX, TXT, etc.)")
            file_question = gr.Textbox(label="Your Question/Request",
                                       placeholder="e.g., 'Summarize the document in 3 bullet points,' or 'Extract all names and addresses'")

            with gr.Row():
                file_processing_timer = gr.Textbox(label="⏱️ Processing Time", interactive=False, value="⏱️ 0s")

            file_output = gr.Markdown(label="AI Response")

            with gr.Row():
                file_submit_btn = gr.Button("Submit File & Question", variant="primary")

            file_submit_btn.click(
                file_question_answer,
                inputs=[uploaded_file, file_question],
                outputs=[file_processing_timer, file_output]
            )

            mic_file.change(
                process_audio_for_file_qa,
                inputs=[mic_file, uploaded_file],
                outputs=[file_processing_timer, file_output]
            )

        with gr.Tab("🎨 Image Generation", visible=False) as image_gen_tab:
            gr.Markdown("### 🎨 Generate stunning images from text prompts")

            with gr.Row():
                mic_img_gen = gr.Audio(label="🎙️ Click to Record Prompt", sources=["microphone"], type="filepath")

            image_prompt = gr.Textbox(label="Image Prompt",
                                      placeholder="e.g., 'A majestic lion wearing a crown, digital art'")
            image_output = gr.Image(label="Generated Image", type="pil", height=512)
            image_status = gr.Textbox(label="Status", interactive=False)
            image_gen_btn = gr.Button("Generate Image", variant="primary")

            image_gen_btn.click(
                generate_image_for_gradio,
                inputs=image_prompt,
                outputs=[image_output, image_status]
            )

            mic_img_gen.change(
                process_audio_for_image_gen,
                inputs=mic_img_gen,
                outputs=[image_output, image_status]
            )

        with gr.Tab("🔍 Image Q&A", visible=False) as image_qa_tab:
            gr.Markdown("### 🖼️ Ask questions about an uploaded image")

            with gr.Row():
                mic_img_qa = gr.Audio(label="🎙️ Click to Record Question", sources=["microphone"], type="filepath")

            qa_image = gr.Image(label="Upload Image", type="pil", height=300)
            qa_prompt = gr.Textbox(label="Question about the Image",
                                   placeholder="e.g., 'Describe what is happening in this photo,' or 'What is the brand of the product?'")
            qa_output = gr.Markdown(label="AI Analysis")
            qa_btn = gr.Button("Analyze Image", variant="primary")

            qa_btn.click(
                query_image_model,
                inputs=[qa_image, qa_prompt],
                outputs=qa_output
            )

            mic_img_qa.change(
                process_audio_for_image_qa,
                inputs=[mic_img_qa, qa_image],
                outputs=qa_output
            )

            gr.Markdown("#### 📋 Or Paste an Image (Ctrl+V):")
            with gr.Row():
                pasted_image = gr.Image(
                    label="Paste Image Here (Ctrl+V or right-click paste)",
                    type="pil",
                    height=300,
                    sources=["clipboard", "upload"]
                )

            search_output = gr.Image(label="Search Result Image", type="pil", height=512)
            search_status = gr.Textbox(label="Status", interactive=False)

            with gr.Row():
                process_paste_btn = gr.Button("Process Pasted Image", variant="secondary")

            process_paste_btn.click(
                handle_pasted_image,
                inputs=pasted_image,
                outputs=[search_output, search_status]
            )

        with gr.Tab("🖼️ Image Search", visible=False) as image_search_tab:
            gr.Markdown("### 🌐 Search and retrieve a relevant image from Google")
            gr.Markdown("**💡 NEW: Paste images with Ctrl+V!** Use the paste area below to paste images from clipboard.")

            with gr.Row():
                mic_img_search = gr.Audio(label="🎙️ Click to Record Query", sources=["microphone"], type="filepath")

            search_query = gr.Textbox(label="Image Search Query", placeholder="e.g., 'latest Mars rover photo'")

            search_output2 = gr.Image(label="Search Result Image", type="pil", height=512)
            search_status2 = gr.Textbox(label="Status", interactive=False)

            with gr.Row():
                search_btn = gr.Button("Search for Image", variant="primary")

            search_btn.click(
                google_image_search,
                inputs=search_query,
                outputs=[search_output2, search_status2]
            )

            mic_img_search.change(
                process_audio_for_image_search,
                inputs=mic_img_search,
                outputs=[search_output2, search_status2]
            )

        with gr.Tab("🎥 Video Generation", visible=False) as video_gen_tab:
            gr.Markdown("### 🎬 Generate AI Videos from Text Prompts")
            gr.Markdown(
                "**Smart Fallback System:** Tries Replicate (premium) first, then falls back to FREE services (Hugging Face + Pollinations)!")
            gr.Markdown(
                "💡 *No billing? No problem!* Free services work automatically if Replicate isn't set up.")

            with gr.Row():
                mic_video_gen = gr.Audio(label="🎙️ Click to Record Prompt", sources=["microphone"], type="filepath")

            video_prompt = gr.Textbox(label="Video Prompt",
                                      placeholder="e.g., 'A futuristic car flying over a neon city at night'")

            video_output = gr.Video(label="Generated Video", height=512, autoplay=False)
            video_status = gr.Textbox(label="Status", interactive=False)
            video_gen_btn = gr.Button("Generate Video", variant="primary")

            video_gen_btn.click(
                video_gen_wrapper,
                inputs=video_prompt,
                outputs=[video_status, video_output]
            )

            mic_video_gen.change(
                process_audio_for_video_gen,
                inputs=mic_video_gen,
                outputs=[video_status, video_output]
            )

        with gr.Tab("🌐 Public IP"):
            gr.Markdown("### 🌎 Check your current public IP address")
            ip_output = gr.Markdown(label="IP Address")
            ip_btn = gr.Button("Get Public IP", variant="primary")

            ip_btn.click(
                get_public_ip,
                inputs=None,
                outputs=ip_output
            )

    # Authentication Event Handlers
    login_btn.click(
        login_user,
        inputs=[login_username, login_password],
        outputs=[login_status, auth_section, main_app, user_info, file_qa_tab, image_gen_tab, image_qa_tab,
                 image_search_tab, video_gen_tab, stats_btn, history_chatbot, guest_chat_warning, chatbot, session_id, mic_chat]
    )

    request_otp_btn.click(
        request_otp,
        inputs=[reg_username, reg_password, reg_email, reg_fullname],
        outputs=[register_status, otp_input, verify_otp_btn]
    )

    verify_otp_btn.click(
        verify_otp_and_register,
        inputs=[reg_email, otp_input],
        outputs=[register_status, otp_input, verify_otp_btn]
    )

    logout_btn.click(
        logout_user,
        inputs=None,
        outputs=[login_status, auth_section, main_app, user_info, file_qa_tab, image_gen_tab, image_qa_tab,
                 image_search_tab, video_gen_tab, stats_btn, history_chatbot, guest_chat_warning, chatbot, session_id, mic_chat]
    )


    def show_stats():
        stats = get_user_stats()
        return gr.update(visible=True), stats


    def close_stats():
        return gr.update(visible=False)


    stats_btn.click(show_stats, outputs=[stats_modal, stats_content])
    close_stats_btn.click(close_stats, outputs=stats_modal)

    demo.load(
        fn=start_as_guest,
        inputs=None,
        outputs=[guest_status, auth_section, main_app, user_info, file_qa_tab, image_gen_tab, image_qa_tab,
                 image_search_tab, video_gen_tab, stats_btn, history_chatbot, guest_chat_warning, chatbot, session_id, mic_chat]
    )

if __name__ == "__main__":
    logging.info("🚀 Starting Manjula AI with Isolated User History...")
    demo.launch()