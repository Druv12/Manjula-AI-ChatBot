import gradio as gr
from dotenv import load_dotenv
import requests
from urllib.parse import quote
from google.ai import generativelanguage as genai
from google.generativeai.protos import Part
from datetime import datetime, timedelta
from PIL import Image
from PIL.Image import Resampling
import io
import time
import openai
from requests.adapters import HTTPAdapter, Retry
import json
import hashlib
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from datetime import datetime
import subprocess
import sys
from pathlib import Path
import tempfile
import logging
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
import argostranslate.package
import argostranslate.translate
# Correct import (one line)
from langdetect import detect, DetectorFactory
import logging
import google.generativeai as genai
import os
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth
from flask import Flask, request, jsonify
import threading
import random
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
flask_app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    firebase_service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "serviceAccountKey.json")

    if os.path.exists(firebase_service_account_path):
        cred = credentials.Certificate(firebase_service_account_path)
        try:
            firebase_admin.get_app()
            logging.info("✅ Firebase already initialized")
        except ValueError:
            firebase_admin.initialize_app(cred)
            logging.info("✅ Firebase Admin SDK initialized successfully from file")
    else:
        firebase_creds_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
        if firebase_creds_json:
            cred_dict = json.loads(firebase_creds_json)
            cred = credentials.Certificate(cred_dict)
            try:
                firebase_admin.get_app()
                logging.info("✅ Firebase already initialized")
            except ValueError:
                firebase_admin.initialize_app(cred)
                logging.info("✅ Firebase Admin SDK initialized from environment variable")
        else:
            logging.warning("⚠️ Firebase service account not configured. Google Sign-In unavailable.")
except Exception as e:
    logging.error(f"❌ Firebase initialization failed: {e}", exc_info=True)

MONGODB_URI = os.getenv("MONGODB_URI")
import traceback
# Now you can safely set the seed
DetectorFactory.seed = 0   # Makes detection consistent across runs

# ✅ ENHANCED LANGUAGE SUPPORT
SUPPORTED_LANGUAGES = {
    # Indian Languages (22)
    "Hindi": "hi",
    "Bengali": "bn",
    "Telugu": "te",
    "Marathi": "mr",
    "Tamil": "ta",
    "Gujarati": "gu",
    "Urdu": "ur",
    "Kannada": "kn",
    "Odia": "or",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Assamese": "as",
    "Maithili": "mai",
    "Sanskrit": "sa",
    "Konkani": "kok",
    "Nepali": "ne",
    "Sindhi": "sd",
    "Dogri": "doi",
    "Kashmiri": "ks",
    "Manipuri": "mni",
    "Bodo": "brx",
    "Santali": "sat",

    # Foreign Languages (30+)
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese (Simplified)": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Arabic": "ar",
    "Portuguese": "pt",
    "Italian": "it",
    "Dutch": "nl",
    "Turkish": "tr",
    "Polish": "pl",
    "Swedish": "sv",
    "Norwegian": "no",
    "Danish": "da",
    "Finnish": "fi",
    "Greek": "el",
    "Czech": "cs",
    "Hungarian": "hu",
    "Romanian": "ro",
    "Ukrainian": "uk",
    "Indonesian": "id",
    "Malay": "ms",
    "Thai": "th",
    "Vietnamese": "vi",
    "Hebrew": "he",
    "Persian": "fa",
    "Swahili": "sw",
}

# MongoDB imports
try:
    from pymongo import MongoClient
    from pymongo.errors import DuplicateKeyError
    MONGODB_AVAILABLE = True
except ImportError:
    logging.warning("⚠️ pymongo not found. Install with: pip install pymongo")
    MONGODB_AVAILABLE = False

# ================================
# FIREBASE ADMIN SDK SETUP
# ================================
FIREBASE_AVAILABLE = False

try:
    # Option 1: Try service account file path from environment
    firebase_service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "serviceAccountKey.json")

    if os.path.exists(firebase_service_account_path):
        cred = credentials.Certificate(firebase_service_account_path)

        # Check if Firebase is already initialized
        try:
            firebase_admin.get_app()
            logging.info("✅ Firebase already initialized")
            FIREBASE_AVAILABLE = True
        except ValueError:
            # Not initialized yet, so initialize it
            firebase_admin.initialize_app(cred)
            FIREBASE_AVAILABLE = True
            logging.info("✅ Firebase Admin SDK initialized successfully from file")
    else:
        # Option 2: Try JSON string from environment variable
        firebase_creds_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
        if firebase_creds_json:
            cred_dict = json.loads(firebase_creds_json)
            cred = credentials.Certificate(cred_dict)

            try:
                firebase_admin.get_app()
                FIREBASE_AVAILABLE = True
            except ValueError:
                firebase_admin.initialize_app(cred)
                FIREBASE_AVAILABLE = True
                logging.info("✅ Firebase Admin SDK initialized from environment variable")
        else:
            logging.warning("⚠️ Firebase service account not configured. Google Sign-In will be unavailable.")
            FIREBASE_AVAILABLE = False

except Exception as e:
    FIREBASE_AVAILABLE = False
    logging.error(f"❌ Firebase initialization failed: {e}")
    logging.error(traceback.format_exc())

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
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
# --- MongoDB Setup ---
db = None
users_collection = None

if MONGODB_AVAILABLE:
    try:
        MONGODB_URI = os.getenv("MONGODB_URI")
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        client.server_info()
        db = client["manjula_ai"]
        users_collection = db["users"]
        users_collection.create_index("username", unique=True)
        users_collection.create_index("email", unique=True)  # ✅ ADD THIS LINE
        users_collection.create_index("firebase_uid", unique=True, sparse=True)  # ✅ ADD THIS LINE
        logging.info("✅ MongoDB connected successfully")
        print("Number of users in the database:", users_collection.count_documents({}))
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

# ============================================================================
# VIDEO GENERATION SETUP - Add this block here
# ============================================================================
# Try to import imageio and numpy (required for local video generation)
IMAGEIO_AVAILABLE = False
imageio = None
np = None

try:
    import imageio
    import numpy as np
    IMAGEIO_AVAILABLE = True
    logging.info("✅ imageio and numpy available for local video generation")
except ImportError:
    logging.warning("⚠️ imageio or numpy not found. Install with: pip install imageio imageio-ffmpeg numpy")
    IMAGEIO_AVAILABLE = False

# Video generation API keys
OPENAI_SORA_API_KEY = os.getenv("OPENAI_SORA_API_KEY")
RUNWAYML_GEN2_API_KEY = os.getenv("RUNWAYML_GEN2_API_KEY")
# ============================================================================

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

# ================================
#  LANGUAGE TRANSLATION MODULE
# ================================
# --- Cache for installed languages ---
INSTALLED_LANGS = {}


def _ensure_language_pack(source_code: str, target_code: str):
    """Download language pack if not already installed."""
    pack_key = f"{source_code}->{target_code}"

    if pack_key in INSTALLED_LANGS:
        return INSTALLED_LANGS[pack_key]

    # Update package index
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()

    # Find package: source → target
    package_to_install = next(
        (pkg for pkg in available_packages
         if pkg.from_code == source_code and pkg.to_code == target_code),
        None
    )

    if package_to_install is None:
        # Try to find via English as intermediate
        logging.warning(f"Direct {source_code}->{target_code} not found. Trying via English...")

        # Install source -> en
        pkg_to_en = next(
            (pkg for pkg in available_packages
             if pkg.from_code == source_code and pkg.to_code == "en"),
            None
        )

        # Install en -> target
        pkg_from_en = next(
            (pkg for pkg in available_packages
             if pkg.from_code == "en" and pkg.to_code == target_code),
            None
        )

        if pkg_to_en:
            download_path = pkg_to_en.download()
            argostranslate.package.install_from_path(download_path)
            logging.info(f"Installed intermediate pack: {source_code}->en")

        if pkg_from_en:
            download_path = pkg_from_en.download()
            argostranslate.package.install_from_path(download_path)
            logging.info(f"Installed intermediate pack: en->{target_code}")

        INSTALLED_LANGS[pack_key] = True
        return True

    # Direct package found - install it
    download_path = package_to_install.download()
    argostranslate.package.install_from_path(download_path)
    INSTALLED_LANGS[pack_key] = True
    logging.info(f"Installed Argos language pack: {source_code}->{target_code}")
    return True


def translate_text(text: str, target_lang: str = "es", source_lang: str = "auto") -> str:
    """
    ENHANCED: Multi-service translation with Gemini as primary (FREE, best quality)
    Falls back to LibreTranslate and ArgosTranslate
    """
    if not text.strip():
        return ""

    # Detect source language if auto
    if source_lang == "auto":
        try:
            detected = detect(text)
            valid_codes = set(SUPPORTED_LANGUAGES.values())
            source_lang = detected if detected in valid_codes else "en"
            logging.info(f"Detected source language: {source_lang}")
        except Exception as e:
            logging.warning(f"Language detection failed: {e}. Assuming English.")
            source_lang = "en"

    # Don't translate if source and target are the same
    if source_lang == target_lang:
        return text

    # Validate target language
    valid_codes = set(SUPPORTED_LANGUAGES.values())
    if target_lang not in valid_codes:
        return f"[Translation failed: Unsupported target language '{target_lang}']"

    # ============================================================
    # METHOD 1: Gemini 2.0 Flash (PRIMARY - FREE, 100+ languages)
    # ============================================================
    if GEMINI_CLIENT:
        logging.info("🌐 Trying Gemini 2.0 Flash translation (FREE)...")
        gemini_result = translate_text_gemini(text, target_lang, source_lang)
        if gemini_result and len(gemini_result) > 0:
            return gemini_result

    # ============================================================
    # METHOD 2: LibreTranslate (FREE, no API key)
    # ============================================================
    try:
        logging.info("🌐 Trying LibreTranslate (FREE)...")
        url = "https://libretranslate.com/translate"
        payload = {
            "q": text,
            "source": source_lang,
            "target": target_lang,
            "format": "text"
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            result = response.json()
            translated = result.get("translatedText", "").strip()
            if translated:
                logging.info(f"✅ LibreTranslate success: {source_lang}->{target_lang}")
                return translated
    except Exception as e:
        logging.warning(f"LibreTranslate failed: {e}")

    # ============================================================
    # METHOD 3: ArgosTranslate (OFFLINE fallback)
    # ============================================================
    try:
        logging.info("🌐 Trying ArgosTranslate (OFFLINE)...")
        _ensure_language_pack(source_lang, target_lang)

        installed_languages = argostranslate.translate.get_installed_languages()
        from_lang = next((lang for lang in installed_languages if lang.code == source_lang), None)
        to_lang = next((lang for lang in installed_languages if lang.code == target_lang), None)

        if not from_lang or not to_lang:
            en_lang = next((lang for lang in installed_languages if lang.code == "en"), None)
            if en_lang and from_lang:
                trans_to_en = from_lang.get_translation(en_lang)
                if trans_to_en:
                    intermediate_text = trans_to_en.translate(text)
                    trans_from_en = en_lang.get_translation(to_lang)
                    if trans_from_en:
                        translated = trans_from_en.translate(intermediate_text)
                        logging.info(f"✅ ArgosTranslate via English: {source_lang}->en->{target_lang}")
                        return translated.strip()
        else:
            translation = from_lang.get_translation(to_lang)
            if translation:
                translated = translation.translate(text)
                logging.info(f"✅ ArgosTranslate direct: {source_lang}->{target_lang}")
                return translated.strip()

    except Exception as e:
        logging.error(f"ArgosTranslate failed: {e}")

    return f"[Translation failed: All services unavailable. Source: {source_lang}, Target: {target_lang}]"


def speak_translation(text: str, lang: str):
    """Generate MP3 audio of the translated text."""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)
        logging.info(f"✅ Audio generated for language: {lang}")
        return tmp.name
    except Exception as e:
        logging.error(f"TTS failed: {e}")
        return None


def translate_text_gemini(text: str, target_lang: str, source_lang: str = "auto") -> str:
    """
    Translate text using Gemini 2.0 Flash (PRIMARY - FREE, 100+ languages)
    """
    if not GEMINI_CLIENT:
        return None

    try:
        # Detect source if auto
        if source_lang == "auto":
            try:
                source_lang = detect(text)
            except:
                source_lang = "en"

        # Get full language names for better results
        source_name = next((name for name, code in SUPPORTED_LANGUAGES.items() if code == source_lang), source_lang)
        target_name = next((name for name, code in SUPPORTED_LANGUAGES.items() if code == target_lang), target_lang)

        # Construct translation prompt
        prompt = f"""Translate the following text from {source_name} to {target_name}.

Rules:
- Provide ONLY the translated text, nothing else
- Maintain the original meaning and tone
- Keep formatting if present
- Do not add explanations or notes

Text to translate:
{text}

Translation:"""

        model = GEMINI_CLIENT.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content(prompt)

        if response and response.text:
            translated = response.text.strip()
            # Remove any common prefixes that might appear
            prefixes_to_remove = [
                "Translation:", "Here is the translation:",
                "Translated text:", "The translation is:"
            ]
            for prefix in prefixes_to_remove:
                if translated.lower().startswith(prefix.lower()):
                    translated = translated[len(prefix):].strip()

            logging.info(f"✅ Gemini translation success: {source_lang}->{target_lang}")
            return translated

        return None

    except Exception as e:
        logging.error(f"Gemini translation failed: {e}")
        return None

# --- Save translation history (only for logged-in users) ---
def save_translation_history(userid, original, translated, target_lang, src_lang="en"):
    """Save translation to MongoDB (logged-in users only)."""
    if not MONGODB_AVAILABLE or not current_user.get("logged_in"):
        return
    try:
        col = db["translation_history"]
        col.insert_one({
            "userid": userid,
            "original": original,
            "translated": translated,
            "source_lang": src_lang,
            "target_lang": target_lang,
            "timestamp": datetime.now()
        })
        logging.info(f"✅ Translation history saved for user: {userid}")
    except Exception as e:
        logging.error(f"Failed to save translation history: {e}")


def perform_translation(text, target_lang):
    """Main translation function with auto-detection"""
    if not text.strip():
        return "", None, "Please enter text to translate."

    # Auto-detect source language
    try:
        source_lang = detect(text)
        logging.info(f"Detected source language: {source_lang}")
    except Exception as e:
        logging.warning(f"Language detection failed: {e}. Assuming English.")
        source_lang = "en"

    # Check if source and target are the same
    if source_lang == target_lang:
        return text, None, f"**Original:** {text}\n\n⚠️ Source and target languages are the same ({source_lang})"

    # Translate
    translated = translate_text(text, target_lang, source_lang)

    # Generate audio
    audio_path = speak_translation(translated, target_lang)

    # Save to history (if logged in)
    if current_user.get("logged_in"):
        save_translation_history(
            current_user["username"],
            text,
            translated,
            target_lang,
            source_lang
        )

    return (
        translated,
        audio_path,
        f"**Original ({source_lang}):** {text}\n\n**Translated ({target_lang}):** {translated}"
    )
# --- User Authentication Functions ---
def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


# ================================
# FIREBASE AUTHENTICATION FUNCTIONS
# ================================

def verify_firebase_token(id_token):
    """Verify Firebase ID token with clock skew tolerance"""

    try:
        firebase_admin.get_app()
    except ValueError:
        print("❌ Firebase not initialized")
        return None

    try:
        print("Calling Firebase Admin SDK verify_id_token()...")

        # ✅ FIX: Add clock_skew_seconds parameter to handle time differences
        decoded_token = firebase_auth.verify_id_token(
            id_token,
            check_revoked=False,
            clock_skew_seconds=10  # ✅ Allows 10 seconds tolerance for clock drift
        )

        print(f"✅ Token decoded successfully")
        print(f"   UID: {decoded_token.get('uid')}")
        print(f"   Email: {decoded_token.get('email')}")
        print(f"   Name: {decoded_token.get('name')}")

        return {
            "uid": decoded_token['uid'],
            "email": decoded_token.get('email'),
            "name": decoded_token.get('name'),
            "picture": decoded_token.get('picture'),
            "email_verified": decoded_token.get('email_verified', False)
        }

    except Exception as e:
        print(f"❌ Verification error: {type(e).__name__}: {str(e)}")
        print(traceback.format_exc())
        return None


def register_or_login_firebase_user(user_info):
    """
    Register or login user from Firebase authentication
    ✅ PREVENTS duplicate email registration
    ✅ Handles both Google and traditional auth users
    """
    global current_session_id, guest_chat_count

    if not MONGODB_AVAILABLE:
        # Fallback without database
        current_user["username"] = user_info["email"].split("@")[0]
        current_user["logged_in"] = True
        current_user["is_guest"] = False
        current_user["email"] = user_info["email"]
        current_user["full_name"] = user_info.get("name", "")
        return True, "Logged in successfully (no database)"

    try:
        username = user_info["email"].split("@")[0].lower()
        email = user_info["email"].lower()  # ✅ FIXED: Lowercase comparison
        full_name = user_info.get("name", "")

        # ✅ Check if user exists by email (case-insensitive)
        existing_user = users_collection.find_one({"email": email})

        if existing_user:
            # ✅ Email already registered - check auth provider
            auth_provider = existing_user.get("auth_provider", "email")

            if auth_provider == "google":
                # Existing Google user - LOG THEM IN
                clear_guest_history()
                guest_chat_count = 0
                current_session_id += 1

                users_collection.update_one(
                    {"email": email},
                    {"$set": {"last_login": datetime.now()}}
                )

                current_user["username"] = existing_user["username"]
                current_user["logged_in"] = True
                current_user["is_guest"] = False
                current_user["email"] = email
                current_user["full_name"] = existing_user.get("full_name", full_name)

                logging.info(f"✅ Existing Google user logged in: {username}")
                return True, f"Welcome back, {full_name or username}!"

            else:
                # Email registered with traditional method - PREVENT Google registration
                logging.warning(f"⚠️ Registration blocked: {email} already exists (traditional auth)")
                return False, (
                    f"❌ **User Already Exists**\n\n"
                    f"The email **{email}** is already registered with a password.\n\n"
                    f"**To access your account:**\n"
                    f"1. Click the **'Login'** tab above\n"
                    f"2. Enter your username and password\n\n"
                    f"💡 *Forgot your password? Contact support for help.*"
                )

        # ✅ Check if username is taken (different from email check)
        existing_username = users_collection.find_one({"username": username})
        if existing_username:
            # Generate a unique username by appending numbers
            base_username = username
            counter = 1
            while users_collection.find_one({"username": f"{base_username}{counter}"}):
                counter += 1
            username = f"{base_username}{counter}"
            logging.info(f"Username {base_username} taken, using {username} instead")

        # ✅ NEW USER - Create account
        try:
            new_user = {
                "username": username,
                "email": email,
                "full_name": full_name,
                "password": None,  # No password for OAuth users
                "auth_provider": "google",
                "firebase_uid": user_info["uid"],
                "profile_picture": user_info.get("picture"),
                "email_verified": user_info.get("email_verified", False),
                "created_at": datetime.now(),
                "last_login": datetime.now(),
                "usage_count": {
                    "chat": 0,
                    "file_qa": 0,
                    "image_gen": 0,
                    "video_gen": 0,
                    "image_search": 0,
                    "image_qa": 0
                },
                "history": [],
                "ip_history": []
            }

            users_collection.insert_one(new_user)

            clear_guest_history()
            guest_chat_count = 0
            current_session_id += 1

            current_user["username"] = username
            current_user["logged_in"] = True
            current_user["is_guest"] = False
            current_user["email"] = email
            current_user["full_name"] = full_name

            logging.info(f"✅ New Google user registered: {username}")
            return True, f"🎉 Welcome to All Mind, {full_name or username}!"

        except DuplicateKeyError as e:
            # Handle race condition (extremely rare)
            error_field = "email" if "email" in str(e) else "username"
            logging.error(f"⚠️ Duplicate key error on {error_field}: {email}")
            return False, (
                f"❌ **Registration Failed**\n\n"
                f"This {error_field} was just registered by another session.\n\n"
                f"Please try logging in instead, or use a different {error_field}."
            )

    except Exception as e:
        logging.error(f"Firebase registration/login failed: {e}")
        return False, f"❌ Authentication failed: {str(e)}"

def request_otp(username, password, email, fullname):
    """Send OTP to email for registration"""
    if not username or not password or not email:
        return (
            "**Error:** Username, password, and email are required!",
            gr.update(visible=False),
            gr.update(visible=False)
        )

    if len(username) < 3:
        return (
            "**Error:** Username must be at least 3 characters long!",
            gr.update(visible=False),
            gr.update(visible=False)
        )

    if len(password) < 6:
        return (
            "**Error:** Password must be at least 6 characters long!",
            gr.update(visible=False),
            gr.update(visible=False)
        )

    # ✅ FIXED: Normalize email and username to lowercase
    email = email.lower().strip()
    username = username.lower().strip()

    # Check if user already exists
    if MONGODB_AVAILABLE:
        try:
            # ✅ Check for existing username
            existing_user = users_collection.find_one({"username": username})
            if existing_user:
                return (
                    f"❌ **Username Already Taken**\n\n"
                    f"The username **'{username}'** is already registered.\n\n"
                    f"**Options:**\n"
                    f"1. Try a different username\n"
                    f"2. If this is your account, use the **Login** tab instead",
                    gr.update(visible=False),
                    gr.update(visible=False)
                )

            # ✅ Check for existing email
            existing_email = users_collection.find_one({"email": email})
            if existing_email:
                auth_provider = existing_email.get("auth_provider", "email")
                if auth_provider == "google":
                    return (
                        f"❌ **Email Already Registered with Google**\n\n"
                        f"The email **{email}** is already registered using Google Sign-In.\n\n"
                        f"**To access your account:**\n"
                        f"1. Click the **'Login'** tab\n"
                        f"2. Use **'Continue with Google'** button\n"
                        f"3. Sign in with the same Google account",
                        gr.update(visible=False),
                        gr.update(visible=False)
                    )
                else:
                    return (
                        f"❌ **Email Already Registered**\n\n"
                        f"The email **{email}** is already registered.\n\n"
                        f"**To access your account:**\n"
                        f"1. Click the **'Login'** tab\n"
                        f"2. Enter your username and password\n\n"
                        f"💡 *Forgot your password? Contact support for help.*",
                        gr.update(visible=False),
                        gr.update(visible=False)
                    )
        except Exception as e:
            logging.error(f"Database check error: {e}")
            return (
                f"**Error:** Database error occurred. Please try again.\n\n{str(e)}",
                gr.update(visible=False),
                gr.update(visible=False)
            )

    # Generate 6-digit OTP
    otp = ''.join(random.choices(string.digits, k=6))

    # Store OTP temporarily (expires in 10 minutes)
    otp_storage[email] = {
        "otp": otp,
        "username": username,
        "password": password,
        "fullname": fullname,
        "expires": datetime.now() + timedelta(minutes=10)
    }

    # Send OTP via email
    try:
        if not SMTP_EMAIL or not SMTP_PASSWORD:
            return (
                "**Error:** Email service not configured. Please contact administrator.",
                gr.update(visible=False),
                gr.update(visible=False)
            )

        msg = MIMEMultipart()
        msg['From'] = SMTP_EMAIL
        msg['To'] = email
        msg['Subject'] = "Your Manjula AI Registration OTP"

        body = f"""
Hello {fullname or username}!

Your OTP for registering with Manjula AI is: {otp}

This OTP will expire in 10 minutes.

If you didn't request this, please ignore this email.

Best regards,
Manjula AI Team
        """

        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        text = msg.as_string()
        server.sendmail(SMTP_EMAIL, email, text)
        server.quit()

        logging.info(f"✅ OTP sent to {email}")

        return (
            f"✅ **OTP sent successfully to {email}!**\n\nPlease check your email and enter the 6-digit code below.",
            gr.update(visible=True),  # otp_input
            gr.update(visible=True)  # verify_otp_btn
        )

    except Exception as e:
        logging.error(f"Failed to send OTP email: {e}")
        return (
            f"**Error:** Failed to send OTP. Please check your email address or try again later.\n\nError: {str(e)}",
            gr.update(visible=False),
            gr.update(visible=False)
        )


def verify_otp_and_register(email, otp):
    """Verify OTP and complete registration"""
    global current_session_id, guest_chat_count

    if not email or not otp:
        return (
            "**Error:** Please enter the OTP code!",
            gr.update(visible=True),
            gr.update(visible=True)
        )

    # Check if OTP exists and is valid
    if email not in otp_storage:
        return (
            "**Error:** No OTP found for this email. Please request a new OTP.",
            gr.update(visible=False),
            gr.update(visible=False)
        )

    stored_data = otp_storage[email]

    # Check if OTP expired
    if datetime.now() > stored_data["expires"]:
        del otp_storage[email]
        return (
            "**Error:** OTP has expired. Please request a new one.",
            gr.update(visible=False),
            gr.update(visible=False)
        )

    # Verify OTP
    if otp.strip() != stored_data["otp"]:
        return (
            "**Error:** Invalid OTP! Please check and try again.",
            gr.update(visible=True),
            gr.update(visible=True)
        )

    # OTP verified - create user account
    username = stored_data["username"]
    password = stored_data["password"]
    fullname = stored_data["fullname"]

    # Remove OTP from storage
    del otp_storage[email]

    if not MONGODB_AVAILABLE:
        # Fallback without database
        clear_guest_history()
        guest_chat_count = 0
        current_session_id += 1

        current_user["username"] = username
        current_user["logged_in"] = True
        current_user["is_guest"] = False
        current_user["email"] = email
        current_user["full_name"] = fullname

        return (
            f"✅ **Registration successful!**\n\nWelcome, {fullname or username}! You can now use all features.",
            gr.update(visible=False),
            gr.update(visible=False)
        )

    # Create user in database
    try:
        new_user = {
            "username": username.lower().strip(),
            "email": email.lower().strip(),
            "full_name": fullname or "",
            "password": hash_password(password),
            "auth_provider": "email",
            "created_at": datetime.now(),
            "last_login": datetime.now(),
            "usage_count": {
                "chat": 0,
                "file_qa": 0,
                "image_gen": 0,
                "video_gen": 0,
                "image_search": 0,
                "image_qa": 0
            },
            "history": [],
            "ip_history": []
        }

        users_collection.insert_one(new_user)

        logging.info(f"✅ New user registered: {username}")

        return (
            f"✅ **Registration successful!**\n\nWelcome to Manjula AI, {fullname or username}!\n\n**Please login to continue.**",
            gr.update(visible=False),
            gr.update(visible=False)
        )

    except DuplicateKeyError:
        return (
            f"**Error:** Username or email already exists! Please try a different one.",
            gr.update(visible=False),
            gr.update(visible=False)
        )
    except Exception as e:
        logging.error(f"Registration error: {e}")
        return (
            f"**Error:** Registration failed. Please try again.\n\nError: {str(e)}",
            gr.update(visible=False),
            gr.update(visible=False)
        )

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

    # ---- reset guest state -------------------------------------------------
    guest_chat_count = 0
    current_user["username"]   = "Guest"
    current_user["logged_in"]  = False
    current_user["is_guest"]   = True

    current_session_id += 1
    clear_guest_history()

    # ---- 16 return values (order must match demo.load outputs) ------------
    return (
        # 1. guest_status (Markdown)
        "Wellcome, Guest! You can try the Chat feature with 10 free messages.\n\n"
        "**Register to unlock:**\n"
        "- File Q&A\n"
        "- Image Generation\n"
        "- Video Generation\n"
        "- Image Search\n"
        "- Image Q&A\n"
        "- Usage Statistics\n"
        "- Unlimited Chat",

        # 2. auth_section (Group) – hide login/register
        gr.update(visible=False),

        # 3. main_app (Group) – show the main UI
        gr.update(visible=True),

        # 4. user_info (Markdown)
        f"Guest Mode | {guest_chat_count}/{GUEST_CHAT_LIMIT} chats used",

        # 5-9. Feature tabs – all hidden for guests
        gr.update(visible=False),   # file_qa_tab
        gr.update(visible=False),   # image_gen_tab
        gr.update(visible=False),   # image_qa_tab
        gr.update(visible=False),   # image_search_tab
        gr.update(visible=False),   # video_gen_tab

        # 10. translation_tab – **must be here**
        gr.update(visible=False),

        # 11. stats_btn (Button) – hidden for guests
        gr.update(visible=False),

        # 12. history_chatbot (Chatbot) – empty
        [],

        # 13. guest_chat_warning (Markdown)
        gr.update(
            visible=True,
            value=(
                f"Guest Mode: You have {GUEST_CHAT_LIMIT}/{GUEST_CHAT_LIMIT} "
                "free chats remaining. Register to get unlimited access!"
            )
        ),

        # 14. chatbot (Chatbot) – empty conversation
        [],

        # 15. session_id (State)
        current_session_id,

        # 16. mic_chat (Audio) – no file yet
        None
    )


# ================================
# FIREBASE WEB CONFIG (for frontend)  ← ADD THIS
# ================================
# ================================
# FIREBASE WEB CONFIG (for frontend)
# ================================
firebase_config = {
    "apiKey": os.getenv("FIREBASE_API_KEY", ""),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN", ""),
    "projectId": os.getenv("FIREBASE_PROJECT_ID", ""),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET", ""),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID", ""),
    "appId": os.getenv("FIREBASE_APP_ID", ""),
    "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID", "")
}

firebase_config_json = json.dumps(firebase_config)

register_html = """
<div style="padding: 20px; text-align: center;">
    <style>
        .google-signin-btn {
            background-color: #4285f4;
            color: white !important;
            border: none;
            padding: 12px 24px;
            font-size: 16px;
            font-weight: 500;
            border-radius: 8px;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 10px;
            transition: all 0.3s;
            margin: 10px auto;
            text-decoration: none;
        }
        .google-signin-btn:hover {
            background-color: #357ae8;
            color: white !important;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(66, 133, 244, 0.4);
        }
    </style>

    <a href="http://127.0.0.1:5000/firebase-auth?action=register" target="_blank" class="google-signin-btn">
        <svg width="18" height="18" viewBox="0 0 48 48">
            <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
            <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
            <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
            <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
        </svg>
        <span style="color: white;">Continue with Google</span>
    </a>
</div>
"""

login_html = """
<div style="padding: 20px; text-align: center;">
    <style>
        .google-signin-btn {
            background-color: #4285f4;
            color: white !important;
            border: none;
            padding: 12px 24px;
            font-size: 16px;
            font-weight: 500;
            border-radius: 8px;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 10px;
            transition: all 0.3s;
            margin: 10px auto;
            text-decoration: none;
        }
        .google-signin-btn:hover {
            background-color: #357ae8;
            color: white !important;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(66, 133, 244, 0.4);
        }
    </style>

    <a href="http://127.0.0.1:5000/firebase-auth?action=login" target="_blank" class="google-signin-btn">
        <svg width="18" height="18" viewBox="0 0 48 48">
            <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
            <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
            <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
            <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
        </svg>
        <span style="color: white;">Continue with Google</span>
    </a>
</div>
"""

# ================================
# FIREBASE LOGIN HANDLER - Returns 16 outputs like login_user()
# ================================

def login_firebase_user(user_info):
    """
    Login user from Firebase authentication - checks if user is registered
    Returns same 16 outputs as login_user() for Gradio UI update
    """
    global current_session_id, guest_chat_count

    # === 1. MONGODB NOT AVAILABLE (Fallback) ===
    if not MONGODB_AVAILABLE:
        current_session_id += 1
        clear_guest_history()
        guest_chat_count = 0

        username = user_info["email"].split("@")[0]
        current_user["username"] = username
        current_user["logged_in"] = True
        current_user["is_guest"] = False
        current_user["email"] = user_info["email"]
        current_user["full_name"] = user_info.get("name", "")

        return (
            f"**Welcome back, {username}!**\n\nYou have full access to all features!",
            gr.update(visible=False),  # auth_section
            gr.update(visible=True),  # main_app
            f"**Logged in as:** {username}",
            gr.update(visible=True),  # file_qa_tab
            gr.update(visible=True),  # image_gen_tab
            gr.update(visible=True),  # image_qa_tab
            gr.update(visible=True),  # image_search_tab
            gr.update(visible=True),  # video_gen_tab
            gr.update(visible=True),  # translation_tab
            gr.update(visible=True),  # stats_btn
            [],  # history_chatbot
            gr.update(visible=False),  # guest_chat_warning
            [],  # chatbot
            current_session_id,  # session_id
            None  # mic_chat
        )

    # === 2. CHECK IF USER EXISTS ===
    try:
        email = user_info["email"]
        existing_user = users_collection.find_one({"email": email})

        if not existing_user:
            # User not registered - return error state
            logging.warning(f"⚠️ Login attempt with unregistered email: {email}")
            error_msg = (
                f"❌ **Account Not Found**\n\n"
                f"The email **{email}** is not registered.\n\n"
                f"**Please register first:**\n"
                f"1. Click the **'Register'** tab above\n"
                f"2. Use **'Continue with Google'** button\n"
                f"3. Sign in with the same Google account\n\n"
                f"Registration takes less than 30 seconds! 🚀"
            )

            return (
                error_msg,  # login_status
                gr.update(visible=True),  # auth_section
                gr.update(visible=False),  # main_app
                "**Not logged in**",  # user_info
                gr.update(visible=False),  # file_qa_tab
                gr.update(visible=False),  # image_gen_tab
                gr.update(visible=False),  # image_qa_tab
                gr.update(visible=False),  # image_search_tab
                gr.update(visible=False),  # video_gen_tab
                gr.update(visible=False),  # translation_tab
                gr.update(visible=False),  # stats_btn
                [],  # history_chatbot
                gr.update(visible=False),  # guest_chat_warning
                [],  # chatbot
                current_session_id,  # session_id
                None  # mic_chat
            )

        # === 3. VALID LOGIN - USER EXISTS ===
        clear_guest_history()
        guest_chat_count = 0
        current_session_id += 1

        users_collection.update_one(
            {"email": email},
            {"$set": {"last_login": datetime.now()}}
        )

        current_user["username"] = existing_user["username"]
        current_user["logged_in"] = True
        current_user["is_guest"] = False
        current_user["email"] = email
        current_user["full_name"] = existing_user.get("full_name", user_info.get("name", ""))

        logging.info(f"✅ Firebase user logged in: {existing_user['username']} | Session ID: {current_session_id}")

        return (
            f"**✅ Welcome back, {current_user['full_name'] or current_user['username']}!**\n\nYou have full access to all features!",
            gr.update(visible=False),  # auth_section
            gr.update(visible=True),  # main_app
            f"**Logged in as:** {current_user['username']}",
            gr.update(visible=True),  # file_qa_tab
            gr.update(visible=True),  # image_gen_tab
            gr.update(visible=True),  # image_qa_tab
            gr.update(visible=True),  # image_search_tab
            gr.update(visible=True),  # video_gen_tab
            gr.update(visible=True),  # translation_tab
            gr.update(visible=True),  # stats_btn
            [],  # history_chatbot
            gr.update(visible=False),  # guest_chat_warning
            [],  # chatbot
            current_session_id,  # session_id
            None  # mic_chat
        )

    except Exception as e:
        logging.error(f"Firebase login failed: {e}")
        return (
            f"**Login failed:** {str(e)}",
            gr.update(visible=True),  # auth_section
            gr.update(visible=False),  # main_app
            "**Not logged in**",
            gr.update(visible=False),  # file_qa_tab
            gr.update(visible=False),  # image_gen_tab
            gr.update(visible=False),  # image_qa_tab
            gr.update(visible=False),  # image_search_tab
            gr.update(visible=False),  # video_gen_tab
            gr.update(visible=False),  # translation_tab
            gr.update(visible=False),  # stats_btn
            [],  # history_chatbot
            gr.update(visible=True),  # guest_chat_warning
            [],  # chatbot
            current_session_id,  # session_id
            None  # mic_chat
        )

def login_user(username, password):
    """Login user - FIXED with complete session isolation & 16 outputs"""
    global current_session_id, guest_chat_count

    # === 1. MONGODB NOT AVAILABLE (Fallback) ===
    if not MONGODB_AVAILABLE:
        current_session_id += 1
        clear_guest_history()
        guest_chat_count = 0,
        current_user["username"] = username
        current_user["logged_in"] = True
        current_user["is_guest"] = False

        return (
            f"**Welcome back, {username}!**\n\nYou have full access to all features!",
            gr.update(visible=False),  # auth_section
            gr.update(visible=True),   # main_app
            f"**Logged in as:** {username}",
            gr.update(visible=True),   # file_qa_tab
            gr.update(visible=True),   # image_gen_tab
            gr.update(visible=True),   # image_qa_tab
            gr.update(visible=True),   # image_search_tab
            gr.update(visible=True),   # video_gen_tab
            gr.update(visible=True),   # translation_tab ← ADDED
            gr.update(visible=True),   # stats_btn
            [],                        # history_chatbot
            gr.update(visible=False),  # guest_chat_warning
            [],                        # chatbot
            current_session_id,        # session_id
            None                       # mic_chat
        )

    # === 2. EMPTY CREDENTIALS ===
    if not username or not password:
        return (
            "**Username and password are required!**",
            gr.update(visible=True),   # auth_section
            gr.update(visible=False),  # main_app
            "**Not logged in**",
            gr.update(visible=False),  # file_qa_tab
            gr.update(visible=False),  # image_gen_tab
            gr.update(visible=False),  # image_qa_tab
            gr.update(visible=False),  # image_search_tab
            gr.update(visible=False),  # video_gen_tab
            gr.update(visible=False),  # translation_tab ← ADDED
            gr.update(visible=False),  # stats_btn
            [],                        # history_chatbot
            gr.update(visible=False),  # guest_chat_warning
            [],                        # chatbot
            current_session_id,        # session_id
            None                       # mic_chat
        )

    # === 3. VALID LOGIN ===
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

            logging.info(f"User logged in: {username} | Session ID: {current_session_id} | Guest data WIPED")

            return (
                f"**Welcome back, {username}!**\n\nYou have full access to all features!",
                gr.update(visible=False),  # auth_section
                gr.update(visible=True),   # main_app
                f"**Logged in as:** {username}",
                gr.update(visible=True),   # file_qa_tab
                gr.update(visible=True),   # image_gen_tab
                gr.update(visible=True),   # image_qa_tab
                gr.update(visible=True),   # image_search_tab
                gr.update(visible=True),   # video_gen_tab
                gr.update(visible=True),   # translation_tab ← ADDED
                gr.update(visible=True),   # stats_btn
                [],                        # history_chatbot
                gr.update(visible=False),  # guest_chat_warning
                [],                        # chatbot
                current_session_id,        # session_id
                None                       # mic_chat
            )
        else:
            # === 4. INVALID CREDENTIALS ===
            return (
                "**Invalid username or password!**",
                gr.update(visible=True),   # auth_section
                gr.update(visible=False),  # main_app
                "**Not logged in**",
                gr.update(visible=False),  # file_qa_tab
                gr.update(visible=False),  # image_gen_tab
                gr.update(visible=False),  # image_qa_tab
                gr.update(visible=False),  # image_search_tab
                gr.update(visible=False),  # video_gen_tab
                gr.update(visible=False),  # translation_tab ← ADDED
                gr.update(visible=False),  # stats_btn
                [],                        # history_chatbot
                gr.update(visible=True),   # guest_chat_warning
                [],                        # chatbot
                current_session_id,        # session_id
                None                       # mic_chat
            )

    # === 5. EXCEPTION ===
    except Exception as e:
        logging.error(f"Login error: {e}")
        return (
            f"**Login failed:** {str(e)}",
            gr.update(visible=True),   # auth_section
            gr.update(visible=False),  # main_app
            "**Not logged in**",
            gr.update(visible=False),  # file_qa_tab
            gr.update(visible=False),  # image_gen_tab
            gr.update(visible=False),  # image_qa_tab
            gr.update(visible=False),  # image_search_tab
            gr.update(visible=False),  # video_gen_tab
            gr.update(visible=False),  # translation_tab ← ADDED
            gr.update(visible=False),  # stats_btn
            [],                        # history_chatbot
            gr.update(visible=True),   # guest_chat_warning
            [],                        # chatbot
            current_session_id,        # session_id
            None                       # mic_chat
        )


def logout_user():
    """FIXED: Logout with all 15 return values"""
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
        gr.update(visible=True),   # auth_section
        gr.update(visible=False),  # main_app
        "👤 **Not logged in**",    # user_info
        gr.update(visible=False),  # file_qa_tab
        gr.update(visible=False),  # image_gen_tab
        gr.update(visible=False),  # image_qa_tab
        gr.update(visible=False),  # image_search_tab
        gr.update(visible=False),  # video_gen_tab
        gr.update(visible=False),  # translation_tab
        gr.update(visible=False),  # stats_btn
        [],                        # history_chatbot
        gr.update(visible=False),  # guest_chat_warning
        [],                        # chatbot
        current_session_id,        # session_id
        None                       # mic_chat ✅ THIS WAS MISSING!
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




# ============================================================================
# ENHANCED API CONFIGURATION WITH MULTIPLE FALLBACKS
# ============================================================================
GEMINI_API_KEY_1 = os.getenv("GEMINI_API_KEY")  # Primary
GEMINI_API_KEY_2 = os.getenv("GEMINI_API_KEY_2")  # Backup 1
GEMINI_API_KEY_3 = os.getenv("GEMINI_API_KEY_3")  # Backup 2
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Initialize multiple Gemini clients
GEMINI_CLIENTS = []
for key in [GEMINI_API_KEY_1, GEMINI_API_KEY_2, GEMINI_API_KEY_3]:
    if key:
        try:
            import google.generativeai as genai_temp
            genai_temp.configure(api_key=key)
            GEMINI_CLIENTS.append(genai_temp)
            logging.info(f"✅ Gemini client initialized (API key #{len(GEMINI_CLIENTS)})")
        except Exception as e:
            logging.warning(f"Failed to initialize Gemini client: {e}")

GEMINI_CLIENT = GEMINI_CLIENTS[0] if GEMINI_CLIENTS else None

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_CLIENT = genai
        logging.info("🟢 Gemini Client successfully initialized.")
    except Exception as e:
        logging.error(f"❌ Failed to initialize Gemini Client: {e}")
        GEMINI_CLIENT = None
else:
    logging.error("❌ GEMINI_API_KEY is not set in environment variables.")

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

# ================================
#  ANY-TO-ANY TRANSLATION FUNCTIONS
# ================================

def _detect_language(text: str) -> str:
    """Auto-detect source language (fallback = en)"""
    try:
        return detect(text)
    except Exception:
        return "en"


def translate_any_to_any(text: str, target_lang: str):
    """Detect source → translate → speak → return markdown + audio path"""
    if current_user["is_guest"]:
        return ("**Register** to use translation.", None)

    if not text.strip():
        return ("Please enter text to translate.", None)

    src_lang = _detect_language(text)
    logging.info(f"Detected source: {src_lang} → target: {target_lang}")

    translated, audio_path, full_msg = perform_translation(text, target_lang)

    if current_user.get("logged_in"):
        save_translation_history(
            current_user["username"],
            text,
            translated,
            target_lang,
            src_lang
        )

    return full_msg, audio_path


def transcribe_and_translate_any(audio_filepath, target_lang):
    """Voice → transcribe → detect → translate → speak"""
    if current_user["is_guest"]:
        return ("**Register** to use voice translation.", None)

    if not audio_filepath:
        return ("No audio recorded.", None)

    text = transcribe_audio(audio_filepath)
    if text.startswith("Could not") or text.startswith("SpeechRecognition"):
        return (text, None)

    return translate_any_to_any(text, target_lang)

def check_guest_feature_access(feature_name):
    """Check if guest user can access a specific feature"""
    if current_user["is_guest"]:
        return f"🚫 **Guest users can only use Text Chat.** Please login or register to access {feature_name}."
    return None


def query_model(prompt, history, session_id_state):
    """Chat function with MULTIPLE API fallbacks - NO RATE LIMITS!"""
    global current_session_id, GEMINI_CLIENTS

    if session_id_state != current_session_id:
        logging.warning(f"⚠️ Session mismatch detected! Clearing stale history.")
        history = []
        session_id_state = current_session_id

    if not prompt or not prompt.strip():
        return history, "", session_id_state

    limit_check = check_guest_chat_limit()
    if limit_check:
        history.append((prompt, limit_check))
        return history, "", session_id_state

    # REMOVED rate limit check - we have fallbacks now!

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

    answer = None
    llm_name = "N/A"

    # ============================================================
    # METHOD 1: Try all Gemini API keys
    # ============================================================
    for idx, gemini_client in enumerate(GEMINI_CLIENTS, 1):
        if answer:
            break

        try:
            llm_name = f"Gemini (Key #{idx})"
            logging.info(f"🔄 Trying {llm_name}...")

            gemini_formatted_messages = []
            for msg in llm_messages:
                role = "model" if msg["role"] == "assistant" else msg["role"]
                gemini_formatted_messages.append({
                    "role": role,
                    "parts": [{"text": msg["content"]}],
                })

            model = gemini_client.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content(gemini_formatted_messages)

            if response and response.text:
                answer = response.text
                logging.info(f"✅ Success with {llm_name}")
                break

        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str:
                logging.warning(f"⚠️ {llm_name} quota exceeded, trying next...")
            else:
                logging.error(f"❌ {llm_name} error: {e}")
            continue

    # ============================================================
    # METHOD 2: OpenAI GPT-3.5
    # ============================================================
    if not answer and OPENAI_KEY:
        try:
            llm_name = "OpenAI GPT-3.5"
            logging.info(f"🔄 Trying {llm_name}...")

            client = openai.OpenAI(api_key=OPENAI_KEY)
            openai_formatted_messages = []
            for msg in llm_messages:
                role = "assistant" if msg["role"] == "model" else msg["role"]
                openai_formatted_messages.append({"role": role, "content": msg["content"]})

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=openai_formatted_messages,
                timeout=30
            )
            answer = response.choices[0].message.content.strip()
            logging.info(f"✅ Success with {llm_name}")

        except Exception as e:
            logging.error(f"❌ {llm_name} error: {e}")

    # ============================================================
    # METHOD 3: Groq (VERY FAST & FREE)
    # ============================================================
    if not answer and GROQ_API_KEY:
        try:
            llm_name = "Groq Llama-3.1"
            logging.info(f"🔄 Trying {llm_name}...")

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
                logging.info(f"✅ Success with {llm_name}")
            else:
                logging.error(f"❌ {llm_name} status: {response.status_code}")

        except Exception as e:
            logging.error(f"❌ {llm_name} error: {e}")

    # ============================================================
    # METHOD 4: Hugging Face (FREE fallback)
    # ============================================================
    if not answer and HF_API_KEY:
        try:
            llm_name = "HuggingFace Mistral"
            logging.info(f"🔄 Trying {llm_name}...")

            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            hf_models = [
                "mistralai/Mistral-7B-Instruct-v0.2",
                "meta-llama/Meta-Llama-3-8B-Instruct"
            ]

            for model_id in hf_models:
                try:
                    response = requests.post(
                        f"https://api-inference.huggingface.co/models/{model_id}",
                        headers=headers,
                        json={"inputs": prompt},
                        timeout=30
                    )

                    if response.status_code == 200:
                        result = response.json()
                        if isinstance(result, list) and len(result) > 0:
                            answer = result[0].get("generated_text", "")
                            if answer:
                                llm_name = f"HF {model_id.split('/')[-1]}"
                                logging.info(f"✅ Success with {llm_name}")
                                break
                except:
                    continue

        except Exception as e:
            logging.error(f"❌ HuggingFace error: {e}")

    # Final fallback message
    if not answer:
        answer = (
            "⚠️ **All AI services are temporarily unavailable.**\n\n"
            "**Solutions:**\n"
            "1. Wait 30 seconds and try again\n"
            "2. Add more API keys to your .env file:\n"
            "   - `GEMINI_API_KEY_2=your_key` (https://makersuite.google.com/app/apikey)\n"
            "   - `GROQ_API_KEY=your_key` (https://console.groq.com/keys)\n\n"
            "💡 With multiple API keys, you'll never see rate limits!"
        )
        llm_name = "None"

    history.append((prompt, answer))

    if current_user["is_guest"]:
        add_to_guest_history("chat", prompt, answer, {"model": llm_name})
    else:
        save_interaction_to_db("chat", prompt, answer, {"model": llm_name})

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

def on_user_input_detect_translation(user_input, user_is_registered):
    if user_is_registered and user_input.lower().startswith("translate"):
        return gr.update(visible=True), gr.update(value="")
    else:
        return gr.update(visible=False), user_input


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
    """
    COMPLETE FILE EXTRACTION WITH ALL API FALLBACKS
    Order: Gemini (all keys) → Groq → OpenAI Vision → OpenAI GPT-4 → OCR
    """
    guest_check = check_guest_feature_access("File Q&A")
    if guest_check:
        yield "🔒 Access Denied", guest_check
        return

    if not file:
        yield "⏱️ 0s", "Error: No file uploaded."
        return

    uploaded_file = None
    start_time = time.time()

    try:
        yield "⏱️ 1s", "⏳ **Step 1:** Uploading file..."
        file_path = file.name
        ext = os.path.splitext(file_path)[-1].lower()

        if ext in ['.docx', '.txt']:
            yield "⏱️ 2s", "Error: Unsupported file type. Please use PDF or images (JPG, PNG)."
            return

        # ============================================================
        # METHOD 1: Try ALL Gemini API Keys (Primary - Best for documents)
        # ============================================================
        for idx, gemini_client in enumerate(GEMINI_CLIENTS, 1):
            try:
                logging.info(f"🔄 Trying Gemini API #{idx} for file extraction...")
                elapsed = int(time.time() - start_time)
                yield f"⏱️ {elapsed}s", f"⏳ **Step 2/{len(GEMINI_CLIENTS) + 2}:** Processing with Gemini API #{idx}..."

                # Upload file to Gemini
                uploaded_file = genai.upload_file(path=file_path)

                extraction_prompt = (
                    f"Analyze the attached document/image thoroughly. Extract ALL text, tables, "
                    f"charts, and key data. Format the output as clean Markdown with proper headers. "
                    f"User's specific request: '{prompt}'"
                )

                contents = [uploaded_file, extraction_prompt]
                model = gemini_client.GenerativeModel('gemini-2.0-flash-exp')
                response = model.generate_content(contents)

                if response and response.text:
                    elapsed = int(time.time() - start_time)
                    final_result = f"**✅ Extraction Complete! (took {elapsed}s)**\n\n**Service:** Gemini API #{idx}\n\n---\n\n{response.text}"

                    # Cleanup
                    try:
                        genai.delete_file(name=uploaded_file.name)
                        logging.info(f"✅ File cleaned up from Gemini")
                    except:
                        pass

                    yield f"⏱️ {elapsed}s ✅", final_result
                    return

            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "quota" in error_str or "resource exhausted" in error_str:
                    logging.warning(f"⚠️ Gemini #{idx} quota exceeded, trying next API...")
                else:
                    logging.error(f"❌ Gemini #{idx} error: {e}")

                # Cleanup on error
                if uploaded_file and hasattr(uploaded_file, 'name'):
                    try:
                        genai.delete_file(name=uploaded_file.name)
                    except:
                        pass
                uploaded_file = None
                continue

        # All Gemini keys failed - proceed to next method
        elapsed = int(time.time() - start_time)
        yield f"⏱️ {elapsed}s", f"⏳ All {len(GEMINI_CLIENTS)} Gemini keys exhausted. Trying Groq..."

        # ============================================================
        # METHOD 2: Groq API (Fast & Free - Works with extracted text)
        # ============================================================
        if GROQ_API_KEY:
            try:
                logging.info("🔄 Trying Groq API for file extraction...")
                elapsed = int(time.time() - start_time)
                yield f"⏱️ {elapsed}s", "⏳ **Fallback 1:** Extracting with Groq AI..."

                # First, extract raw text from file
                extracted_text = ""

                # For images - use OCR
                if ext in ['.jpg', '.jpeg', '.png', '.webp']:
                    try:
                        import pytesseract
                        from PIL import Image
                        img = Image.open(file_path)
                        extracted_text = pytesseract.image_to_string(img)
                        logging.info(f"✅ OCR extracted {len(extracted_text)} characters from image")
                    except ImportError:
                        logging.warning("⚠️ pytesseract not installed. Install with: pip install pytesseract")
                    except Exception as ocr_error:
                        logging.warning(f"⚠️ OCR failed: {ocr_error}")

                # For PDFs - extract text
                elif ext == '.pdf':
                    try:
                        import PyPDF2
                        with open(file_path, 'rb') as pdf_file:
                            pdf_reader = PyPDF2.PdfReader(pdf_file)
                            for page in pdf_reader.pages:
                                extracted_text += page.extract_text() + "\n\n"
                        logging.info(f"✅ Extracted {len(extracted_text)} characters from PDF")
                    except ImportError:
                        logging.warning("⚠️ PyPDF2 not installed. Install with: pip install PyPDF2")
                    except Exception as pdf_error:
                        logging.warning(f"⚠️ PDF extraction failed: {pdf_error}")

                # If we successfully extracted text, analyze with Groq
                if extracted_text.strip():
                    groq_prompt = f"""You are a document analysis expert. Analyze this extracted content and provide a comprehensive response.

**Extracted Content:**
{extracted_text[:6000]}  # Limit for API

**User's Request:** {prompt}

**Instructions:**
- Extract key information
- Format as clean Markdown
- Be comprehensive and accurate
- Address the user's specific request"""

                    response = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {GROQ_API_KEY}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "llama-3.1-8b-instant",
                            "messages": [{"role": "user", "content": groq_prompt}],
                            "max_tokens": 3000,
                            "temperature": 0.3
                        },
                        timeout=45
                    )

                    if response.status_code == 200:
                        result = response.json()["choices"][0]["message"]["content"]
                        elapsed = int(time.time() - start_time)
                        final_result = f"**✅ Extraction Complete! (took {elapsed}s)**\n\n**Service:** Groq AI (Llama 3.1)\n\n---\n\n{result}"
                        yield f"⏱️ {elapsed}s ✅", final_result
                        return
                    else:
                        logging.warning(f"⚠️ Groq API returned status {response.status_code}")
                else:
                    logging.warning("⚠️ No text extracted for Groq to analyze")

            except Exception as e:
                logging.error(f"❌ Groq API error: {e}")

        # Groq failed - proceed to OpenAI
        elapsed = int(time.time() - start_time)
        yield f"⏱️ {elapsed}s", "⏳ Groq unavailable. Trying OpenAI Vision..."

        # ============================================================
        # METHOD 3: OpenAI Vision API (Premium - Best for images)
        # ============================================================
        if OPENAI_KEY and ext in ['.jpg', '.jpeg', '.png', '.webp']:
            try:
                logging.info("🔄 Trying OpenAI Vision API...")
                elapsed = int(time.time() - start_time)
                yield f"⏱️ {elapsed}s", "⏳ **Fallback 2:** Processing with OpenAI Vision..."

                import base64
                with open(file_path, "rb") as f:
                    file_data = base64.b64encode(f.read()).decode('utf-8')

                client = openai.OpenAI(api_key=OPENAI_KEY)

                mime_types = {
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.png': 'image/png',
                    '.webp': 'image/webp'
                }
                mime_type = mime_types.get(ext, 'image/jpeg')

                response = client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Analyze this image/document thoroughly. Extract all text, data, and information. Format as Markdown. User's request: {prompt}"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{file_data}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }],
                    max_tokens=4096
                )

                result = response.choices[0].message.content
                elapsed = int(time.time() - start_time)
                final_result = f"**✅ Extraction Complete! (took {elapsed}s)**\n\n**Service:** OpenAI Vision (GPT-4V)\n\n---\n\n{result}"
                yield f"⏱️ {elapsed}s ✅", final_result
                return

            except Exception as e:
                logging.error(f"❌ OpenAI Vision error: {e}")

        # OpenAI Vision failed - try GPT-4 with text extraction
        elapsed = int(time.time() - start_time)
        yield f"⏱️ {elapsed}s", "⏳ OpenAI Vision unavailable. Trying GPT-4 with text extraction..."

        # ============================================================
        # METHOD 4: OpenAI GPT-4 with Text Extraction
        # ============================================================
        if OPENAI_KEY:
            try:
                logging.info("🔄 Trying OpenAI GPT-4 with text extraction...")
                elapsed = int(time.time() - start_time)
                yield f"⏱️ {elapsed}s", "⏳ **Fallback 3:** Extracting text + GPT-4 analysis..."

                # Extract text content
                file_content = ""

                # For PDFs
                if ext == '.pdf':
                    try:
                        import PyPDF2
                        with open(file_path, 'rb') as pdf_file:
                            pdf_reader = PyPDF2.PdfReader(pdf_file)
                            for page in pdf_reader.pages:
                                file_content += page.extract_text() + "\n\n"
                        logging.info(f"✅ Extracted {len(file_content)} chars from PDF")
                    except Exception as pdf_error:
                        logging.warning(f"PDF extraction failed: {pdf_error}")

                # For images
                elif ext in ['.jpg', '.jpeg', '.png', '.webp']:
                    try:
                        import pytesseract
                        from PIL import Image
                        img = Image.open(file_path)
                        file_content = pytesseract.image_to_string(img)
                        logging.info(f"✅ OCR extracted {len(file_content)} chars")
                    except Exception as ocr_error:
                        logging.warning(f"OCR failed: {ocr_error}")

                # If we have content, analyze with GPT-4
                if file_content.strip():
                    client = openai.OpenAI(api_key=OPENAI_KEY)

                    analysis_prompt = f"""Analyze this extracted document content and provide a comprehensive response.

**Extracted Content:**
{file_content[:10000]}

**User's Request:** {prompt}

**Instructions:**
- Provide detailed analysis
- Format as clean Markdown
- Address the user's specific request"""

                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": analysis_prompt}],
                        max_tokens=3000
                    )

                    result = response.choices[0].message.content
                    elapsed = int(time.time() - start_time)
                    final_result = f"**✅ Extraction Complete! (took {elapsed}s)**\n\n**Service:** OpenAI GPT-4 + Text Extraction\n\n---\n\n{result}"
                    yield f"⏱️ {elapsed}s ✅", final_result
                    return

            except Exception as e:
                logging.error(f"❌ OpenAI GPT-4 error: {e}")

        # ============================================================
        # METHOD 5: Basic OCR Fallback (No AI analysis)
        # ============================================================
        if ext in ['.jpg', '.jpeg', '.png', '.webp']:
            try:
                logging.info("🔄 Trying basic OCR extraction...")
                elapsed = int(time.time() - start_time)
                yield f"⏱️ {elapsed}s", "⏳ **Final Fallback:** Basic OCR extraction..."

                import pytesseract
                from PIL import Image

                img = Image.open(file_path)
                text = pytesseract.image_to_string(img)

                if text.strip():
                    elapsed = int(time.time() - start_time)
                    final_result = f"""**✅ Text Extracted! (took {elapsed}s)**

**Service:** Basic OCR (Tesseract)

**⚠️ Note:** This is raw OCR output without AI analysis. For better results:
1. Add more Gemini API keys: https://makersuite.google.com/app/apikey
2. Get Groq API key: https://console.groq.com/keys
3. Add OpenAI API key for premium analysis

---

**Extracted Text:**

{text}"""
                    yield f"⏱️ {elapsed}s ✅", final_result
                    return

            except ImportError:
                logging.warning("⚠️ pytesseract not installed")
            except Exception as e:
                logging.error(f"❌ OCR extraction error: {e}")

        # ============================================================
        # ALL METHODS FAILED - Show comprehensive error
        # ============================================================
        elapsed = int(time.time() - start_time)

        # Detailed status of what failed
        failure_report = []
        if len(GEMINI_CLIENTS) > 0:
            failure_report.append(f"❌ {len(GEMINI_CLIENTS)} Gemini API key(s) - quota exceeded")
        else:
            failure_report.append("⚠️ No Gemini API keys configured")

        if GROQ_API_KEY:
            failure_report.append("❌ Groq API - failed or no text extracted")
        else:
            failure_report.append("⚠️ No Groq API key configured")

        if OPENAI_KEY:
            failure_report.append("❌ OpenAI APIs - failed")
        else:
            failure_report.append("⚠️ No OpenAI API key configured")

        error_msg = f"""⚠️ **All file extraction services temporarily unavailable**

**Attempted Services ({elapsed}s):**
{chr(10).join(failure_report)}

---

**🔧 SOLUTIONS TO FIX THIS:**

**Option 1: Add More FREE Gemini Keys (5 min - RECOMMENDED)**
→ Get 2-3 more keys at: https://makersuite.google.com/app/apikey
→ Add to .env:
```
GEMINI_API_KEY_2=AIzaSy...your_key
GEMINI_API_KEY_3=AIzaSy...your_key
```
→ Each key = 1,500 requests/day

**Option 2: Add FREE Groq API (2 min - FAST)**
→ Get key at: https://console.groq.com/keys
→ Add to .env:
```
GROQ_API_KEY=gsk_...your_key
```
→ Unlimited requests (rate limited but generous)

**Option 3: Add OpenAI API (Premium)**
→ Get key at: https://platform.openai.com/api-keys
→ Add to .env:
```
OPENAI_API_KEY=sk-...your_key
```

**Option 4: Install OCR for basic extraction**
```bash
pip install pytesseract
pip install PyPDF2
```

**Option 5: Wait 60 seconds**
→ Gemini quotas reset per minute
→ Try uploading again

---

**Current Configuration:**
- Gemini keys: {len(GEMINI_CLIENTS)}/3
- Groq API: {'✅ Configured' if GROQ_API_KEY else '❌ Missing'}
- OpenAI API: {'✅ Configured' if OPENAI_KEY else '❌ Missing'}

**💡 With 3 Gemini keys + Groq, you get ~5,000 FREE extractions/day!**
"""
        yield f"⏱️ {elapsed}s ❌", error_msg

    except Exception as e:
        elapsed = int(time.time() - start_time)
        logging.error(f"File extraction critical error: {e}")
        import traceback
        traceback.print_exc()
        yield f"⏱️ {elapsed}s ❌", f"**Critical Error:** {str(e)}\n\nPlease check logs for details."

    finally:
        # Final cleanup of any remaining uploaded files
        if uploaded_file and hasattr(uploaded_file, 'name'):
            try:
                genai.delete_file(name=uploaded_file.name)
                logging.info(f"✅ Final cleanup: File deleted from Gemini")
            except Exception as cleanup_error:
                logging.debug(f"Cleanup note: {cleanup_error}")

def answer_question_from_content(file_content, user_question):
    """Use multiple LLMs to answer questions about file content"""

    max_content_length = 30000
    if len(file_content) > max_content_length:
        file_content = file_content[:max_content_length] + "\n\n[Content truncated...]"

    system_prompt = f"""You are a highly capable AI assistant.

**File Content:**
{file_content}

**User's Request:**
{user_question}

Provide a helpful response based on the file content."""

    # Try all Gemini clients
    for idx, gemini_client in enumerate(GEMINI_CLIENTS, 1):
        try:
            logging.info(f"🔄 Trying Gemini #{idx} for file Q&A...")
            model = gemini_client.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content(system_prompt)

            if response and response.text:
                return f"**Response:**\n\n{response.text}\n\n---\n**Source:** Gemini #{idx}"
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                continue
            else:
                logging.error(f"Gemini #{idx} error: {e}")
                continue

    # Try OpenAI
    if OPENAI_KEY:
        try:
            logging.info("🔄 Trying OpenAI...")
            client = openai.OpenAI(api_key=OPENAI_KEY)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": system_prompt}],
                timeout=60
            )
            answer = response.choices[0].message.content.strip()
            return f"**Response:**\n\n{answer}\n\n---\n**Source:** OpenAI"
        except Exception as e:
            logging.error(f"OpenAI error: {e}")

    # Try Groq
    if GROQ_API_KEY:
        try:
            logging.info("🔄 Trying Groq...")
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": system_prompt}]},
                timeout=60
            )

            if response.status_code == 200:
                answer = response.json()["choices"][0]["message"]["content"]
                return f"**Response:**\n\n{answer}\n\n---\n**Source:** Groq"
        except Exception as e:
            logging.error(f"Groq error: {e}")

    return (
        f"**Extracted Content:**\n\n{file_content}\n\n---\n\n"
        f"⚠️ All AI services unavailable. Add more API keys to your .env file."
    )

def file_question_answer(file, question):
    """FIXED: File Q&A with proper generator handling"""
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

    # Track the last timer and content
    last_timer = "⏱️ 0s"
    last_content = ""

    # Extract content
    for timer, content in extract_file_content_gemini(file, question):
        last_timer = timer
        last_content = content
        yield timer, content  # Stream updates

    # Check for extraction errors
    if last_content and (
            last_content.lower().startswith("error:") or
            last_content.lower().startswith("extraction failed:")
    ):
        # Save error to history
        if current_user["is_guest"]:
            add_to_guest_history("file_qa", question, last_content,
                                 {"filename": file.name if file else "unknown"})
        else:
            save_interaction_to_db("file_qa", question, last_content,
                                   {"filename": file.name if file else "unknown"})
        return  # Exit on error

    # Process successful extraction
    if last_content and "✅ Extraction Complete!" in last_content:
        # Extract just the content (remove status message)
        content_parts = last_content.split("\n\n", 1)
        extracted_content = content_parts[1] if len(content_parts) > 1 else last_content

        yield last_timer, "⏳ **Generating AI response based on extracted content...**"

        # Generate AI response
        result = answer_question_from_content(extracted_content, question)

        # Save to history
        if current_user["is_guest"]:
            add_to_guest_history("file_qa", question, result,
                                 {"filename": file.name if file else "unknown"})
        else:
            save_interaction_to_db("file_qa", question, result,
                                   {"filename": file.name if file else "unknown"})

        # CRITICAL: Final yield with complete result
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
    """Image Q&A with MULTIPLE API fallbacks"""
    guest_check = check_guest_feature_access("Image Q&A")
    if guest_check:
        return guest_check

    if image is None:
        return "Error: Please upload an image first."

    increment_usage("image_qa")

    # ============================================================
    # Try all Gemini APIs
    # ============================================================
    for idx, gemini_client in enumerate(GEMINI_CLIENTS, 1):
        try:
            logging.info(f"🔄 Trying Gemini #{idx} for image Q&A...")

            resized_image = image.copy()
            resized_image.thumbnail((512, 512), Resampling.LANCZOS)

            contents = [resized_image, prompt]
            model = gemini_client.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content(contents)

            if response and response.text:
                result = response.text

                if current_user["is_guest"]:
                    add_to_guest_history("image_qa", prompt, result, {"model": f"Gemini #{idx}"})
                else:
                    save_interaction_to_db("image_qa", prompt, result, {"model": f"Gemini #{idx}"})

                logging.info(f"✅ Success with Gemini #{idx}")
                return result

        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str:
                logging.warning(f"⚠️ Gemini #{idx} quota exceeded, trying next...")
                continue
            else:
                logging.error(f"❌ Gemini #{idx} error: {e}")
                continue

    # All methods failed
    error_msg = (
        "⚠️ **All image analysis services temporarily unavailable.**\n\n"
        "**Solutions:**\n"
        "1. Add backup API keys: `GEMINI_API_KEY_2`, `GEMINI_API_KEY_3`\n"
        "2. Wait 30-60 seconds\n\n"
        "Get free keys at: https://makersuite.google.com/app/apikey"
    )

    if not current_user["is_guest"]:
        save_interaction_to_db("image_qa", prompt, error_msg, {"error": "all_services_failed"})

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
def install_package(package):
    """Install a package if not already installed"""
    try:
        __import__(package)
        return True
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            return True
        except Exception as e:
            logging.error(f"Failed to install {package}: {e}")
            return False


def generate_video_huggingface_svd_fixed(prompt: str):
    """
    FIXED: Generate actual video using Hugging Face
    Uses the correct working model endpoint
    """
    if not HF_API_KEY:
        return None, "Hugging Face API key not configured."

    try:
        logging.info("🎬 Generating real video with Hugging Face (FREE)...")
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}

        # Step 1: Generate initial frame
        logging.info("🎨 Step 1/2: Generating initial frame...")
        img_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
        img_response = requests.post(
            img_url,
            headers=headers,
            json={"inputs": prompt},
            timeout=120
        )

        if img_response.status_code != 200:
            logging.warning(f"Initial image generation failed: {img_response.status_code}")
            return None, f"Failed to generate starting frame: {img_response.status_code}"

        initial_image = Image.open(io.BytesIO(img_response.content))
        initial_image = initial_image.resize((1024, 576), Resampling.LANCZOS)

        # Save as bytes for video generation
        img_bytes = io.BytesIO()
        initial_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        logging.info("✅ Initial frame generated. Now creating video with motion...")

        # Step 2: Try multiple working video generation models
        video_models = [
            "stabilityai/stable-video-diffusion-img2vid-xt",
            "ali-vilab/i2vgen-xl",
            "damo-vilab/text-to-video-ms-1.7b"
        ]

        for model in video_models:
            try:
                logging.info(f"🎞️ Trying {model}...")
                video_url = f"https://api-inference.huggingface.co/models/{model}"

                video_response = requests.post(
                    video_url,
                    headers=headers,
                    data=img_bytes.getvalue(),
                    timeout=180
                )

                if video_response.status_code == 200:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                        tmp.write(video_response.content)
                        video_path = tmp.name

                    file_size = Path(video_path).stat().st_size / 1024
                    logging.info(f"✅ Real video generated with {model}! Size: {file_size:.1f} KB")
                    return video_path, f"✅ Real video with motion generated! (Size: {file_size:.1f} KB)"

                elif video_response.status_code == 503:
                    logging.warning(f"{model} is loading (503)...")
                    continue
                else:
                    logging.warning(f"{model} failed: {video_response.status_code}")
                    continue

            except Exception as e:
                logging.warning(f"{model} error: {e}")
                continue

        # All HF models failed
        return None, "⏳ Hugging Face video models are loading. Try again in 20-30 seconds."

    except Exception as e:
        logging.error(f"Hugging Face video generation failed: {e}")
        return None, f"❌ Video generation failed: {str(e)}"


def generate_video_animatediff_hf(prompt: str):
    """
    Alternative: Generate video using AnimateDiff on Hugging Face (FREE)
    This is a direct text-to-video model
    """
    if not HF_API_KEY:
        return None, "Hugging Face API key not configured."

    try:
        logging.info("🎬 Generating video with AnimateDiff (Hugging Face)...")
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}

        # Direct text-to-video generation
        video_url = "https://api-inference.huggingface.co/models/guoyww/animatediff-motion-adapter-v1-5-2"

        response = requests.post(
            video_url,
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {
                    "num_frames": 16,
                    "num_inference_steps": 25
                }
            },
            timeout=180
        )

        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(response.content)
                video_path = tmp.name

            file_size = Path(video_path).stat().st_size / 1024
            return video_path, f"✅ Video generated with AnimateDiff! (Size: {file_size:.1f} KB)"

        elif response.status_code == 503:
            return None, "⏳ AnimateDiff model is loading (20-30s). Please try again."
        else:
            logging.error(f"AnimateDiff failed: {response.status_code}")
            return None, f"AnimateDiff failed: {response.status_code}"

    except Exception as e:
        logging.error(f"AnimateDiff HF failed: {e}")
        return None, f"❌ AnimateDiff failed: {str(e)}"


def generate_video_text2video_zero(prompt: str):
    """
    Generate video using Text2Video-Zero (100% FREE, works well)
    """
    if not HF_API_KEY:
        return None, "Hugging Face API key not configured."

    try:
        logging.info("🎬 Generating video with Text2Video-Zero...")
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}

        video_url = "https://api-inference.huggingface.co/models/damo-vilab/text-to-video-ms-1.7b"

        response = requests.post(
            video_url,
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {
                    "num_frames": 16
                }
            },
            timeout=180
        )

        if response.status_code == 200:
            # Response might be a video file or JSON with URL
            content_type = response.headers.get('content-type', '')

            if 'video' in content_type or len(response.content) > 100000:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(response.content)
                    video_path = tmp.name

                file_size = Path(video_path).stat().st_size / 1024
                return video_path, f"✅ Video generated with Text2Video! (Size: {file_size:.1f} KB)"
            else:
                # Try to parse JSON response
                try:
                    result = response.json()
                    video_url = result.get('video_url') or result.get('url')
                    if video_url:
                        vid_response = requests.get(video_url, timeout=60)
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                            tmp.write(vid_response.content)
                            video_path = tmp.name
                        return video_path, "✅ Video generated with Text2Video!"
                except:
                    pass

                return None, "Unexpected response format from Text2Video"

        elif response.status_code == 503:
            return None, "⏳ Text2Video model is loading. Try again in 20-30s."
        else:
            return None, f"Text2Video failed: {response.status_code}"

    except Exception as e:
        logging.error(f"Text2Video failed: {e}")
        return None, f"❌ Text2Video failed: {str(e)}"


def generate_video_replicate_zeroscope(prompt: str):
    """
    Generate video using Replicate's Zeroscope model
    Requires REPLICATE_API_TOKEN and billing credits
    """
    if not REPLICATE_AVAILABLE or not replicate:
        return None, "Replicate not configured or library not installed."

    try:
        logging.info("🎬 Generating video with Replicate Zeroscope...")

        output = replicate.run(
            "anotherjesse/zeroscope-v2-xl:9f747673945c62801b13b84701c783929c0ee784e4748ec062204894dda1a351",
            input={
                "prompt": prompt,
                "num_frames": 24,
                "num_inference_steps": 50,
            }
        )

        # Download the video
        if output:
            video_url = output if isinstance(output, str) else output[0]
            response = requests.get(video_url, timeout=60)

            if response.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(response.content)
                    video_path = tmp.name

                file_size = Path(video_path).stat().st_size / 1024
                return video_path, f"✅ Video generated with Zeroscope! (Size: {file_size:.1f} KB)"

        return None, "Failed to retrieve video from Replicate"

    except Exception as e:
        logging.error(f"Replicate Zeroscope failed: {e}")
        return None, f"❌ Replicate failed: {str(e)}"


def generate_simple_video_locally_enhanced(prompt: str, duration=3, fps=10):
    """
    LOCAL FALLBACK: Generate a simple animated video using images
    This creates a zoom/pan effect on a generated or placeholder image
    """
    if not IMAGEIO_AVAILABLE:
        return None, (
            "❌ Local video generation unavailable.\n\n"
            "**Install required packages:**\n"
            "```bash\n"
            "pip install imageio imageio-ffmpeg numpy\n"
            "```"
        )

    try:
        logging.info("🎨 Creating local animated video...")

        # Try to generate an image first
        base_image = None

        # Method 1: Try Pollinations
        try:
            encoded_prompt = quote(prompt)
            url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=576&nologo=true"
            response = requests.get(url, timeout=30)
            if response.status_code == 200 and len(response.content) > 5000:
                base_image = Image.open(io.BytesIO(response.content))
                logging.info("✅ Got base image from Pollinations")
        except Exception as e:
            logging.warning(f"Pollinations failed for video base: {e}")

        # Method 2: Create a placeholder if image generation fails
        if base_image is None:
            logging.info("Creating placeholder image for animation...")
            base_image = Image.new('RGB', (1024, 576), color=(30, 30, 50))
            draw = ImageDraw.Draw(base_image)

            # Try to use a nice font, fallback to default
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
            except:
                font = ImageFont.load_default()

            # Wrap text
            words = prompt.split()
            lines = []
            current_line = []
            for word in words:
                current_line.append(word)
                if len(' '.join(current_line)) > 40:
                    lines.append(' '.join(current_line[:-1]))
                    current_line = [word]
            if current_line:
                lines.append(' '.join(current_line))

            # Draw text centered
            y_offset = 200
            for line in lines[:4]:  # Max 4 lines
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                x = (1024 - text_width) // 2
                draw.text((x, y_offset), line, fill=(255, 255, 255), font=font)
                y_offset += 60

        # Create animation frames with zoom effect
        frames = []
        total_frames = duration * fps

        for i in range(total_frames):
            # Calculate zoom factor (1.0 to 1.3)
            zoom = 1.0 + (i / total_frames) * 0.3

            # Calculate new size
            new_width = int(1024 * zoom)
            new_height = int(576 * zoom)

            # Resize image
            zoomed = base_image.resize((new_width, new_height), Resampling.LANCZOS)

            # Crop to original size (centered)
            left = (new_width - 1024) // 2
            top = (new_height - 576) // 2
            cropped = zoomed.crop((left, top, left + 1024, top + 576))

            # Convert to numpy array
            frames.append(np.array(cropped))

        # Save as video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            video_path = tmp.name

        imageio.mimsave(video_path, frames, fps=fps, codec='libx264', quality=8)

        file_size = Path(video_path).stat().st_size / 1024

        return video_path, (
            f"✅ Local animated video created! (Size: {file_size:.1f} KB)\n\n"
            f"⚠️ This is a fallback animation since cloud services are unavailable.\n\n"
            f"**For better results, try:**\n"
            f"- Waiting 20-30 seconds (HF models need warm-up)\n"
            f"- Using external services: Pika Labs, RunwayML Gen-2"
        )

    except Exception as e:
        logging.error(f"Local video generation failed: {e}")
        return None, f"❌ Local video generation failed: {str(e)}"

# ============================================================================
# UPDATED generate_video() - Replace your current one with this
# ============================================================================
def generate_video(prompt: str):
    """
    MAIN VIDEO GENERATION with working FREE services
    """
    access_check = check_feature_access("video_gen")
    if access_check:
        yield access_check, None
        return

    limit_msg = check_rate_limit("video_gen")
    if limit_msg:
        yield limit_msg, None
        return

    if not prompt or not prompt.strip():
        yield "❌ Please enter a video prompt.", None
        return

    increment_usage("video_gen")

    # Method 1: Text2Video-Zero (Hugging Face - FREE & Works!)
    if HF_API_KEY:
        yield "⏳ Generating real video with Text2Video-Zero (FREE)...", None
        video_path, status = generate_video_text2video_zero(prompt)
        if video_path:
            if current_user["is_guest"]:
                add_to_guest_history("video_gen", prompt, status, {"service": "Text2Video-Zero"})
            else:
                save_interaction_to_db("video_gen", prompt, status, {"service": "Text2Video-Zero"})
            yield status, video_path
            return
        else:
            yield f"⚠️ Text2Video: {status}. Trying AnimateDiff...", None

        # Method 2: AnimateDiff (Hugging Face - FREE)
        yield "⏳ Trying AnimateDiff (FREE)...", None
        video_path, status = generate_video_animatediff_hf(prompt)
        if video_path:
            if current_user["is_guest"]:
                add_to_guest_history("video_gen", prompt, status, {"service": "AnimateDiff HF"})
            else:
                save_interaction_to_db("video_gen", prompt, status, {"service": "AnimateDiff HF"})
            yield status, video_path
            return
        else:
            yield f"⚠️ AnimateDiff: {status}. Trying Stable Video Diffusion...", None

        # Method 3: Stable Video Diffusion (Hugging Face - FREE)
        yield "⏳ Trying Stable Video Diffusion (FREE)...", None
        video_path, status = generate_video_huggingface_svd_fixed(prompt)
        if video_path:
            if current_user["is_guest"]:
                add_to_guest_history("video_gen", prompt, status, {"service": "Stable Video Diffusion"})
            else:
                save_interaction_to_db("video_gen", prompt, status, {"service": "Stable Video Diffusion"})
            yield status, video_path
            return
        else:
            yield f"⚠️ SVD: {status}. Falling back to local animation...", None

    # Method 4: Replicate (if configured with credits)
    if REPLICATE_AVAILABLE:
        yield "⏳ Trying Replicate services...", None
        video_path, status = generate_video_replicate_zeroscope(prompt)
        if video_path:
            if current_user["is_guest"]:
                add_to_guest_history("video_gen", prompt, status, {"service": "Zeroscope"})
            else:
                save_interaction_to_db("video_gen", prompt, status, {"service": "Zeroscope"})
            yield status, video_path
            return

    # Method 5: Local fallback (animated image)
    yield "⏳ Creating animated video (local fallback)...", None
    video_path, status = generate_simple_video_locally_enhanced(prompt, duration=3, fps=10)

    if video_path:
        if current_user["is_guest"]:
            add_to_guest_history("video_gen", prompt, status, {"service": "Local Animation"})
        else:
            save_interaction_to_db("video_gen", prompt, status, {"service": "Local Animation"})
        yield status, video_path
        return

    # All methods failed
    error_msg = (
        "❌ **All video generation services temporarily unavailable.**\n\n"
        "**What happened:**\n"
        "- Hugging Face models are loading (cold start)\n"
        "- Replicate needs billing credit\n\n"
        "**Solution:**\n"
        "🔄 **Wait 20-30 seconds** and try again - HF models will be ready!\n\n"
        "**Or use these FREE online services:**\n"
        "- Pika Labs: https://pika.art (best quality, instant)\n"
        "- RunwayML Gen-2: https://app.runwayml.com (free tier)\n"
        "- Stable Video: https://huggingface.co/spaces/stabilityai/stable-video-diffusion"
    )

    if not current_user["is_guest"]:
        save_interaction_to_db("video_gen", prompt, "Failed - models loading", {"error": "cold_start"})

    yield error_msg, None


# Wrapper function for Gradio (handles generator properly)
def video_gen_wrapper(prompt):
    """
    FIXED: Proper generator handling for video generation
    """
    final_status = ""
    final_video = None

    try:
        # Iterate through generator
        for status, video in generate_video(prompt):
            final_status = status
            final_video = video
            yield status, video  # Stream updates to UI

        # CRITICAL: Ensure final values are yielded
        if final_status or final_video:
            yield final_status, final_video
        else:
            # Fallback if generator produced nothing
            yield "❌ Video generation failed - no output", None

    except Exception as e:
        logging.error(f"Video generation error in wrapper: {e}")
        yield f"❌ Error: {str(e)}", None



# Audio processing for video generation
def process_audio_for_video_gen(audio_filepath):
    """
    Process audio input for video generation
    """
    guest_check = check_guest_feature_access("Video Generation")
    if guest_check:
        yield guest_check, None
        return

    if audio_filepath is None:
        yield "", None
        return

    transcribed_text = transcribe_audio(audio_filepath)

    if transcribed_text.startswith("❌"):
        yield transcribed_text, None
        return

    # Use the generator
    for status, video in generate_video(transcribed_text):
        yield status, video


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


# ================================
# GLOBAL STATE FOR FIREBASE LOGIN
# ================================
firebase_login_result = None  # Stores the 16-tuple result from login_firebase_user()


# ================================
# GRADIO CHECKER FOR FIREBASE LOGIN
# ================================
def check_firebase_login_and_update():
    """Check if Firebase login completed and return UI updates"""
    global firebase_login_result

    if firebase_login_result is not None:
        result = firebase_login_result
        firebase_login_result = None
        logging.info("🔥 Firebase login detected! Updating Gradio UI...")
        return result

    return tuple([gr.update() for _ in range(16)])

# ================================
# FLASK ROUTE FOR FIREBASE AUTH PAGE
# ================================
@flask_app.route("/firebase-auth", methods=["GET"])
def firebase_auth_page():
    """Serve Firebase auth page with action parameter - NO visible logs"""
    action = request.args.get('action', 'login')

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Signing {action}...</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 600px;
            margin: 50px auto;
            padding: 20px;
            text-align: center;
        }}
        #google-btn {{
            background: #4285f4;
            color: white;
            border: none;
            padding: 15px 30px;
            font-size: 16px;
            border-radius: 8px;
            cursor: pointer;
        }}
        #google-btn:hover {{
            background: #357ae8;
        }}
        .spinner {{
            border: 4px solid #f3f3f3;
            border-top: 4px solid #4285f4;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
            display: none;
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        #status {{
            margin-top: 20px;
            padding: 15px;
            border-radius: 5px;
            display: none;
        }}
        .success {{ 
            background: #d4edda; 
            border: 1px solid #c3e6cb;
            display: block;
        }}
        .error {{ 
            background: #f8d7da; 
            border: 1px solid #f5c6cb;
            display: block;
        }}
    </style>
</head>
<body>
    <h2>🔐 Sign {action} with Google</h2>
    <button id="google-btn">Continue with Google</button>
    <div class="spinner" id="spinner"></div>
    <div id="status"></div>

    <script type="module">
        const statusDiv = document.getElementById('status');
        const spinner = document.getElementById('spinner');
        const btn = document.getElementById('google-btn');
        const action = "{action}";
        const config = {firebase_config_json};

        if (!config.apiKey) {{
            statusDiv.textContent = '❌ Firebase not configured';
            statusDiv.className = 'error';
        }} else {{
            Promise.all([
                import('https://www.gstatic.com/firebasejs/10.7.1/firebase-app.js'),
                import('https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js')
            ]).then(([appModule, authModule]) => {{
                const app = appModule.initializeApp(config);
                const auth = authModule.getAuth(app);
                const provider = new authModule.GoogleAuthProvider();
                provider.setCustomParameters({{ prompt: 'select_account' }});

                btn.addEventListener('click', async () => {{
                    btn.style.display = 'none';
                    spinner.style.display = 'block';

                    try {{
                        const result = await authModule.signInWithPopup(auth, provider);
                        const token = await result.user.getIdToken(true);

                        const response = await fetch('http://127.0.0.1:5000/api/firebase-login', {{
                            method: 'POST',
                            headers: {{ 'Content-Type': 'application/json' }},
                            body: JSON.stringify({{ token: token, action: action }})
                        }});

                        const data = await response.json();

                        if (data.success) {{
                            const actionText = action === 'register' ? 'Registered' : 'Logged in';
                            statusDiv.textContent = '✅ SUCCESS! ' + actionText + ' as ' + result.user.email;
                            statusDiv.className = 'success';
                            spinner.style.display = 'none';

                            alert('🎉 ' + actionText + ' successfully!\\n\\nYou can now:\\n1. Close this window\\n2. Go back to main app\\n3. Refresh the page (F5)');
                            setTimeout(() => window.close(), 2000);
                        }} else {{
                            throw new Error(data.error || 'Backend returned success=false');
                        }}
                    }} catch (error) {{
                        spinner.style.display = 'none';
                        btn.style.display = 'block';
                        statusDiv.textContent = '❌ Error: ' + error.message;
                        statusDiv.className = 'error';
                    }}
                }});
            }}).catch(error => {{
                btn.style.display = 'none';
                statusDiv.textContent = '❌ Failed to load Firebase';
                statusDiv.className = 'error';
            }});
        }}
    </script>
</body>
</html>
"""
    return html


@flask_app.route("/api/firebase-login", methods=["POST", "OPTIONS"])
def firebase_login_endpoint():
    """
    Handle Firebase authentication - SUPPORTS BOTH LOGIN & REGISTER
    ✅ Prevents duplicate email registration
    """
    global firebase_login_result

    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
        response.headers.add("Access-Control-Allow-Methods", "POST,OPTIONS")
        return response, 200

    print("\n" + "=" * 80)
    print("🔥 FIREBASE AUTHENTICATION ENDPOINT CALLED")
    print("=" * 80)

    try:
        data = request.get_json(force=True)
        id_token = data.get("token") if data else None
        action = data.get("action", "login")

        print(f"📋 Action: {action}")
        print(f"🔐 Token present: {bool(id_token)}")

        if not id_token:
            response = jsonify({"success": False, "error": "No token provided"})
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response, 400

        user_info = verify_firebase_token(id_token)

        if not user_info:
            response = jsonify({"success": False, "error": "Invalid or expired token"})
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response, 401

        print(f"✅ User info verified: {user_info.get('email')}")

        # ✅ HANDLE REGISTRATION
        if action == "register":
            success, message = register_or_login_firebase_user(user_info)

            if success:
                # Check if it's actually a new registration or existing user login
                is_new_user = "Welcome to All Mind" in message or "🎉" in message

                response = jsonify({
                    "success": True,
                    "message": message,
                    "is_new_user": is_new_user,
                    "user": {
                        "username": current_user["username"],
                        "email": current_user["email"],
                        "full_name": current_user.get("full_name", "")
                    }
                })
                response.headers.add("Access-Control-Allow-Origin", "*")
                print(f"✅ Registration {'successful' if is_new_user else 'detected as existing user'}")
                return response, 200
            else:
                # ✅ Registration failed - email already exists with different auth method
                print(f"❌ Registration blocked: {message}")
                response = jsonify({"success": False, "error": message})
                response.headers.add("Access-Control-Allow-Origin", "*")
                return response, 409  # ✅ 409 Conflict for duplicate email

        # ✅ HANDLE LOGIN
        else:
            result_tuple = login_firebase_user(user_info)
            firebase_login_result = result_tuple

            success_msg = result_tuple[0]
            is_success = "Welcome back" in success_msg or "✅" in success_msg

            if is_success:
                response = jsonify({
                    "success": True,
                    "message": success_msg,
                    "user": {
                        "username": current_user["username"],
                        "email": current_user["email"],
                        "full_name": current_user.get("full_name", "")
                    }
                })
                response.headers.add("Access-Control-Allow-Origin", "*")
                print(f"✅ Login successful: {current_user['username']}")
                return response, 200
            else:
                # Login failed - user not found or other error
                print(f"❌ Login failed: {success_msg}")
                response = jsonify({"success": False, "error": success_msg})
                response.headers.add("Access-Control-Allow-Origin", "*")
                return response, 404  # ✅ 404 Not Found for unregistered user

    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        response = jsonify({"success": False, "error": str(e)})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response, 500

@flask_app.route("/api/check-auth", methods=["GET"])
def check_auth_status():
    """Check if user is authenticated"""
    return jsonify({
        "logged_in": current_user.get("logged_in", False),
        "username": current_user.get("username"),
        "is_guest": current_user.get("is_guest", True)
    })
@flask_app.route("/test-firebase", methods=["GET"])
def test_firebase():
    """Test Firebase configuration"""
    try:
        app = firebase_admin.get_app()
        return jsonify({
            "status": "✅ Firebase initialized",
            "app_name": app.name,
            "project_id": app.project_id if hasattr(app, 'project_id') else "N/A"
        }), 200
    except ValueError:
        return jsonify({
            "status": "❌ Firebase NOT initialized",
            "error": "No Firebase app found"
        }), 500
    except Exception as e:
        return jsonify({
            "status": "❌ Error",
            "error": str(e)
        }), 500

def check_auth_status():
    """Check if user is authenticated"""
    return jsonify({
        "logged_in": current_user.get("logged_in", False),
        "username": current_user.get("username"),
        "is_guest": current_user.get("is_guest", True)
    })


# Start Flask server in background thread
def run_flask():
    """Run Flask server in a separate thread"""
    flask_app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)


if FIREBASE_AVAILABLE:
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    logging.info("✅ Flask server started on port 5000 for Firebase auth")

# ------------------ GRADIO UI ------------------
with gr.Blocks(
    title="All Mind",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown("# 🤖 All Mind")
    # Login/Register Section
    with gr.Group(visible=False) as auth_section:
        gr.Markdown("## 🔐 Welcome! Please Login or Register")
        guest_status = gr.Markdown("", visible=False)

        with gr.Tab("Login"):
            gr.Markdown("## 🔑 Sign In to Your Account")

            # ADD THIS BLOCK - Google Sign-In for Login
            if FIREBASE_AVAILABLE:
                gr.Markdown("### 🚀 Quick Sign In with Google")
                gr.HTML(login_html)  # ← YOU'RE MISSING THIS!
                gr.Markdown("---\n**OR**\n---")
                gr.Markdown("### Traditional Login")

            with gr.Row():
                login_username = gr.Textbox(label="Username", placeholder="Enter your username")
                login_password = gr.Textbox(label="Password", type="password", placeholder="Enter your password")
            login_btn = gr.Button("🔑 Login", variant="primary", size="lg")
            login_status = gr.Markdown()

        with gr.Tab("Register"):
            gr.Markdown("### 🎉 Create Your Free Account!")

            # Firebase Google Sign-In (if available)
            if FIREBASE_AVAILABLE:
                gr.Markdown("#### 🚀 Quick Sign Up with Google (Recommended)")
                gr.HTML(register_html)
                gr.Markdown("---\n**OR**\n---")
                gr.Markdown("### Traditional Email Registration")
            else:
                gr.Markdown("⚠️ **Google Sign-In temporarily unavailable** - Use email registration below")
                gr.Markdown("### Register with Email")

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
        gr.Markdown(
            "To Use all our features Please Logout from guest mode and register if not and Login to use full features")

        # Create a row with history button on the left and main content on the right
        with gr.Row():
            # Left sidebar for history button (small width)
            with gr.Column(scale=1, min_width=120):
                show_history_btn = gr.Button("📋 Show My History", variant="primary", size="sm")

            # Right side for main content (large width)
            with gr.Column(scale=10):
                pass  # Main content will go here

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

        with gr.Tab("Public IP", visible=False) as public_ip_tab:
            gr.Markdown("### Check your current public IP address")
            ip_output = gr.Markdown(label="IP Address")
            ip_btn = gr.Button("Get Public IP", variant="primary")

            ip_btn.click(
                get_public_ip,
                inputs=None,
                outputs=ip_output
            )

        # CORRECT: Translation tab is NOW OUTSIDE Public IP tab
        with gr.Tab("🌐 Translation", visible=False) as translation_tab:
            gr.Markdown("## 🌍 Universal Translator - Any Language to Any Language")
            gr.Markdown(
                "**Supports:**\n"
                "- 🇮🇳 **All 22 Indian Languages**: Hindi, Bengali, Telugu, Tamil, Marathi, Gujarati, Urdu, Kannada, Malayalam, Punjabi, Odia, Assamese, and more\n"
                "- 🌎 **30+ Foreign Languages**: Spanish, French, German, Chinese, Japanese, Korean, Russian, Arabic, Portuguese, and more\n"
                "- 🎤 **Voice Input**: Speak in any language\n"
                "- 🔊 **Audio Output**: Hear translation in target language\n"
                "- 🤖 **Auto-detection**: Automatically detects source language"
            )

            with gr.Row():
                mic_translate = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="🎙️ Speak Your Text (Any Language)"
                )

            translate_input = gr.Textbox(
                label="📝 Or Type Your Text",
                placeholder="Enter text in any language (Hindi, English, Spanish, etc.)...",
                lines=3
            )

            with gr.Row():
                target_lang_dropdown = gr.Dropdown(
                    label="🎯 Target Language (Select from 50+ languages)",
                    choices=[(name, code) for name, code in sorted(SUPPORTED_LANGUAGES.items())],
                    value="hi",  # Default to Hindi
                    filterable=True
                )
                translate_btn = gr.Button("🔄 Translate & Speak", variant="primary", size="lg")

            translation_result = gr.Markdown(label="📄 Translation Result")
            translation_audio = gr.Audio(label="🔊 Listen to Translation", autoplay=True)

            gr.Examples(
                examples=[
                    ["Hello, how are you?", "hi"],
                    ["नमस्ते, आप कैसे हैं?", "en"],
                    ["Bonjour, comment allez-vous?", "te"],
                    ["你好吗？", "ta"],
                    ["مرحبا، كيف حالك؟", "bn"],
                ],
                inputs=[translate_input, target_lang_dropdown],
                label="💡 Try These Examples"
            )

            translate_btn.click(
                perform_translation,
                inputs=[translate_input, target_lang_dropdown],
                outputs=[translate_input, translation_audio, translation_result]
            )


            def transcribe_and_translate(audio_filepath, target_lang):
                if current_user["is_guest"]:
                    return "", None, "**🔒 Register** to use voice translation."

                if not audio_filepath:
                    return "", None, "No audio recorded."

                text = transcribe_audio(audio_filepath)
                if text.startswith("❌"):
                    return "", None, text

                translated, audio_path, full_msg = perform_translation(text, target_lang)
                return translated, audio_path, full_msg


            mic_translate.change(
                transcribe_and_translate,
                inputs=[mic_translate, target_lang_dropdown],
                outputs=[translate_input, translation_audio, translation_result]
            )
    # Authentication Event Handlers
    login_btn.click(
        login_user,
        inputs=[login_username, login_password],
        outputs=[login_status, auth_section, main_app, user_info, file_qa_tab, image_gen_tab, image_qa_tab,
                 image_search_tab, video_gen_tab,translation_tab, stats_btn, history_chatbot, guest_chat_warning, chatbot, session_id, mic_chat]
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
                 image_search_tab, video_gen_tab, translation_tab, stats_btn, history_chatbot, guest_chat_warning,
                 chatbot, session_id, mic_chat]
    )
    # Firebase login checker - polls every 2 seconds to detect Google login
    firebase_check_timer = gr.Timer(value=2, active=True)

    firebase_check_timer.tick(
        check_firebase_login_and_update,
        inputs=None,
        outputs=[
            login_status, auth_section, main_app, user_info,
            file_qa_tab, image_gen_tab, image_qa_tab,
            image_search_tab, video_gen_tab, translation_tab,
            stats_btn, history_chatbot, guest_chat_warning,
            chatbot, session_id, mic_chat
        ]
    )


    def show_stats():
        stats = get_user_stats()
        return gr.update(visible=True), stats


    def close_stats():
        return gr.update(visible=False)


    stats_btn.click(show_stats, outputs=[stats_modal, stats_content])
    close_stats_btn.click(close_stats, outputs=stats_modal)



    demo.load(
        start_as_guest,
        inputs=None,
        outputs=[
            guest_status, auth_section, main_app, user_info,
            file_qa_tab, image_gen_tab, image_qa_tab,
            image_search_tab, video_gen_tab, translation_tab,
            stats_btn, history_chatbot, guest_chat_warning,
            chatbot, session_id, mic_chat
        ]
    )

if __name__ == "__main__":
    demo.launch()
