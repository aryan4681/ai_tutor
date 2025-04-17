import streamlit as st
import os
import re
import json
from openai import OpenAI
from dotenv import load_dotenv
import time
import random # Needed for shuffling options if desired
import base64 # Needed for image encoding

# --- Page Config MUST be the first Streamlit command ---
st.set_page_config(layout="wide")
# ------------------------------------------------------

# --- Add MathJax support ---
st.markdown("""
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
.stMarkdown {overflow-x: auto;}
</style>
""", unsafe_allow_html=True)
# ------------------------------------------------------

# --- Import Pre-Generated Exam Data ---
# Make sure demo_exam_data.py exists for the demo comprehensive exam
try:
    # Prioritize demo data if it exists
    import demo_exam_data
    if hasattr(demo_exam_data, 'exam_questions') and isinstance(demo_exam_data.exam_questions, list):
        COMPREHENSIVE_EXAM_AVAILABLE = True
        comprehensive_mcq_questions = demo_exam_data.exam_questions # Load from demo file
        COMPREHENSIVE_SUBJECT = "Linear Algebra" # Update subject name
    else:
        st.sidebar.error("`demo_exam_data.py` found, but 'exam_questions' list is missing or invalid.", icon="🚨")
        COMPREHENSIVE_EXAM_AVAILABLE = False
        comprehensive_mcq_questions = []
        COMPREHENSIVE_SUBJECT = None

except ImportError:
    # Fallback to original exam_data if demo is not found (optional, keep original logic)
    try:
        import exam_data
        if hasattr(exam_data, 'exam_questions') and isinstance(exam_data.exam_questions, list):
            COMPREHENSIVE_EXAM_AVAILABLE = True
            comprehensive_mcq_questions = exam_data.exam_questions
            COMPREHENSIVE_SUBJECT = "Linear Algebra" # Original subject
        else:
            # Don't show error if demo_exam_data was intended but missing
            COMPREHENSIVE_EXAM_AVAILABLE = False
            comprehensive_mcq_questions = []
            COMPREHENSIVE_SUBJECT = None
    except ImportError:
        COMPREHENSIVE_EXAM_AVAILABLE = False
        comprehensive_mcq_questions = []
        COMPREHENSIVE_SUBJECT = None
        # Warning shown later if comprehensive exam selected

# --- Configuration ---
load_dotenv()
OPENAI_MODEL = "gpt-4o" # Or "gpt-4-turbo", "gpt-3.5-turbo" etc.
API_KEY_ENV_VAR = "OPENAI_API_KEY"

# --- Fallback Questions (if API fails for Diagnostic/Practice) ---
FALLBACK_QUESTIONS = [
    {
        "question": "What is the result of multiplying the matrix [[1, 2], [3, 4]] by the scalar 2?",
        "options": ["[[2, 4], [6, 8]]", "[[1, 4], [3, 8]]", "[[2, 2], [6, 6]]", "[[3, 4], [5, 6]]"],
        "correct": "[[2, 4], [6, 8]]"
    },
    {
        "question": "Which of the following represents a system of linear equations with no solution?",
        "options": ["x + y = 2, 2x + 2y = 4", "x + y = 2, x - y = 0", "x + y = 2, x + y = 3", "x = 1, y = 1"],
        "correct": "x + y = 2, x + y = 3"
    }
]

# --- Example Topics (Add more as needed) ---
# Ensure the demo subject is listed if needed, or adjust AVAILABLE_TOPICS
AVAILABLE_TOPICS = {
    "Linear Algebra": ["Solving Systems of Linear Equations", "Matrix Operations", "Vector Spaces", "Eigenvalues and Eigenvectors"],
    "Calculus": ["Derivatives", "Integrals", "Limits", "Differential Equations"],
    "Algebra Basics": ["Solving Linear Equations", "Polynomials", "Factoring", "Exponents and Radicals"],
# Add if you want it selectable, maybe copy topics?
}
HIGHLIGHTED_TOPIC = "Solving Systems of Linear Equations" # Topic to visually highlight

# --- AI Personalities --- MOVED EARLIER ---
AI_PERSONALITIES = {
    "Supportive Tutor (Default)": "You are a supportive, encouraging, and clear math tutor.",
    "Silicon Valley Nerd": "you are a sarcastic, slightly jaded, 'too-cool-for-school' silicon valley nerd explaining math. you always respond in lowercase, never use capitalization. you often use tech jargon ironically or out of context (e.g., 'synergize the coefficients', 'bandwidth issue with factoring'). you are helpful, but in a reluctant, eye-rolling kind of way, yet ultimately provide the correct explanation concisely.",
    "Unhinged": "You are an UNHINGED math tutor with absolutely zero filter. Your explanations are correct but delivered with brutal honesty and sarcasm. Roast the student's mistakes mercilessly. You're like a combination of a stand-up comedian and a math genius having a manic episode. You're simultaneously brilliant and chaotic. Talk like a silicon valley nerd, throw in some famous contextual hostorical figures, also current event jokes and references."
}
DEFAULT_PERSONALITY = "Supportive Tutor (Default)"
# ----------------------------------------

# --- OpenAI Client Initialization ---
openai_api_key = None
# 1. Try environment variable FIRST
api_key_from_env = os.getenv(API_KEY_ENV_VAR)
if api_key_from_env:
    openai_api_key = api_key_from_env
# 2. If not found via env var, THEN try Streamlit secrets
elif hasattr(st, 'secrets'):
    try:
        if API_KEY_ENV_VAR in st.secrets:
             openai_api_key = st.secrets[API_KEY_ENV_VAR]
    except Exception as e:
        pass # Silently ignore

openai_client = None
if openai_api_key:
    try:
        openai_client = OpenAI(api_key=openai_api_key)
    except Exception as e:
        # Error will be handled later if client is None
        pass
else:
     # Error will be handled later if client is None
     pass


# --- Helper Functions ---

# Updated to potentially handle multimodal input (text + image)
def query_openai_model(prompt, system_message="You are a helpful AI assistant specializing in math education.", image_bytes=None, image_media_type="image/jpeg"):
    """Sends a prompt to the configured OpenAI model, potentially including image data."""
    if not openai_client:
        return {"error": "OpenAI client not initialized."}

    messages = []
    # System message always first
    if system_message:
        messages.append({"role": "system", "content": system_message})

    # Construct user message content (text + optional image)
    user_content = [{"type": "text", "text": prompt}]
    if image_bytes:
        try:
            # Encode image bytes as base64
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            user_content.append({
                "type": "image_url",
                "image_url": {
                    # Include the media type (e.g., "image/jpeg", "image/png")
                    "url": f"data:{image_media_type};base64,{base64_image}"
                }
            })
            print(f"DEBUG: Added image of type {image_media_type} to the prompt.") # Debugging
        except Exception as e:
            print(f"Error encoding image for OpenAI: {e}") # Log error
            # Decide whether to proceed without image or return error
            # For now, proceed with text only if image encoding fails
            # return {"error": f"Failed to process image for API call: {e}"}


    messages.append({"role": "user", "content": user_content})
    # print("DEBUG: OpenAI Request Messages:", messages) # Careful: Can be very verbose

    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL, # Ensure this model (e.g., gpt-4o) supports vision
            messages=messages,
            temperature=0.7,
            top_p=0.9,
            max_tokens=500 # Increase max_tokens if expecting longer feedback
        )
        # print("DEBUG: OpenAI Response:", response) # Debugging
        return {"generated_text": response.choices[0].message.content}
    except Exception as e:
        print(f"Error during OpenAI API call: {e}") # Log error
        return {"error": f"OpenAI API call failed: {e}"}

def extract_json_from_response(response_data):
    """Attempts to extract JSON from the model's response text."""
    # This function is primarily needed for AI-generated questions now.
    if "error" in response_data or not response_data.get("generated_text"):
        return None

    full_text = response_data["generated_text"]
    try:
        # Try direct JSON parsing first
        first_brace = full_text.find('{')
        first_bracket = full_text.find('[')

        if first_brace == -1 and first_bracket == -1: return None

        start_index = -1; end_char = ''
        if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
             start_index = first_brace; end_char = '}'
        elif first_bracket != -1:
             start_index = first_bracket; end_char = ']'

        if start_index != -1:
            potential_json = full_text[start_index:]
            balance = 0; end_index = -1; in_string = False
            for i, char in enumerate(potential_json):
                if char == '"' and (i == 0 or potential_json[i-1] != '\\'):
                    in_string = not in_string
                elif not in_string:
                    if char == ('[' if end_char == ']' else '{'): balance += 1
                    elif char == end_char: balance -= 1
                    if balance == 0: end_index = i; break
            if end_index != -1:
                 return json.loads(potential_json[:end_index+1])

    except json.JSONDecodeError:
        # Fallback to regex for markdown code blocks
        json_match = re.search(r"```(?:json)?\s*\n(.*)\n```", full_text, re.DOTALL | re.IGNORECASE)
        if json_match:
            try: return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError: return None
    return None


def generate_questions(topic, num_questions=2, difficulty_level=None):
    """Generates topic-specific math questions using the AI model (for Diag/Practice), optionally tailored to a difficulty level."""
    difficulty_prompt_segment = ""
    if difficulty_level and difficulty_level in ["Beginner - Needs foundational review", "Intermediate - Good grasp, needs practice", "Advanced - Strong understanding"]:
         # Incorporate the level extracted from the diagnostic category string
         simple_level = difficulty_level.split(" - ")[0] # Get "Beginner", "Intermediate", or "Advanced"
         difficulty_prompt_segment = f"The student is likely at a {simple_level} level. Please tailor the question difficulty appropriately. "

    prompt = (
        f"Generate {num_questions} distinct multiple-choice questions about '{topic}'. "
        f"{difficulty_prompt_segment}" # Add the difficulty instruction here
        f"Each question must have exactly four options. Clearly indicate the single correct answer. "
        f"Return the result ONLY as a valid JSON list containing {num_questions} objects. "
        f"Each object must have keys: 'question' (string), 'options' (list of 4 strings), and 'correct' (string - the correct option text). "
        f"Ensure the 'correct' value exactly matches one of the strings in the 'options' list."
        f"Do not include any introductory text, explanations, or markdown formatting like ```json."
        f"Just output the raw JSON list."
    )
    system_msg = "You are an expert JSON generator. Create JSON output according to the user's request precisely."
    model_response = query_openai_model(prompt, system_message=system_msg)

    if "error" in model_response:
        st.error(f"Failed to generate questions: {model_response['error']}", icon="📉")
        return FALLBACK_QUESTIONS[:num_questions]

    generated_questions = extract_json_from_response(model_response) # Use the robust extractor

    if not generated_questions or not isinstance(generated_questions, list) or len(generated_questions) == 0:
        st.error(f"Failed to parse valid questions from the model response. Using fallback questions.", icon="📉")
        return FALLBACK_QUESTIONS[:num_questions]

    # --- Validation ---
    valid_questions = []
    for i, q in enumerate(generated_questions):
        if (isinstance(q, dict) and
            all(k in q for k in ['question', 'options', 'correct']) and
            isinstance(q.get('question'), str) and
            isinstance(q.get('options'), list) and len(q['options']) == 4 and
            all(isinstance(opt, str) for opt in q['options']) and
            isinstance(q.get('correct'), str) and
            q['correct'] in q['options']):
            valid_questions.append(q)
        else:
             st.warning(f"AI generated question {i+1} has invalid format or answer. Discarding.", icon="⚠️")

    # Ensure we return the requested number, using fallbacks if needed
    if len(valid_questions) < num_questions:
        st.warning(f"Model generated only {len(valid_questions)} valid questions. Using fallback questions for the remainder.", icon="⚠️")
        needed = num_questions - len(valid_questions)
        valid_fallbacks = [fb for fb in FALLBACK_QUESTIONS if fb['correct'] in fb['options']]
        valid_questions.extend(valid_fallbacks[:needed])

    return valid_questions[:num_questions] # Return exactly the number requested


# --- Feedback, Help, Solution Functions ---
def get_feedback(question, user_answer, personality_name=DEFAULT_PERSONALITY, uploaded_file_data=None):
    """
    Gets feedback on the user's answer using the AI model, adapting to personality.
    Handles both MCQ and open_text questions, including potential file uploads.
    """
    system_msg = AI_PERSONALITIES.get(personality_name, AI_PERSONALITIES[DEFAULT_PERSONALITY])
    q_type = question.get("type", "mcq") # Default to mcq if type is missing

    image_bytes = None
    image_media_type = None
    if uploaded_file_data:
        # uploaded_file_data structure expected: {'name': '...', 'type': '...', 'bytes': b'...'}
        file_type = uploaded_file_data.get('type', '')
        # Allow common image types for direct inclusion
        if file_type in ['image/png', 'image/jpeg', 'image/gif', 'image/webp']:
             image_bytes = uploaded_file_data.get('bytes')
             image_media_type = file_type
             print(f"DEBUG: Preparing image {uploaded_file_data.get('name')} for feedback.")
        # Note: PDF content extraction would need to happen here or before calling this function
        # and be added to the text prompt, as GPT-4o doesn't directly ingest PDF files via API yet.


    if q_type == "mcq":
        prompt = (
             f"The user was asked the following multiple-choice question:\nQuestion: {question['question']}\n"
             f"Options: {question['options']}\n"
             f"Correct Answer: {question['correct']}\n"
             f"User's Answer: {user_answer}\n\n"
             "Provide simple, supportive, and concise feedback (max 30 words). "
             "If the user was correct, acknowledge it appropriately for your personality. "
             "If incorrect, state it and perhaps hint towards the correct concept without giving the direct answer away unless the mistake is very simple."
             "Adhere strictly to the persona defined in the system message."
        )
        # No image for MCQs
        model_response = query_openai_model(prompt, system_message=system_msg)

    elif q_type == "open_text":
        # Check if image data was actually passed
        if not image_bytes:
            # This case might happen if submission logic allowed proceeding without an image
            # Or if file reading failed earlier but wasn't caught properly.
            return "Error: No image data received for evaluation. Please ensure an image was uploaded correctly."

        file_info_prompt = ""
        if uploaded_file_data:
             filename = uploaded_file_data.get('name', 'Unnamed File')
             file_info_prompt = f"The user uploaded the attached image named '{filename}' as their answer."
        else:
             # Should ideally not happen if image_bytes is present, but as fallback:
             file_info_prompt = "The user uploaded the attached image as their answer."


        prompt = (
            f"The user was asked the following open-ended question:\nQuestion: {question['question']}\n\n"
            f"{file_info_prompt}\n\n" # Refer specifically to the *image* upload
            "Please evaluate the work shown in the image based on the question asked. Provide constructive, clear, and supportive feedback (approx. 50-100 words). "
            "Explain if the work shown is correct, partially correct, or incorrect, and why. Suggest areas for improvement if needed. "
            "Adhere strictly to the persona defined in the system message."
            # Added clarification to focus on image:
            "Focus ONLY on the content of the provided image."
            # --- ADDED FORMATTING INSTRUCTION ---
            "\n\nIMPORTANT: When using any mathematical notation (like R^n, specific symbols, matrices, etc.), please enclose it in single dollar signs for proper rendering (e.g., write $\\mathbb{R}^2$ not (R^2) or R^2)."
            # -------------------------------------
        )
        # Pass image bytes
        model_response = query_openai_model(prompt, system_message=system_msg, image_bytes=image_bytes, image_media_type=image_media_type)

    else: # Fallback for unknown question type
        return "Cannot provide feedback for this question type."


    if "error" in model_response or not model_response.get("generated_text"):
        return "Could not get feedback at this time."
    return model_response["generated_text"]

def get_help(question, help_request, personality_name=DEFAULT_PERSONALITY):
    """Gets help related to the current question using the AI model, adapting to personality."""
    system_msg = AI_PERSONALITIES.get(personality_name, AI_PERSONALITIES[DEFAULT_PERSONALITY])
    prompt = (
        f"A student needs help with a math question or concept. They're currently working on this question:\n"
        f"Question: {question['question']}\n"
        f"Options: {question['options']}\n\n"
        f"Their specific request is: '{help_request}'\n\n"
        "Provide a helpful and comprehensive explanation that addresses their request. They might be asking about:"
        "- The specific question they're working on"
        "- A general math concept related to the question"
        "- A different topic in the same subject area"
        "- How to approach similar problems"
        "\n\n"
        "Give a thorough explanation (around 150-300 words) that helps them understand the concept deeply. "
        "If they're asking about the specific question, you can provide guidance and explanation, but avoid directly stating 'the answer is X' for multiple-choice questions."
        "If they're asking about something unrelated to the current question, feel free to address their actual request."
        "Adhere to the persona defined in the system message."
        "\n\n"
        "IMPORTANT FORMATTING INSTRUCTION: When using ANY mathematical notation (variables, equations, formulas, matrices, etc.), "
        "always enclose it in single dollar signs for proper rendering. For example:"
        "- Write $x^2$ not x^2"
        "- Write $\\frac{1}{2}$ not \\frac{1}{2}"
        "- Write $\\begin{bmatrix} a & b \\\\ c & d \\end{bmatrix}$ not \\begin{bmatrix} a & b \\\\ c & d \\end{bmatrix}"
        "- Write $y = mx + b$ not y = mx + b"
        "This is CRITICAL for proper display of mathematics in the user interface."
    )
    model_response = query_openai_model(prompt, system_message=system_msg)
    if "error" in model_response or not model_response.get("generated_text"):
        return "Sorry, I couldn't generate help at this time."
    return model_response["generated_text"]

def get_step_by_step_solution(question, personality_name=DEFAULT_PERSONALITY):
    """Gets a step-by-step solution for the question, adapting to personality."""
    system_msg = AI_PERSONALITIES.get(personality_name, AI_PERSONALITIES[DEFAULT_PERSONALITY])
    prompt = (
        f"Provide a detailed, step-by-step explanation for why '{question['correct']}' is the correct answer to the following multiple-choice math question:\n"
        f"Question: {question['question']}\n"
        f"Options Provided: {question['options']}\n\n"
        "Explain the concepts and calculations needed to arrive at the correct answer. "
        "If relevant, briefly explain why other options might be incorrect (common mistakes)."
        "Format the steps logically (e.g., using numbering or bullet points)."
        "Adhere strictly to the persona defined in the system message."
    )
    model_response = query_openai_model(prompt, system_message=system_msg) # Pass dynamic system_msg
    if "error" in model_response or not model_response.get("generated_text"):
        return "Sorry, I couldn't generate a step-by-step solution at this time."
    return model_response["generated_text"]

# --- UI Function for Displaying a Single Question ---
def display_question_ui(question_state, question_obj, question_index, mode_prefix):
    """Displays the UI elements for a single question and handles interactions."""
    # --- Input Validation ---
    if not question_obj or not isinstance(question_obj, dict):
        st.error(f"Error: Cannot display question {question_index + 1}. Invalid question data received.")
        total_q_key = "total_questions_generated" # Key holding total questions
        if st.button("Skip Invalid Question", key=f"{mode_prefix}_skip_invalid_{question_index}"):
             if question_index + 1 < question_state.get(total_q_key, 0): question_state["current_index"] += 1
             else: question_state["finished"] = True; st.session_state.app_state = f"{mode_prefix}_finished"
             st.rerun()
        return

    q_text = question_obj.get("question")
    q_type = question_obj.get("type", "mcq") # Default to mcq

    if not q_text:
         st.error(f"Error: Question {question_index + 1} text is missing.")
         total_q_key = "total_questions_generated"
         if st.button("Skip Incomplete Question", key=f"{mode_prefix}_skip_incomplete_{question_index}"):
              if question_index + 1 < question_state.get(total_q_key, 0): question_state["current_index"] += 1
              else: question_state["finished"] = True; st.session_state.app_state = f"{mode_prefix}_finished"
              st.rerun()
         return

    # MCQ specific validation
    if q_type == "mcq":
        q_options = question_obj.get("options")
        q_correct = question_obj.get("correct")
        if not q_options or not q_correct or not isinstance(q_options, list) or len(q_options) != 4:
            st.error(f"Error: MCQ Question {question_index + 1} data is incomplete or invalid (options/correct).")
            total_q_key = "total_questions_generated"
            if st.button("Skip Incomplete Question", key=f"{mode_prefix}_skip_incomplete_{question_index}"):
                 if question_index + 1 < question_state.get(total_q_key, 0): question_state["current_index"] += 1
                 else: question_state["finished"] = True; st.session_state.app_state = f"{mode_prefix}_finished"
                 st.rerun()
            return
    # --- End Input Validation ---

    q = question_obj # Use validated object
    
    # Check if this question is already bookmarked
    is_bookmarked = False
    for bq in st.session_state.bookmarked_questions:
        if bq.get('question') == q.get('question'):
            is_bookmarked = True
            break
    
    # Display question with bookmark button
    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        st.markdown(f"**Question {question_index + 1}:** <span style='font-size: 1.4em;'>{q['question']}</span>", unsafe_allow_html=True)
    with col2:
        bookmark_label = "★" if is_bookmarked else "☆"
        if st.button(bookmark_label, key=f"{mode_prefix}_bookmark_{question_index}"):
            if is_bookmarked:
                # Remove from bookmarks
                st.session_state.bookmarked_questions = [
                    bq for bq in st.session_state.bookmarked_questions 
                    if bq.get('question') != q.get('question')
                ]
                st.success("Question removed from bookmarks")
            else:
                # Add to bookmarks
                # Create a copy of the question with source information
                bookmark_data = q.copy()
                bookmark_data['source'] = question_state.get('mode_name', '')
                bookmark_data['subject'] = st.session_state.selected_subject
                bookmark_data['topic'] = question_state.get('topic', '')
                
                st.session_state.bookmarked_questions.append(bookmark_data)
                # Award points for bookmarking
                award_points('bookmark')
                st.success("Question bookmarked!")
            
            st.rerun()

    # State keys using prefix for uniqueness
    answer_key = f"{mode_prefix}_answer_{question_index}"
    submitted_key = f"{mode_prefix}_submitted_{question_index}"
    help_visible_key = f"{mode_prefix}_show_help_{question_index}"
    help_input_key = f"{mode_prefix}_help_input_{question_index}"
    help_text_key = f"{mode_prefix}_help_text_{question_index}"
    show_steps_key = f"{mode_prefix}_show_steps_{question_index}" # For step-by-step
    step_solution_key = f"{mode_prefix}_step_solution_{question_index}" # For step-by-step
    upload_key = f"{mode_prefix}_upload_{question_index}" # For file uploads (open text)
    upload_data_key = f"{mode_prefix}_upload_data_{question_index}" # To store file bytes/info


    # Initialize state if needed
    if answer_key not in question_state: question_state[answer_key] = ""
    if submitted_key not in question_state: question_state[submitted_key] = False
    if help_visible_key not in question_state: question_state[help_visible_key] = False
    if help_input_key not in question_state: question_state[help_input_key] = ""
    if help_text_key not in question_state: question_state[help_text_key] = ""
    if show_steps_key not in question_state: question_state[show_steps_key] = False
    if step_solution_key not in question_state: question_state[step_solution_key] = ""
    if upload_key not in question_state: question_state[upload_key] = None # Holds UploadedFile object
    if upload_data_key not in question_state: question_state[upload_data_key] = None # Holds {'name':.., 'type':.., 'bytes':..}
    if "feedback" not in question_state: question_state["feedback"] = []
    if "score" not in question_state: question_state["score"] = 0 # Score only counts MCQs

    # Get the currently selected personality
    selected_personality = st.session_state.get('selected_personality', DEFAULT_PERSONALITY)

    # --- Display Input Elements and Submit Form ---
    if not question_state[submitted_key]:
        with st.form(key=f"{mode_prefix}_form_{question_index}"):

            if q_type == "mcq":
                # --- Shuffle options for display (MCQ only) ---
                shuffled_options_key = f"{mode_prefix}_shuffled_options_{question_index}"
                if shuffled_options_key not in question_state:
                    original_options = list(q.get('options', [])) # Use q here
                    random.shuffle(original_options)
                    question_state[shuffled_options_key] = original_options
                display_options = question_state.get(shuffled_options_key, list(q.get('options', [])))

                current_selection = question_state.get(answer_key)
                try:
                    current_index = display_options.index(current_selection) if current_selection in display_options else 0
                except (ValueError, AttributeError):
                    current_index = 0

                # Add CSS to increase option font size
                st.markdown('<style>.stRadio label p {font-size: 1.2em !important;}</style>', unsafe_allow_html=True)
                
                question_state[answer_key] = st.radio( # Store selected option text
                    "Choose your answer:",
                    display_options,
                    key=f"{mode_prefix}_radio_{question_index}",
                    index=current_index,
                    label_visibility="collapsed"
                )
                # No file uploader for MCQ

            elif q_type == "open_text":
                # File uploader for open questions (IMAGE ONLY)
                question_state[upload_key] = st.file_uploader(
                    "Upload your answer as an image (PNG, JPG, JPEG)", # Modified label
                    type=['png', 'jpg', 'jpeg'],
                    key=f"{mode_prefix}_uploader_{question_index}",
                    accept_multiple_files=False # Ensure only one file
                )

            # Submit Button (common for both types)
            submit_pressed = st.form_submit_button("Submit Answer")

            if submit_pressed:
                selected_answer = "" # No text answer for open questions now
                if q_type == "mcq":
                     selected_answer = question_state.get(answer_key, "").strip() # Get MCQ answer

                uploaded_file = question_state.get(upload_key) # Get UploadedFile object

                # --- Submission Validation ---
                valid_submission = False
                if q_type == "mcq":
                    if selected_answer:
                        valid_submission = True
                    else:
                        st.warning("Please select an answer before submitting.", icon="👆")
                elif q_type == "open_text":
                    if uploaded_file is not None:
                        valid_submission = True
                    else:
                        # Modified warning: Require image upload
                        st.warning("Please upload an image answer before submitting.", icon="🖼️")
                # --- End Validation ---

                if valid_submission:
                    question_state[submitted_key] = True
                    # Process file upload *if* one exists (only for open_text)
                    file_data_for_feedback = None
                    if q_type == "open_text" and uploaded_file is not None:
                         try:
                              file_bytes = uploaded_file.getvalue()
                              file_data_for_feedback = {
                                   'name': uploaded_file.name,
                                   'type': uploaded_file.type,
                                   'bytes': file_bytes
                              }
                              question_state[upload_data_key] = file_data_for_feedback # Store processed data
                              print(f"DEBUG: Processed uploaded file: {uploaded_file.name}, Type: {uploaded_file.type}, Size: {len(file_bytes)} bytes")
                         except Exception as e:
                              st.error(f"Error reading uploaded file: {e}")
                              file_data_for_feedback = None
                    else:
                        # Ensure file_data is None for MCQs
                         file_data_for_feedback = None

                    # Ensure feedback list has space
                    while len(question_state["feedback"]) <= question_index:
                         question_state["feedback"].append(None)

                    with st.spinner("Evaluating your answer..."):
                        # Pass selected_answer (empty for open) and file data to get_feedback
                        feedback_text = get_feedback(q, selected_answer, selected_personality, file_data_for_feedback)

                    # Store feedback details
                    feedback_entry = {
                        "question": q['question'],
                        "question_type": q_type, # Store type
                        # Store None for user_answer if open_text
                        "user_answer": selected_answer if q_type == "mcq" else None,
                        "evaluation": feedback_text,
                        "uploaded_file_info": {
                            'name': file_data_for_feedback['name'],
                            'type': file_data_for_feedback['type']
                        } if file_data_for_feedback else None
                    }
                    # Only include correct answer and update score for MCQs
                    is_correct = False
                    if q_type == "mcq":
                        q_correct = q.get('correct') # Get correct answer for MCQ
                        feedback_entry["correct_answer"] = q_correct
                        is_correct = (selected_answer == q_correct)
                        if is_correct:
                            question_state["score"] += 1
                    
                    # Award points for completing a question
                    award_points('complete_question', is_correct)
                    
                    question_state["feedback"][question_index] = feedback_entry

                    # Clear specific states (keep answer/upload data until next q)
                    question_state[show_steps_key] = False; question_state[step_solution_key] = ""
                    question_state[help_visible_key] = False; question_state[help_text_key] = ""

                    st.rerun()


    # --- Display Feedback and Interaction Buttons (after submission) ---
    if question_state[submitted_key]:
        # Feedback display logic
        if question_index < len(question_state["feedback"]) and question_state["feedback"][question_index] is not None:
            last_feedback = question_state["feedback"][question_index]
            fb_q_type = last_feedback.get("question_type", "mcq")

            # Display user's answer/upload info
            st.markdown(f"**Your Submission:**")
            if fb_q_type == "open_text":
                 # Removed text_area display
                 uploaded_info = last_feedback.get("uploaded_file_info")
                 if uploaded_info:
                      st.caption(f"📎 You uploaded: {uploaded_info.get('name')} ({uploaded_info.get('type')})")
                      # Optionally display the image itself using st.image if needed
                      # image_bytes = question_state.get(upload_data_key, {}).get('bytes')
                      # if image_bytes: st.image(image_bytes, width=300)
                 else:
                      st.caption("_(No file uploaded or file info missing)_")
            else: # MCQ
                 st.markdown(f"{last_feedback.get('user_answer', 'N/A')}") # Use .get for safety

            # Display evaluation/feedback from AI
            st.markdown(f"**Feedback:**")
            if fb_q_type == "mcq":
                correct = last_feedback['user_answer'] == last_feedback['correct_answer']
                if correct: st.success(f"{last_feedback['evaluation']}", icon="✅")
                else: st.error(f"{last_feedback['evaluation']}", icon="❌"); st.info(f"Correct Answer: **{last_feedback['correct_answer']}**", icon="💡")
            else: # Open question - use info box for qualitative feedback
                st.info(f"{last_feedback['evaluation']}", icon="📝")
        else:
             st.warning("Could not retrieve feedback for this question index.", icon="🤔")

        # Step-by-Step button logic (ONLY FOR MCQ)
        if q_type == "mcq":
            # Removed step-by-step button and related code
            pass
        else:
             # No step-by-step or final answer buttons for open questions
             pass

        # Help Section logic (Common for both)
        st.divider()
        if st.button("Ask for help", key=f"{mode_prefix}_help_btn_{question_index}"):
             question_state[help_visible_key] = not question_state[help_visible_key]
             question_state[help_text_key] = "" # Clear old help
        if question_state[help_visible_key]:
            question_state[help_input_key] = st.text_area(
                 "What specific part of this question or solution do you need help with?",
                 key=f"{mode_prefix}_help_ta_{question_index}",
                 value=question_state.get(help_input_key, "")
                 )
            if st.button("↑", key=f"{mode_prefix}_help_submit_{question_index}"):
                 if question_state[help_input_key].strip():
                     with st.spinner("Asking the assistant for help..."):
                          # Pass the original question object 'q'
                          # Pass personality to get_help
                          question_state[help_text_key] = get_help(q, question_state[help_input_key], selected_personality)
                     st.rerun() # Display hint
                 else:
                      st.warning("Please type your help request first.", icon="✍️")

            if question_state[help_text_key]:
                 # Use st.markdown with HTML/MathJax processing enabled
                 st.markdown(f"""
                 <div class="math-container">
                 <strong>Help:</strong><br>
                 {question_state[help_text_key]}
                 </div>
                 """, unsafe_allow_html=True)
                 if st.button("Clear", key=f"{mode_prefix}_clear_hint_{question_index}"):
                      question_state[help_text_key] = ""
                      question_state[help_input_key] = ""
                      question_state[help_visible_key] = False
                      st.rerun()


        # Next/Finish Buttons logic
        st.divider()
        total_questions = len(question_state.get("questions", []))
        current_mode_finished_state = f"{mode_prefix}_finished"

        if question_index + 1 < total_questions:
            if st.button("Next Question", key=f"{mode_prefix}_next_btn_{question_index}", type="primary"):
                question_state["current_index"] += 1
                # Clear states for next question
                next_idx = question_index + 1
                # Include upload keys in clearing logic
                for key_suffix in ["answer", "submitted", "show_help", "help_text",
                                   "show_steps", "step_solution",
                                   "shuffled_options", "upload", "upload_data"]:
                    state_key = f"{mode_prefix}_{key_suffix}_{next_idx}"
                    if state_key in question_state:
                        if isinstance(question_state[state_key], bool): question_state[state_key] = False
                        elif isinstance(question_state[state_key], list): question_state[state_key] = []
                        # Handle None for file uploader state
                        elif key_suffix in ["upload", "upload_data"]: question_state[state_key] = None
                        else: question_state[state_key] = ""
                st.rerun()
        else:
             # Last question: Show Finish button
             finish_button_label = f"Finish {question_state.get('mode_name', 'Session')}"
             if st.button(finish_button_label, key=f"{mode_prefix}_finish_btn", type="primary"):
                 question_state["finished"] = True
                 # Award points for completing the session
                 if mode_prefix == 'd':
                     award_points('complete_diagnostic')
                 elif mode_prefix == 'p':
                     award_points('complete_practice')
                 elif mode_prefix == 'c':
                     award_points('complete_exam')
                 
                 st.session_state.app_state = current_mode_finished_state
                 st.rerun()

    # display_question_ui should not return True/False, it manages state internally
    # return False # Remove return statement if not used for control flow elsewhere

# --- Mode Functions (Diagnostic / Practice Session) ---
def initialize_mode_state(mode_key, mode_name, topic, level=None):
    """Initializes session state for Diagnostic or Practice Session modes."""
    if mode_key in st.session_state:
        del st.session_state[mode_key]
    st.session_state[mode_key] = {
        "mode_name": mode_name, "started": False, "finished": False,
        "questions": [], "current_index": 0, "score": 0, "feedback": [],
        "topic": topic, "num_questions_selected": 2, "total_questions_generated": 0,
        "level": level # Store the level
    }

def run_mode(mode_key):
    """Runs Diagnostic or Practice Session modes (AI question generation)."""
    if mode_key not in st.session_state:
        st.error(f"Mode state for '{mode_key}' not found."); st.session_state.app_state = 'welcome'; st.rerun(); return

    mode_state = st.session_state[mode_key]
    mode_name = mode_state['mode_name']
    topic = mode_state['topic']
    mode_prefix = mode_key[0] # 'd' or 'p'

    # Setup Screen - simplified UI
    if not mode_state["started"] and not mode_state["finished"]:
        st.header(f"{mode_name}: {topic}")  # Simplified header

        # Add level indicator if practice mode has an associated level - cleaner display
        current_level = mode_state.get('level')
        if mode_key == 'practice' and current_level:
            simple_level = current_level.split(' - ')[0]  # Just show "Beginner", "Intermediate", etc.
            st.info(f"Level: **{simple_level}**") 

        # Cleaner slider with direct label
        num_questions = st.slider("Number of questions", min_value=1, max_value=4, value=mode_state.get("num_questions_selected", 2), key=f"{mode_prefix}_num_q")
        mode_state["num_questions_selected"] = num_questions
        
        # Start button with more whitespace
        st.write("")  # Add some space
        if st.button(f"Start", key=f"{mode_prefix}_start_btn", type="primary"):
            if not openai_client: st.error("Cannot start: OpenAI client not available.", icon="🚫"); return

            level_to_generate = mode_state.get('level') # Get level if set
            spinner_msg = f"Generating questions..."  # Simplified message

            with st.spinner(spinner_msg):
                 # Pass level to generate_questions if available
                 mode_state["questions"] = generate_questions(topic, num_questions, difficulty_level=level_to_generate)

            if mode_state["questions"] and len(mode_state["questions"]) == num_questions:
                mode_state["started"] = True; mode_state["finished"] = False
                mode_state["current_index"] = 0; mode_state["score"] = 0
                mode_state["total_questions_generated"] = len(mode_state["questions"])
                mode_state["feedback"] = [None] * mode_state["total_questions_generated"] # Init feedback list
                st.session_state.app_state = f'{mode_key}_running'; st.rerun()
            else:
                 st.error(f"Could not generate questions. Please try again.", icon="🚫")
                 mode_state["started"] = False; mode_state["questions"] = []
        else: return # Wait for start button

    # Running Screen
    if mode_state["started"] and not mode_state["finished"] and st.session_state.app_state == f'{mode_key}_running':
        # Check if questions list is valid
        if not mode_state.get("questions"):
             st.error(f"Error: No questions loaded for {mode_name}. Please restart the mode.")
             mode_state["started"] = False; mode_state["finished"] = False # Reset state partially
             st.session_state.app_state = f'{mode_key}_setup' # Go back to setup
             st.rerun()
             return

        st.subheader(f"Question {mode_state['current_index'] + 1} of {mode_state['total_questions_generated']}")
        q_index = mode_state["current_index"]
        if q_index < len(mode_state["questions"]):
            current_question = mode_state["questions"][q_index]
            display_question_ui(mode_state, current_question, q_index, mode_prefix)
        else: # Safeguard
             mode_state["finished"] = True
             st.session_state.app_state = f'{mode_prefix}_finished'; st.rerun()

    # Finished Screen
    if mode_state["finished"] and st.session_state.app_state == f'{mode_prefix}_finished':
        st.header(f"{mode_name} Complete!")
        total_questions = mode_state.get("total_questions_generated", 0)
        category = None # Initialize category variable

        percentage = 0 # Initialize percentage BEFORE the calculation block

        if total_questions > 0:
             score = mode_state['score']
             st.write(f"Your score: {score} out of {total_questions}")
             try:
                 percentage = (score / total_questions) * 100 # Assigned only if try succeeds
                 st.progress(percentage / 100); st.write(f"{percentage:.1f}% correct")
             except ZeroDivisionError:
                 percentage = 0; # Already initialized, but setting again is fine
                 st.write("Cannot calculate percentage.")

             if mode_key == 'diagnostic':
                 if percentage < 40: category = "Beginner - Needs foundational review"
                 elif percentage < 75: category = "Intermediate - Good grasp, needs practice"
                 else: category = "Advanced - Strong understanding"
                 st.markdown("<h3 style='font-size:1.5em; font-weight:bold; margin-top:15px;'>Level</h3>", unsafe_allow_html=True)
                 st.info(f"**{category}**")
                 mode_state['diagnostic_level'] = category # <-- STORE the diagnosed level

             # --- Get and Display Topic Suggestions --- Added Section ---
             st.markdown("<h3 style='font-size:1.5em; font-weight:bold; margin-top:15px;'>Suggestions</h3>", unsafe_allow_html=True)
             # Now 'percentage' is guaranteed to have a value here
             suggestion = get_topic_suggestions(
                 subject=st.session_state.selected_subject, # Assuming subject is in session state
                 feedback_list=mode_state.get("feedback", []),
                 questions_list=mode_state.get("questions", []),
                 overall_percentage=percentage,
                 mode_type=mode_key, # 'diagnostic' or 'practice'
                 current_topic=mode_state.get("topic") # Pass the current topic
             )
             st.info(suggestion)
             st.markdown("<h3 style='font-size:1.5em; font-weight:bold; margin-top:15px;'>Review</h3>", unsafe_allow_html=True)
             # ----------------------------------------------------------

             with st.expander("Review Your Answers and Feedback"):
                  feedback_list = mode_state.get("feedback", [])
                  questions_list = mode_state.get("questions", [])
                  if not feedback_list or all(fb is None for fb in feedback_list):
                       st.write("No feedback recorded.")
                  else:
                      for i, q_data in enumerate(questions_list):
                           q_text = q_data.get('question','N/A') if isinstance(q_data, dict) else "Invalid Question Data"
                           q_type = q_data.get('type', 'mcq')
                           st.markdown(f"**Q{i+1} ({q_type.replace('_',' ').title()}): {q_text}**")

                           if i < len(feedback_list) and feedback_list[i] is not None:
                               fb = feedback_list[i]
                               st.markdown(f"**Your Submission:**")
                               if q_type == "open_text":
                                    uploaded_info = fb.get("uploaded_file_info")
                                    if uploaded_info:
                                         st.caption(f"📎 Uploaded: {uploaded_info.get('name')}")
                                    else:
                                         st.caption("_(No file uploaded or file info missing)_")
                               else: # MCQ
                                    st.markdown(f"{fb.get('user_answer', 'N/A')}") # Use .get

                               st.markdown(f"**Feedback:**")
                               if q_type == "mcq":
                                   correct = fb['user_answer'] == fb['correct_answer']
                                   if correct: st.success(f"{fb['evaluation']}", icon="✅")
                                   else: st.error(f"{fb['evaluation']}", icon="❌"); st.info(f"Correct Answer: **{fb['correct_answer']}**", icon="💡")
                               else: # Open Text
                                   st.info(f"{fb['evaluation']}", icon="📝")
                           else:
                               st.write("_(No answer submitted or feedback generated for this question)_")
                           st.divider()
        else:
             st.warning("No questions were attempted in this session.")
             # Even if no questions, 'percentage' is 0, so suggestions might still run if needed
             # Add logic here if suggestions should be skipped when total_questions is 0

        # Post-Mode Actions
        st.subheader("What's next?")
        cols = st.columns(3 if mode_key == 'diagnostic' and category else 2) # 3 columns if diagnostic results are shown

        with cols[0]:
            if st.button(f"Restart {mode_name}", key=f"{mode_prefix}_restart_btn"):
                 current_topic = mode_state['topic']
                 # Reset level if restarting diagnostic, keep if restarting practice (though practice usually goes via diagnostic)
                 level_to_pass = mode_state.get('level') if mode_key == 'practice' else None
                 mode_name_to_pass = mode_state.get('mode_name', mode_name) # Preserve level in name if practice
                 initialize_mode_state(mode_key, mode_name_to_pass, current_topic, level=level_to_pass)
                 st.session_state.app_state = f'{mode_key}_setup'; st.rerun()

        # --- ADD ADAPTIVE PRACTICE BUTTON FOR DIAGNOSTIC ---
        if mode_key == 'diagnostic' and category:
             with cols[1]:
                  simple_level_name = category.split(' - ')[0] # e.g., "Beginner"
                  if st.button(f"Practice at Your Level ({simple_level_name})", key=f"d_practice_adaptive_btn"):
                      current_topic = mode_state['topic']
                      level = mode_state['diagnostic_level']
                      initialize_mode_state('practice', f'Practice Session ({simple_level_name})', current_topic, level=level)
                      st.session_state.app_state = 'practice_setup'; st.rerun()

        # --- Back to Selection Button (Adjust column index) ---
        back_col_index = 2 if mode_key == 'diagnostic' and category else 1
        with cols[back_col_index]:
             if st.button("Choose New Subject/Mode", key=f"{mode_prefix}_back_to_selection_btn"):
                 if 'diagnostic' in st.session_state: del st.session_state['diagnostic']
                 if 'practice' in st.session_state: del st.session_state['practice']
                 if 'comprehensive_exam' in st.session_state: del st.session_state['comprehensive_exam']
                 st.session_state.app_state = 'welcome'; st.rerun()


# --- Comprehensive Exam Functions ---
def initialize_comprehensive_exam_state(subject, questions_list):
    """Initializes state for the exam using pre-generated questions."""
    mode_key = 'comprehensive_exam'
    if mode_key in st.session_state:
        del st.session_state[mode_key]

    # Basic validation of the input questions
    if not questions_list or not isinstance(questions_list, list):
         st.error(f"Cannot start Exam: No valid questions found for {subject}.")
         return False # Indicate failure

    valid_questions = []
    for i, q in enumerate(questions_list):
         if isinstance(q, dict) and q.get('question') and q.get('type'):
              q_type = q.get('type')
              is_valid = False
              if q_type == 'mcq':
                   # Validate MCQ fields
                   if (all(k in q for k in ['options', 'correct']) and
                       isinstance(q.get('options'), list) and len(q['options']) == 4 and
                       q.get('correct') in q.get('options', [])):
                       is_valid = True
              elif q_type == 'open_text':
                   # Open text just needs a question string (already checked)
                   is_valid = True
              # Add other types here if needed

              if is_valid:
                   valid_questions.append(q)
              else:
                   st.warning(f"Invalid format found in demo_exam_data.py for question index {i} (type: {q_type}). Skipping.", icon="⚠️")
         else:
              st.warning(f"Invalid data structure for question index {i} in demo_exam_data.py. Skipping.", icon="⚠️")


    if not valid_questions:
         st.error(f"Cannot start Exam: No valid questions loaded for {subject}.")
         return False

    st.session_state[mode_key] = {
        "mode_name": f"Exam: {subject}", "subject": subject,
        "started": True,
        "finished": False,
        "questions": valid_questions, # Use the validated questions
        "total_questions_generated": len(valid_questions),
        "current_index": 0, "score": 0, # Score only counts MCQs
        "feedback": [None] * len(valid_questions),
    }
    return True


def run_comprehensive_exam():
    """Runs the comprehensive exam mode using pre-generated MCQs."""
    mode_key = 'comprehensive_exam'

    # Ensure state exists and was started correctly
    if mode_key not in st.session_state or not st.session_state[mode_key].get('started'):
        st.error("Comprehensive exam state not found or not ready."); st.session_state.app_state = 'welcome'; st.rerun(); return

    mode_state = st.session_state[mode_key]
    mode_name = mode_state['mode_name']
    mode_prefix = 'c' # Use 'c' for comprehensive

    # --- Running Screen ---
    if mode_state["started"] and not mode_state["finished"] and st.session_state.app_state == 'comprehensive_exam_running':
        st.header(mode_name)
        # Check questions list again
        if not mode_state.get("questions"):
             st.error("Error: No questions available for this comprehensive exam."); st.session_state.app_state = 'welcome'; st.rerun(); return

        total_questions = mode_state['total_questions_generated']
        st.subheader(f"Question {mode_state['current_index'] + 1} of {total_questions}")
        q_index = mode_state["current_index"]

        if q_index < total_questions:
            current_question = mode_state["questions"][q_index]
            display_question_ui(mode_state, current_question, q_index, mode_prefix) # Display the MCQ
        else: # Safeguard
             mode_state["finished"] = True
             st.session_state.app_state = 'c_finished'; st.rerun()

    # --- Finished Screen ---
    if mode_state["finished"] and st.session_state.app_state == 'c_finished':
        st.header(f"{mode_name} Complete!")
        total_questions_attempted = len([fb for fb in mode_state.get("feedback", []) if fb is not None]) # Count non-None feedback entries
        mcq_questions_list = [q for q in mode_state.get("questions", []) if q.get("type") == "mcq"]
        total_mcqs = len(mcq_questions_list)
        score = mode_state.get('score', 0) # Score only tracks MCQs

        st.write(f"**Results Summary:**")
        if total_mcqs > 0:
            st.write(f"Multiple Choice Score: {score} out of {total_mcqs}")
            try:
                percentage = (score / total_mcqs) * 100
                st.progress(percentage / 100)
                st.write(f"{percentage:.1f}% correct (on MCQs)")
            except ZeroDivisionError:
                st.write("No MCQs to calculate percentage.")
        else:
            st.write("No multiple-choice questions in this exam.")

        # --- Get and Display Topic Suggestions --- (Keep existing logic, might need refinement based on open questions)
        st.markdown("<h3 style='font-size:1.5em; font-weight:bold; margin-top:15px;'>Suggestions</h3>", unsafe_allow_html=True)
        suggestion = get_topic_suggestions(
            subject=mode_state.get("subject"),
            feedback_list=mode_state.get("feedback", []),
            questions_list=mode_state.get("questions", []),
            overall_percentage=percentage,
            mode_type='comprehensive', # Explicitly set mode type
            current_topic=None # Not applicable for comprehensive
        )
        st.info(suggestion)
        st.markdown("<h3 style='font-size:1.5em; font-weight:bold; margin-top:15px;'>Review</h3>", unsafe_allow_html=True)
        # ----------------------------------------------------------

        with st.expander("Review Your Answers and Feedback"):
             feedback_list = mode_state.get("feedback", [])
             questions_list = mode_state.get("questions", [])
             if not feedback_list or all(fb is None for fb in feedback_list):
                  st.write("No feedback recorded.")
             else:
                 for i, q_data in enumerate(questions_list):
                      q_text = q_data.get('question','N/A') if isinstance(q_data, dict) else "Invalid Question Data"
                      q_type = q_data.get('type', 'mcq')
                      st.markdown(f"**Q{i+1} ({q_type.replace('_',' ').title()}): {q_text}**")

                      if i < len(feedback_list) and feedback_list[i] is not None:
                          fb = feedback_list[i]
                          st.markdown(f"**Your Submission:**")
                          if q_type == "open_text":
                               uploaded_info = fb.get("uploaded_file_info")
                               if uploaded_info:
                                    st.caption(f"📎 Uploaded: {uploaded_info.get('name')}")
                               else:
                                    st.caption("_(No file uploaded or file info missing)_")
                          else: # MCQ
                               st.markdown(f"{fb.get('user_answer', 'N/A')}") # Use .get

                          st.markdown(f"**Feedback:**")
                          if q_type == "mcq":
                              correct = fb['user_answer'] == fb['correct_answer']
                              if correct: st.success(f"{fb['evaluation']}", icon="✅")
                              else: st.error(f"{fb['evaluation']}", icon="❌"); st.info(f"Correct Answer: **{fb['correct_answer']}**", icon="💡")
                          else: # Open Text
                              st.info(f"{fb['evaluation']}", icon="📝")
                      else:
                          st.write("_(No answer submitted or feedback generated for this question)_")
                      st.divider()

        # --- Post-Mode Actions ---
        st.subheader("What's next?")
        col1, col2 = st.columns(2)
        with col1:
            # Restart button re-initializes with the same pre-generated MCQs
            if st.button(f"Restart Exam", key=f"{mode_prefix}_restart_btn"):
                 current_subject = mode_state['subject']
                 # Re-initialize using the already loaded questions
                 success = initialize_comprehensive_exam_state(current_subject, comprehensive_mcq_questions) # Use original loaded list
                 if success:
                      st.session_state.app_state = 'comprehensive_exam_running'
                 else:
                      # If initialization fails (e.g., data became invalid somehow), go back
                      st.session_state.app_state = 'welcome'
                 st.rerun()
        with col2:
             if st.button("Choose New Subject/Mode", key=f"{mode_prefix}_back_to_selection_btn"):
                 # Clear all mode states
                 if 'diagnostic' in st.session_state: del st.session_state['diagnostic']
                 if 'practice' in st.session_state: del st.session_state['practice']
                 if 'comprehensive_exam' in st.session_state: del st.session_state['comprehensive_exam']
                 st.session_state.app_state = 'welcome'; st.rerun()


# --- Helper Function for Topic Suggestions ---
def get_topic_suggestions(subject, feedback_list, questions_list, overall_percentage, mode_type, current_topic=None):
    """Analyzes results and suggests topics to focus on."""
    HIGH_SCORE_THRESHOLD = 85 # Percentage threshold for "doing well"

    if overall_percentage >= HIGH_SCORE_THRESHOLD:
        return "Great job! You have a strong grasp of this material. Consider reviewing or trying a different subject/topic."

    # For single-topic modes, suggest the current topic if score isn't high
    if mode_type in ['diagnostic', 'practice'] and current_topic:
        return f"Recommendation: Continue practicing on **{current_topic}**."

    # For comprehensive exams, use AI to identify weak topics from incorrect answers
    if mode_type == 'comprehensive':
        incorrect_questions_details = []
        if feedback_list and questions_list:
             for i, fb in enumerate(feedback_list):
                 # Ensure feedback exists, question data exists, and it was answered incorrectly
                 if fb and i < len(questions_list) and questions_list[i] and fb.get('user_answer') is not None and fb.get('user_answer') != fb.get('correct_answer'):
                     q_data = questions_list[i]
                     incorrect_questions_details.append(f"- {q_data.get('question', 'N/A')}") # Append the question text

        if not incorrect_questions_details:
             # This case might occur if the score is < 85 but somehow no specific incorrect answers were logged
             # Or if all questions were skipped
             return "Overall performance seems okay, but review any areas you felt unsure about."

        subject_topics = AVAILABLE_TOPICS.get(subject, [])
        if not subject_topics:
             return "Could not retrieve topics for this subject to provide specific suggestions."

        # --- MODIFICATION START ---
        # Define prompt template using standard multi-line string and placeholders
        prompt_template = """A student took a comprehensive exam on the subject '{subject_name}'.
They answered the following questions incorrectly:
{incorrect_list}

Based ONLY on these incorrect answers, which 1-3 specific topics from the following list seem to be the weakest areas for the student? List only the topic names, separated by commas. Do not add any other text, explanations, or introductory phrases like 'Based on the incorrect answers...' or 'The weak areas seem to be...'. If you cannot reliably determine specific weak topics from the questions provided, respond ONLY with the single word 'None'.

Available topics for {subject_name}: {available_topics_list}"""

        # Format the template string with the actual values
        prompt = prompt_template.format(
            subject_name=subject,
            incorrect_list='\n'.join(incorrect_questions_details),
            available_topics_list=', '.join(subject_topics)
        )
        # --- MODIFICATION END ---


        # Use a system message tailored for this specific task
        system_msg = "You are an analytical assistant identifying weak academic topics based on incorrect answers. Respond ONLY with the requested topic names (comma-separated) or the single word 'None'."
        ai_response = query_openai_model(prompt, system_message=system_msg)

        if "error" in ai_response or not ai_response.get("generated_text"):
            return "Could not analyze weak topics at this time."

        suggested_topics_text = ai_response["generated_text"].strip()

        # Validate the response - check if it's 'None' or seems like a valid topic list
        if suggested_topics_text.lower() == 'none' or not suggested_topics_text:
             return f"Suggestion: Review the incorrect answers above to identify areas needing more practice in {subject}."
        else:
             # Simple validation: check if the response contains parts of the available topics
             # More robust validation could be added if needed
             # Let's check if at least one suggested topic seems related to the available ones
             found_match = False
             potential_topics = [t.strip() for t in suggested_topics_text.split(',')]
             for pt in potential_topics:
                  if any(avail_topic.lower() in pt.lower() or pt.lower() in avail_topic.lower() for avail_topic in subject_topics):
                       found_match = True
                       break
             if found_match:
                return f"Areas to focus on for {subject}: **{suggested_topics_text}**"
             else:
                # AI might have hallucinated topics or provided an explanation instead of just names
                return f"Suggestion: Review the incorrect answers above to identify areas needing more practice in {subject}. (Could not automatically identify specific topics)."


    # Fallback for any unexpected cases
    return "Review your incorrect answers to identify areas for improvement."


# --- Main App Logic ---
def main():
    st.title("AI Math Tutor")  # Removed the brain emoji for cleaner look

    # Initialize Session State variables
    if 'app_state' not in st.session_state: st.session_state.app_state = 'welcome'
    if 'selected_subject' not in st.session_state: st.session_state.selected_subject = None
    if 'selected_topic' not in st.session_state: st.session_state.selected_topic = None
    # Initialize personality state - ADDED
    if 'selected_personality' not in st.session_state:
        st.session_state.selected_personality = DEFAULT_PERSONALITY
    # Initialize study plan state
    if 'study_plan' not in st.session_state:
        st.session_state.study_plan = None
    # Initialize bookmarked questions
    if 'bookmarked_questions' not in st.session_state:
        st.session_state.bookmarked_questions = []
        
    # Initialize gamification elements
    if 'last_activity_date' not in st.session_state:
        st.session_state.last_activity_date = None
    if 'streak' not in st.session_state:
        st.session_state.streak = 0
    if 'points' not in st.session_state:
        st.session_state.points = 0
    if 'completed_activities' not in st.session_state:
        st.session_state.completed_activities = []
    
    # Check and update streak based on today's date
    today = time.strftime("%Y-%m-%d")
    if st.session_state.last_activity_date != today:
        if st.session_state.last_activity_date:
            # Check if last activity was yesterday
            yesterday = time.strftime("%Y-%m-%d", time.localtime(time.time() - 86400))
            if st.session_state.last_activity_date == yesterday:
                st.session_state.streak += 1
            elif st.session_state.last_activity_date != today:
                # Reset streak if not consecutive days
                st.session_state.streak = 1
        else:
            # First time user
            st.session_state.streak = 1
        
        st.session_state.last_activity_date = today

    # --- Sidebar ---
    # Home Button at the top
    if st.sidebar.button("Home", key="home_button", use_container_width=True):
        # Reset to welcome state but keep subject/topic selections
        st.session_state.app_state = 'welcome'
        # Clear active session states
        if 'diagnostic' in st.session_state: del st.session_state['diagnostic']
        if 'practice' in st.session_state: del st.session_state['practice']
        if 'comprehensive_exam' in st.session_state: del st.session_state['comprehensive_exam']
        st.rerun()
    
    st.sidebar.divider()
    
    # OpenAI client warning if not available
    if not openai_client:
        st.sidebar.error("OpenAI API key missing or invalid.", icon="🚨")
        st.warning("AI Tutor features (Diagnostic, Practice, Help, Feedback) are disabled.", icon="⚠️")
    
    # Personality Selector
    st.sidebar.subheader("Tutor Style")  # Simplified header
    personality_list = list(AI_PERSONALITIES.keys())
    current_personality = st.session_state.selected_personality
    # Ensure current selection is valid, default if not
    if current_personality not in personality_list:
        current_personality = DEFAULT_PERSONALITY
        st.session_state.selected_personality = current_personality
    personality_index = personality_list.index(current_personality)

    new_personality = st.sidebar.selectbox(
        "",  # No redundant label
        options=personality_list,
        index=personality_index,
        key='personality_selector'
    )
    if new_personality != st.session_state.selected_personality:
        st.session_state.selected_personality = new_personality
    st.sidebar.divider()

    # 1. Subject Selection
    subjects = list(AVAILABLE_TOPICS.keys())
    default_subject_index = subjects.index(st.session_state.selected_subject) if st.session_state.selected_subject in subjects else 0
    selected_subject = st.sidebar.selectbox(
         "**1. Select Subject:**", options=subjects, index=default_subject_index, key='subject_selector'
         )

    # Handle subject change
    if selected_subject != st.session_state.selected_subject:
        st.session_state.selected_subject = selected_subject; st.session_state.selected_topic = None
        st.session_state.app_state = 'welcome'
        if 'diagnostic' in st.session_state: del st.session_state['diagnostic']
        if 'practice' in st.session_state: del st.session_state['practice']
        if 'comprehensive_exam' in st.session_state: del st.session_state['comprehensive_exam']
        st.rerun()

    subject_chosen = st.session_state.selected_subject # Use state variable

    # 2. Topic Selection (conditional)
    selected_topic = None
    if subject_chosen:
        topics = AVAILABLE_TOPICS.get(subject_chosen, [])
        # Use a prefix for highlighted topic instead of markdown formatting
        display_topics = [f"[Recommended] {t}" if t == HIGHLIGHTED_TOPIC else t for t in topics]
        display_topics.insert(0, "--- Select a Topic ---")

        current_topic_display = None
        if st.session_state.selected_topic:
             current_topic_display = f"[Recommended] {st.session_state.selected_topic}" if st.session_state.selected_topic == HIGHLIGHTED_TOPIC else st.session_state.selected_topic
        current_index = display_topics.index(current_topic_display) if current_topic_display in display_topics else 0

        selected_display_topic = st.sidebar.selectbox(
            "**2. Select Topic:**", options=display_topics, index=current_index, key='topic_selector'
            )

        # Handle topic change/selection
        if selected_display_topic != "--- Select a Topic ---":
            # Remove prefix instead of asterisks
            actual_topic = selected_display_topic.replace("[Recommended] ", "")
            if actual_topic != st.session_state.selected_topic:
                 st.session_state.selected_topic = actual_topic
                 st.session_state.app_state = 'topic_selected'
                 if 'diagnostic' in st.session_state: del st.session_state['diagnostic']
                 if 'practice' in st.session_state: del st.session_state['practice']
                 st.rerun()
            selected_topic = st.session_state.selected_topic # Use state variable
        else: # User selected the placeholder
            if st.session_state.selected_topic is not None:
                 st.session_state.selected_topic = None
                 st.session_state.app_state = 'welcome'
                 st.rerun()

        # --- Comprehensive Exam Button (if available) ---
        st.sidebar.divider()
        st.sidebar.subheader("Full Subject Exam")
        # Check if data exists AND the current subject matches the data's subject
        if COMPREHENSIVE_EXAM_AVAILABLE and subject_chosen == COMPREHENSIVE_SUBJECT:
            if comprehensive_mcq_questions: # Check if questions were loaded ok
                if st.sidebar.button(f"Start Exam ({subject_chosen})", key="start_comp_exam_btn"):
                    # Initialize state using pre-generated MCQs
                    success = initialize_comprehensive_exam_state(subject_chosen, comprehensive_mcq_questions)
                    if success:
                        st.session_state.app_state = 'comprehensive_exam_running'
                    else:
                         # Error shown in init function
                         pass
                    st.rerun()
            # This else case should ideally not be hit if COMPREHENSIVE_EXAM_AVAILABLE is True
            else: st.sidebar.info(f"Error loading questions for {subject_chosen} from exam_data.py.")
        elif subject_chosen: # Only show message if a subject is chosen
            if not COMPREHENSIVE_EXAM_AVAILABLE:
                 st.sidebar.info("`exam_data.py` not found or invalid.")
            elif subject_chosen != COMPREHENSIVE_SUBJECT:
                 st.sidebar.info(f"Exam only available for {COMPREHENSIVE_SUBJECT}.")
    
    # --- Bookmarked Questions Button (always at the end) ---
    st.sidebar.divider()
    st.sidebar.subheader("Saved Questions")
    bookmark_count = len(st.session_state.bookmarked_questions)
    bookmark_label = f"Bookmarked Questions ({bookmark_count})" if bookmark_count > 0 else "Bookmarked Questions"
    if st.sidebar.button(bookmark_label, use_container_width=True, key="view_bookmarks_btn"):
        st.session_state.app_state = 'bookmarks'
        st.rerun()
    # --- End of Sidebar Logic ---

    # --- Main Area Content based on App State ---

    # Welcome State
    if st.session_state.app_state == 'welcome':
        # Clean header at the top
        st.subheader("Dashboard")
        
        # Main content in a prominent container
        with st.container(border=True):
            col1, col2 = st.columns([0.7, 0.3])
            
            with col1:
                # Topic selection guidance
                st.subheader("Get Started")
                if subject_chosen:
                    st.info("Select a **Topic** from the sidebar for Diagnostic/Practice modes.")
                else:
                    st.info("Select a **Subject** from the sidebar to begin.")
            
            with col2:
                # Study plan buttons
                if st.button("Create Study Plan", key="create_study_plan_btn", type="primary", use_container_width=True):
                    st.session_state.app_state = 'study_plan_setup'
                    st.rerun()
                    
                # Show existing study plan button
                if st.session_state.study_plan:
                    if st.button("View Plan", key="view_study_plan_btn", use_container_width=True):
                        st.session_state.app_state = 'view_study_plan'
                        st.rerun()
        
        # Add visual separation with padding
        st.write("")
        st.write("")
        
        # Stats section at the bottom, separated from main content
        with st.container():
            display_stats()

    # Topic Selected State (Topic-Specific Modes)
    elif st.session_state.app_state == 'topic_selected' and selected_topic:
        st.header(f"{subject_chosen}: {selected_topic}")  # Simplified header
        
        # More compact buttons with better spacing
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.subheader("Diagnostic")
            if st.button("Start Diagnostic Test", key="start_diagnostic_btn", type="primary"):
                initialize_mode_state('diagnostic', 'Diagnostic Test', selected_topic)
                st.session_state.app_state = 'diagnostic_setup'; st.rerun()

        with col2:
            st.subheader("Practice")
            if st.button("Start Practice Session", key="start_practice_topic_btn", type="primary"):
                initialize_mode_state('practice', 'Practice Session', selected_topic)
                st.session_state.app_state = 'practice_setup'; st.rerun()

    # --- State Handling for Different Modes ---
    elif st.session_state.app_state == 'diagnostic_setup': run_mode('diagnostic')
    elif st.session_state.app_state == 'diagnostic_running': run_mode('diagnostic')
    elif st.session_state.app_state == 'd_finished': run_mode('diagnostic')

    elif st.session_state.app_state == 'practice_setup': run_mode('practice')
    elif st.session_state.app_state == 'practice_running': run_mode('practice')
    elif st.session_state.app_state == 'p_finished': run_mode('practice')

    elif st.session_state.app_state == 'comprehensive_exam_running': run_comprehensive_exam()
    elif st.session_state.app_state == 'c_finished': run_comprehensive_exam()

    # --- Study Plan States ---
    elif st.session_state.app_state == 'study_plan_setup':
        st.header("Create Your Study Plan")  # Removed emoji
        
        # Add CSS to increase form font size
        st.markdown("""
        <style>
        /* Increase font size for all form elements */
        .stTextInput > label, .stNumberInput > label, .stSlider > label, .stTextArea > label, .stSelectbox > label {
            font-size: 1.2rem !important;
        }
        /* Increase input field text size */
        .stTextInput > div[data-baseweb="input"] input, .stNumberInput > div[data-baseweb="input"] input, 
        .stTextArea textarea, .stSelectbox div[role="listbox"] div[role="option"] {
            font-size: 1.1rem !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        with st.form("study_plan_form"):
            # More concise instructions
            st.write("Enter your exam timeline and study availability.")
            
            # Get subject interest if none selected
            plan_subject = subject_chosen
            if not plan_subject:
                plan_subject = st.selectbox(
                    "Subject",  # Shortened label
                    options=list(AVAILABLE_TOPICS.keys()),
                    key="plan_subject"
                )
            
            # Days until exam - simplified UI
            col1, col2 = st.columns(2)
            with col1:
                days_until_exam = st.number_input(
                    "Days until exam",  # Shortened label
                    min_value=1, max_value=90, value=14
                )
            
            # Hours per day - in second column - increase max to 24
            with col2:
                hours_per_day = st.slider(
                    "Hours per day",  # Shortened label
                    min_value=0.5, max_value=24.0, value=2.0, step=0.5
                )
            
            # Additional notes
            specific_goals = st.text_area(
                "Specific topics to focus on (optional)",  # More specific, clearer label
                height=100  # Control height for more minimal look
            )
            
            # Submit form - more minimal button text
            submitted = st.form_submit_button("Generate Plan")
            
            if submitted:
                with st.spinner("Creating your personalized study plan..."):
                    study_plan = generate_study_plan(
                        subject=plan_subject,
                        days=days_until_exam,
                        hours_per_day=hours_per_day,
                        goals=specific_goals
                    )
                    st.session_state.study_plan = study_plan
                    # Award points for creating a study plan
                    award_points('create_plan')
                    st.session_state.app_state = 'view_study_plan'
                    st.rerun()
        
        # Cancel button outside the form
        if st.button("Cancel", key="cancel_study_plan"):
            st.session_state.app_state = 'welcome'
            st.rerun()
            
    elif st.session_state.app_state == 'view_study_plan':
        st.header("Your Study Plan")  # Removed emoji, simplified header
        
        if not st.session_state.study_plan:
            st.error("No study plan found. Please create one first.")
            if st.button("Create New Plan"):
                st.session_state.app_state = 'study_plan_setup'
                st.rerun()
        else:
            # Display the study plan using markdown (not HTML)
            st.markdown(st.session_state.study_plan)
            
            # Single update button - more minimal
            if st.button("Update Plan", key="update_study_plan", type="primary"):
                st.session_state.app_state = 'study_plan_setup'
                st.rerun()

    # --- Bookmarked Questions State ---
    elif st.session_state.app_state == 'bookmarks':
        st.header("Bookmarked Questions")
        
        if not st.session_state.bookmarked_questions:
            st.info("You haven't bookmarked any questions yet. When viewing questions, click the star (☆) icon to bookmark them for later review.")
        else:
            # Add clear all button
            if st.button("Clear All Bookmarks"):
                st.session_state.bookmarked_questions = []
                st.success("All bookmarks cleared!")
                st.rerun()
            
            # Display all bookmarked questions
            for i, question in enumerate(st.session_state.bookmarked_questions):
                with st.container():
                    st.markdown("---")
                    # Question info/metadata
                    source_text = f"{question.get('source', 'Unknown')} • {question.get('subject', '')} • {question.get('topic', '')}"
                    st.caption(source_text)
                    
                    # Question display
                    col1, col2 = st.columns([0.9, 0.1])
                    with col1:
                        st.markdown(f"<span style='font-size: 1.4em;'>{question.get('question', 'Unknown question')}</span>", unsafe_allow_html=True)
                    with col2:
                        if st.button("✕", key=f"remove_bookmark_{i}"):
                            st.session_state.bookmarked_questions.pop(i)
                            st.success("Bookmark removed")
                            st.rerun()
                    
                    # For MCQ questions, show options and correct answer
                    if question.get('type', 'mcq') == 'mcq':
                        options = question.get('options', [])
                        correct = question.get('correct', '')
                        
                        # Display options with the correct one highlighted
                        for opt in options:
                            is_correct = (opt == correct)
                            if is_correct:
                                st.markdown(f"✓ **{opt}**")
                            else:
                                st.markdown(f"○ {opt}")
                    
                    # For open text questions, just show question
                    else:
                        st.write("(Open response question)")
                        
                    st.markdown("---")
    
    # Fallback State
    else:
        st.warning("An unexpected application state occurred.")
        if st.button("Reset to Welcome"):
             # Clear all mode states
             if 'diagnostic' in st.session_state: del st.session_state['diagnostic']
             if 'practice' in st.session_state: del st.session_state['practice']
             if 'comprehensive_exam' in st.session_state: del st.session_state['comprehensive_exam']
             st.session_state.app_state = 'welcome'
             st.rerun()


# --- Function to generate study plan ---
def generate_study_plan(subject, days, hours_per_day, goals=""):
    """Generates a personalized study plan using the AI model."""
    if not openai_client:
        # Fallback if API not available
        return """
        # Study Plan (Sample - API Unavailable)
        This is a sample study plan. To get a personalized plan, please set up your OpenAI API key.
        
        ## Week 1
        * **Day 1-2:** Review basic concepts using the Practice mode
        * **Day 3-5:** Take diagnostic tests
        """
    
    topics = AVAILABLE_TOPICS.get(subject, [])
    topics_list = ", ".join(topics) if topics else "general topics in this subject"
    
    prompt = f"""
    Create a CONCISE study plan for a student preparing for a {subject} exam in {days} days. 
    They can study approximately {hours_per_day} hours per day.
    
    The key topics in this subject include: {topics_list}
    
    Additional notes from the student: {goals}
    
    IMPORTANT GUIDELINES:
    1. Keep the plan BRIEF and FOCUSED - maximum 500 words total
    2. Focus ONLY on using features of THIS TOOL (Diagnostic Test, Practice Mode, Comprehensive Exam)
    3. DO NOT recommend external resources like books, videos, or lecture notes
    4. Organize by days with clear topic focus for each day
    5. Include specific topics to study each day
    6. Keep language simple and direct - no unnecessary explanations
    
    Format as a short, clear markdown document:
    - Use # for main headings (Week 1, etc.)
    - Use ## for day headings (Day 1, etc.)
    - Use bullet points (*) for activities
    - Be extremely concise - this is a quick reference, not a detailed guide
    
    DO NOT USE HTML. Only use simple markdown that displays properly in Streamlit.
    """
    
    system_msg = "You are a concise study planner who creates brief, focused study schedules. Create short, clear plans with NO fluff, focusing only on using the AI Math Tutor's features (Diagnostic, Practice, Comprehensive Exam)."
    
    model_response = query_openai_model(prompt, system_message=system_msg)
    
    if "error" in model_response or not model_response.get("generated_text"):
        return "# Error Creating Study Plan\nCould not generate a study plan at this time. Please try again later."
    
    plan_text = model_response["generated_text"]
    
    # If the response starts with ```markdown or ```md, strip those tags
    if plan_text.startswith("```markdown") or plan_text.startswith("```md"):
        plan_text = plan_text.split("\n", 1)[1]
    if plan_text.endswith("```"):
        plan_text = plan_text.rsplit("\n", 1)[0]
    
    return plan_text

def display_stats():
    """Display the user's gamification stats in a clean, minimal way."""
    # Calculate level info
    current_level = st.session_state.points // 100
    progress = (st.session_state.points % 100) / 100
    next_level = current_level + 1
    
    # Style just the progress bar
    st.markdown("""
    <style>
    div.stProgress > div > div {
        background-color: #FF9F1C !important;
    }
    div.stProgress {
        height: 15px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create a Streamlit container with an accent color
    container = st.container(border=True)
    
    # Add content to the container - using a more compact layout
    with container:
        st.markdown("#### Your Stats")
        
        # Use simple metric display with plenty of space
        # Display stats in 3 separate rows instead of columns to avoid wrapping
        st.write(f"**Streak:** {st.session_state.streak}")
        st.write(f"**Points:** {st.session_state.points}")
        st.write(f"**Level:** {current_level}")
    
    # Show progress bar outside container
    st.progress(progress)
    st.caption(f"{st.session_state.points % 100}/100 points to Level {next_level}")

# Function to award points for various activities
def award_points(activity, question_correct=False):
    """
    Award points for different activities.
    Activities: 'complete_question', 'complete_diagnostic', 'complete_practice', 'complete_exam', 'create_plan', 'bookmark'
    """
    points_map = {
        'complete_question': 5,  # Base points for attempting a question
        'correct_answer': 10,    # Additional points for correct answer
        'complete_diagnostic': 25,
        'complete_practice': 20,
        'complete_exam': 50,
        'create_plan': 15,
        'bookmark': 2
    }
    
    # Award base points for the activity
    if activity in points_map:
        points = points_map[activity]
        
        # Add bonus for correct answers
        if activity == 'complete_question' and question_correct:
            points += points_map['correct_answer']
            
        st.session_state.points += points
        
        # Record activity to prevent duplicate awards
        activity_id = f"{activity}_{time.time()}"
        st.session_state.completed_activities.append(activity_id)
        
        return activity_id
    
    return None

if __name__ == "__main__":
    main()