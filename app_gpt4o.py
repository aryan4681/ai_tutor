import streamlit as st
import os
import re
import json
from openai import OpenAI
from dotenv import load_dotenv
import time
import random # Needed for shuffling options if desired

# --- Import Pre-Generated Exam Data ---
# Make sure exam_data.py exists and contains the 'exam_questions' list
try:
    import exam_data
    # Check if the required variable exists in the imported module
    if hasattr(exam_data, 'exam_questions') and isinstance(exam_data.exam_questions, list):
        COMPREHENSIVE_EXAM_AVAILABLE = True
        comprehensive_mcq_questions = exam_data.exam_questions
        # *** IMPORTANT: Update this subject if your exam_data.py covers a different subject ***
        COMPREHENSIVE_SUBJECT = "Linear Algebra"
    else:
        st.sidebar.error("`exam_data.py` found, but 'exam_questions' list is missing or invalid.", icon="🚨")
        COMPREHENSIVE_EXAM_AVAILABLE = False
        comprehensive_mcq_questions = []
        COMPREHENSIVE_SUBJECT = None

except ImportError:
    COMPREHENSIVE_EXAM_AVAILABLE = False
    comprehensive_mcq_questions = []
    COMPREHENSIVE_SUBJECT = None
    # Don't show warning yet, wait until subject is selected


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
AVAILABLE_TOPICS = {
    "Linear Algebra": ["Solving Systems of Linear Equations", "Matrix Operations", "Vector Spaces", "Eigenvalues and Eigenvectors"],
    "Calculus": ["Derivatives", "Integrals", "Limits", "Differential Equations"],
    "Algebra Basics": ["Solving Linear Equations", "Polynomials", "Factoring", "Exponents and Radicals"]
}
HIGHLIGHTED_TOPIC = "Solving Systems of Linear Equations" # Topic to visually highlight

# --- AI Personalities --- MOVED EARLIER ---
AI_PERSONALITIES = {
    "Supportive Tutor (Default)": "You are a supportive, encouraging, and clear math tutor.",
    "Silicon Valley Nerd": "you are a sarcastic, slightly jaded, 'too-cool-for-school' silicon valley nerd explaining math. you always respond in lowercase, never use capitalization. you often use tech jargon ironically or out of context (e.g., 'synergize the coefficients', 'bandwidth issue with factoring'). you are helpful, but in a reluctant, eye-rolling kind of way, yet ultimately provide the correct explanation concisely.",
    "Enthusiastic Cheerleader": "You are an extremely enthusiastic and supportive cheerleader coaching math! Use lots of exclamation points!!! 🎉 Positive vibes only! GO TEAM! You explain concepts clearly but with high energy!",
    "Strict Professor": "You are a formal, strict, and precise mathematics professor. Address the student formally using 'the student'. Be exacting and clear, focusing purely on the mathematical concepts without emotional embellishment or unnecessary pleasantries. Your explanations are direct and rigorous.",
    "Pirate Captain": "Ye be Cap'n Teach, a fearsome pirate swabs who happens to know 'is maths! Explain the concepts like yer explainin' a treasure map, usin' pirate lingo ('arrr', 'shiver me timbers', 'matey', 'booty' for results, etc.). Keep it fun, clear, but decidedly piratical, ye scurvy dog!"
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

def query_openai_model(prompt, system_message="You are a helpful AI assistant specializing in math education."):
    """Sends a prompt to the configured OpenAI model."""
    if not openai_client:
        return {"error": "OpenAI client not initialized."}
    try:
        # Optional delay can be added here if hitting rate limits: time.sleep(0.5)
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7, # Can adjust temperature for different modes
            top_p=0.9,
        )
        return {"generated_text": response.choices[0].message.content}
    except Exception as e:
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
def get_feedback(question, user_answer, personality_name=DEFAULT_PERSONALITY):
    """Gets feedback on the user's answer using the AI model, adapting to personality."""
    system_msg = AI_PERSONALITIES.get(personality_name, AI_PERSONALITIES[DEFAULT_PERSONALITY])
    prompt = (
         f"The user was asked the following question:\nQuestion: {question['question']}\n"
         f"Options: {question['options']}\n"
         f"Correct Answer: {question['correct']}\n"
         f"User's Answer: {user_answer}\n\n"
         "Provide simple, supportive, and concise feedback (max 30 words). "
         "If the user was correct, acknowledge it appropriately for your personality. "
         "If incorrect, state it and perhaps hint towards the correct concept without giving the direct answer away unless the mistake is very simple."
         "Adhere strictly to the persona defined in the system message."
    )
    model_response = query_openai_model(prompt, system_message=system_msg) # Pass dynamic system_msg
    if "error" in model_response or not model_response.get("generated_text"):
        return "Could not get feedback at this time."
    return model_response["generated_text"]

def get_help(question, help_request, personality_name=DEFAULT_PERSONALITY):
    """Gets help related to the current question using the AI model, adapting to personality."""
    system_msg = AI_PERSONALITIES.get(personality_name, AI_PERSONALITIES[DEFAULT_PERSONALITY])
    prompt = (
        f"A student needs help with the following math question:\n"
        f"Question: {question['question']}\n"
        f"Options: {question['options']}\n\n"
        f"Their specific help request is: '{help_request}'\n\n"
        "Provide a concise, helpful hint or explanation related to their request and the question. "
        "Do not directly give away the answer to the multiple-choice question. Focus on the concept they are asking about. Keep the explanation clear and relatively brief (around 30-50 words)."
        "Adhere strictly to the persona defined in the system message."

    )
    model_response = query_openai_model(prompt, system_message=system_msg) # Pass dynamic system_msg
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

def get_final_answer(question):
    """Gets only the final answer for the question."""
    return f"The correct answer is: **{question['correct']}**"

# --- UI Function for Displaying a Single Question ---
# Replace your existing display_question_ui function with this one
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
    q_options = question_obj.get("options")
    q_correct = question_obj.get("correct")

    if not q_text or not q_options or not q_correct or not isinstance(q_options, list) or len(q_options) != 4:
        st.error(f"Error: Question {question_index + 1} data is incomplete or invalid.")
        total_q_key = "total_questions_generated"
        if st.button("Skip Incomplete Question", key=f"{mode_prefix}_skip_incomplete_{question_index}"):
             if question_index + 1 < question_state.get(total_q_key, 0): question_state["current_index"] += 1
             else: question_state["finished"] = True; st.session_state.app_state = f"{mode_prefix}_finished"
             st.rerun()
        return
    # --- End Input Validation ---

    q = question_obj # Use validated object
    # Use st.markdown to potentially render LaTeX in the question text
    st.markdown(f"**Question {question_index + 1}:** {q['question']}")

    # --- Shuffle options for display --- ADDED ---
    # State key to store the shuffled order for this question
    shuffled_options_key = f"{mode_prefix}_shuffled_options_{question_index}"

    if shuffled_options_key not in question_state:
        # Shuffle only if not already done for this question instance
        original_options = list(q_options) # Create a mutable copy
        random.shuffle(original_options)
        question_state[shuffled_options_key] = original_options # Store the shuffled order

    # Use the shuffled list for display
    display_options = question_state.get(shuffled_options_key, list(q_options)) # Fallback just in case
    # -----------------------------------------

    # State keys using prefix for uniqueness
    answer_key = f"{mode_prefix}_answer_{question_index}"
    submitted_key = f"{mode_prefix}_submitted_{question_index}"
    help_visible_key = f"{mode_prefix}_show_help_{question_index}"
    help_input_key = f"{mode_prefix}_help_input_{question_index}"
    help_text_key = f"{mode_prefix}_help_text_{question_index}"
    show_steps_key = f"{mode_prefix}_show_steps_{question_index}"
    step_solution_key = f"{mode_prefix}_step_solution_{question_index}"
    show_final_key = f"{mode_prefix}_show_final_{question_index}"
    final_answer_key = f"{mode_prefix}_final_answer_{question_index}"

    # Initialize state if needed (ensure keys exist in question_state dict)
    # Add initialization for the new shuffled options key just in case, though the logic above handles it
    if shuffled_options_key not in question_state: question_state[shuffled_options_key] = []
    if answer_key not in question_state: question_state[answer_key] = ""
    if submitted_key not in question_state: question_state[submitted_key] = False
    if help_visible_key not in question_state: question_state[help_visible_key] = False
    if help_input_key not in question_state: question_state[help_input_key] = ""
    if help_text_key not in question_state: question_state[help_text_key] = ""
    if show_steps_key not in question_state: question_state[show_steps_key] = False
    if step_solution_key not in question_state: question_state[step_solution_key] = ""
    if show_final_key not in question_state: question_state[show_final_key] = False
    if final_answer_key not in question_state: question_state[final_answer_key] = ""
    if "feedback" not in question_state: question_state["feedback"] = []
    if "score" not in question_state: question_state["score"] = 0

    # Get the currently selected personality
    selected_personality = st.session_state.get('selected_personality', DEFAULT_PERSONALITY)

    # --- Display Radio Buttons and Submit Form ---
    if not question_state[submitted_key]:
        current_selection = question_state.get(answer_key)
        # Determine index based on SHUFFLED options if a selection exists
        try:
            # Find the index of the current selection within the *shuffled* list
            current_index = display_options.index(current_selection) if current_selection in display_options else 0
        except (ValueError, AttributeError):
            current_index = 0 # Default to first option if error or no previous selection

        with st.form(key=f"{mode_prefix}_form_{question_index}"):
            # Use the SHUFFLED list for the radio button options
            question_state[answer_key] = st.radio(
                "Choose your answer:",
                display_options, # Use the shuffled list stored in state
                key=f"{mode_prefix}_radio_{question_index}",
                index=current_index, # Index within the shuffled list
                label_visibility="collapsed"
            )

            submit_pressed = st.form_submit_button("Submit Answer")

            if submit_pressed:
                selected_answer = question_state[answer_key] # This is the text of the selected option

                if selected_answer: # Check if an option was selected
                    question_state[submitted_key] = True

                    # Ensure feedback list has space up to current index
                    while len(question_state["feedback"]) <= question_index:
                         question_state["feedback"].append(None)

                    with st.spinner("Evaluating your answer..."):
                        # Use selected_answer directly
                        # q still holds the original question object with the correct answer text
                        # Pass personality to get_feedback
                        feedback_text = get_feedback(q, selected_answer, selected_personality)

                    question_state["feedback"][question_index] = {
                        "question": q['question'],
                        "user_answer": selected_answer, # Store the actual answer text
                        "correct_answer": q_correct, # Store the original correct answer text
                        "evaluation": feedback_text
                    }

                    # Check correctness using the actual answer text vs original correct text
                    if selected_answer == q_correct:
                        question_state["score"] += 1

                    # Clear other states for this question
                    question_state[show_steps_key] = False; question_state[step_solution_key] = ""
                    question_state[show_final_key] = False; question_state[final_answer_key] = ""
                    question_state[help_visible_key] = False; question_state[help_text_key] = ""
                    # The shuffled_options_key will persist until the user moves to the next question
                    # or the state for this mode is otherwise cleared.
                    st.rerun()
                else:
                    st.warning("Please select an answer before submitting.", icon="👆")

    # --- Display Feedback and Interaction Buttons ---
    if question_state[submitted_key]:
        # Feedback display logic (Use markdown for answers/feedback)
        if question_index < len(question_state["feedback"]) and question_state["feedback"][question_index] is not None:
            last_feedback = question_state["feedback"][question_index]
            st.markdown(f"Your answer: **{last_feedback['user_answer']}**")
            is_correct = last_feedback['user_answer'] == last_feedback['correct_answer']
            if is_correct:
                st.success(f"**Feedback:** {last_feedback['evaluation']}", icon="✅")
            else:
                st.error(f"**Feedback:** {last_feedback['evaluation']}", icon="❌")
        else:
             st.warning("Could not retrieve feedback for this question index.", icon="🤔")


        # Step-by-Step and Final Answer buttons logic
        button_cols = st.columns(2)
        with button_cols[0]:
            if st.button("Step-by-Step Solution", key=f"{mode_prefix}_step_btn_{question_index}"):
                question_state[show_steps_key] = not question_state[show_steps_key]
                question_state[show_final_key] = False
                if question_state[show_steps_key] and not question_state[step_solution_key]:
                    with st.spinner("Generating step-by-step solution..."):
                        # Pass the original question object 'q'
                        # Pass personality to get_step_by_step_solution
                        question_state[step_solution_key] = get_step_by_step_solution(q, selected_personality)
                    st.rerun()
        with button_cols[1]:
            if st.button("Show Final Answer", key=f"{mode_prefix}_final_btn_{question_index}"):
                 question_state[show_final_key] = not question_state[show_final_key]
                 question_state[show_steps_key] = False
                 if question_state[show_final_key] and not question_state[final_answer_key]:
                     # Pass the original question object 'q'
                     # Final answer display doesn't use personality
                     question_state[final_answer_key] = get_final_answer(q)
                 st.rerun()

        # Display Step-by-Step / Final Answer logic
        if question_state[show_steps_key] and question_state[step_solution_key]:
             with st.expander("Step-by-Step Solution", expanded=True):
                  # Use markdown to render the solution steps
                  st.markdown(question_state[step_solution_key], unsafe_allow_html=True)
                  if st.button("Hide Steps", key=f"{mode_prefix}_hide_steps_{question_index}"):
                      question_state[show_steps_key] = False; st.rerun()
        if question_state[show_final_key] and question_state[final_answer_key]:
             # Final answer already includes markdown bold tags from get_final_answer
             st.info(f"{question_state[final_answer_key]}", icon="🎯")
             if st.button("Hide Final Answer", key=f"{mode_prefix}_hide_final_{question_index}"):
                 question_state[show_final_key] = False; st.rerun()

        # Help Section logic
        st.divider()
        if st.button("Ask for help on this question?", key=f"{mode_prefix}_help_btn_{question_index}"):
             question_state[help_visible_key] = not question_state[help_visible_key]
             question_state[help_text_key] = "" # Clear old help
        if question_state[help_visible_key]:
            question_state[help_input_key] = st.text_area(
                 "What specific part of this question or solution do you need help with?",
                 key=f"{mode_prefix}_help_ta_{question_index}",
                 value=question_state.get(help_input_key, "")
                 )
            if st.button("Get Hint", key=f"{mode_prefix}_help_submit_{question_index}"):
                 if question_state[help_input_key].strip():
                     with st.spinner("Asking the assistant for help..."):
                          # Pass the original question object 'q'
                          # Pass personality to get_help
                          question_state[help_text_key] = get_help(q, question_state[help_input_key], selected_personality)
                     st.rerun() # Display hint
                 else:
                      st.warning("Please type your help request first.", icon="✍️")

            if question_state[help_text_key]:
                 st.info(f"**Hint:**\n{question_state[help_text_key]}", icon="💡")
                 if st.button("Clear Hint", key=f"{mode_prefix}_clear_hint_{question_index}"):
                      question_state[help_text_key] = ""
                      question_state[help_input_key] = ""
                      question_state[help_visible_key] = False
                      st.rerun()


        # Next/Finish Buttons logic
        st.divider()
        total_questions = len(question_state.get("questions", []))
        # Define specific finished state for the current mode
        current_mode_finished_state = f"{mode_prefix}_finished"

        if question_index + 1 < total_questions:
            if st.button("Next Question", key=f"{mode_prefix}_next_btn_{question_index}", type="primary"):
                question_state["current_index"] += 1
                # Clear states for next question
                next_idx = question_index + 1
                # Include shuffled_options_key in the keys to clear/reset for the *next* question
                for key_suffix in ["answer", "submitted", "show_help", "help_text", "show_steps", "step_solution", "show_final", "final_answer", "shuffled_options"]:
                    state_key = f"{mode_prefix}_{key_suffix}_{next_idx}"
                    if state_key in question_state:
                        # Reset based on expected type (bool to False, string/list to empty)
                        if isinstance(question_state[state_key], bool):
                           question_state[state_key] = False
                        elif isinstance(question_state[state_key], list):
                           question_state[state_key] = []
                        else:
                           question_state[state_key] = ""
                st.rerun()
        else:
             # Last question: Show Finish button
             finish_button_label = f"Finish {question_state.get('mode_name', 'Session')}"
             if st.button(finish_button_label, key=f"{mode_prefix}_finish_btn", type="primary"):
                 question_state["finished"] = True
                 st.session_state.app_state = current_mode_finished_state # Set specific finished state
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

    # Setup Screen
    if not mode_state["started"] and not mode_state["finished"]:
        st.header(f"{mode_name} Setup: {topic}")
        # Add level indicator if practice mode has an associated level
        current_level = mode_state.get('level')
        if mode_key == 'practice' and current_level:
            st.info(f"Practice Level: **{current_level.split(' - ')[0]}** (based on your diagnostic test)") # Show simple level

        num_questions = st.slider(f"Number of questions for {mode_name}", min_value=1, max_value=10, value=mode_state.get("num_questions_selected", 2), key=f"{mode_prefix}_num_q")
        mode_state["num_questions_selected"] = num_questions
        if st.button(f"Start {mode_name}", key=f"{mode_prefix}_start_btn", type="primary"):
            if not openai_client: st.error("Cannot start: OpenAI client not available.", icon="🚫"); return

            level_to_generate = mode_state.get('level') # Get level if set
            spinner_msg = f"Generating {num_questions} questions on '{topic}'..."
            if level_to_generate:
                 spinner_msg = f"Generating {num_questions} {level_to_generate.split(' - ')[0]} level questions on '{topic}'..."

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
                 st.error(f"Could not generate the requested number of questions for {mode_name}. Please try again.", icon="🚫")
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

        if total_questions > 0:
             score = mode_state['score']
             st.write(f"Your score: {score} out of {total_questions}")
             try:
                 percentage = (score / total_questions) * 100
                 st.progress(percentage / 100); st.write(f"{percentage:.1f}% correct")
             except ZeroDivisionError: percentage = 0; st.write("Cannot calculate percentage.")

             if mode_key == 'diagnostic':
                 if percentage < 40: category = "Beginner - Needs foundational review"
                 elif percentage < 75: category = "Intermediate - Good grasp, needs practice"
                 else: category = "Advanced - Strong understanding"
                 st.metric("Suggested Level", category)
                 mode_state['diagnostic_level'] = category # <-- STORE the diagnosed level

             # --- Get and Display Topic Suggestions --- Added Section ---
             st.markdown("--- Suggestions ---") # Separator
             suggestion = get_topic_suggestions(
                 subject=st.session_state.selected_subject, # Assuming subject is in session state
                 feedback_list=mode_state.get("feedback", []),
                 questions_list=mode_state.get("questions", []),
                 overall_percentage=percentage,
                 mode_type=mode_key, # 'diagnostic' or 'practice'
                 current_topic=mode_state.get("topic") # Pass the current topic
             )
             st.info(suggestion)
             st.markdown("--- Review --- ") # Separator
             # ----------------------------------------------------------

             with st.expander("Review Your Answers and Feedback"):
                  feedback_list = mode_state.get("feedback", [])
                  questions_list = mode_state.get("questions", [])
                  if not feedback_list or all(fb is None for fb in feedback_list):
                       st.write("No feedback recorded.")
                  else:
                      for i, q_data in enumerate(questions_list):
                           # Ensure q_data is a dict before accessing keys
                           q_text = q_data.get('question','N/A') if isinstance(q_data, dict) else "Invalid Question Data"
                           st.markdown(f"**Q{i+1}: {q_text}**") # Use markdown
                           if i < len(feedback_list) and feedback_list[i] is not None:
                               fb = feedback_list[i]
                               st.markdown(f"Your Answer: {fb['user_answer']}") # Use markdown
                               correct = fb['user_answer'] == fb['correct_answer']
                               if correct: st.success(f"Evaluation: {fb['evaluation']}", icon="✅")
                               else: st.error(f"Evaluation: {fb['evaluation']}", icon="❌"); st.info(f"Correct Answer: **{fb['correct_answer']}**", icon="💡") # Use markdown for correct answer
                           else: st.write("_(No answer submitted for this question)_")
                           st.divider()
        else: st.warning("No questions were attempted in this session.")

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
                  if st.button(f"✍️ Practice at Your Level ({simple_level_name})", key=f"d_practice_adaptive_btn"):
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
def initialize_comprehensive_exam_state(subject, mcq_questions):
    """Initializes state for the comprehensive exam using pre-generated MCQs."""
    mode_key = 'comprehensive_exam'
    if mode_key in st.session_state:
        del st.session_state[mode_key]

    # Basic validation of the input MCQs
    if not mcq_questions or not isinstance(mcq_questions, list):
         st.error(f"Cannot start Comprehensive Exam: No valid questions found for {subject} in exam_data.py.")
         return False # Indicate failure

    valid_mcqs = []
    for i, q in enumerate(mcq_questions):
         if (isinstance(q, dict) and
             all(k in q for k in ['question', 'options', 'correct']) and
             isinstance(q.get('options'), list) and len(q['options']) == 4 and
             q.get('correct') in q.get('options', [])):
             valid_mcqs.append(q)
         else:
              st.warning(f"Invalid MCQ format found in exam_data.py for question index {i}. Skipping.", icon="⚠️")

    if not valid_mcqs:
         st.error(f"Cannot start Comprehensive Exam: No valid MCQs loaded for {subject}.")
         return False # Indicate failure

    st.session_state[mode_key] = {
        "mode_name": f"Comprehensive Exam: {subject}", "subject": subject,
        "started": True, # Started immediately as questions are pre-loaded
        "finished": False,
        "questions": valid_mcqs, # Use the validated, pre-generated MCQs
        "total_questions_generated": len(valid_mcqs), # Use count of valid questions
        "current_index": 0, "score": 0,
        "feedback": [None] * len(valid_mcqs), # Initialize feedback list size
    }
    # print(f"DEBUG: Initialized comprehensive exam state with {len(valid_mcqs)} MCQs.") # Debugging
    return True # Indicate success


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
        total_questions = mode_state.get("total_questions_generated", 0)
        if total_questions > 0:
            score = mode_state['score']
            st.write(f"Your score: {score} out of {total_questions}")
            try:
                percentage = (score / total_questions) * 100
                st.progress(percentage / 100); st.write(f"{percentage:.1f}% correct")
            except ZeroDivisionError: st.write("Cannot calculate percentage.")

            # --- Get and Display Topic Suggestions --- Added Section ---
            st.markdown("--- Suggestions ---") # Separator
            suggestion = get_topic_suggestions(
                subject=mode_state.get("subject"),
                feedback_list=mode_state.get("feedback", []),
                questions_list=mode_state.get("questions", []),
                overall_percentage=percentage,
                mode_type='comprehensive', # Explicitly set mode type
                current_topic=None # Not applicable for comprehensive
            )
            st.info(suggestion)
            st.markdown("--- Review --- ") # Separator
            # ----------------------------------------------------------

            with st.expander("Review Your Answers and Feedback"):
                 feedback_list = mode_state.get("feedback", [])
                 questions_list = mode_state.get("questions", [])
                 if not feedback_list or all(fb is None for fb in feedback_list):
                       st.write("No feedback recorded.")
                 else:
                      for i, q_data in enumerate(questions_list):
                           q_text = q_data.get('question','N/A') if isinstance(q_data, dict) else "Invalid Question Data"
                           st.markdown(f"**Q{i+1}: {q_text}**") # Use markdown
                           if i < len(feedback_list) and feedback_list[i] is not None:
                               fb = feedback_list[i]
                               st.markdown(f"Your Answer: {fb['user_answer']}") # Use markdown
                               correct = fb['user_answer'] == fb['correct_answer']
                               if correct: st.success(f"Evaluation: {fb['evaluation']}", icon="✅")
                               else: st.error(f"Evaluation: {fb['evaluation']}", icon="❌"); st.info(f"Correct Answer: **{fb['correct_answer']}**", icon="💡") # Use markdown
                           else: st.write("_(No answer submitted for this question)_")
                           st.divider()
        else:
            st.warning("No valid questions were loaded or attempted in this comprehensive exam.")

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
        return "✅ Great job! You have a strong grasp of this material. Consider reviewing or trying a different subject/topic."

    # For single-topic modes, suggest the current topic if score isn't high
    if mode_type in ['diagnostic', 'practice'] and current_topic:
        return f"💡 Recommendation: Continue practicing on **{current_topic}**."

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
             return "✅ Overall performance seems okay, but review any areas you felt unsure about."

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
             return f"💡 Suggestion: Review the incorrect answers above to identify areas needing more practice in {subject}."
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
                return f"💡 Areas to focus on for {subject}: **{suggested_topics_text}**"
             else:
                # AI might have hallucinated topics or provided an explanation instead of just names
                return f"💡 Suggestion: Review the incorrect answers above to identify areas needing more practice in {subject}. (Could not automatically identify specific topics)."


    # Fallback for any unexpected cases
    return "Review your incorrect answers to identify areas for improvement."


# --- Main App Logic ---
def main():
    st.set_page_config(layout="wide")
    st.title("🧠 AI Math Tutor")

    # Initialize Session State variables
    if 'app_state' not in st.session_state: st.session_state.app_state = 'welcome'
    if 'selected_subject' not in st.session_state: st.session_state.selected_subject = None
    if 'selected_topic' not in st.session_state: st.session_state.selected_topic = None
    # Initialize personality state - ADDED
    if 'selected_personality' not in st.session_state:
        st.session_state.selected_personality = DEFAULT_PERSONALITY

    # --- Sidebar ---
    st.sidebar.header("🎯 Select Your Focus")
    if not openai_client:
        st.sidebar.error("OpenAI API key missing or invalid.", icon="🚨")
        st.warning("AI Tutor features (Diagnostic, Practice, Help, Feedback) are disabled.", icon="⚠️")
        # Allow app to run for Comprehensive Exam if available, but warn AI features are off
        # st.stop() # Don't stop if comprehensive is available
    else:
         # Optional: Indicate client is ready
         # st.sidebar.success("OpenAI Client Ready.", icon="✅")
         pass

    # --- Personality Selector --- ADDED ---
    st.sidebar.divider()
    st.sidebar.subheader("🧑‍🏫 Choose Your Tutor's Style")
    personality_list = list(AI_PERSONALITIES.keys())
    current_personality = st.session_state.selected_personality
    # Ensure current selection is valid, default if not
    if current_personality not in personality_list:
        current_personality = DEFAULT_PERSONALITY
        st.session_state.selected_personality = current_personality
    personality_index = personality_list.index(current_personality)

    new_personality = st.sidebar.selectbox(
        "AI Personality:",
        options=personality_list,
        index=personality_index,
        key='personality_selector'
    )
    if new_personality != st.session_state.selected_personality:
        st.session_state.selected_personality = new_personality
        # Optional: could add a small success message temporarily
        # st.sidebar.success(f"Tutor personality set to: {new_personality}")
        # No rerun needed unless it affects current display immediately,
        # which it doesn't until help/feedback is requested.
    st.sidebar.divider()
    # -----------------------------

    # 1. Subject Selection
    subjects = list(AVAILABLE_TOPICS.keys())
    default_subject_index = subjects.index(st.session_state.selected_subject) if st.session_state.selected_subject in subjects else 0
    selected_subject = st.sidebar.selectbox(
         "1. Select Subject:", options=subjects, index=default_subject_index, key='subject_selector'
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
        display_topics = [f"{t} ✨" if t == HIGHLIGHTED_TOPIC else t for t in topics]
        display_topics.insert(0, "--- Select a Topic ---")

        current_topic_display = None
        if st.session_state.selected_topic:
             current_topic_display = f"{st.session_state.selected_topic} ✨" if st.session_state.selected_topic == HIGHLIGHTED_TOPIC else st.session_state.selected_topic
        current_index = display_topics.index(current_topic_display) if current_topic_display in display_topics else 0

        selected_display_topic = st.sidebar.selectbox(
            "2. Select Topic:", options=display_topics, index=current_index, key='topic_selector'
            )

        # Handle topic change/selection
        if selected_display_topic != "--- Select a Topic ---":
            actual_topic = selected_display_topic.replace(" ✨", "")
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


        # --- Comprehensive Exam Button (Sidebar) ---
        st.sidebar.divider()
        st.sidebar.subheader("Full Subject Exam")
        # Check if data exists AND the current subject matches the data's subject
        if COMPREHENSIVE_EXAM_AVAILABLE and subject_chosen == COMPREHENSIVE_SUBJECT:
            if comprehensive_mcq_questions: # Check if questions were loaded ok
                if st.sidebar.button(f"📚 Start Comprehensive Exam ({subject_chosen})", key="start_comp_exam_btn"):
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
                 st.sidebar.info(f"Comprehensive exam only available for {COMPREHENSIVE_SUBJECT}.")
        st.sidebar.divider()
    # --- End of Sidebar Logic ---


    # --- Main Area Content based on App State ---

    # Welcome State
    if st.session_state.app_state == 'welcome':
        st.header("Welcome to the AI Math Tutor!")
        st.write("Please select a **Subject** from the sidebar.")
        if subject_chosen:
             st.info("Now select a specific **Topic** for AI-powered Diagnostic/Practice modes, or start the pre-defined **Comprehensive Exam** for the whole subject (if available).")
        else:
             st.info("Once a subject is selected, further options will appear.")

    # Topic Selected State (Topic-Specific Modes)
    elif st.session_state.app_state == 'topic_selected' and selected_topic:
        st.header(f"Subject: {subject_chosen} | Topic: {selected_topic}")
        st.write("Choose an AI-powered mode for this topic:")
        st.write("")
        # Diagnostic Button
        if st.button(f"🎓 Diagnostic Test ", key="start_diagnostic_btn", help=f"Assess knowledge on {selected_topic} with AI questions"):
            initialize_mode_state('diagnostic', 'Diagnostic Test', selected_topic)
            st.session_state.app_state = 'diagnostic_setup'; st.rerun()

        st.divider()
        # Practice Session Button (Topic-Specific)
        st.subheader(f"Practice: {selected_topic}")
        if st.button(f"✍️ Practice Session ", key="start_practice_topic_btn", help=f"Practice problems on {selected_topic} with AI questions", type="primary"):
            initialize_mode_state('practice', 'Practice Session', selected_topic)
            st.session_state.app_state = 'practice_setup'; st.rerun()
        st.caption(f"Practice AI-generated questions on the selected topic: {selected_topic}.")


    # --- State Handling for Different Modes ---
    elif st.session_state.app_state == 'diagnostic_setup': run_mode('diagnostic')
    elif st.session_state.app_state == 'diagnostic_running': run_mode('diagnostic')
    elif st.session_state.app_state == 'd_finished': run_mode('diagnostic')

    elif st.session_state.app_state == 'practice_setup': run_mode('practice')
    elif st.session_state.app_state == 'practice_running': run_mode('practice')
    elif st.session_state.app_state == 'p_finished': run_mode('practice')

    elif st.session_state.app_state == 'comprehensive_exam_running': run_comprehensive_exam()
    elif st.session_state.app_state == 'c_finished': run_comprehensive_exam()

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


if __name__ == "__main__":
    main()