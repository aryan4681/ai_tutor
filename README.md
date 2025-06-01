# AI Math Tutor (Thesis Project)

This repository contains the source code for an AI Math Tutor, developed as a thesis project. The system leverages OpenAI's GPT-4o model to provide a personalized and interactive learning experience for students, primarily focusing on mathematics.

## Overview

The AI Math Tutor aims to simulate a tutoring environment by offering various modes of interaction, including diagnostic tests, practice sessions, comprehensive exams, and on-demand help. It features adaptive feedback, question generation, and a gamified learning experience to engage students.

## Key Features

*   **Multiple Learning Modes:**
    *   Diagnostic Tests to assess initial understanding.
    *   Targeted Practice Sessions based on topics or diagnosed levels.
    *   Comprehensive Exams for overall assessment.
*   **AI-Powered Interactions:**
    *   Dynamic question generation.
    *   Personalized feedback on answers.
    *   Step-by-step solution explanations.
    *   Conversational help chat.
*   **Personalization & Engagement:**
    *   Selectable AI tutor personalities.
    *   Bookmarkable questions.
    *   Flashcard system with spaced repetition.
    *   Gamification elements (points, streaks, levels).
    *   Study plan generation.
*   **Multimedia Support:**
    *   Handles image-based answers for open-ended questions.
    *   Integration of relevant educational videos (e.g., 3Blue1Brown).
*   **Tools:**
    *   Embedded Desmos graphing calculator.

## Technologies Used

*   **Backend:** Python, Streamlit
*   **AI Model:** OpenAI GPT-4o API
*   **Key Python Libraries:**
    *   `streamlit` for the web application interface.
    *   `openai` for interacting with the GPT model.
    *   `sympy` for mathematical question verification (select topics).
    *   `dotenv` for environment variable management.

## Setup

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd ai_tutor_fresh
    ```
2.  Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Set up your OpenAI API Key:
    *   Create a `.env` file in the project root.
    *   Add your API key: `OPENAI_API_KEY='your_api_key_here'`
    *   Alternatively, set it up as a Streamlit secret if deploying.

## Usage

Run the Streamlit application:
```bash
streamlit run app_gpt4o.py
```
Navigate to the displayed local URL in your web browser to use the AI Tutor.

