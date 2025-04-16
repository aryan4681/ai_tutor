# demo_exam_data.py

# Pre-converted Multiple Choice Questions based on
# ANS Practice Exam (Resit 2021-2022).pdf and its solutions.
# Modified for demo purposes with an open question.

# Note: Mathematical notation uses LaTeX-like syntax.
# Backslashes within strings are escaped (e.g., \\\\begin{matrix}) for Python.

exam_questions = [
    {
        "id": "open_1",
        "type": "open_text", # Mark as open-ended text question
        "question": "Explain in your own words the concept of linear independence for a set of vectors in $\\mathbb{R}^n$. Provide a simple example of a linearly dependent set and a linearly independent set in $\\mathbb{R}^2$.",
        # No 'options' or 'correct' key needed for open questions
    },
    {
        "id": "1",
        "type": "mcq", # Explicitly mark type
        "question": "Let $a_{1}=[\\begin{smallmatrix}1\\\\ -2\\\\ 0\\end{smallmatrix}], a_{2}=[\\begin{smallmatrix}0\\\\ 1\\\\ 2\\end{smallmatrix}], a_{3}=[\\begin{smallmatrix}5\\\\ -6\\\\ 8\\end{smallmatrix}]$, and $b=[\\begin{smallmatrix}2\\\\ -1\\\\ 6\\end{smallmatrix}]$. Which of the following expresses b as a linear combination $c_1a_1 + c_2a_2 + c_3a_3$?",
        "options": [
            "$b = 2a_{1} + 3a_{2} + 0a_{3}$", # Correct from solution example 1
            "$b = a_{1} + a_{2} + a_{3}$",
            "$b = 0a_{1} + 0a_{2} + 0a_{3}$",
            "$b = -3a_{1} - a_{2} - a_{3}$" # Incorrect coefficients based on solution example 2
        ],
        "correct": "$b = 2a_{1} + 3a_{2} + 0a_{3}$"
    },
    {
        "id": "2",
        "type": "mcq", # Explicitly mark type
        "question": "Let $T:\\mathbb{R}^{3}\\rightarrow\\mathbb{R}^{5}$ be a mapping with T([1,0,0])=[-3,1,2,0,5], T([0,1,0])=[0,1,1,2,0], and T([0,0,1])=[0,6,0,0,1]. Can T be a linear transformation?",
        "options": [
            "No, because T(0) is not the zero vector in R^5.", # Correct reason from solution
            "Yes, because T is defined for the standard basis vectors.",
            "No, because the dimensions of the domain and codomain are different.",
            "Yes, because the images of the basis vectors are given."
        ],
        "correct": "No, because T(0) is not the zero vector in R^5."
    },
    {
        "id": "3",
        "type": "mcq", # Explicitly mark type
        "question": "Consider $A=[\\begin{smallmatrix}p&0&0\\\\ 1&p&2\\\\ 0&-1&2\\end{smallmatrix}]$. For which value(s) of p is A not invertible?",
        "options": [
            "p=0 or p=-1", # Correct from solution (det A = p(2p+2) = 0)
            "p=0 only",
            "p=-1 only",
            "p=2 only"
        ],
        "correct": "p=0 or p=-1"
    }
]

# --- End of exam_questions list ---

# Example usage (optional):
if __name__ == '__main__':
    print(f"Loaded {len(exam_questions)} pre-defined demo MCQs.")
    # Print details of the first question
    if exam_questions:
        q1 = exam_questions[0]
        print("\nFirst Question Example:")
        print(f"ID: {q1.get('id')}")
        print(f"Type: {q1.get('type')}")
        print(f"Question: {q1.get('question')}")
        if q1.get('type') == 'mcq':
             print("Options:")
             for i, opt in enumerate(q1.get('options', [])):
                  print(f"  {i+1}. {opt}")
             print(f"Correct Answer: {q1.get('correct')}")

        # Print details of the open question
        q_open = next((q for q in exam_questions if q.get('type') == 'open_text'), None)
        if q_open:
            print("\nOpen Question Example:")
            print(f"ID: {q_open.get('id')}")
            print(f"Type: {q_open.get('type')}")
            print(f"Question: {q_open.get('question')}")
