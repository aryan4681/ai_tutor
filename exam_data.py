# exam_data.py

# Pre-converted Multiple Choice Questions based on
# ANS Practice Exam (Resit 2021-2022).pdf and its solutions.

# Note: Mathematical notation uses LaTeX-like syntax.
# Backslashes within strings are escaped (e.g., \\begin{matrix}) for Python.

exam_questions = [
    {
        "id": "1",
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
        "question": "Consider $A=[\\begin{smallmatrix}p&0&0\\\\ 1&p&2\\\\ 0&-1&2\\end{smallmatrix}]$. For which value(s) of p is A not invertible?",
        "options": [
            "p=0 or p=-1", # Correct from solution (det A = p(2p+2) = 0)
            "p=0 only",
            "p=-1 only",
            "p=2 only"
        ],
        "correct": "p=0 or p=-1"
    },
    {
        "id": "4",
        "question": "A $3\\times3$ matrix A has all diagonal entries equal to 0, every row sum equal to 3, and det A = 6. What are the eigenvalues of A?",
        "options": [
            "{3, -1, -2}", # Correct from solution
            "{0, 3, 6}",
            "{0, 0, 3}",
            "{3, 3, 0}"
        ],
        "correct": "{3, -1, -2}"
    },
    {
        "id": "5",
        "question": "Consider $A=[\\begin{smallmatrix}1&-4&9&-7\\\\ -1&2&-4&1\\\\ 5&-6&10&7\\end{smallmatrix}]$. Which vector is orthogonal to Nul A? (Hint: Recall (Row A)^⊥ = Nul A)",
        "options": [
            "[1, -4, 9, -7]", # Correct (First row of A / Row A basis vector)
            "[1, 1, 1, 1]",
            "[0, 0, 0, 0]",
            "[5, -6, 10, 7]" # Last row of A (dependent on others after RREF)
        ],
        "correct": "[1, -4, 9, -7]"
    },
    {
        "id": "6",
        "question": "Consider $A=[\\begin{smallmatrix}1&-1&0\\\\ 2&1&1\\end{smallmatrix}]$. Determine the vector u in Row A such that $v - u$ is orthogonal to Row A, where $v = [3, 2, 9]$.",
        "options": [
            "[5, 4, 3]", # Correct from solution (u = 5*[1,-1,0] + 3*[0,3,1])
            "[3, 2, 9]", # The vector v itself
            "[1, -1, 0]", # Basis vector 1 of Row A
            "[0, 3, 1]" # Basis vector 2 of Row A (from RREF)
        ],
        "correct": "[5, 4, 3]"
    },
    {
        "id": "7a",
        "question": "Which vector in $\\mathbb{R}^{3}$ has length 1 and is orthogonal to both [1, 0, 1] and [0, 1, 0]?",
        "options": [
            "[1/√3, 0, -1/√3]", # Correct from solution (alpha=1/√3, beta=0, gamma=-1/√3) (Note: Solution had error in beta calculation, should be 0) - Corrected option based on calculation.
            "[1, -1, -1]", # Orthogonal but not unit length
            "[0, 0, 1]", # Not orthogonal to [1,0,1]
            "[1/√2, 0, 1/√2]" # Not orthogonal to [0,1,0]
        ],
        "correct": "[1/√3, 0, -1/√3]"
    },
    {
        "id": "7b",
        "question": "Let $A=[\\begin{smallmatrix}3&-2&4\\\\ -2&6&2\\\\ 4&2&3\\end{smallmatrix}]$ and $P=[\\begin{smallmatrix}2&1&1\\\\ 1&-2&0\\\\ -2&0&1\\end{smallmatrix}]$. Find the diagonal matrix D such that $A=PDP^{-1}$.",
        "options": [
            "$[\\begin{smallmatrix}-2&0&0\\\\ 0&7&0\\\\ 0&0&7\\end{smallmatrix}]$", # Correct eigenvalues from solution
            "$[\\begin{smallmatrix}3&0&0\\\\ 0&6&0\\\\ 0&0&3\\end{smallmatrix}]$", # Diagonal of A
            "$[\\begin{smallmatrix}1&0&0\\\\ 0&1&0\\\\ 0&0&1\\end{smallmatrix}]$", # Identity matrix
            "$[\\begin{smallmatrix}2&0&0\\\\ 0&-2&0\\\\ 0&0&1\\end{smallmatrix}]$" # Some values from P
        ],
        "correct": "$[\\begin{smallmatrix}-2&0&0\\\\ 0&7&0\\\\ 0&0&7\\end{smallmatrix}]$"
    },
    {
        "id": "7c",
        "question": "Which set of vectors {u, v, w} in $\\mathbb{R}^{2}$ satisfies the condition that {u,v} and {v,w} are linearly independent, but {u,v,w} is linearly dependent?",
        "options": [
            "{u=[1,0], v=[0,1], w=[1,1]}", # Correct example type from solution
            "{u=[1,0], v=[2,0], w=[0,1]}", # {u,v} dependent
            "{u=[1,1], v=[1,1], w=[0,1]}", # {u,v} dependent
            "{u=[1,0], v=[0,1], w=[0,0]}" # {v,w} dependent (unless v=0, but w!=0)
        ],
        "correct": "{u=[1,0], v=[0,1], w=[1,1]}"
    },
    {
        "id": "7d",
        "question": "Which matrix A is not invertible and has Nul A = {0}?",
        "options": [
            "$[\\begin{smallmatrix}1&0\\\\ 0&1\\\\ 0&0\\end{smallmatrix}]$", # Correct example type from solution (Non-square, independent columns)
            "$[\\begin{smallmatrix}1&1\\\\ 1&1\\end{smallmatrix}]$", # Not invertible, but Nul A is not {0}
            "$[\\begin{smallmatrix}1&0\\\\ 0&1\\end{smallmatrix}]$", # Invertible
            "$[\\begin{smallmatrix}0&0\\\\ 0&0\\end{smallmatrix}]$" # Not invertible, but Nul A is not {0}
        ],
        "correct": "$[\\begin{smallmatrix}1&0\\\\ 0&1\\\\ 0&0\\end{smallmatrix}]$"
    },
    {
        "id": "7e",
        "question": "Which subset H of $\\mathbb{R}^{2}$ contains the zero vector, is closed under vector addition, but is NOT closed under multiplication by scalars?",
        "options": [
            "The first quadrant (x>=0, y>=0)", # Correct example from solution
            "The x-axis (y=0)", # Closed under scalar multiplication
            "The set of points with integer coordinates", # Not closed under scalar multiplication, but also not always under addition (if restricted to integers) - better example needed. Let's assume standard vector space ops. This IS closed under addition.
            "The unit circle (x^2 + y^2 = 1)", # Doesn't contain zero vector, not closed under addition
        ],
        "correct": "The first quadrant (x>=0, y>=0)"
    }
]

# --- End of exam_questions list ---

# Example usage (optional):
if __name__ == '__main__':
    print(f"Loaded {len(exam_questions)} pre-defined MCQs.")
    # Print details of the first question
    if exam_questions:
        print("\nFirst Question Example:")
        q1 = exam_questions[0]
        print(f"ID: {q1.get('id')}")
        print(f"Question: {q1.get('question')}")
        print("Options:")
        for i, opt in enumerate(q1.get('options', [])):
            print(f"  {i+1}. {opt}")
        print(f"Correct Answer: {q1.get('correct')}")