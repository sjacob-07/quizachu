from fastapi import FastAPI
from pydantic import BaseModel
from quizachu.generate.model import create_generate_model, create_generate_tokenizer, generate_questions
from quizachu.answer.model import create_question_answerer, answer_questions_with_confidence, select_top_n_questions
from quizachu.score.model import create_generate_score_model, check_answer_similarity
from typing import Optional

import time
import more_itertools as mit

class QuestionGenerateRequest(BaseModel):
    context: str
    allow_duplicates: Optional[bool] = False

class AnswerGenerateRequest(BaseModel):
    context: str
    questions: list

class AnswerScoreRequest(BaseModel):
    sentence1: list = []
    sentence2: list = []

app = FastAPI()
app.state.generate_model = None
app.state.generate_tokenizer = None
app.state.question_answerer = None
app.state.score_model = None

@app.get("/ping")
def ping():
    """
    Return a ping if the api is able to respond.
    """
    return {"response": "ping"}

# Question Generation
@app.post("/generate-questions")
async def generate_questions_api(request: QuestionGenerateRequest):
    """ Generate questions

    Generate a list of questions from a given context.

    JSON Fields:
    ------------
    `context` (str): The provided context from which questions should be generated.

    `num_questions` (int, optional): The number of questions to return.

    Returns:
    ____________
    `questions` (list): A list of `str` questions generated from the context of length `num_questions`.
    """

    if not app.state.generate_model:
        app.state.generate_model = create_generate_model()

    if not app.state.generate_tokenizer:
        app.state.generate_tokenizer = create_generate_tokenizer()

    tokenizer = app.state.generate_tokenizer
    model = app.state.generate_model

    questions = generate_questions(model, tokenizer, request.context, 10)

    return questions

# Golden Answer Generator from Context & Question
@app.post("/generate-answers")
async def generate_answers_api(request: AnswerGenerateRequest):
    """ Generate answers
`
    Generate a golden answer from a question and a context.

    JSON Fields:
    ------------
    `context` (str): The provided context from which an answer should be generated.

    `question` (list): The questions to answer.

    Returns:
    ____________
    `golden_answers` (list): The most likely correct answer to the given question.
    """

    if not app.state.question_answerer:
        app.state.question_answerer = create_question_answerer()

    response = select_top_n_questions(app.state.question_answerer, request.context, request.questions)

    return response.to_dict()

@app.post("/generate-questions-and-answers")
async def generate_questions_and_answers_api(request: QuestionGenerateRequest):

    print(request.allow_duplicates)

    start = time.time()

    if not app.state.generate_model:
        app.state.generate_model = create_generate_model()

    if not app.state.generate_tokenizer:
        app.state.generate_tokenizer = create_generate_tokenizer()

    if not app.state.question_answerer:
        app.state.question_answerer = create_question_answerer()

    tokenizer = app.state.generate_tokenizer
    model = app.state.generate_model

    context_length = len(request.context.split())

    questions_lists = []

    # Scale the number of questions/answers generated according to the context length
    # Add another question per 150 words of context
    n_questions = 4 + context_length // 150
    print(n_questions)

    if context_length > 450:
        n_chunks = n_questions
        # The width of each chunk should be n_chunks - 2 (to allow overlapping)
        width_factor = max(n_chunks - 2, 2)

        chunks = [' '.join(window) for window in mit.windowed(request.context.split(), n=context_length//width_factor, step=context_length//n_chunks, fillvalue="")]

        for chunk in chunks:
            chunk_questions = generate_questions(model, tokenizer, chunk, 4)
            questions_lists.append(chunk_questions)

    else:
        questions_lists.append(generate_questions(model, tokenizer, request.context, n_questions*4))

    check1 = time.time()
    print(f"Question generation time: {check1 - start}")

    questions = []
    for l in questions_lists:
        for q in l:
            # Do not append to the list if the question is an empty string
            if q:
                questions.append(q)

    max_repeat_exact_answers=1
    if request.allow_duplicates:
        max_repeat_exact_answers=2

    response = select_top_n_questions(app.state.question_answerer,
                                    request.context,
                                    questions,
                                    c=0.05,
                                    n=n_questions,
                                    max_repeat_exact_answers=max_repeat_exact_answers)

    response.drop(columns=["original_question_number"], inplace=True)
    response.columns = ["confidence_score", "questions", "answers"]
    check2 = time.time()
    print(f"Answer generation time: {check2 - check1}")
    print(f"Total execution time: {check2 - start}")

    return response.to_dict()


# Answer Scoring
@app.post("/score-answers")
async def generate_scores_api(request: AnswerScoreRequest):
    """ Generate answers

    Generate a golden answer from a question and a context.

    JSON Fields:
    ------------
    `sentence1` (list): The golden answer for a question.

    `sentence2` (list): The user answer that needs to be evaluted

    Returns:
    ____________
    `results` (dict): The predication and probability of the given answer
    """
    if not app.state.score_model:
        score_model = create_generate_score_model()

    results = check_answer_similarity(score_model, request.sentence1, request.sentence2)

    return results
