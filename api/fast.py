from fastapi import FastAPI
from pydantic import BaseModel

class QuestionGenerateRequest(BaseModel):
    context: str
    num_questions: int | None = 5

class AnswerGenerateRequest(BaseModel):
    context: str
    questions: list

class AnswerScoreRequest(BaseModel):
    question_ids: list
    user_answers: list

app = FastAPI()

@app.get("/ping")
def ping():
    """
    Return a ping if the api is able to respond.
    """
    return {"response": "ping"}

# Question Generation
@app.post("/generate-questions")
async def generate_questions(request: QuestionGenerateRequest):
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

    return request

# Golden Answer Generator from Context & Question
@app.post("/generate-answers")
async def generate_questions(request: AnswerGenerateRequest):
    """ Generate answers

    Generate a golden answer from a question and a context.

    JSON Fields:
    ------------
    `context` (str): The provided context from which an answer should be generated.

    `question` (list): The questions to answer.

    Returns:
    ____________
    `golden_answers` (list): The most likely correct answer to the given question.
    """

    return request

# Answer Scoring
@app.post("/score-answers")
async def generate_questions(request: AnswerScoreRequest):
    """ Generate answers

    Generate a golden answer from a question and a context.

    JSON Fields:
    ------------
    `question_ids` (list): The ids of served questions which should be evaluated.

    `user_answers` (list): The list of user answers to evaluate.

    Returns:
    ____________
    `scores` (list): The scores of the given answers
    """

    return request
