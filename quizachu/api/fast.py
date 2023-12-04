from fastapi import FastAPI
from pydantic import BaseModel
from quizachu.generate.model import create_generate_model, create_generate_tokenizer, generate_questions

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
app.state.generate_model = create_generate_model()
app.state.generate_tokenizer = create_generate_tokenizer()

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

    tokenizer = app.state.generate_tokenizer
    model = app.state.generate_model

    questions = generate_questions(model, tokenizer, request.context, request.num_questions)

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

    return request

# Answer Scoring
@app.post("/score-answers")
async def generate_scores_api(request: AnswerScoreRequest):
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
