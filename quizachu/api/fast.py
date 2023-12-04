from fastapi import FastAPI
from pydantic import BaseModel
from quizachu.generate.model import create_generate_model, create_generate_tokenizer, generate_questions
from quizachu.answer.model import create_question_answerer, answer_questions_with_confidence, select_top_n_questions

class QuestionGenerateRequest(BaseModel):
    context: str

class AnswerGenerateRequest(BaseModel):
    context: str
    questions: list

class AnswerScoreRequest(BaseModel):
    question_ids: list
    user_answers: list

app = FastAPI()
app.state.generate_model = create_generate_model()
app.state.generate_tokenizer = create_generate_tokenizer()
app.state.question_answerer = create_question_answerer()

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

    questions = generate_questions(model, tokenizer, request.context, 100)

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

    response = select_top_n_questions(app.state.question_answerer, request.context, request.questions)

    return response.to_dict()

@app.post("/generate-questions-and-answers")
async def generate_questions_and_answers_api(request: QuestionGenerateRequest):

    tokenizer = app.state.generate_tokenizer
    model = app.state.generate_model

    questions = generate_questions(model, tokenizer, request.context, 50)

    response = select_top_n_questions(app.state.question_answerer, request.context, questions)
    response.drop(columns=["original_question_number"], inplace=True)
    response.columns = ["confidence_score", "questions", "answers"]

    return response.to_dict()


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
