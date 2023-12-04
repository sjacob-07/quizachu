# quizachu-qna-selector/app/api.py
from fastapi import APIRouter
from .utils import answer_questions_with_confidence, select_top_n_questions

router = APIRouter()

@router.post("/answer_questions_with_confidence")
def answer_questions_with_confidence_endpoint(article: str, questions: list):
    return answer_questions_with_confidence(article, questions)

@router.post("/return_top_n_questions")
def select_top_n_questions_endpoint(article: str, questions: list, c: float = 0.5, n: int = 5):
    return select_top_n_questions(article, questions, n, c)
