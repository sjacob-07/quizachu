# quizachu-qna-selector/select_top_n_questions.py
from transformers import pipeline
from bs4 import BeautifulSoup
import requests
import pandas as pd

question_answerer = pipeline(model='deepset/roberta-base-squad2')

def get_article_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content)
    article = []
    for para in soup.find_all("div", {"data-component": "text-block"}):
        article.append(para.text)
    return " ".join(article)

def answer_questions_with_confidence(context = "You did not specify any content", questions = ["Did you mean to specify a question?"]):
    """Takes a list called 'questions' that contains the questions to answer
    Takes some text called 'content' as a source for answering questions
    Returns a dataframe of the questions with their answers and an assessment of confidence in the answers
    If no context or content is provided, returns a dataframe requesting these"""

    # List to fill with questions, answers, and confidence
    questions_answers = []

    # For each question create an empty dictionary and call the question_answerer model on the question
    for q in questions:
        q_a_dict = {}
        q_a = question_answerer(question=q, context=context)

        # Assign the question, and outputs of the question_answerer model to the dictionary
        q_a_dict['confidence'] = q_a['score']
        q_a_dict['question'] = q
        q_a_dict['answer'] = q_a['answer']

        # Add the dictionary to the list and then convert the final list of dicts to a dataframe
        questions_answers.append(q_a_dict)
    questions_answers = pd.DataFrame(questions_answers)

    # Set a large maxcolwidth to allow for potentially long answers
    pd.options.display.max_colwidth = 20000

    return questions_answers


def select_top_n_questions(context, questions, c = 0.5, n = 5):
    """Selects the top n questions with the highest confidence level c
    User can define how many questions are required and the minimum confidence level"""

    # Call answer_questions to get a df of questions and answers
    questions_answers = answer_questions_with_confidence(context, questions)

    # Filter for confidence
    conf_questions = questions_answers[questions_answers['confidence'] > c]

    # Return n questions ordered by confidence
    selected_questions = conf_questions.sort_values(by='confidence', ascending=False).head(n)\
    .reset_index().rename(columns={'index':'original_question_number'})

    """Check whether enough questions can be returned and explain why if not"""

    # Were enough questions generated?
    if len(questions_answers) < n:
        print(f"Only {len(questions_answers)} questions were generated")

        # Did enough questions meet the confidence requirement?
        if len(selected_questions) == 0:
            print("No questions met your required confidence level.")
        elif len(selected_questions) < n:
            print(f"Not enough questions met your required confidence level,\
 but here {'is' if len(selected_questions) == 1 else 'are'} the {len(selected_questions)} that did:")
        else:
            print(f"Here are your {n} questions")

    else:
        # Did enough questions meet the confidence requirement?
        if len(selected_questions) == 0:
            print("No questions met your required confidence level.")
        elif len(selected_questions) < n:
            print(f"Not enough questions met your required confidence level,\
 but here {'is' if len(selected_questions) == 1 else 'are'} the {len(selected_questions)} that did:")
        else:
            print(f"Here are your {n} questions")

    return selected_questions
