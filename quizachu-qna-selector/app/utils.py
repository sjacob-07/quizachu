# quizachu-qna-selector/app/utils.py
# Any utility functions that might be used across the application can go here

# quizachu-qna-selector/select_top_n_questions.py
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import pandas as pd

"""Import our question answering model"""
question_answerer = pipeline(model = 'deepset/roberta-base-squad2')

#May move this in future as content will be provided via Rob's model
# from bs4 import BeautifulSoup
# import requests
# def get_article_from_url(url):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.content)
#     article = []
#     for para in soup.find_all("div", {"data-component": "text-block"}):
#         article.append(para.text)
#     return " ".join(article)

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
        q_a_dict['answer'] = q_a['answer'].replace('\n', ' ')

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


bbc_context = """A cat whose pictures went viral for regularly visiting a railway station is releasing a Christmas single. Four-year-old Nala has been delighting commuters who have been taking photos of her at Stevenage station. Owner Natasha Ambler revealed the cat was releasing a single called Meow and has been approached for a book deal. The ginger tabby has also recorded a video for the song due to be released this week, under the name Nala the Station Cat. It has been produced by Danny Kirsch, who wrote it with Joe Killington, while Nala is also co-credited as a songwriter, as well as a vocalist. Ms Ambler said "we want to spread the happiness that Stevenage has had, and she's had on socials to the world". The single is officially released on Wednesday and BBC Three Counties Radio's Justin Dealey gave the single an exclusive first play on Sunday. "I'm slightly lost for words," said the presenter after the song finished. Nala's owner replied: "So am I to be fair." The musical cat does not yet have an agent and her owner said "we're all doing our emails ourselves; it's quite new to us". "We'll start small and hopefully she gets in the charts, but number one would be fantastic," she added. Charity campaigners LadBaby have filled the coveted Christmas number one single spot every year for the last five years. All proceeds from the single will be donated to the RSPCA and Stevenage homelessness charity Feed Up Warm Up. The music video, filmed at Stevenage railway station, will be unveiled before Christmas. Follow East of England news on Facebook, Instagram and X. Got a story? Email eastofenglandnews@bbc.co.uk or WhatsApp 0800 169 1830"""
bbc_questions = ['Where will profit go?','Who produced the song?','What is the song called?',\
             'Who gave the song its first play?','When will the song be released?','Who wrote the song?',\
             'Where was the video filmed?','How has nala been delighting commuters?',\
             "Who's pictures went viral?", 'All proceeds from the single will be what?', 'What links Danny and Joe?']

print(answer_questions_with_confidence(bbc_context, bbc_questions))
