from transformers import pipeline
import pandas as pd


def create_question_answerer():
    question_answerer = pipeline(model = 'deepset/roberta-base-squad2')
    return question_answerer

def answer_questions_with_confidence(question_answerer, context = "You did not specify any content", questions = ["Did you mean to specify a question?"]):
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

    return questions_answers

def select_top_n_questions(question_answerer, context, questions, c = 0.3, n = 20, max_repeat_exact_answers=2):
    """Selects the top n questions with the highest confidence level c
    User can define how many questions are required and the minimum confidence level"""

    # Call answer_questions to get a df of questions and answers
    questions_answers = answer_questions_with_confidence(question_answerer, context, questions)

    # Filter for confidence
    conf_questions = questions_answers[questions_answers['confidence'] > c]

    # Create a dictionary with a key for each unique answer
    # which will be updated with frequency of occurrence
    answers_count = {k: 0 for k in conf_questions["answer"].unique()}

    # Sort questions by confidence
    selected_questions = conf_questions.sort_values(by='confidence', ascending=False)

    # Remove questions/answers for which the answer occurs more then `max_repeat_exact_answers` times
    for index, row in selected_questions.iterrows():
        answer = row["answer"]
        answers_count[answer] = answers_count[answer] + 1
        if answers_count[answer] > max_repeat_exact_answers:
            selected_questions.drop(index, inplace=True)

    selected_questions = selected_questions.head(n).reset_index().rename(columns={'index':'original_question_number'})

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
