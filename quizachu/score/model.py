from quizachu.registry import *
from quizachu.score.tokenizer import BertSemanticDataTokenizer
import tensorflow as tf
import numpy as np

sentence1 = "A soccer game with multiple males playing"
sentence2 = "Some men are playing a sport"

test_context = """
The history of the Netherlands extends back long before the founding of the modern Kingdom of the Netherlands in 1815 after the defeat of Napoleon. For thousands of years, people have been living together around the river deltas of this section of the North Sea coast. Records begin with the four centuries during which the region formed a militarized border zone of the Roman Empire. As the Western Roman Empire collapsed and the Middle Ages began, three dominant Germanic peoples coalesced in the area – Frisians in the north and coastal areas, Low Saxons in the northeast, in addition to the Franks in the south. By 800, the Frankish Carolingian dynasty had once again integrated the area into an empire covering a large part of Western Europe. The region was part of the duchy of Lower Lotharingia within the Holy Roman Empire, but neither the empire nor the duchy were governed in a centralized manner. For several centuries, medieval lordships such as Brabant, Holland, Zeeland, Friesland, Guelders and others held a changing patchwork of territories.

By 1433, the Duke of Burgundy had assumed control over most of Lower Lotharingia, creating the Burgundian Netherlands. This included what is now the Netherlands, Belgium, Luxembourg, and a part of France. When their heirs the Catholic kings of Spain took strong measures against Protestantism, the subsequent Dutch revolt led to the splitting in 1581 of the Netherlands into southern and northern parts. The southern "Spanish Netherlands" corresponds approximately to modern Belgium and Luxembourg, and the northern "United Provinces" (or "Dutch Republic)", which spoke Dutch and was predominantly Protestant, was the predecessor of the modern Netherlands."""

def initialize_generate_score_model():
    model_path = get_scoring_model_path()
    model = tf.keras.saving.load_model(model_path)
    # from transformers import TFBertModel
    # model = TFBertModel.from_pretrained("bert-base-uncased")
    return model

def update_score_model_weights(model, weights_path):
    model.load_weights(weights_path)
    return model

def create_generate_score_model():
    from transformers import TFBertModel
    model_path = get_scoring_model_path()
    model = tf.keras.saving.load_model(model_path)
    return model

def check_answer_similarity(model, sentence1, sentence2):
    labels = ["contradiction", "entailment", "neutral"]
    sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
    test_data = BertSemanticDataTokenizer(
        sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
    )

    proba = model.predict(test_data[0])[0]
    idx = np.argmax(proba)
    proba = f"{proba[idx]: .2f}%"
    pred = labels[idx]
    return {"prediction": pred, "probability": proba}

if __name__ == "__main__":
    model = create_generate_score_model()

    results = check_answer_similarity(model, sentence1, sentence2)
    print(results)
