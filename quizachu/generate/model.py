weights = "###"

test_context = """
The history of the Netherlands extends back long before the founding of the modern Kingdom of the Netherlands in 1815 after the defeat of Napoleon. For thousands of years, people have been living together around the river deltas of this section of the North Sea coast. Records begin with the four centuries during which the region formed a militarized border zone of the Roman Empire. As the Western Roman Empire collapsed and the Middle Ages began, three dominant Germanic peoples coalesced in the area – Frisians in the north and coastal areas, Low Saxons in the northeast, in addition to the Franks in the south. By 800, the Frankish Carolingian dynasty had once again integrated the area into an empire covering a large part of Western Europe. The region was part of the duchy of Lower Lotharingia within the Holy Roman Empire, but neither the empire nor the duchy were governed in a centralized manner. For several centuries, medieval lordships such as Brabant, Holland, Zeeland, Friesland, Guelders and others held a changing patchwork of territories.

By 1433, the Duke of Burgundy had assumed control over most of Lower Lotharingia, creating the Burgundian Netherlands. This included what is now the Netherlands, Belgium, Luxembourg, and a part of France. When their heirs the Catholic kings of Spain took strong measures against Protestantism, the subsequent Dutch revolt led to the splitting in 1581 of the Netherlands into southern and northern parts. The southern "Spanish Netherlands" corresponds approximately to modern Belgium and Luxembourg, and the northern "United Provinces" (or "Dutch Republic)", which spoke Dutch and was predominantly Protestant, was the predecessor of the modern Netherlands."""

def initialize_generate_model():
    from transformers import TFT5ForConditionalGeneration
    model = TFT5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    return model

def initialize_generate_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    return tokenizer

def load_weights_from_bucket(model):
    model.load_weights(weights)
    return model

def generate_questions(model, tokenizer, context, n_questions=20):
    tokens = tokenizer(context, return_tensors="tf").input_ids
    generated_tokens = model.generate(
        tokens,
        do_sample=True,
        num_return_sequences=n_questions,
        top_p=0.98)
    questions = []
    for i in range(n_questions):
        questions.append(tokenizer.decode(generated_tokens[i], skip_special_tokens=True))
    return questions

if __name__ == "__main__":
    model = initialize_generate_model()
    tokenizer = initialize_generate_tokenizer()
    model = load_weights_from_bucket(model)

    questions = generate_questions(model, tokenizer, test_context, 10)
    print(questions)
