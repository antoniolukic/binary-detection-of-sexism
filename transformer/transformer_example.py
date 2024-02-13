import pandas as pd
df = pd.read_csv("train_all_tasks.csv")

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.0001, random_state=42)

lista_tekst = df["text"]
lista_label = df["label_sexist"]

for train_index, test_index in sss.split(lista_tekst, lista_label):
    X_train, X_test = lista_tekst[train_index], lista_tekst[test_index]
    y_train, y_test = lista_label[train_index], lista_label[test_index]
    d_train = {"text": X_train, "label": y_train}
    df_train = pd.DataFrame(d_train)
    d_test = {"text": X_test, "label": y_test}
    df_test = pd.DataFrame(d_test)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=42)

lista_tekst = np.array(df_train["text"])
lista_label = np.array(df_train["label"])

for train_index, valid_index in sss.split(lista_tekst, lista_label):
    X_train, X_valid = lista_tekst[train_index], lista_tekst[valid_index]
    y_train, y_valid = lista_label[train_index], lista_label[valid_index]
    d_train = {"text": X_train, "label": y_train}
    df_train = pd.DataFrame(d_train)
    d_valid = {"text": X_valid, "label": y_valid}
    df_valid = pd.DataFrame(d_valid)

print(len(df_test), len(df_valid), len(df_train))

def sentiment_to_label(sentiment):
    return 1 if sentiment == "sexist" else 0

df = {"train": df_train, "test": df_test, "valid": df_valid}
for split in df:
    df[split]["label"] = df[split].label.apply(sentiment_to_label)

def evaluate(y_test, y_test_pred):
    # mjerit ćemo predviđanje modela pomoću matrice konfuzije
    # te sljedećih metrika: preciznost, odziv, f1 (https://en.wikipedia.org/wiki/Precision_and_recall)
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report
    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.metrics import precision_recall_fscore_support

    print(classification_report(y_test, y_test_pred))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average="micro"
    )
    print(f"Precision: {precision:6.2f}")
    print(f"   Recall: {recall:6.2f}")
    print(f"       F1: {f1:6.2f}")

    ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred)
    plt.show()
    return precision, recall, f1

from transformers import AutoModelForSequenceClassification

# Ovime smo definirali da želimo model bert-base-uncased (bert s 12 enkodera koji ne razlikuje mala i velika slova)
# Također smo definirali da želimo BERT model koji može svrstati tekst u 2 klase
# Dodajemo sloj iznad 12-tog enkodera koji omogućava klasifikaciju (taj sloj se zove potpuno povezani sloj)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

from transformers import AutoTokenizer

# učitajmo tokenizator za model koji želimo koristiti (svaki transformer ima svoj tokenizator)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

from transformers import pipeline
# ovdje definiramo da želimo odrezati dulje tekstove od 512 tokena na 512 tokena
clf = pipeline(
    "text-classification",
    model=model,
    tokenizer=lambda x, **kwargs: tokenizer(
        x,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ),
    #device=1, # koristi GPU
)

# {'label': 'LABEL_0', 'score': 0.557502269744873} ovako izgleda jedna predikcija
y_pred = [clf(x) for x in df["test"]["text"]]
# moramo izvući labelu
y_pred = [int(y[0]["label"].split("_")[1]) for y in y_pred]
y_test = df["test"]["label"]

evaluate(y_test, y_pred)

# napravimo tokenizaciju, odrezat ćemo sve reviews koji su dulji od 512 tokena
tokenized_train = df["train"].map(
    lambda x: tokenizer(
        x["text"],
        padding="max_length",
        max_length=512,
        truncation=True,
    ),
    batched=True,
)
tokenized_test = df["test"].map(
    lambda x: tokenizer(
        x["text"],
        padding="max_length",
        max_length=512,
        truncation=True,
    ),
    batched=True,
)

def compute_metrics(eval_pred):
    import numpy as np

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    p, r, f1 = evaluate(y_test, predictions)
    return {"precision": p, "recall": r, "f1": f1}

# train model
from transformers import Trainer, TrainingArguments

# huggingface skriva cijelu logiku treniranja transformera kroz Trainer
# samo predamo argumente koje želimo, kažemo trainer.train() i sve radi!
trainer = Trainer(
    model=model,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
    args=TrainingArguments(
        num_train_epochs=3,
        evaluation_strategy="epoch",
        eval_steps=1,
        output_dir="output",
        seed=42,
        fp16=True,
    ),
)

trainer.train()