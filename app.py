from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EvalPrediction
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
import time
from scipy.special import softmax
import multiprocessing
from functools import partial


app = Flask(__name__, template_folder='templates')


model_name = 'sentence-transformers/all-mpnet-base-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)

model_pred = AutoModelForSequenceClassification.from_pretrained("./output/checkpoint-1800", local_files_only=True)
trainer_new = Trainer(model=model_pred, )
trainer_new.model = model_pred

def import_df():
    df = pd.read_csv('./dataset/test.csv')
    df['text'] = df.text.str.lower()
    df = df[:12000]
    df = Dataset.from_pandas(df)
    return df


def tokenize_sentence(examples):
    return tokenizer(examples['text'], padding=True, truncation=True)


def format_sentence(dataset):
    return {'input_ids': dataset['input_ids'], 'attention_mask': dataset['attention_mask']}


def predict_sentence(sentence):
    examples = [{
        'text': sentence
    }]
    eval_dataset = Dataset.from_list(examples)
    eval_dataset = eval_dataset.map(tokenize_sentence, batched=False)
    # Initialize our Trainer
    trainer = Trainer(model=model_pred, tokenizer=tokenizer)
    predictions = trainer.predict(test_dataset=eval_dataset).predictions

    # Adding a softmax layer to get probabilities. np.argmax(predictions, axis=1) for class labels
    predictions = np.array([softmax(element) for element in predictions])[:, 1]
    return 'predictions: ' + str(predictions)


def predict_df(df):
    train_tokenized = df.map(tokenize_sentence, batched=True)
    train_tokenized = train_tokenized.map(format_sentence)
    trainer = Trainer(model=model_pred, tokenizer=tokenizer)
    predictions = trainer.predict(test_dataset=train_tokenized).predictions
    # Adding a softmax layer to get probabilities. np.argmax(predictions, axis=1) for class labels
    predictions = np.array([softmax(element) for element in predictions])[:, 1]
    return predictions

def parallel_tokenize_function(df_row, tokenize_func):
    return tokenize_func(df_row)

def predict_df_faster(df, batch_size=32, num_processes=None):
    start = time.time()
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=num_processes) as pool:
        partial_tokenize = partial(parallel_tokenize_function, tokenize_func=tokenize_sentence)
        train_tokenized = pool.map(partial_tokenize, df, chunksize=batch_size)

        train_tokenized = [format_sentence(row) for row in train_tokenized]

        trainer = Trainer(model=model_pred, tokenizer=tokenizer)
        predictions = trainer.predict(test_dataset=train_tokenized).predictions
        predictions = np.array([softmax(element) for element in predictions])[:, 1]
    end = time.time()

    return {'time': end-start, 'results':predictions}

def predict(faster=False):
    df = import_df()

    if faster:
        preds = predict_df_faster(df)
    else:
        start = time.time()
        predictions = predict_df(df)
        end = time.time()
        preds = {'predictions': predictions, 'time': end-start}

    return preds

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        sentence = request.form['sentence']
        if sentence != '':
            get_pred = predict_sentence(sentence)
        else:
            print('Getting predictions with multiprocessing')
            get_pred = predict(True)
            print(get_pred)
            print('Getting predictions with normally')
            get_pred = predict()
            print(get_pred)
        return render_template('index.html', sentence=get_pred)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
