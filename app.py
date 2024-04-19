import datetime as dt

from flask import Flask, render_template, request
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
import pandas as pd
import plotly.graph_objs as go
from profanity_check import predict as profanity_predict

app = Flask(__name__)

# Load the fine-tuned model
fine_tuned_model = 'fine_tuned_clf'
model = DistilBertForSequenceClassification.from_pretrained(fine_tuned_model, num_labels=14)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


def decode_emotion(numeric_label):
    class_labels = ['admiration','amusement', 'love', 'neutral', 'optimism', 'sadness', 'anger', 'annoyance', 'approval', 'confusion', 'curiosity', 'disapproval', 'gratitude', 'joy']
    return class_labels[numeric_label]


def predict_emotion(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    _, predicted = torch.max(outputs.logits, 1)
    print('pred item', predicted.item())
    emotion = decode_emotion(predicted.item()) 
    return emotion


def create_plotly_table(df):
    # Create a table trace
        table = go.Table(
            header=dict(
                values=['Date', 'Text', 'Emotion'],
                fill_color='paleturquoise',
                align='center',
                font=dict(color='black', size=14)
            ),
            cells=dict(
                values=[df['date'], df['text'], df['prediction']],
                fill_color='lavender',
                align='center',
                font=dict(color='black', size=12)
            )
        )

        # Define the layout
        layout = go.Layout(
            title='Output Predictions',
            autosize=False,
            width=1500,
        )

        # Create a figure with the table trace and layout
        fig = go.Figure(data=table, layout=layout)
        return fig


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the input text
        text = request.form['text']

        if text:
            # Check for profanity in input text
            if profanity_predict([text])[0] == 1:
                # Profanity detected, don't append to dataframe but make a prediction
                emotion = predict_emotion(text)
                # Append to output_predictions.csv
                try:
                    df = pd.read_csv('output_predictions.csv')
                # Case when the data frame doesn't already exist
                except FileNotFoundError:
                    df = pd.DataFrame(columns=['date', 'text', 'prediction'])

            else:
                # No profanity detected, make a prediction and append to dataframe
                emotion = predict_emotion(text)

                # Append to output_predictions.csv
                try:
                    df = pd.read_csv('output_predictions.csv')
                # Case when the data frame doesn't already exist
                except FileNotFoundError:
                    df = pd.DataFrame(columns=['date', 'text', 'prediction'])
        
                new_row = pd.DataFrame({'date': [dt.datetime.now()], 'text': text, 'prediction': emotion})
                df = pd.concat([df, new_row])
                df.to_csv('output_predictions.csv', index=False)

            # Sort the data frame in reverse order according to 'date' column
            df.date = pd.to_datetime(df.date)
            # Format the datetime into UK format so it's suitable for viewing
            df.date = df.date.dt.strftime('%d/%m/%Y %H:%M:%S')
            df = df.sort_values(by='date', ascending=False)
            
            fig = create_plotly_table(df=df)
            # # Convert the figure to HTML
            plot_html = fig.to_html(full_html=False)
    
            return render_template('index.html', emotion=emotion, plot_html=plot_html)
    
        
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
