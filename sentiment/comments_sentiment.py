import json
import os
import subprocess

import pandas as pd


def get_sentiment(comments):
    # write input to MAX sentiment classifier in input.txt
    if os.path.exists('input.txt'): os.remove('input.txt')
    with open('input.txt', 'a') as text_file:
        print('{ "text": [', file=text_file, end='')
        for comment in comments.iloc[:-1]:
            print(f' "{comment}",', file=text_file, end='')
        print(f' "{comments.iloc[-1]}" ] }}', file=text_file, end='')

    # run MAX sentiment classifier on stored input.txt file
    x = subprocess.Popen([
            'curl',
            '-X', 'POST',
            'http://localhost:5000/model/predict',
            '-H', 'accept: application/json',
            '-H', 'Content-Type: application/json',
            '-d', '@input.txt',
        ],
        shell=True,
        stdout=subprocess.PIPE,
    )
    output_json = json.loads(x.stdout.read())
    predictions = [p['positive'] for p in output_json['predictions']]

    return predictions


if __name__ == '__main__':
    comments = pd.read_csv('../clean/clean_comments.csv')

    predictions = get_sentiment(comments['text'])
    comments = comments.assign(sentiment=predictions)
    comments.to_csv('sentiment_comments.csv', index=False)
