import json
import os
import sys
import subprocess

import pandas as pd


def get_sentiment(comments):
    if os.path.exists('input.txt'): os.remove('input.txt')
    with open('input.txt', 'a') as text_file:
        print('{ "text": [', file=text_file, end='')
        for comment in comments.iloc[:-1]:
            print(f' "{comment}",', file=text_file, end='')
        print(f' "{comments.iloc[-1]}"', file=text_file, end='')
        print(' ] }', file=text_file, end='')

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
    comments = comments.assign(text=comments['text'].str.replace('\"', '\\\"'))

    num_comments = int(sys.argv[1]) if len(sys.argv) == 2 else comments.shape[0]
    command_comments = comments.iloc[range(num_comments)]

    predictions = get_sentiment(command_comments['text'])
    command_comments = command_comments.assign(sentiment=predictions)
    command_comments.to_csv('output.csv', index=False)
