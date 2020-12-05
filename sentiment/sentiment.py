import json
import os
import re
import sys
import subprocess

import pandas as pd


def get_sentiment(comments):
    if os.path.exists('input.txt'): os.remove('input.txt')
    with open('input.txt', 'a') as text_file:
        print('{\n  "text": [', file=text_file)
        for comment in comments.iloc[:-1]:
            print(f'    "{comment}",', file=text_file)
        print(f'    "{comments.iloc[-1]}"', file=text_file)
        print('  ]\n}', file=text_file)

    x = subprocess.Popen(
        ['curl',
         '-X', 'POST',
         'http://localhost:5000/model/predict',
         '-H', 'accept: application/json',
         '-H', 'Content-Type: application/json',
         '-d', '@input.txt'],
        shell=True,
        stdout=subprocess.PIPE,
    )
    output_json = json.loads(x.stdout.read())
    predictions = [p['positive'] for p in output_json['predictions']]

    return predictions


if __name__ == '__main__':
    comments = pd.read_csv('../clean/clean_comments.csv')
    comments = comments.assign(text=comments['text'].str.replace('\"', '\\\"'))

    num_comments = sys.argv[1] if len(sys.argv) == 2 else comments.shape[0]
    command_comments = comments[:range(num_comments)]

    command_comments['sentiment'] = get_sentiment(command_comments['text'])
    command_comments.to_csv('output.csv', index=False)
