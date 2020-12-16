import numpy as np
import pandas as pd

from comments_clean import *


def concatenate_title_description(videos):
    reps = np.sqrt(videos['description'].str.len() / videos['title'].str.len())
    reps = reps.clip(1, reps).astype(int)

    titles = videos['title'].copy()
    for i, title in enumerate(titles):
        titles[i] = '\n'.join([title]*reps[i])

    videos['text'] = titles + '\n' + videos['description']
    return videos


def remove_dirt(videos):
    # remove non-ASCII characters
    dirty_regex = '[^A-Za-z0-9/\?\.,;:\-\+\(\)&\*%\$\!@#\="\' \n\t]+'
    videos = videos.assign(text=videos['text'].str.replace(dirty_regex, ''))

    # replace tabs with 8 spaces
    videos = videos.assign(text=videos['text'].str.replace('\t', ' '*8))

    return videos


if __name__ == '__main__':
    videos = pd.read_csv('../data/videos.csv')
    videos = concatenate_title_description(videos)
    videos = remove_dirt(videos)

    replace_diacritics_html_repeats(videos)
    replace_timecodes(videos)
    replace_links(videos)
    replace_emails(videos)
    replace_hashtags(videos)
    replace_emoticons(videos)
    replace_userhandles(videos)

    videos.to_csv('clean_videos.csv', index=False)
