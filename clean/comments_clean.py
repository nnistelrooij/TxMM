import re

import pandas as pd


def remove_replies_channel_comments(comments):
    """Remove replies and comments made by channel itself."""
    replies = comments['replies'] == -1
    self = comments['channel_id'] == comments['author_channel_id']
    comments = comments[~replies & ~self]

    return comments


def replace_diacritics_html_repeats(comments):
    # get texts from comments DataFrame
    text = comments['text']

    # replace most common diacritics
    text = text.str.replace('á', 'a')
    text = text.str.replace('É', 'E')
    text = text.str.replace('é', 'e')
    text = text.str.replace('ö', 'o')

    # replace HTML character codes
    text = text.str.replace('&amp;', '&')
    text = text.str.replace('&#39;', '\'')
    text = text.str.replace('&quot;', '"')

    # replace long sequences of equal characters
    text = text.str.replace(r'(.|\s)\1{3,}', r'\1\1\1')

    # put new texts in comments DataFrame
    comments.loc[:, 'text'] = text


def replace(comments, regex, repl, key, flags=0):
    textn = comments['text'].map(lambda c: re.subn(regex, repl, c, flags=flags))
    comments[['text', key]] = textn.apply(pd.Series)


def replace_timecodes(comments, repl=''):
    regex = (
        r'((@|~|#) ?)?'  # optional @, ~, or # at the start
        r'(([0-9](:|h))?[0-9]{1,2}(:|m)[0-9]{2}s?\s?-\s?)?'  # optional t range
        r'([0-9](:|h))?[0-9]{1,2}(:|m|\')[0-9]{2}(s|\'\')?'  # time code
        r'(?=[:;\.,\?\!]?(\s|$))'  # optional punctuation, ends with ws or EOF
    )
    replace(comments, regex, repl, 'timecode')
    print(f'We have replaced {sum(comments["timecode"])} time codes.')


def replace_links(comments, repl=''):
    regex = (
        r'(https?://|ftp://|https?://www\.|www\.)'  # starts with protocol
        r'([A-Za-z0-9\-]+\.){1,5}'  # website domain
        r'[a-z]{1,4}'  # top-level domain
        r'/?(([\./]?[A-za-z0-9/%<>\-_@~#:\+&\=\?])+)?'  # optional page or file
    )
    replace(comments, regex, repl, 'link')
    print(f'We have replaced {sum(comments["link"])} links.')


def replace_emails(comments, repl=''):
    regex = (
        r'(^|(?<=\s))'  # start with SOF or ws
        r'[A-Za-z0-9\-\.]+'  # user name
        r'@([A-Za-z0-9\-]+\.){1,5}'  # email domain
        r'[a-z]{1,4}'  # top-level domain
        r'(?=[:;\.,\?\!]?(\s|$))'  # optional punctuation, ends with ws or EOF
    )
    replace(comments, regex, repl, 'email')
    print(f'We have replaced {sum(comments["email"])} e-mails.')


def replace_hashtags(comments, repl=''):
    regex = (
        r'(^|(?<=\s))'  # start with SOF or ws
        r'#[A-Za-z][A-Za-z0-9\!]+'  # hashtag
        r'(?=[:;\.,\?\!]?(\s|$))'  # optional punctuation, ends with ws or EOF
    )
    replace(comments, regex, repl, 'hashtag')
    print(f'We have replaced {sum(comments["hashtag"])} hashtags.')


def replace_emoticons(comments, repl=''):
    regex = (
        r'(^|(?<=\s))'  # start with SOF or ws
        r'(:\w+:|'  # emojis, such as :smile:
        r'<[/\\]?3|'  # (broken) heart
        r'[\(\)\\|\*\$][\-\^]?[:;\=]|'  # smileys
        r'[:;\=8x][\-\^]?[3DOPp@\$\*\\\)\(/\|]|'  # more smileys
        r'\^\^|xd|XD|:(v){1,3}|:0|\-:\)|:9\))'  # ^^ or xd or :v or :0
        r'(?=\s|[\!\.\?]|$)'  # end with EOF or ws
    )
    replace(comments, regex, repl, 'emoticon')
    print(f'We have replaced {sum(comments["emoticon"])} emoticons.')


def replace_userhandles(comments, repl=''):
    regex = (
        r'(^|(?<=\s))'  # start with BOF or ws
        r'@(bprp|Flammable Maths|Stand-up Maths|Standup Maths|PBS|'
        r'PBS Infinite Series|Think Twice|Tipping Point Math|3B1B|'
        r'[a-z0-9]{5,})'  # multitoken handle or generic handle
    )
    replace(comments, regex, repl, 'userhandle', flags=re.IGNORECASE)
    print(f'We have replaced {sum(comments["userhandle"])} user handles.')


def remove_dirt(comments):
    # remove comments with only replacements
    tag_regex = '(\s*(TIMECODE|LINK|EMAIL|HASHTAG|EMOTICON|USERHANDLE))+\s*$'
    comments = comments[~comments['text'].str.match(tag_regex)]

    # remove comments with too many replacements
    tags = ['timecode', 'link', 'email', 'hashtag', 'emoticon', 'userhandle']
    n_tags = comments[tags].sum(axis=1)
    comments = comments[n_tags <= 5]

    # remove comments with too few or too many characters
    comments = comments[comments['text'].str.len() >= 4]
    comments = comments[comments['text'].str.len() <= 500]

    # remove comments with non-ASCII characters or not one lowercase character
    clean_regex = '[A-Za-z0-9/\?\.,;:\-\(\)&%\$\!"\' \n\t]+$'
    comments = comments[comments['text'].str.match(clean_regex)]
    comments = comments[~comments['text'].str.match('[^a-z]+$')]

    # replace tabs with 8 spaces and escape all quotation marks
    comments = comments.assign(text=comments['text'].str.replace('\t', ' '*8))
    comments = comments.assign(text=comments['text'].str.replace(r'"', r'\"'))

    return comments


if __name__ == '__main__':
    comments = pd.read_csv('../data/comments.csv')
    comments = remove_replies_channel_comments(comments)
    replace_diacritics_html_repeats(comments)
    replace_timecodes(comments, 'TIMECODE')
    replace_links(comments, 'LINK')
    replace_emails(comments, 'EMAIL')
    replace_hashtags(comments, 'HASHTAG')
    replace_emoticons(comments, 'EMOTICON')
    replace_userhandles(comments, 'USERHANDLE')
    comments = remove_dirt(comments)

    print(f'Final number of comments: {comments.shape[0]}')
    comments.to_csv('clean_comments.csv', index=False)
