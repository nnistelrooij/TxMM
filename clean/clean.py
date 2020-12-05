import pandas as pd


def remove_replies_channel_comments(comments):
    """Remove replies and comments made by channel itself."""
    replies = comments['replies'] == -1
    self = comments['channel_id'] == comments['author_channel_id']
    comments = comments[~replies & ~self]

    return comments


def replace_diacritics_tags(comments):
    text = comments['text'].str.replace('á', 'a')
    text = text.str.replace('É', 'E')
    text = text.str.replace('é', 'e')
    text = text.str.replace('ö', 'o')

    regex = '(^|(?<=\s)){}((?=\s)|$)'
    text = text.str.replace(regex.format('TIMECODE'), 'timecode')
    text = text.str.replace(regex.format('LINK'), 'link')
    text = text.str.replace(regex.format('HASHTAG'), 'hashtag')
    text = text.str.replace(regex.format('EMOTICON'), 'emoticon')
    text = text.str.replace(regex.format('USERHANDLE'), 'userhandle')

    comments = comments.assign(text=text)
    return comments


def replace_timecodes(comments):
    regex = (
        '((@|~|#) ?)?'  # optional @, ~, or # at the start
        '(([0-9](:|h))?[0-9]{1,2}(:|m)[0-9]{2}s?\s?-\s?)?'  # optional t range
        '([0-9](:|h))?[0-9]{1,2}(:|m)[0-9]{2}s?'  # time code with : or h m s
        '(?=[:;\.,\?\!]?(\s|$))'  # optional punctuation, ends with ws or EOF
    )
    timecodes = comments['text'].str.replace(regex, 'TIMECODE')
    comments = comments.assign(text=timecodes)
    comments['timecode'] = comments['text'].str.match('.*TIMECODE.*')
    print(f'We have replaced {sum(comments["timecode"])} time codes.')

    return comments


def replace_links(comments):
    regex = (
        '(https?:\/\/|ftp:\/\/|https?:\/\/www\.|www\.)'  # starts with protocol
        '([a-z0-9\-]{1,}\.){1,5}'  # website domain
        '[a-z]{1,4}'  # top-level domain
        '\/?(([\.\/]?[A-za-z0-9\/%<>\-_@~#:\+&=\?])+)?'  # optional page or file
    )
    links = comments['text'].str.replace(regex, 'LINK')
    comments = comments.assign(text=links)
    comments['link'] = comments['text'].str.match('.*LINK.*')
    print(f'We have found {sum(comments["link"])} links.')

    return comments


def replace_hashtags(comments):
    regex = (
        '(^|(?<=\s))'  # start with SOF or ws
        '#[A-Za-z][A-Za-z0-9\!]+'  # hashtag
        '((?=\s)|$)'  # end with EOF or ws
    )
    hashtags = comments['text'].str.replace(regex, 'HASHTAG')
    comments = comments.assign(text=hashtags)
    comments['hashtag'] = comments['text'].str.match('.*HASHTAG.*')
    print(f'We have found {sum(comments["hashtag"])} hashtags.')

    return comments


def replace_emoticons(comments):
    regex = (
        '(^|(?<=\s))'  # start with SOF or ws
        '(\:\w+\:|'  # emojis, such as :smile:
        '\<[\/\\]?3|'  # (broken) heart
        '[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|'  # smileys
        '[\:\;\=B8x][\-\^]?[3DOPp\@\$\*\\\)\(\/\|]|'  # more smileys
        '\^\^|xd)'  # ^^ or xd
        '(?=\s|[\!\.\?]|$)'  # end with EOF or ws
    )
    emoticons = comments['text'].str.replace(regex, 'EMOTICON')
    comments = comments.assign(text=emoticons)
    comments['emoticon'] = comments['text'].str.match('.*EMOTICON.*')
    print(f'We have found {sum(comments["emoticon"])} emoticons.')

    return comments


def replace_userhandles(comments):
    regex = (
        '(^|(?<=\s))'  # start with BOF or ws
        '@(bprp|Flammable Maths|Stand-up Maths|Standup Maths|PBS|'
        'PBS Infinite Series|Think Twice|Tipping Point Math|3B1B|'
        '[a-z0-9]{5,})'  # multitoken handle or generic handle
    )
    user_handles = comments['text'].str.replace(regex, 'USERHANDLE', case=False)
    comments = comments.assign(text=user_handles)
    comments['user'] = comments['text'].str.match('.*USERHANDLE.*')
    print(f'We have found {sum(comments["user"])} user handles.')

    return comments


def remove_dirt(comments):
    # remove comments with only replacements
    tag_regex = '(\s*(TIMECODE|LINK|HASHTAG|EMOTICON|USERHANDLE))+\s*$'
    comments = comments[~comments['text'].str.match(tag_regex)]

    # remove comments with 3 or fewer characters
    comments['textlen'] = comments['text'].str.len()
    comments = comments[comments['textlen'] >= 4]

    # remove comments with non-ASCII characters or not one lowercase character
    clean_regex = '[A-Za-z0-9\/\?\.,;:\-\(\)&%\$\!"\' \n\t]+$'
    comments = comments[comments['text'].str.match(clean_regex)]
    comments = comments[~comments['text'].str.match('[^a-z]+$')]

    return comments


if __name__ == '__main__':
    comments = pd.read_csv('../data/comments.csv')
    comments = remove_replies_channel_comments(comments)
    comments = replace_diacritics_tags(comments)
    comments = replace_timecodes(comments)
    comments = replace_links(comments)
    comments = replace_hashtags(comments)
    comments = replace_emoticons(comments)
    comments = replace_userhandles(comments)
    comments = remove_dirt(comments)

    print(f'Final number of comments: {comments.shape[0]}')
    comments.to_csv('clean_comments.csv', index=False)
