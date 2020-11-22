import pandas as pd


def comment_replies_query(youtube, top_level_comment_id):
    page_token = None
    replies_json = []
    while True:
        request = youtube.comments().list(
            part='snippet',
            maxResults=100,
            parentId=top_level_comment_id,
            textFormat='plainText',
            pageToken=page_token,
        )
        response = request.execute()

        # take out replies where author has no channel
        for r in response['items']:
            if 'authorChannelId' in r['snippet']:
                replies_json.append(r['snippet'])

        if 'nextPageToken' in response:
            page_token = response['nextPageToken']
        else:
            break

    return replies_json


def comment_replies_df(replies_json):
    df = {
        'text': [r['textDisplay'].replace('\r', '') for r in replies_json],
        'author_channel_id': [r['authorChannelId']['value'] for r in replies_json],
        'likes': [r['likeCount'] for r in replies_json],
        'published': pd.to_datetime([r['publishedAt'] for r in replies_json]),
    }

    return pd.DataFrame(df)


def comment_video_id(comments_df, top_level_comment_id):
    comment_index = comments_df['comment_id'].eq(top_level_comment_id).idxmax()
    return comments_df.loc[comment_index, 'video_id']


def incomplete_comment_ids(comments_df):
    replies_hist = comments_df['comment_id'].value_counts() - 1
    replies_hist = replies_hist[replies_hist > 0]

    replies_ids = comments_df['comment_id'].isin(replies_hist.index)
    replies_bool = comments_df['replies'] > 0
    top_level_comments = comments_df[replies_ids & replies_bool]

    replies_hist = replies_hist.sort_index()
    top_level_comments = top_level_comments.sort_values(by='comment_id')
    replies_hist.index = range(replies_hist.shape[0])
    top_level_comments.index = range(top_level_comments.shape[0])

    incomplete_comments = replies_hist != top_level_comments['replies']
    comment_ids = top_level_comments.loc[incomplete_comments, 'comment_id']
    nums_replies = replies_hist[incomplete_comments]

    return zip(comment_ids, nums_replies)


def channels_replies_df(youtube, channels, comments_df):
    replies_dfs = []
    for channel, channel_id in channels.items():
        channel_comments_df = comments_df[comments_df['channel'] == channel]
        for comment_id, num_replies in incomplete_comment_ids(channel_comments_df):
            replies_json = comment_replies_query(youtube, comment_id)
            replies_json = replies_json[num_replies:]
            replies_df = comment_replies_df(replies_json)
            replies_df['channel'] = channel
            replies_df['channel_id'] = channel_id
            replies_df['video_id'] = comment_video_id(comments_df, comment_id)
            replies_df['comment_id'] = comment_id
            replies_df['replies'] = -1
            replies_dfs.append(replies_df)

    return pd.concat(replies_dfs, ignore_index=True)
