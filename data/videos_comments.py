import pandas as pd


def video_comments_query(youtube, video_id):
    page_token = None
    comments_json = []
    while True:
        request = youtube.commentThreads().list(
            part='snippet,replies',
            maxResults=100,
            videoId=video_id,
            textFormat='plainText',
            pageToken=page_token,
        )
        response = request.execute()

        # take out comment threads where comment or reply author has no channel
        for ct in response['items']:
            # top level comment author has no channel
            top_level_comment = ct['snippet']['topLevelComment']
            if 'authorChannelId' not in top_level_comment['snippet']:
                continue

            # one of the reply authors has no channel
            if 'replies' in ct:
                replies = ct['replies']['comments']
                authors = ['authorChannelId' in r['snippet'] for r in replies]
                if not all(authors):
                    continue

            comments_json.append(ct)

        if 'nextPageToken' in response:
            page_token = response['nextPageToken']
        else:
            break

    return comments_json


def video_comments_df(comments_json):
    top_comments = [ct['snippet']['topLevelComment'] for ct in comments_json]
    replies = [ct for ct in comments_json if 'replies' in ct]
    replies = [r for rs in replies for r in rs['replies']['comments']]
    comments = [c['snippet'] for c in top_comments + replies]
    df = {
        'comment_id': [c['id'] for c in top_comments] +
                      [r['snippet']['parentId'] for r in replies],
        'replies': [c['snippet']['totalReplyCount'] for c in comments_json] +
                   [-1]*len(replies),
        'text': [c['textDisplay'].replace('\r', '') for c in comments],
        'author_channel_id': [c['authorChannelId']['value'] for c in comments],
        'likes': [c['likeCount'] for c in comments],
        'published': pd.to_datetime([c['publishedAt'] for c in comments]),
    }

    return pd.DataFrame(df)


def channels_comments_df(youtube, channels, videos_df):
    comments_dfs = []
    for channel, channel_id in channels.items():
        video_ids = videos_df.loc[videos_df['channel'] == channel, 'video_id']
        for video_id in video_ids:
            comments_json = video_comments_query(youtube, video_id)
            comments_df = video_comments_df(comments_json)
            comments_df['channel'] = channel
            comments_df['channel_id'] = channel_id
            comments_df['video_id'] = video_id
            comments_dfs.append(comments_df)

    return pd.concat(comments_dfs, ignore_index=True)
