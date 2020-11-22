import pandas as pd


def channel_videos_query(youtube, channel_id, published_after):
    page_token = None
    videos_json = []
    while True:
        request = youtube.search().list(
            part='snippet',
            maxResults=50,
            channelId=channel_id,
            publishedAfter=published_after,
            type='video',
            order='date',
            pageToken=page_token,
        )
        response = request.execute()

        videos_json += response['items']

        if 'nextPageToken' in response:
            page_token = response['nextPageToken']
        else:
            break

    return videos_json


def channel_videos_df(videos_json):
    df = {
        'video_id': [v['id']['videoId'] for v in videos_json],
        'published': pd.to_datetime(
            [v['snippet']['publishedAt'] for v in videos_json]),
        'title': [v['snippet']['title'] for v in videos_json],
    }

    return pd.DataFrame(df)


def channels_videos_df(
    youtube,
    channels,
    published_after,
):
    videos_dfs = []
    for channel, channel_id in channels.items():
        videos_json = channel_videos_query(youtube, channel_id, published_after)
        videos_df = channel_videos_df(videos_json)
        videos_df['channel'] = channel
        videos_df['channel_id'] = channel_id
        videos_dfs.append(videos_df)

    return pd.concat(videos_dfs, ignore_index=True)
