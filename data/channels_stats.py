import pandas as pd


def channels_stats_query(youtube, channel_ids):
    stats_json = []
    for i in range(0, len(channel_ids), 50):
        request = youtube.channels().list(
            part='snippet,statistics',
            id=','.join(channel_ids[i:i + 50]),
            maxResults=50,
        )
        response = request.execute()

        stats_json += response['items']

    return stats_json


def channels_subscribers(stats_json):
    subscribers = []
    for channel_stats in stats_json:
        if channel_stats['statistics']['hiddenSubscriberCount']:
            subscribers.append(-1)
        else:
            subscribers.append(channel_stats['statistics']['subscriberCount'])

    return subscribers


def channels_stats_df(stats_json):
    channel_published = [cs['snippet']['publishedAt'] for cs in stats_json]
    df = {
        'author_channel_id': [cs['id'] for cs in stats_json],
        'channel_published': pd.to_datetime(channel_published),
        'views': [cs['statistics']['viewCount'] for cs in stats_json],
        'subscribers': channels_subscribers(stats_json),
        'videos': [cs['statistics']['videoCount'] for cs in stats_json],
    }

    return pd.DataFrame(df)
