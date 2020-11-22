import pandas as pd


def videos_stats_query(youtube, videos_df):
    stats_json = []
    for i in range(0, videos_df.shape[0], 50):
        request = youtube.videos().list(
            part='statistics',
            id=','.join(videos_df['video_id'][i:i + 50]),
            maxResults=50,
        )
        response = request.execute()

        # take out videos with disabled comments
        for v in response['items']:
            if 'commentCount' in v['statistics']:
                stats_json.append(v)

    return stats_json


def videos_stats_df(stats_json):
    df = {
        'video_id': [v['id'] for v in stats_json],
        'views': [v['statistics']['viewCount'] for v in stats_json],
        'likes': [v['statistics']['likeCount'] for v in stats_json],
        'dislikes': [v['statistics']['dislikeCount'] for v in stats_json],
        'comments': [v['statistics']['commentCount'] for v in stats_json],
    }

    return pd.DataFrame(df)
