import os

from googleapiclient.discovery import build
import pandas as pd


def videos_descriptions_query(youtube, video_ids):
    descriptions_json = []
    for i in range(0, len(video_ids), 50):
        request = youtube.videos().list(
            part='snippet',
            id=','.join(video_ids[i:i + 50]),
            maxResults=50,
        )
        response = request.execute()

        descriptions_json += response['items']

    return descriptions_json


def videos_descriptions_df(descriptions_json):
    df = {
        'video_id': [d['id'] for d in descriptions_json],
        'description': [d['snippet']['description'] for d in descriptions_json],
    }

    return pd.DataFrame(df)


if __name__ == '__main__':
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

    api_service_name = 'youtube'
    api_version = 'v3'
    DEVELOPER_KEY = 'AIzaSyD16H3VkGm0GTD07e5IgVD8oHLIuSEwXh8'

    # initialize YouTube Data API
    youtube = build(api_service_name, api_version, developerKey=DEVELOPER_KEY)

    # get descriptions of all videos
    videos_df = pd.read_csv('videos.csv')
    descriptions_json = videos_descriptions_query(youtube, videos_df['video_id'])
    descriptions_df = videos_descriptions_df(descriptions_json)

    # save video descriptions to storage
    videos_df = pd.merge(videos_df, descriptions_df)
    videos_df.to_csv('videos.csv', index=False)
