import os

from googleapiclient.discovery import build
import pandas as pd

from .channels_stats import channels_stats_df, channels_stats_query
from .channels_videos import channels_videos_df
from .comments_replies import channels_replies_df
from .videos_comments import channels_comments_df
from .videos_stats import videos_stats_df, videos_stats_query


def main(api, channel_ids, published_after):
    # get all videos since published_after
    videos_df = channels_videos_df(api, channel_ids, published_after)

    # add stats of videos
    stats_json = videos_stats_query(api, videos_df)
    stats_df = videos_stats_df(stats_json)
    videos_df = videos_df.merge(stats_df)

    # get all top-level comments and their top 5 replies
    comments_df = channels_comments_df(api, channel_ids, videos_df)

    # add replies 6 and on from top-level comments
    replies_df = channels_replies_df(api, channel_ids, comments_df)
    comments_df = pd.concat([comments_df, replies_df], ignore_index=True)

    # add stats of authors
    channel_ids = pd.unique(comments_df['author_channel_id'])
    stats_json = channels_stats_query(api, channel_ids)
    stats_df = channels_stats_df(stats_json)
    comments_df = comments_df.merge(stats_df)

    # save videos and comments to storage
    videos_df.to_csv('videos.csv', index=False)
    comments_df.to_csv('comments.csv', index=False)


if __name__ == '__main__':
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

    api_service_name = 'youtube'
    api_version = 'v3'
    DEVELOPER_KEY = ''

    youtube = build(api_service_name, api_version, developerKey=DEVELOPER_KEY)

    channels = {
        'singingbanana': 'UCMpizQXRt817D0qpBQZ2TlA',
        'Flammable Maths': 'UCtAIs1VCQrymlAnw3mGonhw',
        'blackpenredpen': 'UC_SvYP0k05UKiJ_2ndB02IA',
        'Mathologer': 'UC1_uAIS3r8Vu6JjXWvastJg',
        'Numberphile': 'UCoxcjq-8xIDTYp3uz647V5A',
        'Stand-up Maths': 'UCSju5G2aFaWMqn-_0YBtq5A',
        '3Blue1Brown': 'UCYO_jab_esuFRV4b17AJtAw',
        'PBS Infinite Series': 'UCs4aHmggTfFrpkPcWSaBN9g',
        'Think Twice': 'UC9yt3wz-6j19RwD5m5f6HSg',
        'Tipping Point Math': 'UCjwOWaOX-c-NeLnj_YGiNEg',
    }

    main(youtube, channels, published_after='2018-01-01T00:00:00Z')
