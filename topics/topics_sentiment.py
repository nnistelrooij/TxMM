import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from videos_topics import *
from sentiment.videos_sentiment import videos_sentiment_distr


def topics_sentiment_distr(video_sentiment_distrs, topic_distrs, lambdas):
    """
    Computes distribution of sentiment of each topic.

    Parameters
    ----------
    video_sentiment_distrs : (n_videos, n_samples) np.ndarray
        Posterior sample distribution of the sentiment of each video.
    topic_distrs : (n_videos, n_topics) np.ndarray
        Distribution of topics of each video.
    lambdas : (n_lambdas,) np.ndarray
        Consecutive intervals of lambda to compute distribution for.

    Returns
    -------
    sentiment_distrs : (n_topics, n_lambdas - 1) np.ndarray
        Distribution of sentiment of each topic.
    """
    # add topics dimension to video_sentiment_distrs
    video_sentiment_distrs = video_sentiment_distrs[:, np.newaxis]

    topic_sentiment_distrs = np.empty((topic_distrs.shape[1], len(lambdas) - 1))
    interval_iter = tqdm(list(zip(lambdas, lambdas[1:])), desc='Topic lambdas')
    for i, lambda_interval in enumerate(interval_iter):
        # posterior = p(left < sentiment <= right|video)
        lambda_left, lambda_right = lambda_interval
        posterior = np.mean(
            (lambda_left < video_sentiment_distrs) &
            (video_sentiment_distrs <= lambda_right),
            axis=-1,
        )

        # joint = p(left < sentiment <= right, topic), marginal = p(topic)
        joint = np.mean(posterior * topic_distrs, axis=0)
        marginal = np.mean(topic_distrs, axis=0)

        # joint / marginal = p(left < sentiment <= right|topic)
        topic_sentiment_distrs[:, i] = joint / marginal

    return topic_sentiment_distrs


if __name__ == '__main__':
    videos_df = pd.read_csv('../clean/clean_videos.csv')
    comments_df = pd.read_csv('../sentiment/sentiment_comments.csv')

    sentiment_distr = videos_sentiment_distr(videos_df, comments_df, verbose=False)

    n_topics = 100
    lda, count_vectorizer = videos_topics(videos_df, n_topics)
    topic_distrs = lda.transform(count_vectorizer.transform(videos_df['text']))

    n_lambdas = 1001
    lambdas = np.linspace(0, 1, n_lambdas)
    final_sentiment_distr = topics_sentiment_distr(
        sentiment_distr, topic_distrs, lambdas,
    )

    width = 1 / (n_lambdas - 1)
    for idx in np.random.randint(n_topics, size=5):
        plt.bar(lambdas[:-1] + width / 2, final_sentiment_distr[idx], width)
        plt.title(f'Posterior probability density of topic {idx}')
        plt.xlabel('$\lambda$')
        plt.ylabel('Probability')
        plt.xlim(0, 1)
        plt.show()
