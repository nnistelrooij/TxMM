import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from videos_topics import print_topics, videos_topics
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
    topic_sentiment_distrs : (n_topics, n_lambdas - 1) np.ndarray
        Distribution of sentiment of each topic.
    lambda_means : (n_lambdas - 1) np.ndarray
        Mean of lambda in each consecutive interval.
    """
    # add topics dimension to video_sentiment_distrs
    video_sentiment_distrs = video_sentiment_distrs[:, np.newaxis]

    n_topics = topic_distrs.shape[1]
    topic_sentiment_distrs = np.empty((n_topics, len(lambdas) - 1))
    lambda_means = np.empty(len(lambdas) - 1)
    interval_iter = tqdm(list(zip(lambdas, lambdas[1:])), desc='Topic lambdas')
    for i, lambda_interval in enumerate(interval_iter):
        # posterior = p(left < sentiment <= right|video)
        lambda_left, lambda_right = lambda_interval
        lambdas_in_range = (
            (lambda_left < video_sentiment_distrs) &
            (video_sentiment_distrs <= lambda_right)
        )
        posterior = np.mean(lambdas_in_range, axis=-1)
        lambda_means[i] = np.mean(video_sentiment_distrs[lambdas_in_range])

        # joint = p(left < sentiment <= right, topic), marginal = p(topic)
        joint = np.mean(posterior * topic_distrs, axis=0)
        marginal = np.mean(topic_distrs, axis=0)

        # joint / marginal = p(left < sentiment <= right|topic)
        topic_sentiment_distrs[:, i] = joint / marginal

    return topic_sentiment_distrs, lambda_means


def plot_example(
    video_sentiment_distrs,
    topic_sentiment_distrs,
    topic_distrs,
    lambdas,
    lambda_means,
):
    """
    Plot example of topic sentiment inference.

    Parameters
    ----------
    video_sentiment_distrs : (n_videos, n_samples) np.ndarray
        Posterior sample distribution of the sentiment of each video.
    topic_sentiment_distrs : (n_topics, n_lambdas - 1) np.ndarray
        Distribution of sentiment of each topic.
    topic_distrs : (n_videos, n_topics) np.ndarray
        Distribution of topics of each video.
    lambdas : (n_lambdas,) np.ndarray
        Consecutive intervals of lambda to compute distribution for.
    lambda_means : (n_lambdas - 1) np.ndarray
        Mean of lambda in each consecutive interval.
    """
    first_range_idx = np.argmin(np.isnan(lambda_means))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    width = 1 / (len(lambdas) - 1)
    bins = lambdas[:-1] + width / 2

    fig = plt.figure(figsize=(9.6, 4.2))
    gs = fig.add_gridspec(2, 4)

    topic_idx = np.random.randint(topic_distrs.shape[1])

    topic_ax = fig.add_subplot(gs[:, 2:])
    topic_distr = -np.sort(-topic_distrs[:, topic_idx])
    topic_mean = np.sum(
        lambda_means[first_range_idx:] *
        topic_sentiment_distrs[topic_idx, first_range_idx:]
    )
    topic_ax.bar(
        bins,
        height=topic_sentiment_distrs[topic_idx],
        width=width,
        label=f'$\mathbb{{E}}[\lambda]={topic_mean:.3f}$'
    )

    cum_distr = np.zeros(len(lambdas) - 1)
    most_prevalent_videos = np.argsort(-topic_distrs[:, topic_idx])[:4]
    for i, video_idx in enumerate(most_prevalent_videos):
        # plot the individual video posterior sentiment distribution
        video_ax = fig.add_subplot(gs[i // 2, i % 2])
        video_mean = np.mean(video_sentiment_distrs[video_idx])
        values, _, _ = video_ax.hist(
            video_sentiment_distrs[video_idx],
            bins=100,
            label=f'$\mathbb{{E}}[\lambda]={video_mean:.2f}$',
            color=colors[i + 1],
            ec=colors[i + 1],
        )
        if i >= 2:
            video_ax.set_xlabel('$\lambda$')
        if i % 2 == 0:
            video_ax.set_ylabel('Posterior  $p(\lambda|\mathcal{D}, a, b)$')
        video_ax.set_xlim(-0.05, 1.05)
        video_ax.axes.yaxis.set_ticks([])
        video_ax.legend(
            fontsize='small', loc='upper left', bbox_to_anchor=(-0.023, 1.029),
        )

        # plot the video distribution as part of the topic distribution
        topic_sentiment_distr = topics_sentiment_distr(
            video_sentiment_distrs[[video_idx]],
            np.ones_like(topic_distrs),
            lambdas,
        )[0][topic_idx] * topic_distr[i] / np.sum(topic_distr)
        topic_ax.bar(
            bins,
            height=topic_sentiment_distr,
            width=width,
            bottom=cum_distr,
            label=f'$\pi_{{{topic_idx},{video_idx}}}={topic_distr[i]:.3f}$',
        )
        cum_distr += topic_sentiment_distr

    topic_ax.set_xlabel('$\lambda$')
    topic_ax.set_ylabel(
        f'Posterior  $p(\lambda|\mathbf{{\pi}}_{{{topic_idx}}},'
        '\mathbf{D}, \mathbf{a}, \mathbf{b})$'
    )
    topic_ax.set_xlim(-0.05, 1.05)
    topic_ax.axes.yaxis.set_ticks([])
    topic_ax.legend()

    plt.tight_layout()
    plt.savefig('topic_inference.pdf')
    plt.show()


def plot_means(topic_sentiment_distrs, lambda_means, n_bins=20):
    """
    Plot distribution of means of topic sentiment distributions.

    Parameters
    ----------
    topic_sentiment_distrs : (n_topics, n_lambdas - 1) np.ndarray
        Distribution of sentiment of each topic.
    lambda_means : (n_lambdas - 1) np.ndarray
        Mean of lambda in each consecutive interval.
    n_bins : int
        Number of bins for the histogram.
    """
    # show distribution of mean topic sentiments
    first_range_idx = np.argmin(np.isnan(lambda_means))
    topic_means = np.sum(
        lambda_means[first_range_idx:] *
        topic_sentiment_distrs[:, first_range_idx:],
        axis=1,
    )
    mean = np.mean(topic_means)
    plt.hist(
        topic_means,
        bins=n_bins,
        label=f'$\mathbb{{E}}[\mathbb{{E}}[\lambda]]={mean:.3f}$',
    )
    plt.xlabel('$\mathbb{{E}}[\lambda]$')
    plt.ylabel('Number of topics')
    plt.xlim(-0.05, 1.05)
    plt.legend()
    plt.savefig('topic_means.pdf')
    plt.show()


def print_table(
    videos_df,
    lda_model,
    count_vectorizer,
    topic_sentiment_distrs,
    topic_distrs,
    lambda_means,
    n_topics=10,
    n_tokens=10,
):
    """Print a LaTeX table that shows the top n_topics best and worst topics.

    Parameters
    ----------
    videos_df : DataFrame
        Pandas DataFrame holding the video YouTube IDs.
    lda_model : sklearn.decomposition.LatentDirichletAllocation
        LDA topic model from the sklearn package.
    count_vectorizer : sklearn.feature_extraction.text.CountVectorizer
        Counts the term occurrences in the video texsts.
    topic_sentiment_distrs : (n_topics, n_lambdas - 1) np.ndarray
        Distribution of sentiment of each topic.
    topic_distrs : (n_videos, n_topics) np.ndarray
        Distribution of topics of each video.
    lambda_means : (n_lambdas - 1) np.ndarray
        Mean of lambda in each consecutive interval.
    n_topics : int
        Number of topics to print for one side of the table.
    n_tokens : int
        Number of tokens to print for each topic.
    """
    topic_distr_idx = np.argsort(-topic_distrs, axis=0)
    topic_tokens = print_topics(
        lda_model, count_vectorizer,
        verbose=False, n_tokens=n_tokens
    )

    first_range_idx = np.argmin(np.isnan(lambda_means))
    topic_means = np.sum(
        lambda_means[first_range_idx:] *
        topic_sentiment_distrs[:, first_range_idx:],
        axis=1,
    )
    worst_idx = np.argsort(topic_means)[:n_topics]
    best_idx = np.argsort(topic_means)[-1:-n_topics - 1:-1]

    format = (
        '\\href{{https://www.youtube.com/watch?v={}}}'
        '{{\\textcolor{{blue}}{{{:.3f}}}}}'
    )
    for best_idx, worst_idx in zip(best_idx, worst_idx):
        print(f'    {topic_means[best_idx]:.3f}', end=' & ')
        print(', '.join(topic_tokens[best_idx]) + ', ...', end=' & ')
        print(format.format(
            videos_df.loc[topic_distr_idx[0, best_idx], 'video_id'],
            topic_distrs[topic_distr_idx[0, best_idx], best_idx],
        ), end=' ')
        print(format.format(
            videos_df.loc[topic_distr_idx[1, best_idx], 'video_id'],
            topic_distrs[topic_distr_idx[1, best_idx], best_idx],
        ), end=' ')
        print(format.format(
            videos_df.loc[topic_distr_idx[2, best_idx], 'video_id'],
            topic_distrs[topic_distr_idx[2, best_idx], best_idx],
        ), end=' ')
        print(format.format(
            videos_df.loc[topic_distr_idx[3, best_idx], 'video_id'],
            topic_distrs[topic_distr_idx[3, best_idx], best_idx],
        ), end=' & ')
    
        print(f'{topic_means[worst_idx]:.3f}', end=' & ')
        print(', '.join(topic_tokens[worst_idx]) + ', ...', end=' & ')
        print(format.format(
            videos_df.loc[topic_distr_idx[0, worst_idx], 'video_id'],
            topic_distrs[topic_distr_idx[0, worst_idx], worst_idx],
        ), end=' ')
        print(format.format(
            videos_df.loc[topic_distr_idx[1, worst_idx], 'video_id'],
            topic_distrs[topic_distr_idx[1, worst_idx], worst_idx],
        ), end=' ')
        print(format.format(
            videos_df.loc[topic_distr_idx[2, worst_idx], 'video_id'],
            topic_distrs[topic_distr_idx[2, worst_idx], worst_idx],
        ), end=' ')
        print(format.format(
            videos_df.loc[topic_distr_idx[3, worst_idx], 'video_id'],
            topic_distrs[topic_distr_idx[3, worst_idx], worst_idx],
        ), end=' \\\\\n')


if __name__ == '__main__':
    # load cleaned data
    videos_df = pd.read_csv('../clean/clean_videos.csv')
    comments_df = pd.read_csv('../sentiment/sentiment_comments.csv')

    # infer the sentiment of all the videos
    video_sentiment_distrs = videos_sentiment_distr(videos_df, comments_df)

    # learn an LDA topic model with 58 topics
    n_topics = 58
    lda_model, count_vectorizer, count_data = videos_topics(videos_df, n_topics)
    topic_distrs = lda_model.transform(count_data)

    # infer the sentiment of all the topics
    n_lambdas = 229
    lambdas = np.linspace(0, 1, n_lambdas)
    topic_sentiment_distrs, lambda_means = topics_sentiment_distr(
        video_sentiment_distrs, topic_distrs, lambdas,
    )

    # plot example of inference of a topic's sentiment
    plot_example(
        video_sentiment_distrs,
        topic_sentiment_distrs,
        topic_distrs,
        lambdas,
        lambda_means,
    )

    # plot distribution of sentiment means of topics
    plot_means(topic_sentiment_distrs, lambda_means)

    # print LaTeX table to show final list of topics ranked on mean sentiment
    print_table(
        videos_df,
        lda_model,
        count_vectorizer,
        topic_sentiment_distrs,
        topic_distrs,
        lambda_means,
    )
