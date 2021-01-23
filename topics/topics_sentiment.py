import matplotlib.pyplot as plt
import numpy as np
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


def plot_example(
    topic_distrs,
    video_sentiment_distrs,
    topic_sentiment_distrs,
    lambdas,
):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    width = 1 / (len(lambdas) - 1)
    bins = lambdas[:-1] + width / 2

    fig = plt.figure(figsize=(9.6, 4.2))
    gs = fig.add_gridspec(2, 4)

    topic_idx = np.random.randint(topic_distrs.shape[1])

    topic_ax = fig.add_subplot(gs[:, 2:])
    topic_distr = -np.sort(-topic_distrs[:, topic_idx])
    topic_mean = np.sum(bins * topic_sentiment_distrs[topic_idx])
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
        )[topic_idx] * topic_distr[i] / np.sum(topic_distr)
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


def plot_means(topic_sentiment_distrs, lambdas):
    width = 1 / (len(lambdas) - 1)
    bins = lambdas[:-1] + width / 2

    # show distribution of mean topic sentiments
    mean = np.mean(np.sum(bins * topic_sentiment_distrs, axis=1))
    plt.hist(
        np.sum(bins * topic_sentiment_distrs, axis=1),
        bins=19,
        label=f'$\mathbb{{E}}[\mathbb{{E}}[\lambda]]={mean:.3f}$',
    )
    plt.xlabel('$\mathbb{{E}}[\lambda]$')
    plt.ylabel('Number of topics')
    plt.xlim(-0.05, 1.05)
    plt.legend()
    plt.savefig('topic_means.pdf')
    plt.show()


# Helper function
def print_topics(lda, count_vectorizer, n_top_words=10):
    words = count_vectorizer.get_feature_names()
    topics = []
    for i, topic in enumerate(lda.components_):
        topic_words = [words[j] for j in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append(topic_words)

    return np.array(topics)


def print_table(
    videos_df,
    lda,
    count_vectorizer,
    topic_distrs,
    sentiment_distrs,
    lambdas,
):
    width = 1 / (len(lambdas) - 1)
    bins = lambdas[:-1] + width / 2
    
    topic_words = print_topics(lda, count_vectorizer, 10)
    worst_idx = np.argsort(np.sum(bins * sentiment_distrs, axis=1))[:10]
    best_idx = np.argsort(np.sum(bins * sentiment_distrs, axis=1))[-1:-11:-1]
    wolla = np.argsort(topic_distrs, axis=0)
    format = (
        '\\href{{https://www.youtube.com/watch?v={}}}'
        '{{\\textcolor{{blue}}{{{}}}}}'
    )
    for best_idx, worst_idx in zip(best_idx, worst_idx):
        print(f'    {np.sum(bins * sentiment_distrs[best_idx]):.3f}', end=' & ')
        print(', '.join(topic_words[best_idx]), end=' & ')
        print(format.format(
            videos_df.loc[wolla[-1, best_idx], 'video_id'],
            topic_distrs[wolla[-1, best_idx], best_idx],
        ), end=' ')
        print(format.format(
            videos_df.loc[wolla[-2, best_idx], 'video_id'],
            topic_distrs[wolla[-2, best_idx], best_idx],
        ), end=' ')
        print(format.format(
            videos_df.loc[wolla[-3, best_idx], 'video_id'],
            topic_distrs[wolla[-3, best_idx], best_idx],
        ), end=' ')
        print(format.format(
            videos_df.loc[wolla[-4, best_idx], 'video_id'],
            topic_distrs[wolla[-4, best_idx], best_idx],
        ), end=' & ')
    
        print(f'{np.sum(bins * sentiment_distrs[worst_idx]):.3f}', end=' & ')
        print(', '.join(topic_words[worst_idx]), end=' & ')
        print(format.format(
            videos_df.loc[wolla[-1, worst_idx], 'video_id'],
            topic_distrs[wolla[-1, worst_idx], worst_idx],
        ), end=' ')
        print(format.format(
            videos_df.loc[wolla[-2, worst_idx], 'video_id'],
            topic_distrs[wolla[-2, worst_idx], worst_idx],
        ), end=' ')
        print(format.format(
            videos_df.loc[wolla[-3, worst_idx], 'video_id'],
            topic_distrs[wolla[-3, worst_idx], worst_idx],
        ), end=' ')
        print(format.format(
            videos_df.loc[wolla[-4, worst_idx], 'video_id'],
            topic_distrs[wolla[-4, worst_idx], worst_idx],
        ), end=' \\\\\n')


if __name__ == '__main__':
    videos_df = pd.read_csv('../clean/clean_videos.csv')
    comments_df = pd.read_csv('../sentiment/sentiment_comments.csv')

    video_sentiment_distrs = videos_sentiment_distr(videos_df, comments_df)

    n_topics = 100
    lda, count_vectorizer = videos_topics(videos_df, n_topics)
    topic_distrs = lda.transform(count_vectorizer.transform(videos_df['text']))

    n_lambdas = 229
    lambdas = np.linspace(0, 1, n_lambdas)
    topic_sentiment_distrs = topics_sentiment_distr(
        video_sentiment_distrs, topic_distrs, lambdas,
    )

    plot_example(
        topic_distrs,
        video_sentiment_distrs,
        topic_sentiment_distrs,
        lambdas,
    )
    plot_means(topic_sentiment_distrs, lambdas)
    print_table(
        videos_df,
        lda,
        count_vectorizer,
        topic_distrs,
        topic_sentiment_distrs,
        lambdas,
    )
