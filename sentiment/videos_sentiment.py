import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from tqdm import trange


def sentiment_prior_params(videos_df):
    # compute the proportion of likes and dislikes for each video
    n_feedback = videos_df['likes'] + videos_df['dislikes']
    like_proportions = (videos_df['likes'] + 1) / (n_feedback + 2)
    dislike_proportions = (videos_df['dislikes'] + 1) / (n_feedback + 2)

    # compute the strength of the prior for each video based on its views
    feedback_weight = np.log2(videos_df['views'] + 1)

    # compute the a and b parameters of the beta prior distribution
    videos_df['a'] = like_proportions * feedback_weight + 1
    videos_df['b'] = dislike_proportions * feedback_weight + 1


def sentiment_likelihood_params(videos_df, comments_df):
    # for each video, add the number of comments and lambda exponents
    for i, video in videos_df.iterrows():
        # get the comments of the current video in comments_df
        comments = comments_df[comments_df['video_id'] == video['video_id']]

        # determine the actual number comments in the data
        n_sentiments = comments.shape[0]
        videos_df.loc[i, 'comments'] = n_sentiments

        # use number and sum of sentiments to make likelihood lambda exponents
        sentiments_sum = np.sum(comments['sentiment'])
        videos_df.loc[i, 'a'] = sentiments_sum
        videos_df.loc[i, 'b'] = n_sentiments - sentiments_sum


def sentiment_joint_params(videos_df, comments_df):
    # for each video, add the number of comments and lambda exponents
    for i, video in videos_df.iterrows():
        # get the comments of the current video in comments_df
        comments = comments_df[comments_df['video_id'] == video['video_id']]

        # determine the actual number comments in the data
        n_sentiments = comments.shape[0]
        videos_df.loc[i, 'comments'] = n_sentiments

        # use number and sum of sentiments to determine joint lambda exponents
        sentiments_sum = np.sum(comments['sentiment'])
        videos_df.loc[i, 'a'] = video['a'] - 1 + sentiments_sum
        videos_df.loc[i, 'b'] = video['b'] - 1 + n_sentiments - sentiments_sum


def sentiment_proposal_params(videos_df):
    # add first guess of half_interval, most videos are covered
    videos_df['half_interval'] = 0.4616 - 0.14 * np.clip(
        np.log(videos_df['comments'] * np.log(videos_df['a']) / videos_df['b']),
        a_min=1.5, a_max=3.2,
    )

    if videos_df.shape[0] > 1:
        # change half_interval for videos with accept rate smaller than 0.33
        videos_df.loc[930, 'half_interval'] /= 18
        videos_df.loc[530, 'half_interval'] /= 6
        videos_df.loc[[1182, 1294, 1180, 526], 'half_interval'] /= 4
        videos_df.loc[1293, 'half_interval'] /= 3.5
        videos_df.loc[[1285, 1023, 1323], 'half_interval'] /= 2.7
        videos_df.loc[[
            1171, 1159, 1160, 1227, 1115, 643
        ], 'half_interval'] /= 1.7
        videos_df.loc[[
            32, 70, 1286, 564, 1309, 681, 689, 691, 747, 893, 1003, 1100, 1112,
            1126, 1141, 1162, 1186, 1241, 1245, 1262, 1267, 1277, 1280, 77, 575,
        ], 'half_interval'] /= 1.6

        # change half_interval for videos with accept rate larger than 0.66
        videos_df.loc[[
            137, 909, 279, 925, 244, 381, 339, 366, 147, 963, 434, 1328, 806
        ], 'half_interval'] *= 1.3


def sentiment_params(videos_df, comments_df):
    # put a and b hyper-parameters of beta prior into videos DataFrame
    sentiment_prior_params(videos_df)

    # put number of comments and joint lambda exponents into videos DataFrame
    sentiment_joint_params(videos_df, comments_df)

    # put interval lengths of proposal distribution for accept rate around 0.5
    sentiment_proposal_params(videos_df)


def cb_consts(probs):
    # fill array with continuous bernoulli constant for prob == 0.5
    consts = np.full_like(probs, 2)

    # substitute default constant whenever prob != 0.5
    pnt5 = probs == 0.5
    consts[~pnt5] *= np.arctanh(1 - 2 * probs[~pnt5]) / (1 - 2 * probs[~pnt5])

    return consts


def sentiment_log_joints(videos_df, probs):
    log_cb_consts = videos_df['comments'] * np.log(cb_consts(probs))
    pos_probs = videos_df['a'] * np.log(probs)
    neg_probs = videos_df['b'] * np.log(1 - probs)

    return log_cb_consts + pos_probs + neg_probs


def log_proposal_prob(videos_df, probs):
    return -np.log(
        np.minimum(probs + videos_df['half_interval'], 1) -
        np.maximum(probs - videos_df['half_interval'], 0)
    )


def acceptance_fn(videos_df, p_candidate, p_current):
    acceptance = (
        sentiment_log_joints(videos_df, p_candidate) +
        log_proposal_prob(videos_df, p_candidate) -
        sentiment_log_joints(videos_df, p_current) -
        log_proposal_prob(videos_df, p_current)
    )

    return np.minimum(1, np.exp(acceptance))


def metropolis_hastings(videos_df, n_samples=100000, burn_in=5000):
    n_videos = videos_df.shape[0]
    n_samples += burn_in

    samples = np.full((n_videos, n_samples), 0.5)
    n_accepted = np.zeros(n_videos)
    for t in trange(1, n_samples, initial=1, total=n_samples, desc='Sampling'):
        # sample new candidates from the uniform proposal distribution
        candidates = np.random.uniform(
            np.maximum(samples[:, t - 1] - videos_df['half_interval'], 0),
            np.minimum(samples[:, t - 1] + videos_df['half_interval'], 1),
        )
        # compute acceptance probabilities of the new candidates
        accept_prob = acceptance_fn(videos_df, candidates, samples[:, t - 1])

        # determine whether the new candidates are accepted
        accepted = np.random.uniform([0]*n_videos, [1]*n_videos) <= accept_prob
        n_accepted += (accepted - n_accepted) / t

        # add the samples of time step t to the big array
        samples[accepted, t] = candidates[accepted]
        samples[~accepted, t] = samples[~accepted, t - 1]

    # remove the first burn_in samples from the samples array
    return samples[:, burn_in:], n_accepted


def videos_sentiment_distr(videos_df, comments_df, verbose=True, n_plots=5):
    sentiment_params(videos_df, comments_df)
    samples, n_accepted = metropolis_hastings(videos_df)

    if verbose:
        print('Few accepted videos:', np.argsort(n_accepted)[:10].to_numpy())
        print('Many accepted videos:', np.argsort(n_accepted)[-10:].to_numpy())

        plt.hist(n_accepted, bins=100)
        plt.title('Acceptance rates during Metropolis-Hastings')
        plt.xlabel('Acceptance rate')
        plt.ylabel('Number of videos')
        plt.show()

        for idx in np.random.randint(videos_df.shape[0], size=n_plots):
            plt.hist(samples[idx], bins=100, density=True)
            plt.title(f'Posterior probability density of video {idx}')
            plt.xlabel('$\lambda$')
            plt.ylabel('Probability density')
            plt.xlim(-0.05, 1.05)
            plt.show()

    return samples


if __name__ == '__main__':
    videos_df = pd.read_csv('../clean/clean_videos.csv')
    comments_df = pd.read_csv('sentiment_comments.csv')

    # initialize the necessary variables for the distributions and figure
    idx = 385
    videos_df = videos_df.loc[[idx]]
    lambdas = np.linspace(0.001, 0.999, 999)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))

    # show the prior sentiment distribution
    sentiment_prior_params(videos_df)
    video = videos_df.loc[idx]
    prior_mean = video['a'] / (video['a'] + video['b'])
    print('Prior mean:', prior_mean)
    ax[0].plot(
        lambdas,
        beta.pdf(lambdas, video['a'], video['b']),
        label=f'$\mathbb{{E}}[\lambda]={prior_mean:.3f}$',
    )
    ax[0].set_xlabel('$\lambda$')
    ax[0].set_ylabel('Prior  Beta$(\lambda|a, b)$')
    ax[0].axes.yaxis.set_ticks([])
    ax[0].legend()

    # show the likelihood sentiment distribution
    sentiment_likelihood_params(videos_df, comments_df)
    video = videos_df.loc[idx]
    const = video['comments'] * np.log(cb_consts(lambdas))
    ll = const + video['a'] * np.log(lambdas) + video['b'] * np.log(1 - lambdas)
    likelihood_mean = np.sum(lambdas * np.exp(ll) / np.sum(np.exp(ll)))
    print('Likelihood mean:', likelihood_mean)
    ax[1].plot(
        lambdas,
        np.exp(ll),
        label=f'$\mathbb{{E}}[\lambda]={likelihood_mean:.3f}$',
    )
    ax[1].set_xlabel('$\lambda$')
    ax[1].set_ylabel('Likelihood  $p(\mathcal{D}|\lambda)$')
    ax[1].axes.yaxis.set_ticks([])
    ax[1].legend()

    # show the posterior sentiment sample distribution
    samples = videos_sentiment_distr(videos_df, comments_df, verbose=False)
    posterior_mean = np.mean(samples[0])
    print('Posterior mean:', posterior_mean)
    n, _, _ = ax[2].hist(
        samples[0],
        bins=100,
        label=f'$\mathbb{{E}}[\lambda]={posterior_mean:.3f}$',
    )
    ax[2].set_xlabel('$\lambda$')
    ax[2].set_ylabel('Posterior  $p(\lambda|\mathcal{D}, a, b)$')
    ax[2].set_xlim(-0.05, 1.05)
    ax[2].set_ylim(bottom=-0.05 * max(n))
    ax[2].axes.yaxis.set_ticks([])
    ax[2].legend()

    plt.tight_layout()
    plt.savefig('inference.pdf')
    plt.show()

    videos_df = pd.read_csv('../clean/clean_videos.csv')
    comments_df = pd.read_csv('sentiment_comments.csv')
    samples = videos_sentiment_distr(videos_df, comments_df)

    mean = np.mean(samples)
    plt.hist(
        np.mean(samples, axis=1),
        bins=100,
        label=f'$\mathbb{{E}}[\mathbb{{E}}[\lambda]]={mean:.3f}$',
    )
    plt.xlabel('$\mathbb{{E}}[\lambda]$')
    plt.ylabel('Number of videos')
    plt.xlim(-0.05, 1.05)
    plt.legend()
    plt.savefig('sentiment_means.pdf')
    plt.show()
