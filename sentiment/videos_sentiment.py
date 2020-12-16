import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta


def sentiment_prior_beta_abs(videos_df):
    # compute the proportion of likes and dislikes for each video
    n_feedback = videos_df['likes'] + videos_df['dislikes']
    like_proportions = (videos_df['likes'] + 1) / (n_feedback + 2)
    dislike_proportions = (videos_df['dislikes'] + 1) / (n_feedback + 2)

    # compute the strength of the prior for each video based on its views
    feedback_weight = np.log2(videos_df['views'] + 1)

    # compute the a and b parameters of the beta prior distribution
    videos_df['a'] = like_proportions * feedback_weight + 1
    videos_df['b'] = dislike_proportions * feedback_weight + 1

    return videos_df


def log_continuous_bernoulli(sentiments, lambdas):
    point5_index = np.argmax(lambdas == 0.5)
    lambdas_prime = np.delete(lambdas, point5_index, axis=0)

    C = np.log(2 * np.arctanh(1 - 2 * lambdas_prime) / (1 - 2 * lambdas_prime))
    C = np.insert(C, point5_index, values=np.log(2), axis=0)

    return C + sentiments*np.log(lambdas) + (1 - sentiments)*np.log(1 - lambdas)


def sentiment_log_likelihoods(sentiments, lambdas):
    sentiments = np.asarray(sentiments).reshape(1, -1)
    lambdas = np.asarray(lambdas).reshape(-1, 1)
    log_probs = log_continuous_bernoulli(sentiments, lambdas)

    return np.sum(log_probs, axis=1)


def sentiment_log_posteriors(videos_df, comments_df, lambdas):
    log_posteriors = np.empty((videos_df.shape[0], lambdas.shape[0]))
    for i, video in videos_df.iterrows():
        # compute log of beta prior given a and b parameters
        log_prior = beta.logpdf(lambdas, video['a'], video['b'])

        # compute log of likelihood given sentiments of video's comments
        id = video['video_id']
        sentiments = comments_df.loc[comments_df['video_id'] == id, 'sentiment']
        log_likelihood = sentiment_log_likelihoods(sentiments, lambdas)

        # compute log of posterior given log prior and log likelihood
        log_posteriors[i] = log_prior + log_likelihood

    return log_posteriors


def sentiment_maps(lambdas, log_posteriors):
    return lambdas[np.argmax(log_posteriors, axis=1)]


def sentiment_means(lambdas, log_posteriors):
    # compute normalized posteriors
    posteriors = np.exp(log_posteriors)
    posteriors = posteriors / np.sum(posteriors, axis=1, keepdims=True)

    # compute mean posterior lambda
    means = np.sum(lambdas * posteriors, axis=1)

    # determine mean instead of NaN numbers
    posterior_1286 = np.exp(log_posteriors[1286] - 24)
    posterior_1286 = posterior_1286 / np.sum(posterior_1286)
    means[1286] = np.sum(lambdas * posterior_1286)

    posterior_1293 = np.exp(log_posteriors[1293] - 1569)
    posterior_1293 = posterior_1293 / np.sum(posterior_1293)
    means[1293] = np.sum(lambdas * posterior_1293)

    return means


if __name__ == '__main__':
    videos_df = pd.read_csv('../clean/clean_videos.csv')
    videos_df = sentiment_prior_beta_abs(videos_df)

    comments_df = pd.read_csv('sentiment_comments.csv')
    lambdas = np.linspace(0.0001, 0.9999, 9999)
    log_posteriors = sentiment_log_posteriors(videos_df, comments_df, lambdas)

    videos_df = videos_df.assign(MAP=sentiment_maps(lambdas, log_posteriors))
    videos_df = videos_df.assign(mean=sentiment_means(lambdas, log_posteriors))
    videos_df.to_csv('sentiment_videos.csv', index=False)

    _, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].hist(videos_df['mean'], bins=100)
    axs[0].set_title('Means')
    axs[1].hist(videos_df['MAP'], bins=100)
    axs[1].set_title('MAPs')
    plt.show()
