import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer


def videos_topics(videos, n_topics=100):
    # initialise the count vectorizer on unigrams and bigrams
    count_vectorizer = CountVectorizer(ngram_range=(1, 2), max_df=0.25)

    # fit and transform the cleaned concatenated titles and descriptions
    count_data = count_vectorizer.fit_transform(videos['text'])

    # create and fit the LDA model
    lda_model = LDA(n_components=n_topics)
    lda_model.fit(count_data)

    return lda_model, count_vectorizer, count_data


def plot_topic_coherences():
    topic_coherences = [
        13.10542855, 10.07430240, 7.39045372, 7.05280857, 6.84086092,
        5.46073961, 5.10062173, 5.24915403, 4.26385513, 4.10724026, 3.95023603,
        3.60677561, 3.59096157, 3.23352124, 3.48632792, 2.37446330, 2.88451832,
        2.93545834, 1.80005155, 1.96767417, 2.20581819, 1.53093132, 1.03782973,
        1.14329335, 1.02046108, 1.23064491, 1.13852566, 0.51374761, 0.06577404,
        -0.11612951, 0.43641142, -0.01236949, -0.25050616, 0.14112911,
        0.10358469, -0.44994464, -0.39790208, -0.17311786, -0.19550505,
        -0.79630481, -0.41375621, -0.75967653, -0.66248884, -0.56598616,
        -0.14012647, -0.85539015, -0.64803524, -0.41943068, -0.98889044,
        -0.64623062, -1.45509967, -1.15375189, -1.42048661, -1.52228527,
        -1.23114186, -1.56184726, -0.94350754, -2.00162230, -1.34769916,
        -1.60850756, -1.69398952, -1.76253316, -1.60087801, -1.36366998,
        -1.64739617, -1.50121661, -2.06342364, -1.92351621, -1.98118651,
        -1.79174578, -1.88786471, -1.81752226, -1.86222102, -2.00271827,
        -2.20810170, -2.15333084, -2.26279020, -2.10433915, -2.29697671,
        -2.34286259, -2.41025434, -2.28386305, -2.24622509, -2.20310440,
        -2.06115300, -2.22616060, -2.01116398, -2.25556844, -2.48931510,
        -2.25678980, -2.37570146, -2.48222114, -2.50038877, -2.28629123,
        -2.34635088, -2.57942416, -2.75233835, -2.47965219, -2.71589476,
        -3.07572026
    ]
    plt.scatter(range(1, len(topic_coherences) + 1), topic_coherences, s=14)
    plt.scatter(58, topic_coherences[57], s=16)
    plt.xlabel('Number of topics')
    plt.ylabel('Mean topic coherence')
    plt.xticks([1, 20, 40, 60, 80, 100])
    plt.yticks([])
    plt.savefig('topic_coherence.pdf')
    plt.show()


def print_topics(lda_model, count_vectorizer, verbose=True, n_tokens=10):
    tokens = count_vectorizer.get_feature_names()
    topics = []
    for i, topic in enumerate(lda_model.components_):
        topic_tokens = [tokens[j] for j in topic.argsort()[:-n_tokens - 1:-1]]
        topics.append(topic_tokens)

        if verbose:
            print(f'Topic #{i + 1}')
            print(topic_tokens, end='\n\n')

    return np.array(topics)


if __name__ == '__main__':
    videos = pd.read_csv('../clean/clean_videos.csv')

    lda_model, count_vectorizer, _ = videos_topics(videos)

    # print the topics found by the LDA model
    print('Topics found via LDA:')
    print(print_topics(lda_model, count_vectorizer))

    # plot the topic coherence for LDA models with 1 to 100 topics.
    plot_topic_coherences()
