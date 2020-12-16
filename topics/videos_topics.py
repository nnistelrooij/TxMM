import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer


# Helper function
def print_topics(lda, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for i, topic in enumerate(lda.components_):
        topic_words = [words[j] for j in topic.argsort()[:-n_top_words - 1:-1]]
        print(f'\nTopic #{i}\n' + ', '.join(topic_words))


if __name__ == '__main__':
    videos = pd.read_csv('../clean/clean_videos.csv')

    # initialise the count vectorizer on unigrams and bigrams
    count_vectorizer = CountVectorizer(ngram_range=(1, 2), max_df=0.25)

    # fit and transform the cleaned and preprocessed video texts
    count_data = count_vectorizer.fit_transform(videos['text'])

    # create and fit the LDA model
    n_topics = 100
    lda = LDA(n_components=n_topics)
    lda.fit(count_data)

    # print the topics found by the LDA model
    print('Topics found via LDA:')
    print_topics(lda, count_vectorizer, n_top_words=10)
