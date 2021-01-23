import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import cohen_kappa_score


def factorize_predictions(sentiment, cut_off):
    wolla = np.full_like(sentiment, 1)
    wolla[sentiment < cut_off] = 0
    wolla[sentiment > (1 - cut_off)] = 2

    return wolla


def make_test_comments():
    comments = pd.read_csv('./sentiment_comments.csv')

    test_comments = comments[['text', 'sentiment']].sample(100)
    test_comments['sentiment_class'] = test_comments['sentiment'].map(factorize_predictions)
    print(test_comments[['sentiment', 'sentiment_class']].head())

    test_comments.to_csv('test_comments.csv', index=False)


def compute_cohens_kappa(comments1, comments2):
    return cohen_kappa_score(
        comments1,
        comments2,
    )


def plot_predictions(sentiments):
    mean = np.mean(sentiments)
    plt.hist(
        sentiments,
        bins=100,
        label=f'mean={mean:.3f}',
    )
    plt.yscale('log')
    plt.xlabel('$p($sentiment$=1|$comment$)$')
    plt.ylabel('Number of comments')
    plt.legend()
    plt.savefig('comment_predictions_log.pdf')
    plt.show()

    exponents = np.linspace(1, 10.66, 101)
    down = 0.5**exponents[::-1]
    up = 1 - 0.5**exponents[1:]
    intervals = np.concatenate([down, up])

    probs = []
    for left, right in zip(intervals, intervals[1:]):
        probs.append(np.sum((left < sentiments) & (sentiments <= right)))

    bins = np.concatenate([
        (-exponents[1:][::-1] + 1) + (exponents[1:][::-1] - exponents[:-1][::-1]) / 2,
        (exponents[1:] - 1) - (exponents[1:] - exponents[:-1]) / 2,
    ])
    plt.bar(
        bins,
        height=probs,
        width=9.66 / 100,
        label=f'mean={np.sum(bins * probs / np.sum(probs)):.3f}',
    )
    plt.xlabel('Confidence')
    plt.ylabel('Number of comments')
    plt.xlim(-11, 11)
    plt.legend()
    plt.savefig('comment_predictions_conf.pdf')
    plt.show()


if __name__ == '__main__':
    comments = pd.read_csv('./sentiment_comments.csv')
    print(comments.loc[np.argsort(comments['sentiment']).iloc[0], 'text'])
    print(comments.loc[np.argsort(comments['sentiment']).iloc[-1], 'text'])

    plot_predictions(comments['sentiment'])

    comments_Niels = pd.read_csv(r'C:\Users\Niels-laptop\Desktop\Test_Niels.csv')
    comments_Niels = comments_Niels.sort_values('text')
    comments_Jelle = pd.read_csv(r'C:\Users\Niels-laptop\Desktop\Test_Jelle.csv', sep=';')
    comments_Jelle = comments_Jelle.sort_values('text')
    print('Niels-Jelle:', compute_cohens_kappa(
        comments_Niels['sentiment_class'],
        comments_Jelle['sentiment_class'])
    )

    test_comments = comments[comments['text'].isin(comments_Niels['text'])]
    test_comments = test_comments.sort_values('text')
    test_predictions = pd.unique(test_comments['sentiment'])
    test_classes = factorize_predictions(test_predictions, 0.1)
    print('Niels-classifier:', compute_cohens_kappa(
        comments_Niels['sentiment_class'],
        test_classes
    ))
    print('Jelle-classifier:', compute_cohens_kappa(
        comments_Jelle['sentiment_class'],
        test_classes
    ))
