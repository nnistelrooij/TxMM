import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import cohen_kappa_score


def factorize_predictions(sentiments, cut_off=0.1):
    factor = np.full_like(sentiments, 1)
    factor[sentiments < cut_off] = 0
    factor[sentiments > (1 - cut_off)] = 2

    return factor


def make_test_comments():
    # make comments that will be annotated
    comments = pd.read_csv('sentiment_comments.csv')

    test_comments = comments[['text', 'sentiment']].sample(100)
    test_comments['class'] = factorize_predictions(test_comments['sentiment'])

    test_comments.to_csv('test_comments.csv', index=False)


def plot_predictions(sentiments):
    # show sentiment predictions with the y-axis log scaled
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

    # show sentiment predictions with the x-axis log scaled
    exponents = np.linspace(1, 10.66, 101)
    down = 0.5**exponents[::-1]
    up = 1 - 0.5**exponents[1:]
    intervals = np.concatenate([down, up])

    probs = []
    for left, right in zip(intervals, intervals[1:]):
        probs.append(np.sum((left < sentiments) & (sentiments <= right)))

    bins = np.concatenate([
        (-exponents[:0:-1] + 1) + (exponents[:0:-1] - exponents[-2::-1]) / 2,
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
    comments = pd.read_csv('sentiment_comments.csv')

    # find out the most negative and most positive comments
    sentiments = np.argsort(comments['sentiment'].to_numpy())
    print('Most negative comment:', comments.loc[sentiments[0], 'text'])
    print('Most positive comment:', comments.loc[sentiments[-1], 'text'])

    # plot sentiment predictions with x-axis or y-axis log scaled
    plot_predictions(comments['sentiment'])

    # compute Cohen's kappa between human annotators
    comments_Niels = pd.read_csv('Test_Niels.csv').sort_values('text')
    comments_Jelle = pd.read_csv('Test_Jelle.csv', sep=';').sort_values('text')
    print('Niels-Jelle:', cohen_kappa_score(
        comments_Niels['class'],
        comments_Jelle['class'],
    ))

    # compute Cohen's kappas between classifier and human annotators
    test_comments = comments[comments['text'].isin(comments_Niels['text'])]
    test_comments = test_comments.sort_values('text')
    test_predictions = pd.unique(test_comments['sentiment'])
    test_classes = factorize_predictions(test_predictions)
    print('Niels-Classifier:', cohen_kappa_score(
        comments_Niels['sentiment_class'],
        test_classes,
    ))
    print('Jelle-Classifier:', cohen_kappa_score(
        comments_Jelle['sentiment_class'],
        test_classes,
    ))
