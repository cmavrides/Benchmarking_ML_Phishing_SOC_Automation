from phishing_benchmark.features.vectorizers import get_tfidf


def test_tfidf_vectorizer_shape():
    texts = ["phishing email", "legitimate newsletter"]
    bundle = get_tfidf(max_features=10)
    matrix = bundle.fit_transform(texts)
    assert matrix.shape[0] == 2
    assert matrix.shape[1] <= 10
