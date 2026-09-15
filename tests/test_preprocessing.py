from src.preprocessing import clean_tweet, label_name


def test_clean_tweet_removes_noise():
    text = "@user I can't wait!!! Visit https://example.com #HappyDay"
    assert clean_tweet(text) == "i can not wait happyday"


def test_label_name_mapping():
    assert label_name(0) == "negative"
    assert label_name(2) == "neutral"
    assert label_name(4) == "positive"
