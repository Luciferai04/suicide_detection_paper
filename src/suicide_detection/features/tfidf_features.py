from dataclasses import dataclass
from typing import Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


@dataclass
class TfidfFeatures:
    """Create TF-IDF features combining word and character n-grams.

    This preserves short-text signals common in social media.
    """

    word_ngram: Tuple[int, int] = (1, 3)
    char_ngram: Tuple[int, int] = (3, 5)
    max_features: int = 200_000
    min_df: int = 1
    sublinear_tf: bool = True

    def build(self) -> FeatureUnion:
        word_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=self.word_ngram,
            max_features=self.max_features,
            min_df=self.min_df,
            sublinear_tf=self.sublinear_tf,
        )
        char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.char_ngram,
            max_features=self.max_features,
            min_df=self.min_df,
            sublinear_tf=self.sublinear_tf,
        )
        return FeatureUnion([("word", word_vec), ("char", char_vec)])
