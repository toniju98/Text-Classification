import spacy
import numpy as np
import json
from spacy.lang.de.stop_words import STOP_WORDS

nlp = spacy.load('de_core_news_sm')
STOP_WORDS.update(["Million", "Milliarde", "Version", "Prozent", "Euro", "US-Dollar",
                   "Dollar", "padding-top", "%", "Jahr", "Quartal", "Zeit", "Datum"])


class CleaningData:
    """class for preparing the data for training

    """

    def __init__(self, data):
        self.data = data

    def get_nouns_list(self):
        """creates a list with the noun strings of each article and adds a new column to the data

        """
        result = [CleaningData.noun_filtering(x, y) for x, y in zip(self.data['headline'], self.data["text"])]
        self.data['just_nouns'] = result

    def split_topics(self):
        """ create columns for all topics

        """
        topic_in_doc = []
        top_topics = self.topic_datas()
        for k in top_topics:
            matches = []
            for doc in self.data["topic"]:
                if k == doc:
                    matches.append(1)
                else:
                    matches.append(0)
            topic_in_doc.append(matches)
        for i, k in enumerate(top_topics):
            self.data[k] = topic_in_doc[i]

    @staticmethod
    def noun_filtering(x, y):
        """filters just the nouns and lemmas

        # Arguments
            x: String, e.g. headline.
            y: String, e.g. content
        # Returns
            article_nouns: String with all nouns of the article
        """
        x = x.replace("-", " ")
        y = y.replace("-", " ")
        doc = nlp(x + " " + y)
        nouns = []
        for n in map(lambda z: z.lemma_, filter(lambda m: (m.pos_ == 'NOUN') and (m.lemma_ not in STOP_WORDS), doc)):
            # evtl. hier Worte mit führenden Ziffern weglassen
            nouns.append(n)
        article_nouns = " ".join(nouns)
        return article_nouns

    def get_data(self):
        """returns the data

            # Returns
                data: pd.Dataframe with the whole data
            """
        return self.data

    def topic_datas(self):
        """Filters the 50 top topics

        :return:
            top_topics: list of the 50 top keywords
        """
        all_topics = self.data["topic"]
        u, indices = np.unique(all_topics, return_inverse=True)
        top_topics = u[np.argsort(np.bincount(indices))[-50::]].tolist()
        with open('topics.json', 'w', encoding='utf-8') as f:
            json.dump(top_topics, f, ensure_ascii=False, indent=4)
        return top_topics
