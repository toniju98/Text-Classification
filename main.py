import load_data
from clean_data import CleaningData
from evaluate_model import EvaluationModel

if __name__ == "__main__":
    # TODO: fix path
    path = "C:/Users/tonij/Downloads/tut-text-ml1"
    data = load_data.read_data(path)
    data = data[:1000]
    cleaning_data = CleaningData(data)
    cleaning_data.keywords_datas()
    cleaning_data.split_keywords()
    data = cleaning_data.get_data()
    evaluation_model = EvaluationModel(data, 'Android')
    evaluation_model.evaluate_testdata()
    # training_model.train_ngram_model(data)
