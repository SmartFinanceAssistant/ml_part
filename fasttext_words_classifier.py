import fasttext
import os


categories = {'_l_travelling': 'Путешествия',
              '_l_groceries': 'Продукты',
              '_l_gadjets': 'Техника',
              '_l_transport': 'Транспорт',
              '_l_education': 'Образование',
              '_l_entertainment': 'Развлечения',
              '_l_health': 'Здоровье',
              '_l_housing': 'Жилье',
              '_l_cloth': 'Одежда'}


def get_predict(words_list):
    model_path = "classifier_fasttext.bin"
    model = fasttext.load_model(model_path)
    answer_dict = dict.fromkeys(words_list, '')
    for word in words_list:
        first_predict, probability = model.predict(word)
        if probability <= 0.25:
            answer_dict[word] = 'Прочее'
        elif probability > 0.25:
            answer_dict[word] = categories[first_predict[0]]
    return answer_dict


def train_model():
    model = fasttext.train_supervised(
        input="data_set_fasttext.csv",
        wordNgrams=10,
        epoch=1000,
        lr=0.1,
        loss='softmax',
        label='_l_'
    )
    model.save_model("classifier_fasttext.bin")


if __name__ == '__main__':
    if not os.path.exists("classifier_fasttext.bin"):
        train_model()