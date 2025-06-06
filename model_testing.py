from fasttext_words_classifier import get_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def model_testing():
    labels = ['Продукты', 'Транспорт', 'Путешествия',
              'Развлечения', 'Здоровье', 'Техника',
              'Образование', 'Одежда', 'Жилье', 'Прочее']

    with open('test_dataset.txt', 'r', encoding="UTF-8") as file:
        words_list = []
        true_categories = []
        for row in file:
            row = row.split()
            true_categories.append(row[-1])
            s = ' '.join(row[:-1])
            words_list.append(s)

    predicted_categories = list(get_predict(words_list).values())
    
    accuracy = accuracy_score(true_categories, predicted_categories)
    conf_matrix = confusion_matrix(true_categories, predicted_categories, labels=labels)
    class_report = classification_report(true_categories, predicted_categories, target_names=labels)

    return accuracy, conf_matrix, class_report


if __name__ == '__main__':
    model_testing()
