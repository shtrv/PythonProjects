from collections import Counter
import string

def count_unique_words(text):
    # Приведение текста к нижнему регистру
    text = text.lower()
    
    # Удаление знаков препинания
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Разделение текста на слова
    words = text.split()
    
    # Подсчет количества каждого слова
    word_counts = Counter(words)
    
    # Сортировка слов по количеству в убывающем порядке
    sorted_word_counts = dict(sorted(word_counts.items(), key=lambda item: item[1], reverse=True))
    
    return sorted_word_counts

# Пример использования функции
text = ""
word_counts = count_unique_words(text)

# Вывод результата
for word, count in word_counts.items():
    print(f"{word} - {count}")
