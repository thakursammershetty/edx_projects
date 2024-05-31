import os
import nltk
import string
import math

def load_files(directory):
    files = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                files[filename] = file.read()
    return files

def tokenize(document):
    nltk.download('punkt')
    nltk.download('stopwords')
    words = nltk.word_tokenize(document.lower())
    stopwords = set(nltk.corpus.stopwords.words("english"))
    words = [word for word in words if word not in string.punctuation and word not in stopwords]
    return words

def compute_idfs(documents):
    idfs = {}
    total_docs = len(documents)
    all_words = set(word for words in documents.values() for word in words)
    
    for word in all_words:
        f = sum(word in documents[doc] for doc in documents)
        idfs[word] = math.log(total_docs / f)
    
    return idfs

def top_files(query, files, idfs, n):
    tfidf_scores = {}
    
    for filename, words in files.items():
        tfidf_scores[filename] = 0
        for word in query:
            if word in words:
                tf = words.count(word)
                tfidf_scores[filename] += tf * idfs[word]
    
    ranked_files = sorted(tfidf_scores, key=tfidf_scores.get, reverse=True)
    return ranked_files[:n]

def top_sentences(query, sentences, idfs, n):
    sentence_scores = []

    for sentence, words in sentences.items():
        idf_score = sum(idfs[word] for word in query if word in words)
        term_density = sum(1 for word in words if word in query) / len(words)
        sentence_scores.append((sentence, idf_score, term_density))
    
    ranked_sentences = sorted(sentence_scores, key=lambda item: (item[1], item[2]), reverse=True)
    return [item[0] for item in ranked_sentences[:n]]

def main():
    directory = "corpus"

    # Calculate IDF values across files
    files = load_files(directory)
    file_words = {filename: tokenize(files[filename]) for filename in files}
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = {}
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCES_MATCHES)
    for match in matches:
        print(match)

if __name__ == "__main__":
    FILE_MATCHES = 1
    SENTENCES_MATCHES = 1
    main()

