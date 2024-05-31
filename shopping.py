import csv
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def load_data(filename):
    months = {
        "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "June": 5, 
        "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11
    }

    evidence = []
    labels = []

    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row

        for row in reader:
            evidence.append([
                int(row[0]),
                float(row[1]),
                int(row[2]),
                float(row[3]),
                int(row[4]),
                float(row[5]),
                float(row[6]),
                float(row[7]),
                float(row[8]),
                float(row[9]),
                months[row[10]],
                int(row[11]),
                int(row[12]),
                int(row[13]),
                int(row[14]),
                1 if row[15] == "Returning_Visitor" else 0,
                1 if row[16] == "TRUE" else 0
            ])
            labels.append(1 if row[17] == "TRUE" else 0)

    return evidence, labels

def train_model(evidence, labels):
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model

def evaluate(labels, predictions):
    true_positive = sum(1 for true, pred in zip(labels, predictions) if true == 1 and pred == 1)
    false_negative = sum(1 for true, pred in zip(labels, predictions) if true == 1 and pred == 0)
    true_negative = sum(1 for true, pred in zip(labels, predictions) if true == 0 and pred == 0)
    false_positive = sum(1 for true, pred in zip(labels, predictions) if true == 0 and pred == 1)

    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)

    return sensitivity, specificity

def main():
    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data("shopping.csv")
    X_train, X_test, y_train, y_test = train_test_split(evidence, labels, test_size=0.4)

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluate model performance
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

if __name__ == "__main__":
    main()

