import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# read data
df = pd.read_table("../data/SMSSpamCollection.csv",
                   sep='\t',
                   header=None,
                   names=['label', 'sms_message']
                   )

# convert labels to numeric 0,1
df['label'] = df.label.map({'ham': 0, 'spam': 1})

# split into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(df['sms_message'],
                                                    df['label'],
                                                    random_state=1
                                                    )
# check sizes
print('rows total    {}'.format(df.shape[0]))
print('rows training {}'.format(X_train.shape[0]))
print('rows test     {}'.format(X_test.shape[0]))

# make bag of words
count_vector = CountVectorizer()

# training data
training_data = count_vector.fit_transform(X_train)

# test data
testing_data = count_vector.transform(X_test)

# naive bayes
nb = MultinomialNB()
nb.fit(training_data, y_train)
print(nb)

# predictions
predictions = nb.predict(testing_data)
print(predictions)

# Accuracy - correct prediction
# Precision - predicted result == actual result (true pos / all positives)
# Recall - (sensitivity) true classified as true (true pos / (true pos + false neg))

print('accuracy ', accuracy_score(y_test, predictions))
print('precision', precision_score(y_test, predictions))
print('recall   ', recall_score(y_test, predictions))
print('f1       ', f1_score(y_test, predictions))
