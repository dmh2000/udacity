import string
from collections import Counter
lc = df.sms_message.str.lower()
np = lc.str.translate(str.maketrans('', '', string.punctuation))
pd = np.str.split(' ');

frequency_list = []
for i in pd:
    freq_count = Counter(i)
    frequency_list.append(freq_count)

print(frequency_list)

# read data
df = pd.read_table("../data/SMSSpamCollection.csv",
                   sep='\t',
                   header=None,
                   names=['label', 'sms_message']
                   )


# vectorize the sms_messages
count_vector = CountVectorizer()
count_vector.fit(df['sms_message'])

# create a matrix of the sms_messages and counts
feature_array = count_vector.transform(df['sms_message']).toarray()
feature_names = count_vector.get_feature_names()
# create new data frame with feature labels for columns
feature_matrix = pd.DataFrame(feature_array,
                              columns=feature_names)