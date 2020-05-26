import jieba
import pandas as pd

jieba.load_userdict('dict.txt')

def read_csv_data(file_name):
    df = pd.read_csv(file_name, encoding='big5')
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]

    data = []
    for i in range(len(df)):
        data.append([x[i], y[i]])
    return data


def calculate_statistics(text_in_label):
    data = []
    data_dict ={'0': [], '1': []}
    document_words = []
    num_positive_sample = 0
    num_negative_sample = 0
    num_data = 0

    for row in text_in_label:
        num_data = num_data + 1
         
        if row[1] == 1:

            split = jieba.cut(row[0], cut_all=False)
            num_positive_sample = num_positive_sample + 1

            for i in split:
                data.append(i)
                document_words.append(i)
                data_dict['1'].append(i)

        if row[1] == 0:

            split = jieba.cut(row[0], cut_all=False)            
            num_negative_sample = num_negative_sample + 1

            for i in split:
                data.append(i)
                document_words.append(i)
                data_dict['0'].append(i)
    document_words = len(set(document_words))
    return data, data_dict, document_words, num_positive_sample, num_negative_sample, num_data

def count_class_freq(dictionary):
    positive_sample_dict = {}
    negative_sample_dict = {}

    for word in dictionary['1']:
        if word not in positive_sample_dict:
            positive_sample_dict[word] = 0

    for word in dictionary['0']:
        if word not in negative_sample_dict:
            negative_sample_dict[word] = 0

    return positive_sample_dict, negative_sample_dict

def calculate_likelihood(dataset, test_data, data_dictionary, positive_sample_dict, negative_sample_dict, vocabulary, alpha = 1):
    prob_positive_words = {}
    prob_negative_words = {}
    positive_words = {}
    negative_words = {}
    num_of_word_in_positive = len(data_dictionary['1'])
    num_of_word_in_negative = len(data_dictionary['0'])

#    print(data_dictionary['1'])
    for word in data_dictionary['1']:
        if word not in data_dictionary['1']:
            positive_words[word] = 0
            
        else:
            if word in positive_words:
                positive_words[word] = positive_words[word] + 1
            else:
                positive_words[word] = 1

    for word in data_dictionary['0']:
        if word not in data_dictionary['0']:
            negative_words[word] = 0
        else:
            if word in negative_words:
                negative_words[word] = negative_words[word] + 1
            else:
                negative_words[word] = 1

    for word in dataset:

        if word not in data_dictionary['1']:
            positive_words[word] = 0

        if word not in data_dictionary['0']:
            negative_words[word] = 0

    for word in test_data:

        if word not in data_dictionary['1']:
            positive_words[word] = 0

        if word not in data_dictionary['0']:
            negative_words[word] = 0

    for word in positive_words:
        prob_positive_words[word] = (positive_words[word] + alpha)/(num_of_word_in_positive + vocabulary)
        #print('word: {} freq: {}'.format(word, positive_words[word]))

    for word in negative_words:
        prob_negative_words[word] = (negative_words[word] + alpha)/(num_of_word_in_negative + vocabulary)
        #print('word: {} freq: {}'.format(word, negative_words[word]))

    return prob_positive_words, prob_negative_words



def predict(test_data, prob_positive, prob_negative, num_positive_sample, num_negative_sample, num_data):
    probability_positive = []
    probability_negative = []
    prediction = []
    positive_meter = 1
    negative_meter = 1

    for word in test_data:
        probability_positive.append(prob_positive[word])
        probability_negative.append(prob_negative[word])

    for value in probability_positive:
        positive_meter = positive_meter * value
    positive_meter = positive_meter * (num_positive_sample/num_data)

    for value in probability_negative:
        negative_meter = negative_meter * value
    negative_meter = negative_meter * (num_negative_sample/num_data)


    if positive_meter > negative_meter:
        prediction.append(1)
    if negative_meter > positive_meter:
        prediction.append(0)
    if positive_meter == negative_meter:
        prediction.append(None)

    prediction_val = max(positive_meter, negative_meter)

    return prediction, prediction_val, negative_meter, positive_meter




file_name = 'dataset.csv'

#讀取CSV檔案

data = read_csv_data(file_name)
#print(data)
#統計總樣本數量、正負樣本數量
dataset, data_dictionary, document_words, num_positive_sample, num_negative_sample, num_data = calculate_statistics(data)
print('Number of data: {}\npositive sample is: {}\nnegative sample is: {}'.format(num_data, num_positive_sample, num_negative_sample))


#將正負樣本出現過的字放到不同字典裡面
positive_sample_dict, negative_sample_dict = count_class_freq(data_dictionary)

#輸入測試樣本
comment = "今天債券市場大漲"
comment = jieba.cut(comment, cut_all=False)

test_data = [] 
for i in comment:
    test_data.append(i)


#計算每個字之likelihood
prob_positive, prob_negative = calculate_likelihood(dataset, test_data, data_dictionary, positive_sample_dict, negative_sample_dict, document_words)
print(prob_negative)

#預測測試樣本
pred, val, negative_meter, positive_meter = predict(test_data, prob_positive, prob_negative, num_positive_sample, num_negative_sample, num_data)

print('prediction: {}, probability is {}'.format(pred, val))
print('positive meter: {}\nnegative meter: {}'.format(positive_meter, negative_meter))
