import loglinear as ll
import random
import utils

STUDENT={'name': 'ZAIDMAN IGAL',
         'ID': '311758866'}

def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.
    return features

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        if ll.predict(features, params) == label:
            good = good + 1
        else:
            bad = bad + 1

    return good / (good + bad)

def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in range(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features) # convert features to a vector.
            y = label                  # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x,y,params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.
            params[0] = params[0] - grads[0] * learning_rate
            params[1] = params[1] - grads[1] * learning_rate

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print (I, train_loss, train_accuracy, dev_accuracy)
    return params

if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    i = j = 0
    all_bigrams = {}
    all_langs = {}
    lang_to_id = {}
    num_iterations = 6
    learning_rate = 0.01

# fill sets of all the languages and bigrams from the train file
    for [lang, bigrams] in utils.TRAIN:
        if lang not in all_langs:
            all_langs[lang] = j
            lang_to_id[j] = lang
            j += 1
        for bigram in bigrams:
            if bigram not in all_bigrams:
                all_bigrams[bigram] = i
                i += 1
                
# extract data from file as language id and features array    
    def fileData(fData):
        data = []
        for [lang, bigrams] in fData:
            features = np.zeros(len(all_bigrams))
            for bigram in bigrams:
                if bigram in all_bigrams:
                    features[all_bigrams[bigram]] += 1
            language = all_langs[lang] if lang in all_langs else -1
            data.append([language, features])
        return data
    
 # process the training and dev data and print accuracy
    params = ll.create_classifier(len(all_bigrams), len(all_langs))
    trained_params = train_classifier(fileData(utils.TRAIN), fileData(utils.DEV), num_iterations, learning_rate, params)

# run prediction on the test data
    predict = []   
    for [label, data] in fileData(utils.TEST):
        predict.append(lang_to_id[ll.predict(data, trained_params)])


# write the prediction to a file
    predict_file = open('test.pred', 'w')
    predict_file.writelines(["%s\n" % item  for item in predict])
    predict_file.close()
