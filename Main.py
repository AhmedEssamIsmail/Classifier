import numpy as np
import os
import pickle
from PIL import Image
import cv2
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from K_means import run_k_means

names = {1: "Cat", 2: "Laptop", 3: "Apple", 4: "Car", 5: "Helicopter"}


def sigmoid(inp):
    denominator = 1 + np.exp(-1 * inp)
    return 1 / denominator


def tanh(inp):
    exp_term = np.exp(-2 * inp)
    return (1 - exp_term) / (1 + exp_term)


# hidden layer is an array that contains the number of neurons in each hidden layer
# example initiate (pca_components,[3,2,3],5,"sigmoid")'''
def initiate(input_layer, hidden_layer, output_layer, activation):
    number_hidden_layers = len(hidden_layer)
    weights = {}
    bias = {}
    previous = input_layer
    for i in range(number_hidden_layers):
        current = int(hidden_layer[i])
        if activation == "tanh" or activation == "sigmoid":  # xavier initialization
            temp = np.random.rand(current, previous) * np.sqrt(1 / previous)
        else:  # relu
            temp = np.random.rand(current, previous) * np.sqrt(2 / previous)
        weights[i] = temp
        bias_temp = np.ones(current)
        # bias_temp = bias_temp.reshape(1, -1)
        bias[i] = bias_temp
        previous = current
    # if activation=="relu"or activation == "tanh" or activation == "sigmoid":  # xavier initialization -- all last layer should be xavier
    weights[number_hidden_layers] = np.random.rand(output_layer, previous) * np.sqrt(1 / previous)
    bias_temp = np.ones(output_layer)
    # bias_temp = bias_temp.reshape(1, -1)
    bias[number_hidden_layers] = bias_temp

    return weights, bias


def forward_pass(features_example, weights, bias, bias_flag, activation):
    num_layers = len(weights)
    z = {}
    current_feed = features_example
    for k in range(num_layers):
        for i in range(weights[k].shape[0]):
            v_net = 0
            for j in range(weights[k].shape[1]):
                v_net += weights[k][i, j] * current_feed[j]
            if bias_flag == 1:
                v_net += bias[k][i]
            if activation == "sigmoid":
                a = sigmoid(v_net)
            elif activation == "tanh":
                a = tanh(v_net)
            # elif activation == "relu": relu is deleted
            #    a = max(v_net,0)
            if i == 0:
                temp = np.array(a)
            else:
                temp = np.append(temp, a)
        # temp = temp.reshape(1,-1)
        current_feed = temp
        last_layer = current_feed
        z[k] = temp
    output = np.zeros(len(current_feed))
    output[np.argmax(current_feed)] = 1
    return last_layer, output, z


def backward_pass(feature_example, desired, weights, bias_flag, bias, activation, z, learning_rate, a,
                  b):  # remaining bias/activation
    num_hidden_layers = len(z) - 1
    local_gradient = {}
    current_desired = np.zeros(len(z[num_hidden_layers]))
    current_desired[desired - 1] = 1
    desired = current_desired
    for i in range(len(z[num_hidden_layers])):  # error for the output layer
        if activation == "sigmoid":
            local_error = a * (desired[i] - z[num_hidden_layers][i]) * z[num_hidden_layers][i] * (
                1 - z[num_hidden_layers][i])
        elif activation == "tanh":
            local_error = (float(b) / a) * (desired[i] - z[num_hidden_layers][i]) * (a - z[num_hidden_layers][i]) * (
                a + z[num_hidden_layers][i])
        if i == 0:
            temp = np.array(local_error)
        else:
            temp = np.append(temp, local_error)
    # temp = temp.reshape(1, -1)
    local_gradient[num_hidden_layers] = temp

    for i in range(num_hidden_layers - 1, -1, -1):  # calculating local gradient
        for j in range(len(z[i])):
            error_sum = 0
            for k in range(len(z[i + 1])):
                error_sum += local_gradient[i + 1][k] * weights[i + 1][k, j]
            if activation == "sigmoid":
                local_error = a * z[i][j] * (1 - z[i][j]) * error_sum
            elif activation == "tanh":
                local_error = (float(b) / a) * (a - z[i][j]) * (a + z[i][j]) * error_sum
            if j == 0:
                temp = np.array(local_error)
            else:
                temp = np.append(temp, local_error)
        # temp = temp.reshape(1,-1)
        local_gradient[i] = temp
    for layer in range(len(weights)):
        z_index = layer - 1
        g_index = layer
        if z_index == -1:
            current_feed = feature_example
        else:
            current_feed = z[z_index]
        gradient = local_gradient[g_index]
        for i in range(weights[layer].shape[0]):
            for j in range(weights[layer].shape[1]):
                weights[layer][i, j] = weights[layer][i, j] + (learning_rate * gradient[i] * current_feed[j])
            if bias_flag == 1:
                bias[layer][i] = bias[layer][i] + (learning_rate * gradient[i] * 1)
    return weights, bias


def mse(samples, desired, weights, bias_flag, bias, activation):
    error = 0
    output_len = weights[len(weights) - 1].shape[0]
    for i in range(samples.shape[0]):
        current_desired = np.zeros(output_len)
        current_desired[desired[i] - 1] = 1
        output, _, _ = forward_pass(samples[i], weights, bias_flag, bias, activation)
        current_error = 0
        for j in range(output_len):
            current_error += (current_desired[j] - output[j]) * (current_desired[j] - output[j])
        current_error /= output_len
        error += current_error
    error /= samples.shape[0]
    return error


def data_pre_processing(data_set, labels):
    labels = np.transpose(labels)

    scaler = StandardScaler()
    scaler.fit(data_set)
    data_set = scaler.transform(data_set)

    pca = PCA()
    pca.fit(data_set)
    data_set = pca.transform(data_set)

    training_data, testing_data, desired_train, desired_test = train_test_split(data_set, labels, test_size=0.2,
                                                                                random_state=0)
    training_data, validation_data, desired_train, desired_validation = train_test_split(training_data, desired_train,
                                                                                         test_size=0.25, random_state=0)

    labels = np.transpose(labels)
    return data_set, testing_data, validation_data, labels, desired_test, desired_validation, pca, scaler


def reading_data(dataset_path):
    # dataset_path = 'C:/Users/Ahmed/Desktop/Data set/Training'
    samples_path = ''
    classes = len(os.listdir(dataset_path))
    for i in range(classes):
        samples_path = dataset_path + '/' + str(i + 1)
        imgs = os.listdir(samples_path)

        for j in range(len(imgs)):
            current_image = np.asarray(
                (Image.open(samples_path + '/' + imgs[j]).convert('L')).resize((50, 50), Image.LANCZOS))
            scurrent_image = current_image.flatten()
            current_image = current_image.reshape(1, -1)
            if i == 0 and j == 0:
                temp = current_image
                desired_temp = np.array(i + 1)
            else:
                temp = np.append(temp, current_image, axis=0)
                desired_temp = np.append(desired_temp, i + 1)
    data_samples = temp
    desired = desired_temp
    prm = np.random.permutation(len(desired))
    desired = desired[prm]
    data_samples = data_samples[prm, :]
    # desired = desired.reshape(1,-1)
    return data_samples, desired, classes


def get_objects(img):
    edges = cv2.Canny(img, 50, 200)
    _, res, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return res


def test_image(path, seg_path, bias_flag, activation):
    original_image = cv2.imread(path)
    grayimage = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    segmented_image = cv2.imread(seg_path)
    objects = get_objects(segmented_image)
    weights, bias, pca, scaler = load_model()
    for obj in objects:
        rect = cv2.boundingRect(obj)
        x, y, w, h = rect

        cropped = grayimage[y:y + h, x:x + w]
        resized = cv2.resize(cropped, (50, 50))
        resized = np.asarray(resized)
        resized = resized.flatten()
        resized = resized.reshape(1, -1)
        resized = scaler.transform(resized)
        resized = pca.transform(resized)
        resized = resized.reshape(-1, 1)

        _, output, _ = forward_pass(resized, weights, bias, bias_flag, activation)
        predicted = np.argmax(output) + 1
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 0), 2)
        cv2.putText(original_image, names[int(predicted)], (x + w - int(1 / 2 * w), y + h - int(1 / 2 * h)), 0, 1, (0, 0, 0))

    cv2.imshow('Result', original_image)
    cv2.waitKey()
    return


def testing(dataset, desired, bias_flag, classes, activation):
    weights, bias, pca, scaler = load_model()
    accuracy = 0
    con_matrix = np.zeros([classes, classes])
    for i in range(dataset.shape[0]):
        _, output, _ = forward_pass(dataset[i], weights, bias, bias_flag, activation)
        predicted = np.argmax(output) + 1
        con_matrix[desired[i] - 1, predicted - 1] += 1
        if predicted == desired[i]:
            accuracy += 1
    accuracy = (accuracy / dataset.shape[0]) * 100
    print(con_matrix, accuracy)
    return


def gui_intermadiate(epochs, hiddens, bias_flag, activation, learning_rate, mse_threshold, stopping_criteria, a, b):
    m = 0
    i = 0
    while i < len(hiddens):
        s = ""
        while i < len(hiddens) and hiddens[i] != ',':
            s += hiddens[i]
            i += 1
        if m == 0:
            temp = np.zeros(1)
            temp[0] = int(s)
            m += 1
        else:
            temp = np.append(temp, int(s))
        i += 1
    hiddens = temp
    runMLP(epochs, hiddens, bias_flag, activation, learning_rate, mse_threshold, stopping_criteria, a, b)
    return


def save_model(weights, bias, pca, scaler):
    with open('weights.pickle', 'wb')as handle:
        pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('bias.pickle', 'wb')as handle:
        pickle.dump(bias, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('pca.pickle', 'wb')as handle:
        pickle.dump(pca, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('scaler.pickle', 'wb')as handle:
        pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return


def load_model():
    with open('weights.pickle', 'rb')as handle:
        weights = pickle.load(handle)
    with open('bias.pickle', 'rb')as handle:
        bias = pickle.load(handle)
    with open('pca.pickle', 'rb') as handle:
        pca = pickle.load(handle)
    with open('scaler.pickle', 'rb') as handle:
        scaler = pickle.load(handle)
    return weights, bias, pca, scaler


def runMLP(epochs, hiddens, bias_flag, activation, learning_rate, mse_threshold, stopping_criteria, a, b):
    data_set, desired, classes = reading_data('C:/Users/Ahmed/Desktop/Pattern/Dataset/Training')
    training_data, testing_data, validation_data, desired_train, desired_test, desired_validation, pca, scaler = data_pre_processing(
        data_set, desired)

    weights, bias = initiate(training_data.shape[1], hiddens, classes, activation)
    prev_error = 9999
    for i in range(epochs):
        for j in range(training_data.shape[0]):
            last_layer, output, z = forward_pass(training_data[j], weights, bias, bias_flag, activation)
            weights, bias = backward_pass(training_data[j], desired_train[j], weights, bias_flag, bias, activation, z,
                                          learning_rate, a, b)
        if stopping_criteria == 1 and i % 50 == 0:  # stopping criteria 1 -> mse threshold
            ms_error = mse(validation_data, desired_validation, weights, bias_flag, bias, activation)
            print(ms_error)
            if mse_threshold >= ms_error:
                save_model(weights, bias, pca, scaler)
                testing(testing_data, desired_test, bias_flag, classes, activation)
                return
        if stopping_criteria == 2 and i % 50 == 0:
            ms_error = mse(validation_data, desired_validation, weights, bias_flag, bias, activation)
            print(ms_error)
            if abs(ms_error - prev_error) <= 0.0001:
                save_model(weights, bias, pca, scaler)
                testing(testing_data, desired_test, bias_flag, classes, activation)
                return
            prev_error = ms_error

    save_model(weights, bias, pca, scaler)  # stopping criteria no_epochs
    testing(testing_data, desired_test, bias_flag, classes, activation)
    return


def ecl(point1, point2):
    dist = 0
    for i in range(len(point1)):
        dist += math.pow(point1[i] - point2[i], 2)
    dist = math.sqrt(dist)
    return dist


def calc_q(centroid, sample, sigma):
    dist = ecl(centroid, sample)
    dist = math.pow(dist, 2)
    q = math.exp(-dist / (2 * sigma))
    return q


def rbf(training_data, desired, number_of_hiddens, mse_threshold, learning_rate, epochs, classes):
    centroids = run_k_means(training_data, number_of_hiddens)
    sigma = 0
    for i in range(centroids.shape[0]):
        for j in range(centroids.shape[0]):
            if i == j:
                continue
            dist = ecl(centroids[i], centroids[j])
            if dist > sigma:
                sigma = dist
    sigma = sigma / math.sqrt(2 * number_of_hiddens)
    Qs = np.zeros([training_data.shape[0], centroids.shape[0]])
    for i in range(centroids.shape[0]):
        for j in range(training_data.shape[0]):
            Qs[j, i] = calc_q(centroids[i], training_data[j], sigma)
    weights = np.random.rand(classes, number_of_hiddens)
    temp_desired = np.zeros([classes, len(desired)])
    for i in range(len(desired)):
        temp_desired[desired[i] - 1, i] = 1
    desired = temp_desired
    for i in range(epochs):
        QT = np.transpose(Qs)
        net = np.dot(weights, QT)
        e = desired - net
        delta = learning_rate * (np.dot(e, Qs))
        weights = weights + delta
        mse = np.power(e, 2)
        mse = np.sum(mse)
        mse /= (desired.shape[0] * desired.shape[1])
        print(mse)
        if mse <= mse_threshold:
            break
    with open('Rbfweights.pickle', 'wb')as handle:
        pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('Rbfcentroids.pickle', 'wb')as handle:
        pickle.dump(centroids, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('sigma.pickle', 'wb')as handle:
        pickle.dump(sigma, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return


def RBFrun(number_of_hiddens, mse_threshold, learning_rate, epochs):
    data_set, desired, classes = reading_data('C:/Users/Ahmed/Desktop/Pattern/Dataset/Training')
    training_data, testing_data, validation_data, desired_train, desired_test, desired_validation, pca, scaler = data_pre_processing(
        data_set, desired)
    rbf(training_data, desired, number_of_hiddens, mse_threshold, learning_rate, epochs, classes)
    return


def test_image_rbf(path, seg_path):
    original_image = cv2.imread(path)
    grayimage = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    segmented_image = cv2.imread(seg_path)
    objects = get_objects(segmented_image)

    with open('Rbfweights.pickle', 'rb')as handle:
        weights = pickle.load(handle)
    with open('Rbfcentroids.pickle', 'rb')as handle:
        centroids = pickle.load(handle)
    with open('pca.pickle', 'rb') as handle:
        pca = pickle.load(handle)
    with open('sigma.pickle', 'rb') as handle:
        sigma = pickle.load(handle)
    with open('scaler.pickle', 'rb') as handle:
        scaler = pickle.load(handle)

    for obj in objects:
        rect = cv2.boundingRect(obj)
        x, y, w, h = rect

        cropped = grayimage[y:y + h, x:x + w]
        resized = cv2.resize(cropped, (50, 50))
        resized = np.asarray(resized)
        resized = resized.flatten()
        resized = resized.reshape(1, -1)
        resized = scaler.transform(resized)
        resized = pca.transform(resized)
        resized = resized.reshape(-1, 1)

        Qs = np.zeros([resized.shape[1], centroids.shape[0]])
        for i in range(centroids.shape[0]):
            Qs[0, i] = calc_q(centroids[i, :], resized, sigma)

        QT = np.transpose(Qs)
        net = np.dot(weights, QT)
        predicted = np.argmax(net) + 1

        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 0), 2)
        cv2.putText(original_image, names[int(predicted)], (x + w - int(1 / 2 * w), y + h - int(1 / 2 * h)), 0, 1, (0, 0, 0))

    cv2.imshow('Result', original_image)
    cv2.waitKey()
    return
