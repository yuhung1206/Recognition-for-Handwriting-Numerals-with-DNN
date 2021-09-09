# Implement Back-Propagation only unsing "numpy" package.
# Without any automatic dierentiation tools


# import module
import numpy as np

"""----------All functions----------"""
def softmax(z):
    # calculate softmax of matrix z
    exp_z = np.exp(z)
    tempsum = np.sum(exp_z, axis=0)
    softmax_z = exp_z / tempsum[:, np.newaxis].T.repeat(10,axis=0)
    return softmax_z

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)

"""----------functions end----------"""


"""-----Read .NPZ file-----"""
TrainData = np.load('train.npz')  # Training set
TestData = np.load('test.npz')  # Testing set
epochs = 300
batchsize = 100
iteration = int(12000/batchsize)
HiddenLayerNum = 2  # previous = 3
ImageSize = 28*28
LayerNum = HiddenLayerNum + 2 # include input, output
NeuralNum = np.array([ImageSize, 320, 160, 10])  # Neural's num of each layer => previous:[ImageSize, 512, 256, 128, 10]
TrainSize = TrainData.f.label.shape[0]
TestSize = TestData.f.label.shape[0]
learningRate = 0.001
Lambda = 0.1

"""-----define "weight matrix", "a vector", "bias vector" for each layer-----"""
# for train
W_matrix = [] # index 0~3 => h1~h4
b_vector = []
a_vector = []
h_vector = []
h_vector.append(np.zeros([ImageSize, batchsize]))
# for derivation
delta_matrix = []
de_W_matrix = []
de_b_vector = []
# for test
a_vector_t = []
h_vector_t = []
h_vector_t.append(np.zeros([ImageSize, TestSize]))

for i in range(len(NeuralNum)-1): # 5 layers only have 4 weight matrix
     # np.random.randn(2, 4)
     W_matrix.append(np.zeros([NeuralNum[i+1],NeuralNum[i]])) # (this layer's neural, last layer's neural)
     b_vector.append(np.zeros([NeuralNum[i+1], 1]))
     a_vector.append(np.zeros([NeuralNum[i+1], batchsize]))
     h_vector.append(np.zeros([NeuralNum[i+1], batchsize]))

     # for backpropagation
     delta_matrix.append(np.zeros([batchsize, NeuralNum[i+1]]))
     de_W_matrix.append(np.zeros([NeuralNum[i],NeuralNum[i+1]]))
     de_b_vector.append(np.zeros([NeuralNum[i+1],1]))

     # for Test
     a_vector_t.append(np.zeros([NeuralNum[i+1], TestSize]))
     h_vector_t.append(np.zeros([NeuralNum[i+1], TestSize]))

"""-----Transfer label[12000X1] to 2D label[12000X10]-----"""
# for train
t_label = np.zeros([TrainSize, 10])
t_digits = list(map(int, TrainData.f.label))
t_label[np.arange(TrainSize),t_digits] = 1

# for train
t_label_t = np.zeros([TestSize, 10])
t_digits_t = list(map(int, TestData.f.label))
t_label_t[np.arange(TestSize),t_digits_t] = 1


Error = np.zeros(epochs) # record each epoch's loss
Train_Error = np.zeros(epochs)
Test_Error = np.zeros(epochs)
#latent_var = np.zeros([epochs, TrainSize, 2]) # to store the latent var in each epoch
for ep in range(epochs):

    """-----forward propagation-----"""
    y_label = np.zeros([TrainSize, 10]) # to store the predict result in each epoch
    for itr in range(iteration):
        y_label_batch = np.zeros([batchsize, 10])  # [100 X 10]
        t_label_batch = t_label[itr*batchsize:(itr + 1)*batchsize]  # [100 X 10]
        # the 0~100 figure to be input [100 X 784]
        InputArray = (TrainData.f.image[itr*batchsize:(itr+1)*batchsize]).reshape(batchsize, NeuralNum[0], )/255.0  # h0 = x(picture i)
        h_vector[0] = InputArray.T  # transpose to [784 X 100]
        """-----Hidden layer computation-----"""
        for k in range(HiddenLayerNum): # index: 0~2
            # hidden: 3 layers -> Activation function: Sigmoid
            a_vector[k] = b_vector[k].repeat(batchsize,axis=1) + np.dot(W_matrix[k], h_vector[k]) # [512 X 100] = [512 X 784][784 X 100]
            """-----use Sigmoid Activation function-----"""
            h_vector[k + 1] = sigmoid(a_vector[k])

        """-----Output layer computation-----"""
        # output layer -> Activation function: softmax
        a_vector[-1] = b_vector[-1].repeat(batchsize,axis=1) + np.dot(W_matrix[-1], h_vector[-2])  # k = 3
        """-----use softmax Activation function-----"""
        h_vector[-1] = softmax(a_vector[-1])  # [10 X 100]
        # predict_class = np.argmax(h_vector[3+1])  # The class be predicted
        """-----get predicted label "y_label"-----"""
        y_label_batch = h_vector[-1].T  # [100 X 10]
        y_label[itr*batchsize:(itr + 1)*batchsize,:] = y_label_batch


        """-----backward propagation to get new "b", "w"-----"""
        # delata_4 = Y - T for last layer
        delta_matrix[-1] = y_label_batch - t_label_batch # size : [5768 X 10]
        de_W_matrix[-1] = np.dot(h_vector[-2], delta_matrix[-1])  # size : [128 X 10]
        W_matrix[-1] = W_matrix[-1] - learningRate * de_W_matrix[-1].T - (learningRate*Lambda)*W_matrix[-1]  # "Regulization term"
        de_b_vector[-1] = np.sum(delta_matrix[-1], axis=0, keepdims=True)  # size : [1 X 10]
        b_vector[-1] = b_vector[-1] - learningRate * de_b_vector[-1].T
        # delta_4 = y_label - t_label # size : [5768 X 10]
        # de_W_matrix_4 = np.dot(h_vector[3], delta_4) # [128 X 10]

        # hidden layer 3~1
        for t in range(HiddenLayerNum-1, -1, -1): # i = 2 ~ 0
            # size : [n X 128]
            delta_matrix[t] = np.dot(delta_matrix[t+1], W_matrix[t+1])*(h_vector[t+1]*(1-h_vector[t+1])).T
            de_W_matrix[t] = np.dot(h_vector[t], delta_matrix[t])   # size : [256 X 128]
            W_matrix[t] = W_matrix[t] - learningRate*de_W_matrix[t].T - learningRate*Lambda*W_matrix[t]
            de_b_vector[t] = np.sum(delta_matrix[t], axis=0, keepdims=True)  # size : [1 X 10]
            b_vector[t] = b_vector[t] - learningRate * de_b_vector[t].T

    """-----calculate learning Error of this epoch-----"""
    Error[ep] = -1 * sum(sum(t_label * np.log(y_label)))/TrainSize 

    """-----calculate training Error of this epoch-----"""
    Train_result_label = np.zeros([TrainSize, 10])
    Train_result_index = np.argmax(y_label, axis=1)  # get max in each row
    Train_result_label[np.arange(TrainSize), Train_result_index] = 1
    Train_Error[ep] = 1 - np.sum(Train_result_label*t_label)/TrainSize


    """-----Testing for this epoch-----"""
    # the 0~5767 figure to be input [5768 X 784]
    Input_Test = (TestData.f.image[0 : TestSize]).reshape(TestSize, NeuralNum[0], ) / 255.0  # h0 = x(picture i)
    h_vector_t[0] = Input_Test.T  # transpose to [784 X 5768]
    Output_Test = np.zeros([TestSize, 10])  # [5768 X 10]
    for k in range(HiddenLayerNum):  # index: 0~2
        # hidden: 3 layers -> Activation function: Sigmoid
        a_vector_t[k] = b_vector[k].repeat(TestSize,axis=1) + np.dot(W_matrix[k], h_vector_t[k])  # [512 X 5768] = [512 X 784][784 X 5768]
        """-----use Sigmoid Activation function-----"""
        h_vector_t[k + 1] = sigmoid(a_vector_t[k])

    """-----Output layer computation-----"""
    # output layer -> Activation function: softmax
    a_vector_t[-1] = np.dot(W_matrix[-1], h_vector_t[-2])  # k = 3
    """-----use softmax Activation function-----"""
    h_vector_t[-1] = softmax(a_vector_t[-1])  # [10 X 5768]
    # predict_class = np.argmax(h_vector[3+1])  # The class be predicted
    """-----get predicted label "y_label"-----"""
    Output_Test = h_vector_t[-1].T  # [5768 X 10]
    """-----calculate testing Error rate of this epoch-----"""
    Test_result_label = np.zeros([TestSize, 10])  # [5768 X 10]
    Test_result_index = np.argmax(Output_Test, axis=1)  # get max in each row
    Test_result_label[np.arange(TestSize), Test_result_index] = 1
    Test_Error[ep] = 1 - np.sum(Test_result_label*t_label_t)/TestSize


    print("---------------------------------------")
    print("Epoch:" + str(ep))
    print("    Train_Error_rate:" + str(Train_Error[ep]))
    print("    Test_Error_rate:" + str(Test_Error[ep]))
    print("    Loss_Error:" + str(Error[ep]))


np.save("Error.npy", Error)
np.save("Train_Error.npy", Train_Error)
np.save("Test_Error.npy", Test_Error)

"""-----calculate confusion matrix for last epoch-----"""
CfMatrix = np.zeros([10,10])
#  CorrctPred = np.sum(Test_result_label*t_label_t, axis=0) # the correction prediction for each digit
#  CfMatrix[np.arange(10),np.arange(10)] = CorrctPred[np.arange(10)]
for Num in range(10):
    num_list = ([i for i in range(TestSize) if t_digits_t[i] == Num])
    CfMatrix[Num,:] = np.sum(Test_result_label[num_list,:], axis=0)

np.save("CfMatrix.npy", CfMatrix)



