#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad
from sklearn.preprocessing import Normalizer


def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    ### YOUR CODE HERE
    # x = x/ (np.sum(x**2, axis=0) ** .5)
    norm = Normalizer()
    x = norm.fit_transform(x)
    ### END YOUR CODE

    return x


def test_normalize_rows():
    print("Testing normalizeRows...")
    x = normalizeRows(np.array([[3.0, 4.0], [1, 2]]))
    print(x)
    ans = np.array([[0.6, 0.8], [0.4472136, 0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print("")


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    ### YOUR CODE HERE
    num, dim = outputVectors.shape
    score = np.dot(outputVectors, predicted)  # (N, )
    prob = softmax(score)  # (N, )
    cost = - np.log(prob[target])  # 1,
    
    # score (N x 1)= outputVectors (N x D) x predicted (D x 1)
    # doutputVectors = dscore (N x 1) x predicted^T (1 x D)
    dscore = prob - np.eye(num)[target]  # (N, )
    grad = np.outer(dscore, predicted)
    gradPred = np.dot(outputVectors.T, dscore)

    ### END YOUR CODE

    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    ### YOUR CODE HERE
    # u = outputVectors[indices, :]  # (K+1, D)
    # mask = - np.ones(len(indices))
    # mask[0] = 1 # 1st word is positive sample, others are negative samples
    # score = mask * np.dot(u, predicted)
    # prob = sigmoid(score)  # (K+1, )
    # cost = - np.sum(np.log(prob))

    # # back propagation
    # dprob = - 1.0 / prob
    # dscore = mask * sigmoid_grad(dprob)
    # du = np.outer(dscore, predicted)  # (K+1, D)
    # dv = np.dot(u.T, dscore)
    # grad = np.zeros_like(outputVectors)
    # for i, index in enumerate(indices):
    #     grad[index] += du[i]
    # gradPred = dv

    predicted_orig_shape = predicted.shape
    outputVectors_orig_shape = outputVectors.shape


    # STEP 0: first let's make the notations consitent with the course and written assignments
    # let D=dimension of hidden layer, |V|=number of tokens in outputvectors, N=number of negative words
    V_c = predicted.reshape(-1, 1)  # the input vector of predicted word --> D x 1
    U = outputVectors.reshape(-1, V_c.shape[0])  # ALL the output vectors --> |V| x D
    U_o = U[target].reshape(1, -1)  # the output vector of predicted word --> 1 x D
    U_negs = U[indices[1:]] # --> N x D
    # -----

    # STEP 1: since the sigmoids of target & all negative samples is needed many times we'll compute and save them
    # Let get the scores first: positive for the target word and negative for the negative word
    targetword_and_negwords_scores = -1 * U[indices, :].dot(V_c) #--> N+1 x 1
    targetword_and_negwords_scores[0] = -1 * targetword_and_negwords_scores[0]
    targetword_and_negwords_sigmoids = sigmoid(targetword_and_negwords_scores) #--> N+1 x 1
    del targetword_and_negwords_scores
    target_sigmoid = targetword_and_negwords_sigmoids[0] #--> 1 x 1, scalar
    neg_sigmoids = targetword_and_negwords_sigmoids[1:] #--> N x 1
    # -----

    # STEP 2: cost = -log(target_word_sigmoid) - sum( neg_words_sigmoids)
    cost = -1.*np.sum(np.log(targetword_and_negwords_sigmoids))
    cost = np.asscalar(cost)
    # -----

    # STEP 3: gradPed = grad_Cost__wrt__V_c
    gradPred = (target_sigmoid -1.) * U_o + (1. - neg_sigmoids).T.dot(U_negs) #--> 1 x D
    gradPred = gradPred.reshape(predicted_orig_shape)
    # -----

    # STEP 4: grad = grad_Cost_wrt_negs_and_target_words_outputvectors, gradient of not(target or negs) are zero
    grad = np.zeros(U.shape) #--> |V| x D
    grad_target_and_negs = (1. - targetword_and_negwords_sigmoids).dot(V_c.T) #--> N+1 x D
    # we negate the grad for the target word as  we found in the formula
    grad_target_and_negs[0] *= -1
    for idx, global_idx in enumerate(indices):

        
        grad[global_idx, :] += grad_target_and_negs[idx, :]
    grad = grad.reshape(outputVectors_orig_shape)
    ### END YOUR CODE

    return cost, gradPred, grad


def skipgram(currentWord,
             C,
             contextWords,
             tokens,
             inputVectors,
             outputVectors,
             dataset,
             word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    current_word_vec_idx = tokens[currentWord]
    current_word_vec = inputVectors[current_word_vec_idx]
    for context_word in contextWords:
        predicted = current_word_vec
        target = tokens[context_word]
        c, grad_pred, grad = word2vecCostAndGradient(predicted, target,
                                                     outputVectors, dataset)
        cost = cost + c
        gradOut = gradOut + grad
        gradIn[current_word_vec_idx] += np.squeeze(grad_pred)
    ### END YOUR CODE
    return cost, gradIn, gradOut


def one_hot(ind, vocab_size):
    rec = np.zeros(vocab_size)
    rec[ind] = 1
    return rec


def cbow(currentWord,
         C,
         contextWords,
         tokens,
         inputVectors,
         outputVectors,
         dataset,
         word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    assert 2 * C >= len(contextWords)
    context_words_indices = [
        tokens[context_word] for context_word in contextWords
    ]
    context_words_input_vectors = inputVectors[context_words_indices, :]
    current_word_idx = tokens[currentWord]
    current_word_out_vector = outputVectors[current_word_idx]
    context_words_input_vectors_avg = np.sum(
        context_words_input_vectors, axis=0)  #--> 1 x D

    # STEP 1: accumulate the gradient (well, we accumulate only once)
    partial_cost, partial_gradIn, partial_gradOut = word2vecCostAndGradient(
        predicted=context_words_input_vectors_avg,
        target=current_word_idx,
        outputVectors=outputVectors,
        dataset=dataset)
    # SHAPES:
    # partial_cost : 1 x 1, scalar
    # pratial_gradIn : 1 x D --> we update only one row : the current_word_input_vector
    # partial_gradOut : |V| x D --> we update all output vectors rows; However in the case of negSamplingCostAndGradient,
    #                               If we knew which words were choosen as negative we could have reduce the size of
    #                               This partial_gradOut to N+1 x D : where N is the number of negative sample
    #                               The extra +1 been added to take into account the gradient of the true context outvector

    cost += partial_cost
    for idx in context_words_indices:
        gradIn[idx] += partial_gradIn
    gradOut += partial_gradOut
    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################


def word2vec_sgd_wrapper(word2vecModel,
                         tokens,
                         wordVectors,
                         dataset,
                         C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N / 2, :]
    outputVectors = wordVectors[N / 2:, :]
    for i in range(batchsize):
        C1 = random.randint(1, C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(centerword, C1, context, tokens,
                                     inputVectors, outputVectors, dataset,
                                     word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N / 2, :] += gin / batchsize / denom
        grad[N / 2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])
    print("==== Gradient check for skip-gram ====")
    gradcheck_naive(
        lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(
        lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)
    print("\n==== Gradient check for CBOW      ====")
    gradcheck_naive(
        lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(
        lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)

    print("\n=== Results ===")
    print(skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens,
                   dummy_vectors[:5, :], dummy_vectors[5:, :], dataset))
    print(skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5, :],
                   dummy_vectors[5:, :], dataset, negSamplingCostAndGradient))
    print(cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens,
               dummy_vectors[:5, :], dummy_vectors[5:, :], dataset))
    print(cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens,
               dummy_vectors[:5, :], dummy_vectors[5:, :], dataset,
               negSamplingCostAndGradient))


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
