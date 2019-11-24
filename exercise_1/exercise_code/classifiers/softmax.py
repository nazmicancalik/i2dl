"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier

# Return the biggest element in the given list.
'''
def get_max(X):
    max = X[0]
    for e in X:
        if e > max:
            max = e
    return max
'''

# Softmax with loops.
def softmax(X):

    # For making the equation numerically stable we subtract the max element from 
    # each element so that the exponentials of the numbers are not so big which prevents
    # overflow and NaN errors.
    max = np.max(X)
    exps = np.exp(X - max)
    return exps / np.sum(exps)


def cross_entropy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    
    n_class = W.shape[1]
    n_train = X.shape[0]

    for index,e in enumerate(X):
        z = np.dot(e,W)
        z -= np.max(z)
        
        sum = np.sum(np.exp(z))

        p = lambda k: np.exp(z[k]) / sum
        loss += -np.log(p(y[index]))
        
        for k in range(n_class):
            # Probability of the class k being the class according to the model.
            p_k = p(k)
            # (S_i - del_yi)*Xj
            dW[:,k] += (p_k - (k == y[index])) * X[index]
    
    loss /= n_train

    # Regularization to avoid overfitting.
    loss += 0.5 * reg * np.sum(W*W)
    
    dW /= n_train
    dW += reg * W 

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    n_train = X.shape[0]
    Z = np.dot(X,W)
    Z -= np.max(Z,axis=1,keepdims=True)

    sum_Z = np.sum(np.exp(Z), axis=1, keepdims=True)
    probs = np.exp(Z)/sum_Z

    loss = np.sum(-np.log(probs[np.arange(n_train),y]))
    loss /= n_train
    loss += 0.5 * reg * np.sum(W * W)

    # Finding the gradient.
    indices = np.zeros_like(probs)
    indices[np.arange(n_train),y] = 1
    dW = X.T.dot(probs - indices)
    
    # Normalize the gradient.
    dW /= n_train

    # Regularize the gradient.
    dW += reg*W
    

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [1e-7, 5e-7]
    regularization_strengths = [2.5e4, 5e4]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################

    pass

    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
