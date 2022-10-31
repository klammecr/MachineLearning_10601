'''
tests.py

Unit tests for HW5 neural network implementation.

To run one test: python -m unittest unitTestsV1.TestLinear
To run all tests: python -m unittest unitTestsV1

If the above commands don't work, try using python3 instead of python
'''

import unittest
import numpy as np
import pickle as pk
from numpy.testing import assert_allclose
import sys

try:
    from neuralnet import (
        Linear, Sigmoid, NN, zero_init, 
        softmax, cross_entropy, d_softmax_cross_entropy
    )
except ImportError:
    sys.exit("Could not find `Linear`, `Sigmoid`, and `NN` classes in neuralnet.py. Unit tests will not be run.")


TOLERANCE = 1e-4

with open("unittest_data.pk", "rb") as f:
    data = pk.load(f)


class TestLinearForward(unittest.TestCase):
    '''
    Note that these tests assume the shape of your weight matrix
    <output dimension> followed by <input dimension>, so if you
    wrote it the other way around, you may have to fiddle around with
    the shapes in these tests to make your implementation pass.
    '''
    def test(self):
        T1, _ = data["linear_forward"]
        # Need to slice off bias term for input X
        w, X, soln = T1[0], T1[1][1:], T1[2]

        # init the Linear layer arbitrarily, then fill in the weight matrix 
        # for the test case
        layer = Linear(1, 1, zero_init, 0.0) 
        layer.w = w
        a = layer.forward(X)
        assert_allclose(np.squeeze(a), soln)


class TestSigmoidForward(unittest.TestCase):
    def test_1(self):
        T1, _ = data["sigmoid_forward"]
        a, soln = T1
        sigmoid = Sigmoid()
        z = sigmoid.forward(a)
        assert_allclose(np.squeeze(z), soln)
    
    def test_2(self):
        _, T2 = data["sigmoid_forward"]
        a, soln = T2
        sigmoid = Sigmoid()
        z = sigmoid.forward(a)
        assert_allclose(np.squeeze(z), soln)


class TestSoftmax(unittest.TestCase):
    def test_1(self):
        T1, _ = data["softmax_forward"]
        z, soln = T1
        yh = softmax(z)
        assert_allclose(np.squeeze(yh), soln)
    
    def test_2(self):
        _, T2 = data["softmax_forward"]
        z, soln = T2
        yh = softmax(z)
        assert_allclose(np.squeeze(yh), soln)


class TestCrossEntropy(unittest.TestCase):
    def test(self):
        T = data["ce_forward"]
        yh, y, soln = T
        loss = cross_entropy(y, yh)
        assert_allclose(np.squeeze(loss), soln)


class TestLinearBackward(unittest.TestCase):
    def test(self):
        T = data["linear_backward"]
        X, w, dxsoln, dwsoln = T[0][1:], T[1], T[2], T[3]
        layer = Linear(1, 1, zero_init, 0.0)
        layer.w = w
        z = layer.forward(X) # forward pass to ensure layer caches values
        dz = np.ones_like(z) # use all 1's for gradient w.r.t output
        dx = layer.backward(dz)
        dw = layer.dw
        assert_allclose(np.squeeze(dx), dxsoln)
        assert_allclose(np.squeeze(dw), dwsoln)


class TestSigmoidBackward(unittest.TestCase):
    def test(self):
        T = data["sigmoid_backward"]
        z, soln = T
        sigmoid = Sigmoid()
        _ = sigmoid.forward(z)
        dz = sigmoid.backward(1)
        assert_allclose(np.squeeze(dz), soln)


class TestDSoftmaxCrossEntropy(unittest.TestCase):
    def test(self):
        T = data["ce_backward"]
        y, yh, soln = T
        db = d_softmax_cross_entropy(y, yh)
        assert_allclose(np.squeeze(db), soln)


class TestNNForward(unittest.TestCase):
    def test(self):
        T1, _ = data["forward_backward"]
        x, y, soln, _, _ = T1
        x = x[1:]
        nn = NN(input_size=len(x), hidden_size=4, output_size=10,
                learning_rate=1, weight_init_fn=zero_init)
        yh = nn.forward(x)
        assert_allclose(np.squeeze(yh), soln)


class TestNNBackward(unittest.TestCase):
    def test(self):
        T1, _ = data["forward_backward"]
        x, y, soln_yh, soln_d_w1, soln_d_w2 = T1
        x = x[1:]
        nn = NN(input_size=len(x), hidden_size=4, output_size=10,
                learning_rate=1, weight_init_fn=zero_init)
        yh = nn.forward(x)
        nn.backward(y, yh)
        assert_allclose(np.squeeze(nn.linear1.dw), soln_d_w1)
        assert_allclose(np.squeeze(nn.linear2.dw), soln_d_w2)
