import variable as var
import numpy as np
import scipy.signal

# 配置
np.set_printoptions(threshold=np.nan)
eps = 1e-6

# 基础类

class Cell:
    def getOutput(self):
        return self.output
    def logOutputValue(self):
        print(self.output.value)
    def logOutputGradient(self):
        print(self.output.gradient)

class SCell(Cell):
    def __init__(self, input):
        self.input = input

class DCell(Cell):
    def __init__(self, input0, input1):
        self.input0 = input0
        self.input1 = input1

# 计算单元类

class AddCell(DCell):
    def forwardPropagation(self):
        self.output = var.Variable(self.input0.getOutput().value + self.input1.getOutput().value)
    def backwardPropagation(self):
        self.input0.getOutput().gradient += self.output.gradient
        self.input1.getOutput().gradient += self.output.gradient
    def logOutputValue(self):
        print(self.output.value.shape, np.sum(self.output.value))
    def logOutputGradient(self):
        print(self.output.gradient.shape, np.sum(self.output.gradient))

class MatMulCell(DCell):
    def forwardPropagation(self):
        self.output = var.Variable(np.dot(self.input0.getOutput().value, self.input1.getOutput().value))
    def backwardPropagation(self):
        self.input0.getOutput().gradient += np.dot(self.output.gradient, self.input1.getOutput().value.T)
        self.input1.getOutput().gradient += np.dot(self.input0.getOutput().value.T, self.output.gradient)
    def logOutputValue(self):
        print(self.output.value.shape, np.sum(self.output.value))
    def logOutputGradient(self):
        print(self.output.gradient.shape, np.sum(self.output.gradient))

class RecurrentCellType1(Cell):
    def __init__(self, input, weight, bias):
        self.input = input
        self.weight = weight
        self.bias = bias
    def forwardPropagation(self):
        [n, d] = self.input.getOutput().value.shape
        [tmp, m] = self.bias.getOutput().value.shape
        self.hidden = [np.zeros([1, m])]
        for i in range(n):
            # h_{t-1} + x_t
            tmp1 = np.append(self.input.getOutput().value[i], self.hidden[-1])
            # (h_{t-1} + x_t) * W
            tmp2 = np.dot(tmp1, self.weight.getOutput().value)
            # (h_{t-1} + x_t) * W + B
            tmp3 = tmp2 + self.bias.getOutput().value
            # tanh((h_{t-1} + x_t) * W + B)
            new_hidden = np.tanh(tmp3)
            self.hidden.append(new_hidden)
        self.output = var.Variable(self.hidden[-1])
    def backwardPropagation(self):
        [n, d] = self.input.getOutput().value.shape
        [tmp, m] = self.bias.getOutput().value.shape
        gradient = self.output.gradient
        for i in reversed(range(n)):
            value = self.hidden.pop()
            gradient = gradient * (1 - value * value)
            self.bias.getOutput().gradient += gradient
            tmp1 = np.append(self.input.getOutput().value[i], self.hidden[-1]).T
            tmp1 = np.reshape(tmp1, [m + d, 1])
            self.weight.getOutput().gradient += np.dot(tmp1, gradient)
            gradient = np.dot(gradient, self.weight.getOutput().value.T)
            gradient = gradient[:,d:]
    def logOutputValue(self):
        print(self.output.value.shape, np.sum(self.output.value))
    def logOutputGradient(self):
        print(self.output.gradient.shape, np.sum(self.output.gradient))

class SigmoidCell(SCell):
    def forwardPropagation(self):
        self.output = var.Variable(1 / (1 + np.exp(-self.input.getOutput().value)))
    def backwardPropagation(self):
        self.input.getOutput().gradient += self.output.gradient * self.output.value * (1 - self.output.value)
    def logOutputGradient(self):
        print(self.output.gradient, self.output.value, self.input.getOutput().value)

class DropoutCell(Cell):
    def __init__(self, input, keeparray):
        self.input = input
        self.keeparray = keeparray
    def forwardPropagation(self):
        # 与分布矩阵相乘，从而“遮罩”某些数值
        self.output = var.Variable(self.input.getOutput().value * self.keeparray.getOutput().value)
    def backwardPropagation(self):
        self.input.getOutput().gradient += self.output.gradient * self.keeparray.getOutput().value
    def logOutputValue(self):
        print(self.output.value.shape, np.sum(self.output.value))
    def logOutputGradient(self):
        print(self.output.gradient.shape, np.sum(self.output.gradient))

class BinaryCrossEntropyCell(DCell):
    def forwardPropagation(self):
        self.output = var.Variable(- np.log(self.input0.getOutput().value) * self.input1.getOutput().value - np.log((1 - self.input0.getOutput().value)) * (1 - self.input1.getOutput().value))
    def backwardPropagation(self):
        self.input0.getOutput().gradient += self.output.gradient * -self.input1.getOutput().value / self.input0.getOutput().value
        self.input1.getOutput().gradient += self.output.gradient * -np.log(self.input0.getOutput().value)
        self.input0.getOutput().gradient += self.output.gradient * (1 - self.input1.getOutput().value) / (1 - self.input0.getOutput().value)
        self.input1.getOutput().gradient += self.output.gradient * np.log(1 - self.input0.getOutput().value)
    def logOutputGradient(self):
        print(self.output.gradient, self.output.value, self.input0.getOutput().value, self.input1.getOutput().value)
            