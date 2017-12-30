import numpy as np
import random
import variable as var
import cell
import network
import csv

# 常数
D = 100
M = 128

# 数据准备

chIndex = {}
chs = []
with open('data/chs.data', 'r', encoding='utf8') as chsfile:
    lines = chsfile.read().splitlines()
    for line in lines:
        chIndex[line] = len(chs)
        chs.append(line)

texts = []
labels = []
with open('data/training.1600000.processed.noemoticon.csv', 'rU', encoding='utf8') as trainingfile:
    spamreader = csv.reader(trainingfile, delimiter=',', quotechar='"')
    for row in spamreader:
        texts.append(row[5])
        labels.append(float(row[0]) / 4)
print(labels[-1])

# 构造网络

rnn = network.Network();

# 输入层
X = var.Variable(np.zeros([0, D]))
Y = var.Variable(np.zeros([1, 1]))
KeepArray = var.Variable(np.zeros([1, M]))

# 循环层
WR = var.Variable(np.random.randn(D + M, M) / 10)
rnn.appendVariable(WR)
BR = var.Variable(np.ones([1, M]) / 10)
rnn.appendVariable(BR)

rCell = cell.RecurrentCellType1(X, WR, BR)
rnn.appendCell(rCell)

# Dropout层
dropoutCell = cell.DropoutCell(rCell, KeepArray)
rnn.appendCell(dropoutCell)

# 全连接层
WFC = var.Variable(np.random.randn(M, 1) / 10)
rnn.appendVariable(WFC)
BFC = var.Variable(np.ones([1, 1]) / 10)
rnn.appendVariable(BFC)

matmulCell = cell.MatMulCell(dropoutCell, WFC)
rnn.appendCell(matmulCell)

addCell = cell.AddCell(matmulCell, BFC)
rnn.appendCell(addCell)

# 激活层 - Sigmoid
sigmoidCell = cell.SigmoidCell(addCell)
rnn.appendCell(sigmoidCell)

# Loss (Cross-entropy)
loss = cell.BinaryCrossEntropyCell(sigmoidCell, Y)
rnn.appendCell(loss)

# Training
BATCH_NUMBER = 20000 # BATCH的数量
BATCH_SIZE = 50 # BATCH的大小
LEARNING_RATE = 0.1 #学习速率
# rnn.logForwardPropgation = True
# rnn.logBackwardPropagation = True
# rnn.logApplyGradient = True
for batch_index in range(BATCH_NUMBER):
    # 构造一个BATCH
    batch_xs = [];
    batch_ys = [];
    for data_index in range(BATCH_SIZE):
        j = random.randint(0, len(texts) - 1)
        x = np.zeros([len(texts[j]), D])
        for i in range(len(texts[j])):
            if texts[j][i] in chIndex:
                x[i][chIndex[texts[j][i]]] = 1
        y = labels[j];
        batch_xs.append(x);
        batch_ys.append(y);
    # print(batch_xs)
    # print(batch_ys)

    # 评估当前BATCH的准确率
    if batch_index % 10 == 0 and True:
        batch_precision = 0
        for data_index in range(BATCH_SIZE):
            x = batch_xs[data_index];
            y = batch_ys[data_index];
            X.takeInput(x, reshape = False);
            Y.takeInput(y);
            KeepArray.takeInput(np.random.binomial(1, 0.5, [1, M]));
            rnn.forwardPropagation() # 正向传播
            predict = sigmoidCell.getOutput().value
            # print(predict, y)
            if (predict < 0.5) ^ (y > 0.5):
                batch_precision += 1 / BATCH_SIZE
        print('precision =', round(batch_precision, 4))

    # 使用这个BATCH进行训练
    batch_loss = 0
    for data_index in range(BATCH_SIZE):
        x = batch_xs[data_index];
        y = batch_ys[data_index];
        X.takeInput(x, reshape = False);
        Y.takeInput(y);
        KeepArray.takeInput(np.random.binomial(1, 0.5, [1, M]));
        rnn.forwardPropagation() # 正向传播
        batch_loss += loss.getOutput().value # 统计整个BATCH的损失
        loss.getOutput().gradient = -1 / BATCH_SIZE # 整个BATCH统一计算梯度，所以单个数据点的输出梯度只有1/BATCH_SIZE
        rnn.backwardPropagation() # 反向传播
        # print('    data', data_index, ', loss =', loss.getOutput().value)
        # break

    # 应用梯度
    rnn.applyGradient(LEARNING_RATE)
    print('batch', batch_index, ', loss =', batch_loss)
    # break

# Test
# precision = 0
# for index in range(len(images_test)):
#     x = images_test[index];
#     y = labels_test[index];
#     X.takeInput(x);
#     Y.takeInput(y);
#     KeepProb.takeInput(np.array(1));
#     cnn.forwardPropagation()
#     predict = np.argmax(softmaxCell.getOutput().value)
#     if predict == np.argmax(y):
#         precision += 1 / len(images_test)
# print('Precision =', precision)