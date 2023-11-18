import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import numpy
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE

# 读取csv文件
import torch
from sklearn import svm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score



# def load_data(path):
#     data_frame = pd.read_csv(path)
#     # data_frame['isDefective'] = data_frame['isDefective'].replace('buggy', 1)
#     # data_frame['isDefective'] = data_frame['isDefective'].replace('clean', 0)
#     # lable = data_frame['isDefective'].values
#     # data_frame = data_frame.drop('isDefective', axis=1)
#     data_frame['class'] = data_frame['class'].replace('buggy', 1)
#     data_frame['class'] = data_frame['class'].replace('clean', 0)
#     lable = data_frame['class'].values
#     data_frame = data_frame.drop('class', axis=1)
#     df_26 = data_frame.iloc[:, 0:26]
#     # df_26 = data_frame.iloc[:, :26]
#
#     # 将数据转换为多维数组
#     # multi_dim_array = data_frame.values
#     multi_dim_array = df_26.values
#     end_train_data = (multi_dim_array[:100], lable[:100],)
#     # end_val_data = (multi_dim_array[100:150], lable[100:150],)
#     end_test_data = (multi_dim_array[-40:], lable[-40:],)
#     train_set_x, train_set_y = make_tensor(end_train_data)
#     # valid_set_x, valid_set_y = make_tensor(end_val_data)
#     test_set_x, test_set_y = make_tensor(end_test_data)
#     # print(len(train_set_x[0]))
#     return [(train_set_x, train_set_y), (test_set_x, test_set_y)]
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import binarize


def load_data(path):
    # data_frame = pd.read_csv(path)
    # data_frame = data_frame.sample(frac=1)
    # last_name = data_frame.columns[-1]
    # data_frame[last_name] = data_frame[last_name].replace('buggy', 1)
    # data_frame[last_name] = data_frame[last_name].replace('clean', 0)
    # data_lable = data_frame[last_name][:300].values
    # # data_lable = data_frame[last_name].values[:300]
    # data_frame = data_frame.iloc[:, 0:61]
    # # data_frame = data_frame.iloc[:, 0:310]
    # data = data_frame[:300].values
    # # data = data_frame[:300].values
    # end_train_data = (data[:240], data_lable[:240],)
    # # end_train_data = (data[:200], data_lable[:200],)
    # end_test_data = (data[-60:], data_lable[-60:],)
    # # end_test_data = (data[-100:], data_lable[-100:],)
    # train_set_x, train_set_y = make_tensor(end_train_data)
    # test_set_x, test_set_y = make_tensor(end_test_data)

    print(path)
    data_frame = pd.read_csv(path)
    sampled_data = data_frame.sample(n=300, random_state=42)
    # data_frame = data_frame.sample(frac=1)
    last_name = sampled_data.columns[-1]
    sampled_data[last_name] = sampled_data[last_name].replace('buggy', 1)
    sampled_data[last_name] = sampled_data[last_name].replace('clean', 0)
    data_lable = sampled_data[last_name].values
    # data_lable = data_frame[last_name].values[:300]
    data_frame = sampled_data.iloc[:, 0:61]
    # data_frame = data_frame.iloc[:, 0:310]
    data = data_frame.values
    # data = data_frame[:300].values

    # 随机划分训练集和测试集，比例为8:2
    data_train, data_test, label_train, label_test = train_test_split(data, data_lable, test_size=0.2, random_state=42)
    end_train_data = (data_train, label_train)
    # end_train_data = (data[:200], data_lable[:200],)
    end_test_data = (data_test, label_test)
    # end_test_data = (data[-100:], data_lable[-100:],)
    train_set_x, train_set_y = make_tensor(end_train_data)
    test_set_x, test_set_y = make_tensor(end_test_data)

    true_num, false_num = check_list(data_lable)
    print("train----true:{},false:{}".format(true_num, false_num))
    true_num, false_num = check_list(data_lable)
    print("test----true:{},false:{}".format(true_num, false_num))

    return [(train_set_x, train_set_y), (test_set_x, test_set_y)]

def check_list(dataset):
    true_list = []
    for i in dataset:
        if i == 1:
            true_list.append(i)
    return len(true_list), len(dataset) - len(true_list)




# def load_data1(path1):
#     data_frame = pd.read_csv(path1)
#     data_frame['class'] = data_frame['class'].replace('buggy', 1)
#     data_frame['class'] = data_frame['class'].replace('clean', 0)
#     lable = data_frame['class'].values
#     data_frame = data_frame.drop('class', axis=1)
#     df_26 = data_frame.iloc[:, 0:26]
#
#     # 将数据转换为多维数组
#     multi_dim_array = df_26.values
#     end_train_data = (multi_dim_array[:100], lable[:100],)
#     # end_val_data = (multi_dim_array[100:150], lable[100:150],)
#     end_test_data = (multi_dim_array[-40:], lable[-40:],)
#     train_set_x, train_set_y = make_tensor(end_train_data)
#     # valid_set_x, valid_set_y = make_tensor(end_val_data)
#     test_set_x, test_set_y = make_tensor(end_test_data)
#     # print(len(train_set_x[0]))
#     return [(train_set_x, train_set_y), (test_set_x, test_set_y)]




def make_tensor(data_xy):
    """converts the input to numpy arrays"""
    data_x, data_y = data_xy
    data_x = torch.tensor(data_x)
    data_y = np.asarray(data_y, dtype='int32')
    # data_y = np.asarray(data_y, dtype='float')
    return data_x, data_y


def svm_classify(data, C):
    """
    trains a linear SVM on the data
    input C specifies the penalty factor of SVM
    """
    train_data, _, train_label = data[0]
    test_data, _, test_label = data[1]

    # 使用SMOTE对训练数据进行采样
    # smote = SMOTE(k_neighbors=8)
    smote = SMOTE()
    train_data_resampled, train_label_resampled = smote.fit_resample(train_data, train_label)

    # 使用Borderline-SMOTE对训练数据进行过采样
    # borderline_smote = BorderlineSMOTE(sampling_strategy=0.8, k_neighbors=5)
    # borderline_smote = BorderlineSMOTE()
    # train_data_resampled, train_label_resampled = borderline_smote.fit_resample(train_data, train_label)

    # 使用ADASYN对训练数据进行过采样
    # adasyn = ADASYN()
    # train_data_resampled, train_label_resampled = adasyn.fit_resample(train_data, train_label)

    # 训练分类器
    print('training SVM...')
    clf = svm.LinearSVC(C=C, dual=False)
    clf.fit(train_data_resampled, train_label_resampled.ravel())

    # print('training SVM...')
    # clf = svm.LinearSVC(C=C, dual=False)
    # clf.fit(train_data, train_label.ravel())

    # p = clf.predict(test_data)
    # test_acc = accuracy_score(test_label, p)
    # p = clf.predict(valid_data)
    # valid_acc = accuracy_score(valid_label, p)

    # p = clf.decision_function(test_data)
    # test_auc = roc_auc_score(test_label, p)
    # p = clf.decision_function(valid_data)
    # valid_auc = roc_auc_score(valid_label, p)

    p = clf.predict(test_data)
    f1_measure = f1_score(test_label, p)

    # p = clf.predict(test_data)
    # test_f1_macro = f1_score(test_label, p, average='macro')
    # test_f1_weighted = f1_score(test_label, p, average='weighted')
    # test_f1_micro = f1_score(test_label, p, average='micro')
    # p = clf.predict(valid_data)
    # valid_f1_macro = f1_score(valid_label, p, average='macro')
    # valid_f1_weighted = f1_score(valid_label, p, average='weighted')

    # p = clf.predict(test_data)
    # tn, fp, fn, tp = confusion_matrix(test_label, p).ravel()
    # print('tn is:', tn)
    # print('fp is:', fp)
    # print('fn is:', fn)
    # print('tp is:', tp)
    # # recall = tp / (tp + fn)
    # # specificity = tn / (tn + fp)
    # # g_measure = 2 * (recall * specificity) / (recall + specificity)
    # # specificity = 1 - (fp / (fp + tn))
    # # g_mean = np.sqrt(recall * specificity)
    # mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))


    # 使用SMOTE对训练数据进行采样
    # smote = SMOTE()
    # train_data_resampled, train_label_resampled = smote.fit_resample(train_data, train_label)
    #
    # # 训练分类器
    # clf = svm.LinearSVC(C=C, dual=False)
    # clf.fit(train_data_resampled, train_label_resampled.ravel())
    #
    # # 进行预测
    # p = clf.predict(test_data)
    # f1_measure = f1_score(test_label, p)

    # return test_f1_micro
    # return test_f1_macro
    # return test_f1_weighted
    # return test_acc
    # return test_auc
    return f1_measure
    # return g_measure
    # return g_mean
    # return mcc