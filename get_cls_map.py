import numpy as np
import matplotlib.pyplot as plt
import torch
def get_classification_map(y_pred, y):

    height = y.shape[0]
    width = y.shape[1]
    k = 0
    cls_labels = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            target = int(y[i, j])
            if target == 0:
                continue
            else:
                #cls_labels[i][j] = y_pred[k]+1#原本
                cls_labels[i][j] = y_pred[k]

                k += 1

    return  cls_labels

def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([147, 67, 46]) / 255.
        if item == 2:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 3:
            y[index] = np.array([255, 100, 0]) / 255.
        if item == 4:
            y[index] = np.array([0, 255, 123]) / 255.
        if item == 5:
            y[index] = np.array([164, 75, 155]) / 255.
        if item == 6:
            y[index] = np.array([101, 174, 255]) / 255.
        if item == 7:
            y[index] = np.array([118, 254, 172]) / 255.
        if item == 8:
            y[index] = np.array([60, 91, 112]) / 255.
        if item == 9:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 10:
            y[index] = np.array([255, 255, 125]) / 255.
        if item == 11:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 12:
            y[index] = np.array([100, 0, 255]) / 255.
        if item == 13:
            y[index] = np.array([0, 172, 254]) / 255.
        if item == 14:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 15:
            y[index] = np.array([171, 175, 80]) / 255.
        if item == 16:
            y[index] = np.array([101, 193, 60]) / 255.

    return y

def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1]*2.0/dpi, ground_truth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0

def test(device, net, test_loader):
    count = 0
    # 模型测试
    net.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs,cbrs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test

def get_cls_map(y_pred, y, map_path,map_path_gt):
    # 使用 torch.nonzero() 获取非零元素的索引
    nonzero_indices = np.nonzero(y_pred)

    # 使用索引提取非零元素
    y_pred = y_pred[nonzero_indices].squeeze()
    y_pred = y_pred.flatten()
    unique_elements_pred = np.unique(y_pred)
    print("....unique_elements_pred{}".format(unique_elements_pred))
    #y_pred, y_new = test(device, net, all_data_loader)
    print("....y.shape{}".format(y.shape))
    print("....y_pred.shape{}".format(y_pred.shape))
    cls_labels = get_classification_map(y_pred, y)#cls_labels二维
    print("....cls_labels.shape{}".format(cls_labels.shape))
    unique_elements = np.unique(cls_labels)
    print("....unique_elements{}".format(unique_elements))
    x = np.ravel(cls_labels)
    print("....x.shape{}".format(x.shape))
    gt = y.flatten()
    print("....gt.shape{}".format(gt.shape))
    unique_elements_y = np.unique(y)
    print("....unique_elements_y{}".format(unique_elements_y))
    y_list = list_to_colormap(x)
    print("....y_list.shape{}".format(y_list.shape))
    y_gt = list_to_colormap(gt)
    print("....y_gt.shape{}".format(y_gt.shape))

    y_re = np.reshape(y_list, (y.shape[0], y.shape[1], 3))
    print("....y_re.shape{}".format(y_re.shape))
    gt_re = np.reshape(y_gt, (y.shape[0], y.shape[1], 3))
    print("....gt_re.shape{}".format(gt_re.shape))
    # classification_map(y_re, y, 300,
    #                    'classification_maps/' + 'IP_predictions.eps')
    classification_map(y_re, y, 300,
                       map_path)
    # classification_map(gt_re, y, 300,
    #                    map_path_gt)
    print('------Get classification maps successful-------')
    return cls_labels



# if __name__ == '__main__':
#     model = SSFTTnet()
#     model.eval()
#     print(model)
#     input = torch.randn(64, 1, 30, 13, 13)
#     y = model(input)
#     print(y.size())
