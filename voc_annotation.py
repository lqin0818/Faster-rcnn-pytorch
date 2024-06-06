import os
import random
import xml.etree.ElementTree as ET

import numpy as np
from utils.utils import get_class_list

#-----------------------------------------------------------------------------------------------------#
#   mode用于指定该文件运行时计算的内容
#   0代表整个标签处理过程，包括获得VOCdevkit/VOC2007/ImageSets里面的txt以及训练用的2007_train.txt、2007_val.txt
#   1代表获得VOCdevkit/VOC2007/ImageSets里面的txt
#   2代表获得训练用的2007_train.txt、2007_val.txt
#-----------------------------------------------------------------------------------------------------#
mode = 2

class_path = 'voc_classes.txt'

#--------------------------------------------------------------------------------------------------------------------------------#
#   trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1
#   train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1
#   仅在annotation_mode为0和1的时候有效
#--------------------------------------------------------------------------------------------------------------------------------#
trainval_percent    = 0.9
train_percent       = 0.9
#-------------------------------------------------------#
#   指向VOC数据集所在的文件夹
#   默认指向根目录下的VOC数据集
#-------------------------------------------------------#
VOCdevkit_path = 'VOCdevkit'

VOCdevkit_sets = [('2007', 'train'), ('2007', 'val')]
class_list, nc = get_class_list(class_path)

#-------------------------------------------------------#
#   统计目标数量
#-------------------------------------------------------#
image_nums  = np.zeros(len(VOCdevkit_sets))
nums        = np.zeros(nc)

def convert_annotation(year, image_set):
    txt_path = os.path.join(VOCdevkit_path, 'VOC{}/ImageSets/Main/{}.txt'.format(year,image_set))
    with open(txt_path, 'r',encoding='utf-8') as f:
        lines = f.readlines()
        image_ids = [line.strip() for line in lines]

    with open('{}_{}.txt'.format(year, image_set), 'w', encoding='utf-8') as list_file:

        for i in image_ids:
            list_file.write('{}/VOC{}/JPEGImages/{}.jpg'.format(os.path.abspath(VOCdevkit_path), year, i))

            xml_path = os.path.join(VOCdevkit_path, 'VOC{}/Annotations/{}.xml'.format(year,i))
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for obj in root.iter('object'):
                cls = obj.find('name').text

                difficult = 0
                if obj.find('difficult'):
                    difficult = obj.find('difficult').text

                if cls not in class_list or int(difficult) == 1:
                    continue
                cls_id = class_list.index(cls)
                xmlbox = obj.find('bndbox')
                xmin = float(xmlbox.find('xmin').text)
                ymin = float(xmlbox.find('ymin').text)
                xmax = float(xmlbox.find('xmax').text)
                ymax = float(xmlbox.find('ymax').text)
                box = (int(xmin), int(ymin), int(xmax), int(ymax))

                list_file.write(" " + ",".join([str(a) for a in box]) + ',' + str(cls_id))

                nums[cls_id] += 1
            list_file.write('\n')

    return len(image_ids)

if __name__ == "__main__":
    random.seed(0)
    if " " in os.path.abspath(VOCdevkit_path):
        raise ValueError("数据集存放的文件夹路径与图片名称中不可以存在空格，否则会影响正常的模型训练，请注意修改。")

    if mode == 0 or mode == 1:
        print("Generate txt in ImageSets.")
        xml_path = os.path.join(VOCdevkit_path, 'VOC2007/Annotations')
        saveBasePath = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Main')

        total_xml = [xml for xml in os.listdir(xml_path) if xml.endswith('.xml')]

        total_name = [xml[:-4] for xml in total_xml]
        #print(total_name)
        num = len(total_name)
        #print(num)
        incidies = range(num)

        train_val_num = int(num*trainval_percent)
        train_num = int(train_val_num*train_percent)

        trainval_idx= random.sample(incidies,train_val_num) #从一个列表中随机提取一定数量的元素，组成新的list
        train_idx = random.sample(trainval_idx,train_num)
        test_idx = list(set(incidies)-set(trainval_idx))

        test_num = len(test_idx)

        print("train and val size",train_val_num)
        print("train size",train_num)
        print("test size", test_num)

        with open (os.path.join(saveBasePath,'trainval.txt'), 'w') as ftrainval,\
             open(os.path.join(saveBasePath,'train.txt'), 'w') as ftrain,\
             open(os.path.join(saveBasePath,'val.txt'), 'w') as fval,\
             open(os.path.join(saveBasePath,'test.txt'), 'w') as ftest:
            for i in trainval_idx:
                ftrainval.write(total_name[i])
                ftrainval.write('\n')
                if i in train_idx:
                    ftrain.write(total_name[i])
                    ftrain.write('\n')
                else:
                    fval.write(total_name[i])
                    fval.write('\n')
            for i in test_idx:
                ftest.write(total_name[i])
                ftest.write('\n')
        print("Generate txt in ImageSets done.")


    if mode == 0 or mode == 2:
        print("Generate 2007_train.txt and 2007_val.txt for train.")
        type_index = 0
        for year, image_set in VOCdevkit_sets:
            x = convert_annotation(year, image_set)

            image_nums[type_index] = x
            type_index += 1
        print("Generate 2007_train.txt and 2007_val.txt for train done.")
        
        def printTable(List1, List2):
            for i in range(len(List1[0])):
                print("|", end=' ')
                for j in range(len(List1)):
                    print(List1[j][i].rjust(int(List2[j])), end=' ')
                    print("|", end=' ')
                print()

        str_nums = [str(int(x)) for x in nums]
        tableData = [
            class_list, str_nums
        ]
        colWidths = [0]*len(tableData)
        len1 = 0
        for i in range(len(tableData)):
            for j in range(len(tableData[i])):
                if len(tableData[i][j]) > colWidths[i]:
                    colWidths[i] = len(tableData[i][j])
        printTable(tableData, colWidths)


