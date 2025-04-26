import json
import os
def load_file_list_recursion(fpath, result):
    allfilelist = os.listdir(fpath)
    for file in allfilelist:
        filepath = os.path.join(fpath, file)
        if os.path.isdir(filepath):
            load_file_list_recursion(filepath, result)
        else:
            result.append(filepath)
            print(len(result))



def scan(input_path, out_put):
    result_list = []
    load_file_list_recursion(input_path, result_list)
    result_list.sort()

    for i in range(len(result_list)):
        print('{}_{}'.format(i, result_list[i]))

    with open(out_put, 'w') as j:
        json.dump(result_list, j)

#scan('/mnt/data/Zz/data/mycard/TrainData/input', './inputDR_train.txt')
#scan('/mnt/data/Zz/data/SD1/SD1train/mask', './maskDR_train.txt')
#scan('/mnt/data/Zz/data/mycard/TrainData/gt', './gtDR_train.txt')
#scan('/mnt/data/Zz/data/mycard/TestData/input', './inputDR_test.txt')
#scan('/mnt/data/Zz/data/SD1/SD1test/mask', './maskDR_test.txt')
#scan('/mnt/data/Zz/data/mycard/TestData/gt', './gtDR_test.txt')
scan('/mnt/data/Zz/data/mycard/TestData/input', './inputDR_train.txt')
scan('/mnt/data/Zz/data/RD/RDtest/mask', './maskDR_train.txt')
scan('/mnt/data/Zz/data/mycard/TestData/gt', './gtDR_train.txt')
scan('/mnt/data/Zz/data/ok/input', './inputDR_test.txt')
scan('/mnt/data/Zz/data/RD/RDtest/mask', './maskDR_test.txt')
scan('/mnt/data/Zz/data/ok/gt', './gtDR_test.txt')