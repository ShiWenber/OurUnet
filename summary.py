# 读取 log 查看 acc 结果
import numpy as np
import os
import sys
# 读取文件中的字符串内容

def get_files_by_re(base, files_re):
  """
  通过re正则匹配获取所有名称符合正则的文件路径

  Args:
      base (str): 起点路径
      files_re (str): 用以匹配文件名的正则表达式

  Returns:
      list: 匹配到的文件完整路径列表
  """
  import re 
  import os
  paths = []
  all_r = list(os.walk(base))
  # dir_path路径名， dir_names在dir_path路径下的子目录名， files在dir_path路径下的文件名
  for dir_path, dir_names, file_names in all_r:
    for file_name in file_names:
      if re.match(files_re, file_name):
        # findall 输出列表，需要取出第一个元素
        paths.append(os.path.join(dir_path, file_name))

  return paths

def get_acc(paths: list()):
    """获取nohup日志中的acc值

    Args:
        path (list(str)): 日志文件路径

    Returns:
        tuple: (acc_max, acc_mean, acc_minus, acc_plus, acc_ls)
    """
    # data = open("./nohup_logs/20230515002301/IMDB-BINARY.log").read()
    mean_dice_ls = []
    mean_hd95_ls = []
    for path in paths:
        data = open(path).read()
        # 按照换行符和逗号分割字符串
        data_ls = data.split("\n")
        for i in data_ls:
            for j in i.split("model: "):
                if (j.startswith("mean_dice :")):
                    # print(j)
                    print(j)
                    ls = [x.split(": ") for x in j.split()]

                    mean_dice = float(ls[2][0])
                    mean_hd95 = float(ls[5][0])
                    if (mean_dice <= 0.20 or mean_hd95 <= 5):
                        continue
                    mean_dice_ls.append(mean_dice)
                    mean_hd95_ls.append(mean_hd95)

    if len(mean_dice_ls) == 0:
        print(f"{path} acc_ls is null")
        return None

    # 计算acc的平均值和+/-误差
    acc_mean = np.mean(mean_dice_ls)
    # print("acc_mean: ",acc_mean)
    mean_dice_max = np.max(mean_dice_ls)
    # 获得其对应的hd95
    d_max_hd95 = mean_hd95_ls[mean_dice_ls.index(mean_dice_max)]
    mean_hd95_min = np.min(mean_hd95_ls)
    # 获得其对应的dice
    hd95_min_dice = mean_dice_ls[mean_hd95_ls.index(mean_hd95_min)]
    print("mean_dice_max", (mean_dice_max, d_max_hd95))
    print("mean_hd95_max", (mean_hd95_min, hd95_min_dice))

    pair_ls = [(mean_dice_ls[i], mean_hd95_ls[i]) for i in range(len(mean_dice_ls))]

    # 按照 dice 从大到小排序，且输出其对应的 hd95
    # print("mean_dice_max->min", sorted(pair_ls, key=lambda x: x[0], reverse=True))
    # 输出到csv文件
    with open("mean_dice_max->min.csv", "w") as f:
        for i in sorted(pair_ls, key=lambda x: x[0], reverse=True):
            f.write(f"{i[0]},{i[1]}\n")
    # 按照 hd95 从小到大排序，且输出其对应的 dice
    # print("mean_hd95_min->max", sorted(pair_ls, key=lambda x: x[1], reverse=False))
    with open("mean_hd95_min->max.csv", "w") as f:
        for i in sorted(pair_ls, key=lambda x: x[1], reverse=False):
            f.write(f"{i[0]},{i[1]}\n")

    # 计算负误差
    acc_minus = acc_mean - np.min(mean_dice_ls)
    acc_plus = np.max(mean_dice_ls) - acc_mean
    # print("acc_minus: ",acc_minus)
    # print("acc_plus: ",acc_plus)
    return (mean_dice_max, d_max_hd95), (mean_hd95_min, hd95_min_dice)

if __name__ == "__main__":
    import os
    args = sys.argv
    dir_path_ls = []
    if args.__len__() == 1:
        print("args is null") 
        print("[Usage]: python summary_logs.py <dir_path> <dir_path> ...")
        exit()
    else:
        for i in args[1:]:
            dir_path = i
            # 检查是否以/结尾
            if dir_path[-1] != "/":
                dir_path += "/"
            print(dir_path)
            dir_path_ls.append(dir_path)
    file_path_ls = []
    for i in dir_path_ls:    
        rootpath = i
        # paths = get_files_by_re(rootpath, "^[\s\S]*log.txt$")
        paths = get_files_by_re(rootpath, "^[\s\S]*log.txt$")
        file_path_ls += paths
    print(file_path_ls)

    acc_dict = {}
    dataname_2_path_ls_dict = {}
    for path in file_path_ls:
        path_ls = []
        file_name = path.split("/")[-1]
        dataname = file_name.split(".")[0]
        if dataname not in dataname_2_path_ls_dict.keys():
            dataname_2_path_ls_dict[dataname] = [path]
        else:
            dataname_2_path_ls_dict[dataname].append(path)
    for i in dataname_2_path_ls_dict.keys():
        print(f"====================================={i}")
        for j in dataname_2_path_ls_dict[i]:
            print(j)
        acc_max = 0
        res = get_acc(dataname_2_path_ls_dict[i])
        if (res == None):
            continue
        name = i
        acc_dict[name] = res

    print(dataname_2_path_ls_dict) 
    print()
    print(acc_dict)
    exit()    
        





    #     print(f"====================================={i}")
    #     path = i
    #     print(path)
    #     res = get_acc(path)
    #     if (res == None):
    #         continue
    #     acc_max = res[0]
    #     name = i.split(".")[0]
    #     acc_dict[name] = acc_max
    # print()
    # print()
    # print("=====================================summary")
    # print(acc_dict)
 