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
    acc_ls = []
    for path in paths:
        data = open(path).read()
        # 按照换行符和逗号分割字符串
        data_ls = data.split("\n")
        for i in data_ls:
            for j in i.split(","):
                if (j.startswith("Acc")):
                    # print(j)
                    j.replace("Acc: ","")
                    acc = float(j.split(": ")[1])
                    acc_ls.append(acc)
    if len(acc_ls) == 0:
        print(f"{path} acc_ls is null")
        return None

    # 计算acc的平均值和+/-误差
    acc_mean = np.mean(acc_ls)
    print("acc_mean: ",acc_mean)
    acc_max = np.max(acc_ls)
    print("acc_max", acc_max)
    # 计算负误差
    acc_minus = acc_mean - np.min(acc_ls)
    acc_plus = np.max(acc_ls) - acc_mean
    print("acc_minus: ",acc_minus)
    print("acc_plus: ",acc_plus)
    return acc_max, acc_mean, acc_minus, acc_plus, acc_ls

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
        paths = get_files_by_re(rootpath, "^[\s\S]*\log.txt$")
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
        acc_max = res[0]
        name = i
        acc_dict[name] = acc_max

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
 