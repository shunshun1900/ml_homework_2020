import os
#获取文件当前工作目录路径
d = os.getcwd()
print(d)
#获取文件当前文件路径
print(os.path.abspath(__file__))

cur_dir = os.path.split(os.path.realpath(__file__))[0]
print(cur_dir)