# 清理测试数据
import os
import re

a = 0
for root, dirs, files in os.walk('./xirou/valid'):
    for name in files:
        # 匹配所有数字命名的png文件
        # if re.match(r'\d+\.png', name) is None:
        if name.endswith('_res.png'):
            os.remove(os.path.join(root, name))
            print(os.path.join(root, name))
            a += 1
print(a)