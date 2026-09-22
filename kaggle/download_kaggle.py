import os
import kagglehub

# 1. 定义你想要的下载文件夹路径
# 这里使用了你报错信息中的桌面路径，r 表示原生字符串，防止反斜杠转义
custom_path = r"Y:\中草药识别\kagglehub"

# 2. 强制将 kagglehub 的缓存目录指向该路径
os.environ["KAGGLEHUB_CACHE"] = custom_path

# （可选）如果你之前没配置 .kaggle 文件夹，可以在这里继续加上账号信息
# os.environ["KAGGLE_USERNAME"] = "你的用户名"
# os.environ["KAGGLE_KEY"] = "你的密钥"

# 3. 开始下载
path = kagglehub.competition_download('chinese-medicine-image')

print("Path to competition files:", path)