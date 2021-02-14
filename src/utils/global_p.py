# Path
DATA_DIR = '../data/'  # 原始数据文件及预处理的数据文件目录
DATASET_DIR = '../dataset/'  # 划分好的数据集目录
MODEL_DIR = '../model/'  # 模型保存路径
LOG_DIR = '../log/'  # 日志输出路径
RESULT_DIR = '../result/'  # 数据集预测结果保存路径
COMMAND_DIR = '../command/'  # run.py所用command文件保存路径
LOG_CSV_DIR = '../log_csv/'  # run.py所用结果csv文件保存路径

# Preprocess/DataLoader
TRAIN_SUFFIX = '.train.csv'  # 训练集文件后缀
VALIDATION_SUFFIX = '.validation.csv'  # 验证集文件后缀
TEST_SUFFIX = '.test.csv'  # 测试集文件后缀
INFO_SUFFIX = '.info.json'  # 数据集统计信息文件后缀
USER_SUFFIX = '.user.csv'  # 数据集用户特征文件后缀
ITEM_SUFFIX = '.item.csv'  # 数据集物品特征文件后缀
TRAIN_GROUP_SUFFIX = '.train_group.csv'  # 训练集用户交互按uid合并之后的文件后缀
TRAIN_POS_SUFFIX = '.train_pos.csv'  # 训练集用户交互按uid合并之后的文件后缀
VT_POS_SUFFIX = '.vt_pos.csv'  # 验证集和测试集用户交互按uid合并之后的文件后缀
VT_GROUP_SUFFIX = '.vt_group.csv'  # 验证集和测试集用户交互按uid合并之后的文件后缀

C_HISTORY = 'history'  # 历史记录column名称
C_HISTORY_LENGTH = 'history_length'  # 历史记录长度column名称
C_HISTORY_NEG = 'history_neg'  # 负反馈历史记录column名称
C_HISTORY_POS_TAG = 'history_pos_tag'  # 用于记录一个交互列表是正反馈1还是负反馈0

# # DataProcessor/feed_dict
X = 'X'
Y = 'Y'
K_IF_TRAIN = 'train'
K_DROPOUT = 'dropout'
K_SAMPLE_ID = 'sample_id'  # 在训练（验证、测试）集中，给每个样本编号。这是该column在data dict和feed dict中的名字。
