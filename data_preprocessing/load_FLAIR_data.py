import json
import os
import random
import logging
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
def load_FLAIR_data(datadir,
                    client_number=10):
    data_info_path = os.path.join(datadir, 'labels_and_metadata.json')
    with open(data_info_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    user_details = get_user_details(data)
    user_details_single_label = create_single_label_details(user_details)
    excluded_labels = [
        'light',
        'celebration',
        'fire',
        'games',
        'recreation',
        'music',
        'religion'
    ]
    label2id = {
        'equipment': 0,
        'structure': 1,
        'material': 2,
        'outdoor': 3,
        'food': 4,
        'plant': 5,
        'animal': 6,
        'liquid': 7,
        'art': 8,
        'interior_room': 9,
        'light': 10,
        'recreation': 11,
        'celebration': 12,
        'fire': 13,
        'music': 14,
        'games': 15,
        'religion': 16
    }
    data_image_path = os.path.join(datadir, 'small_images')
    user_details_10cls = create_filtered_user_details(user_details_single_label, excluded_labels)
    train_data_10cls, val_data_10cls, test_data_10cls = split_user_details_by_partition(user_details_10cls)
    balanced_test_set = create_balanced_test_set(test_data_10cls, samples_per_class=100)
    test_images, test_labels = load_balanced_test_images(balanced_test_set, label2id, data_image_path)
    # matching_users = find_users_by_image_count(train_data_10cls, min_images=50, max_images=5000, sort_by_count=True)
    final_samples = stratified_sample_by_size(train_data_10cls, client_number)
    # final_samples = sample_balanced_clients_by_count(train_data_10cls, label2id, client_number)
    client_X_train_dict, client_y_train_dict, client_fine_labels_dict = {}, {}, {}
    for idx, user_id in enumerate(final_samples):
        logging.info(f"Loading client {idx} data...")
        images_array, label_ids, fine_labels = load_client_data(user_id, train_data_10cls, label2id, data_image_path, image_size=(128, 128))
        client_X_train_dict[idx] = images_array
        client_y_train_dict[idx] = label_ids
    get_label_distribution(client_y_train_dict, test_labels)
    return client_X_train_dict, client_y_train_dict, test_images, test_labels

def get_user_details(data):
    user_details = {}
    for idx, item in enumerate(data):
        user_id = item['user_id']
        partition = item['partition']
        image_id = item['image_id']
        label = item['labels']
        final_label = item['fine_grained_labels']
        
        # 如果是新用户，初始化数据结构
        if user_id not in user_details:
            user_details[user_id] = {
                'train': {'image_info': [],  'num': 0},
                'test': {'image_info': [],  'num': 0},
                'val': {'image_info': [], 'num': 0},
                'total_images': 0
            }
        
        # 更新数据
        user_details[user_id][partition]['image_info'].append((image_id, label, final_label))
        user_details[user_id][partition]['num'] += 1
        user_details[user_id]['total_images'] += 1
    return user_details


def create_single_label_details(user_details):
    """
    为每张图片随机选择一个粗粒度标签和一个细粒度标签
    
    Args:
        user_details: 原始用户详情字典
        random_seed: 随机种子，确保结果可重复
    
    Returns:
        dict: 单标签用户详情字典
    """
    # 设置随机种子
    
    # 创建新的字典来存储单标签数据
    user_details_single_label = {}
    
    # 遍历每个用户
    for user_id, user_info in user_details.items():
        # 初始化新用户的数据结构
        user_details_single_label[user_id] = {
            'train': {'image_info': [], 'num': 0},
            'test': {'image_info': [], 'num': 0},
            'val': {'image_info': [], 'num': 0},
            'total_images': 0
        }
        
        # 处理每个分区的数据
        for partition in ['train', 'test', 'val']:
            for img_data in user_info[partition]['image_info']:
                # 随机选择标签
                coarse_labels = img_data[1]  # 假设是标签列表
                fine_labels = img_data[2]    # 假设是标签列表
                

                coarse_label = coarse_labels[0]
                fine_label = fine_labels[0]
                
                # 更新数据，保存(image_id, coarse_label, fine_label)的元组
                user_details_single_label[user_id][partition]['image_info'].append(
                    (img_data[0], coarse_label, fine_label)
                )
                user_details_single_label[user_id][partition]['num'] += 1
                user_details_single_label[user_id]['total_images'] += 1
    
    return user_details_single_label


def create_filtered_user_details(user_details, excluded_labels):
    """
    创建一个新的用户详情字典，去除包含指定标签的样本
    
    Args:
        user_details: 原始用户详情字典
        excluded_labels: 要排除的标签列表
    
    Returns:
        dict: 新的用户详情字典
    """
    user_details_filtered = {}
    excluded_labels = set(excluded_labels)  # 转换为集合以提高查找效率
    
    # 统计信息
    total_original_images = 0
    total_filtered_images = 0
    
    # 处理每个用户
    for user_id, user_info in user_details.items():
        # 初始化新用户的数据结构
        user_details_filtered[user_id] = {
            'train': {'image_info': [], 'num': 0},
            'test': {'image_info': [], 'num': 0},
            'val': {'image_info': [], 'num': 0},
            'total_images': 0
        }
        
        user_original_images = 0
        user_filtered_images = 0
        
        # 处理每个分区
        for partition in ['train', 'test', 'val']:
            for idx, img_data in enumerate(user_info[partition]['image_info']):
                user_original_images += 1
                total_original_images += 1
                
                # 检查标签是否在排除列表中
                coarse_label = img_data[1]
                if coarse_label not in excluded_labels:
                    # 保留这个样本
                    user_details_filtered[user_id][partition]['image_info'].append(img_data)

                    user_details_filtered[user_id][partition]['num'] += 1
                    user_details_filtered[user_id]['total_images'] += 1
                    user_filtered_images += 1
                    total_filtered_images += 1
        
        # 如果用户没有任何图片了，删除这个用户
        if user_details_filtered[user_id]['total_images'] == 0:
            del user_details_filtered[user_id]
    return user_details_filtered

def split_user_details_by_partition(user_details):
    """
    将用户详情字典按分区分开
    
    Args:
        user_details: 原始用户详情字典
    
    Returns:
        tuple: (train_data, val_data, test_data) 三个字典
    """
    train_data = {}
    val_data = {}
    test_data = {}
    
    # 遍历每个用户
    for user_id, user_info in user_details.items():
        # 为每个分区创建用户数据
        if user_info['train']['num'] > 0:
            train_data[user_id] = {
                'image_ids': user_info['train']['image_info'],
                'num': user_info['train']['num']
            }
            
        if user_info['val']['num'] > 0:
            val_data[user_id] = {
                'image_ids': user_info['val']['image_info'],
                'num': user_info['val']['num']
            }
            
        if user_info['test']['num'] > 0:
            test_data[user_id] = {
                'image_ids': user_info['test']['image_info'],
                'num': user_info['test']['num']
            }
    
    return train_data, val_data, test_data
def stratified_sample_by_size(train_data_10cls, client_num):
    """
    根据数据量分层采样用户
    
    Args:
        train_data_10cls: 用户数据字典，格式为 {user_id: data_list}
        client_num: 需要的总客户端数量
    
    Returns:
        list: 采样得到的用户ID列表
    """
    # 根据数据量对用户分组
    large_users = []    # >2000
    medium_users = []   # 500-2000
    small_users = []    # <500
    random.seed(0)
    for user_id, data in train_data_10cls.items():
        data_size = len(data['image_ids'])
        if data_size > 2000:
            large_users.append((user_id, data_size))
        elif data_size >= 500:
            medium_users.append((user_id, data_size))
        elif data_size >= 50 and data_size < 500:
            small_users.append((user_id, data_size))
    
    
    # 随机采样
    sampled_large = random.sample(large_users, min(int( client_num*0.2), len(large_users)))
    sampled_medium = random.sample(medium_users, min(int(client_num*0.5), len(medium_users)))
    remaining_count = client_num - len(sampled_large) - len(sampled_medium)
    sampled_small = random.sample(small_users, min(remaining_count, len(small_users)))
    
    # 合并结果并打印详情
    final_samples = []
    for user_id, size in sampled_large:
        final_samples.append(user_id)
        
    for user_id, size in sampled_medium:
        final_samples.append(user_id)
        
    for user_id, size in sampled_small:
        final_samples.append(user_id)
    
    return final_samples

def find_users_by_image_count(user_details, min_images=0, max_images=float('inf'), 
                             sort_by_count=True):
    """
    找到数据量在指定区间内的所有用户
    
    Args:
        user_details: 用户详细信息字典
        min_images: 最小图片数量（包含）
        max_images: 最大图片数量（包含）
        sort_by_count: 是否按图片数量排序
    
    Returns:
        list: 包含符合条件的用户信息的列表，每个元素是(user_id, image_count)元组
    """
    matching_users = []
    
    for user_id, user_info in user_details.items():
        image_count = user_info['num']
        
        # 检查是否在指定范围内
        if min_images <= image_count <= max_images:
            matching_users.append((user_id, image_count))
    
    # 按图片数量排序（如果需要）
    if sort_by_count:
        matching_users.sort(key=lambda x: x[1], reverse=True)
    return matching_users

def create_balanced_test_set(test_data, samples_per_class=500):
    """
    从测试数据中每个类别采样固定数量的样本
    
    Args:
        test_data: 测试数据字典
        samples_per_class: 每个类别要采样的数量
    
    Returns:
        list: 包含(image_id, coarse_label, fine_label)的列表
    """
    # 按类别收集所有样本
    class_samples = {}
    
    # 遍历所有用户的测试数据
    for user_info in test_data.values():
        for img_id, coarse_label, fine_label in user_info['image_ids']:
            if coarse_label not in class_samples:
                class_samples[coarse_label] = []
            class_samples[coarse_label].append((img_id, coarse_label, fine_label))
    
    # 从每个类别中随机采样
    balanced_test_set = []
    
    for label, samples in class_samples.items():
        # 确定实际可采样的数量
        actual_samples = min(samples_per_class, len(samples))
        selected_samples = random.sample(samples, actual_samples)
        balanced_test_set.extend(selected_samples)

    
    # 统计每个类别的实际样本数
    label_counts = {}
    fine_label_counts = {}
    for _, coarse_label, fine_label in balanced_test_set:
        label_counts[coarse_label] = label_counts.get(coarse_label, 0) + 1
        fine_label_counts[fine_label] = fine_label_counts.get(fine_label, 0) + 1
    
    # 随机打乱数据
    random.shuffle(balanced_test_set)
    
    return balanced_test_set
def load_balanced_test_images(balanced_test_data, label2id, image_dir):
    """
    加载平衡测试集的图片数据和标签
    
    Args:
        balanced_test_data: 包含(image_id, coarse_label, fine_label)的列表
        label2id: 标签到ID的映射字典
        image_dir: 图片文件夹路径
    
    Returns:
        tuple: (images_array, labels_array)
        - images_array: shape为(N, H, W, C)的numpy数组
        - labels_array: shape为(N,)的numpy数组，包含粗粒度标签的ID
    """
    images = []
    labels = []
    failed_images = []
    
    for img_id, coarse_label, _ in tqdm(balanced_test_data, desc="Loading Test images"):
        img_path = os.path.join(image_dir, f"{img_id}.jpg")
        try:
            # 加载并转换图片
            img = Image.open(img_path)
            img = img.convert('RGB')  # 确保是RGB格式
            img_array = np.array(img)
            
            # 使用label2id转换标签
            label_id = label2id[coarse_label]
            
            # 添加到列表
            images.append(img_array)
            labels.append(label_id)
        except Exception as e:
            print(f"\n加载图片 {img_id} 失败: {str(e)}")
            failed_images.append(img_id)
            continue
    
    if len(images) == 0:
        raise ValueError("Warning: loading no images!!!!!!!!!!!!")
        
    # 转换为numpy数组
    images = np.stack(images, axis=0)
    labels = np.array(labels)
    
    
    return images, labels

def sample_users(matching_users, n_samples):
    """
    从matching_users中均匀采样n个user_id
    
    Args:
        matching_users: 用户ID列表
        n_samples: 需要采样的用户数量
    
    Returns:
        list: 采样得到的用户ID列表
    """
    import random
    
    if n_samples > len(matching_users):
        print(f"警告：请求的采样数量({n_samples})大于总用户数({len(matching_users)})，将返回所有用户")
        return matching_users
    
    # 使用random.sample进行无重复随机采样
    sampled_users = random.sample(matching_users, n_samples)
    
    return sampled_users

def load_client_data(client_id, train_data_dict, label2id, image_dir, image_size=(32, 32)):
    """
    加载指定client的训练数据，将图片调整为指定尺寸，并将标签转换为数字ID
    
    Args:
        client_id: 客户端ID
        train_data_dict: 训练数据字典
        label2id: 标签到ID的映射字典
        image_dir: 图片文件夹路径
        image_size: 目标图片尺寸，默认(32, 32)
    
    Returns:
        tuple: (images_array, label_ids, fine_labels)
        - images_array: shape为(N, H, W, C)的numpy数组
        - label_ids: 粗粒度标签的数字ID列表
        - fine_labels: 细粒度标签列表
    """
    if client_id not in train_data_dict:
        raise ValueError(f"Client ID {client_id} 不存在于训练数据中")
        
    client_data = train_data_dict[client_id]
    images = []
    label_ids = []
    fine_labels = []
    failed_images = []
    
    for img_id, coarse_label, fine_label in tqdm(client_data['image_ids']):
        img_path = os.path.join(image_dir, f"{img_id}.jpg")
        try:
            # 加载并转换图片
            img = Image.open(img_path)
            img = img.convert('RGB')
            # 调整图片尺寸
            img = img.resize(image_size, Image.Resampling.LANCZOS)
            img_array = np.array(img)
            
            # 添加到列表
            images.append(img_array)
            label_ids.append(label2id[coarse_label])
            fine_labels.append(fine_label)
        except Exception as e:
            print(f"\n加载图片 {img_id} 失败: {str(e)}")
            failed_images.append(img_id)
            continue
            
    # 转换为numpy数组
    images_array = np.stack(images, axis=0)
    label_ids = np.array(label_ids)
    
    if failed_images:
        print(f"\n加载失败的图片数：{len(failed_images)}")
        print("失败的图片ID：", failed_images[:5], "..." if len(failed_images) > 5 else "")
    
    
    return images_array, label_ids, fine_labels


def get_label_distribution(client_y_train_dict, test_labels):
    """
    统计训练集和测试集的标签分布
    
    Args:
        client_y_train_dict: 客户端训练标签字典 {client_id: labels_array}
        test_labels: 测试集标签数组
    
    Returns:
        dict: 包含所有分布统计的字典
    """
    import numpy as np
    from collections import Counter
    
    # 统计每个客户端的分布
    client_distributions = {}
    all_train_labels = []
    
    for client_id, labels in client_y_train_dict.items():
        # 统计每个客户端的标签分布并转换为普通整数
        unique_labels, counts = np.unique(labels, return_counts=True)
        client_distributions[client_id] = {int(label): int(count) 
                                         for label, count in zip(unique_labels, counts)}
        all_train_labels.extend(labels)
    
    # 统计整体训练集的分布
    train_counts = Counter(all_train_labels)
    train_distribution = {int(label): int(count) 
                         for label, count in train_counts.items()}
    
    # 统计测试集的分布
    test_counts = Counter(test_labels)
    test_distribution = {int(label): int(count) 
                        for label, count in test_counts.items()}
    print("=== Label distribution of each client ===")
    for client_id, dist in client_distributions.items():
        print("Client {}:{}".format(client_id, dist))

    print("=== Training set label distribution === {}".format(train_distribution))

    print("=== Test set label distribution === {}".format(test_distribution))
    

def sample_balanced_clients_by_count(train_data_10cls, label2id, target_client_num):
    """
    采样客户端，使得整体训练集中每个类别的样本数尽量接近
    
    Args:
        train_data_10cls: 用户数据字典 {user_id: data_list}
        label2id: 标签到ID的映射
        target_client_num: 目标客户端数量
    
    Returns:
        list: 采样得到的客户端ID列表
    """
    import numpy as np
    from collections import Counter
    
    # 计算每个用户的标签分布
    user_distributions = {}
    for user_id, data in train_data_10cls.items():
        labels = [item[1] for item in data['image_ids']]  # 假设item[1]是粗粒度标签
        label_ids = [label2id[label] for label in labels]
        user_distributions[user_id] = Counter(label_ids)
    
    def evaluate_distribution(selected_users):
        """评估选定用户集合的标签分布均衡性"""
        total_dist = Counter()
        for user_id in selected_users:
            total_dist.update(user_distributions[user_id])
        
        if not total_dist:
            return float('inf')
            
        # 计算类别间样本数的差异
        counts = list(total_dist.values())
        return max(counts) - min(counts)  # 返回最大类别和最小类别的样本数差距
    
    # 贪婪选择：每次选择能使类别样本数最均衡的客户端
    selected_users = []
    remaining_users = list(user_distributions.keys())
    
    while len(selected_users) < target_client_num and remaining_users:
        best_diff = float('inf')
        best_user = None
        
        # 尝试添加每个剩余用户，选择最优的
        for user_id in remaining_users:
            temp_selected = selected_users + [user_id]
            diff = evaluate_distribution(temp_selected)
            
            if diff < best_diff:
                best_diff = diff
                best_user = user_id
        
        if best_user:
            selected_users.append(best_user)
            remaining_users.remove(best_user)
    
    # 打印最终的分布情况
    final_distribution = Counter()
    for user_id in selected_users:
        final_distribution.update(user_distributions[user_id])
    
    print("\n=== 采样结果 ===")
    print(f"选择的客户端数量: {len(selected_users)}")
    print("\n各类别样本数:")
    for label, count in sorted(final_distribution.items()):
        print(f"类别 {label}: {count}")
    
    # 计算最大差异
    counts = list(final_distribution.values())
    max_diff = max(counts) - min(counts)
    print(f"\n最大类别差异: {max_diff}")
    print(f"平均每类样本数: {sum(counts)/len(counts):.1f}")
    
    return selected_users

def set_random(seed):
    torch.manual_seed(seed)      
    torch.cuda.manual_seed(seed) 
    np.random.seed(seed)         
    random.seed(seed)           
    torch.backends.cudnn.benchmark = False   
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    client_X_train_dict, client_y_train_dict, test_images, test_labels = load_FLAIR_data('/data/zqy/FLAIR', 10)
    # 使用示例：
    
    set_random(10)


