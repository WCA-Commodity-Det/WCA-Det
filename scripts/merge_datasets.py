import os
import shutil

# ========================================================
# ⚙️ 配置区
# ========================================================
# 原始数据集路径
DATA12_DIR = "D:/spsb/data_12/custom_dataset"
DATA6_DIR = "D:/spsb/data_6/custom_dataset_6classes"

# 融合后的全新数据集保存路径
MERGED_DIR = "D:/spsb/data_merged"

# 🌟 核心映射字典 (将两边的旧 ID 映射到全新的 0~11 ID)
# Data_12 保留的 9 个类: bsk1, bsk1_con, bsk1_wt, bss_large, fd_can, hs_can, kkkl_can, mnd_can, nfsq
# 它们在原来 data_12 的索引是: 0, 1, 2, 3, 4, 7, 9, 10, 11
MAP_DATA12 = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4,
    7: 5, 9: 6, 10: 7, 11: 8
}

# Data_6 保留的 3 个类: xb_can, xb_wt, xb
# 它们在原来 data_6 的索引是: 2, 3, 4
MAP_DATA6 = {
    2: 9, 3: 10, 4: 11
}

NEW_NAMES = [
    'bsk1', 'bsk1_con', 'bsk1_wt', 'bss_large', 'fd_can',
    'hs_can', 'kkkl_can', 'mnd_can', 'nfsq',
    'xb_can', 'xb_wt', 'xb'
]


# ========================================================
# 🚀 融合清洗逻辑
# ========================================================
def setup_directories():
    """创建全新的纯净目录结构"""
    if os.path.exists(MERGED_DIR):
        print(f"⚠️ 发现已存在的 {MERGED_DIR}，正在清理历史数据...")
        shutil.rmtree(MERGED_DIR)

    for split in ['train', 'val']:
        os.makedirs(os.path.join(MERGED_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(MERGED_DIR, 'labels', split), exist_ok=True)


def process_and_copy(src_base, split, class_map, prefix):
    """提取有效标签，重写 ID，并安全复制图片"""
    src_images = os.path.join(src_base, 'images', split)
    src_labels = os.path.join(src_base, 'labels', split)

    dst_images = os.path.join(MERGED_DIR, 'images', split)
    dst_labels = os.path.join(MERGED_DIR, 'labels', split)

    if not os.path.exists(src_images):
        return

    valid_count = 0

    for img_name in os.listdir(src_images):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        base_name = os.path.splitext(img_name)[0]
        lbl_name = base_name + ".txt"
        lbl_path = os.path.join(src_labels, lbl_name)

        # 新的文件名 (加上前缀，防止两个数据集的图片重名互相覆盖)
        new_img_name = f"{prefix}_{img_name}"
        new_lbl_name = f"{prefix}_{lbl_name}"

        new_boxes = []

        # 读取旧标签并转换
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        old_id = int(parts[0])
                        # 只有在保留名单里的商品，才会被写入新文件
                        if old_id in class_map:
                            new_id = class_map[old_id]
                            new_boxes.append(f"{new_id} {' '.join(parts[1:])}\n")

        # 将图片和转换后的标签放入新家
        if new_boxes:
            # 写入新的 txt 标签
            with open(os.path.join(dst_labels, new_lbl_name), 'w') as f:
                f.writelines(new_boxes)
            # 复制图片
            shutil.copy(os.path.join(src_images, img_name), os.path.join(dst_images, new_img_name))
            valid_count += 1

    print(f"✅ [{prefix} - {split}] 成功迁移并清洗了 {valid_count} 张包含有效商品的图片。")


def create_yaml():
    """生成新数据集的终极配置文件"""
    yaml_content = f"""
train: {MERGED_DIR}/images/train
val: {MERGED_DIR}/images/val

nc: 12
names: {NEW_NAMES}
"""
    yaml_path = os.path.join(MERGED_DIR, "dataset_merged.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content.strip())
    print(f"\n📄 融合版配置文件已生成: {yaml_path}")


def main():
    print("🚀 开始执行工业级数据大融合...")
    setup_directories()

    # 处理 Data_12 (打上 data12 前缀)
    process_and_copy(DATA12_DIR, 'train', MAP_DATA12, 'data12')
    process_and_copy(DATA12_DIR, 'val', MAP_DATA12, 'data12')

    # 处理 Data_6 (打上 data6 前缀)
    process_and_copy(DATA6_DIR, 'train', MAP_DATA6, 'data6')
    process_and_copy(DATA6_DIR, 'val', MAP_DATA6, 'data6')

    create_yaml()
    print("\n🎉 全部数据清洗合并完毕！现在你可以用 dataset_merged.yaml 去训练 12 分类的终极大模型了！")


if __name__ == '__main__':
    main()
