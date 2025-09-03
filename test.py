import os
import shutil
import argparse

def delete_all_except_images(folder_path):
    """
    遍历指定文件夹下的所有子文件夹，删除每个子文件夹中除了'images'之外的所有文件和文件夹
    
    Args:
        folder_path (str): 要处理的根文件夹路径
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在")
        return
    
    # 遍历根文件夹下的所有子文件夹
    for subdir in os.listdir(folder_path):
        subdir_path = os.path.join(folder_path, subdir)
        
        # 确保是文件夹而不是文件
        if os.path.isdir(subdir_path):
            print(f"\n处理子文件夹: {subdir}")
            
            # 遍历子文件夹中的所有内容
            for item in os.listdir(subdir_path):
                item_path = os.path.join(subdir_path, item)
                
                # 如果是'images'文件夹，则保留
                if os.path.isdir(item_path) and item.lower() == 'images':
                    print(f"  保留文件夹: {item}")
                    continue
                
                # 删除其他所有文件和文件夹
                try:
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.remove(item_path)
                        print(f"  已删除文件: {item}")
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        print(f"  已删除文件夹: {item}")
                except Exception as e:
                    print(f"  删除失败 {item}: {e}")
    
    print("\n处理完成！")

def safe_delete_all_except_images(folder_path, dry_run=False):
    """
    安全版本：先显示将要删除的内容，需要确认后再执行
    
    Args:
        folder_path (str): 要处理的根文件夹路径
        dry_run (bool): 如果为True，只显示将要删除的内容而不实际删除
    """
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在")
        return
    
    items_to_delete = []
    items_to_keep = []
    
    # 首先收集所有要删除和保留的项目信息
    for subdir in os.listdir(folder_path):
        subdir_path = os.path.join(folder_path, subdir)
        
        if os.path.isdir(subdir_path):
            print(f"\n检查子文件夹: {subdir}")
            
            for item in os.listdir(subdir_path):
                item_path = os.path.join(subdir_path, item)
                
                if os.path.isdir(item_path) and item.lower() == 'images':
                    items_to_keep.append(item_path)
                    print(f"  将保留: {item}/ (文件夹)")
                else:
                    items_to_delete.append(item_path)
                    item_type = "文件" if os.path.isfile(item_path) else "文件夹"
                    print(f"  将删除: {item} ({item_type})")
    
    if not items_to_delete:
        print("\n没有找到需要删除的项目")
        return
    
    if dry_run:
        print(f"\n干燥运行完成，共找到:")
        print(f"  - 将保留: {len(items_to_keep)} 个项目")
        print(f"  - 将删除: {len(items_to_delete)} 个项目")
        return
    
    # 询问用户确认
    print(f"\n总结:")
    print(f"  - 将保留: {len(items_to_keep)} 个项目")
    print(f"  - 将删除: {len(items_to_delete)} 个项目")
    
    confirmation = input("\n确认要执行删除操作吗？此操作不可逆！(y/N): ")
    
    if confirmation.lower() in ['y', 'yes']:
        for item_path in items_to_delete:
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path)
                    print(f"已删除文件: {os.path.basename(item_path)}")
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"已删除文件夹: {os.path.basename(item_path)}")
            except Exception as e:
                print(f"删除失败 {os.path.basename(item_path)}: {e}")
        print("\n处理完成！")
    else:
        print("操作已取消")

def main():
    parser = argparse.ArgumentParser(description='删除子文件夹中除了images文件夹之外的所有文件和文件夹')
    parser.add_argument('--folder_path', help='要处理的根文件夹路径')
    parser.add_argument('--safe', action='store_true', help='使用安全模式（会要求确认）')
    parser.add_argument('--dry-run', action='store_true', help='只显示将要删除的内容而不实际执行')
    
    args = parser.parse_args()
    
    if args.safe or args.dry_run:
        safe_delete_all_except_images(args.folder_path, args.dry_run)
    else:
        # 直接执行模式也会要求确认
        confirmation = input("直接执行模式：确认要删除除了images之外的所有文件和文件夹吗？此操作不可逆！(y/N): ")
        if confirmation.lower() in ['y', 'yes']:
            delete_all_except_images(args.folder_path)
        else:
            print("操作已取消")

if __name__ == "__main__":
    main()