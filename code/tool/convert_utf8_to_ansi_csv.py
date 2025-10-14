import os
import glob
import re

def clean_special_characters(text):
    """
    清理或替换特殊字符
    """
    # 替换4字节的UTF-8字符（如表情符号）
    cleaned_text = re.sub(r'[^\u0000-\uFFFF]', '?', text)
    return cleaned_text

def convert_with_cleaning(source_folder, output_folder, target_encoding='gbk'):
    """
    转换编码前先清理特殊字符，输出到新文件夹
    """
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 查找所有CSV文件
    csv_files = glob.glob(os.path.join(source_folder, "*.csv"))
    
    if not csv_files:
        print(f"在文件夹 {source_folder} 中未找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    success_count = 0
    
    for csv_file in csv_files:
        try:
            filename = os.path.basename(csv_file)
            output_path = os.path.join(output_folder, filename)
            
            # 读取UTF-8编码的文件
            with open(csv_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 清理特殊字符
            cleaned_content = clean_special_characters(content)
            
            # 写入目标编码
            with open(output_path, 'w', encoding=target_encoding, errors='replace') as f:
                f.write(cleaned_content)
            
            print(f"✓ 成功转换: {filename}")
            success_count += 1
            
        except Exception as e:
            print(f"✗ 转换失败 {filename}: {str(e)}")
    
    print(f"\n转换完成！成功转换 {success_count}/{len(csv_files)} 个文件")
    print(f"新文件保存在: {output_folder}")

# 使用示例
if __name__ == "__main__":
    source_folder = r'C:\Users\crwu\Downloads'
    output_folder = r'C:\Users\crwu\Downloads\aftertrans'
    
    # source_folder = r'D:\crwu\data\foodname_classification\before_encoding_trans'
    # output_folder = r'D:\crwu\data\foodname_classification\after_encoding_trans'
    if not os.path.exists(source_folder):
        print("源文件夹路径不存在！")
    else:
        convert_with_cleaning(source_folder, output_folder)