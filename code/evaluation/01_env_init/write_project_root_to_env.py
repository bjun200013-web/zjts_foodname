import os
import platform
import subprocess
import sys

def check_and_add_to_linux_config(config_file, export_line):
    """检查Linux配置文件,如果需要,则添加路径。"""
    full_path = os.path.expanduser(config_file)
    if not os.path.exists(full_path):
        print(f"配置文件 {full_path} 不存在,将跳过。")
        return False, f"配置文件不存在: {config_file}"

    # 读取文件内容,检查是否已存在该路径
    try:
        with open(full_path, 'r') as f:
            # 使用 os.getcwd() 来匹配,因为 export_line 可能包含变量
            if os.getcwd() in f.read():
                print(f"路径已存在于 {config_file} 中,无需操作。")
                return True, "已存在"
    except Exception as e:
        print(f"读取文件 {config_file} 时出错: {e}")
        return False, "读取文件失败"

    # 如果不存在,则追加到文件末尾
    try:
        with open(full_path, 'a') as f:
            f.write("\n# Added by python script to include local modules\n")
            f.write(f"{export_line}\n")
        print(f"成功将路径写入 {config_file}。")
        return True, "写入成功"
    except Exception as e:
        print(f"写入文件 {config_file} 时出错: {e}")
        return False, "写入文件失败"

def handle_windows(current_path):
    """尝试在Windows上设置永久环境变量。"""
    print("检测到 Windows 系统。")
    print("正在尝试使用 `setx` 命令修改系统环境变量...")
    print("这可能需要管理员权限。\n")
    
    # 使用 /m 将变量设置为系统级别。如果只想为当前用户设置,请移除 /m
    command = f'setx PYTHONPATH "%PYTHONPATH%;{current_path}" /m'
    
    try:
        # 运行命令。check=True 会在命令失败时抛出异常。
        # capture_output=True 可以捕获标准输出和标准错误。
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ 命令执行成功！")
        print(result.stdout)
        print("\n请注意：您需要 **重新启动** 您的终端（CMD/PowerShell）才能使更改生效。")
    except subprocess.CalledProcessError as e:
        print("❌ 命令执行失败！")
        print("错误信息:", e.stderr)
        print("\n失败原因很可能是权限不足。")
        print("请手动 **以管理员身份** 打开一个新的命令提示符（CMD）或 PowerShell,然后执行以下命令：\n")
        print(f"    {command}\n")
    except FileNotFoundError:
        print("❌ 错误：`setx` 命令未找到。此工具在较旧的 Windows 版本中可能不可用。")
        print("请参考之前版本中的手动图形界面方法进行设置。")


def handle_linux(current_path):
    """在Linux上将路径添加到shell配置文件中。"""
    print("检测到 Linux 系统。")
    print("正在尝试将路径添加到您的 shell 配置文件中...\n")
    
    shell = os.environ.get("SHELL", "")
    export_line = f'export PYTHONPATH="{current_path}:$PYTHONPATH"'
    config_files = []

    if "bash" in shell:
        config_files.append("~/.bashrc")
    if "zsh" in shell:
        config_files.append("~/.zshrc")
    
    # 作为备选,总是尝试 .profile
    if not config_files or "~/.profile" not in config_files:
         config_files.append("~/.profile")

    success_count = 0
    for config_file in config_files:
        print(f"--- 正在处理 {config_file} ---")
        success, reason = check_and_add_to_linux_config(config_file, export_line)
        if success and reason == "写入成功":
             success_count += 1
        print("-" * (20 + len(config_file)))
        
    print("\n操作完成。")
    if success_count > 0:
        print("✅ 成功修改了您的 shell 配置文件。")
        print("\n\n*** 关键步骤 ***")
        print("为了让更改在 **当前** 终端会话中生效,您必须执行以下命令：\n")
        # 优先建议最可能的文件
        source_file = ""
        if "~/.bashrc" in config_files and "bash" in shell:
            source_file = "~/.bashrc"
        elif "~/.zshrc" in config_files and "zsh" in shell:
            source_file = "~/.zshrc"
        else:
            source_file = config_files[0]
            
        print(f"    source {source_file}\n")
        print("或者,您可以直接打开一个新的终端窗口,更改将自动加载。")
    else:
        print("未对任何配置文件进行新的修改。")


def main():
    """主函数,检测系统并执行相应操作。"""
    current_path = os.getcwd()
    print(f"目标路径: {current_path}\n")

    system = platform.system()

    if system == "Windows":
        handle_windows(current_path)
    elif system == "Linux":
        handle_linux(current_path)
    else:
        print(f"不支持的操作系统: {system}")
        print("本脚本只能在 Windows 和 Linux 上运行。")

if __name__ == "__main__":
    main()