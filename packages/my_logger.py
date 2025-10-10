from datetime import datetime
import logging
import os


# def init_logger(logger_name):
#     # 日志文件名带时间戳，防止覆盖
#     log_time = datetime.now().strftime("%Y%m%d%H%M%S")
#     log_file = f"{logger_name}_{log_time}.log"
#     print(f"日志文件: {os.path.abspath(log_file)}")
#     logging.basicConfig(
#         filename=log_file,
#         filemode="a",
#         format="%(asctime)s %(levelname)s: %(message)s",
#         datefmt="%Y-%m-%d %H:%M:%S",
#         level=logging.INFO,
#     )
#     logger = logging.getLogger(logger_name)
#     return logger

# my_logger.py
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import inspect
# from constants import PROJECT_ROOT


class CustomLogger:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CustomLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self, log_dir=None, script_name=None, level=logging.INFO):
        if not self._initialized:
            current_file = os.path.abspath(__file__)
            packages_dir = os.path.dirname(current_file)
            project_root = os.path.dirname(packages_dir)
            self.log_dir = log_dir or os.path.join(project_root, "logs")
            self.script_name = script_name or self._get_calling_script_name()
            self.level = level
            self.logger = None
            self._setup_logger()
            self._initialized = True

    def _get_calling_script_name(self):
        """获取调用此日志模块的脚本名称"""
        try:
            # 获取调用栈，找到第一个不是本文件的调用者
            stack = inspect.stack()
            for frame_info in stack:
                filename = frame_info.filename
                if filename != __file__:
                    # 返回不带路径和扩展名的纯脚本名
                    return Path(filename).stem
            return "unknown_script"
        except:
            return "unknown_script"

    def _setup_logger(self):
        """设置日志配置"""
        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)

        # 生成日志文件名：日期时间_脚本名称.log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{timestamp}_{self.script_name}.log"
        log_filepath = os.path.join(self.log_dir, log_filename)

        # 创建logger
        self.logger = logging.getLogger(self.script_name)
        self.logger.setLevel(self.level)

        # 避免重复添加handler
        if not self.logger.handlers:
            # 文件处理器
            file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
            file_handler.setLevel(self.level)

            # 控制台处理器
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)

            # 格式化器
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

        # 记录日志初始化信息
        self.logger.info(f"日志系统初始化完成，日志文件: {log_filepath}")
        self.logger.info(f"脚本名称: {self.script_name}")

    def get_logger(self):
        """获取logger实例"""
        return self.logger

    def update_script_name(self, new_script_name):
        """更新脚本名称（用于特殊情况）"""
        self.script_name = new_script_name
        # 重新设置logger
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        self._setup_logger()


# 全局日志实例
_global_logger = None


def setup_logging(log_dir=None, script_name=None, level=logging.INFO):
    """设置全局日志配置"""
    global _global_logger
    _global_logger = CustomLogger(log_dir, script_name, level)
    return _global_logger.get_logger()


def get_logger(name=None):
    """获取logger实例"""
    global _global_logger
    if _global_logger is None:
        # 如果没有初始化，使用默认配置
        _global_logger = CustomLogger()

    if name:
        # 如果指定了名称，返回该名称的logger（但使用相同的文件handler）
        logger = logging.getLogger(name)
        # 确保这个logger也使用相同的配置
        if not logger.handlers:
            # 复制全局logger的handlers
            for handler in _global_logger.logger.handlers:
                logger.addHandler(handler)
            logger.setLevel(_global_logger.level)
        return logger
    else:
        return _global_logger.get_logger()
