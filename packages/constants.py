import inspect
import os

def get_function_location():
    """获取函数所在的文件位置"""
    # 获取当前函数的帧信息
    frame = inspect.currentframe()
    try:
        # 获取函数定义所在的文件路径
        file_path = inspect.getfile(frame)
        # 转换为绝对路径
        abs_path = os.path.abspath(file_path)
        return abs_path
    finally:
        # 重要：删除帧引用以避免循环引用
        del frame

PROJECT_ROOT = os.path.dirname(os.path.dirname(get_function_location()))
EVAL_DATA_IMAGE_ROOT = os.path.join(PROJECT_ROOT, 'data/evaluation/Evaluation_data_0528')
EVAL_DATA_EXCEL_PATH = os.path.join(PROJECT_ROOT, 'data/evaluation/test_data_excel/Evaluation_Data_0530_comments_by_hht.xlsx')
EVAL_RES_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data/evaluation/eval_res_of_llm')
LLM_SCORE_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data/evaluation/llm_score')

GPT_MODEL_NAME =  "gpt-4o-2024-11-20"
GEMINI_MODEL_NAME = "gemini-2.5-flash"
CLAUDE_MODEL_NAME = "claude-3-7-sonnet-20250219"
DEEPSEEK_MODEL_NAME = 'deepseek-v3' # 废弃
QWEN_V2_5_MODEL_NAME = 'qwen2.5-vl-72b-instruct'
QWEN_V3_MODEL_NAME = '' # 青云还没上线

API_URL = "https://api.qingyuntop.top/v1" # 青云的API
API_KEY_DISCOUNT = "sk-mFtES1U9ZQqCpLuoODW1cH6XyDOZEMcfKzBNSq9ROBEBV5YW" # 青云的限时特价令牌
API_KEY_DEFAULT = "sk-QfNYPtgiRGWlk7MzGMqX0IktddTGfvbGIxEEme9N7JOKy8Ak" # 青云的default令牌

if __name__ == '__main__':
    print(EVAL_DATA_EXCEL_PATH)