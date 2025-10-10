from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from requests.exceptions import Timeout, RequestException
import httpx

from packages.my_logger import get_logger

logger = get_logger()


from packages.file_deal import *


@retry(
    stop=stop_after_attempt(3),  # 最多重试3次
    wait=wait_exponential(
        multiplier=1, min=15, max=30
    ),  # 指数退避，等待时间在4-10秒之间
    retry=retry_if_exception_type(
        (Timeout, RequestException, httpx.TimeoutException)
    ),  # 只对超时和请求异常进行重试
    reraise=True,  # 重试失败后抛出原始异常
)
def call_openai_with_timeout(client, model, messages, timeout=30):
    """
    带超时和重试的OpenAI API调用

    Args:
        client: OpenAI客户端实例
        model: 模型名称
        messages: 消息列表
        timeout: 超时时间（秒）
    """
    try:
        return client.chat.completions.create(
            model=model, messages=messages, timeout=timeout, temperature=0  # 设置请求超时时间
        )
    except Exception as e:
        logger.info(f"API调用出错 (将重试): {str(e)}")
        raise  # 抛出异常以触发重试


if __name__ == "__main__":
    pass
