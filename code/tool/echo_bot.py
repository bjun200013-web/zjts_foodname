import os
import subprocess
import time
import lark_oapi as lark
from lark_oapi.api.im.v1 import *
import json

from packages.constants import (
    CODE_DIR,
    EVAL_RES_OUTPUT_PATH,
)

# os.environ["BASE_DOMAIN"] = r'https://open.feishu.cn'
# os.environ["APP_ID"] = r'cli_a87b1a5de31bd00e'
# os.environ["APP_SECRET"] = r'pRfhRO2MbZvK0Y3oZW2erbYbwOgzetOK'

# ============================================================================
# 调用方式
# BASE_DOMAIN=https://open.feishu.cn APP_ID=cli_a87b1a5de31bd00e APP_SECRET=pRfhRO2MbZvK0Y3oZW2erbYbwOgzetOK python /date0/crwu/zjts_foodname/code/tool/echo_bot.py
# ============================================================================

def run_scoring_batch(input_file, log_file):
    """
    运行单个评分批次
    """
    cmd = [
        "python",
        os.path.join(CODE_DIR, "evaluation/03_merge_eval_result/parallel.py"),
        "--eval_only",
        True,
        "--eval_input",
        input_file,
        "--max_test_img_num",
        10,
    ]

    with open(log_file, "w") as log:
        subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)


def get_resource_file(file_key):
    # 构造请求对象
    request: GetFileRequest = GetFileRequest.builder().file_key(file_key).build()

    # 发起请求
    response: GetFileResponse = client.im.v1.file.get(request)

    # 处理失败返回
    if not response.success():
        lark.logger.error(
            f"client.im.v1.file.get failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}"
        )
        return

    # 处理业务结果
    f = open(os.path.join(EVAL_RES_OUTPUT_PATH, "{response.file_name}"), "wb")
    f.write(response.file.read())
    f.close()


def handle_recived_message(message):
    print(f"received message: {message}")


def send_group_text_reply(data, content):
    request: ReplyMessageRequest = (
        ReplyMessageRequest.builder()
        .message_id(data.event.message.message_id)
        .request_body(
            ReplyMessageRequestBody.builder().content(content).msg_type("text").build()
        )
        .build()
    )
    # 使用OpenAPI回复消息
    # Reply to messages using send OpenAPI
    # https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/im-v1/message/reply
    response: ReplyMessageResponse = client.im.v1.message.reply(request)
    if not response.success():
        raise Exception(
            f"client.im.v1.message.reply failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}"
        )


def send_txt_message_p2p(chat_id: str, message_content: str):
    """发送文本消息"""
    request = (
        CreateMessageRequest.builder()
        .receive_id_type("chat_id")
        .request_body(
            CreateMessageRequestBody.builder()
            .receive_id(chat_id)
            .msg_type("text")
            .content(message_content)
            .build()
        )
        .build()
    )
    # 使用OpenAPI发送消息
    # Use send OpenAPI to send messages
    # https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/im-v1/message/create
    response = client.im.v1.message.create(request)

    if not response.success():
        raise Exception(
            f"client.im.v1.message.create failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}"
        )


def send_file_message_p2p(chat_id: str, file_key: str):
    """发送文件消息"""

    request = (
        CreateMessageRequest.builder()
        .receive_id_type("chat_id")
        .request_body(
            CreateMessageRequestBody.builder()
            .receive_id(chat_id)
            .msg_type("file")
            .content(json.dumps({"file_key": file_key}))
            .build()
        )
        .build()
    )
    # 使用OpenAPI发送消息
    # Use send OpenAPI to send messages
    # https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/im-v1/message/create
    response = client.im.v1.message.create(request)

    if not response.success():
        raise Exception(
            f"client.im.v1.message.create failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}"
        )


def download_file(file_key, file_path):
    # 构造请求对象
    request: GetFileRequest = GetFileRequest.builder() \
        .file_key(file_key) \
        .build()

    # 发起请求
    response: GetFileResponse = client.im.v1.file.get(request)

    # 处理失败返回
    if not response.success():
        lark.logger.error(
            f"client.im.v1.file.get failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
        return
    
    # 处理业务结果
    f = open(file_path, "wb")
    f.write(response.file.read())
    f.close()
    
def upload_file(file_path: str) -> str:
    # 构造请求对象
    file = open(file_path, "rb")
    file_name = os.path.basename(file_path)
    request: CreateFileRequest = CreateFileRequest.builder() \
        .request_body(CreateFileRequestBody.builder()
            .file_type("steam")
            .file_name(file_name)
            .file(file)
            .build()) \
        .build()

    # 发起请求
    response: CreateFileResponse = client.im.v1.file.create(request)

    # 处理失败返回
    if not response.success():
        lark.logger.error(
            f"client.im.v1.file.create failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
        return
    
    lark.logger.info(lark.JSON.marshal(response.data, indent=4))
    
    return response.data.file_key

# 注册接收消息事件，处理接收到的消息。
# Register event handler to handle received messages.
# https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/im-v1/message/events/receive
def do_p2_im_message_receive_v1(data: P2ImMessageReceiveV1) -> None:
    res_content = ""
    file_key = ''
    if data.event.message.message_type == "file":
        file_key = json.loads(data.event.message.content)["file_key"]
        file_name = json.loads(data.event.message.content)["file_name"]
        res_content = f"类型为文本, file_key = {file_key}, file_name = {file_name}"

    elif data.event.message.message_type == "text":
        text = json.loads(data.event.message.content)["text"]
        res_content = f"类型为文本, text = {text}"
    else:
        res_content = "解析消息失败，请发送文本消息"

    content = json.dumps({"text": "收到你发送的消息：" + res_content})

    if data.event.message.chat_type == "p2p":
        send_txt_message_p2p(data.event.message.chat_id, content)
    else:
        pass
    
    if file_key:
        # local_file_path = os.path.join(EVAL_RES_OUTPUT_PATH, file_name)
        # download_file(file_key, local_file_path)
        # file_key = upload_file(local_file_path)
        # print(file_key)
        # time.sleep(60)
        # content = json.dumps({"text": "测试回复"})
        # send_txt_message_p2p(data.event.message.chat_id, content)
        # send_file_message_p2p(data.event.message.chat_id, file_key)
        pass


# 注册事件回调
# Register event handler.
event_handler = (
    lark.EventDispatcherHandler.builder("", "")
    .register_p2_im_message_receive_v1(do_p2_im_message_receive_v1)
    .build()
)


# 创建 LarkClient 对象，用于请求OpenAPI, 并创建 LarkWSClient 对象，用于使用长连接接收事件。
# Create LarkClient object for requesting OpenAPI, and create LarkWSClient object for receiving events using long connection.
client = lark.Client.builder().app_id(lark.APP_ID).app_secret(lark.APP_SECRET).build()
wsClient = lark.ws.Client(
    lark.APP_ID,
    lark.APP_SECRET,
    event_handler=event_handler,
    log_level=lark.LogLevel.DEBUG,
)


def main():
    #  启动长连接，并注册事件处理器。
    #  Start long connection and register event handler.
    wsClient.start()


if __name__ == "__main__":
    main()
