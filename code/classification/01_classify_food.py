import csv
import os
import pandas as pd
from openai import OpenAI
from pathlib2 import Path
import pickle as pk
import time
from multiprocessing import Pool
from tqdm import tqdm
from io import StringIO  # 添加 StringIO 导入
from wcwidth import wcwidth

from packages.constants import (
    PROJECT_ROOT,
    EVAL_RES_OUTPUT_PATH,
    EVAL_DATA_IMAGE_ROOT,
    EVAL_DATA_EXCEL_PATH,
    API_URL,
    API_KEY_DISCOUNT,
)
from packages.my_logger import setup_logging

log_dir = os.path.join(PROJECT_ROOT, "logs", "eval_res_of_llm")
logger = setup_logging(log_dir=log_dir)
from packages.text_match import _BOX_RE, exact_match, extract_final_cn
from packages.call_api import call_openai_with_timeout
from packages.file_deal import (
    encode_image_with_resize,
    pk_dump,
    search_pk,
    pk_load,
    shuffle_excel_rows,
)

def chinese_ljust(text, width, fillchar=' '):
    """支持中文字符的左对齐"""
    current_width = sum(wcwidth(char) for char in text)
    padding = max(0, width - current_width)
    return text + fillchar * padding

def format_csv_to_aligned_columns(csv_file_path, output_txt_path=None):
    """
    将CSV文件格式化为对齐的列

    Args:
        csv_file_path: 输入的CSV文件路径
        output_txt_path: 输出的文本文件路径（可选）

    Returns:
        str: 格式化后的文本
    """
    # 读取CSV文件
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        rows = list(reader)

    # 计算每列的最大显示宽度（考虑中文字符）
    if not rows:
        return ""

    num_columns = len(rows[0])
    column_widths = [0] * num_columns

    for row in rows:
        for i, cell in enumerate(row):
            cell_width = sum(wcwidth(char) for char in str(cell))
            column_widths[i] = max(column_widths[i], cell_width)

    # 构建格式化字符串
    formatted_lines = []
    for row in rows:
        formatted_cells = []
        for i, cell in enumerate(row):
            # 左对齐，宽度为该列最大宽度
            formatted_cell = chinese_ljust(str(cell), column_widths[i])
            formatted_cells.append(formatted_cell)
        formatted_lines.append(",".join(formatted_cells))  # 用逗号分隔

    formatted_text = "\n".join(formatted_lines)

    # 输出到文件（如果指定了输出路径）
    if output_txt_path:
        with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(formatted_text)

    return formatted_text

def get_unfinished_tasks(temp_result_dir, total_tasks):
    """
    通过比较临时文件夹中的结果文件和样本数据集，确定未完成的任务

    Args:
        samples_df: 样本数据集DataFrame
        temp_result_dir: 临时结果文件夹路径
        max_tasks: 最大任务数限制，如果指定则只检查前max_tasks个任务
    """
    # 扫描临时文件夹中的结果文件
    completed_tasks = set(search_pk(temp_result_dir))
    all_tasks = set(range(total_tasks))
    return list(all_tasks - completed_tasks)


def parse_csv_from_response(response_text, batch_size):
    """
    从OpenAI响应中解析CSV数据，处理多种可能的格式
    Args:
        response_text: 包含CSV数据的响应文本
    Returns:
        pandas DataFrame对象
    """
    if not isinstance(response_text, str):
        logger.error(f"输入类型错误：预期字符串，实际得到 {type(response_text)}")
        return None

    start_id = 0
    lines = [line.strip(" `") for line in response_text.splitlines()]
    for id, line in enumerate(lines):
        if line.startswith("菜品名,"):
            start_id = id + 1
            break

    if not start_id:
        # 未找到任何CSV数据
        logger.error("未找到CSV格式数据，原文内容:")
        logger.error("-" * 50)
        logger.error(
            response_text[:200] + "..." if len(response_text) > 200 else response_text
        )
        logger.error("-" * 50)
        return None
    if start_id + batch_size > len(response_text.splitlines()):
        logger.warning("警告: 预期的行数超过响应中的实际行数，可能数据不完整")
    #     return None
    for id, line in enumerate(lines[start_id : start_id + batch_size]):
        if len(line.split(",")) != 5:
            logger.warning("警告: 解析的CSV列数不匹配预期，可能数据格式有误")
            logger.warning(f"line id={start_id + id + 1}, line={line}")
            return None

    csv_content = "\n".join(lines[start_id : start_id + batch_size])
    if csv_content:
        try:
            # return pd.read_csv(StringIO(csv_content))
            return csv_content

        except pd.errors.EmptyDataError:
            logger.error("CSV数据为空")
            return None
        except Exception as e:
            logger.error(f"解析CSV数据时出错: {str(e)}")
            logger.error("问题数据内容:")
            logger.error("-" * 50)
            logger.error(
                response_text[:200] + "..."
                if len(response_text) > 200
                else response_text
            )
            logger.error("-" * 50)
            return None


def process_foodname(model_name, batch_num, foodname_list, message_list, fn):
    logger.info(f"Processing {batch_num}...")
    prompt = "\n".join(foodname_list)
    messages = message_list + [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        # 使用带重试和超时的API调用
        response = call_openai_with_timeout(
            client=client,
            model=model_name,
            messages=messages,
            timeout=60,  # 设置60秒超时
        )

        ans = [batch_num, response.choices[0].message.content]
        csv_content = parse_csv_from_response(ans[1], len(foodname_list))
        if csv_content is None:
            return ''
        pk_dump(ans, batch_num, fn)
        logger.info(f"Complete batch {batch_num}, saved to {fn}")
        return ans
    except Exception as e:
        logger.error(f"Error processing batch {batch_num}: {str(e)}")
        return ""
prompt = """[角色与任务]
你将扮演一个顶级的AI美食分类专家。你的唯一任务是接收用户输入的食品/菜品名录，并严格按照以下我们共同建立的、包含十一个维度的分类体系、原则和流程，为每一个条目输出一个分类结果。

[最终输出格式]
请将所有结果整合为CSV格式（以英文逗号分隔）输出，不包含任何多余的说明文字。第一行为列标题行，后续每一行对应一个菜品。列标题和顺序必须严格如下：
菜品名,第一级别,第二级别,第三级别,第四级别,主要食材种类,烹饪方式,菜系（地域）分类,消费场景,营养建议分类,记忆/推理菜
每个分类项均需填写完整，不能留空。对于无法判定的项，请填写“不适用”，而非留空。
除“主要食材种类”以外分类仅可选取一项最合适的，不能选取复数个分类；除需填写“不适用”时，不得自创名称。对于“主要食材种类”，请选择所有适用的食材，并用“/”分隔。

[核心分类流程]
收到待分类菜名后，你将启动一个分析流程，为菜品精准画像并填入11列分类表格。
第一步：菜品识别与理解。
第二步：判定“记忆/推理”属性。
第三步：判定“消费场景”。
第四步：判定“菜系”。
第五步：确定“主要食材种类”与“烹饪方式”。
第六步：进行四级层级分类。遵循“分类判定优先级”规则，严格参照下方的[核心四级分类体系字典]来确定第一至第四级别。
第七步：判定“营养建议分类”。
第八步：整合输出为CSV格式。

[核心四级分类体系字典]
进行层级分类时，请严格参照以下表格中的从属关系和完整名称(括号中的内容不需要输出)：

| 第一级别 | 第二级别 | 第三级别 | 第四级别 (最终精确分类) |
| :--- | :--- | :--- | :--- |
| A. 富含优质蛋白质的菜品 | A1. 红肉类 | 猪肉 | 里脊 |
| | | | 五花肉 |
| | | | 排骨 |
| | | | 梅花肉 |
| | | | 猪蹄/猪脚 |
| | | | 猪肘/蹄髈 |
| | | | 猪头肉/猪耳 |
| | | | 猪颈肉 |
| | | | 腿肉 |
| | | | 肉片 (当部位不明确或菜品以该形态为核心时使用) |
| | | | 肉丝 (当部位不明确或菜品以该形态为核心时使用) |
| | | | 肉末/肉馅 (当部位不明确或菜品以该形态为核心时使用) |
| | | | 肉丸/肉饼 (当部位不明确或菜品以该形态为核心时使用) |
| | | 牛肉 | 牛腩 |
| | | | 牛腱 |
| | | | 牛里脊/牛柳 |
| | | | 牛排 |
| | | | 肥牛 |
| | | | 牛肉片 (当部位不明确或菜品以该形态为核心时使用) |
| | | | 牛肉丝 (当部位不明确或菜品以该形态为核心时使用) |
| | | | 牛肉末/馅 (当部位不明确或菜品以该形态为核心时使用) |
| | | | 牛肉丸 (当部位不明确或菜品以该形态为核心时使用) |
| | | 羊肉 | 羊腿 |
| | | | 羊排 |
| | | | 羊蝎子 |
| | | | 羊肉片 (当部位不明确或菜品以该形态为核心时使用) |
| | | | 羊肉串 (当部位不明确或菜品以该形态为核心时使用) |
| | | 其他畜肉 | 其他畜肉 |
| | A2. 禽肉类 | 鸡肉 | 整鸡 |
| | | | 鸡翅 |
| | | | 鸡腿 |
| | | | 鸡爪/鸡脚 |
| | | | 鸡胸 |
| | | | 鸡块 (当部位不明确或菜品以该形态为核心时使用) |
| | | | 鸡柳/鸡丝 (当部位不明确或菜品以该形态为核心时使用) |
| | | | 鸡丁 (当部位不明确或菜品以该形态为核心时使用) |
| | | | 鸡米花 (当部位不明确或菜品以该形态为核心时使用) |
| | | | 鸡肉丸 (当部位不明确或菜品以该形态为核心时使用) |
| | | 鸭肉 | 整鸭 |
| | | | 鸭掌 |
| | | | 鸭翅 |
| | | | 鸭舌 |
| | | | 鸭块 (当部位不明确或菜品以该形态为核心时使用) |
| | | 鹅肉 | 整鹅 |
| | | | 鹅掌 |
| | | 其他禽肉 | 其他禽肉 |
| | A3. 鱼类 | 高脂肪鱼类 | 高脂肪鱼类 |
| | | 低脂肪鱼类 | 低脂肪鱼类 |
| | A4. 其他水产类 | 虾 | 整虾 |
| | | | 虾仁/虾球 (当部位不明确或菜品以该形态为核心时使用) |
| | | | 虾滑 (当部位不明确或菜品以该形态为核心时使用) |
| | | 蟹 | 整蟹 |
| | | | 蟹黄 |
| | | | 蟹肉/蟹粉 (当部位不明确或菜品以该形态为核心时使用) |
| | | 贝类 | 蛤/蚬/蛏 |
| | | | 螺 |
| | | | 扇贝/带子 |
| | | | 蚝 |
| | | 软体水产 | 鱿鱼/墨鱼 |
| | | | 章鱼 |
| | | 爬行/两栖类 | 甲鱼/鳖 |
| | | | 牛蛙/田鸡 |
| | | 海产干货 | 海参 |
| | | | 鲍鱼 |
| | | | 海蜇 |
| | A5. 内脏类 | 畜肉内脏 | 肝（猪肝等） |
| | | | 肚（牛肚、猪肚等） |
| | | | 肠（肥肠等） |
| | | | 腰/肾 |
| | | | 心 |
| | | | 血制品 |
| | | 禽类内脏 | 胗（鸡胗、鸭胗等） |
| | | | 肝（鸡肝、鸭肝等） |
| | | | 心（鸡心等） |
| | | | 血制品（鸭血等） |
| | A6. 蛋制品类 | 未处理的蛋 | 鸡蛋 |
| | | | 鸭蛋 |
| | | | 鹌鹑蛋 |
| | | | 鹅蛋 |
| | | | 其他禽蛋 |
| | | 经过预处理的蛋 | 皮蛋/松花蛋 |
| | | | 咸鸭蛋 |
| | | | 茶叶蛋 |
| | | | 卤蛋 |
| | A7. 奶制品类 | 咸味奶/奶酪 | 奶油/黄油 |
| | | | 软质奶酪 |
| | | | 硬质/半硬质奶酪 |
| | A8. 植物蛋白类 | 大豆/豆类 | 大豆/豆类 |
| | | 豆制品 | 豆腐 |
| | | | 豆干 |
| | | | 腐竹/豆皮 |
| | | | 素鸡 |
| | | 菌菇 | 菌菇 |
| B. 以碳水化合物食材为主的菜品 | B1. 精制谷物类 | 精米饭/米粉/米制品 | 米饭 |
| | | | 泡饭 |
| | | | 饭团 |
| | | | 盖浇饭 |
| | | | 焖饭 |
| | | | 煲仔饭 |
| | | | 菜包饭 |
| | | | 炒饭 |
| | | | 拌饭 |
| | | | 粥 |
| | | | 米粉 |
| | | | 米线 |
| | | | 年糕 |
| | | 精面粉面食 | 面条 |
| | | | 包子 |
| | | | 饺子 |
| | | | 馄饨 |
| | | | 饼 |
| | | | 烧饼 |
| | | | 馒头 |
| | | | 油条 |
| | | | 面包/吐司 |
| | B2. 全谷物/杂粮类 | 全谷物/杂粮饭 | 杂粮饭 |
| | | | 杂粮粥 |
| | | 全麦/杂粮面食 | 杂粮馒头/窝头 |
| | | | 杂粮饼 |
| | B3. 高淀粉蔬菜/豆类 | 薯类 | 薯类 |
| | | 根茎类 | 根茎类 |
| | | 高淀粉豆 | 高淀粉豆 |
| C. 以蔬菜/水果为主的菜品 | C1. 蔬菜类 | 深色叶菜 | 深色叶菜 |
| | | 浅色叶菜 | 浅色叶菜 |
| | | 瓜果/茄果蔬菜 | 瓜果/茄果蔬菜 |
| | | 高纤维豆类 | 高纤维豆类 |
| | | 葱蒜/洋葱类 | 葱蒜/洋葱类 |
| | | 藻类 | 藻类 |
| | | 根茎类 | 根茎类 |
| | C2. 水果类 | 水果 | 仁果/核果类 |
| | | | 浆果类 |
| | | | 柑橘类 |
| | | | 热带/亚热带水果 |
| | | | 瓜类 |
| D. 高脂肪/高能量菜品 | D. 高脂肪/高能量菜品 | 油炸类 | 油炸-肉类 |
| | | | 油炸-水产 |
| | | | 油炸-蔬菜 |
| | | | 油炸-豆制品 |
| | | | 油炸-面食 |
| | | 源自动物脂肪 | 高脂红肉 |
| | | | 带皮禽肉 |
| | | 富含乳脂 | 奶油/重芝士 |
| | | 富含热带植物油 | 椰浆 |
| F. 汤羹类 | F. 汤羹类 | 清汤 | 清汤 |
| | | 浓汤/奶汤 | 浓汤/奶汤 |
| | | 羹 | 羹 |
| | | 药膳/滋补汤 | 药膳/滋补汤 |
| G. 甜品 | G. 甜品 | 烘焙类 | 烘焙类 |
| | | 冰品/冻品类 | 冰品/冻品类 |
| | | 中式甜汤 | 中式甜汤 |
| | | 中式糕团 | 中式糕团 |
| H. 饮品 | H. 饮品 | 非酒精饮品 | 非酒精饮品 |
| | | 酒精饮品 | 啤酒 |
| | | | 葡萄酒 |
| | | | 中式白酒 |
| | | | 黄酒/米酒 |
| | | | 其他烈酒 |
| | | | 鸡尾酒 |

[平行标签分类列表]

[主要食材种类]
米类,面粉或小麦类,玉米类,杂粮类,叶菜类,根茎类,瓜果类,茄果类,菌菇类,大豆或豆类,高纤维豆类,高淀粉豆类,薯类,豆腐或豆干类,豆浆或腐竹类,发酵豆制品,鸡蛋,鸭蛋,鹌鹑蛋,其他禽蛋,猪肉,牛肉,羊肉,其他畜肉,鸡肉,鸭肉,鹅肉,其他禽肉,畜肉内脏,禽类内脏,淡水鱼,海水鱼,虾类,蟹类,贝类,软体类,爬行或两栖类,海产干货,仁果或核果类,浆果类,柑橘类,热带或亚热带水果,瓜类,奶或奶油类,奶酪或芝士类,发酵乳品,坚果或种子类,茶或咖啡或可可类,藻类,鹅蛋,葱蒜或洋葱类,其他

[烹饪方式]
炒,爆,烧,焖,卤,烩,蒸,煮,炖,涮,炸,溜,凉拌,烤,煎,熏,焗,腌,醉,焯,汆,其他

[菜系（地域）分类]
川菜,粤菜,鲁菜,闽菜,苏菜,浙菜,湘菜,徽菜,京菜,本帮菜,东北菜,客家菜,西北菜,云南菜,港式/茶餐厅,台式,家常/不区分菜系,西餐,日料,韩料,东南亚菜

[消费场景]
家常菜,餐馆菜,宴席菜/大菜,小吃/街头美食,预制菜/工业化食品,其他

[营养建议分类]
分类名称: 主食/淀粉类/高糖/高油
纳入规则：“或”逻辑。满足“主食/高淀粉”、“高糖”、“高油”任一条件即归入此类。不满足则输出“无”。

[记忆/推理菜]
推理菜: 见名知菜。
记忆菜: 需背景知识解码。

[关键判定原则]

[分类判定优先级]
当一个菜品可以被归入多个第一级别类别时，按以下顺序决定其最终归属：
1. 优先级1：判定是否属于 D. 高脂肪/高能量菜品。
2. 优先级2：判定是否属于 A, B, C. 单一营养素主导的菜品。
3. 优先级3：如果前两者皆不适用，则根据“主食功能判定”原则在A类和B类中选择。
    - 主食功能判定原则1：如果菜品通常作为独立主食存在（一碗/一盘即一餐），优先归入 B. 富含碳水化合物的菜品。
    - 主食功能判定原则2：如果菜品是需要搭配主食（如米饭）食用的大菜，优先归入 A. 富含优质蛋白质的菜品。

[特殊条目处理规则]
对于完全无关的输入（如“汽车”），所有11个分类列都统一输出“不适用”。对于食品相关的非菜品条目（如“打豆浆”），在[第一级别]中标注为“(非菜品)”，在[二级分类]中标注具体原因（如“(原材料)”、“(动作/过程)”），并在其他所有列中标注“(不适用)”。

[输出样例]
菜品名,第一级别,第二级别,第三级别,第四级别,主要食材种类,烹饪方式,菜系（地域）分类,消费场景,营养建议分类,记忆/推理菜
西红柿炒蛋,A. 富含优质蛋白质的菜品,A6. 蛋制品类,未处理的蛋,炒鸡蛋,鸡蛋,炒,家常/不区分菜系,家常菜,主食/淀粉类/高糖/高油,推理菜
麻婆豆腐,A. 富含优质蛋白质的菜品,A8. 植物蛋白类,豆制品,烧豆制品,豆腐/豆干类,烧,川菜,家常菜,主食/淀粉类/高糖/高油,记忆菜"""

prompt = """[角色与任务]
你将扮演一个顶级的AI美食分类专家。你的唯一任务是接收用户输入的食品/菜品名录，并严格按照以下我们共同建立的、包含十一个维度的分类体系、原则和流程，为每一个条目输出一个分类结果。

[最终输出格式]
请将所有结果整合为CSV格式（以英文逗号分隔）输出，不包含任何多余的说明文字。第一行为列标题行，后续每一行对应一个菜品。列标题和顺序必须严格如下：
菜品名,第一级别,第二级别,第三级别,第四级别,主要食材种类,烹饪方式,菜系（地域）分类,消费场景,营养建议分类,记忆/推理菜
每个分类项均需填写完整，不能留空。对于无法判定的项，请填写“不适用”，而非留空。
除“主要食材种类”以外分类仅可选取一项最合适的，不能选取复数个分类；除需填写“不适用”时，不得自创名称。对于“主要食材种类”，请选择所有适用的食材，并用“/”分隔。

[核心分类流程]
收到待分类菜名后，你将启动一个分析流程，为菜品精准画像并填入11列分类表格。
第一步：菜品识别与理解。
第二步：判定“记忆/推理”属性。
第三步：判定“消费场景”。
第四步：判定“菜系”。
第五步：确定“主要食材种类”与“烹饪方式”。
第六步：进行四级层级分类。遵循“分类判定优先级”规则，严格参照下方的[核心四级分类体系字典]来确定第一至第四级别。
第七步：判定“营养建议分类”。
第八步：整合输出为CSV格式。

[核心四级分类体系字典]
进行层级分类时，请严格参照以下表格中的从属关系和完整名称(括号中的内容不需要输出)：

| 第一级别 | 第二级别 | 第三级别 | 第四级别 (最终精确分类) |
| :--- | :--- | :--- | :--- |
| A. 富含优质蛋白质的菜品 | A1. 红肉类 | 猪肉 | 里脊 |
| | | | 五花肉 |
| | | | 排骨 |
| | | | 梅花肉 |
| | | | 猪蹄/猪脚 |
| | | | 猪肘/蹄髈 |
| | | | 猪头肉/猪耳 |
| | | | 猪颈肉 |
| | | | 腿肉 |
| | | | 肉片 (当部位不明确或菜品以该形态为核心时使用) |
| | | | 肉丝 (当部位不明确或菜品以该形态为核心时使用) |
| | | | 肉末/肉馅 (当部位不明确或菜品以该形态为核心时使用) |
| | | | 肉丸/肉饼 (当部位不明确或菜品以该形态为核心时使用) |
| | | 牛肉 | 牛腩 |
| | | | 牛腱 |
| | | | 牛里脊/牛柳 |
| | | | 牛排 |
| | | | 肥牛 |
| | | | 牛肉片 (当部位不明确或菜品以该形态为核心时使用) |
| | | | 牛肉丝 (当部位不明确或菜品以该形态为核心时使用) |
| | | | 牛肉末/馅 (当部位不明确或菜品以该形态为核心时使用) |
| | | | 牛肉丸 (当部位不明确或菜品以该形态为核心时使用) |
| | | 羊肉 | 羊腿 |
| | | | 羊排 |
| | | | 羊蝎子 |
| | | | 羊肉片 (当部位不明确或菜品以该形态为核心时使用) |
| | | | 羊肉串 (当部位不明确或菜品以该形态为核心时使用) |
| | | 其他畜肉 | 其他畜肉 |
| | A2. 禽肉类 | 鸡肉 | 整鸡 |
| | | | 鸡翅 |
| | | | 鸡腿 |
| | | | 鸡爪/鸡脚 |
| | | | 鸡胸 |
| | | | 鸡块 (当部位不明确或菜品以该形态为核心时使用) |
| | | | 鸡柳/鸡丝 (当部位不明确或菜品以该形态为核心时使用) |
| | | | 鸡丁 (当部位不明确或菜品以该形态为核心时使用) |
| | | | 鸡米花 (当部位不明确或菜品以该形态为核心时使用) |
| | | | 鸡肉丸 (当部位不明确或菜品以该形态为核心时使用) |
| | | 鸭肉 | 整鸭 |
| | | | 鸭掌 |
| | | | 鸭翅 |
| | | | 鸭舌 |
| | | | 鸭块 (当部位不明确或菜品以该形态为核心时使用) |
| | | 鹅肉 | 整鹅 |
| | | | 鹅掌 |
| | | 其他禽肉 | 其他禽肉 |
| | A3. 鱼类 | 高脂肪鱼类 | 高脂肪鱼类 |
| | | 低脂肪鱼类 | 低脂肪鱼类 |
| | A4. 其他水产类 | 虾 | 整虾 |
| | | | 虾仁/虾球 (当部位不明确或菜品以该形态为核心时使用) |
| | | | 虾滑 (当部位不明确或菜品以该形态为核心时使用) |
| | | 蟹 | 整蟹 |
| | | | 蟹黄 |
| | | | 蟹肉/蟹粉 (当部位不明确或菜品以该形态为核心时使用) |
| | | 贝类 | 蛤/蚬/蛏 |
| | | | 螺 |
| | | | 扇贝/带子 |
| | | | 蚝 |
| | | 软体水产 | 鱿鱼/墨鱼 |
| | | | 章鱼 |
| | | 爬行/两栖类 | 甲鱼/鳖 |
| | | | 牛蛙/田鸡 |
| | | 海产干货 | 海参 |
| | | | 鲍鱼 |
| | | | 海蜇 |
| | A5. 内脏类 | 畜肉内脏 | 肝（猪肝等） |
| | | | 肚（牛肚、猪肚等） |
| | | | 肠（肥肠等） |
| | | | 腰/肾 |
| | | | 心 |
| | | | 血制品 |
| | | 禽类内脏 | 胗（鸡胗、鸭胗等） |
| | | | 肝（鸡肝、鸭肝等） |
| | | | 心（鸡心等） |
| | | | 血制品（鸭血等） |
| | A6. 蛋制品类 | 未处理的蛋 | 鸡蛋 |
| | | | 鸭蛋 |
| | | | 鹌鹑蛋 |
| | | | 鹅蛋 |
| | | | 其他禽蛋 |
| | | 经过预处理的蛋 | 皮蛋/松花蛋 |
| | | | 咸鸭蛋 |
| | | | 茶叶蛋 |
| | | | 卤蛋 |
| | A7. 奶制品类 | 咸味奶/奶酪 | 奶油/黄油 |
| | | | 软质奶酪 |
| | | | 硬质/半硬质奶酪 |
| | A8. 植物蛋白类 | 大豆/豆类 | 大豆/豆类 |
| | | 豆制品 | 豆腐 |
| | | | 豆干 |
| | | | 腐竹/豆皮 |
| | | | 素鸡 |
| | | 菌菇 | 菌菇 |
| B. 以碳水化合物食材为主的菜品 | B1. 精制谷物类 | 精米饭/米粉/米制品 | 米饭 |
| | | | 泡饭 |
| | | | 饭团 |
| | | | 盖浇饭 |
| | | | 焖饭 |
| | | | 煲仔饭 |
| | | | 菜包饭 |
| | | | 炒饭 |
| | | | 拌饭 |
| | | | 粥 |
| | | | 米粉 |
| | | | 米线 |
| | | | 年糕 |
| | | 精面粉面食 | 面条 |
| | | | 包子 |
| | | | 饺子 |
| | | | 馄饨 |
| | | | 饼 |
| | | | 烧饼 |
| | | | 馒头 |
| | | | 油条 |
| | | | 面包/吐司 |
| | B2. 全谷物/杂粮类 | 全谷物/杂粮饭 | 杂粮饭 |
| | | | 杂粮粥 |
| | | 全麦/杂粮面食 | 杂粮馒头/窝头 |
| | | | 杂粮饼 |
| | B3. 高淀粉蔬菜/豆类 | 薯类 | 薯类 |
| | | 根茎类 | 根茎类 |
| | | 高淀粉豆 | 高淀粉豆 |
| C. 以蔬菜/水果为主的菜品 | C1. 蔬菜类 | 深色叶菜 | 深色叶菜 |
| | | 浅色叶菜 | 浅色叶菜 |
| | | 瓜果/茄果蔬菜 | 瓜果/茄果蔬菜 |
| | | 高纤维豆类 | 高纤维豆类 |
| | | 葱蒜/洋葱类 | 葱蒜/洋葱类 |
| | | 藻类 | 藻类 |
| | | 根茎类 | 根茎类 |
| | C2. 水果类 | 水果 | 仁果/核果类 |
| | | | 浆果类 |
| | | | 柑橘类 |
| | | | 热带/亚热带水果 |
| | | | 瓜类 |
| D. 高脂肪/高能量菜品 | D. 高脂肪/高能量菜品 | 油炸类 | 油炸-肉类 |
| | | | 油炸-水产 |
| | | | 油炸-蔬菜 |
| | | | 油炸-豆制品 |
| | | | 油炸-面食 |
| | | 源自动物脂肪 | 高脂红肉 |
| | | | 带皮禽肉 |
| | | 富含乳脂 | 奶油/重芝士 |
| | | 富含热带植物油 | 椰浆 |
| F. 汤羹类 | F. 汤羹类 | 清汤 | 清汤 |
| | | 浓汤/奶汤 | 浓汤/奶汤 |
| | | 羹 | 羹 |
| | | 药膳/滋补汤 | 药膳/滋补汤 |
| G. 甜品 | G. 甜品 | 烘焙类 | 烘焙类 |
| | | 冰品/冻品类 | 冰品/冻品类 |
| | | 中式甜汤 | 中式甜汤 |
| | | 中式糕团 | 中式糕团 |
| H. 饮品 | H. 饮品 | 非酒精饮品 | 非酒精饮品 |
| | | 酒精饮品 | 啤酒 |
| | | | 葡萄酒 |
| | | | 中式白酒 |
| | | | 黄酒/米酒 |
| | | | 其他烈酒 |
| | | | 鸡尾酒 |

[平行标签分类列表]

[主要食材种类]
米类,面粉或小麦类,玉米类,杂粮类,叶菜类,根茎类,瓜果类,茄果类,菌菇类,大豆或豆类,高纤维豆类,高淀粉豆类,薯类,豆腐或豆干类,豆浆或腐竹类,发酵豆制品,鸡蛋,鸭蛋,鹌鹑蛋,其他禽蛋,猪肉,牛肉,羊肉,其他畜肉,鸡肉,鸭肉,鹅肉,其他禽肉,畜肉内脏,禽类内脏,淡水鱼,海水鱼,虾类,蟹类,贝类,软体类,爬行或两栖类,海产干货,仁果或核果类,浆果类,柑橘类,热带或亚热带水果,瓜类,奶或奶油类,奶酪或芝士类,发酵乳品,坚果或种子类,茶或咖啡或可可类,藻类,鹅蛋,葱蒜或洋葱类,其他

[烹饪方式]
炒,爆,烧,焖,卤,烩,蒸,煮,炖,涮,炸,溜,凉拌,烤,煎,熏,焗,腌,醉,焯,汆,其他

[菜系（地域）分类]
川菜,粤菜,鲁菜,闽菜,苏菜,浙菜,湘菜,徽菜,京菜,本帮菜,东北菜,客家菜,西北菜,云南菜,港式/茶餐厅,台式,家常/不区分菜系,西餐,日料,韩料,东南亚菜

[消费场景]
家常菜,餐馆菜,宴席菜/大菜,小吃/街头美食,预制菜/工业化食品,其他

[营养建议分类]
分类名称: 主食/淀粉类/高糖/高油
纳入规则：“或”逻辑。满足“主食/高淀粉”、“高糖”、“高油”任一条件即归入此类。不满足则输出“无”。

[记忆/推理菜]
推理菜: 见名知菜。
记忆菜: 需背景知识解码。

[关键判定原则]

[分类判定优先级]
当一个菜品可以被归入多个第一级别类别时，按以下顺序决定其最终归属：
1. 优先级1：判定是否属于 D. 高脂肪/高能量菜品。
2. 优先级2：判定是否属于 A, B, C. 单一营养素主导的菜品。
3. 优先级3：如果前两者皆不适用，则根据“主食功能判定”原则在A类和B类中选择。
    - 主食功能判定原则1：如果菜品通常作为独立主食存在（一碗/一盘即一餐），优先归入 B. 富含碳水化合物的菜品。
    - 主食功能判定原则2：如果菜品是需要搭配主食（如米饭）食用的大菜，优先归入 A. 富含优质蛋白质的菜品。

[特殊条目处理规则]
对于完全无关的输入（如“汽车”），所有11个分类列都统一输出“不适用”。对于食品相关的非菜品条目（如“打豆浆”），在[第一级别]中标注为“(非菜品)”，在[二级分类]中标注具体原因（如“(原材料)”、“(动作/过程)”），并在其他所有列中标注“(不适用)”。

[输出样例]
菜品名,第一级别,第二级别,第三级别,第四级别,主要食材种类,烹饪方式,菜系（地域）分类,消费场景,营养建议分类,记忆/推理菜
西红柿炒蛋,A. 富含优质蛋白质的菜品,A6. 蛋制品类,未处理的蛋,炒鸡蛋,鸡蛋,炒,家常/不区分菜系,家常菜,主食/淀粉类/高糖/高油,推理菜
麻婆豆腐,A. 富含优质蛋白质的菜品,A8. 植物蛋白类,豆制品,烧豆制品,豆腐/豆干类,烧,川菜,家常菜,主食/淀粉类/高糖/高油,记忆菜"""


prompt = """[角色与任务]
你将扮演一个顶级的AI美食分类专家。你的唯一任务是接收用户输入的食品/菜品名录，并严格按照以下我们共同建立的、包含四个维度的分类体系、原则和流程，为每一个条目输出一个分类结果。

[最终输出格式]
请将所有结果整合为CSV格式（以英文逗号分隔）输出，不包含任何多余的说明文字。第一行为列标题行，后续每一行对应一个菜品。列标题和顺序必须严格如下：
菜品名,营养建议分类,早中晚餐消费场景,食用频率,熟悉程度
每个分类项均需填写完整，不能留空。对于无法判定的项，请填写“不适用”，而非留空。
除“营养建议分类”以外分类仅可选取一项最合适的，不能选取复数个分类；除需填写“不适用”时，不得自创名称。对于“营养建议分类”，请选择所有适用的食材，并用“/”分隔。

[核心分类流程]
收到待分类菜名后，你将启动一个分析流程，为菜品精准画像并填入4列分类表格。
第一步：菜品识别与理解。
第二步：判定“早中晚餐消费场景”属性。
第三步：判定“食用频率”与“熟悉程度”。
第四步：判定“营养建议分类”。
第五步：整合输出为CSV格式。

[平行标签分类列表]

[营养建议分类]
分类名称: 主食,淀粉类,高糖,高油
纳入规则：只满足一项时，归入对应的类别；同时满足多项时，以"/"分隔，如主食或淀粉类/高糖；不满足则输出“无”。

[早中晚餐消费场景]
分类名称：早餐,午餐或晚餐,夜宵,下午茶/零食,其他
分类原则：按该菜品最常见的消费场景进行分类

[食用频率]
分类结果：1分到5分
分类原则：食用频率：1分（极少，仅限特定场合/地域）到 5分（几乎每周都会出现）

[熟悉程度]
分类结果：1分到5分
分类原则：熟悉程度：1分（很多人不知道）到 5分（家喻户晓，人人会点）

[关键判定原则]

[特殊条目处理规则]
对于完全无关的输入（如“汽车”），所有5个分类列都统一输出“不适用”。对于食品相关的非菜品条目（如“打豆浆”），在所有列中标注“(不适用)”。

[输出样例]
菜名,营养建议分类,早中晚餐消费场景,食用频率,熟悉程度
西红柿炒鸡蛋,高油,午餐或晚餐,5,5
酸辣土豆丝,淀粉类/高油,午餐或晚餐,5,5
猪脚饭,高油,午餐或晚餐,3,3
胡辣汤,淀粉类/高油,早餐,2,2
肠旺面,主食/高油,午餐或晚餐,1,2
红烧肉,高油/高糖,午餐或晚餐,3,5
月饼,主食/高糖/高油,其他,1,5
三不粘,淀粉类/高糖/高油,下午茶/零食,1,1
牛瘪火锅,无,午餐或晚餐,1,1
炒粉,主食/高油,夜宵,4,5
烧烤,高油,夜宵,3,5
小龙虾,高油,夜宵,2,5
三不粘,淀粉类/高糖/高油,下午茶/零食,1,1
月饼,主食/高糖/高油,其他,1,5
"""

# def create_context(model_name):

#     logger.info(f"Creating context for {model_name}...")
    
#     message_list = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": prompt},
#             ],
#         }
#     ]
#     try:
#         response = client.chat.completions.create(
#             model=model_name,
#             messages=message_list,
#         )
#         logger.info(f'create cpntext result: {response.choices[0].message.content}')
#         message_list.append(
#             {
#                 "role": response.choices[0].message.role,
#                 "content": [
#                     {"type": "text", "text": response.choices[0].message.content},
#                 ],
#             }
#         )
#         logger.info("Context create complete")
#         return message_list
#     except Exception as e:
#         logger.error(f"Error occurred while creating context: {e}")
#         return ""

def create_context(model_name):

    logger.info(f"Creating context for {model_name}...")
    
    message_list = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]
    message_list.append(
        {
            "role": 'assistant',
            "content": [
                {"type": "text", "text": """菜品名,营养建议分类,早中晚餐消费场景,食用频率,熟悉程度
麻辣香锅,高油,午餐或晚餐,4,4
豆浆油条,主食/高油,早餐,4,5
杨枝甘露,高糖,下午茶/零食,2,4
佛跳墙,无,午餐或晚餐,1,3
螺蛳粉,主食/高油,午餐或晚餐,3,4
方便面,主食/高油,夜宵,4,5
可乐鸡翅,高糖/高油,午餐或晚餐,4,5
地三鲜,高油,午餐或晚餐,3,4
烤冷面,主食/高糖/高油,下午茶/零食,2,3
汽车,不适用,不适用,不适用,不适用"""},
            ],
        }
    )
    logger.info("Context create complete")
    return message_list

if __name__ == "__main__":
    os.environ["OPENAI_BASE_URL"] = API_URL
    os.environ["OPENAI_API_KEY"] = API_KEY_DISCOUNT

    version = 'V100R25C20'
    client = OpenAI()
    file_name = "food_name_all"
    FD = pd.read_csv(f"/date0/crwu/zjts_foodname/data/classification/{file_name}.csv")
    temp_pk_target = Path(
        f"/date0/crwu/zjts_foodname/data/classification/outputs_by_type_{file_name}_{version}/temp_pk"
    )
    os.makedirs(temp_pk_target, exist_ok=True)
    FD["outfn"] = FD["菜名"].apply(lambda s: temp_pk_target / f"{s}.pk")

    # prepare for mapping
    FD_ = FD[FD["outfn"].apply(lambda fn: not os.path.exists(fn))]
    # inputs = FD_[["菜名", "outfn"]].values  # convert to tuple for map function

    model_name = "gemini-2.5-pro"  # "gemini-2.5-flash-preview-04-17-nothinking" #"gemini-2.5-pro" #"qwen-vl-max-latest"

    t_start = time.time()


    food_name_list = FD_["菜名"].tolist()
    # random.shuffle(food_name_list)
    test_num = 30000
    food_name_list = food_name_list[:test_num]

    logger.info(f"{len(food_name_list)} food names remaining for processing...")


    batch_size = 5
    batch_list = [
        food_name_list[i : i + batch_size] for i in range(0, len(food_name_list), batch_size)
    ]
    unfinished_batch_num_list = get_unfinished_tasks(temp_pk_target, len(batch_list))
    unfinished_batch_list = [batch_list[i] for i in unfinished_batch_num_list]
    logger.info(f"{len(unfinished_batch_list)} batches remaining for processing...")
    # test_num = 10
    # unfinished_batch_list=unfinished_batch_list[:test_num]
    failed_batches = []
    while unfinished_batch_num_list:
        failed_batches = []
        message_list = create_context(model_name)

        create_context_time = time.time() - t_start
        logger.info(f"Context creation time: {create_context_time:.2f} seconds")

        with Pool(150) as pool:
            # 用于存储所有异步任务
            async_results = []

            for batch_num, foodname_list in tqdm(
                zip(unfinished_batch_num_list, unfinished_batch_list)
            ):
                logger.info(
                    f"Processing batch {batch_num} with {len(foodname_list)} items..."
                )
                # 存储异步任务结果
                result = pool.apply_async(
                    func=process_foodname,
                    args=(
                        model_name,
                        batch_num,
                        foodname_list,
                        message_list,
                        temp_pk_target,
                    ),
                )
                async_results.append(result)
                # break

            # 等待所有任务完成并获取结果
            logger.info("\n等待所有异步任务完成...")
            for i, async_result in enumerate(tqdm(async_results)):
                try:
                    result = async_result.get(timeout=300)  # 300秒超时
                    if not result:  # 如果返回空字符串，说明处理失败
                        failed_batches.append(unfinished_batch_num_list[i])
                except Exception as e:
                    logger.info(f"Batch {unfinished_batch_num_list[i]} failed: {str(e)}")
                    failed_batches.append(unfinished_batch_num_list[i])
            unfinished_batch_num_list = failed_batches

            # 等待所有进程完成
            pool.close()
            pool.join()

        logger.info("\n所有异步任务已完成...")

    # 从pk文件中读取并合并结果
    logger.info("\n开始合并结果...")
    all_results = []
    pk_files = list(temp_pk_target.glob("*.pk"))
    logger.info(f"\n开始处理 {len(pk_files)} 个pk文件...")

    fail_pk_files = []
    for pk_file in tqdm(pk_files):
        try:
            with open(pk_file, "rb") as f:
                data = pk.load(f)
                if isinstance(data, list) and len(data) == 2:
                    batch_num, response_text = data
                    # 解析CSV数据
                    # df = parse_csv_from_response(response_text, batch_size)
                    csv_text = parse_csv_from_response(response_text, batch_size)
                    # print(df)
                    if csv_text is not None:
                        all_results.append((batch_num, csv_text))
                    else:
                        logger.error(f"文件 {pk_file} 中未能解析出有效的DataFrame")
                        fail_pk_files.append(pk_file)
        except Exception as e:
            logger.error(f"处理文件 {pk_file} 时出错: {str(e)}")
            fail_pk_files.append(pk_file)
        # break

    # 合并所有DataFrame
    if all_results:
        print(f"\n成功解析 {len(all_results)} 个结果文件")
        # 按batch_num排序
        all_results.sort(key=lambda x: x[0])
        # 提取所有DataFrame并合并
        # all_df = pd.concat([df for _, df in all_results], ignore_index=True)

        # 保存合并后的结果        
        output_path = Path(
            f"/date0/crwu/zjts_foodname/data/classification/outputs_by_type_{file_name}_{version}/merged_results.csv"
        )
        output_path2 = Path(
            f"/date0/crwu/zjts_foodname/data/classification/outputs_by_type_{file_name}_{version}/merged_results_formatted.csv"
        )
        if os.path.exists(output_path):
            os.remove(output_path)
        with open(output_path, 'a') as f:
            # headers = '菜品名,第一级别,第二级别,第三级别,第四级别,主要食材种类,烹饪方式,菜系（地域）分类,消费场景,营养建议分类,记忆/推理菜\n'
            headers = '菜品名,营养建议分类,早中晚餐消费场景,食用频率,熟悉程度\n'
            f.writelines(headers)
        line_num = 0
        for _, csv_text in all_results:
            with open(output_path, 'a') as f:
                for line in csv_text.splitlines():
                    f.writelines(line+'\n')
                    line_num +=1
        format_csv_to_aligned_columns(output_path, output_path2)
        print(f"保存合并结果到: {output_path}")
        print(f"保存format结果到: {output_path2}")
        print(f"总计处理了 {line_num} 条数据")
    else:
        print("没有找到有效的数据可以合并")

    t_end = time.time()
    t = t_end - t_start

    logger.info("=== 执行统计 ===")
    logger.info(f"* 待处理批次数: {len(unfinished_batch_list)}")
    if len(failed_batches) > 0:
        logger.error(f"* 失败批次数: {len(failed_batches)}")
        logger.error(f"* 失败批次编号: {failed_batches}")

    logger.info(f"* 待解析文件数: {len(pk_files)}")
    logger.info(f"* 成功解析文件数: {len(all_results)}")
    logger.error(f"* 失败解析文件数: {len(fail_pk_files)}")
    logger.error(f"* 失败解析文件: {fail_pk_files}")
    logger.info(f"* 总执行时间: {t:.2f} 秒")
    for fail_file in fail_pk_files:
        # os.remove(fail_file)
        logger.info(f"删除失败文件: {fail_file}")
