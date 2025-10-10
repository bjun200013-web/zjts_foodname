import os
import pandas as pd
from openai import OpenAI
from pathlib2 import Path
import pickle as pk
import time
from multiprocessing import Pool
from tqdm import tqdm
from io import StringIO  # 添加 StringIO 导入
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.exceptions import Timeout, RequestException
import httpx

def pk_dump(D, fn):
    with open(fn, "wb") as f:
        pk.dump(D, f, protocol=pk.HIGHEST_PROTOCOL)
    return

def get_unfinished_tasks(temp_result_dir, total_tasks):
    """
    通过比较临时文件夹中的结果文件和样本数据集，确定未完成的任务

    Args:
        samples_df: 样本数据集DataFrame
        temp_result_dir: 临时结果文件夹路径
        max_tasks: 最大任务数限制，如果指定则只检查前max_tasks个任务
    """
    completed_tasks = set()
    # 扫描临时文件夹中的结果文件
    for filename in os.listdir(temp_result_dir):
        if filename.endswith(".pk"):
            try:
                idx = int(filename[:-3])  # 从"X.pk"中提取索引
                completed_tasks.add(idx)
            except ValueError:
                continue
            
    all_tasks = set(range(total_tasks))
    return list(all_tasks - completed_tasks)

def parse_csv_from_response_bk(response_text):
    """
    从OpenAI响应中解析CSV数据，处理多种可能的格式
    Args:
        response_text: 包含CSV数据的响应文本
    Returns:
        pandas DataFrame对象
    """
    if not isinstance(response_text, str):
        print(f"输入类型错误：预期字符串，实际得到 {type(response_text)}")
        return None
        
    try:
        # 情况1: 标准的 ```csv 格式
        csv_marker = '```csv\n'
        start_idx = response_text.find(csv_marker)
        if start_idx != -1:
            start_idx += len(csv_marker)
            end_idx = response_text.find('```', start_idx)
            if end_idx != -1:
                csv_content = response_text[start_idx:end_idx].strip()
                if csv_content:
                    return pd.read_csv(StringIO(csv_content))
        
        # 情况2: 无标记的CSV格式（直接以标题行开始）
        if response_text.strip().startswith('菜名/食品名,'):
            return pd.read_csv(StringIO(response_text.strip()))
        
        # 情况3: 有其他文本前缀的CSV（查找第一个CSV样式的行）
        lines = response_text.split('\n')
        for i, line in enumerate(lines):
            if '菜名/食品名,' in line:
                csv_content = '\n'.join(lines[i:])
                if csv_content:
                    return pd.read_csv(StringIO(csv_content))
        
        # 情况4: 有其他标记格式（如 --- 分隔）
        sections = response_text.split('---')
        for section in sections:
            if '菜名/食品名,' in section:
                csv_section = section.strip()
                if csv_section:
                    return pd.read_csv(StringIO(csv_section))
        
        # 未找到任何CSV数据
        print("未找到CSV格式数据，原文内容:")
        print("-" * 50)
        print(response_text[:200] + "..." if len(response_text) > 200 else response_text)
        print("-" * 50)
        return None
        
    except pd.errors.EmptyDataError:
        print("CSV数据为空")
        return None
    except Exception as e:
        print(f"解析CSV数据时出错: {str(e)}")
        print("问题数据内容:")
        print("-" * 50)
        print(response_text[:200] + "..." if len(response_text) > 200 else response_text)
        print("-" * 50)
        return None
        
def parse_csv_from_response(response_text, batch_size):
    """
    从OpenAI响应中解析CSV数据，处理多种可能的格式
    Args:
        response_text: 包含CSV数据的响应文本
    Returns:
        pandas DataFrame对象
    """
    if not isinstance(response_text, str):
        print(f"输入类型错误：预期字符串，实际得到 {type(response_text)}")
        return None
    
    start_id = 0
    lines = [line.strip(' `') for line in response_text.splitlines()]
    for id, line in enumerate(lines):
        if line.startswith('菜名/食品名,'):
            start_id = id+1            
            break
        
    if not start_id:
        # 未找到任何CSV数据
            print("未找到CSV格式数据，原文内容:")
            print("-" * 50)
            print(response_text[:200] + "..." if len(response_text) > 200 else response_text)
            print("-" * 50)
            return None
    if start_id + batch_size > len(response_text.splitlines()):
        print("警告: 预期的行数超过响应中的实际行数，可能数据不完整")
        return None
    for id, line in enumerate(lines[start_id:start_id + batch_size]):
        if len(line.split(',')) != 8:
            print("警告: 解析的CSV列数不匹配预期，可能数据格式有误")
            print(f'line id={start_id + id + 1}, line={line}')
            return None

    csv_content = '\n'.join(lines[start_id:start_id + batch_size])
    if csv_content:
        try:
            return pd.read_csv(StringIO(csv_content))
                    
        except pd.errors.EmptyDataError:
            print("CSV数据为空")
            return None
        except Exception as e:
            print(f"解析CSV数据时出错: {str(e)}")
            print("问题数据内容:")
            print("-" * 50)
            print(response_text[:200] + "..." if len(response_text) > 200 else response_text)
            print("-" * 50)
            return None


@retry(
    stop=stop_after_attempt(2),  # 最多重试3次
    wait=wait_exponential(multiplier=1, min=300, max=600),  # 指数退避，等待时间在300-600秒之间
    retry=retry_if_exception_type((Timeout, RequestException, httpx.TimeoutException)),  # 只对超时和请求异常进行重试
    reraise=True  # 重试失败后抛出原始异常
)
def call_openai_with_timeout(client, model, messages, timeout=30):
    """带超时和重试的OpenAI API调用"""
    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=timeout
        )
    except Exception as e:
        print(f"API调用出错 (将重试): {str(e)}")
        raise

def process_foodname(model_name, batch_num, foodname_list, message_list, fn):
    print(f"Processing {batch_num}...")
    prompt = '\n'.join(foodname_list)
    messages = message_list + [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
        ],
    }]
    
    try:
        # 使用带重试和超时的API调用
        response = call_openai_with_timeout(
            client=client,
            model=model_name,
            messages=messages,
            timeout=300  # 设置60秒超时
        )

        ans = [batch_num, response.choices[0].message.content]
        pk_dump(ans, fn)
        print(f"Complete batch {batch_num}, saved to {fn}")
        return ans
    except Exception as e:
        print(f"Error processing batch {batch_num}: {str(e)}")
        return ""


def create_context(model_name):

    print(f"Creating context for {model_name}...")
    prompt = """你好，请你扮演一个美食专家AI。我需要你为我输入的菜品/食品名录进行分类。请严格遵守以下我们共同建立的、包含七个维度的分类体系和所有规则。

## 一、最终输出格式

请为我输入的每一个菜品/食品，都输出一个包含以下七个标签的分类结果，并使用csv格式呈现：(菜名/食品名，标签1，标签2...标签7)

1.  `一级分类`
2.  `二级分类`
3.  `主要食材种类`
4.  `菜系（地域）分类`
5.  `烹饪方式`
6.  `消费场景`
7.  `营养建议分类`

## 二、详细分类标准与列表

### 1. 一级分类 (按功能与形态)
这是菜品在餐桌上的核心功能分类。

* 主食
* 纯素菜
* 半荤半素
* 纯荤菜
* 汤羹
* 点心/小吃
* 甜品
* 饮品
* 凉菜/前菜

### 2. 二级分类 (按具体食材或形态)
此分类根据“一级分类”的不同而变化，具体规则如下：

* **对于【纯荤菜】/【半荤半素】**: 按主要荤菜食材的精确类别划分（见下文“主要食材种类”的细化列表，如“猪肉”、“鸡肉”等）。
* **对于【纯素菜】**: 按主要素菜食材的类别划分（如“叶菜类”、“豆制品类”等）。
* **对于【主食】**: 按主食的约定俗成的具体品类划分。
* **对于【甜品】**: 按甜品的约定俗成的具体品类划分。
* **对于【汤羹】**: 按汤的形态质地划分（清汤、浓汤、羹）。
* **对于【饮品】**: 按饮品基底划分（茶饮、咖啡等）。
* **对于【凉菜/前菜】**: 按主要食材的荤素划分（素食前菜、肉食前菜、水产前菜）。

### 3. 主要食材种类 (按食材大类)
这是对菜品核心原材料的归属分类。

**规则：** 对于畜肉和禽肉类菜品，此分类应与【二级分类】保持一致，使用最精确的分类。例如，“红烧肉”的二级分类和主要食材种类都应为“猪肉”。

* 谷物类
* 蔬菜类
* 菌菇类
* 豆制品类
* 蛋类
* **猪肉**
* **牛肉**
* **羊肉**
* **其他畜肉**
* **鸡肉**
* **鸭肉**
* **鹅肉**
* **其他禽肉**
* 内脏类
* 鱼类
* 虾蟹类
* 贝类
* 其他水产类
* 水果类
* 乳制品类
* 坚果/种子类
* 茶/咖啡/可可类

### 4. 菜系（地域）分类
这是菜品的文化与风味源流标签。

* 川菜
* 粤菜
* 鲁菜
* 闽菜
* 苏菜
* 浙菜
* 湘菜
* 徽菜
* 京菜
* 本帮菜
* 东北菜
* 客家菜
* 西北菜
* 云南菜
* 港式/茶餐厅
* 台式
* 家常/不区分菜系
* 西餐
* 日料
* 韩料
* 东南亚菜

### 5. 烹饪方式
这是菜品的主要制作工艺标签。

* 炒
* 爆
* 烧
* 焖
* 卤
* 烩
* 蒸
* 煮
* 炖
* 涮
* 炸
* 溜
* 凉拌
* 烤
* 煎
* 熏
* 焗
* 腌
* 醉
* 焯
* 汆

### 6. 消费场景
这是菜品的社会属性与规格分类。

* 家常菜
* 餐馆菜
* 宴席菜/大菜
* 小吃/街头美食
* 预制菜/工业化食品

### 7. 营养建议分类
这是一个基于健康建议的“高能预警”分类。

* **分类名称**: `主食/淀粉类/高糖/高油`
* **纳入规则（核心）**: **“或”逻辑**。任何菜品/食品，只要满足以下四个条件中的**至少一个**，就应被归入此类：
    1.  属于【一级分类】中的**主食**，或主要食材为高**淀粉**类（如土豆、红薯）。
    2.  在烹饪中添加了大量糖或含糖酱汁，属于**高糖**。
    3.  使用油炸、大量油滑炒等烹饪方式，或食材本身脂肪含量高，属于**高油**。
* 如果不满足以上任何一个条件，则输出“**无**”。

### 三、特殊条目处理规则

* 对于非菜品/食品的输入（如“打豆浆”、“包饺子”等动作；“面包粉”等原材料；“面包超人”等无关内容），请在分类列中标注为“(非菜品)”、“(原材料)”等，并在其他列中标注“(不适用)”。"""
    message_list = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=message_list,
        )
        print(response)
        message_list.append(
            {
                "role": response.choices[0].message.role,
                "content": [
                    {"type": "text", "text": response.choices[0].message.content},
                ],
            }
        )
        print("Context create complete")
        return message_list
    except Exception as e:
        print(f"Error occurred while creating context: {e}")
        return ""


if __name__ == "__main__":
    os.environ["OPENAI_BASE_URL"] = "https://api.qingyuntop.top/v1"
    os.environ["OPENAI_API_KEY"] = "sk-mFtES1U9ZQqCpLuoODW1cH6XyDOZEMcfKzBNSq9ROBEBV5YW"

    client = OpenAI()
    file_name = "food_name3"
    FD = pd.read_csv(f"/date0/crwu/classification/data/{file_name}.csv")
    fd_target = Path(f"/date0/crwu/classification/outputs_by_type_{file_name}")

    FD["outfn"] = FD["菜名"].apply(lambda s: fd_target / f"{s}.pk")

    # prepare for mapping
    FD_ = FD[FD["outfn"].apply(lambda fn: not os.path.exists(fn))]
    # inputs = FD_[["菜名", "outfn"]].values  # convert to tuple for map function
    print(f"{len(FD_)} files remaining for processing...")

    model_name = "gemini-2.5-pro"  # "gemini-2.5-flash-preview-04-17-nothinking" #"gemini-2.5-pro" #"qwen-vl-max-latest"

    t_start = time.time()

    batch_size = 50
    batch_list = [FD["菜名"][i:i + batch_size].tolist() for i in range(0, len(FD), batch_size)]
    unfinished_batch_num_list = get_unfinished_tasks(fd_target, len(batch_list))
    unfinished_batch_list = [batch_list[i] for i in unfinished_batch_num_list]
    print(f"{len(unfinished_batch_list)} batches remaining for processing...")
    
    failed_batches = []    
    if unfinished_batch_list:
        message_list = create_context(model_name)

        create_context_time = time.time() - t_start
        print(f"Context creation time: {create_context_time:.2f} seconds")
        
        with Pool(150) as pool:
            # 用于存储所有异步任务
            async_results = []

            for batch_num, foodname_list in tqdm(zip(unfinished_batch_num_list, unfinished_batch_list)):
                print(f"Processing batch {batch_num} with {len(foodname_list)} items...")
                outfn = f'{batch_num}.pk'
                # 存储异步任务结果
                result = pool.apply_async(
                    func=process_foodname, 
                    args=(model_name, batch_num, foodname_list, message_list, fd_target/outfn)
                )
                async_results.append(result)
                # break
                
            # 等待所有任务完成并获取结果
            print("\n等待所有异步任务完成...")
            for i, async_result in enumerate(tqdm(async_results)):
                try:
                    result = async_result.get(timeout=300)  # 300秒超时
                    if not result:  # 如果返回空字符串，说明处理失败
                        failed_batches.append(unfinished_batch_num_list[i])
                except Exception as e:
                    print(f"Batch {unfinished_batch_num_list[i]} failed: {str(e)}")
                    failed_batches.append(unfinished_batch_num_list[i])
            
            # 等待所有进程完成
            pool.close()
            pool.join()
        
        print("\n所有异步任务已完成...")

    # 从pk文件中读取并合并结果
    print("\n开始合并结果...")
    all_results = []
    pk_files = list(fd_target.glob("*.pk"))
    print(f"\n开始处理 {len(pk_files)} 个pk文件...")
    
    fail_pk_files = []
    for pk_file in tqdm(pk_files):
        try:
            with open(pk_file, 'rb') as f:
                data = pk.load(f)
                if isinstance(data, list) and len(data) == 2:
                    batch_num, response_text = data
                    # 解析CSV数据
                    df = parse_csv_from_response(response_text, batch_size)
                    if df is not None:
                        all_results.append((batch_num, df))
                    else:
                        print(f"文件 {pk_file} 中未能解析出有效的DataFrame")
                        fail_pk_files.append(pk_file)
        except Exception as e:
            print(f"处理文件 {pk_file} 时出错: {str(e)}")
            fail_pk_files.append(pk_file)
        # break
    
    # # 合并所有DataFrame
    # if all_results:
    #     print(f"\n成功解析 {len(all_results)} 个结果文件")
    #     # 按batch_num排序
    #     all_results.sort(key=lambda x: x[0])
    #     # 提取所有DataFrame并合并
    #     all_df = pd.concat([df for _, df in all_results], ignore_index=True)
        
    #     # 保存合并后的结果
    #     output_path = fd_target / "merged_results.csv"
    #     all_df.to_csv(output_path, index=False, encoding='utf-8')
    #     print(f"保存合并结果到: {output_path}")
    #     print(f"总计处理了 {len(all_df)} 条数据")
    # else:
    #     print("没有找到有效的数据可以合并")
        
    # # 删除临时文件（可选）
    # for pk_file in pk_files:
    #     pk_file.unlink()

    t_end = time.time()
    t = t_end - t_start
    
    print("=== 执行统计 ===")
    print(f"* 待处理批次数: {len(unfinished_batch_list)}")
    if len(failed_batches) > 0:
        print(f"* 失败批次数: {len(failed_batches)}")
        print(f"* 失败批次编号: {failed_batches}")
        
    print(f"* 待解析文件数: {len(pk_files)}")
    print(f"* 成功解析文件数: {len(all_results)}")
    print(f"* 失败解析文件数: {len(fail_pk_files)}")
    print(f"* 失败解析文件: {fail_pk_files}")
    print(f"* 处理数据条数: {len(all_df) if 'all_df' in locals() else 0}")
    print(f"* 总执行时间: {t:.2f} 秒")
