import argparse
import os

import pandas as pd

from packages.constants import (
    PROJECT_ROOT,
    EVAL_DATA_DIMENSION_EXCEL_PATH,
)
from packages.my_logger import setup_logging

log_dir = os.path.join(PROJECT_ROOT, "logs", "eval_res_of_llm")
logger = setup_logging(log_dir=log_dir)
from packages.file_deal import (
    read_dataset_excel,
)


if __name__ == "__main__":
    # 参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--score_res_input",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-d",
        "--food_dimension_input",
        type=str,
        default=EVAL_DATA_DIMENSION_EXCEL_PATH,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
    )
    args = parser.parse_args()
    score_res_input = args.score_res_input
    food_dimension_input = args.food_dimension_input
    output = args.output if args.output else os.path.dirname(score_res_input)

    # 读取数据集
    samples_df = read_dataset_excel(score_res_input)
    dimension_df = read_dataset_excel(food_dimension_input)
    # print(samples_df)
    # print(dimension_df)
    dimensions = dimension_df.columns.tolist()
    dimensions.remove("food_name")  # 去掉菜名列, 剩下的都是维度
    logger.info(f"维度列表: {dimensions}")    
    new_main_df = pd.DataFrame(columns=[column for column in samples_df.columns if '的评分' not in column] + dimensions)
    
    for column in new_main_df.columns:
        if column in samples_df.columns:
            new_main_df[column] = samples_df[column].values
    dimension_value_dict = {dimension: {} for dimension in dimensions if dimension}
    i = 0
    for food_name, score in zip(
        samples_df["ground_truth_answer"], samples_df["菜名打分平均值"]
    ):
        if food_name == "" or pd.isna(food_name):
            continue
        for dimension in dimensions:
            try:
                dimension_val = dimension_df.loc[
                    dimension_df["food_name"] == food_name, dimension
                ].values[0]
            except Exception as e:
                print(e)
                print(f"未找到菜名 {food_name} 的维度 {dimension} 信息")
            if dimension_val not in dimension_value_dict[dimension]:
                dimension_value_dict[dimension][dimension_val] = []
            dimension_value_dict[dimension][dimension_val].append(score)
            new_main_df.iloc[i, new_main_df.columns.get_loc(dimension)] = dimension_val
            pass
        # print(dimension_value_dict)
        i += 1
        if i == 10:
            # break
            pass
    # print(new_main_df)
    # print(dimension_value_dict)
    # 处理主料种类, 每类主料拆开
    main_ingredients_set = set()
    for main_ingredients in dimension_df["main_ingredients"]:
        main_ingredients_list = [ing.strip() for ing in main_ingredients.split("/")]
        main_ingredients_set.update(main_ingredients_list)
    logger.info(f"主料种类（{len(main_ingredients_set)}种）: {main_ingredients_set}")

    # 处理每种主料对应的评分
    new_main_ingredients_dict = {}
    for main_ingredients, scores in dimension_value_dict["main_ingredients"].items():
        # print(main_ingredients, scores)
        main_ingredients_list = [ing.strip() for ing in main_ingredients.split("/")]
        for main_ingredient in main_ingredients_list:
            if main_ingredient not in new_main_ingredients_dict:
                new_main_ingredients_dict[main_ingredient] = []
            new_main_ingredients_dict[main_ingredient].extend(scores)
            # print(new_main_ingredients_dict)
    dimension_value_dict["main_ingredients"] = new_main_ingredients_dict
    # print(dimension_value_dict)

    # 计算每个维度每个取值的平均分和标准差
    output_path = os.path.join(
        output, score_res_input.replace(".xlsx", "_dimension_analysis.xlsx")
    )
    with pd.ExcelWriter(output_path) as writer:
        new_main_df.to_excel(writer, sheet_name="所有样本", index=False)
        for dimension in dimensions:
            logger.info(f"维度: {dimension}")
            df = pd.DataFrame(columns=["种类", "样本数", "平均分", "标准差"])
            for val, scores in dimension_value_dict[dimension].items():
                avg_score = sum(scores) / len(scores)
                std_score = pd.Series(scores).std()
                new_row = pd.DataFrame(
                    {
                        "种类": [val],
                        "样本数": [len(scores)],
                        "平均分": [avg_score],
                        "标准差": [std_score],
                    }
                )
                df = pd.concat([df, new_row], ignore_index=True)
                # 打印结果
                logger.info(
                    f"  种类: {val}, 样本数: {len(scores)}, 平均分: {avg_score:.3f}, 标准差: {std_score:.3f}"
                )
            # 保存结果
            df.sort_values(by="平均分", ascending=False, inplace=True)
            df.to_excel(writer, sheet_name=dimension, index=False)
    logger.info(f"结果已保存至: {output_path}")
