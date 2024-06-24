import json
import os
import configs
import pandas as pd
import main_scores_parser


def read_all_json_from_dir(dir_name):
    data = {}
    # Iterate over every file in the directory
    for filename in os.listdir(dir_name):
        # Check if the file is a JSON file
        if filename.endswith('.json'):
            # Construct the full file path
            filepath = os.path.join(dir_name, filename)
            with open(filepath, 'r', encoding="utf-8") as file:
                data[filename] = json.load(file)

    return data


def transform_dict_data(manual_scores_dict):
    """
    Returns dataframe of manual scores for GPT4 only
    """
    data_list = []
    for file, pool in manual_scores_dict.items():
        for task in pool:
            for i, input_value in enumerate(task['input_values']):
                gpt_flag = False
                # filter out non-GPT4 translations and bad translations
                if input_value['itemType'] == configs.GOOD_TYPE and input_value['targetID'] == configs.GPT4:
                    gpt_flag = True
                    data_row = [
                        file,
                        task['user_id'],
                        task['task_id'],
                        # because docId has format "docId#1-5", where 1-5 is number of sentences in the doc
                        input_value['documentID'].split('#')[0],
                        input_value['itemID']+1,
                        input_value['targetID'],
                        input_value['itemType'],
                        task['output_values']['result'][i]['value'],
                        task['output_values']['result'][i]['src'],
                        task['output_values']['result'][i]['tgt'],
                    ]
                    data_list.append(data_row)
            # remove last item because it's an aggregation (the entire document, not just one segment)
            if data_list and gpt_flag:
                data_list.pop()

    df = pd.DataFrame(data_list, columns=['RawFileName', 'UserID', 'TaskID', 'DocumentID', 'SegmentCount', 'TargetID',
                                          'ItemType', 'ScoreValue', 'Src', 'Tgt'])
    return df


def add_segment_id(mapping_df, manual_scores_df):
    # add SegmentID column from mapping_df based on 2 keys: DocumentID and SegmentCount
    merged_df = manual_scores_df.merge(mapping_df[['DocumentID', 'SegmentCount', 'SegmentID']],
                         on=['DocumentID', 'SegmentCount'], how='left')

    # drop rows with NaN segment ids
    merged_df.dropna(subset=['SegmentID'], inplace=True)
    merged_df['SegmentID'] = merged_df['SegmentID'].astype(int)

    return merged_df


def get_scores_for_segment_id(manual_df):
    segment_ids = manual_df['SegmentID'].unique()

    # res = manual_df.groupby('SegmentID')['ScoreValue'].agg(list)
    scores_list = []
    for id in segment_ids:
        scores = manual_df[manual_df['SegmentID'] == id]['ScoreValue'].tolist()
        # create a list: SegmentID, list of scores, average score, min score
        scores_list.append([id, scores, int(sum(scores)/len(scores)), min(scores)])

    df = pd.DataFrame(scores_list, columns=['SegmentID', 'Scores', 'AvgScore', 'MinScore'])

    return df


if __name__ == '__main__':
    manual_scores = read_all_json_from_dir(configs.SCORE_MANUAL_FOLDER)
    manual_scores_df = transform_dict_data(manual_scores)

    main_scores_parser.save_to_txt_file(manual_scores_df, "manual_score.txt")

    map_seg = main_scores_parser.unify_segment_ids(configs.METADATA_FILE_NAME)
    manual_scores_df = add_segment_id(map_seg, manual_scores_df)

    main_scores_parser.save_to_txt_file(manual_scores_df, "manual_score_with_ids.txt")

    manual_scores_df = get_scores_for_segment_id(manual_scores_df)
    main_scores_parser.save_to_txt_file(manual_scores_df, "manual_score_mapping.txt")
