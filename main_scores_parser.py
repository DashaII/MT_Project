import configs
import pandas as pd

pd.set_option('display.max_columns', None)


def get_data(file_name, col_name):
    df = pd.read_csv(file_name, sep='\t', header=None, names=[col_name])
    df.index = df.index + 1
    return df


def get_score_data(file_name):
    df = pd.read_csv(file_name, sep='\t', header=None,
                     names=['MetricName', 'FromToLang', 'DataSource', 'Domain', 'DocID',
                            'Ref', 'System', 'SegmentID', 'Score'])
    return df


def read_lines_to_df(file_name, col_name):
    with open(file_name, encoding='utf-8') as f:
        result = [line.strip() for line in f.readlines()]
    df = pd.DataFrame(result, columns=[col_name])
    df.index = df.index + 1
    return df


def save_to_txt_file(df, file_name):
    df.to_csv(file_name, sep='\t', index=False)


def merge_scores(from_lang: str, to_lang: str, score_file: str):
    df = get_score_data(score_file)

    from_to = from_lang + "-" + to_lang

    gpt_df = df[(df['System'] == configs.GPT4) & (df['FromToLang'] == from_to)].rename(
        columns={'Score': 'GPT4_Score'})
    onlineb_df = df[(df['System'] == configs.ONLINEB) & (df['FromToLang'] == from_to)].rename(
        columns={'Score': 'OnlineB_Score'})
    onlinew_df = df[(df['System'] == configs.ONLINEW) & (df['FromToLang'] == from_to)].rename(
        columns={'Score': 'OnlineW_Score'})

    merged_df = gpt_df.merge(onlineb_df[['SegmentID', 'OnlineB_Score']], on='SegmentID', how='left')
    merged_df = merged_df.merge(onlinew_df[['SegmentID', 'OnlineW_Score']], on='SegmentID', how='left')

    return merged_df


def merge_data(scores_df, from_lang: str, to_lang: str):
    source_file_name = configs.SOURCE_FILE_NAME + from_lang + '-' + to_lang + '.src.' + from_lang
    ref_file_name = configs.REF_FILE_NAME + from_lang + '-' + to_lang + '.ref.refA.' + to_lang
    output_gpt_file_name = configs.OUTPUT_FILE_NAME + from_lang + '-' + to_lang + '.hyp.' + configs.GPT4 + '.' + to_lang
    output_onlineB_file_name = configs.OUTPUT_FILE_NAME + from_lang + '-' + to_lang + '.hyp.' + configs.ONLINEB + '.' + to_lang
    output_onlineW_file_name = configs.OUTPUT_FILE_NAME + from_lang + '-' + to_lang + '.hyp.' + configs.ONLINEW + '.' + to_lang

    df_scr = read_lines_to_df(source_file_name, 'Source')
    df_ref = read_lines_to_df(ref_file_name, 'Reference')
    df_gpt_out = read_lines_to_df(output_gpt_file_name, 'GPT_Output')
    df_onlineb_out = read_lines_to_df(output_onlineB_file_name, 'OnlineB_Output')
    df_onlinew_out = read_lines_to_df(output_onlineW_file_name, 'OnlineW_Output')

    # index serves as merging key for right table
    merged_scr = scores_df.merge(df_scr[['Source']], left_on='SegmentID', right_index=True, how='left')
    merged_ref = merged_scr.merge(df_ref[['Reference']], left_on='SegmentID', right_index=True, how='left')
    merged_gpt_out = merged_ref.merge(df_gpt_out[['GPT_Output']], left_on='SegmentID', right_index=True, how='left')
    merged_onlineb_out = merged_gpt_out.merge(df_onlineb_out[['OnlineB_Output']], left_on='SegmentID', right_index=True,
                                              how='left')
    merged_result = merged_onlineb_out.merge(df_onlinew_out[['OnlineW_Output']], left_on='SegmentID', right_index=True,
                                             how='left')

    return merged_result


def get_auto_score_file(from_lang: str, to_lang: str, score_file: str):
    """
    Gets From and To language parameters.
    Creates .txt and .csv files with GPT4, OnlineB and OnlineW scores
    aligned with Source text, Reference translation and GPT4, OnlineB and OnlineW translations
    """
    merged_scores = merge_scores(from_lang, to_lang, score_file)
    merged_data = merge_data(merged_scores, from_lang, to_lang)

    save_to_txt_file(merged_data, from_lang + '_' + to_lang + '_auto_scores.txt')
    merged_data.to_csv(from_lang + '_' + to_lang + '_auto_scores.csv', index=False)


def unify_segment_ids(file_name):
    """
        Gets Metadata file from general test.
        Creates mapping table where pair DocumentID and SegmentCount (= from 1 to number_of_segments_in_doc)
        is mapped to SegmentID (= from 1 to total_number_of_segments)
    """
    with open(file_name, encoding='utf-8') as f:
        result = [line.strip() for line in f.readlines()]

    doc_ids = [line.split()[1] for line in result]
    segment_nums = []
    segment_ids = []
    id_count = 0
    num_count = 0
    for i, doc_id in enumerate(doc_ids):
        id_count += 1
        if i == 0 or doc_ids[i] == doc_ids[i - 1]:
            num_count += 1
        elif i > 0:
            num_count = 1
        segment_nums.append(num_count)
        segment_ids.append(id_count)

    data = {'DocumentID': doc_ids, 'SegmentCount': segment_nums, 'SegmentID': segment_ids}
    segm_df = pd.DataFrame(data)

    return segm_df


def enrich_human_score_with_segment_id(human_score_file_name, mapping_df, from_lang, to_lang):
    df = pd.read_csv(human_score_file_name)
    df.index = df.index + 1

    df = df.rename(columns={'SegmentID': 'SegmentCount'})
    df.drop(df[df['IsDocument'] == 1].index, inplace=True)
    df['SegmentCount'] = df['SegmentCount'].astype(str)
    mapping_df['SegmentCount'] = mapping_df['SegmentCount'].astype(str)
    mapping_df['SegmentID'] = mapping_df['SegmentID'].astype(str)

    # add SegmentID column from mapping_df based on 2 keys: DocumentID and SegmentCount
    merged_df = df.merge(mapping_df[['DocumentID', 'SegmentCount', 'SegmentID']],
                         on=['DocumentID', 'SegmentCount'], how='left')
    # filter GPT4 only, from-to languages and good estimates (type=tgt is good, type=bad is bad)
    merged_df = merged_df[(merged_df['SystemID'] == configs.GPT4) & (merged_df['SourceLanguage'] == from_lang) &
                          (merged_df['TargetLanguage'] == to_lang) & (merged_df['Type'] == configs.GOOD_TYPE)]
    # drop lines with SegmentID=NaN which may occur since human score contains
    # all source documents (challenge and general), but mapping_df has general docs only
    df.drop(df[df['IsDocument'] == 1].index, inplace=True)
    merged_df.dropna(subset=['SegmentID'], inplace=True)

    merged_df.to_csv("test.csv")

    return df


def get_manual_score_file(from_lang: str, to_lang: str):
    pass


if __name__ == '__main__':
    get_auto_score_file(from_lang=configs.EN, to_lang=configs.RU, score_file=configs.SCORE_COMET_FILE_NAME)
    get_auto_score_file(from_lang=configs.EN, to_lang=configs.RU, score_file=configs.SCORE_BLEU_FILE_NAME)

    # get_auto_score_file(from_lang=configs.EN, to_lang=configs.ZH)

    # map_seg = unify_segment_ids(configs.METADATA_FILE_NAME)
    # map_seg.to_csv("mapping.csv")
    # enrich_human_score_with_segment_id(configs.SCORE_HUMAN, map_seg, configs.ENG, configs.ZHO)
