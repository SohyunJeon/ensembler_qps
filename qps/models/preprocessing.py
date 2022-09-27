import pandas as pd
import numpy as np






def remove_same_values(x_data: pd.DataFrame, same_ratio: float) -> pd.DataFrame:
    same_criteria = int(x_data.shape[0] * same_ratio)
    same_val_cnt = x_data.apply(lambda x: x.value_counts().values[0])
    same_val_col = [col for col, val in zip(same_val_cnt.index, same_val_cnt.values) \
                    if val > same_criteria]
    print(f'Same column count : {len(same_val_col)}')
    df = x_data.drop(same_val_col, axis=1)
    print(f'After Remove Same value columns, shape : {df.shape}')
    return df



def drop_na(data: pd.DataFrame) -> pd.DataFrame:
    df = data.dropna(axis=1)
    df = df.dropna()
    print(f'After Remove NA, shape : {df.shape}')
    return df


def remove_high_corr(x_data: pd.DataFrame, corr_criteria: float) -> pd.DataFrame:
    corr_matrix = x_data.corr().abs()
    # 상관계수 matrix의 윗부분
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > corr_criteria)]
    df = x_data.drop(to_drop, axis=1)
    print(f'High correlation column count : {len(to_drop)}')
    return df


def fill_na_params(data: pd.DataFrame, scale_info: dict) -> (pd.DataFrame, str):
    if not all(item in data.columns for item in scale_info['X_mean'].index):
        missing_scale = np.setdiff1d(scale_info['X_mean'].index, data.columns)
        comment = f'There is lack of scale info. Fill with mean value. : {missing_scale}'
        for scale in missing_scale:
            data[scale] = scale_info['X_mean'][scale]
    else:
        comment = ''
    return data, comment
