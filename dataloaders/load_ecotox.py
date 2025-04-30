import pandas as pd
import numpy as np
import ast

def load_ecotox_data(
    adore_path,
    chemicals_path,
    # -- Optional concentration choice --
    use_molar=True,
    mol_col="result_conc1_mean_mol_log",
    mg_col="result_conc1_mean_log",
    # -- Selfies options --
    use_selfies=False,
    selfies_path=None,
    selfies_col="selfies_embed",
    # -- Mol2Vec options --
    use_mol2vec=False,
    mol2vec_path=None,
    mol2vec_cols=None,
    # -- Fingerprint options --
    use_fingerprint=False,
    fingerprint_path=None,
    fp_col="morgan_fp",
    # -- Misc --
    shuffle=True,
    random_state=42
):
    # -------------------------------------------------------------------------
    # 1. Read base CSVs and standard merges
    # -------------------------------------------------------------------------
    adore = pd.read_csv(adore_path, low_memory=False)
    chemicals = pd.read_csv(chemicals_path, low_memory=False)

    df = adore.merge(chemicals, on='test_cas', how='left')
    df.rename(columns={
        'tax_gs': 'species',
        'test_cas': 'CAS',
        'result_obs_duration_mean': 'duration'
    }, inplace=True)

    target_col = mol_col if use_molar else mg_col
    if target_col not in df.columns:
        raise ValueError(
            f"Chosen target_col='{target_col}' not found in the merged DataFrame. "
            "Check your ADORE/chemicals CSV for the correct column name."
        )
    df.rename(columns={target_col: 'conc'}, inplace=True)

    # -------------------------------------------------------------------------
    # 2. Optionally merge SELFIES
    # -------------------------------------------------------------------------
    if use_selfies:
        if not selfies_path:
            raise ValueError("`use_selfies=True` but `selfies_path` was not provided.")
        df_selfies = pd.read_csv(selfies_path)
        df = df.merge(df_selfies, on='CAS', how='left')
        def parse_selfies(x):
            return ast.literal_eval(x) if isinstance(x, str) else x
        df[selfies_col] = df[selfies_col].apply(parse_selfies)

    # -------------------------------------------------------------------------
    # 3. Optionally merge Mol2Vec
    # -------------------------------------------------------------------------
    if use_mol2vec:
        if mol2vec_path is None:
            # Assume mol2vec columns are already present in the merged dataframe.
            if not mol2vec_cols:
                raise ValueError("When use_mol2vec=True and mol2vec_path is None, you must provide mol2vec_cols.")
            missing_cols = [col for col in mol2vec_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Mol2Vec columns missing in chemicals file: {missing_cols}")
        else:
            if not mol2vec_cols:
                raise ValueError("`use_mol2vec=True` requires mol2vec_cols.")
            df_m2v = pd.read_csv(mol2vec_path)
            columns_to_merge = ['CAS'] + mol2vec_cols
            missing_cols = [c for c in columns_to_merge if c not in df_m2v.columns]
            if missing_cols:
                raise ValueError(f"Mol2Vec file is missing columns: {missing_cols}")
            df = df.merge(df_m2v[columns_to_merge], on='CAS', how='left')

    # -------------------------------------------------------------------------
    # 4. Optionally merge Fingerprints
    # -------------------------------------------------------------------------
    if use_fingerprint:
        if not fingerprint_path:
            raise ValueError("`use_fingerprint=True` but `fingerprint_path` was not provided.")
        df_fp = pd.read_csv(fingerprint_path)
        if 'CAS' not in df_fp.columns or fp_col not in df_fp.columns:
            raise ValueError(f"Fingerprint file must contain 'CAS' and '{fp_col}' columns. Found: {list(df_fp.columns)}")
        df = df.merge(df_fp[['CAS', fp_col]], on='CAS', how='left')
        def parse_fp(x):
            return ast.literal_eval(x) if isinstance(x, str) else x
        df[fp_col] = df[fp_col].apply(parse_fp)

    # -------------------------------------------------------------------------
    # 5. Clean up DataFrame
    # -------------------------------------------------------------------------
    df['species'] = pd.Categorical(df['species'])
    df['CAS'] = pd.Categorical(df['CAS'])
    df['duration'] = df['duration'].astype(float)

    if shuffle:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 6. Center the target
    # -------------------------------------------------------------------------
    y_mean = df['conc'].mean()
    df['conc_centered'] = df['conc'] - y_mean

    return df, df['conc_centered'].values
