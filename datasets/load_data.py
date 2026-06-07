#%%
import os
import pickle
import numpy as np


#%%
def load_eod_data(config: dict):
    """EOD price data + mask + ground truth + base price 로드."""
    dataset_path = os.path.join(config['data_root'], config['market'])
    with open(os.path.join(dataset_path, 'eod_data.pkl'), 'rb') as f:
        eod_data = pickle.load(f)
    with open(os.path.join(dataset_path, 'mask_data.pkl'), 'rb') as f:
        mask_data = pickle.load(f)
    with open(os.path.join(dataset_path, 'gt_data.pkl'), 'rb') as f:
        gt_data = pickle.load(f)
    with open(os.path.join(dataset_path, 'price_data.pkl'), 'rb') as f:
        price_data = pickle.load(f)
    return eod_data, mask_data, gt_data, price_data


#%%
def load_kg(config: dict):
    """
    KG 인접 행렬 로드.

    graph_type == 'gcn'  → D^{-1/2} A D^{-1/2} 정규화 adjacency
    graph_type == 'gat'  → 이진 마스크 (연결: 0.0, 비연결: -1e9)
    graph_type == 'none' → None

    새로운 KG 소스는 config['kg_sources']에 경로만 추가하면 바로 사용 가능.
    """
    graph_type = config.get('graph_type', 'none')
    if graph_type == 'none':
        return None

    kg_source = config.get('kg_source', 'wikidata')
    kg_sources = config.get('kg_sources', {})
    if kg_source not in kg_sources:
        raise ValueError(f"kg_source '{kg_source}' not in config. Available: {list(kg_sources.keys())}")

    rel_path = os.path.join(config['data_root'], config['market'], kg_sources[kg_source])
    if not os.path.exists(rel_path):
        raise FileNotFoundError(f"KG file not found: {rel_path}")

    relation = np.load(rel_path)  # [S, S, R]
    rel_shape = [relation.shape[0], relation.shape[1]]
    connected = ~np.equal(
        np.zeros(rel_shape, dtype=int),
        np.sum(relation, axis=2)
    )  # [S, S] bool, True = 연결

    if graph_type == 'gcn':
        return _normalize_adj(connected.astype(float))
    elif graph_type == 'gat':
        return np.where(connected, 0.0, -1e9).astype(np.float32)
    elif graph_type == 'hgat':
        # (S, S, R) per-relation mask: edge=0.0, no-edge=-1e9
        return np.where(relation > 0, 0.0, -1e9).astype(np.float32)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")


#%%
def load_kg_snapshots(config: dict) -> dict:
    """
    news_dynamic / llm_dynamic용: {year: processed_adj} dict 반환.
    graph_type에 따라 gcn/gat 전처리 적용.
    파일 없는 연도는 건너뜀 (job 미완료 시 안전하게 동작).
    """
    graph_type = config.get('graph_type', 'none')
    kg_source  = config.get('kg_source', 'news_dynamic')

    if kg_source == 'llm_dynamic':
        years   = config.get('llm_dynamic_years', [2013, 2014, 2015, 2016, 2017])
        pattern = config['kg_sources']['llm_dynamic']
    else:
        years   = config.get('news_kg_years', [2013, 2014, 2015, 2016, 2017])
        pattern = config['kg_sources']['news_dynamic']
    snapshots  = {}

    for yr in years:
        path = os.path.join(
            config['data_root'], config['market'],
            pattern.format(year=yr)
        )
        if not os.path.exists(path):
            print(f'[load_kg_snapshots] {yr} KG 없음: {path} — 건너뜀')
            continue
        relation  = np.load(path)
        rel_shape = [relation.shape[0], relation.shape[1]]
        connected = ~np.equal(
            np.zeros(rel_shape, dtype=int),
            np.sum(relation, axis=2)
        )
        if graph_type == 'gcn':
            snapshots[yr] = _normalize_adj(connected.astype(float))
        elif graph_type == 'gat':
            snapshots[yr] = np.where(connected, 0.0, -1e9).astype(np.float32)
        elif graph_type == 'hgat':
            snapshots[yr] = np.where(relation > 0, 0.0, -1e9).astype(np.float32)

    print(f'[load_kg_snapshots] 로드 완료: {list(snapshots.keys())}')
    return snapshots


def _normalize_adj(adj: np.ndarray) -> np.ndarray:
    """D^{-1/2} A D^{-1/2} symmetric normalization."""
    degree = np.sum(adj, axis=0)
    degree = np.where(degree == 0, 1.0, degree)
    degree = 1.0 / degree
    np.sqrt(degree, degree)
    D = np.diag(degree)
    return np.dot(np.dot(D, adj), D).astype(np.float32)
