#%%
DATA_ROOT = '/gpfs/home1/pz29075/Capstone/StockMixer/dataset'

#%%
DATASET_CONFIGS = {
    'NASDAQ': {
        'stock_num': 1026,
        'fea_num': 5,
        'market_num': 20,
        'lookback_length': 16,
        'valid_index': 756,
        'test_index': 1008,
        'scale_factor': 3,
        'steps': 1,
        # 새로운 KG 소스 추가 시 여기에만 경로 추가
        'kg_sources': {
            'wikidata':        'relation/wikidata/NASDAQ_wiki_relation.npy',
            'sector_industry': 'relation/sector_industry/NASDAQ_industry_relation.npy',
            'institutional':   'relation/institutional/NASDAQ_institutional_relation.npy',
            'board':           'relation/board/NASDAQ_board_relation.npy',
            'supply_chain':    'relation/supply_chain/NASDAQ_supply_chain_relation.npy',
            'llm':             'relation/llm/NASDAQ_llm_relation.npy',
            'llm_v2':          'relation/llm_v2/NASDAQ_llm_relation.npy',
            # 동적 KG: 연도별 파일 패턴 ({year} 치환)
            'news_dynamic':    'relation/news_{year}/NASDAQ_news_relation.npy',
            'llm_dynamic':     'relation/llm_dynamic_{year}/NASDAQ_llm_dynamic_relation.npy',
        },
        'news_kg_years':    [2013, 2014, 2015, 2016, 2017],
        'llm_dynamic_years': [2013, 2014, 2015, 2016, 2017],
    },
    'SP500': {
        'stock_num': 474,
        'fea_num': 5,
        'market_num': 20,
        'lookback_length': 16,
        'valid_index': 886,   # 2015-11-19 (NASDAQ과 동일 캘린더)
        'test_index': 1138,   # 2016-11-18 (NASDAQ과 동일 캘린더)
        'scale_factor': 3,
        'steps': 1,
        'kg_sources': {
            'sector_industry': 'relation/sector_industry/SP500_industry_relation.npy',
            'llm':             'relation/llm_v2/SP500_llm_relation.npy',
        },
    },
    'NYSE': {
        'stock_num': 1737,
        'fea_num': 5,
        'market_num': 20,
        'lookback_length': 16,
        'valid_index': 756,
        'test_index': 1008,
        'scale_factor': 3,
        'steps': 1,
        'kg_sources': {
            'wikidata':        'relation/wikidata/NYSE_wiki_relation.npy',
            'sector_industry': 'relation/sector_industry/NYSE_industry_relation.npy',
        },
    },
}


def get_config(market: str) -> dict:
    cfg = DATASET_CONFIGS[market].copy()
    cfg['data_root'] = DATA_ROOT
    cfg['market'] = market
    return cfg
