
ABLATION_CONFIGS = {
    'sequence_only': {
        'use_cfg': False,
        'use_dfg': False,
    },
    'cfg_only': {
        'use_cfg': True,
        'use_dfg': False,
    },
    'cfg_dfg': {
        'use_cfg': True,
        'use_dfg': True,
    },
    'hierarchical_full': {
        'use_cfg': True,
        'use_dfg': True,
        'use_graph_pooling': True,
        'use_runtime_suppression': True,
    }
}
