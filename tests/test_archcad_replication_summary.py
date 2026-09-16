import copy
import pytest

from scripts.research.summarize_archcad_replication import comparable_signature, summarize


def config():
    return dict(selection_seed=1337, train=['a'], val=['b'], model={'dim': 64}, representation='v1',
                split_fingerprint='fixed', epochs=20, lr=.0003, max_primitives=1024, queries=64,
                training={'batch': 1}, decoder={'threshold': .1}, metric_profile='local-v1',
                implementation_sha256={'model': 'hash'}, seed=1337)


def test_replication_allows_model_seed_but_not_data_or_code_changes():
    a, b = config(), config()
    b['seed'] = 2024
    assert comparable_signature(a) == comparable_signature(b)
    b['val'] = ['different-image']
    with pytest.raises(ValueError, match='confounded'):
        summarize([{'config': a}, {'config': b}])
    b = copy.deepcopy(a)
    b['seed'] = 2024
    b['implementation_sha256']['model'] = 'changed-code'
    with pytest.raises(ValueError, match='confounded'):
        summarize([{'config': a}, {'config': b}])


def test_repeated_seed_is_not_replication():
    with pytest.raises(ValueError, match='duplicate'):
        summarize([{'config': config()}, {'config': config()}])
