import json
import pytest
from t2mfdf.experiments import summarize


def test_aggregate_four_targets_multiple_seeds(tmp_path):
    for dataset in range(1,5):
        for seed,value in [(42,.5),(43,.7)]:
            path = tmp_path/'full'/'fraction_0.01'/f'dataset{dataset}'/f'seed{seed}'
            path.mkdir(parents=True)
            metric = dict(accuracy=value,macro_f1=value,present_class_macro_f1=value,samples=20)
            (path/'metrics.json').write_text(json.dumps(dict(test=metric,transfer=metric,data_sha256='fixture')))
    result = summarize(tmp_path)
    assert len(result['summary']) == 8
    assert result['summary'][0]['accuracy_mean'] == pytest.approx(.6)
    assert result['summary'][0]['accuracy_std'] == pytest.approx(0.1414213562373095)
    assert result['transfer_radar_area'][0]['raw_area'] == pytest.approx(.72)
