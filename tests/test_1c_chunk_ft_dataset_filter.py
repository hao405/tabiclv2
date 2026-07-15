from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_ft_module():
    module_path = Path(__file__).resolve().parent.parent / "1C_Chunk_FT.py"
    spec = importlib.util.spec_from_file_location("tabicl_chunk_ft_filter_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_exclude_ttt_inactive_dataset_dirs_uses_builtin_dataset_names(tmp_path: Path):
    module = load_ft_module()
    data_root = tmp_path / "data"
    for dataset_name in [
        "BLE_RSSI_dataset_for_Indoor_localization",
        "ASP-POTASSCO-classification",
        "UJI_Pen_Characters",
    ]:
        (data_root / dataset_name).mkdir(parents=True)

    dataset_dirs = module.find_dataset_dirs(data_root)
    filtered_dirs = module.exclude_ttt_inactive_dataset_dirs(dataset_dirs)

    assert [path.name for path in filtered_dirs] == ["BLE_RSSI_dataset_for_Indoor_localization"]


def test_default_args_run_full_data200_without_inactive_filter():
    module = load_ft_module()
    args = module.build_arg_parser().parse_args([])

    assert args.data_root == "data200"
    assert args.out_dir == "result/compare/Tabiclv2_ft_ensemble32_data200"
    assert args.exclude_ttt_inactive_datasets is False


def test_inactive_filter_is_explicit_opt_in():
    module = load_ft_module()
    args = module.build_arg_parser().parse_args(["--exclude-ttt-inactive-datasets"])

    assert args.exclude_ttt_inactive_datasets is True
