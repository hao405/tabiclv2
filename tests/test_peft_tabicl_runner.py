from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "PEFT_Tabicl" / "1C_Chunk_PEFT.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_peft_module():
    spec = importlib.util.spec_from_file_location("peft_tabicl_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeEncoder(torch.nn.Module):
    def __init__(self, num_blocks: int = 3):
        super().__init__()
        self.blocks = torch.nn.ModuleList([FakeTransformerBlock() for _ in range(num_blocks)])


class FakeTransformerBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim=4, num_heads=2, dropout=0.0, batch_first=True)
        self.linear1 = torch.nn.Linear(4, 8)
        self.linear2 = torch.nn.Linear(8, 4)
        self.norm1 = torch.nn.LayerNorm(4)
        self.norm2 = torch.nn.LayerNorm(4)


class FakeColEmbedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_linear = torch.nn.Linear(1, 4)
        self.y_encoder = torch.nn.Linear(2, 4)
        self.tf_col = FakeEncoder(num_blocks=1)
        self.other = torch.nn.Linear(4, 4)


class FakeRowInteractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tf_row = FakeEncoder(num_blocks=2)
        self.out_ln = torch.nn.LayerNorm(4)


class FakeIclPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tf_icl = FakeEncoder(num_blocks=3)
        self.ln = torch.nn.LayerNorm(4)
        self.y_encoder = torch.nn.Linear(2, 4)
        self.decoder = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.GELU(), torch.nn.Linear(8, 2))


class FakeTabICL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.col_embedder = FakeColEmbedder()
        self.row_interactor = FakeRowInteractor()
        self.icl_predictor = FakeIclPredictor()


class FakeClassifier:
    def __init__(self, model):
        self.model_ = model


class FakeTabPFNV2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_group_embedder = torch.nn.Linear(2, 4)
        self.target_embedder = torch.nn.Linear(2, 4)
        self.feature_positional_embedding_embeddings = torch.nn.Linear(1, 4)
        self.blocks = torch.nn.ModuleList([FakeTransformerBlock() for _ in range(3)])
        self.output_projection = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.GELU(),
            torch.nn.Linear(8, 2),
        )


def trainable_names(model):
    return {name for name, param in model.named_parameters() if param.requires_grad}


def test_lora_only_trains_lora_params_and_merged_state_excludes_temporary_keys():
    peft = load_peft_module()
    model = FakeTabICL()
    classifier = FakeClassifier(model)

    trainable_params = peft._configure_ttt_trainable_params(
        classifier,
        peft.TTTConfig(
            enabled=True,
            peft_method="lora",
            peft_targets="row,icl",
            lora_rank=2,
            lora_alpha=4.0,
        ),
    )

    names = trainable_names(model)
    assert trainable_params
    assert names
    assert all("._ttt_lora_" in name for name in names)
    assert not any(param.requires_grad for param in model.col_embedder.parameters())
    assert model.row_interactor.tf_row.blocks[0].attn._ttt_lora_in_proj_installed is True
    assert model.icl_predictor.tf_icl.blocks[0].attn._ttt_lora_in_proj_installed is True

    merged = peft._merged_ttt_state_dict(model)
    assert not any("_ttt_lora_" in name or ".parametrizations." in name for name in merged)
    assert "row_interactor.tf_row.blocks.0.linear1.weight" in merged
    assert "icl_predictor.tf_icl.blocks.0.attn.in_proj_weight" in merged


def test_last_layers_only_unfreezes_final_icl_block_and_head():
    peft = load_peft_module()
    model = FakeTabICL()
    classifier = FakeClassifier(model)

    peft._configure_ttt_trainable_params(
        classifier,
        peft.TTTConfig(
            enabled=True,
            peft_method="last_layers",
            peft_targets="icl",
            last_n_icl_blocks=1,
        ),
    )

    names = trainable_names(model)
    assert names
    assert any(name.startswith("icl_predictor.tf_icl.blocks.2.") for name in names)
    assert any(name.startswith("icl_predictor.decoder.") for name in names)
    assert any(name.startswith("icl_predictor.ln.") for name in names)
    assert not any(name.startswith("icl_predictor.tf_icl.blocks.0.") for name in names)
    assert not any(name.startswith("icl_predictor.tf_icl.blocks.1.") for name in names)
    assert not any(name.startswith("row_interactor.") for name in names)
    assert not any(name.startswith("col_embedder.") for name in names)


def test_ln_head_embedding_unfreezes_only_layernorm_head_and_embedding_params():
    peft = load_peft_module()
    model = FakeTabICL()
    classifier = FakeClassifier(model)

    peft._configure_ttt_trainable_params(
        classifier,
        peft.TTTConfig(
            enabled=True,
            peft_method="ln_head_embedding",
            peft_targets="col,row,icl",
        ),
    )

    names = trainable_names(model)
    assert names
    assert any(name.startswith("col_embedder.in_linear.") for name in names)
    assert any(name.startswith("col_embedder.y_encoder.") for name in names)
    assert any(name.startswith("row_interactor.out_ln.") for name in names)
    assert any(name.startswith("icl_predictor.decoder.") for name in names)
    assert any(name.startswith("icl_predictor.y_encoder.") for name in names)
    assert any("norm" in name or ".ln." in name for name in names)
    assert not any(name.startswith("row_interactor.tf_row.blocks.0.linear") for name in names)
    assert not any(name.startswith("col_embedder.other.") for name in names)


def test_tabpfn_lora_only_trains_low_rank_parameters():
    peft = load_peft_module()
    model = FakeTabPFNV2()

    trainable = peft._configure_tabpfn_trainable_params(
        model,
        peft.TTTConfig(enabled=True, peft_method="lora", lora_rank=2, lora_alpha=4.0),
    )

    names = trainable_names(model)
    assert trainable
    assert names
    assert all("._ttt_lora_" in name for name in names)
    assert model.blocks[0].attn._ttt_lora_in_proj_installed is True
    assert not any("_ttt_lora_" in name for name in peft._merged_ttt_state_dict(model))


def test_tabpfn_last_layers_unfreezes_last_block_and_output_head():
    peft = load_peft_module()
    model = FakeTabPFNV2()

    peft._configure_tabpfn_trainable_params(
        model,
        peft.TTTConfig(enabled=True, peft_method="last_layers", last_n_icl_blocks=1),
    )

    names = trainable_names(model)
    assert any(name.startswith("blocks.2.") for name in names)
    assert any(name.startswith("output_projection.") for name in names)
    assert not any(name.startswith("blocks.0.") for name in names)
    assert not any(name.startswith("blocks.1.") for name in names)
    assert not any(name.startswith("feature_group_embedder.") for name in names)


def test_tabpfn_ln_head_embedding_selects_only_norm_embedding_and_head():
    peft = load_peft_module()
    model = FakeTabPFNV2()

    peft._configure_tabpfn_trainable_params(
        model,
        peft.TTTConfig(enabled=True, peft_method="ln_head_embedding"),
    )

    names = trainable_names(model)
    assert any(name.startswith("feature_group_embedder.") for name in names)
    assert any(name.startswith("target_embedder.") for name in names)
    assert any(name.startswith("output_projection.") for name in names)
    assert any("norm" in name for name in names)
    assert not any(name.startswith("blocks.0.linear") for name in names)
    assert not any(name.startswith("blocks.0.attn.q_projection") for name in names)


def test_matrix_expansion_is_model_major_and_has_six_trials():
    peft = load_peft_module()
    assert peft.expand_matrix_trials("all", "all") == [
        ("tabiclv2", "lora"),
        ("tabiclv2", "last_layers"),
        ("tabiclv2", "ln_head_embedding"),
        ("tabpfnv2", "lora"),
        ("tabpfnv2", "last_layers"),
        ("tabpfnv2", "ln_head_embedding"),
    ]


def test_matrix_command_routes_one_concrete_trial(tmp_path):
    peft = load_peft_module()
    args = peft.build_arg_parser().parse_args(
        [
            "--model-family",
            "all",
            "--ttt-peft-method",
            "all",
            "--workers",
            "1",
            "--gpu-groups",
            "2",
        ]
    )
    command = peft.build_matrix_trial_command(
        args,
        model_family="tabpfnv2",
        peft_method="last_layers",
        trial_dir=tmp_path / "trial",
    )
    joined = " ".join(command)
    assert "--model-family tabpfnv2" in joined
    assert "--ttt-peft-method last_layers" in joined
    assert "--gpu-groups 2" in joined
    assert f"--out-dir {tmp_path / 'trial'}" in joined


def test_tabpfn_peft_disables_reentrant_activation_checkpointing(tmp_path):
    peft = load_peft_module()
    module = peft._load_tabpfn_runner_module()
    args = peft.build_arg_parser().parse_args(
        [
            "--model-family",
            "tabpfnv2",
            "--out-dir",
            str(tmp_path / "trial"),
            "--workers",
            "1",
            "--gpu-groups",
            "2",
        ]
    )
    tabpfn_args = peft._tabpfn_args_from_common(args, module)
    assert tabpfn_args.ttt_activation_checkpointing is False


def test_resume_requires_success_config_and_all_artifacts(tmp_path):
    peft = load_peft_module()
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    (trial_dir / "all_classification_results.csv").write_text(
        "status,ttt_applied,accuracy\nok,True,0.5\n",
        encoding="utf-8",
    )
    (trial_dir / "summary.txt").write_text("ok_count: 1\n", encoding="utf-8")
    (trial_dir / "run_config.json").write_text(
        json.dumps({"status": "success"}),
        encoding="utf-8",
    )
    assert peft._trial_is_complete(trial_dir) is True
    (trial_dir / "summary.txt").unlink()
    assert peft._trial_is_complete(trial_dir) is False


def test_tabpfn_finetuning_base_exposes_noop_pre_optimizer_hook():
    source = (
        REPO_ROOT
        / "baseline_compare/TabPFN-main/src/tabpfn/finetuning/finetuned_base.py"
    ).read_text(encoding="utf-8")
    assert "def _configure_model_for_optimization" in source
    assert "self._configure_model_for_optimization(self.finetuned_estimator_.model_)" in source
