"""A replayed depth frame must not overwrite the previous semantic token."""

from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

import voxtream.model as model_module


def test_depth_graph_frame_has_independent_storage():
    model = model_module.Model.__new__(model_module.Model)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(num_phone_states=1, audio_vocab_size=8, audio_pad_size=0)
    model.temp_former = SimpleNamespace(caches_are_enabled=lambda: True)
    model.temp_former_causal_mask = torch.ones(1, 1, dtype=torch.bool)
    model.dep_former = SimpleNamespace(reset_caches=lambda: None)
    model.sink_attention = SimpleNamespace(
        get_context=lambda **kwargs: (kwargs["input_pos"], None),
        append_step=lambda *args: None,
    )
    model.sem_head = torch.nn.Identity()
    model._temp_former = lambda h, _pos, _mask: h
    model._embed_audio_tokens = MethodType(
        lambda _self, _tokens: torch.zeros(2, 1, 1, 4), model
    )

    graph_output = torch.zeros(1, 2, dtype=torch.long)
    replay = 0

    def depth_graph(_hidden, _c0, _speaker):
        nonlocal replay
        replay += 1
        graph_output[0, 0] = replay
        return graph_output

    model._depth_graph = depth_graph
    inputs = dict(
        config=SimpleNamespace(),
        phone_emb=torch.zeros(2, 1, 4),
        audio_tokens=torch.zeros(2, 1, 1, dtype=torch.long),
        input_pos=torch.zeros(2, 1, dtype=torch.long),
        spk_embeddings=torch.zeros(2, 1, 4),
        cfg_gamma=1.5,
        spk_rate_weight=1.0,
        cur_spk_rate_cnt=torch.zeros(1),
        target_spk_rate_cnt=torch.ones(1),
    )
    with patch.object(
        model_module,
        "sample_semantic_token",
        return_value=(torch.zeros(1, 1, dtype=torch.long), None, None),
    ):
        first, _, _ = model.generate_frame(**inputs)
        previous_semantic_token = first[:, :1]
        second, _, _ = model.generate_frame(**inputs)

    assert first[0, 0].item() == previous_semantic_token[0, 0].item() == 1
    assert second[0, 0].item() == graph_output[0, 0].item() == 2
