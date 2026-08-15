import importlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from fastapi.testclient import TestClient
from torch import nn
from transformers import PreTrainedModel


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.o_proj = nn.Linear(4, 4, bias=False)
        self.head_dim = 2
        self.num_key_value_groups = 1


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 6, bias=False)
        self.up_proj = nn.Linear(4, 6, bias=False)
        self.down_proj = nn.Linear(6, 4, bias=False)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = _Mlp()


class _Llm(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer()])
        self.config = types.SimpleNamespace(rope_parameters={"rope_theta": 10_000})


class InferenceMemoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_flashinfer = types.ModuleType("flashinfer")
        fake_flashinfer.norm = types.SimpleNamespace(rmsnorm=lambda x, _w, eps: x)
        fake_flashinfer.rope = types.SimpleNamespace(
            apply_rope_pos_ids_inplace=lambda *_args, **_kwargs: None
        )
        fake_flashinfer.activation = types.SimpleNamespace(
            silu_and_mul=lambda x: torch.nn.functional.silu(x[..., : x.shape[-1] // 2])
            * x[..., x.shape[-1] // 2 :]
        )
        fake_flashinfer.BatchPrefillWithRaggedKVCacheWrapper = object
        with mock.patch.dict(sys.modules, {"flashinfer": fake_flashinfer}):
            cls.flashinfer_patch = importlib.import_module(
                "omnivoice.models.omnivoice_flashinfer"
            )
        cls.flashinfer_patch.flashinfer = fake_flashinfer

    def test_fused_forwards_drop_original_projection_parameters(self):
        torch.manual_seed(0)
        llm = _Llm()
        layer = llm.layers[0]
        hidden = torch.randn(1, 3, 4)

        q = layer.self_attn.q_proj(hidden[0]).view(3, 2, 2)
        k = layer.self_attn.k_proj(hidden[0]).view(3, 2, 2)
        v = layer.self_attn.v_proj(hidden[0]).view(3, 2, 2)
        expected_attention = layer.self_attn.o_proj((q + k + v).reshape(3, 4))
        gate = layer.mlp.gate_proj(hidden[0])
        up = layer.mlp.up_proj(hidden[0])
        expected_mlp = layer.mlp.down_proj(torch.nn.functional.silu(gate) * up)

        self.flashinfer_patch._patch_attention_forward(llm)
        self.flashinfer_patch._patch_mlp(llm)
        self.flashinfer_patch._CTX.update(
            wrapper=types.SimpleNamespace(run=lambda q, k, v: q + k + v),
            pos_ids=torch.arange(3, dtype=torch.int32),
            doc_slots=None,
        )

        attention, _ = layer.self_attn(hidden)
        mlp = layer.mlp(hidden)

        self.assertEqual(attention.shape, hidden.shape)
        self.assertEqual(mlp.shape, hidden.shape)
        torch.testing.assert_close(attention, expected_attention.unsqueeze(0))
        torch.testing.assert_close(mlp, expected_mlp.unsqueeze(0))
        parameter_names = set(dict(llm.named_parameters()))
        for name in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
            self.assertFalse(any(name in parameter for parameter in parameter_names))
        self.assertNotIn("_fi_w_qkv", layer.self_attn.state_dict())
        self.assertNotIn("_fi_w_gate_up", layer.mlp.state_dict())
        self.assertIsNone(layer.self_attn._fi_w_qkv.grad_fn)
        self.assertIsNone(layer.mlp._fi_w_gate_up.grad_fn)

        llm.to(dtype=torch.float64)
        self.assertEqual(layer.self_attn._fi_w_qkv.dtype, torch.float64)
        self.assertEqual(layer.mlp._fi_w_gate_up.dtype, torch.float64)

    def test_audio_tokenizer_stays_float32_with_half_precision_model(self):
        from omnivoice.models.omnivoice import OmniVoice

        model = types.SimpleNamespace(
            device=torch.device("cuda:0"),
            dtype=torch.float16,
            text_tokenizer=None,
            audio_tokenizer=None,
            feature_extractor=None,
            sampling_rate=None,
            duration_estimator=None,
            _asr_model_name=None,
            _asr_device=None,
        )
        tokenizer = types.SimpleNamespace()
        feature_extractor = types.SimpleNamespace(sampling_rate=24_000)

        with tempfile.TemporaryDirectory() as checkpoint:
            Path(checkpoint, "audio_tokenizer").mkdir()
            with (
                mock.patch.object(
                    PreTrainedModel, "from_pretrained", return_value=model
                ),
                mock.patch("omnivoice.models.omnivoice.AutoTokenizer.from_pretrained"),
                mock.patch(
                    "omnivoice.models.omnivoice.HiggsAudioV2TokenizerModel.from_pretrained",
                    return_value=tokenizer,
                ) as load_audio_tokenizer,
                mock.patch(
                    "omnivoice.models.omnivoice.AutoFeatureExtractor.from_pretrained",
                    return_value=feature_extractor,
                ),
            ):
                loaded = OmniVoice.from_pretrained(checkpoint, dtype=torch.float16)

        self.assertIs(loaded.audio_tokenizer, tokenizer)
        load_audio_tokenizer.assert_called_once_with(
            str(Path(checkpoint, "audio_tokenizer")),
            device_map=torch.device("cuda:0"),
            dtype=torch.float32,
        )


class HttpGenerationOptionsTest(unittest.TestCase):
    def test_service_chunk_defaults_and_form_overrides_reach_both_endpoints(self):
        calls = []

        class DummyModel:
            dtype = torch.float16
            audio_tokenizer = types.SimpleNamespace(dtype=torch.float16)
            sampling_rate = 24_000

            def generate(self, **kwargs):
                calls.append(kwargs)
                count = len(kwargs["text"]) if isinstance(kwargs["text"], list) else 1
                return [np.zeros(16, dtype=np.float32) for _ in range(count)]

        dummy_model = DummyModel()
        fake_patch = types.ModuleType("omnivoice.models.omnivoice_flashinfer")
        fake_patch.apply_flashinfer = lambda model: model
        fake_setproctitle = types.ModuleType("setproctitle")
        fake_setproctitle.setproctitle = lambda _title: None

        import omnivoice

        server_path = Path(__file__).parents[1] / "http_server.py"
        spec = importlib.util.spec_from_file_location("http_server_test", server_path)
        server = importlib.util.module_from_spec(spec)
        env = {
            "OMNIVOICE_AUDIO_CHUNK_DURATION": "12",
            "OMNIVOICE_AUDIO_CHUNK_THRESHOLD": "20",
            "OMNIVOICE_DEVICE": "cpu",
        }
        with (
            mock.patch.dict(os.environ, env),
            mock.patch.object(
                omnivoice.OmniVoice, "from_pretrained", return_value=dummy_model
            ),
            mock.patch.dict(
                sys.modules,
                {
                    "omnivoice.models.omnivoice_flashinfer": fake_patch,
                    "setproctitle": fake_setproctitle,
                },
            ),
        ):
            spec.loader.exec_module(server)

        client = TestClient(server.app)
        response = client.post(
            "/synthesize",
            data={
                "text": "hello",
                "audio_chunk_duration": "8",
                "audio_chunk_threshold": "14",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(calls[-1]["audio_chunk_duration"], 8.0)
        self.assertEqual(calls[-1]["audio_chunk_threshold"], 14.0)

        response = client.post(
            "/synthesize_batch",
            data={"items": json.dumps([{"text": "hello"}, {"text": "world"}])},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(calls[-1]["audio_chunk_duration"], 12.0)
        self.assertEqual(calls[-1]["audio_chunk_threshold"], 20.0)

        health = client.get("/health").json()
        self.assertEqual(health["dtype"], "float16")
        self.assertEqual(health["audio_tokenizer_dtype"], "float16")
        self.assertEqual(
            health["generation_defaults"],
            {"audio_chunk_duration": 12.0, "audio_chunk_threshold": 20.0},
        )


if __name__ == "__main__":
    unittest.main()
