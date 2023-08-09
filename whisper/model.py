import base64
import gzip
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Conv1d, LayerNorm, Linear

from .decoding import KVCacheEntry, KVBlockCacheEntry
from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .transcribe import transcribe as transcribe_function


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment *
                               torch.arange(channels // 2))
    scaled_time = torch.arange(
        length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        cache_entry: Optional[KVCacheEntry] = None,
    ) -> Tuple[Tensor, Tensor, KVCacheEntry]:
        q = self.query(x)

        if cache_entry is None or xa is None:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = cache_entry.k
            v = cache_entry.v

        if cache_entry is not None and xa is None:
            # This is a self-attention call, so we need to prepend the cached key/value tensors.
            k = torch.cat([cache_entry.k, k], dim=1)
            v = torch.cat([cache_entry.v, v], dim=1)

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk, KVCacheEntry(k=k, v=v)

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(q.shape[0], q.shape[1], self.n_head, -
                   1).permute(0, 2, 1, 3) * scale
        k = k.view(k.shape[0], k.shape[1], self.n_head, -
                   1).permute(0, 2, 3, 1) * scale
        v = v.view(v.shape[0], v.shape[1], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        block_cache_entry: Optional[KVBlockCacheEntry] = None,
    ) -> Tuple[Tensor, Optional[KVBlockCacheEntry]]:
        attn_kv_tensor: Optional[KVCacheEntry] = None
        cross_attn_kv_tensor: Optional[KVCacheEntry] = None
        if block_cache_entry is not None:
            attn_kv_tensor = block_cache_entry.attn
            cross_attn_kv_tensor = block_cache_entry.cross_attn

        attn_result = self.attn(self.attn_ln(
            x), mask=mask, cache_entry=attn_kv_tensor)
        x = x + attn_result[0]
        attn_kv_tensor = attn_result[2]
        if self.cross_attn is not None:
            cross_attn_result = self.cross_attn(
                self.cross_attn_ln(x), xa, cache_entry=cross_attn_kv_tensor)
            x = x + cross_attn_result[0]
            cross_attn_kv_tensor = cross_attn_result[2]
        x = x + self.mlp(self.mlp_ln(x))
        if attn_kv_tensor is None or cross_attn_kv_tensor is None:
            return x, None
        else:
            return x, KVBlockCacheEntry(attn=attn_kv_tensor, cross_attn=cross_attn_kv_tensor)


class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3,
                            stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)[0]

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[List[KVBlockCacheEntry]] = None) -> Tuple[Tensor, List[KVBlockCacheEntry]]:
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        offset = 0
        if kv_cache is not None:
            offset = kv_cache[0].attn.k.shape[1]
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset: offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)

        new_block_cache_entries: List[KVBlockCacheEntry] = []
        for i, block in enumerate(self.blocks):
            inner_kv_cache: Optional[KVBlockCacheEntry] = None
            if kv_cache is not None:
                inner_kv_cache = kv_cache[i]
            x, new_block_cache_entry = block(
                x, xa, mask=self.mask, block_cache_entry=inner_kv_cache)
            assert new_block_cache_entry is not None
            new_block_cache_entries.append(new_block_cache_entry)

        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits, new_block_cache_entries


def _rearrange_kv_cache_entry(source_indicies: Tensor, entry: KVCacheEntry) -> KVCacheEntry:
    return KVCacheEntry(k=entry.k[source_indicies], v=entry.v[source_indicies])


def _rearrange_kv_block_cache_entry(source_indicies: Tensor, entry: KVBlockCacheEntry) -> KVBlockCacheEntry:
    return KVBlockCacheEntry(
        attn=_rearrange_kv_cache_entry(source_indicies, entry.attn),
        cross_attn=entry.cross_attn)


class InferenceDecoder(nn.Module):
    initial_token_length: int
    kv_cache: Optional[List[KVBlockCacheEntry]]

    def __init__(self, decoder: TextDecoder):
        super().__init__()

        self.decoder = decoder
        self.initial_token_length = 0
        self.kv_cache = None

    def forward(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        if tokens.shape[-1] > self.initial_token_length:
            # only need to use the last token except in the first forward pass
            tokens = tokens[:, -1:]

        x, self.kv_cache = self.decoder(
            tokens, audio_features, kv_cache=self.kv_cache)

        return x

    @torch.jit.export
    def initialize_decoding(self, initial_token_length: int):
        self.initial_token_length = initial_token_length
        self.kv_cache = None

    @torch.jit.export
    def cleanup_caching(self):
        self.kv_cache = None

    @torch.jit.export
    def rearrange_kv_cache(self, source_indices: Tensor):
        if not torch.equal(source_indices, torch.range(0, source_indices.size()[0] - 1)) and self.kv_cache is not None:
            kv_cache: Optional[List[KVBlockCacheEntry]] = self.kv_cache
            assert kv_cache is not None
            for i in range(len(self.kv_cache)):
                kv_cache[i] = _rearrange_kv_block_cache_entry(
                    source_indices, kv_cache[i])


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = torch.jit.script(AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        ))
        self.decoder = torch.jit.script(TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        ))

        # use the last half layers for alignment by default; see `set_alignment_heads()` below
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2:] = True
        self.register_buffer(
            "alignment_heads", all_heads.to_sparse(), persistent=False)

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer(
            "alignment_heads", mask.to_sparse(), persistent=False)

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)[0]

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865
    
    def get_inference_decoder(self) -> InferenceDecoder:
        return torch.jit.script(InferenceDecoder(self.decoder))

    def install_kv_cache_hooks(self, cache: Optional[Dict[int, Tensor]] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module: nn.Module, _, output):
            if id(module) not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[id(module)] = output
            else:
                cache[id(module)] = torch.cat(
                    [cache[id(module)], output], dim=1).detach()
            return cache[id(module)]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function

    def to_torch_script(self) -> (torch.jit.ScriptModule, torch.jit.ScriptModule):
        return (torch.jit.script(self.encoder), torch.jit.script(self.decoder))
