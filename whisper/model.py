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


@torch.jit.script
class DecoderKeyValueCacheEntry:
    def __init__(self):
        new_cached_pair: Optional[Tuple[Tensor, Tensor]] = None
        self._cached_pair = new_cached_pair

    def clear(self):
        self._cached_pair = None

    def has_value(self):
        return self._cached_pair is not None
    
    def get_key_value(self) -> Tuple[Tensor, Tensor]:
        cached_pair = self._cached_pair
        assert cached_pair is not None
        return cached_pair

    def set(self, keys: Tensor, values: Tensor):
        self._cached_pair = (keys, values)

    def push_layer(self, keys: Tensor, values: Tensor) -> Tuple[Tensor, Tensor]:
        cached_pair = self._cached_pair
        if cached_pair is None:
            self._cached_pair = (keys, values)
            return keys, values
        else:
            curr_keys, curr_values = cached_pair
            new_keys = torch.cat([curr_keys, keys], dim=1)
            new_values = torch.cat([curr_values, values], dim=1)
            self._cached_pair = (new_keys, new_values)
            return new_keys, new_values

    def get_offset(self) -> int:
        cached_pair = self._cached_pair
        if cached_pair is None:
            return 0
        else:
            return cached_pair[0].shape[1]
    
    def _rearrange_kv_cache(self, source_indices: Tensor):
        cached_pair = self._cached_pair
        if cached_pair is not None:
            k, v = cached_pair
            k = k[source_indices]
            v = v[source_indices]
            self._cached_pair = (k, v)


@torch.jit.script
class DecoderKeyValueBlockCacheEntry:
    def __init__(self):
        self._attn = DecoderKeyValueCacheEntry()
        self._cross_attn = DecoderKeyValueCacheEntry()

    def clear(self):
        self._attn.clear()
        self._cross_attn.clear()

    def attn(self) -> DecoderKeyValueCacheEntry:
        return self._attn

    def cross_attn(self) -> DecoderKeyValueCacheEntry:
        return self._cross_attn

    def get_offset(self) -> int:
        return self._attn.get_offset()

    def _rearrange_kv_cache(self, source_indices: Tensor):
        self._attn._rearrange_kv_cache(source_indices)

@torch.jit.script
class DecoderKeyValueCache:
    _layers: List[DecoderKeyValueBlockCacheEntry]
    def __init__(self, n_layer: int):
        self._layers = [DecoderKeyValueBlockCacheEntry() for _ in range(n_layer)]

    def clear(self):
        for layer in self._layers:
            layer.clear()

    def get_block(self, layer: int) -> DecoderKeyValueBlockCacheEntry:
        return self._layers[layer]

    def get_offset(self):
        return self._layers[0].get_offset()

    def rearrange_kv_cache(self, source_indices: Tensor):
        for layer in self._layers:
            layer._rearrange_kv_cache(source_indices)

@torch.jit.script
def qkv_attention(
    n_head: int, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    n_batch, n_ctx, n_state = q.shape
    scale = (n_state // n_head) ** -0.25
    q = q.view(q.shape[0], q.shape[1], n_head, -
                1).permute(0, 2, 1, 3) * scale
    k = k.view(k.shape[0], k.shape[1], n_head, -
                1).permute(0, 2, 3, 1) * scale
    v = v.view(v.shape[0], v.shape[1], n_head, -1).permute(0, 2, 1, 3)

    qk = q @ k
    if mask is not None:
        qk = qk + mask[:n_ctx, :n_ctx]
    qk = qk.float()

    w = F.softmax(qk, dim=-1).to(q.dtype)
    return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class MultiHeadCrossAttention(nn.Module):
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
        xa: Tensor,
        cache_entry: DecoderKeyValueCacheEntry
    ) -> Tuple[Tensor, Tensor]:
        q = self.query(x)
        if cache_entry.has_value():
            k, v = cache_entry.get_key_value()
        else:
            k = self.key(xa)
            v = self.value(xa)
            cache_entry.set(k, v)
            
        wv, qk = qkv_attention(self.n_head, q, k, v)
        return self.out(wv), qk


class MultiHeadAttention(nn.Module):
    cache_entry: Optional[DecoderKeyValueCacheEntry]
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
        mask: Optional[Tensor] = None,
        cache_entry: Optional[DecoderKeyValueCacheEntry] = None,
    ) -> Tuple[Tensor, Tensor]:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        if cache_entry is not None:
            # This is a self-attention call, so we need to prepend the cached key/value tensors.
            k, v = cache_entry.push_layer(k, v)

        wv, qk = qkv_attention(self.n_head, q, k, v, mask)
        return self.out(wv), qk

class EncoderResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = x + self.attn(self.attn_ln(x), mask=mask)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x



class DecoderResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()

        # cache_entry is only set when CrossAttention is enabled

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = MultiHeadCrossAttention(n_state, n_head)
        self.cross_attn_ln = LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Tensor,
        cache_entry: DecoderKeyValueBlockCacheEntry,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = x + self.attn(self.attn_ln(x), mask=mask, cache_entry = cache_entry.attn())[0]
        x = x + self.cross_attn(self.cross_attn_ln(x), xa, cache_entry = cache_entry.cross_attn())[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


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
            [EncoderResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
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
            x = block(x)

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
                DecoderResidualAttentionBlock(n_state, n_head)
                for i in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, cache: DecoderKeyValueCache) -> Tensor:
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        offset = cache.get_offset()
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset: offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)

        for i, block in enumerate(self.blocks):
            x = block(
                x, xa, cache.get_block(i), mask=self.mask)

        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits


class InferenceDecoder(nn.Module):
    initial_token_length: int
    cache: DecoderKeyValueCache

    def __init__(self, n_layers: int, decoder: TextDecoder):
        super().__init__()

        self.decoder = decoder
        self.initial_token_length = 0
        self.cache = DecoderKeyValueCache(n_layers)

    def forward(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        if tokens.shape[-1] > self.initial_token_length:
            # only need to use the last token except in the first forward pass
            tokens = tokens[:, -1:]

        return self.decoder(
            tokens, audio_features, self.cache)

        return x

    @torch.jit.export
    def initialize_decoding(self, initial_token_length: int):
        self.initial_token_length = initial_token_length
        self.cache.clear()

    @torch.jit.export
    def cleanup_caching(self):
        self.cache.clear()

    @torch.jit.export
    def rearrange_kv_cache(self, source_indices: Tensor):
        if not torch.equal(source_indices, torch.range(0, source_indices.size()[0] - 1)):
            self.cache.rearrange_kv_cache(source_indices)


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
        return self.decoder(tokens, audio_features, DecoderKeyValueCache(self.dims.n_text_layer))

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
        return torch.jit.script(InferenceDecoder(self.dims.n_text_layer, self.decoder))

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
