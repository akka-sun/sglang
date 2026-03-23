from __future__ import annotations

import copy
import dataclasses
import multiprocessing as mp
import queue
import re
import time
from typing import Any, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    EvictParams,
    EvictResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey


@dataclasses.dataclass
class SelectiveNodeRef:
    cache_type: str  # "public" | "private"
    node: Any


@dataclasses.dataclass
class UnifiedNodeMeta:
    private_tag: int  # 0: public, 1: private
    creator_id: str
    access_epoch: int = 0
    tier: str = "rule"


class SelectivePrefixCache(BasePrefixCache):
    """Selective cache with unified privacy/public index semantics.

    This class emulates the Figure-4 style access model:
    - `private_tag` + `creator_id` access control.
    - Public/private selective sharing.
    - Metadata propagation for repeated prefixes.
    - Progressive eviction preference (public first).

    Implementation detail:
    - Storage still uses two internal `RadixCache` instances because the current KV
      allocator does not support one KV slot being referenced by two index owners.
      The metadata layer provides a unified logical view.
    """

    def __init__(
        self,
        params: CacheInitParams,
        public_extra_key: str = "__public__",
        sensitive_patterns: Optional[list[str]] = None,
        force_private_without_text: bool = True,
        enable_async_privacy_detector: bool = True,
        detector_queue_size: int = 8192,
    ):
        self.disable = params.disable
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size

        self.public_extra_key = public_extra_key
        self.force_private_without_text = force_private_without_text
        self._sensitive_regexes = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in (sensitive_patterns or self._default_sensitive_patterns())
            if pattern
        ]
        self._epoch = 0
        self._node_meta: dict[tuple[str, int], UnifiedNodeMeta] = {}
        self._prefix_label_cache: dict[tuple[str, tuple[int, ...]], str] = {}
        self._submitted_detection_keys: set[tuple[str, tuple[int, ...]]] = set()
        self.enable_async_privacy_detector = enable_async_privacy_detector
        self._detector_ctx = None
        self._detector_task_queue = None
        self._detector_result_queue = None
        self._detector_process = None

        if self.enable_async_privacy_detector:
            self._detector_ctx = mp.get_context("spawn")
            self._detector_task_queue = self._detector_ctx.Queue(maxsize=detector_queue_size)
            self._detector_result_queue = self._detector_ctx.Queue(maxsize=detector_queue_size)
            self._detector_process = self._detector_ctx.Process(
                target=self._privacy_detector_worker_stub,
                args=(self._detector_task_queue, self._detector_result_queue),
                daemon=True,
            )
            self._detector_process.start()

        if params.enable_metrics:
            self.init_metrics_collector()

        self.public_cache = RadixCache(params)
        self.private_cache = RadixCache(params)

    @staticmethod
    def _default_sensitive_patterns() -> list[str]:
        return [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b(?:\d[ -]?){13,19}\b",  # card-like numbers
            r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",  # email
            r"\b(?:\+?\d{1,3}[ -]?)?(?:\(?\d{3}\)?[ -]?)\d{3}[ -]?\d{4}\b",  # phone
            r"\b(?:password|passwd|secret|api[_-]?key|token)\b",
        ]

    def _get_private_key(self, req, base_key: Optional[str]) -> str:
        if base_key:
            return f"private:{base_key}"
        if req is not None and getattr(req, "session_id", None):
            return f"private:session:{req.session_id}"
        if req is not None and getattr(req, "rid", None):
            return f"private:rid:{req.rid}"
        return "private:anonymous"

    def _classify_request(self, req, key: RadixKey) -> tuple[str, str]:
        """Return (decision, tier): decision in {public, private, pending}.

        pending = unknown yet, treated as private for safety.
        """
        if req is None:
            return "public", "default"

        custom_params = getattr(getattr(req, "sampling_params", None), "custom_params", None)
        if isinstance(custom_params, dict):
            mode = custom_params.get("kv_privacy_mode")
            if mode == "private":
                return "private", "override"
            creator_id = self._get_private_key(req, key.extra_key)
            token_ids = list(key.token_ids)
            detector_label = custom_params.get("kv_privacy_label")
            if detector_label in ("private", "public", "pending"):
                self._propagate_label(
                    creator_id=creator_id,
                    token_ids=token_ids,
                    label=("private" if detector_label == "pending" else detector_label),
                )
            public_prefix_len = custom_params.get("kv_public_prefix_len")
            if isinstance(public_prefix_len, int) and public_prefix_len > 0:
                public_len = min(public_prefix_len, len(token_ids))
                self.mark_prefix_public(
                    creator_id=creator_id,
                    token_ids=token_ids[:public_len],
                )

        creator_id = self._get_private_key(req, key.extra_key)
        token_ids = tuple(key.token_ids)
        for length in range(len(token_ids), 0, -self.page_size):
            cached_label = self._prefix_label_cache.get((creator_id, token_ids[:length]))
            if cached_label is not None:
                return cached_label, "cached"

        self._enqueue_detection_task(req=req, creator_id=creator_id, token_ids=list(token_ids))
        return "private", "pending"

    def _cache_from_type(self, cache_type: str) -> RadixCache:
        return self.public_cache if cache_type == "public" else self.private_cache

    def _node_type(self, node_ref: Any) -> str:
        if isinstance(node_ref, SelectiveNodeRef):
            return node_ref.cache_type
        return "private"

    def _make_req_view(self, req, cache_type: str, node_ref: Any):
        req_view = copy.copy(req)
        if cache_type == "public":
            req_view.extra_key = self.public_extra_key
        else:
            req_view.extra_key = self._get_private_key(req, getattr(req, "extra_key", None))
        req_view.last_node = node_ref.node if isinstance(node_ref, SelectiveNodeRef) else node_ref
        return req_view

    def _propagate_label(self, creator_id: str, token_ids: list[int], label: str):
        if not token_ids:
            return
        step = max(1, self.page_size)
        for end in range(step, len(token_ids) + 1, step):
            self._prefix_label_cache[(creator_id, tuple(token_ids[:end]))] = label

    def mark_prefix_public(self, creator_id: str, token_ids: list[int]):
        """External hook for async detector: mark a prefix as public."""
        self._propagate_label(creator_id=creator_id, token_ids=token_ids, label="public")

    def mark_prefix_private(self, creator_id: str, token_ids: list[int]):
        """External hook for async detector: mark a prefix as private."""
        self._propagate_label(creator_id=creator_id, token_ids=token_ids, label="private")

    @staticmethod
    def _privacy_detector_worker_stub(task_queue, result_queue):
        """Concept scaffold for async privacy detector process.

        Current stub intentionally classifies everything as private.
        Replace this function with real detector logic later.
        """
        while True:
            task = task_queue.get()
            if task is None:
                break
            result_queue.put(
                {
                    "creator_id": task["creator_id"],
                    "token_ids": task["token_ids"],
                    "label": "private",
                    "tier": "async_stub",
                }
            )

    def _enqueue_detection_task(self, req, creator_id: str, token_ids: list[int]):
        if not self.enable_async_privacy_detector:
            return
        if not token_ids:
            return
        key = (creator_id, tuple(token_ids))
        if key in self._submitted_detection_keys:
            return
        self._submitted_detection_keys.add(key)
        if self._detector_task_queue is None:
            return
        task = {
            "creator_id": creator_id,
            "token_ids": token_ids,
            "prompt_text": getattr(req, "origin_input_text", None),
            "timestamp": time.time(),
        }
        try:
            self._detector_task_queue.put_nowait(task)
        except queue.Full:
            pass

    def _drain_detection_results(self):
        if not self.enable_async_privacy_detector or self._detector_result_queue is None:
            return
        while True:
            try:
                result = self._detector_result_queue.get_nowait()
            except queue.Empty:
                break
            label = result.get("label")
            creator_id = result.get("creator_id")
            token_ids = result.get("token_ids")
            if not creator_id or not token_ids:
                continue
            if label == "public":
                self.mark_prefix_public(creator_id=creator_id, token_ids=token_ids)
            else:
                self.mark_prefix_private(creator_id=creator_id, token_ids=token_ids)

    def shutdown_detector(self):
        if not self.enable_async_privacy_detector:
            return
        if self._detector_task_queue is not None:
            try:
                self._detector_task_queue.put_nowait(None)
            except Exception:
                pass
        if self._detector_process is not None and self._detector_process.is_alive():
            self._detector_process.join(timeout=0.5)
            if self._detector_process.is_alive():
                self._detector_process.terminate()
        self._detector_process = None
        self._detector_task_queue = None
        self._detector_result_queue = None

    def _update_node_meta(
        self,
        node_ref: Any,
        cache_type: str,
        creator_id: str,
        tier: str,
    ):
        node = node_ref.node if isinstance(node_ref, SelectiveNodeRef) else node_ref
        node_id = getattr(node, "id", None)
        if node_id is None:
            return
        self._epoch += 1
        self._node_meta[(cache_type, node_id)] = UnifiedNodeMeta(
            private_tag=(1 if cache_type == "private" else 0),
            creator_id=creator_id,
            access_epoch=self._epoch,
            tier=tier,
        )

    def reset(self):
        self._drain_detection_results()
        self.public_cache.reset()
        self.private_cache.reset()
        self._epoch = 0
        self._node_meta.clear()
        self._prefix_label_cache.clear()
        self._submitted_detection_keys.clear()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        if self.disable:
            return self.private_cache.match_prefix(params)
        self._drain_detection_results()

        req = params.req
        key = params.key
        decision, tier = self._classify_request(req, key)
        creator_id = self._get_private_key(req, key.extra_key)

        private_key = RadixKey(
            token_ids=key.token_ids,
            extra_key=creator_id,
            is_bigram=key.is_bigram,
        )
        private_result = self.private_cache.match_prefix(
            MatchPrefixParams(key=private_key, cow_mamba=params.cow_mamba, req=req)
        )

        if decision == "private":
            cache_type = "private"
            selected = private_result
        else:
            public_key = RadixKey(
                token_ids=key.token_ids,
                extra_key=self.public_extra_key,
                is_bigram=key.is_bigram,
            )
            public_result = self.public_cache.match_prefix(
                MatchPrefixParams(key=public_key, cow_mamba=params.cow_mamba, req=req)
            )
            # pending is handled in private by default; explicit public/cached public can use public path.
            if decision == "pending":
                cache_type = "private"
                selected = private_result
            elif len(public_result.device_indices) > len(private_result.device_indices):
                cache_type = "public"
                selected = public_result
            else:
                cache_type = "private"
                selected = private_result

        if req is not None:
            req._selective_cache_type = cache_type
            req._selective_privacy_decision = decision
            req._selective_creator_id = creator_id
            req._selective_privacy_tier = tier

        self._update_node_meta(
            node_ref=selected.last_device_node,
            cache_type=cache_type,
            creator_id=creator_id,
            tier=tier,
        )

        return MatchResult(
            device_indices=selected.device_indices,
            last_device_node=SelectiveNodeRef(cache_type=cache_type, node=selected.last_device_node),
            last_host_node=SelectiveNodeRef(cache_type=cache_type, node=selected.last_host_node),
            host_hit_length=selected.host_hit_length,
            mamba_branching_seqlen=selected.mamba_branching_seqlen,
        )

    def cache_finished_req(self, req, is_insert: bool = True, **kwargs):
        if self.disable:
            return self.private_cache.cache_finished_req(req, is_insert=is_insert, **kwargs)
        self._drain_detection_results()

        source_cache_type = self._node_type(req.last_node)
        source_cache = self._cache_from_type(source_cache_type)

        decision = getattr(req, "_selective_privacy_decision", "private")
        creator_id = getattr(
            req,
            "_selective_creator_id",
            self._get_private_key(req, getattr(req, "extra_key", None)),
        )
        privacy_tier = getattr(req, "_selective_privacy_tier", "default")
        desired_cache_type = "public" if decision == "public" else "private"
        can_switch = req.cache_protected_len == 0
        target_cache_type = (
            desired_cache_type if (is_insert and can_switch) else source_cache_type
        )
        target_cache = self._cache_from_type(target_cache_type)

        # Safe migration only when no protected prefix was reused.
        if target_cache_type != source_cache_type:
            source_node = req.last_node.node if isinstance(req.last_node, SelectiveNodeRef) else req.last_node
            source_cache.dec_lock_ref(source_node)
            req_view = self._make_req_view(
                req=req,
                cache_type=target_cache_type,
                node_ref=SelectiveNodeRef(
                    cache_type=target_cache_type, node=target_cache.root_node
                ),
            )
            req_view.cache_protected_len = 0
            result = target_cache.cache_finished_req(req_view, is_insert=is_insert, **kwargs)
        else:
            req_view = self._make_req_view(req=req, cache_type=source_cache_type, node_ref=req.last_node)
            result = source_cache.cache_finished_req(req_view, is_insert=is_insert, **kwargs)

        # Prefix label propagation approximates Figure-4 descendant metadata propagation.
        self._propagate_label(
            creator_id=creator_id,
            token_ids=getattr(req, "fill_ids", []) or [],
            label=("public" if target_cache_type == "public" else "private"),
        )
        self._update_node_meta(
            node_ref=req.last_node,
            cache_type=source_cache_type,
            creator_id=creator_id,
            tier=privacy_tier,
        )
        return result

    def cache_unfinished_req(self, req, **kwargs):
        if self.disable:
            return self.private_cache.cache_unfinished_req(req, **kwargs)
        self._drain_detection_results()

        cache_type = getattr(req, "_selective_cache_type", self._node_type(req.last_node))
        cache = self._cache_from_type(cache_type)
        req_view = self._make_req_view(req=req, cache_type=cache_type, node_ref=req.last_node)
        cache.cache_unfinished_req(req_view, **kwargs)

        req.prefix_indices = req_view.prefix_indices
        req.cache_protected_len = req_view.cache_protected_len
        req.last_node = SelectiveNodeRef(cache_type=cache_type, node=req_view.last_node)
        creator_id = getattr(
            req,
            "_selective_creator_id",
            self._get_private_key(req, getattr(req, "extra_key", None)),
        )
        self._update_node_meta(
            node_ref=req.last_node,
            cache_type=cache_type,
            creator_id=creator_id,
            tier=getattr(req, "_selective_privacy_tier", "default"),
        )

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()
        self._drain_detection_results()

        # Prefer evicting public entries first to preserve private-user locality.
        public_result = self.public_cache.evict(
            EvictParams(num_tokens=params.num_tokens, swa_num_tokens=0, mamba_num=0)
        )
        remaining = max(0, params.num_tokens - public_result.num_tokens_evicted)
        private_result = self.private_cache.evict(
            EvictParams(num_tokens=remaining, swa_num_tokens=0, mamba_num=0)
        )

        return EvictResult(
            num_tokens_evicted=(
                public_result.num_tokens_evicted + private_result.num_tokens_evicted
            )
        )

    def inc_lock_ref(self, node: Any):
        if self.disable:
            return 0
        if isinstance(node, SelectiveNodeRef):
            return self._cache_from_type(node.cache_type).inc_lock_ref(node.node)
        return self.private_cache.inc_lock_ref(node)

    def dec_lock_ref(self, node: Any, swa_uuid_for_lock: Optional[str] = None):
        if self.disable:
            return 0
        if isinstance(node, SelectiveNodeRef):
            return self._cache_from_type(node.cache_type).dec_lock_ref(node.node)
        return self.private_cache.dec_lock_ref(node)

    def evictable_size(self):
        return self.public_cache.evictable_size() + self.private_cache.evictable_size()

    def protected_size(self):
        return self.public_cache.protected_size() + self.private_cache.protected_size()

    def total_size(self):
        return self.public_cache.total_size() + self.private_cache.total_size()

    def pretty_print(self):
        print("=== SelectivePrefixCache: public ===")
        self.public_cache.pretty_print()
        print("=== SelectivePrefixCache: private ===")
        self.private_cache.pretty_print()

    def take_events(self):
        self._drain_detection_results()
        return self.public_cache.take_events() + self.private_cache.take_events()

    def __del__(self):
        try:
            self.shutdown_detector()
        except Exception:
            pass
