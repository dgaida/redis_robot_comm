# Bug Report - redis_robot_comm

## Summary of Findings
A systematic analysis of the `redis_robot_comm` repository revealed several bugs, type inconsistencies, and architectural gaps.

| BUG-ID | Severity | Category | Component | Description |
|--------|----------|----------|-----------|-------------|
| BUG-001 | Critical | Functional | Subscription | Infinite loop in `subscribe_*` methods when callback fails. |
| BUG-002 | Medium | Code Quality | All Managers | 20 Mypy type errors related to Redis-py and union types. |
| BUG-003 | Medium | Edge Case | ImageStreamer | Missing validation for JPEG quality (1-100) and dimensions. |
| BUG-004 | Low | Code Quality | ImageStreamer | `ImageStreamConfig` from `config.py` is underutilized. |
| BUG-005 | Medium | Functional | All Managers | Inconsistent error handling; some Redis failures return `None` instead of raising exceptions. |
| BUG-006 | Low | Documentation | Repository | Doc coverage (86.7%) is below the mandated 95% threshold. |

## Detailed Analysis

### BUG-001: Infinite Loop in Subscriptions
**Files:** `redis_client.py`, `redis_image_streamer.py`, `redis_label_manager.py`, `redis_text_overlay.py`
**Root Cause:** The `last_id = msg_id` update happens inside the message processing loop, but if the callback or decoding fails, it might skip the update or be trapped in a loop depending on implementation. In `RedisMessageBroker.subscribe_objects`, it's inside the inner loop but if an exception occurs before `last_id = msg_id`, it retries.
**Impact:** High CPU usage and system hang if a "poison pill" message arrives.

### BUG-002: Mypy Type Errors
**Files:** Multiple
**Root Cause:** Incompatibility with `redis-py` type stubs and incorrect handling of `Awaitable` types in synchronous code.

### BUG-003: Missing Validation
**Files:** `redis_image_streamer.py`
**Root Cause:** `publish_image` accepts `quality` but doesn't check its range.

### BUG-005: Inconsistent Error Handling
**Files:** `redis_client.py`, `redis_image_streamer.py`
**Root Cause:** Some catch blocks return empty lists or `None` without informing the caller of a Redis-level failure.
