"""
"""
from __future__ import annotations

import os
import time
import warnings
from datetime import timedelta
from typing import Any

import gradio as gr
import httpx
from packaging import version
from typing_extensions import assert_never

from .. import utils
from ..config import Config
from .api import APIClient
from .api import AuthLevel
from .api import QuotaInfos
from .api import ScheduleResponse
from .gradio import info
from .gradio import error
from .gradio import get_event
from .gradio import supports_auth


TOKEN_HEADER = 'X-IP-Token'
DEFAULT_SCHEDULE_DURATION = 60

QUOTA_MESSAGE = "You have exceeded your GPU quota"
UNUSED_MESSAGE = "GPU device not used"
NO_GPU_MESSAGE_REGULAR = "No GPU was available"
NO_GPU_MESSAGE_INQUEUE = "No GPU was available after 60s"

SIGNUP_ON_HF_TXT = "Create a free account"
SIGNUP_ON_HF_URL = "https://huggingface.co/join"
SUBSCRIBE_TO_PRO_TXT = "Subscribe to Pro"
SUBSCRIBE_TO_PRO_URL = "https://huggingface.co/settings/billing/subscription"


def api_client():
    assert Config.zero_device_api_url is not None
    httpx_client = httpx.Client(base_url=Config.zero_device_api_url, timeout=60, verify=False)
    return APIClient(httpx_client)


def startup_report():
    retries, max_retries = 0, 2
    client = api_client()
    while (status := client.startup_report()) is httpx.codes.NOT_FOUND: # pragma: no cover
        time.sleep(1)
        if (retries := retries + 1) > max_retries:
            raise RuntimeError("Error while initializing ZeroGPU: NotFound")
    if status is not httpx.codes.OK: # pragma: no cover
        raise RuntimeError("Error while initializing ZeroGPU: Unknown")


def html_string(html_contents: str, text_contents: str): # pragma: no cover
    class HTMLString(str):
        def __str__(self):
            return text_contents
    return HTMLString(html_contents)


def _toast_action(
    auth: AuthLevel | None,
    supports_html: bool,
    pro_message: str,
    unlogged_desc: str,
    logged_desc: str,
    ending: str,
) -> tuple[str, str]: # pragma: no cover
    if not supports_auth() or auth == 'pro':
        return pro_message, pro_message
    html = ""
    link = SIGNUP_ON_HF_URL if auth is None else SUBSCRIBE_TO_PRO_URL
    text = SIGNUP_ON_HF_TXT if auth is None else SUBSCRIBE_TO_PRO_TXT
    desc = unlogged_desc if auth is None else logged_desc
    desc += f" {ending}."
    style = ";".join([
        "white-space: nowrap",
        "text-underline-offset: 2px",
        "color: var(--body-text-color)",
    ])
    if supports_html:
        html += f'<a style="{style}" href="{link}">'
    html += text
    if supports_html:
        html += '</a>'
    html += f" {desc}"
    markdown = f'[{text}]({link}) {desc}'
    return html, markdown


def schedule(
    task_id: int,
    request: gr.Request | None = None,
    duration: timedelta | None = None,
    _first_attempt: bool = True,
) -> ScheduleResponse:

    if not (gradio_version := version.parse(gr.__version__)).major >= 4: # pragma: no cover
        raise RuntimeError("ZeroGPU is only compatible with Gradio 4+")

    GRADIO_HTML_TOASTS = gradio_version >= version.Version('4.39')

    if (token_and_payload := _get_token_and_payload(request)):
        if (token_error := token_and_payload[1].get('error')):
            message = f"Falling back to IP-based quotas ({token_error})"
            info("ZeroGPU client warning", message, level='warning')

    res, meta = api_client().schedule(
        cgroup_path=utils.self_cgroup_device_path(),
        task_id=task_id,
        token=token_and_payload[0],
        duration_seconds=duration.seconds if duration is not None else None,
    )

    auth = meta.auth

    if isinstance(res, ScheduleResponse):
        return res

    if isinstance(res, QuotaInfos): # pragma: no cover
        requested = duration.seconds if duration is not None else DEFAULT_SCHEDULE_DURATION
        if res.wait < timedelta(0):
            message = (
                f"The requested GPU duration ({requested}s) "
                f"is larger than the maximum allowed"
            )
            raise error("ZeroGPU illegal duration", message)
        else:
            gpu = "Pro GPU" if auth == 'pro' else ("free GPU" if auth == 'regular' else "GPU")
            message = (
                f"You have exceeded your {gpu} quota "
                f"({requested}s requested vs. {res.left}s left)."
            )
            details_html, details_markdown = _toast_action(
                auth=auth,
                supports_html=GRADIO_HTML_TOASTS,
                pro_message=f"Try again in {res.wait}",
                unlogged_desc="to get more",
                logged_desc="to get 5x more",
                ending="daily usage quota",
            )
            message_html = f"{message} {details_html}"
            message_text = f"{message} {details_markdown}"
            message = html_string(message_html, message_text)
            raise error("ZeroGPU quota exceeded", message, html=True)

    if not isinstance(res, httpx.codes): # pragma: no cover
        if meta.queuing_reason in ('node', None):
            info("ZeroGPU queue", "Waiting for a GPU to become available")
        elif meta.queuing_reason == 'concurrency':
            info("ZeroGPU queue", "Waiting for a GPU slot on this Space")
        else:
            assert_never(meta.queuing_reason)
        # TODO: Sign-up message if not authenticated (after some time ?)
        connection_event = get_event()
        if connection_event is None and request is not None:
            warnings.warn("ZeroGPU: Cannot get Gradio app Queue instance")
        while True:
            try:
                event = next(res)
            except StopIteration:
                raise RuntimeError("Unexpected end of stream")
            except httpx.RemoteProtocolError:
                if not _first_attempt:
                    raise RuntimeError("Error while re-trying after queue disconnect")
                return schedule(task_id, request, duration, _first_attempt=False)
            if event.event == 'ping':
                if connection_event is not None and not connection_event.alive:
                    res.close()
                    raise RuntimeError("Connection closed by visitor while queueing")
                continue
            if event.event == 'failed':
                details_html, details_markdown = _toast_action(
                    auth=auth,
                    supports_html=GRADIO_HTML_TOASTS,
                    pro_message="Retry later",
                    unlogged_desc="to get a higher",
                    logged_desc="to get the highest",
                    ending="priority in ZeroGPU queues",
                )
                message_html = f"{NO_GPU_MESSAGE_INQUEUE}. {details_html}"
                message_text = f"{NO_GPU_MESSAGE_INQUEUE} {details_markdown}"
                message = html_string(message_html, message_text)
                raise error("ZeroGPU queue timeout", message, html=True)
            if event.event == 'succeeded':
                assert event.data is not None
                if connection_event is not None and not connection_event.alive:
                    release(event.data.allowToken)
                    raise RuntimeError("Connection closed by visitor on queue success")
                info("ZeroGPU queue", "Successfully acquired a GPU", level='success')
                return event.data

    if res is httpx.codes.SERVICE_UNAVAILABLE:
        raise error("ZeroGPU client error", NO_GPU_MESSAGE_REGULAR)

    if res is httpx.codes.UNAUTHORIZED: # pragma: no cover
        raise error("ZeroGPU client error", "Expired ZeroGPU proxy token")

    # TODO: Find a way to log 'detail' response field
    raise RuntimeError(f"ZeroGPU API /schedule error: {res} ({httpx.codes.get_reason_phrase(res)})") # pragma: no cover


def allow(allow_token: str) -> None:
    pid = os.getpid()
    assert pid != 1, "Allowing PID 1 on ZeroGPU will end up killing your Space"
    assert api_client().allow(allow_token=allow_token, pid=pid) is httpx.codes.OK


def release(
    allow_token: str, *,
    fail: bool = False,
    allow_404: bool = False,
) -> None:

    res = api_client().release(
        allow_token=allow_token,
        fail=fail,
    )

    if res is httpx.codes.NO_CONTENT: # pragma: no cover
        try:
            info("ZeroGPU client warning", UNUSED_MESSAGE, level='warning')
        except AttributeError:
            pass
        warnings.warn(UNUSED_MESSAGE, RuntimeWarning)
        return None

    if res is httpx.codes.NOT_FOUND:
        if not allow_404:
            warnings.warn("ZeroGPU API /release warning: 404 Not Found")
        return None

    if httpx.codes.is_success(res):
        return None

    # TODO: Find a way to log 'detail' response field
    # TODO: Only raise in dev environment. Simply warn in production ?
    raise RuntimeError(f"ZeroGPU API /release error: {res} ({httpx.codes.get_reason_phrase(res)})") # pragma: no cover


def _get_token(request: gr.Request | None) -> str | None:

    if request is None:
        return None

    headers = getattr(request, 'headers', None)
    if headers is None or not hasattr(headers, '__dict__'):
        raise error("ZeroGPU client error", "Internal Gradio error")

    # Compatibility trick
    if not hasattr(headers, 'get'):
        headers = headers.__dict__ # pragma: no cover

    return headers.get(TOKEN_HEADER.lower())


def _get_token_and_payload(request: gr.Request | None) -> tuple[str | None, dict[str, Any]]:
    if (token := _get_token(request)) is None:
        return None, {}
    try:
        payload = utils.jwt_payload(token)
    except Exception: # pragma: no cover
        warnings.warn("Error while decoding X-IP-Token JWT")
        return token, {}
    return token, payload
