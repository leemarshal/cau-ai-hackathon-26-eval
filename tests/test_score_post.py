from __future__ import annotations

import io
import math
import sys
import unittest
from email.message import Message
from http.client import HTTPException
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError, URLError


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ta_grading.score_post as score_post  # noqa: E402


class FakeResponse:
    def __init__(self, status: int, body: bytes = b"") -> None:
        self.status = status
        self.body = io.BytesIO(body)
        self.read_sizes: list[int] = []
        self.closed = False

    def getcode(self) -> int:
        return self.status

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return self.body.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        self.closed = True


class ScorePostTests(unittest.TestCase):
    def test_posts_exact_json_with_explicit_headers_and_timeout(self) -> None:
        response = FakeResponse(204, b"ok")
        with mock.patch.object(
            score_post._OPENER, "open", return_value=response
        ) as open_url:
            score_post.post_score(
                "https://api.minds.ai.kr/submit",
                8,
                0.7321,
                timeout_seconds=3.25,
            )

        request = open_url.call_args.args[0]
        self.assertEqual(open_url.call_args.kwargs, {"timeout": 3.25})
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.full_url, "https://api.minds.ai.kr/submit")
        self.assertEqual(request.data, b'{"team_id":8,"score":0.7321}')
        headers = {key.lower(): value for key, value in request.header_items()}
        self.assertEqual(
            headers,
            {
                "content-type": "application/json",
                "accept": "application/json",
                "user-agent": "cau-ai-hackathon-26-eval/1.0",
            },
        )
        self.assertEqual(response.read_sizes, [])
        self.assertTrue(response.closed)

    def test_user_agent_does_not_regress_to_python_urllib_default(self) -> None:
        response = FakeResponse(200)
        with mock.patch.object(
            score_post._OPENER, "open", return_value=response
        ) as open_url:
            score_post.post_score("https://example.test/submit", 1, 0.5)

        request = open_url.call_args.args[0]
        self.assertEqual(request.get_header("User-agent"), score_post.USER_AGENT)
        self.assertNotIn("Python-urllib", request.get_header("User-agent"))

    def test_integer_score_is_encoded_as_a_json_float(self) -> None:
        response = FakeResponse(200)
        with mock.patch.object(
            score_post._OPENER, "open", return_value=response
        ) as open_url:
            score_post.post_score("https://example.test/submit", 2, 1)

        request = open_url.call_args.args[0]
        self.assertEqual(request.data, b'{"team_id":2,"score":1.0}')

    def test_rejects_non_2xx_after_consuming_and_closing_response(self) -> None:
        response = FakeResponse(300, b"redirect")
        with mock.patch.object(score_post._OPENER, "open", return_value=response):
            with self.assertRaisesRegex(
                score_post.ScorePostError, "invalid HTTP status 300"
            ):
                score_post.post_score("https://example.test/submit", 1, 0.5)

        self.assertTrue(response.closed)
        self.assertEqual(
            response.read_sizes,
            [score_post.DEFAULT_MAX_RESPONSE_BYTES],
        )

    def test_handles_http_error_without_exposing_or_unbounding_its_body(self) -> None:
        error = HTTPError(
            "https://example.test/submit",
            503,
            "unavailable",
            hdrs=Message(),
            fp=io.BytesIO(b"private error details"),
        )
        with mock.patch.object(score_post._OPENER, "open", side_effect=error):
            with self.assertRaisesRegex(score_post.ScorePostError, "HTTP 503"):
                score_post.post_score("https://example.test/submit", 1, 0.5)

        self.assertTrue(error.closed)

    def test_accepts_2xx_without_reading_its_response_body(self) -> None:
        response = FakeResponse(200, b"x" * 20)
        with mock.patch.object(score_post._OPENER, "open", return_value=response):
            score_post.post_score(
                "https://example.test/submit",
                1,
                0.5,
                max_response_bytes=8,
            )

        self.assertEqual(response.read_sizes, [])
        self.assertEqual(response.body.tell(), 0)
        self.assertTrue(response.closed)

    def test_redirect_handler_refuses_to_follow_any_redirect(self) -> None:
        handler = score_post._RejectRedirects()
        self.assertIsNone(
            handler.redirect_request(
                mock.sentinel.request,
                mock.sentinel.file,
                307,
                "temporary redirect",
                mock.sentinel.headers,
                "https://other.example.test/submit",
            )
        )

    def test_wraps_transport_errors(self) -> None:
        with mock.patch.object(
            score_post._OPENER, "open", side_effect=URLError("offline")
        ):
            with self.assertRaisesRegex(score_post.ScorePostError, "offline"):
                score_post.post_score("https://example.test/submit", 1, 0.5)

    def test_wraps_http_protocol_errors(self) -> None:
        with mock.patch.object(
            score_post._OPENER,
            "open",
            side_effect=HTTPException("malformed response"),
        ):
            with self.assertRaisesRegex(
                score_post.ScorePostError, "malformed response"
            ):
                score_post.post_score("https://example.test/submit", 1, 0.5)

    def test_invalid_inputs_fail_before_network_access(self) -> None:
        cases = [
            ("http://example.test/submit", 1, 0.5, {}),
            ("https://example.test/submit#fragment", 1, 0.5, {}),
            ("https://user:pass@example.test/submit", 1, 0.5, {}),
            ("https://example.test/submit", True, 0.5, {}),
            ("https://example.test/submit", 1, math.nan, {}),
            (
                "https://example.test/submit",
                1,
                0.5,
                {"timeout_seconds": 0},
            ),
            (
                "https://example.test/submit",
                1,
                0.5,
                {"max_response_bytes": -1},
            ),
        ]
        with mock.patch.object(score_post._OPENER, "open") as open_url:
            for endpoint_url, team_id, score, kwargs in cases:
                with self.subTest(
                    endpoint_url=endpoint_url,
                    team_id=team_id,
                    score=score,
                    kwargs=kwargs,
                ):
                    with self.assertRaises((TypeError, ValueError)):
                        score_post.post_score(
                            endpoint_url, team_id, score, **kwargs
                        )

        open_url.assert_not_called()


if __name__ == "__main__":
    unittest.main()
