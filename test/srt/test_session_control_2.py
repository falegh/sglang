"""
Usage:
python3 -m unittest test_session_control.TestSessionControl.test_session_control
python3 -m unittest test_session_control.TestSessionControl.test_session_control_with_branching
python3 -m unittest test_session_control.TestSessionControl.test_session_control_backtrack_with_abort
python3 -m unittest test_session_control.TestSessionControlVision.test_session_control
"""

import asyncio
import json
import unittest

import aiohttp
import requests

from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


def remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix) :] if text.startswith(prefix) else text


class TestSessionControl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model, cls.base_url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_AA_session(self, gen_len=12):
        chunks = [
            "Let me tell you something about France.",
            "The capital of France is",
            "The population of the city is",
            "A brief history about that city is",
        ]
        tokenizer = get_tokenizer(self.model)
        chunks_ids = [tokenizer.encode(x) for x in chunks]
        for i in range(1, len(chunks_ids)):
            if chunks_ids[i][0] == tokenizer.bos_token_id:
                chunks_ids[i] = chunks_ids[i][1:]

        # Run two identical sessions
        outputs_session_1 = self.run_session_control(chunks_ids, gen_len)
        outputs_session_2 = self.run_session_control(chunks_ids, gen_len)

        print("outputs from session control 1:\n", outputs_session_1)
        print("outputs from session control 2:\n", outputs_session_2)

        assert outputs_session_1 == outputs_session_2

    def test_AA_no_session(self, gen_len=12):
        chunks = [
            "Let me tell you something about France.",
            "The capital of France is",
            "The population of the city is",
            "A brief history about that city is",
        ]
        tokenizer = get_tokenizer(self.model)
        chunks_ids = [tokenizer.encode(x) for x in chunks]
        for i in range(1, len(chunks_ids)):
            if chunks_ids[i][0] == tokenizer.bos_token_id:
                chunks_ids[i] = chunks_ids[i][1:]

        outputs_normal_1 = self.run_no_session_control(chunks_ids, tokenizer, gen_len)
        outputs_normal_2 = self.run_no_session_control(chunks_ids, tokenizer, gen_len)

        print("outputs from normal 1:\n", outputs_normal_1)
        print("outputs from normal 2:\n", outputs_normal_2)

        assert outputs_normal_1 == outputs_normal_2

    def test_session_control(self, gen_len=12):
        chunks = [
            "Let me tell you something about France.",
            "The capital of France is",
            "The population of the city is",
            "A brief history about that city is",
        ]
        tokenizer = get_tokenizer(self.model)
        chunks_ids = [tokenizer.encode(x) for x in chunks]
        for i in range(1, len(chunks_ids)):
            if chunks_ids[i][0] == tokenizer.bos_token_id:
                chunks_ids[i] = chunks_ids[i][1:]

        outputs_session = self.run_session_control(chunks_ids, gen_len)
        outputs_normal = self.run_no_session_control(chunks_ids, tokenizer, gen_len)

        print("outputs from session control:\n", outputs_session)
        print("outputs from normal:\n", outputs_normal)

        assert outputs_session == outputs_normal

    def run_session_control(self, chunks_ids, gen_len=12):
        # Open new session
        # requests.post(self.base_url + "/flush_cache")
        session_id = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 1000},
        ).json()
        rid = None
        first_rid = None
        outputs = []

        # Process chunks
        for i, chunk_ids in enumerate(chunks_ids):
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": chunk_ids,
                    "session_params": {
                        "id": session_id,
                        "rid": rid,
                        "offset": -1,
                        "replace": True,
                    },
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": (
                            gen_len if i > 0 else 1
                        ),  # prefill only for the first chunk
                        "no_stop_trim": True,
                        "skip_special_tokens": False,
                    },
                },
            ).json()
            rid = response["meta_info"]["id"]
            if i == 0:
                first_rid = rid
            if i > 0:
                outputs.append(response["text"])

        # Backtrack to first request and regenerate
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": chunks_ids[-1],
                "session_params": {
                    "id": session_id,
                    "rid": first_rid,
                    "offset": -1,
                    "replace": True,
                },
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": gen_len,
                    "no_stop_trim": True,
                    "skip_special_tokens": False,
                },
            },
        ).json()
    
        requests.post(
            self.base_url + "/close_session",
            json={"session_id": session_id},
        )

        outputs.append(response["text"])
        return outputs

    def run_no_session_control(self, chunks_ids, tokenizer, gen_len=12):
        requests.post(self.base_url + "/flush_cache")
        input_ids_first_req = None
        input_ids = []
        outputs = []
        for i, chunk_ids in enumerate(chunks_ids):
            input_ids += chunk_ids
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": input_ids,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": (
                            gen_len if i > 0 else 1
                        ),  # prefill only for the first chunk
                        "no_stop_trim": True,
                        "skip_special_tokens": False,
                    },
                },
            ).json()
            if i > 0:
                output_ids = tokenizer.encode(response["text"])
                if output_ids[0] == tokenizer.bos_token_id:
                    output_ids = output_ids[1:]
                input_ids += output_ids[:-1]
                outputs.append(response["text"])
            if i == 0:
                input_ids_first_req = input_ids.copy()

        input_ids_first_req += chunks_ids[-1]
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": input_ids_first_req,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": gen_len,
                    "no_stop_trim": True,
                    "skip_special_tokens": False,
                },
            },
        ).json()
        outputs.append(response["text"])
        return outputs


if __name__ == "__main__":
    unittest.main()