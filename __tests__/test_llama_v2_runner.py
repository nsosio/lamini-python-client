from lamini import LlamaV2Runner
from lamini.api.lamini import Lamini
from unittest.mock import patch
import unittest
import os


class TestLlamaV2Runner(unittest.TestCase):
    @patch("lamini.api.lamini.Lamini.generate")
    def test_llama_v2_runner(self, mock_generate):
        mock_generate.return_value = "My favorite food is pizza"
        model = LlamaV2Runner(api_key="test", model_name="meta-llama/Llama-2-7b-chat")
        answer = model.call("What is your favorite food?")
        self.assertEqual(answer, "My favorite food is pizza")

    @patch("lamini.api.inference_queue.InferenceQueue.submit")
    def test_llama_v2_runner(self, mock_submit):
        mock_submit.return_value = [{"output": "my output"}]
        runner = LlamaV2Runner(model_name="hf-internal-testing/tiny-random-gpt2")
        prompt = "A"
        answer = runner.call(prompt)
        self.assertEqual(answer, "my output")

    @patch("lamini.api.inference_queue.InferenceQueue.submit")
    def test_llama_v2_runner_w_output_type(self, mock_submit):
        mock_submit.return_value = [
            {
                "question_1": "my_answer_1",
                "question_2": "my_answer_2",
                "question_3": "my_answer_3",
            }
        ]
        runner = LlamaV2Runner(model_name="hf-internal-testing/tiny-random-gpt2")
        questions_type = {"question_1": "str", "question_2": "str", "question_3": "str"}
        prompt = "A"
        answer = runner.call(prompt, output_type=questions_type)
        self.assertEqual(
            answer,
            {
                "question_1": "my_answer_1",
                "question_2": "my_answer_2",
                "question_3": "my_answer_3",
            },
        )

    def test_llama_v2_runner_load_data(self):
        runner = LlamaV2Runner(model_name="hf-internal-testing/tiny-random-gpt2")
        dir_path = os.path.dirname(os.path.realpath(__file__))

        runner.load_data_from_jsonlines(
            dir_path + "/test_data.jsonlines", input_key="question", output_key="answer"
        )
        print(len(runner.data))
