import unittest
import lamini


class TestLamini(unittest.TestCase):
    def setUp(self):
        self.engine = lamini.Lamini(
            api_key="test_k",
            model_name="hf-internal-testing/tiny-random-gpt2",
        )

    def test_make_llm_req_data_single_input(self):
        prompt = "What is the hottest day of the year?"
        out_type = {"Answer": "An answer to the question"}
        model_name = "hf-internal-testing/tiny-random-gpt2"
        stop_tokens = None
        max_tokens = None
        req_data = self.engine.make_llm_req_map(
            model_name,
            prompt,
            out_type,
            stop_tokens,
            max_tokens,
        )
        wanted_data = {
            "model_name": "hf-internal-testing/tiny-random-gpt2",
            "prompt": "What is the hottest day of the year?",
            "out_type": {"Answer": "An answer to the question"},
        }
        self.assertEqual(req_data, wanted_data)

    def test_passing_model_config(self):
        con = {"production.key": ""}
        engine = lamini.Lamini(
            api_key="test_k",
            model_name="hf-internal-testing/tiny-random-gpt2",
            config=con,
        )
        self.assertIsNone(engine.model_config)

        con = {
            "model_config": {"rope_scaling.type": "dynamic", "rope_scaling.factor": 3.0}
        }
        engine = lamini.Lamini(
            api_key="test_k",
            model_name="hf-internal-testing/tiny-random-gpt2",
            config=con,
        )
        self.assertEqual(engine.model_config["rope_scaling.type"], "dynamic")
        self.assertEqual(engine.model_config["rope_scaling.factor"], 3.0)

    # multiple input values and output types
    def test_make_llm_req_data_multiple_input(self):
        prompt = "What is the hottest day of the year?"
        out_type = {
            "Answer": "An answer to the question",
            "Answer2": "An answer to the question2",
        }
        model_name = "hf-internal-testing/tiny-random-gpt2"
        stop_tokens = None
        max_tokens = None
        req_data = self.engine.make_llm_req_map(
            model_name,
            prompt,
            out_type,
            stop_tokens,
            max_tokens,
        )
        wanted_data = {
            "model_name": "hf-internal-testing/tiny-random-gpt2",
            "prompt": "What is the hottest day of the year?",
            "out_type": {
                "Answer": "An answer to the question",
                "Answer2": "An answer to the question2",
            },
        }
        self.assertEqual(req_data, wanted_data)

    def test_make_llm_req_data_user_specified_out_type(self):
        # user can specify boolean with both "#bool" and "#boolean"

        prompt = "What is the answer?"
        out_type = {"Answer": "An answer to the question"}
        model_name = "hf-internal-testing/tiny-random-gpt2"
        stop_tokens = None
        max_tokens = None
        req_data = self.engine.make_llm_req_map(
            model_name,
            prompt,
            out_type,
            stop_tokens,
            max_tokens,
        )
        wanted_data = {
            "model_name": "hf-internal-testing/tiny-random-gpt2",
            "prompt": "What is the answer?",
            "out_type": {"Answer": "An answer to the question"},
        }
        self.assertEqual(req_data, wanted_data)

        out_type = {"Answer": "An answer to the question"}
        model_name = "hf-internal-testing/tiny-random-gpt2"
        stop_tokens = None
        max_tokens = None
        req_data = self.engine.make_llm_req_map(
            model_name,
            prompt,
            out_type,
            stop_tokens,
            max_tokens,
        )
        self.assertEqual(req_data, wanted_data)

        # user can specify integer with both "#int" and "#integer"

        out_type = {"Answer": "An answer to the question"}
        model_name = "hf-internal-testing/tiny-random-gpt2"
        stop_tokens = None
        max_tokens = None
        req_data = self.engine.make_llm_req_map(
            model_name,
            prompt,
            out_type,
            stop_tokens,
            max_tokens,
        )
        wanted_data["out_type"]["Answer"] = "An answer to the question"
        self.assertEqual(req_data, wanted_data)

        out_type = {"Answer": "An answer to the question"}
        model_name = "hf-internal-testing/tiny-random-gpt2"
        stop_tokens = None
        max_tokens = None
        req_data = self.engine.make_llm_req_map(
            model_name,
            prompt,
            out_type,
            stop_tokens,
            max_tokens,
        )
        self.assertEqual(req_data, wanted_data)

        # user can specify string with both "#str" and "#string"

        out_type = {"Answer": "An answer to the question"}
        model_name = "hf-internal-testing/tiny-random-gpt2"
        stop_tokens = None
        max_tokens = None
        req_data = self.engine.make_llm_req_map(
            model_name,
            prompt,
            out_type,
            stop_tokens,
            max_tokens,
        )
        wanted_data["out_type"]["Answer"] = "An answer to the question"
        self.assertEqual(req_data, wanted_data)

        out_type = {"Answer": "An answer to the question"}
        model_name = "hf-internal-testing/tiny-random-gpt2"
        stop_tokens = None
        max_tokens = None
        req_data = self.engine.make_llm_req_map(
            model_name,
            prompt,
            out_type,
            stop_tokens,
            max_tokens,
        )
        self.assertEqual(req_data, wanted_data)

        # user can specify number with both "#float" and "#number"

        out_type = {"Answer": "An answer to the question"}
        model_name = "hf-internal-testing/tiny-random-gpt2"
        stop_tokens = None
        max_tokens = None
        req_data = self.engine.make_llm_req_map(
            model_name,
            prompt,
            out_type,
            stop_tokens,
            max_tokens,
        )
        wanted_data["out_type"]["Answer"] = "An answer to the question"
        self.assertEqual(req_data, wanted_data)

        out_type = {"Answer": "An answer to the question"}
        model_name = "hf-internal-testing/tiny-random-gpt2"
        stop_tokens = None
        max_tokens = None
        req_data = self.engine.make_llm_req_map(
            model_name,
            prompt,
            out_type,
            stop_tokens,
            max_tokens,
        )
        self.assertEqual(req_data, wanted_data)
