from wandering_light import solver
from wandering_light.solver import (
    TokenGeneratorPredictor,
    TrainedLLMTokenGenerator,
    TrajectorySolver,
)


class DummyPipeline:
    def __init__(self):
        self.prompts = []

        class DummyTokenizer:
            eos_token_id = 0
            pad_token_id = 0

        self.tokenizer = DummyTokenizer()

    def __call__(
        self,
        prompt,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        pad_token_id=None,
        batch_size=None,
    ):
        self.prompts.append(prompt)
        if isinstance(prompt, list):
            # Batch processing
            return [{"generated_text": f"{p} completion"} for p in prompt]
        else:
            # Single prompt
            return [{"generated_text": f"{prompt} completion"}]


def test_trained_llm_token_generator(monkeypatch):
    def fake_pipeline(task, model):
        assert task == "text-generation"
        assert model == "my-model"
        return DummyPipeline()

    monkeypatch.setattr(solver, "hf_pipeline", fake_pipeline)
    gen = TrainedLLMTokenGenerator("my-model")
    out = gen.generate("hello")
    assert out == "completion"
    assert gen.llm_io_history == [("hello", "completion")]


def test_get_solver_by_name_trained_local(monkeypatch):
    class DummyGen:
        def __init__(self, model_or_path):
            self.model_or_path = model_or_path
            self.llm_io_history = []

        def generate(self, prompt):
            return ""

        def generate_batch(self, prompts):
            return [self.generate(p) for p in prompts]

    monkeypatch.setattr(solver, "TrainedLLMTokenGenerator", DummyGen)
    sol = solver.get_solver_by_name("trained_local", model_name="checkpoint")
    assert isinstance(sol, TrajectorySolver)
    assert isinstance(sol.predictor, TokenGeneratorPredictor)
    assert isinstance(sol.predictor.token_generator, DummyGen)
    assert sol.predictor.token_generator.model_or_path == "checkpoint"
