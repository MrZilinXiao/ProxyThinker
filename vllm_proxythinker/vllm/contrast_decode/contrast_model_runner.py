from vllm.worker.model_runner_base import (ModelRunnerBase,
                                           ModelRunnerWrapperBase)

class ContrastModelRunner(ModelRunnerWrapperBase):
    def __init__(self, model_runner: ModelRunnerBase):
        super().__init__(model_runner)