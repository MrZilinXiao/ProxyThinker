from collections import defaultdict
from functools import cached_property
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import torch
import copy
from vllm.config import (
    ContrastiveDecodingConfig,
    ModelConfig,
)
from vllm.distributed.communication_op import (
    broadcast_tensor_dict, get_tp_group
)
from vllm.distributed import get_pp_group, model_parallel_is_initialized
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.sampler import SamplerOutput, Sampler
from vllm.sequence import (
    ExecuteModelRequest,
    SequenceGroupMetadata,
)
from vllm.spec_decode.multi_step_worker import MultiStepWorker
from vllm.spec_decode.smaller_tp_proposer_worker import SmallerTpProposerWorker
from vllm.contrast_decode.contrast_model_runner import ContrastModelRunner

from vllm.worker.worker import Worker
from vllm.worker.worker_base import LoRANotSupportedWorkerBase, WorkerBase

logger = init_logger(__name__)


def create_contrastive_worker(*args, **kwargs) -> "ContrastiveDecodeWorker":
    # assert "contrastive_decoding_config" in kwargs
    # contrastive_decoding_config: ContrastiveDecodingConfig = kwargs.get(
    #     "contrastive_decoding_config"
    # )
    # assert contrastive_decoding_config is not None
    
    vllm_config = kwargs.get("vllm_config")
    contrastive_decoding_config: ContrastiveDecodingConfig = vllm_config.contrastive_decoding_config
    assert contrastive_decoding_config is not None
    
    contrastive_worker_kwargs = kwargs.copy()

    kwargs["model_runner_cls"] = ContrastModelRunner
    # print("create Worker with kwargs: ", args, kwargs)
    base_worker = Worker(*args, **kwargs)

    contrastive_worker_kwargs.update(
        vllm_config=vllm_config,
    )

    contrastive_decode_worker = ContrastiveDecodeWorker.create_worker(
        base_worker=base_worker,
        worker_kwargs=contrastive_worker_kwargs,
        positive_model_config=contrastive_decoding_config.positive_model_config,
        negative_model_config=contrastive_decoding_config.negative_model_config,
        sampler_alpha=contrastive_decoding_config.sampler_alpha,
        put_on_diff_gpus=contrastive_decoding_config.put_on_diff_gpus,
    )

    return contrastive_decode_worker


class ContrastiveDecodeWorker(LoRANotSupportedWorkerBase):

    @classmethod
    def create_worker(
        cls,
        base_worker: WorkerBase,
        worker_kwargs: Dict[str, Any],
        positive_model_config: Optional[ModelConfig],
        negative_model_config: Optional[ModelConfig],
        sampler_alpha: float,
        put_on_diff_gpus: bool
    ) -> "ContrastiveDecodeWorker":
        """
        Create a ContrastiveDecodeWorker from the given arguments.
        """
        assert (
            positive_model_config is not None or negative_model_config is not None
        ), "Either positive_model_config or negative_model_config must be specified."

        if positive_model_config is None:
            positive_worker = None
        else:
            positive_worker_kwargs = worker_kwargs.copy()
            
            # *** FIX START ***
            # Deep copy the config to avoid modifying the original or shared config
            # You might need deepcopy if config objects are mutable and shared
            positive_vllm_config = copy.deepcopy(positive_worker_kwargs['vllm_config'])
            target_tp_size = positive_vllm_config.parallel_config.tensor_parallel_size
            
            # Set the specific model config for the positive worker
            # whihc will leaves pos_neg_workers to have the same parallel_config as the base worker
            positive_vllm_config.model_config = positive_model_config
            
            # IMPORTANT: Ensure this config doesn't recursively trigger contrastive creation
            # Set the contrastive config part to None for this specific worker instance
            positive_vllm_config.parallel_config = positive_vllm_config.contrastive_decoding_config.parallel_config  # this could takes a smaller TP
            positive_vllm_config.parallel_config.worker_cls = \
                        "vllm.worker.multi_step_worker.MultiStepWorker"
                        # "vllm.worker.multi_step_worker.MultiStepWorker"

            
            # Update the kwargs with the modified, isolated config
            positive_worker_kwargs['vllm_config'] = positive_vllm_config
            # *** FIX END ***
            positive_worker = MultiStepWorker(**positive_worker_kwargs)
            
            # print(f"draft tp: {positive_vllm_config.parallel_config.tensor_parallel_size}, "
            #       f"target tp: {target_tp_size}")
            
            positive_worker = SmallerTpProposerWorker.maybe_wrap_worker(
                positive_worker, 
                draft_tensor_parallel_size=positive_vllm_config.parallel_config.tensor_parallel_size,
                target_tensor_parallel_size=target_tp_size,
                is_positive=True,
            )
            
        if negative_model_config is None:
            negative_worker = None
        else:
            negative_worker_kwargs = worker_kwargs.copy()
            # *** FIX START ***
            negative_vllm_config = copy.deepcopy(negative_worker_kwargs['vllm_config'])
            target_tp_size = negative_vllm_config.parallel_config.tensor_parallel_size
            
            negative_vllm_config.model_config = negative_model_config
            negative_vllm_config.parallel_config = negative_vllm_config.contrastive_decoding_config.parallel_config
            
            negative_vllm_config.parallel_config.worker_cls = \
                        "vllm.worker.multi_step_worker.MultiStepWorker"
            negative_worker_kwargs['vllm_config'] = negative_vllm_config
             # *** FIX END ***
            negative_worker = MultiStepWorker(**negative_worker_kwargs)
            negative_worker = SmallerTpProposerWorker.maybe_wrap_worker(
                negative_worker,
                draft_tensor_parallel_size=negative_vllm_config.parallel_config.tensor_parallel_size,
                target_tensor_parallel_size=target_tp_size,
                # is_positive=False,
                is_positive=not put_on_diff_gpus,   # if put_on_diff_gpus, then put negative worker on 4,5,6,7
            )

            
            
        return cls(
            base_worker=base_worker,
            worker_kwargs=worker_kwargs,
            positive_worker=positive_worker,
            negative_worker=negative_worker,
            sampler_alpha=sampler_alpha,
            put_on_diff_gpus=put_on_diff_gpus,
        )

    def __init__(
        self,
        base_worker: WorkerBase,
        worker_kwargs: Dict[str, Any],
        positive_worker: Optional[WorkerBase],
        negative_worker: Optional[WorkerBase],
        sampler_alpha: float,
        put_on_diff_gpus: bool,
    ):
        self.base_worker = base_worker
        self.worker_kwargs = worker_kwargs
        self.positive_worker = positive_worker
        self.negative_worker = negative_worker
        self.sampler_alpha = sampler_alpha
        self.put_on_diff_gpus = put_on_diff_gpus
        

    def init_device(self) -> None:
        self.base_worker.init_device()
        if self.positive_worker is not None:
            print(f"Initializing positive device in contrast_worker ({type(self.positive_worker)})")
            self.positive_worker.init_device()
        
        if isinstance(self.positive_worker, SmallerTpProposerWorker):
            # need a global barrier
            get_tp_group().barrier()
            
        # maybe we should wait until all positive workers are initialized
        if self.negative_worker is not None:
            print(f"Initializing negative device in contrast_worker ({type(self.negative_worker)})")
            self.negative_worker.init_device()
            
        if isinstance(self.negative_worker, SmallerTpProposerWorker):
            # need a global barrier
            get_tp_group().barrier()

        # self.base_worker.load_model()
        # if self.positive_worker is not None:
        #     self.positive_worker.load_model()
        # if self.negative_worker is not None:
        #     self.negative_worker.load_model()

    def load_model(self, *args, **kwargs):
        self.base_worker.load_model()
        if self.positive_worker is not None:
            print(f"Loading positive model in contrast_worker ({type(self.positive_worker)})")
            self.positive_worker.load_model()
        if self.negative_worker is not None:
            self.negative_worker.load_model()
            
        if (isinstance(self.negative_worker, SmallerTpProposerWorker) 
                and get_tp_group().rank_in_group == self._negative_driver_rank
                and self.put_on_diff_gpus):
            logger.info(
                "SmallerTpProposerWorker: Setting is_driver_worker to "
                "True for rank %d", get_tp_group().rank_in_group
            )
            self.negative_worker.is_driver_worker = True
            self.negative_worker._worker.is_driver_worker = True
            self.negative_worker._worker.worker.is_driver_worker = True  # this should propoagate to the base worker
            # also do the same with negative_worker model runner
            self.negative_worker._worker.model_runner.is_driver_worker = True   # spec_decode worker
            self.negative_worker._worker.worker.model_runner.is_driver_worker = True  # real MultiStepWorkerModelRunner
            self.negative_worker._worker.worker.model_runner._base_model_runner.is_driver_worker = True # real GPUModelRunnerBase
            
            # check here self.negative_worker._tp_group() -> ranks, rank, local_rank, rank_in_group
            logger.info(f"SmallerTpProposerWorker negative tp_group_info (put_on_diff_gpus: {self.put_on_diff_gpus}): {self.negative_worker._tp_group}"
                        "ranks: %s, rank: %d, local_rank: %d, rank_in_group: %d, world_size: %d",
                        self.negative_worker._tp_group.ranks,
                        self.negative_worker._tp_group.rank,
                        self.negative_worker._tp_group.local_rank,
                        self.negative_worker._tp_group.rank_in_group, 
                        self.negative_worker._tp_group.world_size)   # 4! not 8!
            
        # for debug usage: print all negative_worker is_driver_worker when is available
        if isinstance(self.negative_worker, SmallerTpProposerWorker):
            first_is_driver_worker = self.negative_worker.is_driver_worker if hasattr(self.negative_worker, "is_driver_worker") else "None"
            second_is_driver_worker = self.negative_worker._worker.is_driver_worker if hasattr(self.negative_worker, "_worker") and hasattr(self.negative_worker._worker, "is_driver_worker") else "None"
            third_is_driver_worker = self.negative_worker._worker.worker.is_driver_worker if hasattr(self.negative_worker, "_worker") and hasattr(self.negative_worker._worker, "worker") and hasattr(self.negative_worker._worker.worker, "is_driver_worker") else "None"
            logger.info(f"SmallerTpProposerWorker (rank {get_tp_group().rank_in_group}) negative_worker is_driver_worker: {first_is_driver_worker}, "
                        f"negative_worker._worker.is_driver_worker: {second_is_driver_worker}, "
                        f"negative_worker._worker.worker.is_driver_worker: {third_is_driver_worker}")
            # print the same for model_runner
            logger.info(f"SmallerTpProposerWorker (rank {get_tp_group().rank_in_group}) negative_worker model_runner is_driver_worker: {self.negative_worker._worker.model_runner.is_driver_worker}, "
                        f"negative_worker._worker.worker.model_runner.is_driver_worker: {self.negative_worker._worker.worker.model_runner.is_driver_worker}")

    def get_cache_block_size_bytes(self) -> int:
        pass

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of cache blocks to use.

        This is done by profiling the base model (which is typically the
        larger of the two). Then the total memory which would be used by the
        base model KV is divided evenly between the positive and negative model KV,
        such that the number of blocks is equal in both KV caches.
        """
        num_gpu_blocks, num_cpu_blocks = self.base_worker.determine_num_available_blocks()
        return num_gpu_blocks, num_cpu_blocks
        # Fix: let's reserve the min gpu blocks for each model
        # return min(num_gpu_blocks, positive_num_gpu_blocks, negative_num_gpu_blocks), num_cpu_blocks
    
    
    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the cache engine of the scorer and proposer workers.
        """
        self.base_worker.initialize_cache(num_gpu_blocks=num_gpu_blocks,
                                            num_cpu_blocks=num_cpu_blocks)
        if self.positive_worker is not None:
            self.positive_worker.initialize_cache(num_gpu_blocks=num_gpu_blocks,
                                              num_cpu_blocks=num_cpu_blocks)
        if self.negative_worker is not None:
            self.negative_worker.initialize_cache(num_gpu_blocks=num_gpu_blocks,
                                              num_cpu_blocks=num_cpu_blocks)
    
    @torch.inference_mode()
    def execute_model(
        self, execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        """Perform contrastive decoding on the input batch."""
        # print(f"Running current rank {self.rank}, driver rank {self._driver_rank}: {str(execute_model_req)[:10]}")
        
        rank = get_tp_group().rank_in_group if model_parallel_is_initialized() else self.rank
        # if self.rank != self._driver_rank:
        if rank != self._driver_rank:
            self._run_non_driver_rank()
            return []
        
        if execute_model_req is None:
            """
            This signals that there's no more requests to process for now.
            All workers are running infinite loop with broadcast_tensor_dict,
            and it stops the loop when the driver broadcasts an empty input.
            Send an empty input to notify all other workers to stop their
            execution loop.
            """
            broadcast_tensor_dict({}, src=0)
            return []
        

        disable_all_contrastive_decoding = (
            self._should_disable_all_contrastive_decoding(execute_model_req)
        )
        boardcast_dict = dict(
            disable_all_contrastive_decoding=disable_all_contrastive_decoding,
        )
        # this triggers the `_run_non_driver_rank` to run 
        broadcast_tensor_dict(boardcast_dict, src=self._driver_rank)

        if disable_all_contrastive_decoding:
            return self._run_no_contrastive_decoding(execute_model_req)

        return self._run_contrastive_decoding(execute_model_req)

    def _should_disable_all_contrastive_decoding(
        self, execute_model_req: ExecuteModelRequest
    ) -> bool:
        """
        Determine if all contrastive decoding should be disabled.
        """
        # TODO: Implement this
        return False
    
    @torch.inference_mode()
    def start_worker_execution_loop(self) -> None:
        """Execute model loop to perform speculative decoding
        in parallel worker."""
        while self._run_non_driver_rank():
            pass

    def _run_non_driver_rank(self) -> bool:
        assert self.rank != self._driver_rank
        data = broadcast_tensor_dict(src=self._driver_rank)
        if not data:  # driver sent termination signal
            return False

        # disable_all_contrastive_decoding = data.get("disable_all_contrastive_decoding", False)

        # --------------------------------------------------
        # 1. Perform local forward passes
        # --------------------------------------------------
        self.base_worker.execute_model()
        if self.positive_worker is not None:
            self.positive_worker.execute_model()
            
        negative_sampler_output = None
        if self.negative_worker is not None:
            recv_dict = dict()
            
            if self.put_on_diff_gpus and self.rank == self._negative_driver_rank:
                # receive execute_model_req from driver rank
                # logger.info(f"Receiving execute_model_req from driver rank {self._driver_rank} to rank {self._negative_driver_rank}")
                recv_dict = get_tp_group().recv_tensor_dict(src=self._driver_rank)
                # logger.info(f"Received execute_model_req from driver rank {self._driver_rank} to rank {self._negative_driver_rank}: {recv_dict}")
                recv_dict.pop("sender_rank", None)  # remove sender_rank key
                
            negative_sampler_output = self.negative_worker.execute_model(**recv_dict)
            
        # logger.info(f"Non-driver rank {self.rank} get negative_sampler_output: {negative_sampler_output}")
        # Missing 4 here!
        
        # --------------------------------------------------
        # 2. If this rank is the designated negative driver, send its logits to rank 0
        # --------------------------------------------------
        if self.put_on_diff_gpus and self.rank == self._negative_driver_rank and negative_sampler_output is not None:
            neg_logits = negative_sampler_output[0].logits
            # Send negative logits to rank 0 instead of broadcasting directly
            send_dict = {"negative_logits": neg_logits, "sender_rank": self.rank}
            # Use point-to-point communication instead of broadcast
            # logger.info(f"Sending negative logits from rank {self.rank} to driver rank {self._driver_rank}")
            get_tp_group().send_tensor_dict(send_dict, dst=self._driver_rank)
            
            # logger.info(f"Sent negative logits from rank {self.rank} to driver rank {self._driver_rank}: {send_dict}")
        # --------------------------------------------------
        
        return True
    
    # usually we don't get here
    def _run_no_contrastive_decoding(
        self, execute_model_req: ExecuteModelRequest
    ) -> List[SamplerOutput]:
        """
        Run the model without contrastive decoding.
        """
        assert False, "This should not be called in contrastive decoding mode."
    
    # def _run_negative_driver_rank(self):
    #     assert self.rank == self._negative_driver_rank
    #     data = broadcast_tensor_dict(src=self._driver_rank)

    #     if not data:  # no more data to process -- signal to stop
    #         return False
        
    #     # run all models but receive the logits from the negative logits
    #     self.base_worker.execute_model()

    #     if self.positive_worker is not None:
    #         self.positive_worker.execute_model()

    #     if self.negative_worker is not None:
    #         negative_sampler_output = self.negative_worker.execute_model()
        
    #     negative_logits = negative_sampler_output[0].logits
    
    # only runs on driver worker (0)
    def _run_contrastive_decoding(
        self, execute_model_req: ExecuteModelRequest
    ) -> List[SamplerOutput]:
        """
        Run the model with contrastive decoding.
        """
        base_sampler_output = self.base_worker.execute_model(execute_model_req)
        positive_sampler_output: List[SamplerOutput] = []
        if self.positive_worker is not None:
            positive_sampler_output = self.positive_worker.execute_model(execute_model_req)
        else:
            positive_sampler_output = []
            
        if self.negative_worker is not None:
            # send execute_model_req if negative_worker is a SmallerTpProposerWorker
            if isinstance(self.negative_worker, SmallerTpProposerWorker) and self.put_on_diff_gpus:
                broadcast_dict = dict(
                    execute_model_req=execute_model_req,
                    sender_rank=self.rank,
                )
                # Use point-to-point communication instead of broadcast
                # logger.info(f"Sending execute_model_req from rank {self.rank} to driver rank {self._negative_driver_rank}")
                get_tp_group().send_tensor_dict(broadcast_dict, dst=self._negative_driver_rank)
                # logger.info(f"Sent execute_model_req from rank {self.rank} to driver rank {self._negative_driver_rank}: {broadcast_dict}")
            
            negative_sampler_output = self.negative_worker.execute_model(execute_model_req)
        else:
            negative_sampler_output = []
        # --------------------------------------------------
        # Receive the negative logits from the negative‑driver rank (if any)
        # --------------------------------------------------
        
        positive_logits = positive_sampler_output[0].logits if positive_sampler_output else None
        negative_logits: Optional[torch.Tensor] = None
        
        # logger.info(f"Driver rank {self._driver_rank} received positive logits: {positive_logits}")

        if self.negative_worker is not None:
            if self._negative_driver_rank in (self._driver_rank, -1):   # same device
                negative_logits = negative_sampler_output[0].logits
            else:
                # Receive negative logits directly from the negative driver rank
                # instead of broadcasting
                # logger.info(f"Receiving negative logits from driver rank {self._negative_driver_rank} to rank {self.rank}")
                recv_dict = get_tp_group().recv_tensor_dict(src=self._negative_driver_rank)
                # logger.info(f"Received negative logits from driver rank {self._negative_driver_rank} to rank {self.rank}: {recv_dict}")
                negative_logits = recv_dict.get("negative_logits")

        is_pp = model_parallel_is_initialized() and get_pp_group().world_size > 1
        
        if is_pp and not get_pp_group().is_last_rank:
            return []
        
        generators = self.base_worker.model_runner.get_generators(
            execute_model_req.finished_requests_ids)
        
        input_tokens_tensor, seq_lens, query_lens = self._prepare_input_tensors(
            execute_model_req.seq_group_metadata_list,
        )

        sampling_metadata = SamplingMetadata.prepare(
            execute_model_req.seq_group_metadata_list,
            seq_lens,
            query_lens,
            self.device,
            self.base_worker.model_runner.pin_memory,
            generators,
        )
        
        # logger.info(f"Barrier current rank {self.rank}, driver rank {self._driver_rank}: {positive_sampler_output} | {negative_sampler_output}")
        # I guess we have positive but not negative here

        contrastive_sampler_output = self._create_contrastive_sampler_output(
            sampling_metadata,
            base_sampler_output,
            positive_logits=positive_logits,
            negative_logits=negative_logits,
        )
        return contrastive_sampler_output

    def _create_contrastive_sampler_output(
        self,
        sampling_metadata: SamplingMetadata,
        base_sampler_output: List[SamplerOutput],
        positive_logits: Optional[torch.Tensor] = None,
        negative_logits: Optional[torch.Tensor] = None,
    ) -> List[SamplerOutput]:
        """
        Create a contrastive sampler output.
        """
        # by Claude: Safety check for empty or None outputs
        if not base_sampler_output or base_sampler_output[0] is None or base_sampler_output[0].logits is None:
            # Handle this case appropriately - either return early or provide default logits
            logger.warning("Missing logits in base_sampler_output. You should never see this.")
            # You might need to implement a fallback strategy here
            return []

        # Sample the next token.
        logits = base_sampler_output[0].logits
        # Align different logits shapes caused by tokenizer
        
        # hack: 72b and 7b model has a 152064 lm head while 2b model only has 151936
        
        larger_shape = max(logits.shape[-1], positive_logits.shape[-1])
        
        if logits.shape[-1] != larger_shape:  # positive logits are with larger vocab -- truncate positive_logits
            flag = True
            positive_logits = positive_logits[:,:logits.shape[-1]]
            negative_logits = negative_logits[0].logits[:,:logits.shape[-1]]
        else:  # logits are with larger vocab  -- truncate base_logits
            flag = False
            logits = logits[:,:positive_logits.shape[-1]]
        
        # base_logits = logits.clone()

        if self.positive_worker is not None:
            logits = logits + self.sampler_alpha * positive_logits
        if self.negative_worker is not None:
            logits = logits - self.sampler_alpha * negative_logits

        if not flag:
            # pad float('-inf') to the logits
            logits = torch.cat([logits, torch.full((logits.shape[0], 128), float('-inf'), device=self.device)], dim=-1)
        
        output: SamplerOutput = self.base_worker.model_runner.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )

        return [output]

    def _prepare_input_tensors(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
    ) -> Tuple[torch.Tensor, List[int], List[int]]:
        if not seq_group_metadata_list:
            return torch.empty(0, device=self.device), [], []

        input_tokens: List[int] = []
        seq_lens: List[int] = []
        query_lens: List[int] = []

        for seq_group_metadata in seq_group_metadata_list:
            is_prompt = seq_group_metadata.is_prompt

            for seq_data in seq_group_metadata.seq_data.values():
                seq_data_len = seq_data.get_len()
                if is_prompt:
                    context_len = seq_data.get_num_computed_tokens()
                    seq_len = min(
                        seq_data_len,
                        context_len + seq_group_metadata.token_chunk_size)
                    tokens = seq_data.get_token_ids()[context_len:seq_len]
                    seq_lens.append(seq_len)
                    input_tokens.extend(tokens)
                    query_lens.append(seq_len - context_len)
                else:
                    seq_lens.append(seq_data_len)
                    input_tokens.append(seq_data.get_last_token_id())
                    query_lens.append(1)

        input_tokens_tensor = torch.tensor(input_tokens,
                                           dtype=torch.long,
                                           device=self.device)
        return input_tokens_tensor, seq_lens, query_lens
    
    @cached_property
    def vocab_size(self) -> int:
        return self.base_worker.vocab_size

    @property
    def rank(self) -> int:
        return self.base_worker.rank

    @property
    def device(self) -> torch.device:
        return self.base_worker.device
    
    @property
    def _negative_driver_rank(self) -> int:
        if self.negative_worker is not None and isinstance(
                self.negative_worker, SmallerTpProposerWorker):
            return self.negative_worker._draft_ranks[0]
        else:
            return -1
        
    @property
    def _input_output_transfer_needed(self) -> bool:
        return self._negative_driver_rank != self._driver_rank

    @property
    def _driver_rank(self) -> int:
        return 0
    
    @property
    def _all_driver_ranks(self): 
        return [self._driver_rank, self._negative_driver_rank]