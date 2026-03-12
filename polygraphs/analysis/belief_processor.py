import os
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import pandas as pd
import torch


class BeliefProcessor:
    def __init__(self, device="cpu"):
        requested = "cuda" if device == "gpu" else device
        if requested == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but no GPU available; falling back to CPU", flush=True)
            self.device = "cpu"
        else:
            self.device = requested
        print(f"BeliefProcessor device = {self.device}", flush=True)

    def get_beliefs(self, hd5_file_path, graph):
        node_ids = list(graph.nodes)

        with h5py.File(hd5_file_path, "r") as fp:
            keys = sorted(map(int, fp["beliefs"].keys()))

            initial_beliefs = graph.pg["ndata"]["beliefs"]
            if not isinstance(initial_beliefs, torch.Tensor):
                initial_beliefs = torch.as_tensor(
                    initial_beliefs, dtype=torch.float32
                )

            initial_beliefs = initial_beliefs.to(self.device)
            belief_tensors = [initial_beliefs]

            for key in keys:
                arr = np.asarray(fp["beliefs"][str(key)], dtype=np.float32)
                belief_tensors.append(torch.from_numpy(arr).to(self.device))

        all_beliefs = torch.stack(belief_tensors, dim=0)
        all_beliefs_cpu = all_beliefs.cpu().numpy()

        iteration_ids = [0] + keys
        index = pd.MultiIndex.from_product(
            [iteration_ids, node_ids],
            names=["iteration", "node"],
        )

        iterations_df = pd.DataFrame(
            {"beliefs": all_beliefs_cpu.reshape(-1)},
            index=index,
        )

        return iterations_df


class Beliefs:
    def __init__(
        self,
        dataframe,
        belief_processor,
        graphs,
        parallel=True,
        max_workers=None,
    ):
        self.hd5_file_path = dataframe["hd5_file_path"]
        self.belief_processor = belief_processor
        self.graphs = graphs
        self.beliefs = [None] * len(dataframe)
        self.parallel = parallel and len(self.beliefs) > 1
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self._futures = None
        self.index = 0

        if self.parallel:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
            self._futures = [
                self._executor.submit(self._load_beliefs, idx)
                for idx in range(len(self.beliefs))
            ]
        else:
            self._executor = None

    def __getitem__(self, index):
        self._check_index(index)
        return self.get(index)

    def __len__(self):
        return len(self.beliefs)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.beliefs):
            self.index = 0
            raise StopIteration
        value = self.get(self.index)
        self.index += 1
        return value

    def get(self, index):
        self._check_index(index)

        if self.beliefs[index] is not None:
            return self.beliefs[index]

        if self._futures is not None:
            self.beliefs[index] = self._futures[index].result()
            return self.beliefs[index]

        self.beliefs[index] = self._load_beliefs(index)
        return self.beliefs[index]

    def _check_index(self, index):
        if index < 0 or index >= len(self.beliefs):
            raise IndexError("Simulation index out of range")

    def _load_beliefs(self, index):
        return self.belief_processor.get_beliefs(
            self.hd5_file_path[index],
            self.graphs[index],
        )
