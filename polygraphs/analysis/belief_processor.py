import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd  # Importing pandas library for data manipulation
import h5py  # Importing h5py library for working with HDF5 files


class BeliefProcessor:
    def __init__(self, device="cpu"):
        self.device = "cuda" if device == "gpu" else device

    def get_beliefs(self, hd5_file_path, graph):
        # Open the HDF5 file in read mode
        with h5py.File(hd5_file_path, "r") as fp:
            # Extract the keys (iteration numbers) from the 'beliefs' group in the HDF5 file
            _keys = sorted(map(int, fp["beliefs"].keys()))
            # Initialize a list to store iteration number and corresponding beliefs
            # with the initial beliefs from the .bin file graph
            initial_beliefs = graph.pg["ndata"]["beliefs"]
            if hasattr(initial_beliefs, "to"):
                initial_beliefs = initial_beliefs.to(device=self.device)
            iterations = [(0, initial_beliefs.tolist())]

            # Iterate over each key (iteration number) in the HDF5 file
            for key in _keys:
                # Retrieve beliefs data for the current iteration
                beliefs = fp["beliefs"][str(key)]

                # Append the iteration number and beliefs data to the list
                iterations.append((key, list(beliefs)))

        # Create a MultiIndex for DataFrame indexing with iteration number and node as indices
        index = pd.MultiIndex.from_product(
            [[0, *_keys], list(graph.nodes)], names=["iteration", "node"]
        )

        # Create an empty DataFrame with the defined MultiIndex
        iterations_df = pd.DataFrame(index=index, columns=["beliefs"], dtype="Float32")

        # Populate the DataFrame with beliefs data for each iteration
        for key, beliefs in iterations:
            iterations_df.loc[key, "beliefs"] = beliefs

        # Return the populated DataFrame containing beliefs data for each iteration
        return iterations_df


class Beliefs:
    """
    The Beliefs class stores the beliefs of simulations that have been
    explicitly loaded for analysis using the Belief Processor

    This class provides an iterator and get item to access beliefs
    """

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

        # Return a saved beliefs dataframe using its index or load from file
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
