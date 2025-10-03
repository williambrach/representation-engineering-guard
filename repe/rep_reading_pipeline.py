import numpy as np
import torch
from transformers import Pipeline

from .rep_readers import (
    DIRECTION_FINDERS,
    ClusterMeanRepReader,
    PCARepReader,
    RandomRepReader,
    RepReader,
)


class RepReadingPipeline(Pipeline):
    """
    A Hugging Face pipeline for "Representation Reading." This pipeline extracts internal hidden states
    from a model and uses a RepReader to analyze how abstract concepts are represented.
    It can be used in two ways:
    1. To train a RepReader by finding concept directions (using `get_directions`).
    2. To get concept scores for new inputs using a trained RepReader.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _get_hidden_states(
        self,
        outputs: dict,
        rep_token: str | int = -1,
        hidden_layers: list[int] | int = -1,
        which_hidden_states: str | None = None,
    ) -> dict:
        """
        Extracts hidden states from the model's outputs for specified layers and a specific token position.
        """
        # For encoder-decoder models, select which set of hidden states to use ('encoder' or 'decoder').
        if hasattr(outputs, "encoder_hidden_states") and hasattr(
            outputs, "decoder_hidden_states"
        ):
            outputs["hidden_states"] = outputs[f"{which_hidden_states}_hidden_states"]

        hidden_states_layers = {}
        # Loop through the specified layers to extract their hidden states.
        if not isinstance(hidden_layers, list):
            hidden_layers = [hidden_layers]
        for layer in hidden_layers:
            hidden_states = outputs["hidden_states"][layer]
            # Select the hidden state for the specific representative token (e.g., the last token).
            hidden_states = hidden_states[:, rep_token, :].detach()
            # Ensure the data type is float32 for consistency, converting from bfloat16 if necessary.
            if hidden_states.dtype == torch.bfloat16:
                hidden_states = hidden_states.float()
            hidden_states_layers[layer] = hidden_states.detach()

        return hidden_states_layers

    def _sanitize_parameters(
        self,
        rep_reader: RepReader | None = None,
        rep_token: str | int = -1,
        hidden_layers: list[int] | int = -1,
        component_index: int = 0,
        which_hidden_states: str | None = None,
        **tokenizer_kwargs,
    ) -> tuple:
        """
        Standard pipeline method to process and validate pipeline arguments,
        organizing them for the preprocess, forward, and postprocess steps.
        """
        preprocess_params = tokenizer_kwargs
        forward_params = {}
        postprocess_params = {}

        forward_params["rep_token"] = rep_token
        # Ensure hidden_layers is a list for consistent processing.
        if not isinstance(hidden_layers, list):
            hidden_layers = [hidden_layers]

        # Verify that the RepReader has a direction for every specified layer.
        assert rep_reader is None or len(rep_reader.directions) == len(
            hidden_layers
        ), f"expect total rep_reader directions ({len(rep_reader.directions)})== total hidden_layers ({len(hidden_layers)})"
        forward_params["rep_reader"] = rep_reader
        forward_params["hidden_layers"] = hidden_layers
        forward_params["component_index"] = component_index
        forward_params["which_hidden_states"] = which_hidden_states

        return preprocess_params, forward_params, postprocess_params

    def preprocess(
        self, inputs: str | list[str] | list[list[str]], **tokenizer_kwargs
    ) -> dict:
        """
        Tokenizes the input text. Handles both text and image models.
        """
        if self.image_processor:
            return self.image_processor(
                inputs, add_end_of_utterance_token=False, return_tensors="pt"
            )
        return self.tokenizer(inputs, return_tensors=self.framework, **tokenizer_kwargs)

    def postprocess(self, outputs):  # noqa: ANN001, ANN201
        """
        Simply returns the final outputs without modification.
        """
        return outputs

    def _forward(
        self,
        model_inputs: dict,
        rep_token: int,
        hidden_layers: list[int] | int,
        rep_reader: RepReader | None = None,
        component_index: int = 0,
        which_hidden_states: str | None = None,
        pad_token_id: int | None = None,
    ) -> dict:
        """
        Performs the forward pass of the model to get hidden states and optionally transforms them.
        If a `rep_reader` is provided, it projects the hidden states onto the concept direction to get scores.
        Otherwise, it returns the raw hidden states.
        Args:
        - which_hidden_states (str): Specifies which part of the model (encoder, decoder, or both) to compute the hidden states from.
                        It's applicable only for encoder-decoder models. Valid values: 'encoder', 'decoder'.
        """
        # Ensure no gradients are calculated, as this is for inference.
        # get model hidden states and optionally transform them with a RepReader
        with torch.no_grad():
            # Special handling for encoder-decoder models to provide a decoder start token.
            if hasattr(self.model, "encoder") and hasattr(self.model, "decoder"):
                decoder_start_token = [self.tokenizer.pad_token] * model_inputs[
                    "input_ids"
                ].size(0)
                decoder_input = self.tokenizer(
                    decoder_start_token, return_tensors="pt"
                ).input_ids
                model_inputs["decoder_input_ids"] = decoder_input
            # Run the model and instruct it to output hidden states.
            outputs = self.model(**model_inputs, output_hidden_states=True)
        hidden_states = self._get_hidden_states(
            outputs, rep_token, hidden_layers, which_hidden_states
        )

        # If no RepReader is provided, return the raw hidden states.
        if rep_reader is None:
            return hidden_states

        # If a RepReader is provided, use it to transform hidden states into concept scores.
        return rep_reader.transform(hidden_states, hidden_layers, component_index)

    def _batched_string_to_hiddens(
        self,
        train_inputs,  # noqa: ANN001
        rep_token,  # noqa: ANN001
        hidden_layers,  # noqa: ANN001
        batch_size,  # noqa: ANN001
        which_hidden_states,  # noqa: ANN001
        **tokenizer_args,
    ) -> dict:
        """
        A helper function to efficiently get hidden states for a large list of strings by processing them in batches.
        """
        # This calls the pipeline's main `__call__` method, which will process inputs in batches.
        # Wrapper method to get a dictionary hidden states from a list of strings
        hidden_states_outputs = self(
            train_inputs,
            rep_token=rep_token,
            hidden_layers=hidden_layers,
            batch_size=batch_size,
            rep_reader=None,
            which_hidden_states=which_hidden_states,
            **tokenizer_args,
        )
        # Aggregate the results from all batches into a single dictionary.
        hidden_states = {layer: [] for layer in hidden_layers}
        for hidden_states_batch in hidden_states_outputs:
            for layer in hidden_states_batch:
                hidden_states[layer].extend(hidden_states_batch[layer])
        return {k: np.vstack(v) for k, v in hidden_states.items()}

    def _validate_params(self, n_difference: int, direction_method: str) -> None:
        """
        Validates parameters for the `get_directions` method.
        """
        # validate params for get_directions
        if direction_method == "clustermean":
            assert n_difference == 1, "n_difference must be 1 for clustermean"

    def get_directions(
        self,
        train_inputs: str | list[str] | list[list[str]],
        rep_token: str | int = -1,
        hidden_layers: list[int] | int = -1,
        n_difference: int = 1,
        batch_size: int = 8,
        train_labels: list[int] | None = None,
        direction_method: str = "pca",
        direction_finder_kwargs: dict | None = None,
        which_hidden_states: str | None = None,
        **tokenizer_args,
    ) -> PCARepReader | ClusterMeanRepReader | RandomRepReader:
        """
        Trains a RepReader to find concept directions from a training dataset. This implements the LAT methodology.
        Train a RepReader on the training data.
        Args:
            batch_size: batch size to use when getting hidden states
            direction_method: string specifying the RepReader strategy for finding directions
            direction_finder_kwargs: kwargs to pass to RepReader constructor
        """
        if direction_finder_kwargs is None:
            direction_finder_kwargs = {}

        if not isinstance(hidden_layers, list):
            assert isinstance(hidden_layers, int)
            hidden_layers = [hidden_layers]

        self._validate_params(n_difference, direction_method)

        # Initialize the chosen direction finding algorithm (e.g., PCARepReader).
        # initialize a DirectionFinder
        direction_finder = DIRECTION_FINDERS[direction_method](
            **direction_finder_kwargs
        )

        # if relevant, get the hidden state data for training set
        hidden_states = None
        relative_hidden_states = None
        # If the method requires hidden states (like PCA), collect them.
        if direction_finder.needs_hiddens:
            # get raw hidden states for the train inputs
            # Get raw hidden states for all training examples.
            hidden_states = self._batched_string_to_hiddens(
                train_inputs,
                rep_token,
                hidden_layers,
                batch_size,
                which_hidden_states,
                **tokenizer_args,
            )
            direction_finder.hidden_states = hidden_states

            # Calculate the difference between hidden states of paired examples (e.g., positive vs. negative).
            # get differences between pairs
            relative_hidden_states = {k: np.copy(v) for k, v in hidden_states.items()}
            for layer in hidden_layers:
                for _ in range(n_difference):
                    relative_hidden_states[layer] = (
                        relative_hidden_states[layer][::2]
                        - relative_hidden_states[layer][1::2]
                    )
            direction_finder.relative_hidden_states = relative_hidden_states

        # Use the chosen method to find the concept directions from the processed hidden states.
        # get the directions
        direction_finder.directions = direction_finder.get_rep_directions(
            self.model,
            self.tokenizer,
            relative_hidden_states,
            hidden_layers,
            train_choices=train_labels,
        )
        # Ensure directions are float32.
        for layer in direction_finder.directions:
            if isinstance(direction_finder.directions[layer], np.ndarray):
                # if type(direction_finder.directions[layer]) == np.ndarray:
                direction_finder.directions[layer] = direction_finder.directions[
                    layer
                ].astype(np.float32)

        # If labels are provided, determine the sign of the direction (e.g., if positive scores mean "more honest").
        if train_labels is not None:
            direction_finder.direction_signs = direction_finder.get_signs(
                hidden_states, train_labels, hidden_layers
            )

        return direction_finder

    def get_hidden_and_direction_vectors(
        self,
        train_inputs: str | list[str] | list[list[str]],
        rep_token: str | int = -1,
        hidden_layers: list[int] | None = None,
        n_difference: int = 1,
        batch_size: int = 8,
        train_labels: list[int] | None = None,
        direction_method: str = "pca",
        direction_finder_kwargs: dict | None = None,
        which_hidden_states: str | None = None,
        **tokenizer_args,
    ) -> tuple:
        if hidden_layers is None:
            hidden_layers = [-1]
        hidden_states = None
        direction_vectors = None
        # If the method requires hidden states (like PCA), collect them.
        hidden_states = self._batched_string_to_hiddens(
                train_inputs,
                rep_token,
                hidden_layers,
                batch_size,
                which_hidden_states,
                **tokenizer_args,
            )
        direction_vectors = {k: np.copy(v) for k, v in hidden_states.items()}
        for layer in hidden_layers:
            for _ in range(n_difference):
                direction_vectors[layer] = (
                    direction_vectors[layer][::2]
                    - direction_vectors[layer][1::2]
                )
        return hidden_states, direction_vectors

