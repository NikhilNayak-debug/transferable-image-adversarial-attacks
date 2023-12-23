from abc import ABC, abstractmethod

from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self, hf_name: str, hf_processor: str) -> None:
        """
        Initialize the Model class with a name and processor for the Hugging Face model.

        Parameters:
        - hf_name (str): Name of the Hugging Face model.
        - hf_processor (str): Name of the Hugging Face processor.
        """
        pass

    @abstractmethod
    def forward(self):
        """
        Abstract method to define the forward pass of the model.
        """
        pass

    @abstractmethod
    def preprocess_text(self, text: str):
        """
        Abstract method to preprocess text data before feeding it into the model.

        Parameters:
        - text (str): Input text.

        Returns:
        - input_ids: Preprocessed input tensor for text.
        """
        pass

    @abstractmethod
    def preprocess_image(self, image):
        """
        Abstract method to preprocess image data before feeding it into the model.

        Parameters:
        - image: Input image.

        Returns:
        - pixel_values: Preprocessed input tensor for images.
        """
        pass

    @property
    @abstractmethod
    def loss(self):
        """
        Abstract property to define the loss function of the model.
        """
        pass

    def assert_inputs(self, fun):
        """
        A decorator function to ensure that text and image inputs have been preprocessed before calling a function.

        Parameters:
        - fun: The function to be decorated.

        Returns:
        - new_fun: Decorated function.
        """
        def new_fun(*args, **kwargs):
            assert self._input_ids is not None, "You need to preprocess text first"
            assert self._pixel_values is not None, "You need to preprocess image first"
            return fun(*args, **kwargs)
        return new_fun

