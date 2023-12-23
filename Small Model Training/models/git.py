from abc import ABC, abstractmethod
from typing import List
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModelForCausalLM, ViltProcessor, ViltForQuestionAnswering
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from .model import Model  # Importing the base class Model


class GitModel(Model):
    def __init__(self, hf_name="microsoft/git-base-textvqa", hf_processor="microsoft/git-base-textvqa") -> None:
        """
        Constructor to initialize the GitModel with Hugging Face model name and processor.

        Parameters:
        - hf_name (str, optional): Hugging Face model name. Defaults to "microsoft/git-base-textvqa".
        - hf_processor (str, optional): Hugging Face processor name. Defaults to "microsoft/git-base-textvqa".
        """
        # Initialize Hugging Face processor
        self._processor = AutoProcessor.from_pretrained(hf_processor)
        # Load and move model to device (GPU)
        self._model = AutoModelForCausalLM.from_pretrained(hf_name).to(device).cuda()

    def forward(self, input_ids=None, pixel_values=None):
        """
        Forward pass of the model.

        Parameters:
        - input_ids (torch.Tensor, optional): Input tensor for text. Defaults to None.
        - pixel_values (torch.Tensor, optional): Input tensor for images. Defaults to None.

        Returns:
        - output: Output tensor from the model.
        """
        if input_ids is None:
            assert self._input_ids is not None, "You need to preprocess text first"
            input_ids = self._input_ids
        if pixel_values is None:
            assert self._pixel_values is not None, "You need to preprocess image first"
            pixel_values = self._pixel_values

        # Preprocess target text to generate answer tokens
        answer_tokens = self._processor(text=self._target, return_tensors="pt", add_special_tokens=False).input_ids
        labels = input_ids[0].clone()
        masking_idx = answer_tokens.shape[1]
        labels[:-masking_idx] = -100
        labels = torch.tensor(labels).unsqueeze(0).to(device)

        self._labels = labels

        # Model forward pass with input, pixel values, and labels
        output = self._model(input_ids=input_ids, pixel_values=pixel_values, labels=self._labels)
        self._output = output
        return output

    def preprocess_text(self, text: str):
        """
        Preprocess text data before feeding it into the model.

        Parameters:
        - text (str): Input text.

        Returns:
        - input_ids: Preprocessed input tensor for text.
        """
        assert self._target is not None, "You need to set label first"
        text_total = text + self._target
        self._input_ids = self._processor(text=text_total, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

        # Create labels for the input_ids with appropriate masking
        question_tokens = self._processor.tokenizer(text, return_tensors="pt").input_ids
        labels = self._input_ids[0].clone()
        labels[:question_tokens.shape[1] - 1] = -100
        labels = torch.tensor(labels).unsqueeze(0).to(device)
        self._labels = labels
        return self._input_ids

    def preprocess_image(self, image):
        """
        Preprocess image data before feeding it into the model.

        Parameters:
        - image: Input image.

        Returns:
        - pixel_values: Preprocessed input tensor for images.
        """
        self._pixel_values = self._processor.image_processor(images=image, return_tensors="pt").pixel_values.to(device)
        return self._pixel_values

    def show_image(self, image_tensor):
        """
        Display the preprocessed image.

        Parameters:
        - image_tensor: Tensor representing the preprocessed image.
        """
        adv_img = (image_tensor).detach().cpu() \
                    * torch.tensor(self._processor.image_processor.image_std).view(1, 3, 1, 1) \
                    + torch.tensor(self._processor.image_processor.image_mean).view(1, 3, 1, 1)

        plt.imshow(adv_img[0].permute(1, 2, 0))

    def generate_answer(self, question, image):
        """
        Generate an answer for a given question and image.

        Parameters:
        - question (str): Input question.
        - image: Input image.

        Returns:
        - result: Generated answer.
        """
        if not torch.is_tensor(image):
            image = self._processor.image_processor(images=image, return_tensors="pt").pixel_values.to(device)
        input_tokens = self._processor(text=question, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        generated_ids = self._model.generate(pixel_values=image, input_ids=input_tokens, max_length=50)
        return self._processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    @property
    def target(self):
        """
        Property for accessing the target attribute.
        """
        pass

    @target.setter
    def target(self, target):
        """
        Setter for the target attribute.

        Parameters:
        - target: Target value.
        """
        self._target = target

    @target.getter
    def target(self):
        """
        Getter for the target attribute.

        Returns:
        - target: Current target value.
        """
        return self._target

    @property
    def loss(self):
        """
        Property for accessing the loss attribute.

        Returns:
        - loss: Loss value.
        """
        assert self._output is not None, "You need to call forward first"
        loss = self._output.loss
        return loss
