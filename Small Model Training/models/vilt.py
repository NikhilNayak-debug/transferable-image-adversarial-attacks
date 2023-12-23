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


class VILTModel(Model):
    def __init__(self, hf_name="dandelin/vilt-b32-finetuned-vqa", hf_processor="dandelin/vilt-b32-finetuned-vqa") -> None:
        """
        Initialize the VILTModel.

        Parameters:
        - hf_name (str, optional): Hugging Face model name. Defaults to "dandelin/vilt-b32-finetuned-vqa".
        - hf_processor (str, optional): Hugging Face processor name. Defaults to "dandelin/vilt-b32-finetuned-vqa".
        """
        # Constructor to initialize the VILTModel with Hugging Face model name and processor
        self._model = ViltForQuestionAnswering.from_pretrained(hf_name).to(device)  # Load VILT model and move to device
        self._processor = ViltProcessor.from_pretrained(hf_processor)  # Initialize VILT processor
        self.loss_fn = lambda input, target: -torch.log(torch.sum(input * target))  # Define the loss function

    def forward(self, input_ids=None, pixel_values=None):
        """
        Forward pass of the VILT model.

        Parameters:
        - input_ids (torch.Tensor, optional): Input tensor for text. Defaults to None.
        - pixel_values (torch.Tensor, optional): Input tensor for images. Defaults to None.

        Returns:
        - output: Output tensor from the VILT model.
        """
        if input_ids is None:
            assert self._input_ids is not None, "You need to preprocess text first"
            input_ids = self._input_ids
        if pixel_values is None:
            assert self._pixel_values is not None, "You need to preprocess image first"
            pixel_values = self._pixel_values
        output = self._model(input_ids=input_ids, pixel_values=pixel_values)
        self._output = output
        return output

    def preprocess_text(self, text: str):
        """
        Preprocess text data before feeding it into the VILT model.

        Parameters:
        - text (str): Input text.

        Returns:
        - input_ids: Preprocessed input tensor for text.
        """
        self._input_ids = self._processor.tokenizer(text, return_tensors="pt").input_ids.to(device)
        return self._input_ids

    def preprocess_image(self, image):
        """
        Preprocess image data before feeding it into the VILT model.

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
        Generate an answer for a given question and image using the VILT model.

        Parameters:
        - question (str): Input question.
        - image: Input image.

        Returns:
        - result: Generated answer.
        """
        if not torch.is_tensor(image):
            image = self._processor.image_processor(images=image, return_tensors="pt").pixel_values.to(device)
        input_tokens = self._processor.tokenizer(text=question, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        outputs = self._model(input_ids=input_tokens, pixel_values=image)
        logits = outputs.logits
        sigmoids = torch.sigmoid(logits)
        idx = torch.sigmoid(logits).argmax(-1).item()
        answer = self._model.config.id2label[idx]
        return question + answer

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
        idx_target = self._model.config.label2id[target]
        target = torch.zeros((1, len(self._model.config.id2label))).to(device)
        target[0, idx_target] = 1
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
        - loss_vilt: Loss value.
        """
        assert self._output is not None, "You need to call forward first"
        assert self._target is not None, "You need to set target first"
        logits = self._output.logits
        sigmoids = torch.sigmoid(logits)
        loss_vilt = self.loss_fn(sigmoids, self._target)
        return loss_vilt
