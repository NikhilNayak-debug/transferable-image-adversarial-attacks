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


class Trainer():
    def __init__(self,
                 model,
                 images: List,
                 texts: List,
                 epochs: int,
                 delta: float = 0.3,
                 epsilon: float = 10e-4,
                 optimizer=optim.Adam,
                 optimizer_params={"lr": 10e-4}
                 ) -> None:
        """
        Initialize the Trainer.

        Parameters:
        - model (Model): The model to be trained.
        - images (List): List of images for training.
        - texts (List): List of text inputs corresponding to the images.
        - epochs (int): Number of training epochs.
        - delta (float, optional): Perturbation magnitude for adversarial training. Defaults to 0.3.
        - epsilon (float, optional): A small constant to avoid division by zero. Defaults to 10e-4.
        - optimizer: The optimizer used for training. Defaults to optim.Adam.
        - optimizer_params (dict, optional): Parameters for the optimizer. Defaults to {"lr": 10e-4}.
        """
        assert not (len(images) > 1 and len(texts) > 1), "You cannot train with both multiple images and multiple texts"
        self._model = model
        self._images = images
        self._texts = texts
        self._optimizer = optimizer
        self._optimizer_params = optimizer_params
        self._epochs = epochs
        self._epsilon = epsilon
        self._delta = delta

    def train(self):
        """
        Train the model.

        Returns:
        - change (torch.Tensor): Perturbation tensor applied during training.
        - adversarial_images (torch.Tensor): Images after applying the perturbation.
        """
        epoch_size = max(map(len, [self._images, self._texts]))
        self._images.extend([self._images[0]] * (epoch_size - len(self._images)))
        self._texts.extend([self._texts[0]] * (epoch_size - len(self._texts)))

        # Preprocess images and texts and initialize the change tensor for adversarial training
        image_encodings = [self._model.preprocess_image(image) for image in self._images]
        text_encodings = [self._model.preprocess_text(text) for text in self._texts]
        change = torch.zeros_like(image_encodings[0]).to(device)
        change.requires_grad = True
        optimizer = self._optimizer([change], **self._optimizer_params)

        # Training loop
        pbar = tqdm(range(self._epochs))
        for epoch in pbar:
            for image, text in zip(image_encodings, text_encodings):
                self._model.forward(input_ids=text, pixel_values=image + change)
                loss = self._model.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    change[:] = change.clamp(-self._delta, self._delta)
                pbar.set_postfix({'loss': loss.item()})

        return change, image + change
