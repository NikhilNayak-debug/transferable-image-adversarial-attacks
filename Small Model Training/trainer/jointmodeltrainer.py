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


class JointTrainer():
    def __init__(self,
                 models,
                 images: List,
                 texts: List[str],
                 epochs: int,
                 delta: float = 0.05,
                 epsilon: float = 10e-4,
                 optimizer=optim.Adam,
                 optimizer_params={"lr": 0.05}
                 ) -> None:
        """
        Initialize the JointTrainer.

        Parameters:
        - models (List[Model]): List of models to be trained jointly.
        - images (List): List of images for training.
        - texts (List[str]): List of text inputs corresponding to the images.
        - epochs (int): Number of training epochs.
        - delta (float, optional): Perturbation magnitude for adversarial training. Defaults to 0.05.
        - epsilon (float, optional): A small constant to avoid division by zero. Defaults to 10e-4.
        - optimizer: The optimizer used for training. Defaults to optim.Adam.
        - optimizer_params (dict, optional): Parameters for the optimizer. Defaults to {"lr": 0.05}.
        """
        assert not (len(images) > 1 and len(texts) > 1), "You cannot train with both multiple images and multiple texts"
        self._models = models
        self._images = images
        self._texts = texts
        self._optimizer = optimizer
        self._optimizer_params = optimizer_params
        self._epochs = epochs
        self._epsilon = epsilon
        self._delta = delta

    def train(self):
        """
        Train the models jointly.

        Returns:
        - change (torch.Tensor): Perturbation tensor applied during training.
        - adversarial_images (torch.Tensor): Images after applying the perturbation.
        """
        epoch_size = max(map(len, [self._images, self._texts]))
        self._images.extend([self._images[0]] * (epoch_size - len(self._images)))
        self._texts.extend([self._texts[0]] * (epoch_size - len(self._texts)))

        # Preprocess the first image to get encoding dimensions
        img_encoding = self._models[0].preprocess_image(self._images[0])

        # Initialize perturbation tensor
        change = torch.zeros_like(img_encoding).to(device)
        change.requires_grad = True

        # Initialize optimizer
        optimizer = self._optimizer([change], **self._optimizer_params)

        counter = 0
        for epoch in range(self._epochs):
            for image, text in zip(self._images, self._texts):
                loss = torch.zeros(1).to(device)
                counter += 1

                # Forward pass through each model and accumulate the loss
                for model in self._models:
                    input_ids = model.preprocess_text(text)
                    model.forward(input_ids=input_ids, pixel_values=img_encoding + change)
                    loss += model.loss

                    # Print internal loss for each model every 10 iterations
                    if counter % 10 == 0:
                        print("Model_internal_loss:", model.loss.item())

                # Print total loss every 49 iterations
                if counter % 49 == 0:
                    print("Total loss:", loss.item())

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Clamp perturbation tensor
                with torch.no_grad():
                    change[:] = change.clamp(-self._delta, self._delta)

                optimizer.zero_grad()

        return change, img_encoding + change
