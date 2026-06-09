import numpy as np
import pytest
import torch

from src.models.cae import CAE
from src.models.dcnn import BlockB, DCNN
from src.train.common import EarlyStopping
from src.train.train_cae import compute_tau
from src.utils.focal_loss import FocalLoss


def test_cae_rejects_runtime_shape_mismatch():
    model = CAE(input_shape=(3, 8, 8), latent_dim=4)
    with pytest.raises(ValueError, match='input must have shape'):
        model(torch.zeros(1, 3, 16, 16))


def test_dcnn_validates_constructor_arguments():
    with pytest.raises(ValueError, match='num_classes'):
        DCNN(num_classes=1)
    with pytest.raises(ValueError, match='dropout'):
        DCNN(dropout=1.0)


def test_block_b_has_no_residual_connection():
    block = BlockB().eval()
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.zero_()
        output = block(torch.ones(1, 256, 4, 4))
    assert torch.count_nonzero(output) == 0


def test_focal_loss_validates_reduction():
    with pytest.raises(ValueError, match='reduction'):
        FocalLoss(reduction='invalid')


def test_cae_tau_requires_validation_scores():
    with pytest.raises(ValueError, match='validation-normal'):
        compute_tau(np.array([], dtype=np.float32))


def test_early_stopping_restores_best_weights():
    model = torch.nn.Linear(2, 1)
    stopper = EarlyStopping(patience=2, mode='min')
    stopper.step(0.5, model)
    best_weight = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(10.0)
    stopper.restore_best(model)
    assert torch.equal(model.weight, best_weight)
