import torch

from scene.gaussian_model import GaussianModel


class DummyColorNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.use_drop_out = True
        self.linear = torch.nn.Linear(1, 1)


class DummyMapGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.use_features_mask = True
        self.bn = torch.nn.BatchNorm2d(1)


def make_model():
    model = GaussianModel.__new__(GaussianModel)
    model.eval_mode = False
    model.use_color_net = True
    model.color_net = DummyColorNet()
    model.use_kmap_pjmap = True
    model.use_okmap = False
    model.map_generator = DummyMapGenerator()
    model.use_features_mask = True
    if hasattr(model, "_eval_state_stack"):
        delattr(model, "_eval_state_stack")
    return model


def test_set_eval_sets_appearance_modules_to_eval_and_restores_train_state():
    model = make_model()
    model.color_net.train(True)
    model.map_generator.train(True)

    model.set_eval(True)

    assert model.eval_mode is True
    assert model.color_net.training is False
    assert model.map_generator.training is False
    assert model.color_net.use_drop_out is False
    assert model.use_features_mask is False
    assert model.map_generator.use_features_mask is False

    model.set_eval(False)

    assert model.eval_mode is False
    assert model.color_net.training is True
    assert model.map_generator.training is True
    assert model.color_net.use_drop_out is True
    assert model.use_features_mask is True
    assert model.map_generator.use_features_mask is True


def test_set_eval_restores_modules_that_were_already_eval():
    model = make_model()
    model.color_net.eval()
    model.map_generator.eval()
    model.color_net.use_drop_out = False
    model.map_generator.use_features_mask = False
    model.use_features_mask = False

    model.set_eval(True)
    model.set_eval(False)

    assert model.eval_mode is False
    assert model.color_net.training is False
    assert model.map_generator.training is False
    assert model.color_net.use_drop_out is False
    assert model.use_features_mask is False
    assert model.map_generator.use_features_mask is False


def test_nested_set_eval_restores_stack_in_order():
    model = make_model()
    model.set_eval(True)
    model.set_eval(True)

    model.set_eval(False)
    assert model.eval_mode is True
    assert model.color_net.training is False
    assert model.map_generator.training is False

    model.set_eval(False)
    assert model.eval_mode is False
    assert model.color_net.training is True
    assert model.map_generator.training is True
