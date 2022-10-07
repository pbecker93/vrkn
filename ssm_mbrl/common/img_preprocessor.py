import torch

jit = torch.jit


class ImgPreprocessor(jit.ScriptModule):

    __constants__ = ["_depth_bits", "_add_cb_noise"]
    _depth_bits: int
    _add_cb_noise: bool

    def __init__(self, depth_bits: int, add_cb_noise: bool):
        super(ImgPreprocessor, self).__init__()
        assert 1 <= depth_bits <= 8
        self._depth_bits = depth_bits
        self._add_cb_noise = add_cb_noise

    @torch.jit.script_method
    def forward(self, img):
        img = img.float()
        img.div_(2 ** (8 - self._depth_bits)).floor_().div_(2 ** self._depth_bits).sub_(0.5)
        if self._add_cb_noise:
            img.add_(torch.rand_like(img).div_(2 ** self._depth_bits))
        return img

    @torch.jit.script_method
    def revert_preprocessing(self, img):
        img = ((img + 0.5) * 255.0).floor().type(torch.uint8)
        return img
