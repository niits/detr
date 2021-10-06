from typing import List, Tuple
from torch import nn, no_grad


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        class_cost_weight: float = 1,
        bbox_cost_weight: float = 1,
        giou_cost_weight: float = 1,
    ):
        """[summary]

        Args:
            class_cost_weight (float, optional): Trọng số cho giá trị của việc phân loại. Defaults to 1.
            bbox_cost_weight (float, optional): Trọng số cho giá trị của việc tìm bounding box. Defaults to 1.
            giou_cost_weight (float, optional): Trọng số cho giá trị của việc tính giou. Defaults to 1.
        """
        super(HungarianMatcher, self).__init__()
        self.class_cost_weight = class_cost_weight
        self.bbox_cost_weight = bbox_cost_weight
        self.giou_cost_weight = giou_cost_weight
        assert (
            class_cost_weight * bbox_cost_weight * giou_cost_weight != 0
        ), "all costs cant be 0"

    @no_grad()
    def forward(self, outputs: dict, targets: dict) -> List[Tuple[int, int]]:

        pass
