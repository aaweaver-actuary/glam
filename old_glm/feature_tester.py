from typing import Protocol
from hit_ratio.old_glm.model_result import BaseModelResult
from hit_ratio.old_glm.model_fitter import BaseModelFitter
from hit_ratio.old_glm.glm

class BaseFeatureTester(Protocol):
    """Protocol for testing new features to add to the model."""

    def test(
        self, new_feature: str, model: BaseModelResult, fitter: BaseModelFitter
    ) -> None:
        """Test the model, before and after adding the new feature."""
        ...


class FeatureTester:
    """Concrete implementation of the feature tester protocol."""

    def test(
        self, new_feature: str, model: BaseModelResult, fitter: BaseModelFitter
    ) -> None:
        """Test the model, before and after adding the new feature."""
        current_model = model
        current_features = current_model.features
        new_features = current_features + [new_feature]
        new_model = model.add_feature(new_feature, fitter)
