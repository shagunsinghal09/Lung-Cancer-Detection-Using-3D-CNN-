from ml.pipeline import LungCancerPipeline


class PipelineService:
    def __init__(self) -> None:
        self.pipeline = LungCancerPipeline()

    def predict(self, path: str) -> dict:
        outcome = self.pipeline.predict(path)
        return {
            "label": outcome["label"],
            "probability": round(float(outcome["probability"]), 4),
            "confidence": round(float(outcome["confidence"]), 4),
            "disclaimer": "Research/educational output only. Not a clinical diagnosis.",
        }
