import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

class ProbabilityCalibration:
    """Fit only on the separate post-training calibration segment."""
    def __init__(self,method="none"):
        self.method=method;self.model=None
    def fit(self,probability,labels):
        p=np.asarray(probability)
        if self.method=="none":
            return self
        if len(np.unique(labels))!=2:
            raise ValueError("Calibration requires both label classes")
        if self.method=="sigmoid":
            self.model=LogisticRegression(C=1e6,max_iter=1000).fit(self._logits(p),labels)
        elif self.method=="isotonic":
            self.model=IsotonicRegression(out_of_bounds="clip").fit(p,labels)
        else:
            raise ValueError("Unknown calibration method")
        return self
    @staticmethod
    def _logits(p):
        clipped=np.clip(p,1e-6,1-1e-6)
        return np.log(clipped/(1-clipped)).reshape(-1,1)
    def predict(self,probability):
        p=np.asarray(probability)
        if self.method=="none":return p
        if self.model is None:raise ValueError("Calibrator not fitted")
        return self.model.predict_proba(self._logits(p))[:,1] if self.method=="sigmoid" else self.model.predict(p)
