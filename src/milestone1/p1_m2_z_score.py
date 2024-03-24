import numpy as np 

class Zscorer:
    """
    -------------------------------------------------------------------------------------------------------------------------------
    Implements the z-score anomaly detection.

    Atributes:
        mu: mean value of the analyzed data

        sigma: standard deviation of the analyzed data
    -------------------------------------------------------------------------------------------------------------------------------
    """

    def __init__(self, data) -> None:
        """
        -------------------------------------------------------------------------------------------------------------------------------
        Args:
            data: array of data to be analyzed
        -------------------------------------------------------------------------------------------------------------------------------
        """
        self.data = data
        self.mu = np.mean(data)
        self.sigma = np.std(data)

    def score(self, sample):
        """
        -------------------------------------------------------------------------------------------------------------------------------
        Args: 
            sample: array of instances to be scored

        Returns:
            Z score of sample based on the analyzed data
        -------------------------------------------------------------------------------------------------------------------------------
        """
        return [abs(sample - self.mu)/(self.sigma) for sample in sample]

