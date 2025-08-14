from sklearn.decomposition import PCA
import numpy as np

class ImagePCACompressor:
    def __init__(self, variance_ratio: float):
        """
        variance_ratio: float in (0,1], e.g., 0.95 means keep 95% variance
        """
        self.variance_ratio = variance_ratio
        self.pca_models = []

    def fit_transform(self, img: np.ndarray) -> np.ndarray:
        """
        Apply PCA on each color channel and reconstruct image.
        img: numpy array (H, W, 3)
        """
        compressed_channels = []
        self.pca_models = []

        for channel in range(img.shape[2]):
            pca = PCA(self.variance_ratio, svd_solver='full')
            transformed = pca.fit_transform(img[:, :, channel])
            reconstructed = pca.inverse_transform(transformed)
            compressed_channels.append(reconstructed)
            self.pca_models.append(pca)

        compressed_img = np.stack(compressed_channels, axis=2)
        compressed_img = np.clip(compressed_img, 0, 255).astype(np.uint8)
        return compressed_img

    def compression_stats(self, original_size_bytes: int, compressed_size_bytes: int) -> dict:
        """
        Calculate reduction stats based on PCA components.
        """
        total_components = sum(int(pca.n_components_) for pca in self.pca_models)
        return {
            "variance_ratio": float(self.variance_ratio),
            "total_components": int(total_components),
            "original_size_MB": round(original_size_bytes / (1024 * 1024), 3),
            "original_size_KB": round(original_size_bytes / 1024, 1),
            "compressed_size_MB": round(compressed_size_bytes / (1024 * 1024), 3),
            "compressed_size_KB": round(compressed_size_bytes / 1024, 1),
            "reduction_percent": round(
                100 * (1 - compressed_size_bytes / original_size_bytes), 2
            )
        }