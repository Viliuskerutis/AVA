import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score


class ExperimentAnalysis:
    def __init__(self, similarity_matrix, image_names):
        self.similarity_matrix = similarity_matrix
        self.image_names = image_names

    def extract_scores_optimized(self, similarity_threshold=-1):
        painting_names = [name.rsplit("_", 1)[0] for name in self.image_names]
        num_images = len(self.image_names)
        painting_name_matrix = np.equal.outer(painting_names, painting_names)
        diagonal_scores = np.diag(self.similarity_matrix)
        upper_triangle_indices = np.triu_indices(num_images, k=1)
        upper_triangle_matching_mask = painting_name_matrix[upper_triangle_indices]
        upper_triangle_scores = self.similarity_matrix[upper_triangle_indices]

        if similarity_threshold == -1:
            return diagonal_scores, upper_triangle_scores[~upper_triangle_matching_mask]
        else:
            return (
                np.concatenate(
                    [
                        diagonal_scores,
                        upper_triangle_scores[
                            (upper_triangle_matching_mask)
                            & (upper_triangle_scores > similarity_threshold)
                        ],
                    ]
                ),
                upper_triangle_scores[
                    ~(
                        (upper_triangle_matching_mask)
                        & (upper_triangle_scores > similarity_threshold)
                    )
                ],
            )

    def run_analysis(self, sim_type):
        matching_scores, non_matching_scores = self.extract_scores_optimized()
        self.plot_combined_similarity_histogram(
            matching_scores, non_matching_scores, sim_type
        )
        self.plot_similarity_boxplots(matching_scores, non_matching_scores, sim_type)
        self.plot_roc_curve_with_thresholds(
            matching_scores, non_matching_scores, sim_type
        )
        self.plot_threshold_analysis(
            matching_scores,
            non_matching_scores,
            sim_type,
            start=0.825,
            end=0.9,
            step=0.01,
            tolerance=0.025,
        )

    def plot_combined_similarity_histogram(
        self, matching_scores, non_matching_scores, sim_type
    ):
        fig, ax1 = plt.subplots(figsize=(12, 8))
        bins = np.linspace(0, 1, 30)
        ax1.hist(
            non_matching_scores,
            bins=bins,
            alpha=0.6,
            label="Non-Matching Scores",
            color="tab:orange",
        )
        ax1.set_xlabel("Similarity Score")
        ax1.set_ylabel("Frequency (Non-Matching Scores)", color="tab:orange")
        ax2 = ax1.twinx()
        ax2.hist(
            matching_scores,
            bins=bins,
            alpha=0.6,
            label="Matching Scores",
            color="tab:blue",
        )
        ax2.set_ylabel("Frequency (Matching Scores)", color="tab:blue")
        plt.title(f"{sim_type} Combined Histogram of Similarity Scores")
        fig.tight_layout()
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    def plot_similarity_boxplots(self, matching_scores, non_matching_scores, sim_type):
        plt.figure(figsize=(8, 6))
        plt.boxplot(
            [matching_scores, non_matching_scores], labels=["Matching", "Non-Matching"]
        )
        plt.title(f"{sim_type} Box Plot of Similarity Scores")
        plt.ylabel("Similarity Score")
        plt.grid(True)
        plt.show()

    def plot_roc_curve_with_thresholds(
        self, matching_scores, non_matching_scores, sim_type
    ):
        y_true = [1] * len(matching_scores) + [0] * len(non_matching_scores)
        y_scores = np.concatenate((matching_scores, non_matching_scores))
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auc_score = roc_auc_score(y_true, y_scores)

        # Compute Youden's J statistic
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]

        print(f"Optimal Threshold (Youden's J): {optimal_threshold:.4f}")

        # Plot ROC curve
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})", color="blue")
        plt.plot([0, 1], [0, 1], "--", color="gray", label="Random Guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{sim_type} ROC Curve for Similarity Thresholds")
        plt.legend()
        plt.grid(True)

        # Mark the optimal threshold based on Youden's J statistic
        plt.axvline(
            fpr[optimal_idx],
            color="red",
            linestyle="--",
            label=f"Optimal Threshold (Youden's J): {optimal_threshold:.4f}",
        )
        plt.scatter(
            fpr[optimal_idx],
            tpr[optimal_idx],
            color="red",
            label="Best Threshold Point",
            zorder=3,
        )

        plt.legend()
        plt.show()

    def plot_threshold_analysis(
        self,
        matching_scores,
        non_matching_scores,
        sim_type,
        start=0.5,
        end=1,
        step=0.025,
        tolerance=0.05,
    ):
        thresholds = np.arange(start, end + 0.0001, step)
        tpr, fpr, accuracies, f1_scores = [], [], [], []

        y_true = [1] * len(matching_scores) + [0] * len(non_matching_scores)
        y_scores = np.concatenate((matching_scores, non_matching_scores))

        for threshold in tqdm(thresholds):
            y_pred = (y_scores >= threshold).astype(int)

            tp = sum(score >= threshold for score in matching_scores)
            fn = sum(score < threshold for score in matching_scores)
            fp = sum(score >= threshold for score in non_matching_scores)
            tn = sum(score < threshold for score in non_matching_scores)

            tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
            accuracies.append(accuracy_score(y_true, y_pred))
            f1_scores.append(f1_score(y_true, y_pred))

        optimal_fpr_idx = np.argmin(np.abs(np.array(fpr) - tolerance))
        optimal_threshold_fpr = thresholds[optimal_fpr_idx]
        optimal_accuracy_idx = np.argmax(np.array(accuracies) >= (1 - tolerance))
        optimal_threshold_accuracy = thresholds[optimal_accuracy_idx]
        optimal_f1_idx = np.argmax(f1_scores)
        optimal_threshold_f1 = thresholds[optimal_f1_idx]
        highest_f1_score = f1_scores[optimal_f1_idx]  # Store the highest F1-score

        plt.figure(figsize=(12, 8))
        plt.plot(thresholds, tpr, label="TPR", color="green")
        plt.plot(thresholds, fpr, label="FPR", color="red")
        plt.plot(thresholds, accuracies, label="Accuracy", color="blue")
        plt.plot(thresholds, f1_scores, label="F1-Score", color="purple")

        plt.axvline(
            optimal_threshold_fpr,
            color="orange",
            linestyle="--",
            label=f"FPR<{tolerance*100}%: {optimal_threshold_fpr:.2f}",
        )
        plt.axvline(
            optimal_threshold_accuracy,
            color="cyan",
            linestyle="--",
            label=f"Accuracy>{(1-tolerance)*100}%: {optimal_threshold_accuracy:.2f}",
        )
        plt.axvline(
            optimal_threshold_f1,
            color="magenta",
            linestyle="--",
            label=f"Best F1-Score: {optimal_threshold_f1:.2f} (F1={highest_f1_score:.4f})",
        )

        plt.xticks(np.arange(start, end + step, step), rotation=45)
        plt.xlabel("Threshold")
        plt.ylabel("Score / Rate")
        plt.title(
            f"{sim_type} Threshold Analysis\nHighest F1-score: {highest_f1_score:.4f}"
        )
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.show()
