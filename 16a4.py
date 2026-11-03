"""Day 16 / Session 4 - Assignment: Segment a Synthetic Customer Dataset with K-Means."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SEED = 42
OUT = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["monthly_spend", "visits_per_month", "tenure_years", "support_tickets"]

# 1. reproducible unlabeled-looking data; the true labels are kept aside and only opened at the very end
X, y_true = make_blobs(n_samples=800, n_features=4, centers=4, cluster_std=1.3, random_state=SEED)
X = X * np.array([40.0, 3.0, 1.5, 1.0])                                          # very different column scales
X = X - X.min(axis=0) + np.array([20.0, 1.0, 0.5, 0.0])                          # keep every feature non-negative (spend, visits, years, tickets)
print("[1] data", X.shape)

# 2. standardise (K-Means uses Euclidean distance, so a large-scale column would dominate the clustering)
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
print("[2] scaling: K-Means assigns points by Euclidean distance, so a column measured in the hundreds (monthly_spend) would "
      "swamp one measured in single digits unless every feature is put on the same scale.")

# 3. elbow
ks = range(1, 11)
inertias = [KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xs).inertia_ for k in ks]
# elbow = the point farthest from the straight line joining the first and last inertia (kneedle idea)
xs = (np.arange(1, 11) - 1) / 9.0                                   # normalise both axes to [0, 1]
ys = (np.array(inertias) - min(inertias)) / (max(inertias) - min(inertias))
dist = np.abs((ys[-1] - ys[0]) * (xs - xs[0]) - (xs[-1] - xs[0]) * (ys - ys[0]))   # distance to the first-last chord
k_elbow = int(np.argmax(dist) + 1)
plt.figure(figsize=(7, 4))
plt.plot(list(ks), inertias, "o-")
plt.axvline(k_elbow, color="r", ls="--")
plt.annotate(f"elbow at k={k_elbow}", (k_elbow, inertias[k_elbow - 1]), xytext=(k_elbow + 1, inertias[k_elbow - 1] * 1.4),
             arrowprops={"arrowstyle": "->"})
plt.xlabel("number of clusters k")
plt.ylabel("inertia")
plt.title("Elbow method")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "16a4_elbow.png"), dpi=120)
plt.close()
print(f"[3] inertias: {[round(v, 1) for v in inertias]}  -> elbow suggests k={k_elbow}")

# 4. silhouette
sil = {k: silhouette_score(Xs, KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(Xs)) for k in range(2, 11)}
k_sil = max(sil, key=sil.get)
plt.figure(figsize=(7, 4))
plt.plot(list(sil), list(sil.values()), "o-")
plt.axvline(k_sil, color="r", ls="--", label=f"peak k={k_sil}")
plt.xlabel("number of clusters k")
plt.ylabel("silhouette score")
plt.title("Silhouette analysis")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "16a4_silhouette.png"), dpi=120)
plt.close()
print(f"[4] silhouette by k: { {k: round(v, 3) for k, v in sil.items()} }  -> peak at k={k_sil}; "
      f"elbow says {k_elbow}, silhouette says {k_sil}: {'they agree' if k_elbow == k_sil else 'they disagree'}")

# 5. final model (silhouette peak is the primary choice because the elbow is a visual judgement)
k = k_sil
km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xs)
sizes = np.bincount(km.labels_)
final_sil = silhouette_score(Xs, km.labels_)
print(f"\n[5] final KMeans k={k}: cluster_centers_.shape={km.cluster_centers_.shape}, sizes={sizes.tolist()}, "
      f"inertia={km.inertia_:.1f}, silhouette={final_sil:.3f}")

# 6. segment profiles in original units
centers = scaler.inverse_transform(km.cluster_centers_)
overall = X.mean(axis=0)
print("\n[6] segment profiles")
for c in range(k):
    hi = [f"{FEATURES[j]} {centers[c, j]:.1f}" for j in np.argsort(-(centers[c] - overall) / X.std(axis=0))[:2]]
    print(f"    segment {c}: {sizes[c]} customers ({sizes[c] / len(X):.0%}); centre " +
          ", ".join(f"{f}={v:.1f}" for f, v in zip(FEATURES, centers[c])) + f"; stands out on {hi[0]} and {hi[1]}")

# 7. only now open the answer key
true_k = len(np.unique(y_true))
print("\nConclusion:")
print(f"I chose k={k}: the silhouette score peaks there ({final_sil:.3f}) and the elbow method suggests k={k_elbow}, so the two methods "
      f"{'agree' if k_elbow == k_sil else 'disagree, and I trusted the silhouette because the elbow is a visual judgement'}. "
      f"The final model has inertia {km.inertia_:.1f} with clusters of {sizes.tolist()} customers. "
      f"Comparing with the answer key at the very end, the data was generated from {true_k} blobs, so the recovered k "
      f"{'matches' if k == true_k else 'does not match'} the planted structure. "
      f"A limitation of K-Means is that it needs k up front and assumes round, similarly sized clusters, so elongated or "
      f"very uneven segments would be split or merged incorrectly.")
