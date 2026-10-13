# Unsupervised outputs adapted from the supplied screenshots

The screenshots are output examples for employee attrition. PolicyPulse now provides the corresponding analysis for insurance customers, preserving its existing theme, header, navigation and cards.

All three algorithms fit every varying prepared predictor on their reproducible sample. ID and Response are excluded. PCA is fitted on the same inputs after clustering, solely to display the assigned labels in two dimensions. The axis labels report explained variance; the interface explains that the projection can overlap even when full-feature clusters differ.

| Model | Controls and outputs |
| --- | --- |
| K-Means | Silhouette / Elbow / Manual K, selected K and best silhouette K cards, highlighted optimization curves, PCA plot with projected centers, customer summaries and WCSS/silhouette metrics |
| Hierarchical | Ward / Complete / Average linkage, K control, cluster/linkage cards, truncated dendrogram with a dashed K cutoff, PCA plot and customer summaries |
| DBSCAN | EPS and Minimum Samples, six summary cards, cluster colors and red crosses for noise in PCA, customer summaries including noise, and silhouette when defined |

The default view no longer asks for two original feature axes. Every summary table includes count, population percentage, mean age, annual premium and vintage in days, previously insured percentage, vehicle damage percentage, and response rate. Response is read only after clustering to describe segments. Rounded percentages may sum slightly above or below 100%; underlying percentages use exact counts.

K-Means uses up to 3,500 customers, hierarchical uses 1,200 and DBSCAN uses 2,500. The pages show analyzed and total record counts. Silhouette uses up to 1,500 reproducibly selected assigned points and excludes DBSCAN noise. DBSCAN can legitimately return all noise or a single cluster, for which silhouette is undefined. No reference screenshot values are hardcoded.

The elbow recommendation uses the largest normalized distance from the endpoint chord. The dendrogram cutoff lies between the last within-group and next between-group merge. If the merge heights tie, the page discloses that a horizontal cutoff cannot uniquely represent K.

Validation: 61 automated checks passed, including independent SVD checks of PCA variance, raw-record summary reconciliation, all-feature fitting, noise percentages, and matching dendrogram cuts to hierarchical assignments for all three linkages. The full-data route verifier and independent model audit also passed. Browser form submissions verified the three result pages, working automatic/manual K controls, and desktop and narrow-screen chart layouts. Preview images are saved beside this document.
