# Pattern2
PROJECT: SUPERVISED STATE AGGREGATION FOR EURUSD

Build a complete Python research framework from scratch.

The objective is to compare:

Gini Decision Tree
Cross-Entropy Decision Tree
Jensen-Shannon State Clustering
SSE (Euclidean) State Clustering

for aggregating M5 Markov states.

IMPORTANT CONCEPTUAL REQUIREMENTS

There are TWO separate families of models.

FAMILY A — DECISION TREES

These are supervised models.

Features:

X = [s1, s2, s3, s4, s5]

where:

s1 = state(t-5)
s2 = state(t-4)
s3 = state(t-3)
s4 = state(t-2)
s5 = state(t-1)

Target:

Y = state(t)

The target is ALWAYS the next day's state.

Target values:

-1
 0
+1

Do NOT use:

Direction
Entropy
Gini
Classification Error
Confidence

as features.

These are derived statistics and are NOT tree inputs.

FAMILY B — STATE CLUSTERING

After transition probabilities have been estimated for every M5 state:

Represent each state as:

[P(-1), P(0), P(+1)]

Then cluster those probability vectors using:

Jensen-Shannon Distance
SSE / Euclidean Distance

These clustering methods are completely independent of the decision trees.

DATA ACQUISITION

Use:

yfinance

Ticker:

EURUSD=X

Start date:

2003-12-01

End date:

2025-12-31

Download only:

Close

column.

IMPORTANT

DO NOT:

generate synthetic data
generate random data
create fallback data
fill missing data with random values

If download fails:

raise RuntimeError(...)

and stop execution.

DATA PREPARATION

Compute log returns:

log_ret = np.log(
    Close / Close.shift(1)
)

Compute:

q30 = 30th percentile
q70 = 70th percentile

of all log returns.

Convert returns into states:

return < q30      -> -1

q30 <= return <= q70 -> 0

return > q70      -> +1

Store:

State
STATE CONSTRUCTION

The code must support:

M1
M2
M3
M4
M5

through a user parameter:

LOOKBACK = 5

For M5:

s1 = State.shift(5)
s2 = State.shift(4)
s3 = State.shift(3)
s4 = State.shift(2)
s5 = State.shift(1)

Y = State

Drop NA rows.

STATE ORDERING

THIS IS EXTREMELY IMPORTANT.

All state tables must be ordered so that the OLDEST STATE changes fastest.

The ordering must be:

(-1,-1,-1,-1,-1)
( 0,-1,-1,-1,-1)
( 1,-1,-1,-1,-1)

(-1, 0,-1,-1,-1)
( 0, 0,-1,-1,-1)
( 1, 0,-1,-1,-1)

...

The ordering loop must therefore be:

order = []

for s5 in [-1,0,1]:
    for s4 in [-1,0,1]:
        for s3 in [-1,0,1]:
            for s2 in [-1,0,1]:
                for s1 in [-1,0,1]:

                    order.append(
                        (s1,s2,s3,s4,s5)
                    )

Use this ordering consistently for:

state tables
cluster member tables
exports
TRANSITION TABLE

For every M5 state:

Compute raw counts:

Count(-1)
Count(0)
Count(+1)

Then compute:

P(-1)
P(0)
P(+1)

using empirical frequencies only.

IMPORTANT

DO NOT USE:

Bayesian shrinkage
Laplace smoothing
alpha parameters
probability smoothing

Use raw empirical probabilities only.

DERIVED STATE STATISTICS

For every state compute:

Direction
Direction = P(+1) - P(-1)
Classification Error

Use the ISLR definition:

E=1−max
k
	​

(p
k
	​

)

where:

k ∈ {-1,0,+1}
Gini Index

Use the exact ISLR definition:

G=Σp
k
	​

(1−p
k
	​

)

Do NOT rewrite as:

1−Σp
2

even though mathematically equivalent.

Use the book definition in comments and documentation.

Cross-Entropy

Use the exact ISLR definition:

D=−Σp
k
	​

log(p
k
	​

)

Handle:

log(0)

safely.

Confidence
Confidence = max(P)
METHOD 1 — GINI DECISION TREE

Features:

[s1,s2,s3,s4,s5]

Target:

Y

Criterion:

criterion='gini'

Tune:

max_depth = 1..10

using:

StratifiedKFold
cross_val_score

Select best depth.

Train final tree.

METHOD 2 — CROSS-ENTROPY DECISION TREE

Features:

[s1,s2,s3,s4,s5]

Target:

Y

Criterion:

criterion='entropy'

This corresponds to Cross-Entropy / Information Gain.

Tune:

max_depth = 1..10

using:

StratifiedKFold
cross_val_score

Select best depth.

Train final tree.

TREE CLUSTERS

After fitting:

Use:

tree.apply(X)

to obtain leaf IDs.

Each leaf node becomes a cluster.

Therefore:

Cluster = Tree Leaf
CLUSTER TRANSITION PROBABILITIES

For every tree cluster:

Pool all raw observations.

Example:

Y=-1 : 40

Y=0 : 30

Y=+1 : 30

Then:

P(-1)=0.40

P(0)=0.30

P(+1)=0.30

DO NOT average state probabilities.

Always recompute from pooled raw observations.

METHOD 3 — JENSEN-SHANNON CLUSTERING

Represent each state by:

[P(-1),P(0),P(+1)]

Distance:

scipy.spatial.distance.jensenshannon

Compute full pairwise distance matrix.

Use:

AgglomerativeClustering

or hierarchical clustering.

Linkage:

average

DO NOT use Ward.

Determine optimal cluster count:

k = 2..20

using:

silhouette_score

Select best k automatically.

METHOD 4 — SSE / EUCLIDEAN CLUSTERING

Represent each state by:

[P(-1),P(0),P(+1)]

Distance:

Σ(P
i
	​

−P
j
	​

)
2

Equivalent to Euclidean distance.

Use:

AgglomerativeClustering

Linkage:

average

Determine:

k = 2..20

using silhouette score.

Select best k automatically.

CLUSTER RE-ESTIMATION

For ALL FOUR METHODS:

After assigning states to clusters:

Pool raw counts.

Recompute:

P(-1)
P(0)
P(+1)

from pooled counts.

DO NOT average probabilities.

CLUSTER STATISTICS

For every cluster calculate:

N_States

N_Observations

P(-1)

P(0)

P(+1)

Direction

Classification_Error

Gini

Cross_Entropy

Confidence

using the ISLR formulas above.

CLUSTER ORDERING

Order clusters by:

Direction = P(+1) - P(-1)

ascending.

The final order should be:

Most Bearish
↓
Bearish
↓
Neutral
↓
Bullish
↓
Most Bullish
REQUIRED OUTPUT FILES

Export:

state_transition_table.csv

gini_tree_cluster_summary.csv

entropy_tree_cluster_summary.csv

js_cluster_summary.csv

sse_cluster_summary.csv

gini_tree_cluster_members.csv

entropy_tree_cluster_members.csv

js_cluster_members.csv

sse_cluster_members.csv

comparison_report.csv
VISUALIZATIONS

Generate:

Gini Tree Diagram
Cross-Entropy Tree Diagram
Jensen-Shannon Dendrogram
SSE Dendrogram
Cluster Transition Probability Heatmaps
Cluster Size Charts
Cluster Direction Charts
COMPARISON REPORT

Create a final comparison table containing:

Method

Best_Depth

CV_Accuracy

Number_of_Clusters

Average_Cluster_Size

Average_CrossEntropy

Average_Confidence

Average_Direction

Silhouette_Score

where applicable.

CODE QUALITY REQUIREMENTS

The code must be:

fully runnable
modular
heavily commented
research quality
reproducible

Every major block must explain:

Mathematical objective.
Statistical interpretation.
Why the step is performed.
How the output should be interpreted.

Generate complete Python code implementing all requirements above.
