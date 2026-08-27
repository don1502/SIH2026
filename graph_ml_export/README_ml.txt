================================================================
GRAPH-ML-READY EXPORT — README
Crime Intelligence Network Dataset — Link Prediction Task
================================================================

1. HOW THE GRAPH WAS CONSTRUCTED
---------------------------------
Nodes (4,920 total) were taken directly from the entity tables, one
node per row, typed as follows:

    node_type       source table          count
    -----------     -------------------   -----
    person          persons.csv           1,000
    organization    organizations.csv       120
    location        locations.csv            150
    phone_number    phones.csv             1,400
    account         accounts.csv           1,200
    vehicle         vehicles.csv             800
    case            cases.csv                250

No synthetic or inferred nodes were added.

Edges (16,433 total) were derived from the junction and activity
tables, exactly as specified — no relationships were invented:

    edge_type          source table(s)                count
    ---------------    ----------------------------   -----
    OWNS               person_phones, person_accounts,
                        person_vehicles                3,714
    ASSOCIATED_WITH     person_organizations,
                        person_locations                2,765
    PARTICIPATED_IN     case_persons                      954
    CALLED              call_records (phone-phone,
                        with timestamp + duration)      5,000
    TRANSACTED_WITH     transactions (account-account,
                        with timestamp + amount)         4,000

Two of the activity tables reference entities by a natural key
rather than the entity's primary key:
  - call_records.caller_number / receiver_number store the
    phone_number, not phones.phone_id.
  - transactions.sender_account / receiver_account store the
    account_number, not accounts.account_id.
Both were resolved to the corresponding *_id via a lookup against
phones.csv / accounts.csv before edges were written, so every
source_id / target_id in the edge files is a graph node id
(PER_xxxx, PH_xxxx, ACC_xxxx, ORG_xxxx, LOC_xxxx, VEH_xxxx,
CASE_xxxx) consistent with node_features.csv.

The graph is treated as undirected for connectivity purposes
(OWNS/ASSOCIATED_WITH/PARTICIPATED_IN are naturally symmetric
relationships; CALLED and TRANSACTED_WITH keep their original
source/target direction in the CSV but were also treated as
undirected edges when computing connectivity and the split, since
link prediction here is about "is there a relationship" rather than
strict directionality). If your model needs directed edges, use
source_id/target_id as given; if it needs undirected, symmetrize by
adding the reverse edge.

Of the 4,920 total nodes, 4,588 participate in at least one edge of
the five modeled types; the remaining 332 (mostly organizations,
locations, or cases with no direct person/account/phone link in the
junction tables provided) are isolated and included in
node_features.csv for completeness but will not appear in the edge
lists.

2. TRAIN / VAL / TEST SPLIT METHOD
-----------------------------------
Split ratios: 70% train / 15% val / 15% test, applied to EDGES
(not nodes) — every node still appears in the training graph.

Method (connectivity-preserving edge split):
  1. Build the undirected adjacency from all 16,433 edges.
  2. Run BFS from every unvisited node to compute a spanning tree
     of each connected component, marking one edge per tree link
     as "protected" (i.e. required to stay in train to keep the
     graph connected). The dataset is a single connected component
     of 4,588 non-isolated nodes, so this produced a spanning tree
     of 4,587 protected edges.
  3. All protected edges are placed in train. Every remaining edge
     (parallel/extra edges beyond the spanning tree — e.g. a second
     call between the same two phones) is shuffled (seed=42) and
     split 15% / 15% into val / test, with the rest added to train.
  4. Result: train=11,503 (70.0%), val=2,465 (15.0%),
     test=2,465 (15.0%), total=16,433.

Verification performed: the training graph alone was confirmed to
have exactly 1 connected component spanning all 4,588 non-isolated
nodes — identical to the full graph — so removing val/test edges
does not disconnect any node and val/test pairs are guaranteed to
be reachable in the train graph for negative sampling.

Files:
  edges_train.csv  — 11,503 rows
  edges_val.csv    — 2,465 rows
  edges_test.csv   — 2,465 rows

Each file has columns: source_id, target_id, edge_type, timestamp,
weight
  - timestamp is populated only for CALLED edges (ISO 8601 UTC) and
    is blank for OWNS/ASSOCIATED_WITH/PARTICIPATED_IN.
  - weight holds duration_seconds for CALLED edges and amount for
    TRANSACTED_WITH edges; blank otherwise.

3. NODE FEATURES
-----------------
node_features.csv contains, per node:
  - node_id, node_type
  - degree            : total edge count across ALL edge types
                         (computed over the full 16,433-edge graph)
  - call_count        : number of CALLED edges touching this node
  - transaction_count : number of TRANSACTED_WITH edges touching
                         this node

No PII (names, phone numbers, IMEIs, account numbers, addresses,
etc.) is included — only IDs, type, and structural/aggregate counts,
per the privacy constraint.

4. RECOMMENDED EVALUATION FOR LINK PREDICTION
-----------------------------------------------
Standard practice for this task:
  - Train an encoder (e.g. GraphSAGE, GCN, or a simple node2vec /
    matrix-factorization baseline) on edges_train.csv only.
  - For each positive edge in edges_val.csv / edges_test.csv,
    sample an equal number of negative edges (node pairs of
    compatible types with no edge in the FULL graph, not just
    train) — this is standard negative sampling for link
    prediction and avoids leaking val/test positives as negatives.
  - Score all positive + negative pairs with the model (e.g. dot
    product or MLP over the two node embeddings).
  - Report:
      * ROC-AUC — ranking quality between positive and negative
        pairs, threshold-independent.
      * AP (Average Precision) — area under the precision-recall
        curve; more informative than AUC when negatives vastly
        outnumber positives, which is typical for sparse graphs
        like this one.
  - Use edges_val.csv for model selection / early stopping and
    hyperparameter tuning, and edges_test.csv only once, for the
    final reported metric.
  - Since edge_type is heterogeneous (5 relation types across 7
    node types), consider evaluating AUC/AP per edge_type as well
    as overall, since a single pooled score can hide poor
    performance on a rare relation (e.g. PARTICIPATED_IN, the
    smallest class at 954 edges).
  - CALLED and TRANSACTED_WITH edges carry timestamps; if a
    temporal model is desired, an alternative time-based split
    (train on earlier timestamps, test on later ones) can be
    substituted for the random split provided here — the random,
    connectivity-preserving split in this export is the general-
    purpose default requested.
