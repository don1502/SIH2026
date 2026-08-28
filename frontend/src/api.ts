import axios from "axios";

const api = axios.create({ baseURL: "/api" });

export interface SearchResult {
  id: string;
  name: string | null;
  labels: string[];
  role: string | null;
  risk_score: string | null;
  pagerank: number | null;
}

export interface CyNode {
  data: {
    id: string;
    label: string;
    type: string;
    role?: string | null;
    risk_score?: string | null;
    pagerank?: number | null;
    community?: number | null;
    is_center?: boolean;
    suspect_probability?: number | null;
    is_suspect?: boolean;
    anomaly_score?: number | null;
    is_anomalous?: boolean;
  };
}

export interface CyEdge {
  data: {
    id: string;
    source: string;
    target: string;
    label: string;
    [key: string]: unknown;
  };
}

export interface Subgraph {
  nodes: CyNode[];
  edges: CyEdge[];
}

export interface EntityProfile {
  id: string;
  labels: string[];
  properties: Record<string, unknown>;
  relationship_counts: { rel_type: string; count: number }[];
  neighbors: {
    rel_type: string;
    outgoing: boolean;
    neighbor_id: string;
    neighbor_name: string | null;
    neighbor_labels: string[];
    rel_props: Record<string, unknown>;
  }[];
}

export interface GraphStats {
  total_nodes: number;
  total_relationships: number;
  node_labels: Record<string, number>;
  relationship_types: { rel_type: string; count: number }[];
}

export interface RankedEntity {
  id: string;
  name: string;
  score: number;
  community: number | null;
  role: string | null;
  risk_score: string | null;
}

export interface Suspect {
  person_id: string;
  name: string;
  age: string | null;
  gender: string | null;
  role: string | null;
  risk_score: string | null;
  suspect_probability: number;
  is_suspect: boolean;
  anomaly_score?: number;
  indicators: string[];
}

export interface PredictResult {
  suspects: Suspect[];
  graph: Subgraph;
  summary: {
    persons_scored: number;
    flagged_suspects: number;
    top_probability: number;
  };
  ingestion?: {
    recognized: Record<string, string>;
    unrecognized: string[];
  };
  anomalies?: AnomalyBundle;
}

export interface MLMetrics {
  task: string;
  label: string;
  n_labeled: number;
  n_positive: number;
  n_negative: number;
  test: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    roc_auc: number;
    pr_auc: number;
  };
  cv_roc_auc_mean: number;
  cv_roc_auc_std: number;
  feature_importances: { feature: string; importance: number }[];
}

export interface AnomalyTxn {
  transaction_id: string;
  sender_account: string;
  receiver_account: string;
  sender_owner: string;
  receiver_owner: string;
  amount: number;
  transaction_type: string;
  timestamp: string;
  anomaly_score: number;
  reasons: string[];
}

export interface AnomalyCall {
  phone_id: string;
  phone_number: string;
  owner: string;
  call_count: number;
  distinct_contacts: number;
  max_calls_in_hour: number;
  total_duration: number;
  anomaly_score: number;
  reasons: string[];
}

export interface AnomalyPerson {
  person_id: string;
  name: string;
  role: string | null;
  risk_score: string | null;
  anomaly_score: number;
  reasons: string[];
}

export interface AnomalyBundle {
  transactions: AnomalyTxn[];
  calls: AnomalyCall[];
  persons: AnomalyPerson[];
}

export const getHealth = () => api.get("/health").then((r) => r.data);
export const getStats = () => api.get<GraphStats>("/stats").then((r) => r.data);
export const search = (q: string) =>
  api.get<SearchResult[]>("/entities/search", { params: { q, limit: 15 } }).then((r) => r.data);
export const getProfile = (id: string) =>
  api.get<EntityProfile>(`/entities/${id}`).then((r) => r.data);
export const getSubgraph = (id: string, hops = 1) =>
  api.get<Subgraph>(`/entities/${id}/subgraph`, { params: { hops, limit: 250 } }).then((r) => r.data);
export const getTop = (metric: string) =>
  api.get<RankedEntity[]>("/analytics/top", { params: { metric, limit: 10 } }).then((r) => r.data);
export const getCommunities = () =>
  api.get<{ num_communities: number; sizes: Record<string, number> }>("/analytics/communities").then((r) => r.data);
export const getMlMetrics = () => api.get<MLMetrics>("/ml/metrics").then((r) => r.data);
export const predictSample = (caseId?: string) =>
  api.get<PredictResult>("/predict/sample", { params: caseId ? { case_id: caseId } : {} }).then((r) => r.data);
export const predictUpload = (files: File[]) => {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  return api
    .post<PredictResult>("/predict/upload", form, { headers: { "Content-Type": "multipart/form-data" } })
    .then((r) => r.data);
};
export const getAnomalyTransactions = (limit = 20) =>
  api.get<AnomalyTxn[]>("/anomalies/transactions", { params: { limit } }).then((r) => r.data);
export const getAnomalyCalls = (limit = 20) =>
  api.get<AnomalyCall[]>("/anomalies/calls", { params: { limit } }).then((r) => r.data);
export const getAnomalyPersons = (limit = 20) =>
  api.get<AnomalyPerson[]>("/anomalies/persons", { params: { limit } }).then((r) => r.data);
