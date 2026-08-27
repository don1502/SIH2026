import { useEffect, useState } from "react";
import {
  GraphStats,
  RankedEntity,
  SearchResult,
  getCommunities,
  getStats,
  getTop,
  search,
} from "../api";

interface Props {
  onSelect: (id: string) => void;
}

export default function Sidebar({ onSelect }: Props) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [stats, setStats] = useState<GraphStats | null>(null);
  const [top, setTop] = useState<RankedEntity[]>([]);
  const [communities, setCommunities] = useState<{ num_communities: number } | null>(null);

  useEffect(() => {
    getStats().then(setStats).catch(() => {});
    getTop("pagerank").then(setTop).catch(() => {});
    getCommunities().then(setCommunities).catch(() => {});
  }, []);

  useEffect(() => {
    if (query.trim().length < 1) {
      setResults([]);
      return;
    }
    const t = setTimeout(() => {
      search(query).then(setResults).catch(() => {});
    }, 250);
    return () => clearTimeout(t);
  }, [query]);

  return (
    <aside className="sidebar">
      <div className="search-box">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search person, org, phone, account…"
        />
      </div>

      {results.length > 0 && (
        <div className="result-list">
          {results.map((r) => (
            <div key={r.id} className="result-item" onClick={() => onSelect(r.id)}>
              <span className="result-name">{r.name || r.id}</span>
              <span className="result-meta">
                {r.labels.filter((l) => l !== "Entity")[0]}
                {r.role ? ` · ${r.role}` : ""}
              </span>
            </div>
          ))}
        </div>
      )}

      {stats && (
        <div className="panel">
          <h3>Knowledge Graph</h3>
          <div className="stat-row">
            <span>{stats.total_nodes.toLocaleString()} nodes</span>
            <span>{stats.total_relationships.toLocaleString()} edges</span>
          </div>
          <div className="chips">
            {Object.entries(stats.node_labels).map(([k, v]) => (
              <span key={k} className="chip">{k} {v}</span>
            ))}
          </div>
          {communities && (
            <p className="hint">{communities.num_communities} detected communities</p>
          )}
        </div>
      )}

      {stats && (
        <div className="panel">
          <h3>Relationships</h3>
          <div className="chips">
            {stats.relationship_types.map((r) => (
              <span key={r.rel_type} className="chip">{r.rel_type} · {r.count}</span>
            ))}
          </div>
        </div>
      )}

      {top.length > 0 && (
        <div className="panel">
          <h3>Most central persons</h3>
          <div className="rank-list">
            {top.map((l, i) => (
              <div key={l.id} className="rank-item" onClick={() => onSelect(l.id)}>
                <span className="rank-idx">{i + 1}</span>
                <span className="rank-name">{l.name}</span>
                <span className="rank-score">{l.role || ""}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <p className="disclaimer">
        Signals are investigative leads, not proof of guilt.
      </p>
    </aside>
  );
}
