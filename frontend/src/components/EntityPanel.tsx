import { EntityProfile } from "../api";

interface Props {
  profile: EntityProfile | null;
  edge: Record<string, unknown> | null;
  onSelect: (id: string) => void;
  onExpand: (id: string) => void;
}

export default function EntityPanel({ profile, edge, onSelect, onExpand }: Props) {
  if (edge) {
    const skip = new Set(["id", "source", "target", "label"]);
    return (
      <div className="detail">
        <h2>Relationship</h2>
        <p className="rel-context">{String(edge.label)}</p>
        <div className="evidence-card">
          <Field k="From" v={String(edge.source)} />
          <Field k="To" v={String(edge.target)} />
          {Object.entries(edge)
            .filter(([k, v]) => !skip.has(k) && v !== null && v !== "")
            .map(([k, v]) => (
              <Field key={k} k={k} v={String(v)} />
            ))}
        </div>
        <p className="hint">Relationships are derived from the case data and activity records.</p>
      </div>
    );
  }

  if (!profile) {
    return (
      <div className="detail empty">
        <p>Select an entity to view its profile and connections.</p>
      </div>
    );
  }

  const p = profile.properties as Record<string, string | number>;
  const label = profile.labels.filter((l) => l !== "Entity")[0] || "Entity";

  return (
    <div className="detail">
      <div className="entity-head">
        <span className={`type-badge ${label}`}>{label}</span>
        <h2>{(p.name as string) || profile.id}</h2>
        <span className="entity-id">{profile.id}</span>
      </div>

      <div className="prop-grid">
        {p.role && <Field k="Role" v={String(p.role)} />}
        {p.risk_score !== undefined && <Field k="Risk score" v={String(p.risk_score)} />}
        {p.age && <Field k="Age" v={String(p.age)} />}
        {p.gender && <Field k="Gender" v={String(p.gender)} />}
        {p.type && label !== "Person" && <Field k="Type" v={String(p.type)} />}
        {p.threat_level && <Field k="Threat" v={String(p.threat_level)} badge />}
        {p.crime_type && <Field k="Crime" v={String(p.crime_type)} />}
        {p.status && <Field k="Status" v={String(p.status)} badge />}
        {p.pagerank !== undefined && <Field k="PageRank" v={Number(p.pagerank).toFixed(4)} />}
      </div>

      <button className="btn" onClick={() => onExpand(profile.id)}>Expand 2 hops</button>

      <h3>Connections</h3>
      <div className="chips">
        {profile.relationship_counts.map((rc) => (
          <span key={rc.rel_type} className="chip">{rc.rel_type} · {rc.count}</span>
        ))}
      </div>

      <h3>Neighbors</h3>
      <div className="neighbor-list">
        {profile.neighbors.slice(0, 60).map((n, i) => (
          <div key={i} className="neighbor-item">
            <span className="dir">{n.outgoing ? "→" : "←"}</span>
            <span className="rel">{n.rel_type}</span>
            <span className="nbr" onClick={() => onSelect(n.neighbor_id)}>
              {n.neighbor_name || n.neighbor_id}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function Field({ k, v, badge }: { k: string; v: string | number; badge?: boolean }) {
  return (
    <div className="field">
      <span className="field-k">{k}</span>
      {badge ? (
        <span className={`status-badge s-${String(v)}`}>{v}</span>
      ) : (
        <span className="field-v">{v}</span>
      )}
    </div>
  );
}
