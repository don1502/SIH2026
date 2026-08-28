import { useEffect, useState } from "react";
import GraphView from "./GraphView";
import {
  AnomalyCall,
  AnomalyPerson,
  AnomalyTxn,
  Subgraph,
  getAnomalyCalls,
  getAnomalyPersons,
  getAnomalyTransactions,
  getSubgraph,
} from "../api";

type Selected =
  | { kind: "person"; item: AnomalyPerson }
  | { kind: "txn"; item: AnomalyTxn }
  | { kind: "call"; item: AnomalyCall };

export default function AnomaliesPage() {
  const [txns, setTxns] = useState<AnomalyTxn[]>([]);
  const [calls, setCalls] = useState<AnomalyCall[]>([]);
  const [persons, setPersons] = useState<AnomalyPerson[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<Selected | null>(null);
  const [graph, setGraph] = useState<Subgraph | null>(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([getAnomalyTransactions(20), getAnomalyCalls(20), getAnomalyPersons(20)])
      .then(([t, c, p]) => {
        setTxns(t);
        setCalls(c);
        setPersons(p);
        if (p[0]) {
          setSelected({ kind: "person", item: p[0] });
          getSubgraph(p[0].person_id, 1).then(setGraph).catch(() => setGraph(null));
        }
      })
      .catch((e) => setError(e?.response?.data?.detail || "Failed to load anomalies"))
      .finally(() => setLoading(false));
  }, []);

  const pickPerson = (p: AnomalyPerson) => {
    setSelected({ kind: "person", item: p });
    getSubgraph(p.person_id, 1).then(setGraph).catch(() => setGraph(null));
  };

  return (
    <div className="anomaly-layout">
      <div className="anomaly-col">
        <div className="panel">
          <h3>Anomalous persons</h3>
          <p className="hint">Highest scores from owned accounts and phones.</p>
          {loading && <p className="hint">Scoring…</p>}
          {error && <p className="error-text">{error}</p>}
          <div className="suspect-list">
            {persons.map((p, i) => (
              <div
                key={p.person_id}
                className={`suspect-item ${selected?.kind === "person" && selected.item.person_id === p.person_id ? "active" : ""}`}
                onClick={() => pickPerson(p)}
              >
                <span className="suspect-rank">{i + 1}</span>
                <div className="suspect-main">
                  <span className="suspect-name">{p.name}</span>
                  <span className="suspect-role">{p.role}</span>
                </div>
                <span className="prob hi">{(p.anomaly_score * 100).toFixed(0)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="anomaly-center">
        <div className="stage-header">
          <span className="crumb">Suspicious money flows and call bursts</span>
          <div className="legend">
            <Legend color="#ff8c42" label="Anomalous person" />
            <Legend color="#e74c3c" label="Transaction" />
            <Legend color="#9b59b6" label="Call burst" />
          </div>
        </div>
        <div className="anomaly-lists">
          <div className="panel">
            <h3>Money flows</h3>
            {txns.map((t) => (
              <div
                key={t.transaction_id}
                className={`anomaly-row ${selected?.kind === "txn" && selected.item.transaction_id === t.transaction_id ? "active" : ""}`}
                onClick={() => setSelected({ kind: "txn", item: t })}
              >
                <div className="anomaly-row-main">
                  <span className="anomaly-id">{t.transaction_id}</span>
                  <span className="anomaly-meta">
                    {t.sender_owner || t.sender_account} → {t.receiver_owner || t.receiver_account}
                  </span>
                  <div className="chips">
                    {t.reasons.slice(0, 2).map((r) => (
                      <span key={r} className="chip">{r}</span>
                    ))}
                  </div>
                </div>
                <div className="anomaly-row-side">
                  <span className="anomaly-amt">₹{t.amount.toLocaleString()}</span>
                  <span className="prob hi">{(t.anomaly_score * 100).toFixed(0)}</span>
                </div>
              </div>
            ))}
          </div>
          <div className="panel">
            <h3>Call bursts</h3>
            {calls.map((c) => (
              <div
                key={c.phone_id}
                className={`anomaly-row ${selected?.kind === "call" && selected.item.phone_id === c.phone_id ? "active" : ""}`}
                onClick={() => {
                  setSelected({ kind: "call", item: c });
                  getSubgraph(c.phone_id, 1).then(setGraph).catch(() => setGraph(null));
                }}
              >
                <div className="anomaly-row-main">
                  <span className="anomaly-id">{c.owner || c.phone_number}</span>
                  <span className="anomaly-meta">
                    {c.call_count} calls · {c.max_calls_in_hour} in one hour · {c.distinct_contacts} contacts
                  </span>
                  <div className="chips">
                    {c.reasons.slice(0, 2).map((r) => (
                      <span key={r} className="chip">{r}</span>
                    ))}
                  </div>
                </div>
                <span className="prob hi">{(c.anomaly_score * 100).toFixed(0)}</span>
              </div>
            ))}
          </div>
        </div>
        <GraphView data={graph} onSelectNode={() => {}} />
      </div>

      <div className="anomaly-detail">
        {selected?.kind === "person" && <PersonDetail p={selected.item} />}
        {selected?.kind === "txn" && <TxnDetail t={selected.item} />}
        {selected?.kind === "call" && <CallDetail c={selected.item} />}
        {!selected && (
          <div className="detail empty">
            <p>Select a person, transfer, or phone to inspect why it was flagged.</p>
          </div>
        )}
      </div>
    </div>
  );
}

function PersonDetail({ p }: { p: AnomalyPerson }) {
  return (
    <div className="detail">
      <div className="entity-head">
        <span className="type-badge anomalous">ANOMALY</span>
        <h2>{p.name}</h2>
        <span className="entity-id">{p.person_id}</span>
      </div>
      <ScoreGauge value={p.anomaly_score} caption="person anomaly score" />
      <div className="prop-grid">
        <Field k="Role" v={p.role || "-"} />
        <Field k="Risk score" v={p.risk_score || "-"} />
      </div>
      <Reasons items={p.reasons} />
      <p className="disclaimer">Unsupervised IsolationForest score — investigative lead, not proof.</p>
    </div>
  );
}

function TxnDetail({ t }: { t: AnomalyTxn }) {
  return (
    <div className="detail">
      <div className="entity-head">
        <span className="type-badge anomalous">MONEY FLOW</span>
        <h2>{t.transaction_id}</h2>
        <span className="entity-id">{t.transaction_type}</span>
      </div>
      <ScoreGauge value={t.anomaly_score} caption="transaction anomaly score" />
      <div className="prop-grid">
        <Field k="Amount" v={`₹${t.amount.toLocaleString()}`} />
        <Field k="When" v={t.timestamp || "-"} />
        <Field k="From" v={t.sender_owner || t.sender_account} />
        <Field k="To" v={t.receiver_owner || t.receiver_account} />
      </div>
      <Reasons items={t.reasons} />
    </div>
  );
}

function CallDetail({ c }: { c: AnomalyCall }) {
  return (
    <div className="detail">
      <div className="entity-head">
        <span className="type-badge anomalous">CALL BURST</span>
        <h2>{c.owner || c.phone_number}</h2>
        <span className="entity-id">{c.phone_id}</span>
      </div>
      <ScoreGauge value={c.anomaly_score} caption="phone anomaly score" />
      <div className="prop-grid">
        <Field k="Number" v={c.phone_number} />
        <Field k="Calls" v={String(c.call_count)} />
        <Field k="Peak / hour" v={String(c.max_calls_in_hour)} />
        <Field k="Contacts" v={String(c.distinct_contacts)} />
        <Field k="Duration (s)" v={String(c.total_duration)} />
      </div>
      <Reasons items={c.reasons} />
    </div>
  );
}

function ScoreGauge({ value, caption }: { value: number; caption: string }) {
  return (
    <div className="prob-gauge">
      <div className="prob-value">{(value * 100).toFixed(1)}</div>
      <div className="prob-caption">{caption}</div>
      <div className="prob-track">
        <div className="prob-fill" style={{ width: `${value * 100}%`, background: "#ff8c42" }} />
      </div>
    </div>
  );
}

function Reasons({ items }: { items: string[] }) {
  if (!items.length) return null;
  return (
    <>
      <h3>Why flagged</h3>
      <ul className="indicator-list">
        {items.map((r) => (
          <li key={r}>{r}</li>
        ))}
      </ul>
    </>
  );
}

function Field({ k, v }: { k: string; v: string }) {
  return (
    <div className="field">
      <span className="field-k">{k}</span>
      <span className="field-v">{v}</span>
    </div>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="legend-item">
      <span className="legend-dot" style={{ background: color }} />
      {label}
    </span>
  );
}
