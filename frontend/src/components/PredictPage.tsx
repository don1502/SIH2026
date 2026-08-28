import { useEffect, useRef, useState } from "react";
import GraphView from "./GraphView";
import {
  AnomalyTxn,
  MLMetrics,
  PredictResult,
  Suspect,
  getMlMetrics,
  predictSample,
  predictUpload,
} from "../api";

export default function PredictPage() {
  const [files, setFiles] = useState<File[]>([]);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<MLMetrics | null>(null);
  const [selected, setSelected] = useState<Suspect | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getMlMetrics().then(setMetrics).catch(() => {});
  }, []);

  const onFiles = (list: FileList | null) => {
    if (!list) return;
    setFiles(Array.from(list));
    setError(null);
  };

  const runUpload = () => {
    if (files.length === 0) return;
    setLoading(true);
    setError(null);
    predictUpload(files)
      .then((r) => {
        setResult(r);
        setSelected(r.suspects[0] || null);
      })
      .catch((e) => setError(e?.response?.data?.detail || "Prediction failed"))
      .finally(() => setLoading(false));
  };

  const runSample = () => {
    setLoading(true);
    setError(null);
    predictSample()
      .then((r) => {
        setResult(r);
        setSelected(r.suspects[0] || null);
        setFiles([]);
      })
      .catch((e) => setError(e?.response?.data?.detail || "Sample failed"))
      .finally(() => setLoading(false));
  };

  return (
    <div className="predict-layout">
      <div className="predict-left">
        <div className="panel">
          <h3>Upload case files</h3>
          <p className="hint">
            Drop the CSV files for a case (persons, phones, accounts, call_records,
            transactions, and their relations). A persons file is required.
          </p>
          <div
            className="dropzone"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              onFiles(e.dataTransfer.files);
            }}
            onClick={() => inputRef.current?.click()}
          >
            <input
              ref={inputRef}
              type="file"
              multiple
              accept=".csv"
              hidden
              onChange={(e) => onFiles(e.target.files)}
            />
            {files.length === 0 ? (
              <span>Drag & drop CSV files here, or click to browse</span>
            ) : (
              <div className="file-chips">
                {files.map((f) => (
                  <span key={f.name} className="chip">{f.name}</span>
                ))}
              </div>
            )}
          </div>
          <button className="btn primary" onClick={runUpload} disabled={loading || files.length === 0}>
            {loading ? "Analyzing…" : "Build graph & predict suspects"}
          </button>
          <button className="btn" onClick={runSample} disabled={loading}>
            Use a sample case from the dataset
          </button>
          {error && <p className="error-text">{error}</p>}
          {result?.ingestion && (
            <div className="ingest-report">
              <span className="ok">{Object.keys(result.ingestion.recognized).length} files recognized</span>
              {result.ingestion.unrecognized.length > 0 && (
                <span className="warn">{result.ingestion.unrecognized.length} skipped</span>
              )}
            </div>
          )}
        </div>

        {metrics && (
          <div className="panel">
            <h3>Model</h3>
            <p className="model-label">{metrics.label}</p>
            <div className="metric-grid">
              <Metric label="ROC-AUC" value={metrics.test.roc_auc} highlight />
              <Metric label="PR-AUC" value={metrics.test.pr_auc} highlight />
              <Metric label="F1" value={metrics.test.f1} />
              <Metric label="Recall" value={metrics.test.recall} />
            </div>
            <h4>Top predictive features</h4>
            <div className="feat-list">
              {metrics.feature_importances.slice(0, 6).map((f) => (
                <div key={f.feature} className="feat-row">
                  <span className="feat-name">{f.feature}</span>
                  <div className="feat-bar">
                    <div style={{ width: `${(f.importance / metrics.feature_importances[0].importance) * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {result && (
          <div className="panel">
            <h3>Predicted suspects</h3>
            <div className="stat-row">
              <span>{result.summary.persons_scored} scored</span>
              <span>{result.summary.flagged_suspects} flagged</span>
            </div>
            <div className="suspect-list">
              {result.suspects.map((s, i) => (
                <div
                  key={s.person_id}
                  className={`suspect-item ${selected?.person_id === s.person_id ? "active" : ""}`}
                  onClick={() => setSelected(s)}
                >
                  <span className="suspect-rank">{i + 1}</span>
                  <div className="suspect-main">
                    <span className="suspect-name">{s.name}</span>
                    <span className="suspect-role">{s.role}</span>
                  </div>
                  <span className={`prob ${s.is_suspect ? "hi" : ""}`}>
                    {(s.suspect_probability * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="predict-center">
        <div className="stage-header">
          <span className="crumb">
            {result ? "Reconstructed case network" : "Upload files or load a sample to begin"}
          </span>
          <div className="legend">
            <Legend color="#ff4d4d" label="Predicted suspect" />
            <Legend color="#4f9dff" label="Person" />
            <Legend color="#9b59b6" label="Phone" />
            <Legend color="#e74c3c" label="Account" />
          </div>
        </div>
        <GraphView data={result?.graph || null} onSelectNode={() => {}} showSuspects />
      </div>

      <div className="predict-right">
        {selected ? (
          <div className="detail">
            <div className="entity-head">
              <span className={`type-badge ${selected.is_suspect ? "suspect" : ""}`}>
                {selected.is_suspect ? "SUSPECT" : "Person"}
              </span>
              <h2>{selected.name}</h2>
              <span className="entity-id">{selected.person_id}</span>
            </div>
            <div className="prob-gauge">
              <div className="prob-value">{(selected.suspect_probability * 100).toFixed(1)}%</div>
              <div className="prob-caption">suspect probability</div>
              <div className="prob-track">
                <div
                  className="prob-fill"
                  style={{
                    width: `${selected.suspect_probability * 100}%`,
                    background: selected.is_suspect ? "#ff4d4d" : "#4f9dff",
                  }}
                />
              </div>
            </div>
            <div className="prop-grid">
              <Field k="Age" v={selected.age || "-"} />
              <Field k="Gender" v={selected.gender || "-"} />
              <Field k="Role" v={selected.role || "-"} />
              <Field k="Risk score" v={selected.risk_score || "-"} />
              <Field k="Anomaly" v={selected.anomaly_score != null ? selected.anomaly_score.toFixed(3) : "-"} />
            </div>
            {selected.indicators.length > 0 && (
              <>
                <h3>Why flagged</h3>
                <ul className="indicator-list">
                  {selected.indicators.map((ind) => (
                    <li key={ind}>{ind}</li>
                  ))}
                </ul>
              </>
            )}
            {result?.anomalies && result.anomalies.transactions.length > 0 && (
              <>
                <h3>Flagged money flows</h3>
                <div className="mini-anomaly">
                  {result.anomalies.transactions.slice(0, 5).map((t: AnomalyTxn) => (
                    <div key={t.transaction_id} className="mini-anomaly-row">
                      <span>{t.transaction_id}</span>
                      <span>₹{t.amount.toLocaleString()}</span>
                    </div>
                  ))}
                </div>
              </>
            )}
            <p className="disclaimer">
              Model output is an investigative lead, not proof of guilt.
            </p>
          </div>
        ) : (
          <div className="detail empty">
            <p>Predicted suspects and their supporting indicators will appear here.</p>
          </div>
        )}
      </div>
    </div>
  );
}

function Metric({ label, value, highlight }: { label: string; value: number; highlight?: boolean }) {
  return (
    <div className={`metric ${highlight ? "metric-hi" : ""}`}>
      <span className="metric-val">{value?.toFixed?.(2) ?? value}</span>
      <span className="metric-label">{label}</span>
    </div>
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
