import { useEffect, useState } from "react";
import GraphView from "./components/GraphView";
import Sidebar from "./components/Sidebar";
import EntityPanel from "./components/EntityPanel";
import PredictPage from "./components/PredictPage";
import { EntityProfile, Subgraph, getProfile, getSubgraph } from "./api";

type Tab = "explore" | "predict";

export default function App() {
  const [tab, setTab] = useState<Tab>("predict");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [subgraph, setSubgraph] = useState<Subgraph | null>(null);
  const [profile, setProfile] = useState<EntityProfile | null>(null);
  const [edge, setEdge] = useState<Record<string, unknown> | null>(null);

  const selectEntity = (id: string, hops = 1) => {
    setSelectedId(id);
    setEdge(null);
    getProfile(id).then(setProfile).catch(() => setProfile(null));
    getSubgraph(id, hops).then(setSubgraph).catch(() => setSubgraph(null));
  };

  useEffect(() => {
    if (tab === "explore" && !selectedId) {
      getSubgraph("PER_0001").then(setSubgraph).catch(() => {});
      getProfile("PER_0001").then(setProfile).catch(() => {});
      setSelectedId("PER_0001");
    }
  }, [tab]);

  return (
    <div className="root">
      <header className="topbar">
        <div className="brand">
          <h1>CrimeGraph<span>AI</span></h1>
          <span className="subtitle">Criminal Intelligence · Suspect Prediction</span>
        </div>
        <nav className="tabs">
          <button className={tab === "predict" ? "tab active" : "tab"} onClick={() => setTab("predict")}>
            Predict Suspects
          </button>
          <button className={tab === "explore" ? "tab active" : "tab"} onClick={() => setTab("explore")}>
            Explore Graph
          </button>
        </nav>
      </header>

      {tab === "predict" ? (
        <PredictPage />
      ) : (
        <div className="app">
          <Sidebar onSelect={(id) => selectEntity(id)} />
          <main className="stage">
            <div className="stage-header">
              <span className="crumb">{selectedId ? `Exploring ${selectedId}` : "Select an entity"}</span>
              <div className="legend">
                <Legend color="#4f9dff" label="Person" />
                <Legend color="#f6a623" label="Org" />
                <Legend color="#9b59b6" label="Phone" />
                <Legend color="#e74c3c" label="Account" />
                <Legend color="#95a5a6" label="Case" />
              </div>
            </div>
            <GraphView
              data={subgraph}
              onSelectNode={(id) => selectEntity(id)}
              onSelectEdge={(e) => setEdge(e)}
            />
          </main>
          <EntityPanel
            profile={profile}
            edge={edge}
            onSelect={(id) => selectEntity(id)}
            onExpand={(id) => selectEntity(id, 2)}
          />
        </div>
      )}
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
