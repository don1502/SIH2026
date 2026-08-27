import { useEffect, useRef } from "react";
import cytoscape, { Core, ElementDefinition } from "cytoscape";
import { Subgraph } from "../api";

const TYPE_COLORS: Record<string, string> = {
  Person: "#4f9dff",
  Organization: "#f6a623",
  Location: "#2ecc71",
  Phone: "#9b59b6",
  Account: "#e74c3c",
  Vehicle: "#e67e22",
  Case: "#95a5a6",
  Entity: "#7f8c8d",
};

interface Props {
  data: Subgraph | null;
  onSelectNode: (id: string) => void;
  onSelectEdge?: (edge: Record<string, unknown>) => void;
  showSuspects?: boolean;
}

export default function GraphView({ data, onSelectNode, onSelectEdge, showSuspects }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const cy = cytoscape({
      container: containerRef.current,
      style: [
        {
          selector: "node",
          style: {
            "background-color": (ele: any) => TYPE_COLORS[ele.data("type")] || TYPE_COLORS.Entity,
            label: "data(label)",
            color: "#e8eef7",
            "font-size": "9px",
            "text-valign": "bottom",
            "text-halign": "center",
            "text-margin-y": 3,
            width: 22,
            height: 22,
            "text-outline-color": "#0d1117",
            "text-outline-width": 2,
          },
        },
        {
          selector: "node[type = 'Person']",
          style: {
            width: (ele: any) =>
              showSuspects
                ? 20 + (ele.data("suspect_probability") || 0) * 30
                : 20 + Math.min((ele.data("pagerank") || 0) * 3000, 30),
            height: (ele: any) =>
              showSuspects
                ? 20 + (ele.data("suspect_probability") || 0) * 30
                : 20 + Math.min((ele.data("pagerank") || 0) * 3000, 30),
          },
        },
        {
          selector: "node[?is_center]",
          style: { "border-width": 4, "border-color": "#ffd23f" },
        },
        {
          selector: "node[?is_suspect]",
          style: {
            "background-color": "#ff4d4d",
            "border-width": 3,
            "border-color": "#ff9999",
          },
        },
        {
          selector: "edge",
          style: {
            width: 1.4,
            "line-color": "#3a4a63",
            "target-arrow-color": "#3a4a63",
            "target-arrow-shape": "triangle",
            "arrow-scale": 0.7,
            "curve-style": "bezier",
            label: "data(label)",
            "font-size": "7px",
            color: "#6d7a8f",
            "text-rotation": "autorotate",
          },
        },
        { selector: "edge[label = 'CALLED']", style: { "line-color": "#9b59b6", "target-arrow-color": "#9b59b6" } },
        { selector: "edge[label = 'TRANSACTED_WITH']", style: { "line-color": "#e74c3c", "target-arrow-color": "#e74c3c" } },
        { selector: "edge[label = 'PARTICIPATED_IN']", style: { "line-color": "#95a5a6", "target-arrow-color": "#95a5a6" } },
        { selector: ":selected", style: { "border-width": 4, "border-color": "#ffffff", "line-color": "#ffffff" } },
      ],
      layout: { name: "cose", animate: false },
      wheelSensitivity: 0.2,
    });

    cy.on("tap", "node", (evt) => onSelectNode(evt.target.data("id")));
    cy.on("tap", "edge", (evt) => onSelectEdge?.(evt.target.data()));

    cyRef.current = cy;
    return () => cy.destroy();
  }, []);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !data) return;
    const elements: ElementDefinition[] = [...data.nodes, ...data.edges];
    cy.elements().remove();
    cy.add(elements);
    cy.layout({ name: "cose", animate: false, nodeRepulsion: () => 9000 } as any).run();
    cy.fit(undefined, 40);
  }, [data]);

  return <div ref={containerRef} className="graph-canvas" />;
}
