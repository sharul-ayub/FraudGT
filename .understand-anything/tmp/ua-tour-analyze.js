const fs = require('fs');

function main() {
  const [inputPath, outputPath] = process.argv.slice(2);
  if (!inputPath || !outputPath) {
    console.error('Usage: node ua-tour-analyze.js <input.json> <output.json>');
    process.exit(1);
  }

  const graph = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
  const nodes = Array.isArray(graph.nodes) ? graph.nodes : [];
  const edges = Array.isArray(graph.edges) ? graph.edges : [];
  const layers = Array.isArray(graph.layers) ? graph.layers : [];
  const nodeById = new Map(nodes.map((node) => [node.id, node]));

  const fanIn = new Map(nodes.map((node) => [node.id, 0]));
  const fanOut = new Map(nodes.map((node) => [node.id, 0]));
  const adjacency = new Map(nodes.map((node) => [node.id, []]));

  for (const edge of edges) {
    if (!nodeById.has(edge.source) || !nodeById.has(edge.target)) continue;
    fanOut.set(edge.source, (fanOut.get(edge.source) || 0) + 1);
    fanIn.set(edge.target, (fanIn.get(edge.target) || 0) + 1);
    if (edge.type === 'imports' || edge.type === 'calls') {
      adjacency.get(edge.source).push(edge.target);
    }
  }

  const rank = (map, label) => [...map.entries()]
    .map(([id, value]) => ({ id, [label]: value, name: nodeById.get(id)?.name || id }))
    .sort((a, b) => b[label] - a[label] || a.id.localeCompare(b.id))
    .slice(0, 20);

  const fanInRanking = rank(fanIn, 'fanIn');
  const fanOutRanking = rank(fanOut, 'fanOut');
  const topFanOut = new Set(fanOutRanking.slice(0, Math.max(1, Math.ceil(nodes.length * 0.1))).map((item) => item.id));
  const sortedFanIn = [...fanIn.entries()].sort((a, b) => a[1] - b[1]);
  const lowFanIn = new Set(sortedFanIn.slice(0, Math.max(1, Math.ceil(nodes.length * 0.25))).map((item) => item[0]));
  const entryName = /^(index|main|app|server|manage|run|__main__)\.(ts|js|py|rs|go|c|cpp)$/i;
  const specificEntry = /^(wsgi|asgi)\.py$|^Application\.(java|kt)$|^Main\.java$|^Program\.cs$|^config\.ru$|^index\.php$|^App\.swift$/i;

  const entryPointCandidates = nodes
    .map((node) => {
      let score = 0;
      const path = node.filePath || '';
      const parts = path.split(/[\\/]/).filter(Boolean);
      if (node.type === 'file') {
        if (entryName.test(node.name || '') || specificEntry.test(node.name || '')) score += 3;
        if (parts.length <= 2) score += 1;
        if (topFanOut.has(node.id)) score += 1;
        if (lowFanIn.has(node.id)) score += 1;
      }
      if (node.type === 'document') {
        if (path === 'README.md') score += 5;
        else if (parts.length === 1 && /\.md$/i.test(path)) score += 2;
      }
      return { id: node.id, score, name: node.name, summary: node.summary };
    })
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score || a.id.localeCompare(b.id))
    .slice(0, 5);

  const codeStart = entryPointCandidates.find((item) => nodeById.get(item.id)?.type === 'file')?.id;
  const bfsTraversal = { startNode: codeStart || null, order: [], depthMap: {}, byDepth: {} };
  if (codeStart) {
    const queue = [codeStart];
    const seen = new Set([codeStart]);
    bfsTraversal.depthMap[codeStart] = 0;
    while (queue.length) {
      const current = queue.shift();
      const depth = bfsTraversal.depthMap[current];
      bfsTraversal.order.push(current);
      bfsTraversal.byDepth[String(depth)] ||= [];
      bfsTraversal.byDepth[String(depth)].push(current);
      for (const next of adjacency.get(current) || []) {
        if (seen.has(next)) continue;
        seen.add(next);
        bfsTraversal.depthMap[next] = depth + 1;
        queue.push(next);
      }
    }
  }

  const nonCodeFiles = { documentation: [], infrastructure: [], data: [], config: [] };
  for (const node of nodes) {
    const item = { id: node.id, name: node.name, type: node.type, summary: node.summary };
    if (node.type === 'document') nonCodeFiles.documentation.push(item);
    else if (['service', 'pipeline', 'resource'].includes(node.type)) nonCodeFiles.infrastructure.push(item);
    else if (['table', 'schema', 'endpoint'].includes(node.type)) nonCodeFiles.data.push(item);
    else if (node.type === 'config') nonCodeFiles.config.push(item);
  }

  const edgeKey = new Set(edges.map((edge) => `${edge.source}\t${edge.target}\t${edge.type}`));
  const clusters = [];
  const clustered = new Set();
  for (const edge of edges) {
    if (edge.type !== 'imports' && edge.type !== 'calls') continue;
    if (!edgeKey.has(`${edge.target}\t${edge.source}\t${edge.type}`)) continue;
    if (clustered.has(edge.source) && clustered.has(edge.target)) continue;
    const cluster = new Set([edge.source, edge.target]);
    let expanded = true;
    while (expanded && cluster.size < 5) {
      expanded = false;
      for (const candidate of nodes.map((node) => node.id)) {
        if (cluster.has(candidate)) continue;
        let links = 0;
        for (const member of cluster) {
          if (edgeKey.has(`${candidate}\t${member}\timports`) || edgeKey.has(`${member}\t${candidate}\timports`) ||
              edgeKey.has(`${candidate}\t${member}\tcalls`) || edgeKey.has(`${member}\t${candidate}\tcalls`)) {
            links += 1;
          }
        }
        if (links >= 2) {
          cluster.add(candidate);
          expanded = true;
          break;
        }
      }
    }
    for (const id of cluster) clustered.add(id);
    const clusterNodes = [...cluster];
    const edgeCount = edges.filter((candidate) => cluster.has(candidate.source) && cluster.has(candidate.target)).length;
    clusters.push({ nodes: clusterNodes, edgeCount });
  }
  clusters.sort((a, b) => b.edgeCount - a.edgeCount).splice(10);

  const nodeSummaryIndex = Object.fromEntries(nodes.map((node) => [
    node.id,
    { name: node.name, type: node.type, summary: node.summary }
  ]));

  fs.writeFileSync(outputPath, JSON.stringify({
    scriptCompleted: true,
    entryPointCandidates,
    fanInRanking,
    fanOutRanking,
    bfsTraversal,
    nonCodeFiles,
    clusters,
    layers: { count: layers.length, list: layers.map(({ id, name, description }) => ({ id, name, description })) },
    nodeSummaryIndex,
    totalNodes: nodes.length,
    totalEdges: edges.length
  }, null, 2));
}

try {
  main();
} catch (error) {
  console.error(error && error.stack ? error.stack : String(error));
  process.exit(1);
}
