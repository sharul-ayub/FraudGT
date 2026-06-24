#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

function writeJson(file, data) {
  fs.writeFileSync(file, JSON.stringify(data, null, 2));
}

function commonPrefixSegments(paths) {
  if (!paths.length) return [];
  const split = paths.map(p => p.split(/[\\/]+/).filter(Boolean));
  const prefix = [];
  for (let i = 0; i < split[0].length; i += 1) {
    const segment = split[0][i];
    if (split.every(parts => parts[i] === segment)) prefix.push(segment);
    else break;
  }
  return prefix;
}

function extensionGroup(filePath) {
  const base = path.posix.basename(filePath);
  if (/(__init__\.py|index\.[jt]sx?)$/.test(base)) return 'entry';
  if (/\.(test|spec)\./.test(base) || /^test_/.test(base)) return 'test';
  if (/\.(ya?ml|json|toml|ini|cfg)$/.test(base)) return 'config';
  if (/\.(md|rst|txt)$/.test(base)) return 'documentation';
  if (/\.sh$/.test(base)) return 'scripts';
  if (/\.py$/.test(base)) return 'python';
  return 'root';
}

function groupFor(filePath, prefix) {
  const parts = filePath.split(/[\\/]+/).filter(Boolean);
  const rest = parts.slice(prefix.length);
  if (rest.length > 1) return rest[0];
  return 'root';
}

function classifyPattern(group, ids, byId) {
  const lower = group.toLowerCase();
  const groupPatterns = [
    [/^(routes|api|controllers|endpoints|handlers|serializers|routers|blueprints)$/, 'api'],
    [/^(services|core|lib|domain|logic|internal|signals|jobs|channels|mailers|composables)$/, 'service'],
    [/^(models|db|data|persistence|repository|entities|migrations|database|schema|datasets?)$/, 'data'],
    [/^(components|views|pages|ui|layouts|screens)$/, 'ui'],
    [/^(middleware|plugins|interceptors|guards)$/, 'middleware'],
    [/^(utils|helpers|common|shared|tools|pkg)$/, 'utility'],
    [/^(config|configs|constants|env|settings|management|commands)$/, 'config'],
    [/^(__tests__|test|tests|spec|specs)$/, 'test'],
    [/^(types|interfaces|schemas|contracts|dtos|dto|request|response)$/, 'types'],
    [/^(assets|static|public)$/, 'assets'],
    [/^(docs|documentation|wiki)$/, 'documentation'],
    [/^(\.github|\.gitlab|\.circleci)$/, 'ci-cd'],
    [/^(deploy|deployment|infra|infrastructure|k8s|kubernetes|helm|charts|terraform|tf|docker)$/, 'infrastructure'],
    [/^(bin|run|scripts)$/, 'entry']
  ];
  for (const [regex, label] of groupPatterns) {
    if (regex.test(lower)) return label;
  }
  const labels = ids.map(id => {
    const n = byId.get(id);
    const fp = n.filePath;
    const base = path.posix.basename(fp);
    if (/\.(test|spec)\./.test(base) || /^test_/.test(base)) return 'test';
    if (base === 'Dockerfile' || /^docker-compose\./.test(base) || base === 'Makefile') return 'infrastructure';
    if (/\.sh$/.test(base)) return 'entry';
    if (/\.(md|rst)$/.test(base)) return 'documentation';
    if (/\.(ya?ml|json|toml|ini|cfg)$/.test(base)) return 'config';
    if (base === '__init__.py') return 'entry';
    return null;
  }).filter(Boolean);
  if (labels.length && labels.length === ids.length) {
    const counts = labels.reduce((acc, label) => (acc[label] = (acc[label] || 0) + 1, acc), {});
    return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
  }
  return 'uncategorized';
}

function edgeNodeType(id) {
  const idx = id.indexOf(':');
  return idx === -1 ? 'unknown' : id.slice(0, idx);
}

function main() {
  const [inputPath, outputPath] = process.argv.slice(2);
  if (!inputPath || !outputPath) {
    console.error('usage: ua-arch-analyze.js <input> <output>');
    process.exit(1);
  }
  const input = readJson(inputPath);
  const fileNodes = input.fileNodes || [];
  const importEdges = input.importEdges || [];
  const allEdges = input.allEdges || [];
  const byId = new Map(fileNodes.map(n => [n.id, n]));
  const prefix = commonPrefixSegments(fileNodes.map(n => n.filePath)).filter(Boolean);
  const directoryGroups = {};
  for (const node of fileNodes) {
    const group = fileNodes.every(n => !n.filePath.includes('/')) ? extensionGroup(node.filePath) : groupFor(node.filePath, prefix);
    (directoryGroups[group] ||= []).push(node.id);
  }
  const nodeTypeGroups = {};
  for (const node of fileNodes) (nodeTypeGroups[node.type] ||= []).push(node.id);

  const fileFanIn = {};
  const fileFanOut = {};
  const interGroup = new Map();
  const intraGroupDensity = {};
  const groupById = new Map();
  for (const [group, ids] of Object.entries(directoryGroups)) ids.forEach(id => groupById.set(id, group));
  for (const node of fileNodes) {
    fileFanIn[node.id] = 0;
    fileFanOut[node.id] = 0;
  }
  for (const edge of importEdges) {
    if (!byId.has(edge.source) || !byId.has(edge.target)) continue;
    fileFanOut[edge.source] += 1;
    fileFanIn[edge.target] += 1;
    const from = groupById.get(edge.source) || 'root';
    const to = groupById.get(edge.target) || 'root';
    const key = `${from}\u0000${to}`;
    interGroup.set(key, (interGroup.get(key) || 0) + 1);
  }
  for (const group of Object.keys(directoryGroups)) {
    let internalEdges = 0;
    let totalEdges = 0;
    for (const edge of importEdges) {
      const from = groupById.get(edge.source);
      const to = groupById.get(edge.target);
      if (from === group || to === group) {
        totalEdges += 1;
        if (from === group && to === group) internalEdges += 1;
      }
    }
    intraGroupDensity[group] = {
      internalEdges,
      totalEdges,
      density: totalEdges ? Number((internalEdges / totalEdges).toFixed(3)) : 0
    };
  }

  const crossMap = new Map();
  const nonCodeConnections = [];
  for (const edge of allEdges) {
    const sourceType = edgeNodeType(edge.source);
    const targetType = edgeNodeType(edge.target);
    const key = `${sourceType}\u0000${targetType}\u0000${edge.type}`;
    crossMap.set(key, (crossMap.get(key) || 0) + 1);
    if (sourceType !== 'file' || targetType !== 'file') nonCodeConnections.push(edge);
  }
  const crossCategoryEdges = [...crossMap.entries()].map(([key, count]) => {
    const [fromType, toType, edgeType] = key.split('\u0000');
    return { fromType, toType, edgeType, count };
  }).sort((a, b) => b.count - a.count);

  const interGroupImports = [...interGroup.entries()].map(([key, count]) => {
    const [from, to] = key.split('\u0000');
    return { from, to, count };
  }).sort((a, b) => b.count - a.count);

  const patternMatches = {};
  for (const [group, ids] of Object.entries(directoryGroups)) {
    patternMatches[group] = classifyPattern(group, ids, byId);
  }

  const infraFiles = fileNodes
    .filter(n => n.type === 'service' || n.type === 'resource' || n.type === 'pipeline' || /(^|\/)(Dockerfile|docker-compose|\.github|\.gitlab|Jenkinsfile|Makefile)/.test(n.filePath))
    .map(n => n.filePath);
  const deploymentTopology = {
    hasDockerfile: fileNodes.some(n => path.posix.basename(n.filePath) === 'Dockerfile'),
    hasCompose: fileNodes.some(n => /^docker-compose\./.test(path.posix.basename(n.filePath))),
    hasK8s: fileNodes.some(n => /(^|\/)(k8s|kubernetes|helm|charts)(\/|$)/.test(n.filePath)),
    hasTerraform: fileNodes.some(n => /\.tf(vars)?$/.test(n.filePath)),
    hasCI: fileNodes.some(n => /(^|\/)(\.github\/workflows|\.gitlab-ci\.yml|Jenkinsfile)/.test(n.filePath)),
    infraFiles
  };

  const dataPipeline = {
    schemaFiles: fileNodes.filter(n => /\.(graphql|gql|proto|prisma)$/.test(n.filePath)).map(n => n.filePath),
    migrationFiles: fileNodes.filter(n => /migrations?\//.test(n.filePath) || /\.sql$/.test(n.filePath)).map(n => n.filePath),
    dataModelFiles: fileNodes.filter(n => /(datasets?|models?|data)\//.test(n.filePath)).map(n => n.filePath),
    apiHandlerFiles: fileNodes.filter(n => /(routes?|api|controllers?|handlers?)\//.test(n.filePath)).map(n => n.filePath)
  };

  const groupsWithDocs = Object.entries(directoryGroups).filter(([, ids]) => ids.some(id => /\.(md|rst)$/i.test(byId.get(id).filePath))).map(([g]) => g);
  const undocumentedGroups = Object.keys(directoryGroups).filter(g => !groupsWithDocs.includes(g));
  const docCoverage = {
    groupsWithDocs: groupsWithDocs.length,
    totalGroups: Object.keys(directoryGroups).length,
    coverageRatio: Object.keys(directoryGroups).length ? Number((groupsWithDocs.length / Object.keys(directoryGroups).length).toFixed(3)) : 0,
    undocumentedGroups
  };

  const dependencyDirection = [];
  const groupPairs = new Set();
  for (const item of interGroupImports) {
    if (item.from === item.to) continue;
    const pair = [item.from, item.to].sort().join('\u0000');
    groupPairs.add(pair);
  }
  for (const pair of groupPairs) {
    const [a, b] = pair.split('\u0000');
    const ab = interGroup.get(`${a}\u0000${b}`) || 0;
    const ba = interGroup.get(`${b}\u0000${a}`) || 0;
    if (ab > ba) dependencyDirection.push({ dependent: a, dependsOn: b, count: ab });
    else if (ba > ab) dependencyDirection.push({ dependent: b, dependsOn: a, count: ba });
  }

  const filesPerGroup = Object.fromEntries(Object.entries(directoryGroups).map(([g, ids]) => [g, ids.length]));
  const nodeTypeCounts = Object.fromEntries(Object.entries(nodeTypeGroups).map(([t, ids]) => [t, ids.length]));
  writeJson(outputPath, {
    scriptCompleted: true,
    commonPrefix: prefix.join('/'),
    directoryGroups,
    nodeTypeGroups,
    crossCategoryEdges,
    interGroupImports,
    intraGroupDensity,
    patternMatches,
    deploymentTopology,
    dataPipeline,
    docCoverage,
    dependencyDirection,
    nonCodeConnections,
    fileStats: {
      totalFileNodes: fileNodes.length,
      filesPerGroup,
      nodeTypeCounts
    },
    fileFanIn,
    fileFanOut
  });
}

try {
  main();
} catch (error) {
  console.error(error && error.stack ? error.stack : String(error));
  process.exit(1);
}
