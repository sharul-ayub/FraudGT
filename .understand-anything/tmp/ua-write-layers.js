#!/usr/bin/env node
const fs = require('fs');

const inputPath = '.understand-anything/tmp/architecture-input.json';
const outputPath = '.understand-anything/intermediate/layers.json';
const input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
const nodes = input.fileNodes || [];

const layers = [
  {
    id: 'layer:documentation-environment',
    name: 'Documentation & Environment',
    description: 'Project overview and dependency documentation that explain how to install, understand, and run FraudGT experiments.',
    nodeIds: []
  },
  {
    id: 'layer:experiment-configurations',
    name: 'Experiment Configurations',
    description: 'YAML experiment configurations for AML and Ethereum fraud-detection runs across datasets, model variants, transfer setups, optimizers, and positional encodings.',
    nodeIds: []
  },
  {
    id: 'layer:launch-entrypoints',
    name: 'Launch Entrypoints',
    description: 'Shell and Python entry points that start FraudGT runs, manual sweeps, grid jobs, and package initialization.',
    nodeIds: []
  },
  {
    id: 'layer:runtime-configuration',
    name: 'Runtime Configuration',
    description: 'FraudGT configuration extension modules that register dataset, model, optimizer, split, pretrained, positional-encoding, and Weights & Biases defaults.',
    nodeIds: []
  },
  {
    id: 'layer:data-loading-sampling',
    name: 'Data Loading & Sampling',
    description: 'Dataset preprocessors, temporal loaders, encoding generators, split builders, samplers, transfer helpers, and graph transforms for AML and Ethereum data.',
    nodeIds: []
  },
  {
    id: 'layer:model-architecture',
    name: 'Model Architecture',
    description: 'FraudGT encoders, graph transformer layers, GNN and MLP networks, activation hooks, and prediction heads used for node and edge fraud tasks.',
    nodeIds: []
  },
  {
    id: 'layer:training-evaluation-utilities',
    name: 'Training, Evaluation & Utilities',
    description: 'Training loops, optimizer and loss registrations, metric wrappers, finetuning helpers, logging, timing, aggregation, and run utility code.',
    nodeIds: []
  },
  {
    id: 'layer:graphgym-framework',
    name: 'GraphGym Framework',
    description: 'Local GraphGym-derived framework code for registries, model builders, loaders, training loops, contrib extensions, utilities, and baseline components.',
    nodeIds: []
  }
];

const byId = Object.fromEntries(layers.map(layer => [layer.id, layer]));

function assign(node) {
  const p = node.filePath;
  if (node.type === 'document') return 'layer:documentation-environment';
  if (p.startsWith('configs/')) return 'layer:experiment-configurations';
  if (p.startsWith('run/') || p === 'fraudGT/main.py' || p === 'fraudGT/__init__.py') return 'layer:launch-entrypoints';
  if (p.startsWith('fraudGT/config/')) return 'layer:runtime-configuration';
  if (
    p.startsWith('fraudGT/datasets/') ||
    p.startsWith('fraudGT/loader/') ||
    p.startsWith('fraudGT/sampler/') ||
    p.startsWith('fraudGT/transform/')
  ) return 'layer:data-loading-sampling';
  if (
    p.startsWith('fraudGT/act/') ||
    p.startsWith('fraudGT/encoder/') ||
    p.startsWith('fraudGT/head/') ||
    p.startsWith('fraudGT/layer/') ||
    p.startsWith('fraudGT/network/')
  ) return 'layer:model-architecture';
  if (p.startsWith('fraudGT/graphgym/')) return 'layer:graphgym-framework';
  if (
    p === 'fraudGT/agg_runs.py' ||
    p === 'fraudGT/finetuning.py' ||
    p === 'fraudGT/logger.py' ||
    p.startsWith('fraudGT/loss/') ||
    p === 'fraudGT/metric_wrapper.py' ||
    p === 'fraudGT/metrics_ogb.py' ||
    p.startsWith('fraudGT/optimizer/') ||
    p === 'fraudGT/timer.py' ||
    p.startsWith('fraudGT/train/') ||
    p === 'fraudGT/utils.py'
  ) return 'layer:training-evaluation-utilities';
  throw new Error(`No layer rule for ${node.id} (${p})`);
}

for (const node of nodes) {
  byId[assign(node)].nodeIds.push(node.id);
}

for (const layer of layers) layer.nodeIds.sort();

const seen = new Map();
for (const layer of layers) {
  for (const id of layer.nodeIds) {
    if (seen.has(id)) throw new Error(`${id} assigned to both ${seen.get(id)} and ${layer.id}`);
    seen.set(id, layer.id);
  }
}
const missing = nodes.map(n => n.id).filter(id => !seen.has(id));
if (missing.length) throw new Error(`Missing ${missing.length} node assignments: ${missing.slice(0, 10).join(', ')}`);
const known = new Set(nodes.map(n => n.id));
const invented = [...seen.keys()].filter(id => !known.has(id));
if (invented.length) throw new Error(`Invented ${invented.length} node IDs: ${invented.slice(0, 10).join(', ')}`);
if (seen.size !== nodes.length) throw new Error(`Assigned ${seen.size}, expected ${nodes.length}`);
if (layers.some(layer => layer.nodeIds.length === 0)) throw new Error('Empty layer created');

fs.mkdirSync('.understand-anything/intermediate', { recursive: true });
fs.writeFileSync(outputPath, JSON.stringify(layers, null, 2));
console.log(JSON.stringify(layers.map(layer => ({ id: layer.id, name: layer.name, count: layer.nodeIds.length })), null, 2));
