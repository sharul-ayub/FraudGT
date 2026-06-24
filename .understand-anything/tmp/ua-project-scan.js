#!/usr/bin/env node
const fs = require("fs");
const path = require("path");
const cp = require("child_process");

const DEFAULT_IGNORE_PATTERNS = [
  "node_modules/",
  ".git/",
  "vendor/",
  "venv/",
  ".venv/",
  "__pycache__/",
  "dist/",
  "build/",
  "out/",
  "coverage/",
  ".next/",
  ".cache/",
  ".turbo/",
  "target/",
  "obj/",
  "*.lock",
  "package-lock.json",
  "yarn.lock",
  "pnpm-lock.yaml",
  "*.png",
  "*.jpg",
  "*.jpeg",
  "*.gif",
  "*.svg",
  "*.ico",
  "*.woff",
  "*.woff2",
  "*.ttf",
  "*.eot",
  "*.mp3",
  "*.mp4",
  "*.pdf",
  "*.zip",
  "*.tar",
  "*.gz",
  "*.min.js",
  "*.min.css",
  "*.map",
  "*.generated.*",
  ".idea/",
  ".vscode/",
  "LICENSE",
  ".gitignore",
  ".editorconfig",
  ".prettierrc",
  ".eslintrc*",
  "*.log",
];

const EXCLUDE_SCAN_ARTIFACTS = [".understand-anything/", ".agents/"];

function normalizeRel(p) {
  return p.replace(/\\/g, "/").replace(/^\/+/, "");
}

function isUnderToolingDir(rel) {
  const normalized = normalizeRel(rel);
  return EXCLUDE_SCAN_ARTIFACTS.some((prefix) => normalized === prefix.slice(0, -1) || normalized.startsWith(prefix));
}

function hasSegment(rel, segment) {
  return normalizeRel(rel).split("/").includes(segment);
}

function basename(rel) {
  return path.posix.basename(normalizeRel(rel));
}

function wildcardToRegExp(pattern) {
  const escaped = pattern.replace(/[.+^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*").replace(/\?/g, ".");
  return new RegExp(`^${escaped}$`);
}

function baselineIgnored(rel) {
  const n = normalizeRel(rel);
  const base = basename(n);
  if (isUnderToolingDir(n)) return true;
  for (const dir of ["node_modules", ".git", "vendor", "venv", ".venv", "__pycache__", "dist", "build", "out", "coverage", ".next", ".cache", ".turbo", "target", "obj", ".idea", ".vscode"]) {
    if (hasSegment(n, dir)) return true;
  }
  if (base === "LICENSE" || base === ".gitignore" || base === ".editorconfig" || base === ".prettierrc") return true;
  if (/^\.eslintrc/.test(base)) return true;
  const ext = path.posix.extname(base).toLowerCase();
  if ([".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".woff", ".woff2", ".ttf", ".eot", ".mp3", ".mp4", ".pdf", ".zip", ".tar", ".gz", ".lock", ".log", ".map"].includes(ext)) return true;
  if (base === "package-lock.json" || base === "yarn.lock" || base === "pnpm-lock.yaml") return true;
  if (/\.min\.(js|css)$/i.test(base)) return true;
  if (/\.generated\./i.test(base)) return true;
  return false;
}

function userIgnoreFilesExist(root) {
  return fs.existsSync(path.join(root, ".understand-anything", ".understandignore")) || fs.existsSync(path.join(root, ".understandignore"));
}

async function createCoreIgnoreFilter(root) {
  const candidates = [
    "@understand-anything/core",
    pathToFileUrl(path.join("C:", "Users", "hchun", ".understand-anything", "repo", "understand-anything-plugin", "packages", "core", "dist", "index.js")),
  ];
  for (const candidate of candidates) {
    try {
      const mod = await import(candidate);
      if (typeof mod.createIgnoreFilter === "function") return mod.createIgnoreFilter(root);
    } catch (_) {
      // Try the next candidate.
    }
  }
  return null;
}

function pathToFileUrl(filePath) {
  let resolved = path.resolve(filePath).replace(/\\/g, "/");
  if (!resolved.startsWith("/")) resolved = `/${resolved}`;
  return `file://${resolved.replace(/#/g, "%23").replace(/\?/g, "%3F")}`;
}

function loadSimpleUserPatterns(root) {
  const files = [path.join(root, ".understand-anything", ".understandignore"), path.join(root, ".understandignore")];
  const patterns = [];
  for (const file of files) {
    if (!fs.existsSync(file)) continue;
    for (const raw of fs.readFileSync(file, "utf8").split(/\r?\n/)) {
      const line = raw.trim();
      if (!line || line.startsWith("#") || line.startsWith("!")) continue;
      patterns.push(line);
    }
  }
  return patterns;
}

function simplePatternIgnored(rel, patterns) {
  const n = normalizeRel(rel);
  const base = basename(n);
  return patterns.some((pattern) => {
    const p = normalizeRel(pattern);
    if (!p) return false;
    if (p.endsWith("/")) return n.startsWith(p) || n.includes(`/${p}`);
    if (!p.includes("/")) return wildcardToRegExp(p).test(base);
    return wildcardToRegExp(p).test(n);
  });
}

function discoverWithGit(root) {
  try {
    const out = cp.execFileSync("git", ["ls-files"], { cwd: root, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] });
    return out.split(/\r?\n/).map(normalizeRel).filter(Boolean);
  } catch (_) {
    return [];
  }
}

function discoverRecursive(root) {
  const results = [];
  function walk(dir) {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const abs = path.join(dir, entry.name);
      const rel = normalizeRel(path.relative(root, abs));
      if (entry.isDirectory()) {
        if (!isUnderToolingDir(rel) && ![".git", "node_modules", "vendor", "venv", ".venv", "__pycache__"].includes(entry.name)) walk(abs);
      } else if (entry.isFile()) {
        results.push(rel);
      }
    }
  }
  walk(root);
  return results;
}

function languageFor(rel) {
  const n = normalizeRel(rel);
  const base = basename(n);
  const ext = path.posix.extname(base).toLowerCase();
  if (base === "Dockerfile") return "dockerfile";
  if (base === "Makefile") return "makefile";
  if (base === "Jenkinsfile") return "jenkinsfile";
  const map = {
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".py": "python",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".h": "cpp",
    ".hpp": "cpp",
    ".c": "c",
    ".cs": "csharp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".php": "php",
    ".vue": "vue",
    ".svelte": "svelte",
    ".sh": "shell",
    ".bash": "shell",
    ".md": "markdown",
    ".rst": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".sql": "sql",
    ".graphql": "graphql",
    ".gql": "graphql",
    ".proto": "protobuf",
    ".tf": "terraform",
    ".tfvars": "terraform",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "css",
    ".sass": "css",
    ".less": "css",
    ".xml": "xml",
    ".cfg": "config",
    ".ini": "config",
    ".env": "config",
  };
  if (base === ".env" || base === ".env.example") return "config";
  return map[ext] || "unknown";
}

function categoryFor(rel) {
  const n = normalizeRel(rel);
  const base = basename(n);
  const ext = path.posix.extname(base).toLowerCase();
  if ([".md", ".rst", ".txt"].includes(ext) && base !== "LICENSE") return "docs";
  if (base === "Dockerfile" || base === "Makefile" || base === "Jenkinsfile" || base === "Procfile" || base === "Vagrantfile" || base.startsWith("docker-compose.") || n.startsWith(".github/workflows/") || base === ".gitlab-ci.yml" || n.startsWith(".circleci/") || ext === ".tf" || ext === ".tfvars" || /\.k8s\.ya?ml$/i.test(base) || n.includes("/k8s/") || n.includes("/kubernetes/")) return "infra";
  if ([".yaml", ".yml", ".json", ".toml", ".xml", ".cfg", ".ini", ".env"].includes(ext) || [".env", ".env.example", "tsconfig.json", "package.json", "pyproject.toml", "Cargo.toml", "go.mod"].includes(base)) return "config";
  if ([".sql", ".graphql", ".gql", ".proto", ".prisma", ".csv"].includes(ext) || /\.schema\.json$/i.test(base)) return "data";
  if ([".sh", ".bash", ".ps1", ".bat"].includes(ext)) return "script";
  if ([".html", ".htm", ".css", ".scss", ".sass", ".less"].includes(ext)) return "markup";
  return "code";
}

function countLines(abs) {
  const content = fs.readFileSync(abs, "utf8");
  if (content.length === 0) return 0;
  const matches = content.match(/\n/g);
  return (matches ? matches.length : 0) + (content.endsWith("\n") ? 0 : 1);
}

function readText(root, rel) {
  try {
    return fs.readFileSync(path.join(root, rel), "utf8");
  } catch (_) {
    return "";
  }
}

function detectFrameworks(root, files) {
  const frameworks = new Set();
  const req = readText(root, "requirements.txt");
  const pythonFrameworks = {
    django: "Django",
    djangorestframework: "Django REST Framework",
    fastapi: "FastAPI",
    flask: "Flask",
    sqlalchemy: "SQLAlchemy",
    alembic: "Alembic",
    celery: "Celery",
    pydantic: "Pydantic",
    uvicorn: "Uvicorn",
    gunicorn: "Gunicorn",
    aiohttp: "aiohttp",
    tornado: "Tornado",
    starlette: "Starlette",
    pytest: "pytest",
    hypothesis: "Hypothesis",
    channels: "Django Channels",
  };
  for (const line of req.split(/\r?\n/)) {
    const pkg = line.trim().split(/[<>=!~;\s[]/)[0].toLowerCase();
    if (pythonFrameworks[pkg]) frameworks.add(pythonFrameworks[pkg]);
    if (pkg === "torchmetrics") frameworks.add("PyTorch");
    if (pkg === "wandb") frameworks.add("Weights & Biases");
  }
  const allPaths = new Set(files);
  if (allPaths.has("Dockerfile")) frameworks.add("Docker");
  if ([...allPaths].some((p) => basename(p) === "docker-compose.yml" || basename(p) === "docker-compose.yaml")) frameworks.add("Docker Compose");
  if ([...allPaths].some((p) => p.endsWith(".tf"))) frameworks.add("Terraform");
  if ([...allPaths].some((p) => p.startsWith(".github/workflows/") && /\.ya?ml$/i.test(p))) frameworks.add("GitHub Actions");
  if (allPaths.has(".gitlab-ci.yml")) frameworks.add("GitLab CI");
  if (allPaths.has("Jenkinsfile")) frameworks.add("Jenkins");
  return [...frameworks].sort((a, b) => a.localeCompare(b));
}

function projectName(root) {
  const pyproject = readText(root, "pyproject.toml");
  const pyName = pyproject.match(/^\s*name\s*=\s*["']([^"']+)["']/m);
  if (pyName) return pyName[1];
  return path.basename(root);
}

function resolvePythonImport(rel, dots, moduleName, fileSet) {
  const parts = normalizeRel(rel).split("/");
  parts.pop();
  const levels = Math.max(0, dots.length - 1);
  for (let i = 0; i < levels; i++) parts.pop();
  const moduleParts = moduleName ? moduleName.split(".").filter(Boolean) : [];
  const base = parts.concat(moduleParts).join("/");
  const candidates = [`${base}.py`, `${base}/__init__.py`].filter((x) => x && x !== ".py");
  return candidates.find((candidate) => fileSet.has(candidate)) || null;
}

function buildImportMap(root, fileRecords) {
  const fileSet = new Set(fileRecords.map((f) => f.path));
  const importMap = {};
  for (const file of fileRecords) {
    importMap[file.path] = [];
    if (file.fileCategory !== "code" || file.language !== "python") continue;
    const content = readText(root, file.path);
    const resolved = new Set();
    const re = /^\s*from\s+(\.+)([\w.]+)?\s+import\s+([\w*,\s()]+)/gm;
    let match;
    while ((match = re.exec(content))) {
      const target = resolvePythonImport(file.path, match[1], match[2] || "", fileSet);
      if (target) resolved.add(target);
    }
    importMap[file.path] = [...resolved].sort((a, b) => a.localeCompare(b));
  }
  return importMap;
}

function descriptionFromReadme(head) {
  const clean = head
    .split(/\r?\n/)
    .map((line) => line.replace(/^#+\s*/, "").replace(/!\[[^\]]*\]\([^)]*\)/g, "").trim())
    .filter(Boolean)
    .join(" ");
  const sentence = clean || "No description available";
  return sentence.length > 240 ? `${sentence.slice(0, 237).trim()}...` : sentence;
}

async function main() {
  const root = path.resolve(process.argv[2] || "");
  const outputPath = path.resolve(process.argv[3] || "");
  if (!root || !outputPath || !fs.existsSync(root) || !fs.statSync(root).isDirectory()) {
    console.error("Usage: node ua-project-scan.js <project-root> <output-path>");
    process.exit(1);
  }

  const rawFiles = discoverWithGit(root);
  const discovered = (rawFiles.length ? rawFiles : discoverRecursive(root)).filter((rel) => fs.existsSync(path.join(root, rel)));
  const baselineFiles = discovered.filter((rel) => !baselineIgnored(rel));

  let finalFiles = baselineFiles;
  let filteredByIgnore = 0;
  if (userIgnoreFilesExist(root)) {
    const coreFilter = await createCoreIgnoreFilter(root);
    if (coreFilter) {
      const coreFiltered = discovered.filter((rel) => !isUnderToolingDir(rel) && !coreFilter.isIgnored(rel));
      filteredByIgnore = Math.max(0, baselineFiles.length - coreFiltered.length);
      finalFiles = coreFiltered;
    } else {
      const patterns = loadSimpleUserPatterns(root);
      const simpleFiltered = discovered.filter((rel) => !baselineIgnored(rel) && !simplePatternIgnored(rel, patterns));
      filteredByIgnore = Math.max(0, baselineFiles.length - simpleFiltered.length);
      finalFiles = simpleFiltered;
    }
  }

  const records = finalFiles
    .sort((a, b) => a.localeCompare(b))
    .map((rel) => ({
      path: rel,
      language: languageFor(rel),
      sizeLines: countLines(path.join(root, rel)),
      fileCategory: categoryFor(rel),
    }));

  const languages = [...new Set(records.map((f) => f.language).filter((l) => l !== "unknown"))].sort((a, b) => a.localeCompare(b));
  const readmeHead = readText(root, "README.md").split(/\r?\n/).slice(0, 10).join("\n");
  const frameworks = detectFrameworks(root, records.map((f) => f.path));
  const totalFiles = records.length;
  const estimatedComplexity = totalFiles <= 30 ? "small" : totalFiles <= 150 ? "moderate" : totalFiles <= 500 ? "large" : "very-large";
  const name = projectName(root);
  let description = "FraudGT is a graph transformer codebase for financial fraud detection, with AML and Ethereum dataset processing, experiment configs, training scripts, and GraphGym-style model components.";
  if (!readText(root, "README.md")) description = descriptionFromReadme(readmeHead);
  if (totalFiles > 100) description += " Note: this project has over 100 source files; consider scoping analysis to a subdirectory for faster results.";

  const result = {
    name,
    description,
    languages,
    frameworks,
    files: records,
    totalFiles,
    filteredByIgnore,
    estimatedComplexity,
    importMap: buildImportMap(root, records),
  };

  if (result.totalFiles !== result.files.length) throw new Error("totalFiles mismatch");
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${JSON.stringify(result, null, 2)}\n`);
}

main().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
