import express from 'express';
import cors from 'cors';
import { createServer } from 'node:http';
import { Server } from 'socket.io';
import Database from 'better-sqlite3';
import {
  createHash,
  pbkdf2Sync,
  randomBytes,
  randomUUID,
  timingSafeEqual
} from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { mkdirSync } from 'node:fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT || 7860);
const DB_PATH = process.env.DB_PATH || path.join(__dirname, 'data', 'ai-studio.sqlite');
const PUBLIC_DIR = path.join(__dirname, 'public');
const NGROK_HEADERS = {
  'ngrok-skip-browser-warning': 'true',
  'content-type': 'application/json'
};

mkdirSync(path.dirname(DB_PATH), { recursive: true });

const db = new Database(DB_PATH);
db.pragma('journal_mode = WAL');
db.pragma('foreign_keys = ON');

db.exec(`
  CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    token TEXT NOT NULL UNIQUE,
    token_hash TEXT NOT NULL UNIQUE,
    hf_token TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS workers (
    id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    name TEXT NOT NULL,
    url TEXT NOT NULL,
    worker_token_hash TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'offline',
    metrics TEXT NOT NULL DEFAULT '{}',
    last_seen TEXT,
    updated_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (owner_id) REFERENCES users(id) ON DELETE CASCADE
  );

  CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    worker_id TEXT NOT NULL,
    model TEXT NOT NULL,
    dataset TEXT NOT NULL,
    task TEXT NOT NULL,
    output_repo TEXT,
    push_to_hf INTEGER NOT NULL DEFAULT 0,
    params TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'queued',
    progress REAL NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (owner_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (worker_id) REFERENCES workers(id) ON DELETE CASCADE
  );

  CREATE TABLE IF NOT EXISTS job_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    level TEXT NOT NULL DEFAULT 'info',
    message TEXT NOT NULL,
    progress REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
    FOREIGN KEY (owner_id) REFERENCES users(id) ON DELETE CASCADE
  );

  CREATE INDEX IF NOT EXISTS idx_workers_owner ON workers(owner_id);
  CREATE INDEX IF NOT EXISTS idx_jobs_owner ON jobs(owner_id);
  CREATE INDEX IF NOT EXISTS idx_jobs_worker ON jobs(worker_id);
  CREATE INDEX IF NOT EXISTS idx_job_logs_job ON job_logs(job_id);
`);

const app = express();
const httpServer = createServer(app);
const io = new Server(httpServer, {
  cors: {
    origin: true,
    credentials: false
  }
});

app.use(cors());
app.use(express.json({ limit: '5mb' }));
app.use(express.static(PUBLIC_DIR));

function now() {
  return new Date().toISOString();
}

function sha256(value) {
  return createHash('sha256').update(String(value)).digest('hex');
}

function randomToken(prefix) {
  return `${prefix}_${randomBytes(24).toString('hex')}`;
}

function hashPassword(password) {
  const salt = randomBytes(32);
  const hash = pbkdf2Sync(password, salt, 310_000, 64, 'sha256');
  return `${salt.toString('hex')}:${hash.toString('hex')}`;
}

function verifyPassword(password, storedHash) {
  try {
    const [saltHex, hashHex] = storedHash.split(':');
    const salt = Buffer.from(saltHex, 'hex');
    const hash = pbkdf2Sync(password, salt, 310_000, 64, 'sha256');
    return timingSafeEqual(hash, Buffer.from(hashHex, 'hex'));
  } catch {
    return false;
  }
}

function parseJson(value, fallback = {}) {
  if (!value) return fallback;
  try {
    return JSON.parse(value);
  } catch {
    return fallback;
  }
}

function intParam(value, fallback, min, max) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.min(max, Math.max(min, Math.round(number)));
}

function floatParam(value, fallback, min, max) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.min(max, Math.max(min, number));
}

function normalizeUrl(url) {
  return String(url || '').trim().replace(/\/+$/, '');
}

function sanitizeTrainingParams(input = {}) {
  return {
    epochs: intParam(input.epochs, 3, 1, 100),
    batch_size: intParam(input.batch_size ?? input.batchSize, 2, 1, 128),
    learning_rate: floatParam(input.learning_rate ?? input.learningRate, 0.0002, 0.000001, 0.01),
    max_seq_length: intParam(input.max_seq_length ?? input.maxSeqLength, 512, 64, 8192),
    lora_rank: intParam(input.lora_rank ?? input.loraRank, 16, 1, 256),
    lora_alpha: intParam(input.lora_alpha ?? input.loraAlpha, 32, 1, 512),
    gradient_accumulation_steps: intParam(input.gradient_accumulation_steps ?? input.gradientAccumulationSteps, 1, 1, 256),
    warmup_ratio: floatParam(input.warmup_ratio ?? input.warmupRatio, 0.03, 0, 0.5),
    weight_decay: floatParam(input.weight_decay ?? input.weightDecay, 0.0, 0, 1),
    logging_steps: intParam(input.logging_steps ?? input.loggingSteps, 10, 1, 1000)
  };
}

function getWorkers(ownerId) {
  return db
    .prepare(`
      SELECT id, owner_id AS ownerId, name, url, status, metrics, last_seen AS lastSeen, created_at AS createdAt
      FROM workers
      WHERE owner_id = ?
      ORDER BY last_seen DESC, created_at DESC
    `)
    .all(ownerId)
    .map((worker) => ({
      ...worker,
      metrics: parseJson(worker.metrics, {})
    }));
}

function getWorker(ownerId, workerId) {
  return db
    .prepare(`
      SELECT id, owner_id AS ownerId, name, url, status, metrics, last_seen AS lastSeen, created_at AS createdAt
      FROM workers
      WHERE id = ? AND owner_id = ?
    `)
    .get(workerId, ownerId);
}

function getJobs(ownerId) {
  return db
    .prepare(`
      SELECT
        j.id,
        j.owner_id AS ownerId,
        j.worker_id AS workerId,
        w.name AS workerName,
        j.model,
        j.dataset,
        j.task,
        j.output_repo AS outputRepo,
        j.push_to_hf AS pushToHf,
        j.params,
        j.status,
        j.progress,
        j.error,
        j.created_at AS createdAt,
        j.updated_at AS updatedAt
      FROM jobs j
      LEFT JOIN workers w ON w.id = j.worker_id
      WHERE j.owner_id = ?
      ORDER BY j.created_at DESC
    `)
    .all(ownerId)
    .map((job) => ({
      ...job,
      params: parseJson(job.params, {}),
      pushToHf: Boolean(job.pushToHf)
    }));
}

function getJob(ownerId, jobId) {
  return db
    .prepare(`
      SELECT
        j.id,
        j.owner_id AS ownerId,
        j.worker_id AS workerId,
        w.name AS workerName,
        j.model,
        j.dataset,
        j.task,
        j.output_repo AS outputRepo,
        j.push_to_hf AS pushToHf,
        j.params,
        j.status,
        j.progress,
        j.error,
        j.created_at AS createdAt,
        j.updated_at AS updatedAt
      FROM jobs j
      LEFT JOIN workers w ON w.id = j.worker_id
      WHERE j.id = ? AND j.owner_id = ?
    `)
    .get(jobId, ownerId);
}

function getJobLogs(ownerId, jobId) {
  return db
    .prepare(`
      SELECT id, job_id AS jobId, owner_id AS ownerId, level, message, progress, created_at AS createdAt
      FROM job_logs
      WHERE job_id = ? AND owner_id = ?
      ORDER BY id ASC
      LIMIT 2000
    `)
    .all(jobId, ownerId);
}

function authRequired(req, res, next) {
  const header = req.headers.authorization || '';
  const token = header.startsWith('Bearer ') ? header.slice(7).trim() : header.trim();

  if (!token) {
    return res.status(401).json({ error: 'Token utilisateur requis.' });
  }

  const user = db
    .prepare('SELECT * FROM users WHERE token_hash = ?')
    .get(sha256(token));

  if (!user) {
    return res.status(401).json({ error: 'Token utilisateur invalide.' });
  }

  req.user = user;
  return next();
}

function workerAuthRequired(req, res, next) {
  const header = req.headers.authorization || '';
  const token = header.startsWith('Bearer ') ? header.slice(7).trim() : header.trim();

  if (!token) {
    return res.status(401).json({ error: 'Worker token requis.' });
  }

  const worker = db
    .prepare('SELECT * FROM workers WHERE worker_token_hash = ?')
    .get(sha256(token));

  if (!worker) {
    return res.status(401).json({ error: 'Worker token invalide.' });
  }

  req.worker = worker;
  return next();
}

async function workerRequest(worker, method, route, body) {
  const url = `${normalizeUrl(worker.url)}${route.startsWith('/') ? route : `/${route}`}`;

  const response = await fetch(url, {
    method,
    headers: NGROK_HEADERS,
    body: body === undefined ? undefined : JSON.stringify(body)
  });

  const text = await response.text();
  let data = {};

  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { raw: text.slice(0, 1000) };
  }

  if (!response.ok) {
    throw new Error(`Erreur worker ${response.status}: ${response.statusText} - ${text.slice(0, 500)}`);
  }

  return data;
}

function broadcastUser(userId, event, payload) {
  io.to(`user:${userId}`).emit(event, payload);
}

app.get('/api/health', (req, res) => {
  res.json({ ok: true, service: 'ai-studio-server', time: now() });
});

app.post('/api/auth/register', async (req, res) => {
  try {
    const username = String(req.body.username || '').trim();
    const password = String(req.body.password || '');

    if (username.length < 3) {
      return res.status(400).json({ error: 'Le nom d’utilisateur doit contenir au moins 3 caractères.' });
    }

    if (password.length < 6) {
      return res.status(400).json({ error: 'Le mot de passe doit contenir au moins 6 caractères.' });
    }

    const id = randomUUID();
    const token = randomToken('user');
    const passwordHash = hashPassword(password);

    db.prepare(`
      INSERT INTO users (id, username, password_hash, token, token_hash)
      VALUES (?, ?, ?, ?, ?)
    `).run(id, username, passwordHash, token, sha256(token));

    return res.json({
      token,
      user: { id, username }
    });
  } catch (error) {
    if (String(error.code).includes('SQLITE_CONSTRAINT')) {
      return res.status(409).json({ error: 'Ce nom d’utilisateur existe déjà.' });
    }
    console.error(error);
    return res.status(500).json({ error: 'Erreur serveur lors de l’inscription.' });
  }
});

app.post('/api/auth/login', (req, res) => {
  const username = String(req.body.username || '').trim();
  const password = String(req.body.password || '');
  const user = db.prepare('SELECT * FROM users WHERE username = ?').get(username);

  if (!user || !verifyPassword(password, user.password_hash)) {
    return res.status(401).json({ error: 'Nom d’utilisateur ou mot de passe invalide.' });
  }

  return res.json({
    token: user.token,
    user: {
      id: user.id,
      username: user.username
    }
  });
});

app.get('/api/me', authRequired, (req, res) => {
  res.json({
    user: {
      id: req.user.id,
      username: req.user.username
    },
    userToken: req.user.token,
    hfTokenConfigured: Boolean(req.user.hf_token)
  });
});

app.post('/api/hf/token', authRequired, (req, res) => {
  const hfToken = String(req.body.token || '').trim();

  if (!hfToken) {
    db.prepare('UPDATE users SET hf_token = NULL WHERE id = ?').run(req.user.id);
    return res.json({ ok: true, hfTokenConfigured: false });
  }

  db.prepare('UPDATE users SET hf_token = ? WHERE id = ?').run(hfToken, req.user.id);
  return res.json({ ok: true, hfTokenConfigured: true });
});

app.get('/api/hf/search', authRequired, async (req, res) => {
  try {
    const type = req.query.type === 'dataset' ? 'datasets' : 'models';
    const query = String(req.query.q || req.query.query || '').trim();
    const limit = Math.min(intParam(req.query.limit, 20, 1, 100), 100);
    const url = new URL(`https://huggingface.co/api/${type}`);

    if (query) url.searchParams.set('search', query);
    url.searchParams.set('limit', String(limit));

    const headers = {};
    if (req.user.hf_token) {
      headers.Authorization = `Bearer ${req.user.hf_token}`;
    }

    const response = await fetch(url, { headers });
    const data = await response.json();

    if (!response.ok) {
      return res.status(response.status).json({
        error: 'Erreur Hugging Face API.',
        details: data
      });
    }

    return res.json({ type, items: data });
  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: 'Erreur lors de la recherche Hugging Face.' });
  }
});

app.get('/api/workers', authRequired, (req, res) => {
  res.json({ workers: getWorkers(req.user.id) });
});

app.post('/api/workers', authRequired, (req, res) => {
  const name = String(req.body.name || '').trim();
  const url = normalizeUrl(req.body.url || req.body.ngrokUrl || '');

  if (!name) {
    return res.status(400).json({ error: 'Le nom de la machine est requis.' });
  }

  if (!url) {
    return res.status(400).json({ error: 'L’URL ngrok de la machine est requise.' });
  }

  const id = randomUUID();
  const workerToken = randomToken('worker');

  db.prepare(`
    INSERT INTO workers (id, owner_id, name, url, worker_token_hash, status)
    VALUES (?, ?, ?, ?, ?, 'offline')
  `).run(id, req.user.id, name, url, sha256(workerToken));

  const worker = getWorker(req.user.id, id);

  broadcastUser(req.user.id, 'workers:update', getWorkers(req.user.id));

  return res.status(201).json({
    worker,
    workerToken
  });
});

app.get('/api/workers/:id', authRequired, (req, res) => {
  const worker = getWorker(req.user.id, req.params.id);

  if (!worker) {
    return res.status(404).json({ error: 'Machine introuvable.' });
  }

  res.json({ worker });
});

app.post('/api/workers/register', (req, res) => {
  const workerToken = String(req.body.workerToken || '');
  const status = req.body.status === 'busy' ? 'busy' : 'online';
  const metrics = req.body.metrics || {};
  const worker = db
    .prepare('SELECT * FROM workers WHERE worker_token_hash = ?')
    .get(sha256(workerToken));

  if (!worker) {
    return res.status(401).json({ error: 'Worker token invalide.' });
  }

  db.prepare(`
    UPDATE workers
    SET status = ?, metrics = ?, last_seen = ?, updated_at = ?
    WHERE id = ?
  `).run(status, JSON.stringify(metrics), now(), now(), worker.id);

  const updatedWorker = getWorker(worker.owner_id, worker.id);
  broadcastUser(worker.owner_id, 'workers:update', getWorkers(worker.owner_id));

  return res.json({
    ok: true,
    worker: updatedWorker
  });
});

app.get('/api/workers/:id/health', authRequired, async (req, res) => {
  const worker = getWorker(req.user.id, req.params.id);

  if (!worker) {
    return res.status(404).json({ error: 'Machine introuvable.' });
  }

  try {
    const remote = await workerRequest(worker, 'GET', '/health');
    return res.json({ ok: true, remote });
  } catch (error) {
    db.prepare('UPDATE workers SET status = ?, last_seen = ? WHERE id = ?').run('offline', now(), worker.id);
    broadcastUser(req.user.id, 'workers:update', getWorkers(req.user.id));
    return res.status(502).json({ ok: false, error: error.message });
  }
});

app.get('/api/workers/:id/metrics', authRequired, async (req, res) => {
  const worker = getWorker(req.user.id, req.params.id);

  if (!worker) {
    return res.status(404).json({ error: 'Machine introuvable.' });
  }

  try {
    const remote = await workerRequest(worker, 'GET', '/metrics');
    db.prepare('UPDATE workers SET metrics = ?, last_seen = ? WHERE id = ?').run(
      JSON.stringify(remote),
      now(),
      worker.id
    );
    broadcastUser(req.user.id, 'metrics:update', { workerId: worker.id, metrics: remote });
    return res.json({ ok: true, metrics: remote });
  } catch (error) {
    db.prepare('UPDATE workers SET status = ?, last_seen = ? WHERE id = ?').run('offline', now(), worker.id);
    broadcastUser(req.user.id, 'workers:update', getWorkers(req.user.id));
    return res.status(502).json({ ok: false, error: error.message });
  }
});

app.delete('/api/workers/:id', authRequired, (req, res) => {
  const worker = getWorker(req.user.id, req.params.id);

  if (!worker) {
    return res.status(404).json({ error: 'Machine introuvable.' });
  }

  const activeJob = db
    .prepare('SELECT id FROM jobs WHERE worker_id = ? AND status IN (?, ?)')
    .get(worker.id, 'queued', 'running');

  if (activeJob) {
    return res.status(409).json({ error: 'Impossible de supprimer une machine qui exécute un job.' });
  }

  db.prepare('DELETE FROM workers WHERE id = ? AND owner_id = ?').run(worker.id, req.user.id);
  broadcastUser(req.user.id, 'workers:update', getWorkers(req.user.id));

  return res.json({ ok: true });
});

app.get('/api/jobs', authRequired, (req, res) => {
  res.json({ jobs: getJobs(req.user.id) });
});

app.get('/api/jobs/:id', authRequired, (req, res) => {
  const job = getJob(req.user.id, req.params.id);

  if (!job) {
    return res.status(404).json({ error: 'Job introuvable.' });
  }

  res.json({ job });
});

app.get('/api/jobs/:id/logs', authRequired, (req, res) => {
  const job = getJob(req.user.id, req.params.id);

  if (!job) {
    return res.status(404).json({ error: 'Job introuvable.' });
  }

  res.json({ logs: getJobLogs(req.user.id, req.params.id) });
});

app.post('/api/jobs', authRequired, async (req, res) => {
  try {
    const workerId = String(req.body.workerId || '');
    const model = String(req.body.model || '').trim();
    const dataset = String(req.body.dataset || '').trim();
    const task = String(req.body.task || 'text-generation');
    const outputRepo = String(req.body.outputRepo || '').trim();
    const pushToHf = Boolean(req.body.pushToHf);
    const params = sanitizeTrainingParams(req.body.params || {});

    const allowedTasks = ['text-generation', 'instruction-tuning', 'text-classification'];
    if (!allowedTasks.includes(task)) {
      return res.status(400).json({ error: 'Tâche d’entraînement invalide.' });
    }

    if (!model) {
      return res.status(400).json({ error: 'Le modèle est requis.' });
    }

    if (!dataset) {
      return res.status(400).json({ error: 'Le dataset est requis.' });
    }

    const worker = getWorker(req.user.id, workerId);

    if (!worker) {
      return res.status(404).json({ error: 'Machine introuvable.' });
    }

    if (worker.status === 'offline') {
      return res.status(400).json({ error: 'La machine sélectionnée est offline.' });
    }

    const activeJob = db
      .prepare('SELECT id FROM jobs WHERE worker_id = ? AND status IN (?, ?)')
      .get(worker.id, 'queued', 'running');

    if (activeJob) {
      return res.status(409).json({ error: 'Cette machine exécute déjà un job. Les GPU ne sont pas partagés.' });
    }

    const hfTokenRow = db.prepare('SELECT hf_token AS hfToken FROM users WHERE id = ?').get(req.user.id);
    const jobId = randomUUID();

    const insert = db.prepare(`
      INSERT INTO jobs (
        id, owner_id, worker_id, model, dataset, task, output_repo, push_to_hf, params, status, progress
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', 0)
    `);

    insert.run(
      jobId,
      req.user.id,
      worker.id,
      model,
      dataset,
      task,
      outputRepo || null,
      pushToHf ? 1 : 0,
      JSON.stringify(params)
    );

    const payload = {
      jobId,
      model,
      dataset,
      task,
      outputRepo: outputRepo || null,
      pushToHf,
      params,
      hfToken: hfTokenRow.hfToken || ''
    };

    try {
      await workerRequest(worker, 'POST', '/start-job', payload);
    } catch (error) {
      db.prepare(`
        UPDATE jobs
        SET status = 'failed', progress = 0, error = ?, updated_at = ?
        WHERE id = ?
      `).run(error.message, now(), jobId);

      broadcastUser(req.user.id, 'jobs:update', getJobs(req.user.id));
      broadcastUser(req.user.id, 'job:log', {
        jobId,
        level: 'error',
        message: `Échec de l’envoi du job au worker: ${error.message}`,
        progress: 0
      });

      return res.status(502).json({
        ok: false,
        error: error.message,
        job: getJob(req.user.id, jobId)
      });
    }

    db.prepare(`
      UPDATE jobs
      SET status = 'running', updated_at = ?
      WHERE id = ?
    `).run(now(), jobId);

    db.prepare('UPDATE workers SET status = ?, updated_at = ? WHERE id = ?')
      .run('busy', now(), worker.id);

    const job = getJob(req.user.id, jobId);
    broadcastUser(req.user.id, 'jobs:update', getJobs(req.user.id));
    broadcastUser(req.user.id, 'workers:update', getWorkers(req.user.id));
    broadcastUser(req.user.id, 'job:log', {
      jobId,
      level: 'info',
      message: `Job envoyé à la machine ${worker.name}.`,
      progress: 0
    });

    return res.status(201).json({ ok: true, job });
  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: 'Erreur lors de la création du job.' });
  }
});

app.post('/api/jobs/:id/cancel', authRequired, async (req, res) => {
  const job = getJob(req.user.id, req.params.id);

  if (!job) {
    return res.status(404).json({ error: 'Job introuvable.' });
  }

  const terminalStatuses = ['completed', 'failed', 'cancelled'];
  if (terminalStatuses.includes(job.status)) {
    return res.status(400).json({ error: 'Ce job est déjà terminé.' });
  }

  const worker = getWorker(req.user.id, job.workerId);
  db.prepare(`
    UPDATE jobs
    SET status = 'cancelling', updated_at = ?
    WHERE id = ?
  `).run(now(), job.id);

  broadcastUser(req.user.id, 'jobs:update', getJobs(req.user.id));

  if (worker) {
    try {
      await workerRequest(worker, 'POST', '/stop-job', { jobId: job.id });
    } catch (error) {
      db.prepare('INSERT INTO job_logs (job_id, owner_id, level, message) VALUES (?, ?, ?, ?)')
        .run(job.id, req.user.id, 'error', `Erreur pendant la demande d’arrêt: ${error.message}`);
    }
  }

  return res.json({ ok: true, job: getJob(req.user.id, job.id) });
});

app.post('/api/jobs/:jobId/logs', workerAuthRequired, (req, res) => {
  const job = db.prepare('SELECT * FROM jobs WHERE id = ?').get(req.params.jobId);

  if (!job) {
    return res.status(404).json({ error: 'Job introuvable.' });
  }

  if (job.owner_id !== req.worker.owner_id) {
    return res.status(403).json({ error: 'Ce worker n’est pas autorisé à écrire dans ce job.' });
  }

  const level = String(req.body.level || 'info');
  const message = String(req.body.message || '');
  const progress = req.body.progress === undefined ? null : floatParam(req.body.progress, 0, 0, 100);

  if (!message) {
    return res.status(400).json({ error: 'Message de log requis.' });
  }

  db.prepare(`
    INSERT INTO job_logs (job_id, owner_id, level, message, progress)
    VALUES (?, ?, ?, ?, ?)
  `).run(job.id, job.owner_id, level, message, progress);

  if (progress !== null) {
    db.prepare('UPDATE jobs SET progress = ?, updated_at = ? WHERE id = ?').run(progress, now(), job.id);
  }

  broadcastUser(job.owner_id, 'job:log', {
    jobId: job.id,
    level,
    message,
    progress
  });

  return res.json({ ok: true });
});

app.post('/api/jobs/:jobId/status', workerAuthRequired, (req, res) => {
  const job = db.prepare('SELECT * FROM jobs WHERE id = ?').get(req.params.jobId);

  if (!job) {
    return res.status(404).json({ error: 'Job introuvable.' });
  }

  if (job.owner_id !== req.worker.owner_id) {
    return res.status(403).json({ error: 'Ce worker n’est pas autorisé à modifier ce job.' });
  }

  const allowedStatuses = ['queued', 'running', 'completed', 'failed', 'cancelled', 'cancelling'];
  const status = String(req.body.status || job.status);

  if (!allowedStatuses.includes(status)) {
    return res.status(400).json({ error: 'Statut invalide.' });
  }

  const progress = req.body.progress === undefined ? job.progress : floatParam(req.body.progress, 0, 0, 100);
  const error = req.body.error ? String(req.body.error) : null;

  db.prepare(`
    UPDATE jobs
    SET status = ?, progress = ?, error = COALESCE(?, error), updated_at = ?
    WHERE id = ?
  `).run(status, progress, error, now(), job.id);

  if (['completed', 'failed', 'cancelled'].includes(status)) {
    db.prepare('UPDATE workers SET status = ?, updated_at = ? WHERE id = ?').run('online', now(), job.worker_id);
  }

  const updatedJob = getJob(job.owner_id, job.id);
  broadcastUser(job.owner_id, 'jobs:update', getJobs(job.owner_id));
  broadcastUser(job.owner_id, 'workers:update', getWorkers(job.owner_id));

  return res.json({ ok: true, job: updatedJob });
});

app.get('/api/system/info', (req, res) => {
  res.json({
    ok: true,
    nodeVersion: process.version,
    socketIoEnabled: true,
    ngrokHeader: 'ngrok-skip-browser-warning'
  });
});

io.on('connection', (socket) => {
  const token = socket.handshake.auth?.token;

  if (!token) {
    socket.emit('error', { message: 'Token utilisateur requis.' });
    socket.disconnect(true);
    return;
  }

  const user = db.prepare('SELECT id, username, token FROM users WHERE token_hash = ?').get(sha256(token));

  if (!user) {
    socket.emit('error', { message: 'Token utilisateur invalide.' });
    socket.disconnect(true);
    return;
  }

  socket.data.user = user;
  socket.join(`user:${user.id}`);

  socket.emit('me', {
    user: { id: user.id, username: user.username },
    userToken: user.token
  });
  socket.emit('workers:update', getWorkers(user.id));
  socket.emit('jobs:update', getJobs(user.id));
});

app.get('*', (req, res) => {
  res.sendFile(path.join(PUBLIC_DIR, 'index.html'));
});

app.use((error, req, res, next) => {
  console.error(error);
  return res.status(500).json({ error: 'Erreur serveur interne.' });
});

httpServer.listen(PORT, () => {
  console.log(`AI Studio serveur démarré sur le port ${PORT}`);
  console.log(`Base de données: ${DB_PATH}`);
});