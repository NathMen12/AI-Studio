const API_BASE = '';
const STORAGE_TOKEN_KEY = 'ai_studio_user_token';

const state = {
  token: localStorage.getItem(STORAGE_TOKEN_KEY) || '',
  user: null,
  userToken: '',
  workers: [],
  jobs: [],
  logs: [],
  loadedLogIds: new Set(),
  selectedJobId: null,
  socket: null,
  pollingTimer: null
};

const $ = (id) => document.getElementById(id);

function setMessage(element, message, type = '') {
  element.textContent = message || '';
  element.className = `message ${type}`.trim();
}

async function api(path, options = {}) {
  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers || {})
  };

  if (state.token) {
    headers.Authorization = `Bearer ${state.token}`;
  }

  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers
  });

  const text = await response.text();
  let data = {};

  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { raw: text };
  }

  if (!response.ok) {
    throw new Error(data.error || data.message || `Erreur HTTP ${response.status}`);
  }

  return data;
}

function percent(used, total) {
  const usedNumber = Number(used || 0);
  const totalNumber = Number(total || 0);
  if (!totalNumber) return 0;
  return Math.max(0, Math.min(100, Math.round((usedNumber / totalNumber) * 100)));
}

function formatBytes(bytes) {
  const value = Number(bytes || 0);
  if (!value) return '0 GB';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let index = 0;
  let current = value;

  while (current >= 1024 && index < units.length - 1) {
    current /= 1024;
    index += 1;
  }

  return `${current.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function statusClass(status) {
  return String(status || 'offline').toLowerCase();
}

function showApp() {
  $('authPanel').classList.add('hidden');
  $('appPanel').classList.remove('hidden');
  $('logoutButton').classList.remove('hidden');
  $('userTokenDisplay').value = state.userToken || state.token;
  $('profileText').textContent = state.user
    ? `Connecté en tant que ${state.user.username}.`
    : 'Connecté.';
}

function showAuth() {
  $('authPanel').classList.remove('hidden');
  $('appPanel').classList.add('hidden');
  $('logoutButton').classList.add('hidden');
}

function connectSocket() {
  if (!window.io) {
    setMessage($('jobMessage'), 'Socket.IO non chargé. Le rafraîchissement automatique REST reste actif.', 'error');
    return;
  }

  if (state.socket) {
    state.socket.disconnect();
  }

  const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
  const socketUrl = `${protocol}//${window.location.host}`;

  state.socket = io(socketUrl, {
    auth: { token: state.token },
    reconnection: true,
    reconnectionAttempts: 20
  });

  state.socket.on('connect', () => {
    setMessage($('jobMessage'), 'Connexion temps réel active.', 'success');
  });

  state.socket.on('me', (payload) => {
    state.user = payload.user;
    state.userToken = payload.userToken;
    $('userTokenDisplay').value = payload.userToken;
    $('profileText').textContent = `Connecté en tant que ${payload.user.username}.`;
  });

  state.socket.on('workers:update', (payload) => {
    state.workers = payload || [];
    renderWorkers();
    renderWorkerSelect();
  });

  state.socket.on('jobs:update', (payload) => {
    state.jobs = payload || [];
    renderJobs();
  });

  state.socket.on('job:log', (payload) => {
    appendLog(payload);
  });

  state.socket.on('metrics:update', (payload) => {
    const worker = state.workers.find((item) => item.id === payload.workerId);
    if (worker) {
      worker.metrics = payload.metrics || {};
      renderWorkers();
    }
  });

  state.socket.on('error', (payload) => {
    setMessage($('jobMessage'), payload?.message || 'Erreur Socket.IO.', 'error');
  });
}

function startPolling() {
  if (state.pollingTimer) {
    clearInterval(state.pollingTimer);
  }

  state.pollingTimer = setInterval(() => {
    if (!state.token) return;
    refreshWorkers();
    refreshJobs();
  }, 5000);
}

async function init() {
  if (!state.token) {
    showAuth();
    return;
  }

  try {
    const me = await api('/api/me');
    state.user = me.user;
    state.userToken = me.userToken || state.token;
    showApp();
    connectSocket();
    startPolling();
    await refreshAll();
  } catch (error) {
    localStorage.removeItem(STORAGE_TOKEN_KEY);
    state.token = '';
    showAuth();
    setMessage($('loginMessage'), `Session invalide: ${error.message}`, 'error');
  }
}

async function refreshAll() {
  await Promise.allSettled([refreshWorkers(), refreshJobs()]);
}

async function refreshWorkers() {
  const data = await api('/api/workers');
  state.workers = data.workers || [];
  renderWorkers();
  renderWorkerSelect();
}

async function refreshJobs() {
  const data = await api('/api/jobs');
  state.jobs = data.jobs || [];
  renderJobs();

  if (state.selectedJobId) {
    await loadLogs(state.selectedJobId, false);
  }
}

function renderWorkers() {
  const container = $('workersList');

  if (!state.workers.length) {
    container.innerHTML = '<p class="muted">Aucune machine GPU enregistrée. Ajoute une machine avec son URL ngrok.</p>';
    return;
  }

  container.innerHTML = state.workers.map((worker) => {
    const metrics = worker.metrics || {};
    const gpuName = metrics.gpu_name || metrics.gpu?.name || metrics.gpu || 'GPU inconnu';
    const vramUsed = Number(metrics.vram_used || 0);
    const vramTotal = Number(metrics.vram_total || 0);
    const ramUsed = Number(metrics.ram_used || 0);
    const ramTotal = Number(metrics.ram_total || 0);
    const cpu = Number(metrics.cpu_percent || metrics.cpu?.percent || 0);
    const vramPercent = percent(vramUsed, vramTotal);
    const ramPercent = percent(ramUsed, ramTotal);

    return `
      <article class="worker-card">
        <header>
          <div>
            <strong>${escapeHtml(worker.name)}</strong>
            <p class="small">${escapeHtml(worker.url)}</p>
          </div>
          <span class="status-pill ${statusClass(worker.status)}">${escapeHtml(worker.status)}</span>
        </header>

        <div class="metric">
          <span><b>GPU</b><span>${escapeHtml(gpuName)}</span></span>
        </div>

        <div class="metric">
          <span><b>VRAM</b><span>${formatBytes(vramUsed)} / ${formatBytes(vramTotal)} · ${vramPercent}%</span></span>
          <div class="progress"><i style="width:${vramPercent}%"></i></div>
        </div>

        <div class="metric">
          <span><b>RAM</b><span>${formatBytes(ramUsed)} / ${formatBytes(ramTotal)} · ${ramPercent}%</span></span>
          <div class="progress"><i style="width:${ramPercent}%"></i></div>
        </div>

        <div class="metric">
          <span><b>CPU</b><span>${cpu.toFixed(1)}%</span></span>
          <div class="progress"><i style="width:${Math.max(0, Math.min(100, cpu))}%"></i></div>
        </div>

        <p class="small">Dernier heartbeat: ${worker.lastSeen ? new Date(worker.lastSeen).toLocaleString() : 'jamais'}</p>

        <div class="row">
          <button class="button small" type="button" data-action="health" data-worker-id="${worker.id}">Tester santé</button>
          <button class="button small" type="button" data-action="metrics" data-worker-id="${worker.id}">Métriques</button>
          <button class="button small secondary" type="button" data-action="delete" data-worker-id="${worker.id}">Supprimer</button>
        </div>
      </article>
    `;
  }).join('');

  container.querySelectorAll('[data-action="health"]').forEach((button) => {
    button.addEventListener('click', async () => {
      try {
        const data = await api(`/api/workers/${button.dataset.workerId}/health`);
        setMessage($('hfTokenMessage'), `Santé worker: ${JSON.stringify(data.remote)}`, 'success');
      } catch (error) {
        setMessage($('hfTokenMessage'), error.message, 'error');
      }
    });
  });

  container.querySelectorAll('[data-action="metrics"]').forEach((button) => {
    button.addEventListener('click', async () => {
      try {
        const data = await api(`/api/workers/${button.dataset.workerId}/metrics`);
        setMessage($('hfTokenMessage'), `Métriques reçues: ${JSON.stringify(data.metrics)}`, 'success');
      } catch (error) {
        setMessage($('hfTokenMessage'), error.message, 'error');
      }
    });
  });

  container.querySelectorAll('[data-action="delete"]').forEach((button) => {
    button.addEventListener('click', async () => {
      if (!confirm('Supprimer cette machine GPU ?')) return;

      try {
        await api(`/api/workers/${button.dataset.workerId}`, { method: 'DELETE' });
        await refreshWorkers();
      } catch (error) {
        setMessage($('hfTokenMessage'), error.message, 'error');
      }
    });
  });
}

function renderWorkerSelect() {
  const select = $('jobWorkerId');
  select.innerHTML = '';

  if (!state.workers.length) {
    const option = document.createElement('option');
    option.value = '';
    option.textContent = 'Aucune machine disponible';
    select.appendChild(option);
    select.disabled = true;
    return;
  }

  select.disabled = false;

  for (const worker of state.workers) {
    const option = document.createElement('option');
    option.value = worker.id;
    option.disabled = worker.status === 'busy' || worker.status === 'offline';
    option.textContent = `${worker.name} - ${worker.status}`;
    select.appendChild(option);
  }
}

function renderResults(items, type) {
  const container = $('hfResults');

  if (!items || !items.length) {
    container.innerHTML = '<p class="muted">Aucun résultat.</p>';
    return;
  }

  container.innerHTML = items.map((item) => {
    const id = item.modelId || item.id;
    const likes = item.likes ?? item.downloads ?? 0;
    const privateText = item.private ? 'privé' : item.gated ? 'gated' : 'public';
    const pipeline = item.pipeline_tag ? ` · ${item.pipeline_tag}` : '';

    return `
      <article class="result-card">
        <header>
          <strong>${escapeHtml(id)}</strong>
          <span class="status-pill ${item.private || item.gated ? 'busy' : 'online'}">${privateText}</span>
        </header>
        <p class="small">${escapeHtml(pipeline || (type === 'dataset' ? 'Dataset Hugging Face' : 'Modèle Hugging Face'))}</p>
        <p class="small">Likes / popularité: ${likes}</p>
        <button
          class="button small primary"
          type="button"
          data-use="${type === 'dataset' ? 'dataset' : 'model'}"
          data-value="${escapeHtml(id)}"
        >
          Utiliser
        </button>
      </article>
    `;
  }).join('');

  container.querySelectorAll('[data-use]').forEach((button) => {
    button.addEventListener('click', () => {
      if (button.dataset.use === 'dataset') {
        $('jobDataset').value = button.dataset.value;
      } else {
        $('jobModel').value = button.dataset.value;
      }
    });
  });
}

function renderJobs() {
  const container = $('jobsList');

  if (!state.jobs.length) {
    container.innerHTML = '<p class="muted">Aucun job d’entraînement.</p>';
    return;
  }

  container.innerHTML = state.jobs.map((job) => {
    const terminal = ['completed', 'failed', 'cancelled'].includes(job.status);
    const canCancel = !terminal && job.status !== 'cancelling';

    return `
      <article class="job-card">
        <header>
          <div>
            <strong>Job ${escapeHtml(job.id)}</strong>
            <p class="small">${new Date(job.createdAt).toLocaleString()}</p>
          </div>
          <span class="status-pill ${statusClass(job.status)}">${escapeHtml(job.status)}</span>
        </header>

        <div class="job-meta">
          <code>Modèle: ${escapeHtml(job.model)}</code>
          <code>Dataset: ${escapeHtml(job.dataset)}</code>
          <code>Worker: ${escapeHtml(job.workerName || job.workerId)}</code>
          <code>Tâche: ${escapeHtml(job.task)}</code>
        </div>

        <div class="metric">
          <span><b>Progression</b><span>${Number(job.progress || 0).toFixed(1)}%</span></span>
          <div class="progress"><i style="width:${Math.max(0, Math.min(100, Number(job.progress || 0)))}%"></i></div>
        </div>

        ${job.error ? `<p class="message error">${escapeHtml(job.error)}</p>` : ''}

        <div class="row">
          <button class="button small" type="button" data-action="logs" data-job-id="${job.id}">Charger les logs</button>
          ${canCancel ? `<button class="button small secondary" type="button" data-action="cancel" data-job-id="${job.id}">Arrêter</button>` : ''}
        </div>
      </article>
    `;
  }).join('');

  container.querySelectorAll('[data-action="logs"]').forEach((button) => {
    button.addEventListener('click', async () => {
      state.selectedJobId = button.dataset.jobId;
      await loadLogs(state.selectedJobId, true);
    });
  });

  container.querySelectorAll('[data-action="cancel"]').forEach((button) => {
    button.addEventListener('click', async () => {
      if (!confirm('Arrêter ce job ?')) return;

      try {
        await api(`/api/jobs/${button.dataset.jobId}/cancel`, { method: 'POST' });
        await refreshJobs();
      } catch (error) {
        setMessage($('jobMessage'), error.message, 'error');
      }
    });
  });
}

async function loadLogs(jobId, replace) {
  try {
    const data = await api(`/api/jobs/${jobId}/logs`);
    if (replace) {
      state.logs = [];
      state.loadedLogIds.clear();
      $('logsOutput').textContent = '';
    }

    for (const log of data.logs || []) {
      if (!replace && log.id && state.loadedLogIds.has(log.id)) {
        continue;
      }

      if (log.id) {
        state.loadedLogIds.add(log.id);
      }

      appendLog({
        jobId: log.jobId,
        level: log.level,
        message: log.message,
        progress: log.progress,
        createdAt: log.createdAt
      }, false);
    }
  } catch (error) {
    setMessage($('jobMessage'), error.message, 'error');
  }
}

function appendLog(payload, scroll = true) {
  if (!payload || !payload.message) return;

  if (payload.id && state.loadedLogIds.has(payload.id)) {
    return;
  }

  if (payload.id) {
    state.loadedLogIds.add(payload.id);
  }

  const time = payload.createdAt ? new Date(payload.createdAt).toLocaleTimeString() : new Date().toLocaleTimeString();
  const line = `[${time}] [${String(payload.level || 'info').toUpperCase()}] ${payload.message}`;
  state.logs.push(line);

  if (state.logs.length > 3000) {
    state.logs = state.logs.slice(-3000);
  }

  $('logsOutput').textContent = state.logs.join('\n');

  if (scroll) {
    $('logsOutput').scrollTop = $('logsOutput').scrollHeight;
  }
}

function escapeHtml(value) {
  const map = {
    '&': String.fromCharCode(38) + 'amp;',
    '<': String.fromCharCode(38) + 'lt;',
    '>': String.fromCharCode(38) + 'gt;',
    '"': String.fromCharCode(38) + 'quot;',
    "'": String.fromCharCode(38) + '#039;'
  };

  return String(value ?? '').replace(/[&<>"']/g, (char) => map[char]);
}

$('loginForm').addEventListener('submit', async (event) => {
  event.preventDefault();

  try {
    setMessage($('loginMessage'), 'Connexion...');
    const data = await api('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({
        username: $('loginUsername').value,
        password: $('loginPassword').value
      })
    });

    state.token = data.token;
    state.user = data.user;
    state.userToken = data.token;
    localStorage.setItem(STORAGE_TOKEN_KEY, data.token);
    showApp();
    connectSocket();
    startPolling();
    await refreshAll();
  } catch (error) {
    setMessage($('loginMessage'), error.message, 'error');
  }
});

$('registerForm').addEventListener('submit', async (event) => {
  event.preventDefault();

  try {
    setMessage($('registerMessage'), 'Création du compte...');
    const data = await api('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify({
        username: $('registerUsername').value,
        password: $('registerPassword').value
      })
    });

    state.token = data.token;
    state.user = data.user;
    state.userToken = data.token;
    localStorage.setItem(STORAGE_TOKEN_KEY, data.token);
    showApp();
    connectSocket();
    startPolling();
    await refreshAll();
  } catch (error) {
    setMessage($('registerMessage'), error.message, 'error');
  }
});

$('logoutButton').addEventListener('click', () => {
  localStorage.removeItem(STORAGE_TOKEN_KEY);
  state.token = '';
  state.user = null;
  state.userToken = '';
  state.workers = [];
  state.jobs = [];
  state.logs = [];

  if (state.socket) {
    state.socket.disconnect();
    state.socket = null;
  }

  if (state.pollingTimer) {
    clearInterval(state.pollingTimer);
    state.pollingTimer = null;
  }

  showAuth();
});

$('addWorkerForm').addEventListener('submit', async (event) => {
  event.preventDefault();

  try {
    const data = await api('/api/workers', {
      method: 'POST',
      body: JSON.stringify({
        name: $('workerName').value,
        url: $('workerUrl').value
      })
    });

    $('workerTokenDisplay').textContent = data.workerToken;
    $('workerTokenBox').classList.remove('hidden');

    const serverUrl = window.location.origin;
    $('colabCommand').textContent =
      `python worker.py --server-url ${serverUrl} --worker-token ${data.workerToken}`;

    setMessage($('hfTokenMessage'), 'Machine créée. Lance le worker sur ta machine GPU.', 'success');
    await refreshWorkers();
  } catch (error) {
    setMessage($('hfTokenMessage'), error.message, 'error');
  }
});

$('copyWorkerTokenButton').addEventListener('click', async () => {
  const token = $('workerTokenDisplay').textContent;

  try {
    await navigator.clipboard.writeText(token);
    setMessage($('hfTokenMessage'), 'Worker token copié.', 'success');
  } catch {
    setMessage($('hfTokenMessage'), 'Copie impossible. Sélectionne le token manuellement.', 'error');
  }
});

$('saveHfTokenButton').addEventListener('click', async () => {
  try {
    const data = await api('/api/hf/token', {
      method: 'POST',
      body: JSON.stringify({ token: $('hfTokenInput').value })
    });

    $('hfTokenInput').value = '';
    setMessage($('hfTokenMessage'), data.hfTokenConfigured ? 'Token HF enregistré.' : 'Token HF supprimé.', 'success');
  } catch (error) {
    setMessage($('hfTokenMessage'), error.message, 'error');
  }
});

$('clearHfTokenButton').addEventListener('click', async () => {
  try {
    const data = await api('/api/hf/token', {
      method: 'POST',
      body: JSON.stringify({ token: '' })
    });

    setMessage($('hfTokenMessage'), data.hfTokenConfigured ? 'Token HF enregistré.' : 'Token HF supprimé.', 'success');
  } catch (error) {
    setMessage($('hfTokenMessage'), error.message, 'error');
  }
});

$('hfSearchForm').addEventListener('submit', async (event) => {
  event.preventDefault();

  try {
    setMessage($('hfSearchMessage'), 'Recherche...');
    const type = $('hfSearchType').value;
    const query = $('hfSearchQuery').value;
    const data = await api(`/api/hf/search?type=${encodeURIComponent(type)}&q=${encodeURIComponent(query)}&limit=20`);
    renderResults(data.items, data.type);
    setMessage($('hfSearchMessage'), `${data.items.length} résultat(s).`, 'success');
  } catch (error) {
    setMessage($('hfSearchMessage'), error.message, 'error');
  }
});

$('startJobButton').addEventListener('click', async () => {
  try {
    $('startJobButton').disabled = true;
    setMessage($('jobMessage'), 'Envoi du job...');

    const workerId = $('jobWorkerId').value;
    const worker = state.workers.find((item) => item.id === workerId);

    if (!worker) {
      throw new Error('Choisis une machine GPU valide.');
    }

    if (worker.status === 'busy' || worker.status === 'offline') {
      throw new Error('Cette machine n’est pas disponible.');
    }

    const data = await api('/api/jobs', {
      method: 'POST',
      body: JSON.stringify({
        workerId,
        model: $('jobModel').value,
        dataset: $('jobDataset').value,
        task: $('jobTask').value,
        outputRepo: $('jobOutputRepo').value,
        pushToHf: $('jobPushToHf').checked,
        params: {
          epochs: $('jobEpochs').value,
          batch_size: $('jobBatchSize').value,
          learning_rate: $('jobLearningRate').value,
          max_seq_length: $('jobMaxSeqLength').value,
          lora_rank: $('jobLoraRank').value,
          lora_alpha: $('jobLoraAlpha').value,
          gradient_accumulation_steps: $('jobGradientAccumulationSteps').value,
          warmup_ratio: $('jobWarmupRatio').value,
          weight_decay: $('jobWeightDecay').value,
          logging_steps: $('jobLoggingSteps').value
        }
      })
    });

    state.selectedJobId = data.job.id;
    appendLog({
      jobId: data.job.id,
      level: 'info',
      message: `Job créé: ${data.job.id}`
    });

    await refreshJobs();
    await loadLogs(data.job.id, true);
    setMessage($('jobMessage'), 'Job envoyé au worker GPU.', 'success');
  } catch (error) {
    setMessage($('jobMessage'), error.message, 'error');
  } finally {
    $('startJobButton').disabled = false;
  }
});

$('refreshWorkersButton').addEventListener('click', async () => {
  try {
    await refreshWorkers();
    setMessage($('hfTokenMessage'), 'Machines rafraîchies.', 'success');
  } catch (error) {
    setMessage($('hfTokenMessage'), error.message, 'error');
  }
});

$('refreshJobsButton').addEventListener('click', async () => {
  try {
    await refreshJobs();
    setMessage($('jobMessage'), 'Jobs rafraîchis.', 'success');
  } catch (error) {
    setMessage($('jobMessage'), error.message, 'error');
  }
});

$('clearLogsButton').addEventListener('click', () => {
  state.logs = [];
  state.loadedLogIds.clear();
  $('logsOutput').textContent = '';
});

init();