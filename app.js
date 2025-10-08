// app.js — Async RunPod variant (uses /run + /status polling)
document.addEventListener('DOMContentLoaded', () => {
  // -------------------- Globals --------------------
  let photoData = [null, null, null];
  let qualityResults = [null, null, null];
  window.photoData = photoData;
  window.qualityResults = qualityResults;

  // -------------------- Helpers --------------------
  const asArray = (v) => (Array.isArray(v) ? v : []);
  const normalizeEndpoint = (url) => (url || '').trim().replace(/\/+$/, '');

  function getCreds() {
    return {
      apiEndpoint: normalizeEndpoint(document.getElementById('apiEndpoint')?.value),
      apiKey: (document.getElementById('apiKey')?.value || '').trim(),
    };
  }

  // Derive /run and /status routes from whatever the user typed
  function deriveRoutes(endpoint) {
    if (!endpoint) return {};
    let base = endpoint;
    base = base.replace(/\/status\/?.*$/i, '');
    base = base.replace(/\/runsync$/i, '');
    base = base.replace(/\/run$/i, '');
    base = base.replace(/\/+$/, '');
    const runUrl = `${base}/run`;
    const statusUrl = `${base}/status`;
    return { runUrl, statusUrl };
  }

  function parseRunpodResponse(raw) {
    return raw?.output?.[0]?.output ?? raw?.output ?? raw;
  }

  function extractJobId(raw) {
    return raw?.id || raw?.jobId || raw?.requestId || raw?.output?.id || raw?.[0]?.id || null;
  }

  const wait = (ms) => new Promise((r) => setTimeout(r, ms));

  async function pollStatus(statusUrl, jobId, apiKey, opts = {}) {
    const {
      timeoutMs = 180000,
      startDelayMs = 800,
      maxDelayMs = 3000,
    } = opts;

    let delay = startDelayMs;
    const started = Date.now();

    while (true) {
      if (Date.now() - started > timeoutMs) {
        throw new Error('Polling timeout exceeded');
      }

      const res = await fetch(`${statusUrl}/${encodeURIComponent(jobId)}`, {
        method: 'GET',
        headers: { Authorization: `Bearer ${apiKey}` },
      });

      if (!res.ok) {
        const body = await res.text().catch(() => '');
        throw new Error(`Status HTTP ${res.status} ${res.statusText}: ${body}`);
      }

      const json = await res.json();
      const status = json?.status || json?.state || json?.job?.status;
      if (status === 'COMPLETED' || status === 'SUCCEEDED' || status === 'SUCCESS') {
        return parseRunpodResponse(json);
      }
      if (status === 'FAILED' || status === 'CANCELLED' || status === 'CANCELED' || status === 'ERROR') {
        const msg = json?.error || json?.message || 'Job failed';
        throw new Error(msg);
      }

      await wait(delay);
      delay = Math.min(maxDelayMs, Math.round(delay * 1.5));
    }
  }

  function estimateBase64KB(dataUrl) {
    const size = Math.ceil((dataUrl.length - dataUrl.indexOf(',') - 1) * 3 / 4);
    return Math.round(size / 1024);
  }

  function updateGenerateButton() {
    const allPhotosUploaded = photoData.every((p) => p !== null);
    const apiEndpoint = document.getElementById('apiEndpoint')?.value;
    const apiKey = document.getElementById('apiKey')?.value;
    const btn = document.getElementById('generateBtn');
    if (btn) btn.disabled = !allPhotosUploaded || !apiEndpoint || !apiKey;
  }

  function fileToResizedDataURL(file, maxSide = 1280, mime = 'image/jpeg', quality = 0.85) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      const reader = new FileReader();
      reader.onload = (e) => {
        img.onload = () => {
          const canvas = document.createElement('canvas');
          let { width, height } = img;
          const scale = Math.min(1, maxSide / Math.max(width, height));
          width = Math.round(width * scale);
          height = Math.round(height * scale);
          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(img, 0, 0, width, height);
          resolve(canvas.toDataURL(mime, quality));
        };
        img.onerror = reject;
        img.src = e.target.result;
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  function clearQualityUI() {
    for (let i = 1; i <= 3; i++) {
      const card = document.getElementById(`card${i}`);
      const preview = document.getElementById(`preview${i}`);
      if (!card || !preview) continue;
      card.classList.remove('quality-bad');
      const img = preview.querySelector('img.preview-image');
      preview.innerHTML = '';
      if (img) preview.appendChild(img);
    }
  }

  // -------------------- File Inputs --------------------
  for (let i = 1; i <= 3; i++) {
    const input = document.getElementById(`photo${i}`);
    const preview = document.getElementById(`preview${i}`);
    const card = document.getElementById(`card${i}`);
    if (!input || !preview || !card) continue;

    input.addEventListener('change', async (e) => {
      const file = e.target.files?.[0];
      if (!file) return;

      try {
        const dataUrl = await fileToResizedDataURL(file);
        photoData[i - 1] = dataUrl;

        const kb = estimateBase64KB(dataUrl);
        preview.innerHTML = `
          <img src="${dataUrl}" class="preview-image">
          <div>✓ Photo uploaded (${kb} KB)</div>
        `;
        card.classList.add('has-image');
        updateGenerateButton();
      } catch (err) {
        preview.innerHTML = `<div style="color:#dc3545">Upload failed: ${err?.message || err}</div>`;
      }
    });
  }

  document.getElementById('apiEndpoint')?.addEventListener('input', updateGenerateButton);
  document.getElementById('apiKey')?.addEventListener('input', updateGenerateButton);

  // -------------------- Result UI + Viewer --------------------
  function showResult(type, message) {
    const section = document.getElementById('resultSection');
    const content = document.getElementById('resultContent');
    if (!section || !content) return;

    section.style.display = 'block';
    content.innerHTML =
      type === 'success'
        ? `<div class="success-message">${message}</div>`
        : `<div class="error-message">${message}</div>`;
    section.scrollIntoView({ behavior: 'smooth' });
  }
  window.showResult = showResult;

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = src;
      s.onload = resolve;
      s.onerror = reject;
      document.head.appendChild(s);
    });
  }
  async function ensureThree() {
    if (window.THREE && window.THREE.OBJLoader) return;
    if (!window.THREE) {
      await loadScript('https://unpkg.com/three@0.158.0/build/three.min.js');
    }
    if (!window.THREE?.OBJLoader) {
      await loadScript('https://unpkg.com/three@0.158.0/examples/js/loaders/OBJLoader.js');
    }
  }
  function showObjInViewer(objText) {
    const container = document.getElementById('viewer');
    if (!container) return;
    const w = container.clientWidth || container.parentElement.clientWidth || 800;
    const h = 420;

    const init = () => {
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 1000);
      const renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setSize(w, h);
      container.innerHTML = '';
      container.appendChild(renderer.domElement);

      scene.add(new THREE.AmbientLight(0xffffff, 0.8));
      const dir = new THREE.DirectionalLight(0xffffff, 0.7);
      dir.position.set(1, 1, 1);
      scene.add(dir);

      const loader = new THREE.OBJLoader();
      const obj = loader.parse(objText);
      obj.traverse((c) => {
        if (c.isMesh) {
          c.material = new THREE.MeshPhongMaterial({ color: 0xeeeeee });
        }
      });
      scene.add(obj);

      obj.rotation.x = -Math.PI / 2;
      obj.position.set(0, -0.5, 0);
      camera.position.set(0, 0.5, 2.5);

      function animate() {
        requestAnimationFrame(animate);
        obj.rotation.y += 0.01;
        renderer.render(scene, camera);
      }
      animate();

      window.addEventListener('resize', () => {
        const w2 = container.clientWidth || container.parentElement.clientWidth || w;
        renderer.setSize(w2, h);
        camera.aspect = w2 / h;
        camera.updateProjectionMatrix();
      });
    };

    if (window.THREE && window.THREE.OBJLoader) {
      init();
    } else {
      ensureThree().then(init).catch((e) => {
        console.warn('Three.js load failed:', e);
      });
    }
  }
  function downloadMesh(base64Data) {
    const objData = atob(base64Data);
    showObjInViewer(objData);
    const blob = new Blob([objData], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'digital_twin.obj';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
  window.downloadMesh = downloadMesh;

  function showSuccessResult(result) {
    const section = document.getElementById('resultSection');
    const content = document.getElementById('resultContent');
    if (!section || !content) return;

    section.style.display = 'block';
    const stats = result.statistics || {};
    content.innerHTML = `
      <div class="success-message">
        <strong>✓ Digital Twin Generated Successfully!</strong>
      </div>
      <div class="stats-grid">
        <div class="stat-item"><div class="stat-value">${stats.num_vertices ?? '-'}</div><div class="stat-label">Vertices</div></div>
        <div class="stat-item"><div class="stat-value">${stats.num_faces ?? '-'}</div><div class="stat-label">Faces</div></div>
        <div class="stat-item"><div class="stat-value">${stats.num_matches ?? '-'}</div><div class="stat-label">Matches</div></div>
      </div>
      <div id="viewer" style="margin-top:20px;height:420px;background:#000;border-radius:10px;"></div>
      <div style="margin-top:20px;">
        <button onclick="downloadMesh('${result.mesh_obj_base64}')" 
          style="width:100%;padding:15px;background:#28a745;color:white;border:none;border-radius:10px;font-size:16px;cursor:pointer;">
          📥 Download OBJ File
        </button>
      </div>
    `;
    section.scrollIntoView({ behavior: 'smooth' });

    try {
      const objText = atob(result.mesh_obj_base64);
      showObjInViewer(objText);
    } catch (e) {
      console.warn('OBJ preview failed:', e);
    }
  }
  window.showSuccessResult = showSuccessResult;

  // -------------------- Analyze (Async) --------------------
  const analyzeBtn = document.getElementById('analyzeBtn');
  analyzeBtn?.addEventListener('click', async () => {
    const { apiEndpoint, apiKey } = getCreds();
    if (!/^https?:\/\//i.test(apiEndpoint) || !apiKey) {
      alert('Please enter a valid RunPod API endpoint (e.g., https://api.runpod.ai/v2/ENDPOINT_ID) and API key');
      return;
    }
    const routes = deriveRoutes(apiEndpoint);
    if (!routes.runUrl || !routes.statusUrl) {
      alert('Could not derive /run and /status routes from the endpoint.');
      return;
    }

    const anyPhoto = photoData.some((p) => p !== null);
    if (!anyPhoto) {
      alert('Please upload at least one photo');
      return;
    }

    analyzeBtn.innerHTML = '<div class="spinner"></div> Queuing & analyzing...';
    analyzeBtn.disabled = true;
    clearQualityUI();

    try {
      for (let i = 0; i < 3; i++) {
        if (!photoData[i]) continue;

        // submit
        const submit = await fetch(routes.runUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${apiKey}`,
          },
          body: JSON.stringify({ input: { action: 'analyze_photo', photo: photoData[i] } }),
        });

        if (!submit.ok) {
          const txt = await submit.text().catch(() => '');
          throw new Error(`Run HTTP ${submit.status} ${submit.statusText}: ${txt}`);
        }

        const job = await submit.json();
        const jobId = extractJobId(job);
        if (!jobId) throw new Error('No job id returned from /run');

        // poll
        const final = await pollStatus(routes.statusUrl, jobId, apiKey);
        const result = parseRunpodResponse(final);

        const card = document.getElementById(`card${i + 1}`);
        const preview = document.getElementById(`preview${i + 1}`);
        if (!card || !preview) continue;

        if (result?.error || typeof result?.is_good === 'undefined') {
          card.classList.add('quality-bad');
          const msg = result?.error || 'Unexpected response format.';
          preview.innerHTML += `
            <span class="quality-badge quality-bad">✗ Error</span>
            <div class="quality-issues">${msg}</div>
          `;
          continue;
        }

        const issues = asArray(result.issues);
        if (result.is_good) {
          card.classList.remove('quality-bad');
          preview.innerHTML += `<span class="quality-badge quality-good">✓ Good Quality</span>`;
        } else {
          card.classList.add('quality-bad');
          const issuesHtml = issues.map((x) => `• ${x}`).join('<br>');
          preview.innerHTML += `
            <span class="quality-badge quality-bad">✗ Quality Issues</span>
            <div class="quality-issues">${issuesHtml}</div>
          `;
        }
      }

      showResult('success', 'Photo quality analysis complete!');
    } catch (err) {
      showResult('error', `Error analyzing photos: ${err.message}`);
    } finally {
      analyzeBtn.innerHTML = '🔍 Analyze Photo Quality';
      analyzeBtn.disabled = false;
    }
  });

  // -------------------- Generate (Async) --------------------
  const generateBtn = document.getElementById('generateBtn');
  generateBtn?.addEventListener('click', async () => {
    const { apiEndpoint, apiKey } = getCreds();
    if (!/^https?:\/\//i.test(apiEndpoint) || !apiKey) {
      alert('Please enter a valid RunPod API endpoint (e.g., https://api.runpod.ai/v2/ENDPOINT_ID) and API key');
      return;
    }
    const routes = deriveRoutes(apiEndpoint);
    if (!routes.runUrl || !routes.statusUrl) {
      alert('Could not derive /run and /status routes from the endpoint.');
      return;
    }

    const measurements = {
      height: parseInt(document.getElementById('height')?.value),
      weight: parseInt(document.getElementById('weight')?.value),
      shoulderWidth: parseInt(document.getElementById('shoulder')?.value),
      chestCircumference: parseInt(document.getElementById('chest')?.value),
      waistCircumference: parseInt(document.getElementById('waist')?.value),
      hipCircumference: parseInt(document.getElementById('hips')?.value),
    };

    generateBtn.innerHTML = '<div class="spinner"></div> Queued… generating digital twin...';
    generateBtn.disabled = true;

    try {
      // submit
      const submit = await fetch(routes.runUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          input: { action: 'generate_twin', photos: photoData, measurements },
        }),
      });

      if (!submit.ok) {
        const txt = await submit.text().catch(() => '');
        throw new Error(`Run HTTP ${submit.status} ${submit.statusText}: ${txt}`);
      }

      const job = await submit.json();
      const jobId = extractJobId(job);
      if (!jobId) throw new Error('No job id returned from /run');

      // poll
      const final = await pollStatus(routes.statusUrl, jobId, apiKey, { timeoutMs: 300000 }); // allow up to 5 min
      const result = parseRunpodResponse(final);

      if (result?.success) {
        showSuccessResult(result);
      } else {
        const message = result?.error || 'Failed to generate digital twin.';
        showResult('error', message);
      }
    } catch (err) {
      showResult('error', `Error: ${err.message}`);
    } finally {
      generateBtn.innerHTML = '🎯 Generate Digital Twin';
      generateBtn.disabled = false;
    }
  });

  // -------------------- Kick things off --------------------
  updateGenerateButton();
});
