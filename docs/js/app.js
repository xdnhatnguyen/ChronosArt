/**
 * ZooSketch - Living Wildlife in Training-Free Style Personalization
 * Clean Academic Interface inspired by Roblox/PixVid project page
 */

document.addEventListener('DOMContentLoaded', async () => {
  let animalsData = [];
  let currentTheme = 'sketch'; // 'sketch' | 'origami'
  let currentIndex = 0;
  let currentSeedIndex = 0;
  let isLivingActive = true;
  let isTransparent = true;
  let animationInterval = null;

  // 1. Fetch Animals Dual Data
  try {
    const res = await fetch('data/animals_dual.json');
    animalsData = await res.json();
  } catch (err) {
    console.error('Failed to load animals_dual.json', err);
    return;
  }

  // DOM Elements
  const stage = document.getElementById('animal-stage');
  const superWord = document.getElementById('super-backdrop-word');
  const displayName = document.getElementById('animal-display-name');
  const scientificName = document.getElementById('animal-scientific-name');
  const promptBadge = document.getElementById('animal-prompt-badge');
  const counterDisplay = document.getElementById('counter-display');
  const thumbsStrip = document.getElementById('thumbs-strip');
  const galleryGrid = document.getElementById('gallery-grid');

  const btnPrev = document.getElementById('btn-prev');
  const btnNext = document.getElementById('btn-next');
  const btnToggleLive = document.getElementById('btn-toggle-living');
  const btnToggleBg = document.getElementById('btn-toggle-bg');
  const btnThemeSketch = document.getElementById('btn-theme-sketch');
  const btnThemeOrigami = document.getElementById('btn-theme-origami');
  const copyBibBtn = document.getElementById('copy-bib-btn');

  // =========================================================================
  // 2. SUBTLE SPARKLE CANVAS (Hạt lấp lánh nhẹ nhàng, tinh tế)
  // =========================================================================
  const canvas = document.getElementById('particle-canvas');
  const ctx = canvas.getContext('2d');
  let particles = [];
  let width = (canvas.width = window.innerWidth);
  let height = (canvas.height = window.innerHeight);

  window.addEventListener('resize', () => {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  });

  class Particle {
    constructor(x, y, theme) {
      this.x = x;
      this.y = y;
      const angle = Math.random() * Math.PI * 2;
      const speed = Math.random() * 1.8 + 0.5;
      this.vx = Math.cos(angle) * speed;
      this.vy = Math.sin(angle) * speed - 0.3;
      this.size = Math.random() * 2.5 + 1.0;
      this.maxLife = Math.random() * 35 + 20;
      this.life = this.maxLife;

      if (theme === 'origami') {
        const colors = ['#f43f5e', '#38bdf8', '#34d399', '#fbbf24', '#a855f7'];
        this.color = colors[Math.floor(Math.random() * colors.length)];
      } else {
        const colors = ['#94a3b8', '#cbd5e1', '#e2e8f0', '#64748b'];
        this.color = colors[Math.floor(Math.random() * colors.length)];
      }
    }

    update() {
      this.x += this.vx;
      this.y += this.vy;
      this.vx *= 0.95;
      this.vy *= 0.95;
      this.life--;
    }

    draw(ctx) {
      const progress = this.life / this.maxLife;
      ctx.save();
      ctx.globalAlpha = progress * 0.7;
      ctx.fillStyle = this.color;
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.size * progress, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  }

  function emitParticles(x, y, count = 2) {
    for (let i = 0; i < count; i++) {
      particles.push(new Particle(x + (Math.random() * 8 - 4), y + (Math.random() * 8 - 4), currentTheme));
    }
  }

  window.addEventListener('mousemove', (e) => {
    emitParticles(e.clientX, e.clientY, 1);
  });

  if (stage) {
    stage.addEventListener('mousemove', (e) => {
      emitParticles(e.clientX, e.clientY, 3);
    });
  }

  function renderParticles() {
    ctx.clearRect(0, 0, width, height);
    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i];
      p.update();
      p.draw(ctx);
      if (p.life <= 0) particles.splice(i, 1);
    }
    requestAnimationFrame(renderParticles);
  }
  renderParticles();

  // =========================================================================
  // 3. ANIMAL ASSETS HELPERS
  // =========================================================================
  function getAnimalFrames(animal) {
    if (!animal || !animal[currentTheme]) return [];
    const styleObj = animal[currentTheme];
    return isTransparent ? styleObj.nobg_webp : styleObj.full_webp;
  }

  function renderStage(index) {
    if (animalsData.length === 0) return;
    const animal = animalsData[index];
    if (!animal) return;

    if (counterDisplay) {
      counterDisplay.textContent = `${String(index + 1).padStart(2, '0')} / ${String(animalsData.length).padStart(2, '0')}`;
    }

    if (superWord) {
      superWord.textContent = animal.id.toUpperCase();
    }

    if (displayName) displayName.textContent = animal.common_name;
    if (scientificName) scientificName.textContent = animal.scientific_name;

    const currentPrompt = animal[currentTheme]?.prompt || '';
    if (promptBadge) {
      promptBadge.textContent = `Prompt: "${currentPrompt}"`;
    }

    if (stage) {
      stage.innerHTML = '';
      const frames = getAnimalFrames(animal);

      frames.forEach((file, idx) => {
        const img = document.createElement('img');
        img.src = file;
        img.alt = `${animal.common_name} ${currentTheme} frame ${idx}`;
        img.className = `stage-frame-img ${idx === currentSeedIndex ? 'active' : ''}`;
        stage.appendChild(img);
      });
    }

    // Highlight active thumbnail
    document.querySelectorAll('.thumb-item').forEach((item, idx) => {
      item.classList.toggle('active', idx === index);
    });

    startLivingLoop();
  }

  function startLivingLoop() {
    if (animationInterval) {
      clearInterval(animationInterval);
      animationInterval = null;
    }

    if (isLivingActive && animalsData.length > 0) {
      animationInterval = setInterval(() => {
        const animal = animalsData[currentIndex];
        const frames = getAnimalFrames(animal);
        if (frames.length <= 1) return;

        currentSeedIndex = (currentSeedIndex + 1) % frames.length;
        const allFrames = stage.querySelectorAll('.stage-frame-img');
        allFrames.forEach((frame, idx) => {
          frame.classList.toggle('active', idx === currentSeedIndex);
        });
      }, 260);
    }
  }

  // =========================================================================
  // 4. THUMBNAILS STRIP (10 ANIMALS)
  // =========================================================================
  function renderThumbnailStrip() {
    if (!thumbsStrip) return;
    thumbsStrip.innerHTML = '';

    animalsData.forEach((anim, idx) => {
      const frames = getAnimalFrames(anim);
      const thumbUrl = frames.length > 0 ? frames[0] : '';

      const thumb = document.createElement('div');
      thumb.className = `thumb-item ${idx === currentIndex ? 'active' : ''}`;
      thumb.title = anim.common_name;
      thumb.innerHTML = `<img src="${thumbUrl}" alt="${anim.common_name}">`;

      thumb.addEventListener('click', () => {
        currentIndex = idx;
        currentSeedIndex = 0;
        renderStage(currentIndex);
      });

      thumbsStrip.appendChild(thumb);
    });
  }

  // =========================================================================
  // 5. GALLERY GRID (CLEAN ACADEMIC CARDS)
  // =========================================================================
  function renderGallery() {
    if (!galleryGrid) return;
    galleryGrid.innerHTML = '';

    animalsData.forEach((anim, idx) => {
      const frames = getAnimalFrames(anim);
      const firstImg = frames.length > 0 ? frames[0] : '';

      const card = document.createElement('div');
      card.className = 'gallery-card';
      card.innerHTML = `
        <div class="card-img-wrap">
          <img src="${firstImg}" alt="${anim.common_name}">
        </div>
        <div class="card-meta">
          <div class="card-name">${anim.common_name}</div>
          <div class="card-latin">${anim.scientific_name}</div>
        </div>
      `;

      card.addEventListener('click', () => {
        currentIndex = idx;
        currentSeedIndex = 0;
        renderStage(currentIndex);
        document.getElementById('stage').scrollIntoView({ behavior: 'smooth' });
      });

      // Hover living preview
      let hoverInterval = null;
      let hoverIdx = 0;
      card.addEventListener('mouseenter', () => {
        const imgEl = card.querySelector('img');
        if (frames.length > 1) {
          hoverInterval = setInterval(() => {
            hoverIdx = (hoverIdx + 1) % frames.length;
            imgEl.src = frames[hoverIdx];
          }, 220);
        }
      });
      card.addEventListener('mouseleave', () => {
        if (hoverInterval) clearInterval(hoverInterval);
        card.querySelector('img').src = firstImg;
      });

      galleryGrid.appendChild(card);
    });
  }

  // =========================================================================
  // 6. NAVIGATION & CONTROLS HANDLERS
  // =========================================================================
  if (btnPrev) {
    btnPrev.addEventListener('click', () => {
      currentIndex = (currentIndex - 1 + animalsData.length) % animalsData.length;
      currentSeedIndex = 0;
      renderStage(currentIndex);
    });
  }

  if (btnNext) {
    btnNext.addEventListener('click', () => {
      currentIndex = (currentIndex + 1) % animalsData.length;
      currentSeedIndex = 0;
      renderStage(currentIndex);
    });
  }

  window.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight') btnNext && btnNext.click();
    else if (e.key === 'ArrowLeft') btnPrev && btnPrev.click();
  });

  if (btnToggleLive) {
    btnToggleLive.addEventListener('click', () => {
      isLivingActive = !isLivingActive;
      btnToggleLive.classList.toggle('active', isLivingActive);
      btnToggleLive.querySelector('.status-text').textContent = isLivingActive ? 'Living Motion' : 'Paused';
      startLivingLoop();
    });
  }

  if (btnToggleBg) {
    btnToggleBg.addEventListener('click', () => {
      isTransparent = !isTransparent;
      btnToggleBg.classList.toggle('active', isTransparent);
      btnToggleBg.querySelector('.status-text').textContent = isTransparent ? 'Transparent (SAM)' : 'Full Canvas';
      renderStage(currentIndex);
      renderThumbnailStrip();
      renderGallery();
    });
  }

  // Theme Switcher: Sketch vs Origami
  function setTheme(theme) {
    currentTheme = theme;
    if (btnThemeSketch) btnThemeSketch.classList.toggle('active', theme === 'sketch');
    if (btnThemeOrigami) btnThemeOrigami.classList.toggle('active', theme === 'origami');

    renderStage(currentIndex);
    renderThumbnailStrip();
    renderGallery();
  }

  if (btnThemeSketch) {
    btnThemeSketch.addEventListener('click', () => setTheme('sketch'));
  }
  if (btnThemeOrigami) {
    btnThemeOrigami.addEventListener('click', () => setTheme('origami'));
  }

  // Copy BibTeX
  if (copyBibBtn) {
    copyBibBtn.addEventListener('click', () => {
      const code = document.getElementById('bib-code').innerText;
      navigator.clipboard.writeText(code).then(() => {
        copyBibBtn.textContent = 'Copied!';
        setTimeout(() => {
          copyBibBtn.textContent = 'Copy';
        }, 2000);
      });
    });
  }

  // Initialize
  renderStage(0);
  renderThumbnailStrip();
  renderGallery();
});
