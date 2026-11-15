/**
 * ChronosArt - Living Wildlife Camera Roll & Stochastic Artistry
 * Interactive Brainstorm / Camera Roll Whiteboard inspired by Camroll VQA
 */

document.addEventListener('DOMContentLoaded', async () => {
  let animalsData = [];
  let currentTheme = 'sketch'; // 'sketch' | 'origami'
  let currentModalIndex = 0;
  let currentSeedIndex = 0;
  let isModalJumping = true;
  let isCutout = true;
  let jumpSpeed = 230; // ms per frame
  let modalInterval = null;

  // 1. Fetch Animals Dual Data
  try {
    const res = await fetch('data/animals_dual.json');
    animalsData = await res.json();
  } catch (err) {
    console.error('Failed to load animals_dual.json', err);
    return;
  }

  // DOM Elements
  const boardViewport = document.getElementById('board-viewport');
  const boardCanvas = document.getElementById('board-canvas');
  const cameraRollGrid = document.getElementById('camera-roll-grid');
  const heroAnchor = document.getElementById('hero-anchor');

  const btnThemeSketch = document.getElementById('btn-theme-sketch');
  const btnThemeOrigami = document.getElementById('btn-theme-origami');
  const btnShuffle = document.getElementById('btn-shuffle');

  // Modal Elements
  const spotlightOverlay = document.getElementById('spotlight-overlay');
  const spotlightCloseBtn = document.getElementById('spotlight-close-btn');
  const spotlightBackdropWord = document.getElementById('spotlight-backdrop-word');
  const spotlightFramesWrap = document.getElementById('spotlight-frames-wrap');
  const btnModalTogglePlay = document.getElementById('btn-modal-toggle-play');
  const playPauseIcon = document.getElementById('play-pause-icon');
  const playPauseText = document.getElementById('play-pause-text');
  const btnModalToggleCutout = document.getElementById('btn-modal-toggle-cutout');
  const speedButtons = document.querySelectorAll('.speed-btn');

  const modalCategory = document.getElementById('modal-category');
  const modalTitle = document.getElementById('modal-title');
  const modalLatin = document.getElementById('modal-latin');
  const modalLore = document.getElementById('modal-lore');
  const modalStatsGrid = document.getElementById('modal-stats-grid');
  const modalCounter = document.getElementById('modal-counter');
  const seedLiveTag = document.getElementById('seed-live-tag');
  const seedButtons = document.querySelectorAll('.seed-btn');
  const modalBtnPrev = document.getElementById('modal-btn-prev');
  const modalBtnNext = document.getElementById('modal-btn-next');

  // =========================================================================
  // 2. SUBTLE SPARKLE CANVAS (Hạt lấp lánh nhẹ nhàng khi di chuột)
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
      const speed = Math.random() * 1.6 + 0.4;
      this.vx = Math.cos(angle) * speed;
      this.vy = Math.sin(angle) * speed - 0.2;
      this.size = Math.random() * 2.2 + 0.8;
      this.maxLife = Math.random() * 30 + 18;
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
      ctx.globalAlpha = progress * 0.65;
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
  // 3. WHITEBOARD PANNING & DRAGGING (Smooth Infinite Board Feel)
  // =========================================================================
  let panX = 0;
  let panY = 0;
  let isDragging = false;
  let startX = 0;
  let startY = 0;
  let hasDragged = false;

  function updateBoardTransform() {
    boardCanvas.style.transform = `translate(${panX}px, ${panY}px)`;
  }

  // Initial Center on Hero Anchor
  function centerHero() {
    const vpW = boardViewport.clientWidth;
    const vpH = boardViewport.clientHeight;
    // Hero center: (650 + 350 = 1000, 400 + 180 = 580)
    panX = Math.round(vpW / 2 - 1000);
    panY = Math.round(vpH / 2 - 580);
    updateBoardTransform();
  }
  centerHero();
  window.addEventListener('resize', centerHero);

  boardViewport.addEventListener('mousedown', (e) => {
    // Only drag when clicking background or canvas (not buttons or notes)
    if (e.target.closest('button') || e.target.closest('a') || e.target.closest('.spotlight-overlay')) {
      return;
    }
    isDragging = true;
    hasDragged = false;
    startX = e.clientX - panX;
    startY = e.clientY - panY;
    boardViewport.classList.add('is-dragging');
  });

  window.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    const newPanX = e.clientX - startX;
    const newPanY = e.clientY - startY;
    if (Math.abs(newPanX - panX) > 4 || Math.abs(newPanY - panY) > 4) {
      hasDragged = true;
    }
    panX = newPanX;
    panY = newPanY;
    updateBoardTransform();
  });

  window.addEventListener('mouseup', () => {
    if (isDragging) {
      isDragging = false;
      boardViewport.classList.remove('is-dragging');
    }
  });

  // Mouse wheel pan
  boardViewport.addEventListener('wheel', (e) => {
    if (spotlightOverlay.classList.contains('active')) return;
    e.preventDefault();
    panX -= e.deltaX * 0.8;
    panY -= e.deltaY * 0.8;
    updateBoardTransform();
  }, { passive: false });

  // Touch Support for Mobile / Tablet
  let touchStartX = 0;
  let touchStartY = 0;
  boardViewport.addEventListener('touchstart', (e) => {
    if (e.touches.length === 1) {
      isDragging = true;
      hasDragged = false;
      touchStartX = e.touches[0].clientX - panX;
      touchStartY = e.touches[0].clientY - panY;
    }
  }, { passive: true });

  window.addEventListener('touchmove', (e) => {
    if (!isDragging || e.touches.length !== 1) return;
    const newPanX = e.touches[0].clientX - touchStartX;
    const newPanY = e.touches[0].clientY - touchStartY;
    if (Math.abs(newPanX - panX) > 4 || Math.abs(newPanY - panY) > 4) {
      hasDragged = true;
    }
    panX = newPanX;
    panY = newPanY;
    updateBoardTransform();
  }, { passive: true });

  window.addEventListener('touchend', () => {
    isDragging = false;
  });

  // =========================================================================
  // 4. SCATTERED CAMERA ROLL POSITIONS (Around Central Hero)
  // =========================================================================
  const PASTEL_THEMES = ['mint', 'peach', 'rose', 'sky', 'lemon'];
  const SIGNATURES = [
    '+ wild.chronos',
    '+ minh-nhat.fit',
    '+ stochastic.eye',
    '+ ortho.sketch',
    '+ paper.crease',
    '+ arctic.trail',
    '+ savanna.chronicle',
    '+ forest.whisper',
    '+ predatory.brush',
    '+ wildlife.canvas'
  ];

  // Base coordinates around the central hero (Hero is at 650, 400; size 700x360)
  const BASE_POSITIONS = [
    { top: 70,  left: 140, rotStack: -4, rotNote: 2, notePos: 'right' },  // 0: Elephant
    { top: 60,  left: 780, rotStack: 3,  rotNote: -2, notePos: 'right' }, // 1: Eagle
    { top: 80,  left: 1460, rotStack: -3, rotNote: 3, notePos: 'left' },  // 2: Lion
    { top: 480, left: 80,  rotStack: 4,  rotNote: -3, notePos: 'right' }, // 3: Tiger
    { top: 480, left: 1480, rotStack: -5, rotNote: 2, notePos: 'left' },  // 4: Wolf
    { top: 880, left: 130, rotStack: -2, rotNote: 4, notePos: 'right' },  // 5: Panda
    { top: 860, left: 1440, rotStack: 4,  rotNote: -2, notePos: 'left' },  // 6: Bear
    { top: 960, left: 820, rotStack: -4, rotNote: 2, notePos: 'right' },  // 7: Deer
    { top: 920, left: 470, rotStack: 3,  rotNote: -3, notePos: 'right' }, // 8: Fox
    { top: 90,  left: 1140, rotStack: -2, rotNote: 3, notePos: 'left' }   // 9: Owl
  ];

  function getAnimalFrames(animal, theme = currentTheme, cutout = isCutout) {
    if (!animal || !animal[theme]) return [];
    const styleObj = animal[theme];
    return cutout ? styleObj.nobg_webp : styleObj.full_webp;
  }

  // =========================================================================
  // 5. RENDER CAMERA ROLL ITEMS (Polaroid Stacks & Sticky Notes)
  // =========================================================================
  function renderCameraRoll() {
    if (!cameraRollGrid) return;
    cameraRollGrid.innerHTML = '';

    animalsData.forEach((anim, idx) => {
      const pos = BASE_POSITIONS[idx % BASE_POSITIONS.length];
      const pastelColor = PASTEL_THEMES[idx % PASTEL_THEMES.length];
      const authorSig = SIGNATURES[idx % SIGNATURES.length];
      const frames = getAnimalFrames(anim);
      const topImgSrc = frames.length > 0 ? frames[0] : '';

      const itemWrap = document.createElement('div');
      itemWrap.className = 'camera-roll-item';
      itemWrap.style.top = `${pos.top}px`;
      itemWrap.style.left = `${pos.left}px`;
      itemWrap.dataset.index = idx;

      // Polaroid Stack HTML
      const polaroidHtml = `
        <div class="polaroid-stack" style="transform: rotate(${pos.rotStack}deg);" data-index="${idx}">
          <div class="stack-layer stack-layer-back"></div>
          <div class="stack-layer stack-layer-mid"></div>
          <div class="polaroid-top">
            <div class="stack-badge">
              <span>📸</span> 3 SEEDS
            </div>
            <div class="polaroid-img-wrap">
              <img class="polaroid-img" src="${topImgSrc}" alt="${anim.common_name}">
            </div>
            <div class="polaroid-caption">
              <span class="polaroid-species">${anim.common_name}</span>
              <span class="polaroid-indicator">#${String(idx + 1).padStart(2, '0')}</span>
            </div>
          </div>
        </div>
      `;

      // Pastel Sticky Note HTML (No long prompt! Just lore & identity)
      const stickyHtml = `
        <div class="sticky-note ${pastelColor}" style="transform: rotate(${pos.rotNote}deg);" data-index="${idx}">
          <div class="sticky-tape"></div>
          <div class="sticky-category">${anim.category || 'Wildlife'}</div>
          <div class="sticky-name">${anim.common_name}</div>
          <div class="sticky-latin">${anim.scientific_name}</div>
          <p class="sticky-text">${anim.lore || ''}</p>
          <div class="sticky-author">${authorSig}</div>
        </div>
      `;

      // Order based on notePos
      if (pos.notePos === 'left') {
        itemWrap.innerHTML = stickyHtml + polaroidHtml;
      } else {
        itemWrap.innerHTML = polaroidHtml + stickyHtml;
      }

      // Micro-hover stochastic preview on polaroid
      const stackEl = itemWrap.querySelector('.polaroid-stack');
      const imgEl = itemWrap.querySelector('.polaroid-img');
      let hoverInterval = null;
      let hoverFrameIdx = 0;

      stackEl.addEventListener('mouseenter', () => {
        if (frames.length > 1) {
          hoverInterval = setInterval(() => {
            hoverFrameIdx = (hoverFrameIdx + 1) % frames.length;
            imgEl.src = frames[hoverFrameIdx];
          }, 240);
        }
      });

      stackEl.addEventListener('mouseleave', () => {
        if (hoverInterval) clearInterval(hoverInterval);
        imgEl.src = frames[0];
      });

      // Click to open Stochastic Spotlight
      itemWrap.addEventListener('click', (e) => {
        if (hasDragged) return; // Prevent triggering if user was panning the board
        openSpotlight(idx);
      });

      cameraRollGrid.appendChild(itemWrap);
    });
  }

  // Shuffle button slightly jiggles positions & rotations
  if (btnShuffle) {
    btnShuffle.addEventListener('click', () => {
      document.querySelectorAll('.camera-roll-item').forEach((item, idx) => {
        const base = BASE_POSITIONS[idx % BASE_POSITIONS.length];
        const dX = (Math.random() * 60 - 30);
        const dY = (Math.random() * 60 - 30);
        const dR = (Math.random() * 8 - 4);
        item.style.transform = `translate(${dX}px, ${dY}px)`;
        const stack = item.querySelector('.polaroid-stack');
        if (stack) stack.style.transform = `rotate(${base.rotStack + dR}deg)`;
      });
      emitParticles(window.innerWidth / 2, window.innerHeight / 2, 10);
    });
  }

  // =========================================================================
  // 6. SPOTLIGHT MODAL (Live Stochastic Jump & Inspector)
  // =========================================================================
  function openSpotlight(index) {
    currentModalIndex = index;
    currentSeedIndex = 0;
    isModalJumping = true;

    renderSpotlightContent();
    spotlightOverlay.classList.add('active');
    spotlightOverlay.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';

    startStochasticJump();
  }

  function closeSpotlight() {
    spotlightOverlay.classList.remove('active');
    spotlightOverlay.setAttribute('aria-hidden', 'true');
    document.body.style.overflow = '';
    stopStochasticJump();
  }

  function renderSpotlightContent() {
    const animal = animalsData[currentModalIndex];
    if (!animal) return;

    if (modalCategory) modalCategory.textContent = animal.category || 'Wildlife';
    if (modalTitle) modalTitle.textContent = animal.common_name;
    if (modalLatin) modalLatin.textContent = animal.scientific_name;
    if (modalLore) modalLore.textContent = animal.lore || '';
    if (spotlightBackdropWord) spotlightBackdropWord.textContent = animal.id.toUpperCase();
    if (modalCounter) {
      modalCounter.textContent = `${String(currentModalIndex + 1).padStart(2, '0')} / ${String(animalsData.length).padStart(2, '0')}`;
    }

    // Render Stats
    if (modalStatsGrid) {
      modalStatsGrid.innerHTML = '';
      if (animal.stats) {
        Object.entries(animal.stats).forEach(([k, v]) => {
          const cell = document.createElement('div');
          cell.className = 'stat-cell';
          cell.innerHTML = `
            <div class="stat-lbl">${k.replace('_', ' ')}</div>
            <div class="stat-val">${v}</div>
          `;
          modalStatsGrid.appendChild(cell);
        });
      }
    }

    // Render Frames in Spotlight
    if (spotlightFramesWrap) {
      spotlightFramesWrap.innerHTML = '';
      const frames = getAnimalFrames(animal, currentTheme, isCutout);

      frames.forEach((file, idx) => {
        const img = document.createElement('img');
        img.src = file;
        img.alt = `${animal.common_name} frame ${idx}`;
        img.className = `spotlight-frame-img ${idx === currentSeedIndex ? 'active' : ''}`;
        spotlightFramesWrap.appendChild(img);
      });
    }

    updateSeedUI();
  }

  function startStochasticJump() {
    stopStochasticJump();
    if (!isModalJumping) return;

    updatePlayPauseUI(true);

    modalInterval = setInterval(() => {
      const animal = animalsData[currentModalIndex];
      const frames = getAnimalFrames(animal, currentTheme, isCutout);
      if (frames.length <= 1) return;

      currentSeedIndex = (currentSeedIndex + 1) % frames.length;
      applyFrameIndex(currentSeedIndex);
      updateSeedUI();
    }, jumpSpeed);
  }

  function stopStochasticJump() {
    if (modalInterval) {
      clearInterval(modalInterval);
      modalInterval = null;
    }
    updatePlayPauseUI(false);
  }

  function applyFrameIndex(idx) {
    const allFrames = spotlightFramesWrap.querySelectorAll('.spotlight-frame-img');
    allFrames.forEach((frame, fIdx) => {
      frame.classList.toggle('active', fIdx === idx);
    });
  }

  function updateSeedUI() {
    const seedLabels = ['Seed #42 (Variant A)', 'Seed #108 (Variant B)', 'Seed #999 (Variant C)'];
    if (seedLiveTag) {
      seedLiveTag.textContent = `● ${seedLabels[currentSeedIndex] || `Seed Frame ${currentSeedIndex}`}`;
    }
    seedButtons.forEach((btn, idx) => {
      btn.classList.toggle('active', idx === currentSeedIndex);
    });
  }

  function updatePlayPauseUI(jumping) {
    if (playPauseIcon) playPauseIcon.textContent = jumping ? '⏸' : '▶';
    if (playPauseText) playPauseText.textContent = jumping ? 'Jumping Live' : 'Paused';
    if (btnModalTogglePlay) btnModalTogglePlay.classList.toggle('active', jumping);
  }

  // Play / Pause Button
  if (btnModalTogglePlay) {
    btnModalTogglePlay.addEventListener('click', () => {
      isModalJumping = !isModalJumping;
      if (isModalJumping) {
        startStochasticJump();
      } else {
        stopStochasticJump();
      }
    });
  }

  // Speed Selector Buttons
  speedButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      speedButtons.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      jumpSpeed = parseInt(btn.dataset.speed, 10);
      if (isModalJumping) {
        startStochasticJump();
      }
    });
  });

  // Seed Scrubber Button Clicks (Freeze on specific seed)
  seedButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const idx = parseInt(btn.dataset.seedIdx, 10);
      currentSeedIndex = idx;
      isModalJumping = false;
      stopStochasticJump();
      applyFrameIndex(currentSeedIndex);
      updateSeedUI();
    });
  });

  // Toggle Cutout vs Full Canvas in Modal
  if (btnModalToggleCutout) {
    btnModalToggleCutout.addEventListener('click', () => {
      isCutout = !isCutout;
      btnModalToggleCutout.classList.toggle('active', isCutout);
      btnModalToggleCutout.innerHTML = isCutout
        ? '<span>✂️ Transparent (SAM)</span>'
        : '<span>🖼️ Full Canvas</span>';
      renderSpotlightContent();
      if (isModalJumping) startStochasticJump();
    });
  }

  // Previous & Next Navigation in Modal
  if (modalBtnPrev) {
    modalBtnPrev.addEventListener('click', () => {
      currentModalIndex = (currentModalIndex - 1 + animalsData.length) % animalsData.length;
      currentSeedIndex = 0;
      renderSpotlightContent();
      if (isModalJumping) startStochasticJump();
    });
  }

  if (modalBtnNext) {
    modalBtnNext.addEventListener('click', () => {
      currentModalIndex = (currentModalIndex + 1) % animalsData.length;
      currentSeedIndex = 0;
      renderSpotlightContent();
      if (isModalJumping) startStochasticJump();
    });
  }

  // Close Modal Button & Overlay Click
  if (spotlightCloseBtn) {
    spotlightCloseBtn.addEventListener('click', closeSpotlight);
  }

  if (spotlightOverlay) {
    spotlightOverlay.addEventListener('click', (e) => {
      if (e.target === spotlightOverlay) closeSpotlight();
    });
  }

  // Keyboard Shortcuts (Arrow keys & Esc & Space)
  window.addEventListener('keydown', (e) => {
    if (!spotlightOverlay.classList.contains('active')) return;

    if (e.key === 'Escape') {
      closeSpotlight();
    } else if (e.key === 'ArrowLeft') {
      if (modalBtnPrev) modalBtnPrev.click();
    } else if (e.key === 'ArrowRight') {
      if (modalBtnNext) modalBtnNext.click();
    } else if (e.key === ' ') {
      e.preventDefault();
      if (btnModalTogglePlay) btnModalTogglePlay.click();
    }
  });

  // =========================================================================
  // 7. THEME EXHIBITION SWITCHER (Sketch vs Origami)
  // =========================================================================
  function switchTheme(theme) {
    if (theme === currentTheme) return;
    currentTheme = theme;

    btnThemeSketch.classList.toggle('active', theme === 'sketch');
    btnThemeOrigami.classList.toggle('active', theme === 'origami');

    // Update board items
    renderCameraRoll();

    // If modal is open, re-render it
    if (spotlightOverlay.classList.contains('active')) {
      renderSpotlightContent();
      if (isModalJumping) startStochasticJump();
    }
  }

  if (btnThemeSketch) {
    btnThemeSketch.addEventListener('click', () => switchTheme('sketch'));
  }

  if (btnThemeOrigami) {
    btnThemeOrigami.addEventListener('click', () => switchTheme('origami'));
  }

  // Initial Render
  renderCameraRoll();
});
